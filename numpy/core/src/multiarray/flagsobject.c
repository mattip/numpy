/* Array Flags Object */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"
#include "array_assign.h"

#include "common.h"

static void
_UpdateContiguousFlags(PyArrayObject *ap);

/*HPY_NUMPY_API
 *
 * Get New ArrayFlagsObject
 */
NPY_NO_EXPORT HPy
HPyArray_NewFlagsObject(HPyContext *ctx, HPy flags_type, HPy obj)
{
    HPy flagobj;
    PyArrayFlagsObject *data;
    int flags;
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    if (HPy_IsNull(obj)) {
        flags = NPY_ARRAY_C_CONTIGUOUS |
                NPY_ARRAY_OWNDATA |
                NPY_ARRAY_F_CONTIGUOUS |
                NPY_ARRAY_ALIGNED;
    }
    else {
        if (!HPy_TypeCheck(ctx, obj, array_type)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Need a NumPy array to create a flags object");
            return HPy_NULL;
        }

        flags = PyArray_FLAGS(PyArrayObject_AsStruct(ctx, obj));
    }
    flagobj = HPy_New(ctx, flags_type, &data);
    HPy_Close(ctx, array_type);
    if (HPy_IsNull(flagobj)) {
        return HPy_NULL;
    }
    if (HPy_IsNull(obj)) {
        data->arr = HPyField_NULL;
    } else {
        HPyField_Store(ctx, flagobj, &(data->arr), obj);
    }
    data->flags = flags;
    return flagobj;
}

/*NUMPY_API
 *
 * Get New ArrayFlagsObject
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFlagsObject(PyObject *obj)
{
    HPyContext *ctx = npy_get_context();
    HPy h_res;
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPy h_flags_type = HPy_FromPyObject(ctx, (PyObject *) &PyArrayFlags_Type);
    PyObject *res;
    h_res = HPyArray_NewFlagsObject(ctx, h_flags_type, h_obj);
    HPy_Close(ctx, h_flags_type);
    HPy_Close(ctx, h_obj);
    
    res = HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    
    return res;
}


/*NUMPY_API
 * Update Several Flags at once.
 */
NPY_NO_EXPORT void
PyArray_UpdateFlags(PyArrayObject *ret, int flagmask)
{
    /* Always update both, as its not trivial to guess one from the other */
    if (flagmask & (NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_C_CONTIGUOUS)) {
        _UpdateContiguousFlags(ret);
    }
    if (flagmask & NPY_ARRAY_ALIGNED) {
        if (IsAligned(ret)) {
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_ALIGNED);
        }
        else {
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_ALIGNED);
        }
    }
    /*
     * This is not checked by default WRITEABLE is not
     * part of UPDATE_ALL
     */
    if (flagmask & NPY_ARRAY_WRITEABLE) {
        if (_IsWriteable(ret)) {
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_WRITEABLE);
        }
        else {
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
        }
    }
    return;
}

static void _hpy_UpdateContiguousFlags(HPyContext *ctx, HPy h_ap, PyArrayObject *ap);

/*HPY_NUMPY_API
 * Update Several Flags at once.
 */
NPY_NO_EXPORT void
HPyArray_UpdateFlags(HPyContext *ctx, HPy h_ret, PyArrayObject *ret, int flagmask)
{
    /* Always update both, as its not trivial to guess one from the other */
    if (flagmask & (NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_C_CONTIGUOUS)) {
        _hpy_UpdateContiguousFlags(ctx, h_ret, ret);
    }
    if (flagmask & NPY_ARRAY_ALIGNED) {
        if (HPyIsAligned(ctx, h_ret, ret)) {
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_ALIGNED);
        }
        else {
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_ALIGNED);
        }
    }
    /*
     * This is not checked by default WRITEABLE is not
     * part of UPDATE_ALL
     */
    if (flagmask & NPY_ARRAY_WRITEABLE) {
        if (_IsWriteable(ret)) {
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_WRITEABLE);
        }
        else {
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
        }
    }
    return;
}

/*
 * Check whether the given array is stored contiguously
 * in memory. And update the passed in ap flags appropriately.
 *
 * The traditional rule is that for an array to be flagged as C contiguous,
 * the following must hold:
 *
 * strides[-1] == itemsize
 * strides[i] == shape[i+1] * strides[i + 1]
 *
 * And for an array to be flagged as F contiguous, the obvious reversal:
 *
 * strides[0] == itemsize
 * strides[i] == shape[i - 1] * strides[i - 1]
 *
 * According to these rules, a 0- or 1-dimensional array is either both
 * C- and F-contiguous, or neither; and an array with 2+ dimensions
 * can be C- or F- contiguous, or neither, but not both. Though there
 * there are exceptions for arrays with zero or one item, in the first
 * case the check is relaxed up to and including the first dimension
 * with shape[i] == 0. In the second case `strides == itemsize` will
 * can be true for all dimensions and both flags are set.
 *
 * When NPY_RELAXED_STRIDES_CHECKING is set, we use a more accurate
 * definition of C- and F-contiguity, in which all 0-sized arrays are
 * contiguous (regardless of dimensionality), and if shape[i] == 1
 * then we ignore strides[i] (since it has no affect on memory layout).
 * With these new rules, it is possible for e.g. a 10x1 array to be both
 * C- and F-contiguous -- but, they break downstream code which assumes
 * that for contiguous arrays strides[-1] (resp. strides[0]) always
 * contains the itemsize.
 */
static void
_UpdateContiguousFlags(PyArrayObject *ap)
{
    npy_intp sd;
    npy_intp dim;
    int i;
    npy_bool is_c_contig = 1;

    sd = PyArray_ITEMSIZE(ap);
    for (i = PyArray_NDIM(ap) - 1; i >= 0; --i) {
        dim = PyArray_DIMS(ap)[i];
#if NPY_RELAXED_STRIDES_CHECKING
        /* contiguous by definition */
        if (dim == 0) {
            PyArray_ENABLEFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
            PyArray_ENABLEFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
            return;
        }
        if (dim != 1) {
            if (PyArray_STRIDES(ap)[i] != sd) {
                is_c_contig = 0;
            }
            sd *= dim;
        }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if (PyArray_STRIDES(ap)[i] != sd) {
            is_c_contig = 0;
            break;
        }
        /* contiguous, if it got this far */
        if (dim == 0) {
            break;
        }
        sd *= dim;
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
    }
    if (is_c_contig) {
        PyArray_ENABLEFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
    }
    else {
        PyArray_CLEARFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
    }

    /* check if fortran contiguous */
    sd = PyArray_ITEMSIZE(ap);
    for (i = 0; i < PyArray_NDIM(ap); ++i) {
        dim = PyArray_DIMS(ap)[i];
#if NPY_RELAXED_STRIDES_CHECKING
        if (dim != 1) {
            if (PyArray_STRIDES(ap)[i] != sd) {
                PyArray_CLEARFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
                return;
            }
            sd *= dim;
        }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if (PyArray_STRIDES(ap)[i] != sd) {
            PyArray_CLEARFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
            return;
        }
        if (dim == 0) {
            break;
        }
        sd *= dim;
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
    }
    PyArray_ENABLEFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
    return;
}

static void
_hpy_UpdateContiguousFlags(HPyContext *ctx, HPy h_ap, PyArrayObject *ap)
{
    npy_intp sd;
    npy_intp dim;
    int i;
    npy_bool is_c_contig = 1;

    sd = HPyArray_ITEMSIZE(ctx, h_ap, ap);
    for (i = PyArray_NDIM(ap) - 1; i >= 0; --i) {
        dim = PyArray_DIMS(ap)[i];
#if NPY_RELAXED_STRIDES_CHECKING
        /* contiguous by definition */
        if (dim == 0) {
            PyArray_ENABLEFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
            PyArray_ENABLEFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
            return;
        }
        if (dim != 1) {
            if (PyArray_STRIDES(ap)[i] != sd) {
                is_c_contig = 0;
            }
            sd *= dim;
        }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if (PyArray_STRIDES(ap)[i] != sd) {
            is_c_contig = 0;
            break;
        }
        /* contiguous, if it got this far */
        if (dim == 0) {
            break;
        }
        sd *= dim;
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
    }
    if (is_c_contig) {
        PyArray_ENABLEFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
    }
    else {
        PyArray_CLEARFLAGS(ap, NPY_ARRAY_C_CONTIGUOUS);
    }

    /* check if fortran contiguous */
    sd = HPyArray_ITEMSIZE(ctx, h_ap, ap);
    for (i = 0; i < PyArray_NDIM(ap); ++i) {
        dim = PyArray_DIMS(ap)[i];
#if NPY_RELAXED_STRIDES_CHECKING
        if (dim != 1) {
            if (PyArray_STRIDES(ap)[i] != sd) {
                PyArray_CLEARFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
                return;
            }
            sd *= dim;
        }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if (PyArray_STRIDES(ap)[i] != sd) {
            PyArray_CLEARFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
            return;
        }
        if (dim == 0) {
            break;
        }
        sd *= dim;
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
    }
    PyArray_ENABLEFLAGS(ap, NPY_ARRAY_F_CONTIGUOUS);
    return;
}


#define _define_get(UPPER, lower) \
    static HPy \
    arrayflags_ ## lower ## _get_impl( \
            HPyContext *ctx, HPy self, void *NPY_UNUSED(ignored)) \
    { \
        PyArrayFlagsObject *data = PyArrayFlagsObject_AsStruct(ctx, self); \
        return HPyBool_FromLong(ctx, (data->flags & (UPPER)) == (UPPER)); \
    }

static char *msg = "future versions will not create a writeable "
    "array from broadcast_array. Set the writable flag explicitly to "
    "avoid this warning.";

#define _define_get_warn(UPPER, lower) \
    static HPy \
    arrayflags_ ## lower ## _get_impl( \
            HPyContext *ctx, HPy self, void *NPY_UNUSED(ignored)) \
    { \
        PyArrayFlagsObject *data = PyArrayFlagsObject_AsStruct(ctx, self); \
        if (data->flags & NPY_ARRAY_WARN_ON_WRITE) { \
            if (HPyErr_WarnEx(ctx, ctx->h_FutureWarning, msg, 1) < 0) {\
                return HPy_NULL; \
            } \
        } \
        return HPyBool_FromLong(ctx, (data->flags & (UPPER)) == (UPPER)); \
    }


HPyDef_GET(arrayflags_contiguous_get, "contiguous", arrayflags_contiguous_get_impl)
_define_get(NPY_ARRAY_C_CONTIGUOUS, contiguous)

HPyDef_GET(arrayflags_fortran_get, "fortran", arrayflags_fortran_get_impl)
_define_get(NPY_ARRAY_F_CONTIGUOUS, fortran)

HPyDef_GET(arrayflags_owndata_get, "owndata", arrayflags_owndata_get_impl)
_define_get(NPY_ARRAY_OWNDATA, owndata)

HPyDef_GET(arrayflags_writeable_no_warn_get, "_writeable_no_warn", arrayflags_writeable_no_warn_get_impl)
_define_get(NPY_ARRAY_WRITEABLE, writeable_no_warn)

HPyDef_GET(arrayflags_behaved_get, "behaved", arrayflags_behaved_get_impl)
_define_get_warn(NPY_ARRAY_ALIGNED|
            NPY_ARRAY_WRITEABLE, behaved)

HPyDef_GET(arrayflags_carray_get, "carray", arrayflags_carray_get_impl)
_define_get_warn(NPY_ARRAY_ALIGNED|
            NPY_ARRAY_WRITEABLE|
            NPY_ARRAY_C_CONTIGUOUS, carray)

HPyDef arrayflags_c_contiguous_get = {
        .kind = HPyDef_Kind_GetSet,
        .getset = {
            .name = "c_contiguous",
            .getter_impl = (HPyCFunction)arrayflags_contiguous_get_impl,
            .getter_cpy_trampoline = (cpy_getter)arrayflags_contiguous_get_get_trampoline,
        }
};

HPyDef arrayflags_f_contiguous_get = {
        .kind = HPyDef_Kind_GetSet,
        .getset = {
            .name = "f_contiguous",
            .getter_impl = (HPyCFunction)arrayflags_fortran_get_impl,
            .getter_cpy_trampoline = (cpy_getter)arrayflags_fortran_get_get_trampoline,
        }
};

HPyDef_GET(arrayflags_forc_get, "forc", arrayflags_forc_get_impl)
static HPy
arrayflags_forc_get_impl(HPyContext *ctx, HPy self, void *NPY_UNUSED(ignored))
{
    PyArrayFlagsObject *data = PyArrayFlagsObject_AsStruct(ctx, self);
    HPy item;

    if (((data->flags & NPY_ARRAY_F_CONTIGUOUS) == NPY_ARRAY_F_CONTIGUOUS) ||
        ((data->flags & NPY_ARRAY_C_CONTIGUOUS) == NPY_ARRAY_C_CONTIGUOUS)) {
        item = ctx->h_True;
    }
    else {
        item = ctx->h_False;
    }
    return HPy_Dup(ctx, item);
}

HPyDef_GET(arrayflags_fnc_get, "fnc", arrayflags_fnc_get_impl)
static HPy
arrayflags_fnc_get_impl(HPyContext *ctx, HPy self, void *NPY_UNUSED(ignored))
{
    PyArrayFlagsObject *data = PyArrayFlagsObject_AsStruct(ctx, self);
    HPy item;

    if (((data->flags & NPY_ARRAY_F_CONTIGUOUS) == NPY_ARRAY_F_CONTIGUOUS) &&
        !((data->flags & NPY_ARRAY_C_CONTIGUOUS) == NPY_ARRAY_C_CONTIGUOUS)) {
        item = ctx->h_True;
    }
    else {
        item = ctx->h_False;
    }
    return HPy_Dup(ctx, item);
}

HPyDef_GET(arrayflags_farray_get, "farray", arrayflags_farray_get_impl)
static HPy
arrayflags_farray_get_impl(HPyContext *ctx, HPy self, void *NPY_UNUSED(ignored))
{
    PyArrayFlagsObject *data = PyArrayFlagsObject_AsStruct(ctx, self);
    HPy item;

    if (((data->flags & (NPY_ARRAY_ALIGNED|
                         NPY_ARRAY_WRITEABLE|
                         NPY_ARRAY_F_CONTIGUOUS)) != 0) &&
        !((data->flags & NPY_ARRAY_C_CONTIGUOUS) != 0)) {
        item = ctx->h_True;
    }
    else {
        item = ctx->h_False;
    }
    return HPy_Dup(ctx, item);
}

HPyDef_GET(arrayflags_num_get, "num", arrayflags_num_get_impl)
static HPy
arrayflags_num_get_impl(HPyContext *ctx, HPy self, void *NPY_UNUSED(ignored))
{
    PyArrayFlagsObject *data = PyArrayFlagsObject_AsStruct(ctx, self);
    return HPyLong_FromLong(ctx, data->flags);
}

/* relies on setflags order being write, align, uic */
HPyDef_GETSET(arrayflags_writebackifcopy_getset, "writebackifcopy", arrayflags_writebackifcopy_get_impl, arrayflags_writebackifcopy_set_impl)
_define_get(NPY_ARRAY_WRITEBACKIFCOPY, writebackifcopy)
static int
arrayflags_writebackifcopy_set_impl(
        HPyContext *ctx, HPy self, HPy obj, void *NPY_UNUSED(ignored))
{
    HPy res;
    HPy h_arr, h_setflags, h_args;
    PyArrayFlagsObject *data;

    if (HPy_IsNull(obj)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete flags writebackifcopy attribute");
        return -1;
    }
    data = PyArrayFlagsObject_AsStruct(ctx, self);
    h_arr = HPyField_Load(ctx, self, data->arr);
    if (HPy_IsNull(h_arr)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Cannot set flags on array scalars.");
        return -1;
    }
    h_args = HPyTuple_Pack(ctx, 3,
                           ctx->h_None,
                           ctx->h_None, 
                           HPy_IsTrue(ctx, obj) ? ctx->h_True : ctx->h_False);
    h_setflags = HPy_GetAttr_s(ctx, h_arr, "setflags");
    res = HPy_CallTupleDict(ctx, h_setflags, h_args, HPy_NULL);
    HPy_Close(ctx, h_setflags);
    HPy_Close(ctx, h_args);
    HPy_Close(ctx, h_arr);

    if (HPy_IsNull(res)) {
        return -1;
    }
    HPy_Close(ctx, res);
    return 0;
}

HPyDef_GETSET(arrayflags_aligned_getset, "aligned", arrayflags_aligned_get_impl, arrayflags_aligned_set_impl)
_define_get(NPY_ARRAY_ALIGNED, aligned)
static int
arrayflags_aligned_set_impl(
        HPyContext *ctx, HPy self, HPy obj, void *NPY_UNUSED(ignored))
{
    HPy res;
    HPy h_arr, h_setflags, h_args;
    PyArrayFlagsObject *data;

    if (HPy_IsNull(obj)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete flags aligned attribute");
        return -1;
    }
    data = PyArrayFlagsObject_AsStruct(ctx, self);
    h_arr = HPyField_Load(ctx, self, data->arr);
    if (HPy_IsNull(h_arr)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Cannot set flags on array scalars.");
        return -1;
    }
    h_args = HPyTuple_Pack(ctx, 3,
                           ctx->h_None, 
                           HPy_IsTrue(ctx, obj) ? ctx->h_True : ctx->h_False,
                           ctx->h_None);
    h_setflags = HPy_GetAttr_s(ctx, h_arr, "setflags");
    res = HPy_CallTupleDict(ctx, h_setflags, h_args, HPy_NULL);
    HPy_Close(ctx, h_setflags);
    HPy_Close(ctx, h_args);
    HPy_Close(ctx, h_arr);

    if (HPy_IsNull(res)) {
        return -1;
    }
    HPy_Close(ctx, res);
    return 0;
}


HPyDef_GETSET(arrayflags_writeable_getset, "writeable", arrayflags_writeable_get_impl, arrayflags_writeable_set_impl)
_define_get_warn(NPY_ARRAY_WRITEABLE, writeable)
static int
arrayflags_writeable_set_impl(
        HPyContext *ctx, HPy self, HPy obj, void *NPY_UNUSED(ignored))
{
    HPy res;
    HPy h_arr, h_setflags, h_args;
    PyArrayFlagsObject *data;

    if (HPy_IsNull(obj)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete flags writeable attribute");
        return -1;
    }
    data = PyArrayFlagsObject_AsStruct(ctx, self);
    h_arr = HPyField_Load(ctx, self, data->arr);
    if (HPy_IsNull(h_arr)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Cannot set flags on array scalars.");
        return -1;
    }
    h_args = HPyTuple_Pack(ctx, 3, HPy_IsTrue(ctx, obj) ? ctx->h_True : ctx->h_False, ctx->h_None, ctx->h_None);
    h_setflags = HPy_GetAttr_s(ctx, h_arr, "setflags");
    res = HPy_CallTupleDict(ctx, h_setflags, h_args, HPy_NULL);
    HPy_Close(ctx, h_setflags);
    HPy_Close(ctx, h_args);
    HPy_Close(ctx, h_arr);

    if (HPy_IsNull(res)) {
        return -1;
    }
    HPy_Close(ctx, res);
    return 0;
}

HPyDef_SET(arrayflags_warn_on_write_set, "_warn_on_write", arrayflags_warn_on_write_set_impl)
static int
arrayflags_warn_on_write_set_impl(HPyContext *ctx,
        HPy self, HPy obj, void *NPY_UNUSED(ignored))
{
    /*
     * This code should go away in a future release, so do not mangle the
     * array_setflags function with an extra kwarg
     */
    int ret;
    HPy h_arr;
    PyArrayFlagsObject *data;
    PyArrayObject *array_data;

    if (HPy_IsNull(obj)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete flags _warn_on_write attribute");
        return -1;
    }
    ret = HPy_IsTrue(ctx, obj);
    if (ret > 0) {
        data = PyArrayFlagsObject_AsStruct(ctx, self);
        h_arr = HPyField_Load(ctx, self, data->arr);
        array_data = PyArrayObject_AsStruct(ctx, h_arr);
        if (!(PyArray_FLAGS(array_data) & NPY_ARRAY_WRITEABLE)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                        "cannot set '_warn_on_write' flag when 'writable' is "
                        "False");
            return -1;
        }
        PyArray_ENABLEFLAGS(array_data, NPY_ARRAY_WARN_ON_WRITE);
    }
    else if (ret < 0) {
        return -1;
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "cannot clear '_warn_on_write', set "
                        "writeable True to clear this private flag");
        return -1;
    }
    return 0;
}

HPyDef_SLOT(arrayflags_getitem, arrayflags_getitem_impl, HPy_mp_subscript)
static HPy
arrayflags_getitem_impl(HPyContext *ctx, HPy self, HPy ind)
{
    char *key = NULL;
    char buf[16];
    int n;
    if (HPyUnicode_Check(ctx, ind)) {
        HPy tmp_str;
        tmp_str = HPyUnicode_AsUTF8String(ctx, ind);
        if (HPy_IsNull(tmp_str)) {
            return HPy_NULL;
        }
        key = HPyBytes_AS_STRING(ctx, tmp_str);
        n = HPyBytes_GET_SIZE(ctx, tmp_str);
        if (n > 16) {
            HPy_Close(ctx, tmp_str);
            goto fail;
        }
        memcpy(buf, key, n);
        HPy_Close(ctx, tmp_str);
        key = buf;
    }
    else if (HPyBytes_Check(ctx, ind)) {
        key = HPyBytes_AS_STRING(ctx, ind);
        n = HPyBytes_GET_SIZE(ctx, ind);
    }
    else {
        goto fail;
    }
    switch(n) {
    case 1:
        switch(key[0]) {
        case 'C':
            return arrayflags_contiguous_get_impl(ctx, self, NULL);
        case 'F':
            return arrayflags_fortran_get_impl(ctx, self, NULL);
        case 'W':
            return arrayflags_writeable_get_impl(ctx, self, NULL);
        case 'B':
            return arrayflags_behaved_get_impl(ctx, self, NULL);
        case 'O':
            return arrayflags_owndata_get_impl(ctx, self, NULL);
        case 'A':
            return arrayflags_aligned_get_impl(ctx, self, NULL);
        case 'X':
            return arrayflags_writebackifcopy_get_impl(ctx, self, NULL);
        default:
            goto fail;
        }
        break;
    case 2:
        if (strncmp(key, "CA", n) == 0) {
            return arrayflags_carray_get_impl(ctx, self, NULL);
        }
        if (strncmp(key, "FA", n) == 0) {
            return arrayflags_farray_get_impl(ctx, self, NULL);
        }
        break;
    case 3:
        if (strncmp(key, "FNC", n) == 0) {
            return arrayflags_fnc_get_impl(ctx, self, NULL);
        }
        break;
    case 4:
        if (strncmp(key, "FORC", n) == 0) {
            return arrayflags_forc_get_impl(ctx, self, NULL);
        }
        break;
    case 6:
        if (strncmp(key, "CARRAY", n) == 0) {
            return arrayflags_carray_get_impl(ctx, self, NULL);
        }
        if (strncmp(key, "FARRAY", n) == 0) {
            return arrayflags_farray_get_impl(ctx, self, NULL);
        }
        break;
    case 7:
        if (strncmp(key,"FORTRAN",n) == 0) {
            return arrayflags_fortran_get_impl(ctx, self, NULL);
        }
        if (strncmp(key,"BEHAVED",n) == 0) {
            return arrayflags_behaved_get_impl(ctx, self, NULL);
        }
        if (strncmp(key,"OWNDATA",n) == 0) {
            return arrayflags_owndata_get_impl(ctx, self, NULL);
        }
        if (strncmp(key,"ALIGNED",n) == 0) {
            return arrayflags_aligned_get_impl(ctx, self, NULL);
        }
        break;
    case 9:
        if (strncmp(key,"WRITEABLE",n) == 0) {
            return arrayflags_writeable_get_impl(ctx, self, NULL);
        }
        break;
    case 10:
        if (strncmp(key,"CONTIGUOUS",n) == 0) {
            return arrayflags_contiguous_get_impl(ctx, self, NULL);
        }
        break;
    case 12:
        if (strncmp(key, "C_CONTIGUOUS", n) == 0) {
            return arrayflags_contiguous_get_impl(ctx, self, NULL);
        }
        if (strncmp(key, "F_CONTIGUOUS", n) == 0) {
            return arrayflags_fortran_get_impl(ctx, self, NULL);
        }
        break;
    case 15:
        if (strncmp(key, "WRITEBACKIFCOPY", n) == 0) {
            return arrayflags_writebackifcopy_get_impl(ctx, self, NULL);
        }
        break;
    }

 fail:
    HPyErr_SetString(ctx, ctx->h_KeyError, "Unknown flag");
    return HPy_NULL;
}

HPyDef_SLOT(arrayflags_setitem, arrayflags_setitem_impl, HPy_mp_ass_subscript)
static int
arrayflags_setitem_impl(HPyContext *ctx, HPy self, HPy ind, HPy item)
{
    char *key;
    char buf[16];
    int n;
    if (HPyUnicode_Check(ctx, ind)) {
        HPy tmp_str;
        /* TODO HPY LABS PORT: should be HPyUnicode_AsASCIIString */
        tmp_str = HPyUnicode_AsUTF8String(ctx, ind);
        key = HPyBytes_AS_STRING(ctx, tmp_str);
        n = HPyBytes_GET_SIZE(ctx, tmp_str);
        if (n > 16) n = 16;
        memcpy(buf, key, n);
        HPy_Close(ctx, tmp_str);
        key = buf;
    }
    else if (HPyBytes_Check(ctx, ind)) {
        key = HPyBytes_AS_STRING(ctx, ind);
        n = HPyBytes_GET_SIZE(ctx, ind);
    }
    else {
        goto fail;
    }
    if (((n==9) && (strncmp(key, "WRITEABLE", n) == 0)) ||
        ((n==1) && (strncmp(key, "W", n) == 0))) {
        return arrayflags_writeable_set_impl(ctx, self, item, NULL);
    }
    else if (((n==7) && (strncmp(key, "ALIGNED", n) == 0)) ||
             ((n==1) && (strncmp(key, "A", n) == 0))) {
        return arrayflags_aligned_set_impl(ctx, self, item, NULL);
    }
    else if (((n==15) && (strncmp(key, "WRITEBACKIFCOPY", n) == 0)) ||
             ((n==1) && (strncmp(key, "X", n) == 0))) {
        return arrayflags_writebackifcopy_set_impl(ctx, self, item, NULL);
    }

 fail:
    HPyErr_SetString(ctx, ctx->h_KeyError, "Unknown flag");
    return -1;
}

static char *
_torf_(int flags, int val)
{
    if ((flags & val) == val) {
        return "True";
    }
    else {
        return "False";
    }
}

/* TODO HPY LABS PORT: HPy_tp_str not yet available
HPyDef arrayflags_str = {
    .kind = HPyDef_Kind_Slot,
    .slot = {
        .slot = HPy_tp_str,
        .impl = (HPyCFunction)arrayflags_print_impl,
        .cpy_trampoline = (cpy_PyCFunction)arrayflags_repr_trampoline
    }
};
*/

HPyDef_SLOT(arrayflags_repr, arrayflags_print_impl, HPy_tp_repr)
static HPy
arrayflags_print_impl(HPyContext *ctx, HPy self)
{
    static const char *_warn_on_write_true = "  (with WARN_ON_WRITE=True)";
    static const char fmt_str[] = 
                        "  C_CONTIGUOUS : %s\n  F_CONTIGUOUS : %s\n"
                        "  OWNDATA : %s\n  WRITEABLE : %s%s\n"
                        "  ALIGNED : %s\n  WRITEBACKIFCOPY : %s\n";

    PyArrayFlagsObject *data = PyArrayFlagsObject_AsStruct(ctx, self);
    int fl = data->flags;
    const char *_warn_on_write = "";
    HPy res;
    int fmt_rc;
    
    /*
     * Computation of buffer size: 
     * strlen(fmt_str) + 6 * strlen("False") + sizeof(_warn_on_write_true)
     */
    #define BUFSIZE (sizeof(fmt_str) + 30 * sizeof(char) \
                     + sizeof(_warn_on_write_true))

    if (fl & NPY_ARRAY_WARN_ON_WRITE) {
        _warn_on_write = _warn_on_write_true;
    }
    char *buf = (char *) PyMem_RawMalloc(BUFSIZE);
    if (!buf) {
        return HPy_NULL;
    }
    
    fmt_rc = snprintf(buf, BUFSIZE, fmt_str, 
                        _torf_(fl, NPY_ARRAY_C_CONTIGUOUS),
                        _torf_(fl, NPY_ARRAY_F_CONTIGUOUS),
                        _torf_(fl, NPY_ARRAY_OWNDATA),
                        _torf_(fl, NPY_ARRAY_WRITEABLE),
                        _warn_on_write,
                        _torf_(fl, NPY_ARRAY_ALIGNED),
                        _torf_(fl, NPY_ARRAY_WRITEBACKIFCOPY));
#undef BUFSIZE
    if (fmt_rc >= 0) {
        res = HPyUnicode_FromString(ctx, buf);
    } else {
        /* formatting failed */
        res = HPy_NULL;
    }
    PyMem_RawFree(buf);
    return res;
}

HPyDef_SLOT(arrayflags_richcompare, arrayflags_richcompare_impl, HPy_tp_richcompare)
static HPy
arrayflags_richcompare_impl(HPyContext *ctx, HPy self, HPy other, HPy_RichCmpOp cmp_op)
{
    PyArrayFlagsObject *self_data, *other_data;
    HPy arrayflags_type = HPy_Type(ctx, self);
    int tc = HPy_TypeCheck(ctx, other, arrayflags_type);
    HPy_Close(ctx, arrayflags_type);

    if (!tc) {
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }

    self_data = PyArrayFlagsObject_AsStruct(ctx, self);
    other_data = PyArrayFlagsObject_AsStruct(ctx, other);
    npy_bool eq = self_data->flags == other_data->flags;

    if (cmp_op == HPy_EQ) {
        return HPyBool_FromLong(ctx, eq);
    }
    else if (cmp_op == HPy_NE) {
        return HPyBool_FromLong(ctx, !eq);
    }
    else {
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
}

HPyDef_SLOT(arrayflags_new, arrayflags_new_impl, HPy_tp_new)
static HPy
arrayflags_new_impl(HPyContext *ctx, HPy self, HPy *args, HPy_ssize_t nargs, HPy NPY_UNUSED(kw))
{
    HPy arg = HPy_NULL;
    HPy array_type = HPy_FromPyObject(ctx, (PyObject *) &PyArray_Type);
    
    if (nargs == 1) {
        arg = args[0];
    } else if(nargs != 0) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "expected 0 or 1 argument");
        return HPy_NULL;
    }
    
    if (HPy_IsNull(arg) || !HPy_TypeCheck(ctx, arg, array_type)) {
        arg = HPy_NULL;
    }
    HPy_Close(ctx, array_type);
    return HPyArray_NewFlagsObject(ctx, self, arg);
}

/*
static PyType_Slot arrayflags_slots[] = {
        //{Py_tp_str, arrayflags_print},
        {0, NULL}
};
*/

static HPyDef *arrayflags_defines[] = {
        &arrayflags_new,
        &arrayflags_repr,
        &arrayflags_getitem,
        &arrayflags_setitem,
        &arrayflags_richcompare,
        &arrayflags_contiguous_get,
        &arrayflags_c_contiguous_get,
        &arrayflags_fortran_get,
        &arrayflags_f_contiguous_get,
        &arrayflags_writebackifcopy_getset,
        &arrayflags_owndata_get,
        &arrayflags_aligned_getset,
        &arrayflags_writeable_getset,
        &arrayflags_writeable_no_warn_get,
        &arrayflags_warn_on_write_set,
        &arrayflags_fnc_get,
        &arrayflags_forc_get,
        &arrayflags_behaved_get,
        &arrayflags_carray_get,
        &arrayflags_farray_get,
        &arrayflags_num_get,
        NULL
};

NPY_NO_EXPORT HPyType_Spec PyArrayFlags_Type_Spec = {
    .name = "numpy.core.multiarray.flagsobj",
    .basicsize = sizeof(PyArrayFlagsObject),
    .flags = HPy_TPFLAGS_DEFAULT,
    .defines = arrayflags_defines,
};

NPY_NO_EXPORT PyTypeObject *_PyArrayFlags_Type_p;

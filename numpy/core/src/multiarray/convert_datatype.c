#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"
#include "lowlevel_strided_loops.h"

#include "npy_pycompat.h"
#include "numpy/npy_math.h"

#include "array_coercion.h"
#include "common.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "common_dtype.h"
#include "scalartypes.h"
#include "mapping.h"
#include "legacy_dtype_implementation.h"

#include "abstractdtypes.h"
#include "convert_datatype.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "array_method.h"
#include "usertypes.h"
#include "dtype_transfer.h"

// added by HPy porting:
#include "arraytypes.h"


/*
 * Required length of string when converting from unsigned integer type.
 * Array index is integer size in bytes.
 * - 3 chars needed for cast to max value of 255 or 127
 * - 5 chars needed for cast to max value of 65535 or 32767
 * - 10 chars needed for cast to max value of 4294967295 or 2147483647
 * - 20 chars needed for cast to max value of 18446744073709551615
 *   or 9223372036854775807
 */
NPY_NO_EXPORT npy_intp REQUIRED_STR_LEN[] = {0, 3, 5, 10, 10, 20, 20, 20, 20};


static HPy
HPyArray_GetGenericToVoidCastingImpl(HPyContext *ctx);

static HPy
HPyArray_GetVoidToGenericCastingImpl(HPyContext *ctx);

static HPy
HPyArray_GetGenericToObjectCastingImpl(HPyContext *ctx);

static HPy
HPyArray_GetObjectToGenericCastingImpl(HPyContext *ctx);


/**
 * Fetch the casting implementation from one DType to another.
 *
 * @params from
 * @params to
 *
 * @returns A castingimpl (PyArrayDTypeMethod *), None or NULL with an
 *          error set.
 */
NPY_NO_EXPORT PyObject *
PyArray_GetCastingImpl(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to)
{
    HPyContext *ctx = npy_get_context();
    HPy h_from = HPy_FromPyObject(ctx, (PyObject *)from);
    HPy h_to = HPy_FromPyObject(ctx, (PyObject *)to);
    CAPI_WARN("PyArray_GetCastingImpl");
    HPy h_res = HPyArray_GetCastingImpl(ctx, h_from, from, h_to, to);
    PyObject *res = HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_Close(ctx, h_to);
    HPy_Close(ctx, h_from);
    return res;
}

NPY_NO_EXPORT HPy
HPyArray_GetCastingImpl(HPyContext *ctx, 
                            HPy /* PyArray_DTypeMeta * */ h_from, 
                            PyArray_DTypeMeta *from, 
                            HPy /* PyArray_DTypeMeta * */ h_to, 
                            PyArray_DTypeMeta * to)
{
    HPy res;
    if (HPy_Is(ctx, h_from, h_to)) {
        res = HPY_DTYPE_SLOTS_WITHIN_DTYPE_CASTINGIMPL(ctx, h_from, from);
    }
    else {
        HPy tmp = HPY_DTYPE_SLOTS_CASTINGIMPL(ctx, h_from, from);
        res = HPyDict_GetItemWithError(ctx, tmp, h_to);
        HPy_Close(ctx, tmp);
    }
    if (!HPy_IsNull(res) || HPyErr_Occurred(ctx)) {
        return res;
    }

    /*
        * The following code looks up CastingImpl based on the fact that anything
        * can be cast to and from objects or structured (void) dtypes.
        *
        * The last part adds casts dynamically based on legacy definition
        */
    if (from->type_num == NPY_OBJECT) {
        res = HPyArray_GetObjectToGenericCastingImpl(ctx);
    }
    else if (to->type_num == NPY_OBJECT) {
        res = HPyArray_GetGenericToObjectCastingImpl(ctx);
    }
    else if (from->type_num == NPY_VOID) {
        res = HPyArray_GetVoidToGenericCastingImpl(ctx);
    }
    else if (to->type_num == NPY_VOID) {
        res = HPyArray_GetGenericToVoidCastingImpl(ctx);
    }
    else if (from->type_num < NPY_NTYPES && to->type_num < NPY_NTYPES) {
        /* All builtin dtypes have their casts explicitly defined. */
        // PyErr_Format(PyExc_RuntimeError,
        //         "builtin cast from %S to %S not found, this should not "
        //         "be possible.", from, to);
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "builtin cast from %S to %S not found, this should not "
                "be possible.");
        return HPy_NULL;
    }
    else {
        if (NPY_DT_is_parametric(from) || NPY_DT_is_parametric(to)) {
            return HPy_Dup(ctx, ctx->h_None);
        }
        /* Reject non-legacy dtypes (they need to use the new API) */
        if (!NPY_DT_is_legacy(from) || !NPY_DT_is_legacy(to)) {
            return HPy_Dup(ctx, ctx->h_None);
        }
        if (from != to) {
            /* A cast function must have been registered */
            HPy singleton = HPyField_Load(ctx, h_from, from->singleton);
            PyArray_VectorUnaryFunc *castfunc = HPyArray_GetCastFunc(ctx,
                    singleton, to->type_num);
            HPy_Close(ctx, singleton);
            if (castfunc == NULL) {
                HPyErr_Clear(ctx);
                /* Remember that this cast is not possible */
                HPy castingimpls = HPyField_Load(ctx, h_from, NPY_DT_SLOTS(from)->castingimpls);
                if (HPy_SetItem(ctx, castingimpls, h_to, ctx->h_None) < 0) {
                    HPy_Close(ctx, castingimpls);
                    return HPy_NULL;
                }
                HPy_Close(ctx, castingimpls);
                return HPy_Dup(ctx, ctx->h_None);
            }
        }

        /* PyArray_AddLegacyWrapping_CastingImpl find the correct casting level: */
        /*
            * TODO: Possibly move this to the cast registration time. But if we do
            *       that, we have to also update the cast when the casting safety
            *       is registered.
            */
        if (HPyArray_AddLegacyWrapping_CastingImpl(ctx, h_from, h_to, -1) < 0) {
            return HPy_NULL;
        }
        return HPyArray_GetCastingImpl(ctx, h_from, from, h_to, to);
    }

    if (HPy_IsNull(res)) {
        return HPy_NULL;
    }
    if (HPy_Is(ctx, h_from, h_to)) {
        // PyErr_Format(PyExc_RuntimeError,
        //         "Internal NumPy error, within-DType cast missing for %S!", from);
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "Internal NumPy error, within-DType cast missing for %S!");
        HPy_Close(ctx, res);
        return HPy_NULL;
    }
    HPy castingimpls = HPyField_Load(ctx, h_from, NPY_DT_SLOTS(from)->castingimpls);
    if (HPy_SetItem(ctx, castingimpls, h_to, res) < 0) {
        HPy_Close(ctx, castingimpls);
        HPy_Close(ctx, res);
        return HPy_NULL;
    }
    HPy_Close(ctx, castingimpls);
    return res;
}


/**
 * Fetch the (bound) casting implementation from one DType to another.
 *
 * @params from
 * @params to
 *
 * @returns A bound casting implementation or None (or NULL for error).
 */
static HPy
HPyArray_GetBoundCastingImpl(HPyContext *ctx, 
                                HPy /* PyArray_DTypeMeta * */ from, 
                                HPy /* PyArray_DTypeMeta * */ to)
{
    PyArray_DTypeMeta *from_data = PyArray_DTypeMeta_AsStruct(ctx, from);
    PyArray_DTypeMeta *to_data = PyArray_DTypeMeta_AsStruct(ctx, to);
    HPy method = HPyArray_GetCastingImpl(ctx, from, from_data, to, to_data);
    if (HPy_IsNull(method) || HPy_Is(ctx, method, ctx->h_None)) {
        return method;
    }

    /* TODO: Create better way to wrap method into bound method */
    PyBoundArrayMethodObject *res;
    HPy h_PyBoundArrayMethod_Type = HPyGlobal_Load(ctx, HPyBoundArrayMethod_Type);
    HPy h_res = HPy_New(ctx, h_PyBoundArrayMethod_Type, &res);
    HPy_Close(ctx, h_PyBoundArrayMethod_Type);
    if (HPy_IsNull(h_res)) {
        return HPy_NULL;
    }
    HPyField_Store(ctx, h_res, &res->method, method);
    HPy_Close(ctx, method);
    res->dtypes = (HPyField *)calloc(2, sizeof(HPyField /* PyArray_DTypeMeta * */)); // PyMem_Malloc
    if (res->dtypes == NULL) {
        HPy_Close(ctx, h_res);
        return HPy_NULL;
    }
    // Py_INCREF(from);
    HPyField_Store(ctx, h_res, &res->dtypes[0], from);
    // Py_INCREF(to);
    HPyField_Store(ctx, h_res, &res->dtypes[1], to);

    return h_res;
}


HPyDef_METH(_get_castingimpl, "_get_castingimpl", _get_castingimpl_impl, HPyFunc_VARARGS)
NPY_NO_EXPORT HPy
_get_castingimpl_impl(HPyContext *ctx, HPy NPY_UNUSED(module), HPy *args, HPy_ssize_t nargs)
{
    HPy from, to; // PyArray_DTypeMeta *
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OO:_get_castingimpl",
            &from, &to)) {
        return HPy_NULL;
    }
    HPy h_PyArrayDTypeMeta_Type = HPyGlobal_Load(ctx, HPyArrayDTypeMeta_Type);
    HPy from_type = HPy_Type(ctx, from);
    if (!HPyType_IsSubtype(ctx, from_type, h_PyArrayDTypeMeta_Type)) {
        HPy_Close(ctx, from_type);
        HPy_Close(ctx, h_PyArrayDTypeMeta_Type);
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "must be ?, not ?"); // TODO
        return HPy_NULL;
    }
    HPy_Close(ctx, from_type);
    HPy to_type = HPy_Type(ctx, to);
    if (!HPy_Is(ctx, to_type, h_PyArrayDTypeMeta_Type)) {
        HPy_Close(ctx, to_type);
        HPy_Close(ctx, h_PyArrayDTypeMeta_Type);
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "must be ?, not ?"); // TODO
        return HPy_NULL;
    }
    HPy_Close(ctx, to_type);
    HPy_Close(ctx, h_PyArrayDTypeMeta_Type);

    return HPyArray_GetBoundCastingImpl(ctx, from, to);
}


/**
 * Find the minimal cast safety level given two cast-levels as input.
 * Supports the NPY_CAST_IS_VIEW check, and should be preferred to allow
 * extending cast-levels if necessary.
 * It is not valid for one of the arguments to be -1 to indicate an error.
 *
 * @param casting1
 * @param casting2
 * @return The minimal casting error (can be -1).
 */
NPY_NO_EXPORT NPY_CASTING
PyArray_MinCastSafety(NPY_CASTING casting1, NPY_CASTING casting2)
{
    if (casting1 < 0 || casting2 < 0) {
        return -1;
    }
    /* larger casting values are less safe */
    if (casting1 > casting2) {
        return casting1;
    }
    return casting2;
}


/*NUMPY_API
 * For backward compatibility
 *
 * Cast an array using typecode structure.
 * steals reference to dtype --- cannot be NULL
 *
 * This function always makes a copy of arr, even if the dtype
 * doesn't change.
 */
NPY_NO_EXPORT PyObject *
PyArray_CastToType(PyArrayObject *arr, PyArray_Descr *dtype, int is_f_order)
{
    HPyContext *ctx = npy_get_context();
    HPy h_arr = HPy_FromPyObject(ctx, (PyObject *)arr);
    HPy h_dtype = HPy_FromPyObject(ctx, (PyObject *)dtype);
    HPy h_res = HPyArray_CastToType(ctx, h_arr, h_dtype, is_f_order);
    PyObject *res = HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_Close(ctx, h_dtype);
    HPy_Close(ctx, h_arr);
    // simulate stealing
    Py_XDECREF(dtype);
    return res;
}

/*HPY_NUMPY_API
 * Similar to PyArray_CastToType but *DOES NOT* steal reference to dtype.
 */
NPY_NO_EXPORT HPy
HPyArray_CastToType(HPyContext *ctx, HPy /* (PyArrayObject *) */ arr, HPy /* (PyArray_Descr *) */ dtype, int is_f_order)
{
    HPy out;

    if (HPy_IsNull(dtype)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "dtype is NULL in PyArray_CastToType");
        return HPy_NULL;
    }

    HPy updated_dtype = HPyArray_AdaptDescriptorToArray(ctx, arr, dtype);
    if (HPy_IsNull(updated_dtype)) {
        return HPy_NULL;
    }

    HPy arr_type = HPy_Type(ctx, arr);
    PyArrayObject *arr_data = PyArrayObject_AsStruct(ctx, arr);
    out = HPyArray_NewFromDescr(ctx, arr_type, updated_dtype,
                               PyArray_NDIM(arr_data),
                               PyArray_DIMS(arr_data),
                               NULL, NULL,
                               is_f_order,
                               arr);
    HPy_Close(ctx, updated_dtype);

    if (HPy_IsNull(out)){
        return HPy_NULL;
    }

    if (HPyArray_CopyInto(ctx, out, arr) < 0) {
        HPy_Close(ctx, out);
        return HPy_NULL;
    }

    return out;
}

/*NUMPY_API
 * Get a cast function to cast from the input descriptor to the
 * output type_number (must be a registered data-type).
 * Returns NULL if un-successful.
 */
NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num)
{
    HPyContext *ctx = npy_get_context();
    HPy h_descr = HPy_FromPyObject(ctx, (PyObject *)descr);
    PyArray_VectorUnaryFunc *ret = HPyArray_GetCastFunc(ctx, h_descr, type_num);
    HPy_Close(ctx, h_descr);
    return ret;
}

/*HPY_NUMPY_API
 * Get a cast function to cast from the input descriptor to the
 * output type_number (must be a registered data-type).
 * Returns NULL if un-successful.
 */
NPY_NO_EXPORT PyArray_VectorUnaryFunc *
HPyArray_GetCastFunc(HPyContext *ctx, HPy /* PyArray_Descr * */ h_descr, int type_num)
{
    PyArray_VectorUnaryFunc *castfunc = NULL;

    PyArray_Descr *descr = PyArray_Descr_AsStruct(ctx, h_descr);
    if (type_num < NPY_NTYPES_ABI_COMPATIBLE) {
        castfunc = descr->f->cast[type_num];
    }
    else {
        if (!HPyField_IsNull(descr->f->castdict)) { 
            HPy obj = HPyField_Load(ctx, h_descr, descr->f->castdict);
            if (HPyDict_Check(ctx, obj)) {
                HPy key;
                HPy cobj;

                key = HPyLong_FromLong(ctx, type_num);
                cobj = HPy_GetItem(ctx, obj, key);
                HPy_Close(ctx, key);
                HPy cobj_type = HPy_Type(ctx, cobj);
                if (!HPy_IsNull(cobj) && HPy_Is(ctx, cobj_type, ctx->h_CapsuleType)) {
                    HPy_Close(ctx, cobj_type);
                    castfunc = HPyCapsule_GetPointer(ctx, cobj, NULL);
                    if (castfunc == NULL) {
                        return NULL;
                    }
                } else {
                    HPy_Close(ctx, cobj_type);
                }
            }
        }
    }
    if (PyTypeNum_ISCOMPLEX(descr->type_num) &&
            !PyTypeNum_ISCOMPLEX(type_num) &&
            PyTypeNum_ISNUMBER(type_num) &&
            !PyTypeNum_ISBOOL(type_num)) {
        HPy cls = HPy_NULL, obj = HPy_NULL;
        int ret;
        obj = HPyImport_ImportModule(ctx, "numpy.core");

        if (!HPy_IsNull(obj)) {
            cls = HPy_GetAttr_s(ctx, obj, "ComplexWarning");
            HPy_Close(ctx, obj);
        }
        ret = HPyErr_WarnEx(ctx, cls,
                "Casting complex values to real discards "
                "the imaginary part", 1);
        HPy_Close(ctx, cls);
        if (ret < 0) {
            return NULL;
        }
    }
    if (castfunc) {
        return castfunc;
    }

    HPyErr_SetString(ctx, ctx->h_ValueError,
            "No cast function available.");
    return NULL;
}


/*
 * Must be broadcastable.
 * This code is very similar to PyArray_CopyInto/PyArray_MoveInto
 * except casting is done --- NPY_BUFSIZE is used
 * as the size of the casting buffer.
 */

/*NUMPY_API
 * Cast to an already created array.
 */
NPY_NO_EXPORT int
PyArray_CastTo(PyArrayObject *out, PyArrayObject *mp)
{
    /* CopyInto handles the casting now */
    return PyArray_CopyInto(out, mp);
}

/*NUMPY_API
 * Cast to an already created array.  Arrays don't have to be "broadcastable"
 * Only requirement is they have the same number of elements.
 */
NPY_NO_EXPORT int
PyArray_CastAnyTo(PyArrayObject *out, PyArrayObject *mp)
{
    /* CopyAnyInto handles the casting now */
    return PyArray_CopyAnyInto(out, mp);
}


static NPY_CASTING
_get_cast_safety_from_castingimpl(PyArrayMethodObject *castingimpl,
        PyArray_DTypeMeta *dtypes[2], PyArray_Descr *from, PyArray_Descr *to,
        npy_intp *view_offset)
{
    PyArray_Descr *descrs[2] = {from, to};
    PyArray_Descr *out_descrs[2];

    *view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = resolve_descriptors_trampoline(castingimpl->resolve_descriptors,
            castingimpl, dtypes, descrs, out_descrs, view_offset);
    if (casting < 0) {
        return -1;
    }
    /* The returned descriptors may not match, requiring a second check */
    if (out_descrs[0] != descrs[0]) {
        npy_intp from_offset = NPY_MIN_INTP;
        NPY_CASTING from_casting = PyArray_GetCastInfo(
                descrs[0], out_descrs[0], NULL, &from_offset);
        casting = PyArray_MinCastSafety(casting, from_casting);
        if (from_offset != *view_offset) {
            /* `view_offset` differs: The multi-step cast cannot be a view. */
            *view_offset = NPY_MIN_INTP;
        }
        if (casting < 0) {
            goto finish;
        }
    }
    if (descrs[1] != NULL && out_descrs[1] != descrs[1]) {
        npy_intp from_offset = NPY_MIN_INTP;
        NPY_CASTING from_casting = PyArray_GetCastInfo(
                descrs[1], out_descrs[1], NULL, &from_offset);
        casting = PyArray_MinCastSafety(casting, from_casting);
        if (from_offset != *view_offset) {
            /* `view_offset` differs: The multi-step cast cannot be a view. */
            *view_offset = NPY_MIN_INTP;
        }
        if (casting < 0) {
            goto finish;
        }
    }

  finish:
    Py_DECREF(out_descrs[0]);
    Py_DECREF(out_descrs[1]);
    /*
     * Check for less harmful non-standard returns.  The following two returns
     * should never happen:
     * 1. No-casting must imply a view offset of 0.
     * 2. Equivalent-casting + 0 view offset is (usually) the definition
     *    of a "no" cast.  However, changing the order of fields can also
     *    create descriptors that are not equivalent but views.
     * Note that unsafe casts can have a view offset.  For example, in
     * principle, casting `<i8` to `<i4` is a cast with 0 offset.
     */
    if (*view_offset != 0) {
        assert(casting != NPY_NO_CASTING);
    }
    else {
        assert(casting != NPY_EQUIV_CASTING
               || (PyDataType_HASFIELDS(from) && PyDataType_HASFIELDS(to)));
    }
    return casting;
}

static NPY_CASTING
_hget_cast_safety_from_castingimpl(HPyContext *hctx, HPy castingimpl,
        HPy dtypes[2], HPy from, HPy to,
        npy_intp *view_offset)
{
    HPy descrs[2] = {from, to};
    HPy out_descrs[2];
    PyArrayMethodObject *castingimpl_data = PyArrayMethodObject_AsStruct(hctx, castingimpl);

    *view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = castingimpl_data->resolve_descriptors(hctx,
            castingimpl, dtypes, descrs, out_descrs, view_offset);
    if (casting < 0) {
        return -1;
    }
    /* The returned descriptors may not match, requiring a second check */
    if (!HPy_Is(hctx, out_descrs[0], descrs[0])) {
        npy_intp from_offset = NPY_MIN_INTP;
        NPY_CASTING from_casting = HPyArray_GetCastInfo(hctx,
                descrs[0], out_descrs[0], HPy_NULL, &from_offset);
        casting = PyArray_MinCastSafety(casting, from_casting);
        if (from_offset != *view_offset) {
            /* `view_offset` differs: The multi-step cast cannot be a view. */
            *view_offset = NPY_MIN_INTP;
        }
        if (casting < 0) {
            goto finish;
        }
    }
    if (!HPy_IsNull(descrs[1]) && !HPy_Is(hctx, out_descrs[1], descrs[1])) {
        npy_intp from_offset = NPY_MIN_INTP;
        NPY_CASTING from_casting = HPyArray_GetCastInfo(hctx,
                descrs[1], out_descrs[1], HPy_NULL, &from_offset);
        casting = PyArray_MinCastSafety(casting, from_casting);
        if (from_offset != *view_offset) {
            /* `view_offset` differs: The multi-step cast cannot be a view. */
            *view_offset = NPY_MIN_INTP;
        }
        if (casting < 0) {
            goto finish;
        }
    }

  finish:
    HPy_Close(hctx, out_descrs[0]);
    HPy_Close(hctx, out_descrs[1]);
    /*
     * Check for less harmful non-standard returns.  The following two returns
     * should never happen:
     * 1. No-casting must imply a view offset of 0.
     * 2. Equivalent-casting + 0 view offset is (usually) the definition
     *    of a "no" cast.  However, changing the order of fields can also
     *    create descriptors that are not equivalent but views.
     * Note that unsafe casts can have a view offset.  For example, in
     * principle, casting `<i8` to `<i4` is a cast with 0 offset.
     */
    if (*view_offset != 0) {
        assert(casting != NPY_NO_CASTING);
    }
    else {
        assert(casting != NPY_EQUIV_CASTING
               || (PyDataType_HASFIELDS(PyArray_Descr_AsStruct(hctx, from)) &&
                   PyDataType_HASFIELDS(PyArray_Descr_AsStruct(hctx, to))));
    }
    return casting;
}


/**
 * Given two dtype instances, find the correct casting safety.
 *
 * Note that in many cases, it may be preferable to fetch the casting
 * implementations fully to have them available for doing the actual cast
 * later.
 *
 * @param from
 * @param to The descriptor to cast to (may be NULL)
 * @param to_dtype If `to` is NULL, must pass the to_dtype (otherwise this
 *        is ignored).
 * @param[out] view_offset
 * @return NPY_CASTING or -1 on error or if the cast is not possible.
 */
NPY_NO_EXPORT NPY_CASTING
PyArray_GetCastInfo(
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype,
        npy_intp *view_offset)
{
    if (to != NULL) {
        to_dtype = NPY_DTYPE(to);
    }
    PyObject *meth = PyArray_GetCastingImpl(NPY_DTYPE(from), to_dtype);
    if (meth == NULL) {
        return -1;
    }
    if (meth == Py_None) {
        Py_DECREF(Py_None);
        return -1;
    }

    PyArrayMethodObject *castingimpl = (PyArrayMethodObject *)meth;
    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(from), to_dtype};
    NPY_CASTING casting = _get_cast_safety_from_castingimpl(castingimpl,
            dtypes, from, to, view_offset);
    Py_DECREF(meth);

    return casting;
}

NPY_NO_EXPORT NPY_CASTING
HPyArray_GetCastInfo(HPyContext *ctx,
        HPy from, HPy to, HPy to_dtype,
        npy_intp *view_offset)
{
    if (!HPy_IsNull(to)) {
        to_dtype = HNPY_DTYPE(ctx, to);
    }
    HPy from_meta = HNPY_DTYPE(ctx, from);
    PyArray_DTypeMeta *from_meta_data = PyArray_DTypeMeta_AsStruct(ctx, from_meta);
    PyArray_DTypeMeta *to_dtype_data = PyArray_DTypeMeta_AsStruct(ctx, to_dtype);
    HPy meth = HPyArray_GetCastingImpl(ctx, from_meta, from_meta_data, to_dtype, to_dtype_data);
    HPy_Close(ctx, from_meta);
    if (HPy_IsNull(meth)) {
        if (!HPy_IsNull(to)) {
            HPy_Close(ctx, to_dtype);
        }
        return -1;
    }
    if (HPy_Is(ctx, meth, ctx->h_None)) {
        HPy_Close(ctx, meth);
        return -1;
    }

    HPy dtypes[2] = {HNPY_DTYPE(ctx, from), to_dtype};
    NPY_CASTING casting = _hget_cast_safety_from_castingimpl(ctx, meth,
            dtypes, from, to, view_offset);
    HPy_Close(ctx, dtypes[0]);
    if (!HPy_IsNull(to)) {
        HPy_Close(ctx, to_dtype);
    }
    HPy_Close(ctx, meth);

    return casting;
}


/**
 * Check whether a cast is safe, see also `PyArray_GetCastInfo` for
 * a similar function.  Unlike GetCastInfo, this function checks the
 * `castingimpl->casting` when available.  This allows for two things:
 *
 * 1. It avoids  calling `resolve_descriptors` in some cases.
 * 2. Strings need to discover the length, but in some cases we know that the
 *    cast is valid (assuming the string length is discovered first).
 *
 * The latter means that a `can_cast` could return True, but the cast fail
 * because the parametric type cannot guess the correct output descriptor.
 * (I.e. if `object_arr.astype("S")` did _not_ inspect the objects, and the
 * user would have to guess the string length.)
 *
 * @param casting the requested casting safety.
 * @param from
 * @param to The descriptor to cast to (may be NULL)
 * @param to_dtype If `to` is NULL, must pass the to_dtype (otherwise this
 *        is ignored).
 * @return 0 for an invalid cast, 1 for a valid and -1 for an error.
 */
NPY_NO_EXPORT int
PyArray_CheckCastSafety(NPY_CASTING casting,
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype)
{
    if (to != NULL) {
        to_dtype = NPY_DTYPE(to);
    }
    PyObject *meth = PyArray_GetCastingImpl(NPY_DTYPE(from), to_dtype);
    if (meth == NULL) {
        return -1;
    }
    if (meth == Py_None) {
        Py_DECREF(Py_None);
        return -1;
    }
    PyArrayMethodObject *castingimpl = (PyArrayMethodObject *)meth;

    if (PyArray_MinCastSafety(castingimpl->casting, casting) == casting) {
        /* No need to check using `castingimpl.resolve_descriptors()` */
        Py_DECREF(meth);
        return 1;
    }

    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(from), to_dtype};
    npy_intp view_offset;
    NPY_CASTING safety = _get_cast_safety_from_castingimpl(castingimpl,
            dtypes, from, to, &view_offset);
    Py_DECREF(meth);
    /* If casting is the smaller (or equal) safety we match */
    if (safety < 0) {
        return -1;
    }
    return PyArray_MinCastSafety(safety, casting) == casting;
}


NPY_NO_EXPORT int
HPyArray_CheckCastSafety(HPyContext *ctx, NPY_CASTING casting,
        HPy h_from, HPy h_to, HPy h_to_dtype_in)
{
    HPy h_to_dtype;
    if (HPy_IsNull(h_to)) {
        h_to_dtype = HPy_Type(ctx, h_to);
    } else {
        h_to_dtype = HPy_Dup(ctx, h_to_dtype_in);
    }
    HPy h_from_dtype = HPy_Type(ctx, h_from);
    PyArray_DTypeMeta *h_from_dtype_data = PyArray_DTypeMeta_AsStruct(ctx, h_from_dtype);
    PyArray_DTypeMeta *h_to_dtype_data = PyArray_DTypeMeta_AsStruct(ctx, h_to_dtype);
    HPy meth = HPyArray_GetCastingImpl(ctx, h_from_dtype, h_from_dtype_data, h_to_dtype, h_to_dtype_data);
    HPy_Close(ctx, h_from_dtype);
    if (HPy_IsNull(meth)) {
        return -1;
    }
    if (HPy_Is(ctx, meth, ctx->h_None)) {
        HPy_Close(ctx, meth);
        return -1;
    }

    PyArrayMethodObject *castingimpl = PyArrayMethodObject_AsStruct(ctx, meth);
    if (PyArray_MinCastSafety(castingimpl->casting, casting) == casting) {
        /* No need to check using `castingimpl.resolve_descriptors()` */
        HPy_Close(ctx, meth);
        return 1;
    }

    HPy dtypes[2] = {
        h_from_dtype, 
        h_to_dtype
    };
    npy_intp view_offset;
    NPY_CASTING safety = _hget_cast_safety_from_castingimpl(ctx, meth,
            dtypes, h_from, h_to, &view_offset);
    HPy_Close(ctx, meth);
    /* If casting is the smaller (or equal) safety we match */
    if (safety < 0) {
        return -1;
    }
    return PyArray_MinCastSafety(safety, casting) == casting;
}


/*NUMPY_API
 *Check the type coercion rules.
 */
NPY_NO_EXPORT int
PyArray_CanCastSafely(int fromtype, int totype)
{
    /* Identity */
    if (fromtype == totype) {
        return 1;
    }
    /*
     * As a micro-optimization, keep the cast table around.  This can probably
     * be removed as soon as the ufunc loop lookup is modified (presumably
     * before the 1.21 release).  It does no harm, but the main user of this
     * function is the ufunc-loop lookup calling it until a loop matches!
     *
     * (The table extends further, but is not strictly correct for void).
     * TODO: Check this!
     */
    if ((unsigned int)fromtype <= NPY_CLONGDOUBLE &&
            (unsigned int)totype <= NPY_CLONGDOUBLE) {
        return _npy_can_cast_safely_table[fromtype][totype];
    }

    PyArray_DTypeMeta *from = PyArray_DTypeFromTypeNum(fromtype);
    if (from == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    PyArray_DTypeMeta *to = PyArray_DTypeFromTypeNum(totype);
    if (to == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    PyObject *castingimpl = PyArray_GetCastingImpl(from, to);
    Py_DECREF(from);
    Py_DECREF(to);

    if (castingimpl == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    else if (castingimpl == Py_None) {
        Py_DECREF(Py_None);
        return 0;
    }
    NPY_CASTING safety = ((PyArrayMethodObject *)castingimpl)->casting;
    int res = PyArray_MinCastSafety(safety, NPY_SAFE_CASTING) == NPY_SAFE_CASTING;
    Py_DECREF(castingimpl);
    return res;
}



/*NUMPY_API
 * leaves reference count alone --- cannot be NULL
 *
 * PyArray_CanCastTypeTo is equivalent to this, but adds a 'casting'
 * parameter.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastTo(PyArray_Descr *from, PyArray_Descr *to)
{
    return PyArray_CanCastTypeTo(from, to, NPY_SAFE_CASTING);
}


/* Provides an ordering for the dtype 'kind' character codes */
NPY_NO_EXPORT int
dtype_kind_to_ordering(char kind)
{
    switch (kind) {
        /* Boolean kind */
        case 'b':
            return 0;
        /* Unsigned int kind */
        case 'u':
            return 1;
        /* Signed int kind */
        case 'i':
            return 2;
        /* Float kind */
        case 'f':
            return 4;
        /* Complex kind */
        case 'c':
            return 5;
        /* String kind */
        case 'S':
        case 'a':
            return 6;
        /* Unicode kind */
        case 'U':
            return 7;
        /* Void kind */
        case 'V':
            return 8;
        /* Object kind */
        case 'O':
            return 9;
        /*
         * Anything else, like datetime, is special cased to
         * not fit in this hierarchy
         */
        default:
            return -1;
    }
}

/* Converts a type number from unsigned to signed */
static int
type_num_unsigned_to_signed(int type_num)
{
    switch (type_num) {
        case NPY_UBYTE:
            return NPY_BYTE;
        case NPY_USHORT:
            return NPY_SHORT;
        case NPY_UINT:
            return NPY_INT;
        case NPY_ULONG:
            return NPY_LONG;
        case NPY_ULONGLONG:
            return NPY_LONGLONG;
        default:
            return type_num;
    }
}


/*NUMPY_API
 * Returns true if data of type 'from' may be cast to data of type
 * 'to' according to the rule 'casting'.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting)
{
    PyArray_DTypeMeta *to_dtype = NPY_DTYPE(to);

    /*
     * NOTE: This code supports U and S, this is identical to the code
     *       in `ctors.c` which does not allow these dtypes to be attached
     *       to an array. Unlike the code for `np.array(..., dtype=)`
     *       which uses `PyArray_ExtractDTypeAndDescriptor` it rejects "m8"
     *       as a flexible dtype instance representing a DType.
     */
    /*
     * TODO: We should grow support for `np.can_cast("d", "S")` being
     *       different from `np.can_cast("d", "S0")` here, at least for
     *       the python side API.
     *       The `to = NULL` branch, which considers "S0" to be "flexible"
     *       should probably be deprecated.
     *       (This logic is duplicated in `PyArray_CanCastArrayTo`)
     */
    if (PyDataType_ISUNSIZED(to) && to->subarray == NULL) {
        to = NULL;  /* consider mainly S0 and U0 as S and U */
    }

    int is_valid = PyArray_CheckCastSafety(casting, from, to, to_dtype);
    /* Clear any errors and consider this unsafe (should likely be changed) */
    if (is_valid < 0) {
        PyErr_Clear();
        return 0;
    }
    return is_valid;
}

/*HPY_NUMPY_API
 * Returns true if data of type 'from' may be cast to data of type
 * 'to' according to the rule 'casting'.
 */
NPY_NO_EXPORT npy_bool
HPyArray_CanCastTypeTo(HPyContext *ctx, HPy h_from, HPy h_to,
        NPY_CASTING casting)
{
    PyArray_Descr *to = PyArray_Descr_AsStruct(ctx, h_to);

    /*
     * NOTE: This code supports U and S, this is identical to the code
     *       in `ctors.c` which does not allow these dtypes to be attached
     *       to an array. Unlike the code for `np.array(..., dtype=)`
     *       which uses `PyArray_ExtractDTypeAndDescriptor` it rejects "m8"
     *       as a flexible dtype instance representing a DType.
     */
    /*
     * TODO: We should grow support for `np.can_cast("d", "S")` being
     *       different from `np.can_cast("d", "S0")` here, at least for
     *       the python side API.
     *       The `to = NULL` branch, which considers "S0" to be "flexible"
     *       should probably be deprecated.
     *       (This logic is duplicated in `PyArray_CanCastArrayTo`)
     */
    if (PyDataType_ISUNSIZED(to) && to->subarray == NULL) {
        to = NULL;  /* consider mainly S0 and U0 as S and U */
    }

    HPy to_meta = HPy_Type(ctx, h_to);
    int is_valid = HPyArray_CheckCastSafety(ctx, casting, h_from, h_to, to_meta);
    HPy_Close(ctx, to_meta);
    /* Clear any errors and consider this unsafe (should likely be changed) */
    if (is_valid < 0) {
        HPyErr_Clear(ctx);
        return 0;
    }
    return is_valid;
}


/* CanCastArrayTo needs this function */
static int min_scalar_type_num(char *valueptr, int type_num,
                                            int *is_small_unsigned);


/*
 * NOTE: This function uses value based casting logic for scalars. It will
 *       require updates when we phase out value-based-casting.
 */
NPY_NO_EXPORT npy_bool
can_cast_scalar_to(HPyContext *ctx, HPy /* (PyArray_Descr *) */ scal_type, char *scal_data,
                    HPy /* (PyArray_Descr *) */ to, NPY_CASTING casting)
{
    /*
     * If the two dtypes are actually references to the same object
     * or if casting type is forced unsafe then always OK.
     *
     * TODO: Assuming that unsafe casting always works is not actually correct
     */
    if (HPy_Is(ctx, scal_type, to) || casting == NPY_UNSAFE_CASTING ) {
        return 1;
    }

    HPy to_meta = HNPY_DTYPE(ctx, to);
    int valid = HPyArray_CheckCastSafety(ctx, casting, scal_type, to, to_meta);
    HPy_Close(ctx, to_meta);
    if (valid == 1) {
        /* This is definitely a valid cast. */
        return 1;
    }
    if (valid < 0) {
        /* Probably must return 0, but just keep trying for now. */
        HPyErr_Clear(ctx);
    }

    /*
     * If the scalar isn't a number, value-based casting cannot kick in and
     * we must not attempt it.
     * (Additional fast-checks would be possible, but probably unnecessary.)
     */
    PyArray_Descr *scal_type_data = PyArray_Descr_AsStruct(ctx, scal_type);
    if (!PyTypeNum_ISNUMBER(scal_type_data->type_num)) {
        return 0;
    }

    /*
     * At this point we have to check value-based casting.
     */
    HPy dtype; /* (PyArray_Descr *) */
    int is_small_unsigned = 0, type_num;
    /* An aligned memory buffer large enough to hold any builtin numeric type */
    npy_longlong value[4];

    int swap = !PyArray_ISNBO(scal_type_data->byteorder);
    scal_type_data->f->copyswap(&value, scal_data, swap, NULL);

    type_num = min_scalar_type_num((char *)&value, scal_type_data->type_num,
                                    &is_small_unsigned);

    /*
     * If we've got a small unsigned scalar, and the 'to' type
     * is not unsigned, then make it signed to allow the value
     * to be cast more appropriately.
     */
    PyArray_Descr *to_data = PyArray_Descr_AsStruct(ctx, to);
    if (is_small_unsigned && !(PyTypeNum_ISUNSIGNED(to_data->type_num))) {
        type_num = type_num_unsigned_to_signed(type_num);
    }

    dtype = HPyArray_DescrFromType(ctx, type_num);
    if (HPy_IsNull(dtype)) {
        return 0;
    }
#if 0
    printf("min scalar cast ");
    PyObject_Print(dtype, stdout, 0);
    printf(" to ");
    PyObject_Print(to, stdout, 0);
    printf("\n");
#endif
    npy_bool ret = HPyArray_CanCastTypeTo(ctx, dtype, to, casting);
    HPy_Close(ctx, dtype);
    return ret;
}

/*
 * NOTE: This function uses value based casting logic for scalars. It will
 *       require updates when we phase out value-based-casting.
 */
NPY_NO_EXPORT npy_bool
hpy_can_cast_scalar_to(HPyContext *ctx, HPy scal_type, char *scal_data,
                    HPy to, NPY_CASTING casting)
{
    /*
     * If the two dtypes are actually references to the same object
     * or if casting type is forced unsafe then always OK.
     *
     * TODO: Assuming that unsafe casting always works is not actually correct
     */
    if (HPy_Is(ctx, scal_type, to) || casting == NPY_UNSAFE_CASTING ) {
        return 1;
    }

    int valid = HPyArray_CheckCastSafety(ctx, casting, scal_type, to, HPyArray_DTYPE(ctx, to));
    if (valid == 1) {
        /* This is definitely a valid cast. */
        return 1;
    }
    if (valid < 0) {
        /* Probably must return 0, but just keep trying for now. */
        HPyErr_Clear(ctx);
    }

    PyArray_Descr *scal_type_data = PyArray_Descr_AsStruct(ctx, scal_type);
    /*
     * If the scalar isn't a number, value-based casting cannot kick in and
     * we must not attempt it.
     * (Additional fast-checks would be possible, but probably unnecessary.)
     */
    if (!PyTypeNum_ISNUMBER(scal_type_data->type_num)) {
        return 0;
    }

    /*
     * At this point we have to check value-based casting.
     */
    int is_small_unsigned = 0, type_num;
    /* An aligned memory buffer large enough to hold any builtin numeric type */
    npy_longlong value[4];

    int swap = !PyArray_ISNBO(scal_type_data->byteorder);
    CAPI_WARN("Not clear what scal_type_data->f->copyswap may call...");
    scal_type_data->f->copyswap(&value, scal_data, swap, NULL);

    type_num = min_scalar_type_num((char *)&value, scal_type_data->type_num,
                                    &is_small_unsigned);

    /*
     * If we've got a small unsigned scalar, and the 'to' type
     * is not unsigned, then make it signed to allow the value
     * to be cast more appropriately.
     */
    PyArray_Descr *to_data = PyArray_Descr_AsStruct(ctx, to);
    if (is_small_unsigned && !(PyTypeNum_ISUNSIGNED(to_data->type_num))) {
        type_num = type_num_unsigned_to_signed(type_num);
    }

    HPy dtype = HPyArray_DescrFromType(ctx, type_num);
    if (HPy_IsNull(dtype)) {
        return 0;
    }
#if 0
    printf("min scalar cast ");
    PyObject_Print(dtype, stdout, 0);
    printf(" to ");
    PyObject_Print(to, stdout, 0);
    printf("\n");
#endif
    npy_bool ret = HPyArray_CanCastTypeTo(ctx, dtype, to, casting);
    HPy_Close(ctx, dtype);
    return ret;
}

/*NUMPY_API
 * Returns 1 if the array object may be cast to the given data type using
 * the casting rule, 0 otherwise.  This differs from PyArray_CanCastTo in
 * that it handles scalar arrays (0 dimensions) specially, by checking
 * their value.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastArrayTo(PyArrayObject *arr, PyArray_Descr *to,
                        NPY_CASTING casting)
{
    HPyContext *ctx = npy_get_context();
    HPy h_arr = HPy_FromPyObject(ctx, (PyObject*)arr);
    HPy h_to = HPy_FromPyObject(ctx, (PyObject*)to);
    return HPyArray_CanCastArrayTo(ctx, h_arr, h_to, casting);
}

/*HPY_NUMPY_API
 * Returns 1 if the array object may be cast to the given data type using
 * the casting rule, 0 otherwise.  This differs from PyArray_CanCastTo in
 * that it handles scalar arrays (0 dimensions) specially, by checking
 * their value.
 */
NPY_NO_EXPORT npy_bool
HPyArray_CanCastArrayTo(HPyContext *ctx, HPy /* (PyArrayObject *) */ arr,
                        HPy /* (PyArray_Descr *) */ to, NPY_CASTING casting)
{
    HPy from = HPyArray_GetDescr(ctx, arr); /* (PyArray_Descr *) */
    HPy to_dtype = HNPY_DTYPE(ctx, to); /* (PyArray_DTypeMeta *) */

    /* NOTE, TODO: The same logic as `PyArray_CanCastTypeTo`: */
    PyArray_Descr *to_data = PyArray_Descr_AsStruct(ctx, to);
    if (PyDataType_ISUNSIZED(to_data)) {
        CAPI_WARN("HPyArray_CanCastArrayTo: access to legacy field PyArray_Descr.subarray");
        if (to_data->subarray == NULL) {
            to = HPy_NULL;
        }
    }

    /*
     * If it's a scalar, check the value.  (This only currently matters for
     * numeric types and for `to == NULL` it can't be numeric.)
     */
    PyArrayObject *arr_data = PyArrayObject_AsStruct(ctx, arr);
    if (PyArray_NDIM(arr_data) == 0 && !HPyArray_HASFIELDS(ctx, arr, arr_data) && !HPy_IsNull(to)) {
        return can_cast_scalar_to(ctx, from, PyArray_DATA(arr_data), to, casting);
    }

    /* Otherwise, use the standard rules (same as `PyArray_CanCastTypeTo`) */
    int is_valid = HPyArray_CheckCastSafety(ctx, casting, from, to, to_dtype);
    /* Clear any errors and consider this unsafe (should likely be changed) */
    if (is_valid < 0) {
        HPyErr_Clear(ctx);
        return 0;
    }
    return is_valid;
}


NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";
        case NPY_EQUIV_CASTING:
            return "'equiv'";
        case NPY_SAFE_CASTING:
            return "'safe'";
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";
        default:
            return "<unknown>";
    }
}


/**
 * Helper function to set a useful error when casting is not possible.
 *
 * @param src_dtype
 * @param dst_dtype
 * @param casting
 * @param scalar Whether this was a "scalar" cast (includes 0-D array with
 *               PyArray_CanCastArrayTo result).
 */
NPY_NO_EXPORT void
npy_set_invalid_cast_error(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar)
{
    char *msg;

    if (!scalar) {
        msg = "Cannot cast array data from %R to %R according to the rule %s";
    }
    else {
        msg = "Cannot cast scalar from %R to %R according to the rule %s";
    }
    PyErr_Format(PyExc_TypeError,
            msg, src_dtype, dst_dtype, npy_casting_to_string(casting));
}

NPY_NO_EXPORT void
hpy_npy_set_invalid_cast_error(HPyContext *ctx,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar)
{
    char *msg;

    if (!scalar) {
        msg = "Cannot cast array data from %R to %R according to the rule %s";
    }
    else {
        msg = "Cannot cast scalar from %R to %R according to the rule %s";
    }
    // PyErr_Format(PyExc_TypeError,
    //         msg, src_dtype, dst_dtype, npy_casting_to_string(casting));
    HPyErr_SetString(ctx, ctx->h_TypeError, msg);
}

/*NUMPY_API
 * See if array scalars can be cast.
 *
 * TODO: For NumPy 2.0, add a NPY_CASTING parameter.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastScalar(PyTypeObject *from, PyTypeObject *to)
{
    int fromtype;
    int totype;

    fromtype = _typenum_fromtypeobj((PyObject *)from, 0);
    totype = _typenum_fromtypeobj((PyObject *)to, 0);
    if (fromtype == NPY_NOTYPE || totype == NPY_NOTYPE) {
        return NPY_FALSE;
    }
    return (npy_bool) PyArray_CanCastSafely(fromtype, totype);
}

/*
 * Internal promote types function which handles unsigned integers which
 * fit in same-sized signed integers specially.
 */
static HPy
hpy_promote_types(HPyContext *ctx, HPy h_type1, HPy h_type2,
                        int is_small_unsigned1, int is_small_unsigned2)
{
    PyArray_Descr *type1  = PyArray_Descr_AsStruct(ctx, h_type1);
    PyArray_Descr *type2  = PyArray_Descr_AsStruct(ctx, h_type2);
    if (is_small_unsigned1) {
        int type_num1 = type1->type_num;
        int type_num2 = type2->type_num;
        int ret_type_num;

        if (type_num2 < NPY_NTYPES && !(PyTypeNum_ISBOOL(type_num2) ||
                                        PyTypeNum_ISUNSIGNED(type_num2))) {
            /* Convert to the equivalent-sized signed integer */
            type_num1 = type_num_unsigned_to_signed(type_num1);

            ret_type_num = _npy_type_promotion_table[type_num1][type_num2];
            /* The table doesn't handle string/unicode/void, check the result */
            if (ret_type_num >= 0) {
                return HPyArray_DescrFromType(ctx, ret_type_num);
            }
        }

        return HPyArray_PromoteTypes(ctx, h_type1, h_type2);
    }
    else if (is_small_unsigned2) {
        int type_num1 = type1->type_num;
        int type_num2 = type2->type_num;
        int ret_type_num;

        if (type_num1 < NPY_NTYPES && !(PyTypeNum_ISBOOL(type_num1) ||
                                        PyTypeNum_ISUNSIGNED(type_num1))) {
            /* Convert to the equivalent-sized signed integer */
            type_num2 = type_num_unsigned_to_signed(type_num2);

            ret_type_num = _npy_type_promotion_table[type_num1][type_num2];
            /* The table doesn't handle string/unicode/void, check the result */
            if (ret_type_num >= 0) {
                return HPyArray_DescrFromType(ctx, ret_type_num);
            }
        }

        return HPyArray_PromoteTypes(ctx, h_type1, h_type2);
    }
    else {
        return HPyArray_PromoteTypes(ctx, h_type1, h_type2);
    }

}

/*
 * Returns a new reference to type if it is already NBO, otherwise
 * returns a copy converted to NBO.
 */
NPY_NO_EXPORT PyArray_Descr *
ensure_dtype_nbo(PyArray_Descr *type)
{
    if (PyArray_ISNBO(type->byteorder)) {
        Py_INCREF(type);
        return type;
    }
    else {
        return PyArray_DescrNewByteorder(type, NPY_NATIVE);
    }
}

NPY_NO_EXPORT HPy
hensure_dtype_nbo(HPyContext *ctx, HPy type)
{
    if (PyArray_ISNBO(PyArray_Descr_AsStruct(ctx, type)->byteorder)) {
        return HPy_Dup(ctx, type);
    }
    else {
        return HPyArray_DescrNewByteorder(ctx, type, NPY_NATIVE);
    }
}


/**
 * This function should possibly become public API eventually.  At this
 * time it is implemented by falling back to `PyArray_AdaptFlexibleDType`.
 * We will use `CastingImpl[from, to].resolve_descriptors(...)` to implement
 * this logic.
 * Before that, the API needs to be reviewed though.
 *
 * WARNING: This function currently does not guarantee that `descr` can
 *          actually be cast to the given DType.
 *
 * @param descr The dtype instance to adapt "cast"
 * @param given_DType The DType class for which we wish to find an instance able
 *        to represent `descr`.
 * @returns Instance of `given_DType`. If `given_DType` is parametric the
 *          descr may be adapted to hold it.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType)
{
    HPyContext *ctx = npy_get_context();
    HPy h_descr = HPy_FromPyObject(ctx, (PyObject *)descr);
    HPy h_given_DType = HPy_FromPyObject(ctx, (PyObject *)given_DType);
    HPy h_res = HPyArray_CastDescrToDType(ctx, h_descr, h_given_DType);
    PyArray_Descr *res = (PyArray_Descr *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_Close(ctx, h_given_DType);
    HPy_Close(ctx, h_descr);
    return res;
}

//NPY_NO_EXPORT PyArray_Descr *
//PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType)
NPY_NO_EXPORT HPy
HPyArray_CastDescrToDType(HPyContext *ctx, HPy descr, HPy given_DType)
{
    HPy res = HPy_NULL;
    HPy descr_type = HNPY_DTYPE(ctx, descr);
    if (HPy_Is(ctx, descr_type, given_DType)) {
        res = HPy_Dup(ctx, descr);
        goto success;
    }
    PyArray_DTypeMeta *given_DType_data = PyArray_DTypeMeta_AsStruct(ctx, given_DType);
    if (!NPY_DT_is_parametric(given_DType_data)) {
        /*
         * Don't actually do anything, the default is always the result
         * of any cast.
         */
        res = HNPY_DT_CALL_default_descr(ctx, given_DType, given_DType_data);
        goto success;
    }
    if (HPy_TypeCheck(ctx, descr, given_DType)) {
        res = HPy_Dup(ctx, descr);
        goto success;
    }

    PyArray_DTypeMeta *descr_type_data = PyArray_DTypeMeta_AsStruct(ctx, descr_type);
    HPy tmp = HPyArray_GetCastingImpl(ctx, descr_type, descr_type_data, given_DType, given_DType_data);
    if (HPy_IsNull(tmp) || HPy_Is(ctx, tmp, ctx->h_None)) {
        HPy_Close(ctx, tmp);
        goto error;
    }
    HPy dtypes[2] = {descr_type, given_DType}; /* (PyArray_DTypeMeta *) */
    HPy given_descrs[2] = {descr, HPy_NULL}; /* (PyArray_Descr *) */
    HPy loop_descrs[2]; /* (PyArray_Descr *) */

    // PyArrayMethodObject *meth = (PyArrayMethodObject *)tmp;
    PyArrayMethodObject *meth = PyArrayMethodObject_AsStruct(ctx, tmp);
    npy_intp view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = meth->resolve_descriptors(ctx,
            tmp, dtypes, given_descrs, loop_descrs, &view_offset);
    HPy_Close(ctx, tmp);
    if (casting < 0) {
        goto error;
    }
    HPy_Close(ctx, loop_descrs[0]);
    res = loop_descrs[1];
success:
    HPy_Close(ctx, descr_type);
    return res;

  error:;  /* (; due to compiler limitations) */
    HPy_Close(ctx, descr_type);
    // TODO HPY LABS PORT: PyErr_Fetch
    // PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
    // PyErr_Fetch(&err_type, &err_value, &err_traceback);
    HPyErr_SetString(ctx, ctx->h_TypeError,
            "cannot cast dtype %S to %S.");
    // TODO HPY LABS PORT: PyErr_Format
    // PyErr_Format(PyExc_TypeError,
    //         "cannot cast dtype %S to %S.", descr, given_DType);
    // TODO HPY LABS PORT: npy_PyErr_ChainExceptionsCause
    // npy_PyErr_ChainExceptionsCause(err_type, err_value, err_traceback);
    return HPy_NULL;
}


/*
 * Helper to find the target descriptor for multiple arrays given an input
 * one that may be a DType class (e.g. "U" or "S").
 * Works with arrays, since that is what `concatenate` works with. However,
 * unlike `np.array(...)` or `arr.astype()` we will never inspect the array's
 * content, which means that object arrays can only be cast to strings if a
 * fixed width is provided (same for string -> generic datetime).
 *
 * As this function uses `PyArray_ExtractDTypeAndDescriptor`, it should
 * eventually be refactored to move the step to an earlier point.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_FindConcatenationDescriptor(
        npy_intp n, PyArrayObject **arrays, PyObject *requested_dtype)
{
    if (requested_dtype == NULL) {
        return PyArray_LegacyResultType(n, arrays, 0, NULL);
    }

    PyArray_DTypeMeta *common_dtype;
    PyArray_Descr *result = NULL;
    if (PyArray_ExtractDTypeAndDescriptor(
            requested_dtype, &result, &common_dtype) < 0) {
        return NULL;
    }
    if (result != NULL) {
        if (result->subarray != NULL) {
            PyErr_Format(PyExc_TypeError,
                    "The dtype `%R` is not a valid dtype for concatenation "
                    "since it is a subarray dtype (the subarray dimensions "
                    "would be added as array dimensions).", result);
            Py_SETREF(result, NULL);
        }
        goto finish;
    }
    assert(n > 0);  /* concatenate requires at least one array input. */

    /*
     * NOTE: This code duplicates `PyArray_CastToDTypeAndPromoteDescriptors`
     *       to use arrays, copying the descriptors seems not better.
     */
    PyArray_Descr *descr = PyArray_DESCR(arrays[0]);
    result = PyArray_CastDescrToDType(descr, common_dtype);
    if (result == NULL || n == 1) {
        goto finish;
    }
    for (npy_intp i = 1; i < n; i++) {
        descr = PyArray_DESCR(arrays[i]);
        PyArray_Descr *curr = PyArray_CastDescrToDType(descr, common_dtype);
        if (curr == NULL) {
            Py_SETREF(result, NULL);
            goto finish;
        }
        Py_SETREF(result, NPY_DT_SLOTS(common_dtype)->common_instance(result, curr));
        Py_DECREF(curr);
        if (result == NULL) {
            goto finish;
        }
    }

  finish:
    Py_DECREF(common_dtype);
    return result;
}


/*NUMPY_API
 * Produces the smallest size and lowest kind type to which both
 * input types can be cast.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_PromoteTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    HPyContext *ctx = npy_get_context();
    HPy h_type1 = HPy_FromPyObject(ctx, (PyObject *)type1);
    HPy h_type2 = HPy_FromPyObject(ctx, (PyObject *)type2);
    HPy h_res = HPyArray_PromoteTypes(ctx, h_type1, h_type2);
    PyArray_Descr *res = (PyArray_Descr *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_type1);
    HPy_Close(ctx, h_type2);
    HPy_Close(ctx, h_res);
    return res;
}

/*HPY_NUMPY_API
 * Produces the smallest size and lowest kind type to which both
 * input types can be cast.
 */
NPY_NO_EXPORT HPy
HPyArray_PromoteTypes(HPyContext *ctx, HPy h_type1, HPy h_type2)
{
    PyArray_Descr *type1 = PyArray_Descr_AsStruct(ctx, h_type1);
    // PyArray_DTypeMeta *common_dtype;
    HPy res;

    /* Fast path for identical inputs (NOTE: This path preserves metadata!) */
    if (HPy_Is(ctx, h_type1, h_type2) && PyArray_ISNBO(type1->byteorder)) {
        return HPy_Dup(ctx, h_type1);
    }

    CAPI_WARN("HPyArray_PromoteTypes: calling PyArray_CommonDType (common_dtype)");
    PyArray_DTypeMeta *py_dtype1 = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_type1);
    PyArray_DTypeMeta *py_dtype2 = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_type2);
    PyArray_DTypeMeta *common_dtype = PyArray_CommonDType(NPY_DTYPE(py_dtype1), NPY_DTYPE(py_dtype2));
    HPy h_common_dtype = HPy_FromPyObject(ctx, (PyObject *)common_dtype);
    Py_DECREF(py_dtype1);
    Py_DECREF(py_dtype2);

    if (HPy_IsNull(h_common_dtype)) {
        return HPy_NULL;
    }

    if (!NPY_DT_is_parametric(common_dtype)) {
        /* Note that this path loses all metadata */
        res = HNPY_DT_CALL_default_descr(ctx, h_common_dtype, common_dtype);
        HPy_Close(ctx, h_common_dtype);
        return res;
    }

    /* Cast the input types to the common DType if necessary */
    HPy hh_type1 = HPyArray_CastDescrToDType(ctx, h_type1, h_common_dtype);
    if (HPy_IsNull(hh_type1)) {
        HPy_Close(ctx, h_common_dtype);
        return HPy_NULL;
    }
    HPy hh_type2 = HPyArray_CastDescrToDType(ctx, h_type2, h_common_dtype);
    if (HPy_IsNull(hh_type2)) {
        HPy_Close(ctx, hh_type1);
        HPy_Close(ctx, h_common_dtype);
        return HPy_NULL;
    }

    /*
     * And find the common instance of the two inputs
     * NOTE: Common instance preserves metadata (normally and of one input)
     */
    CAPI_WARN("HPyArray_PromoteTypes: calling NPY_DT_SLOTS(common_dtype)->common_instance");
    PyArray_Descr *py_type1 = (PyArray_Descr *)HPy_AsPyObject(ctx, hh_type1);
    PyArray_Descr *py_type2 = (PyArray_Descr *)HPy_AsPyObject(ctx, hh_type2);
    PyArray_Descr *py_res = NPY_DT_SLOTS(common_dtype)->common_instance(py_type1, py_type2);
    res = HPy_FromPyObject(ctx, (PyObject *)py_res);
    HPy_Close(ctx, hh_type1);
    HPy_Close(ctx, hh_type2);
    HPy_Close(ctx, h_common_dtype);
    return res;
}

/*
 * Produces the smallest size and lowest kind type to which all
 * input types can be cast.
 *
 * Roughly equivalent to functools.reduce(PyArray_PromoteTypes, types)
 * but uses a more complex pairwise approach.
 */
NPY_NO_EXPORT HPy
HPyArray_PromoteTypeSequence(HPyContext *ctx, HPy *types, npy_intp ntypes)
{
    if (ntypes == 0) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "at least one type needed to promote");
        return HPy_NULL;
    }
    return HPyArray_ResultType(ctx, 0, NULL, ntypes, types);
}

/*
 * NOTE: While this is unlikely to be a performance problem, if
 *       it is it could be reverted to a simple positive/negative
 *       check as the previous system used.
 *
 * The is_small_unsigned output flag indicates whether it's an unsigned integer,
 * and would fit in a signed integer of the same bit size.
 */
static int min_scalar_type_num(char *valueptr, int type_num,
                                            int *is_small_unsigned)
{
    switch (type_num) {
        case NPY_BOOL: {
            return NPY_BOOL;
        }
        case NPY_UBYTE: {
            npy_ubyte value = *(npy_ubyte *)valueptr;
            if (value <= NPY_MAX_BYTE) {
                *is_small_unsigned = 1;
            }
            return NPY_UBYTE;
        }
        case NPY_BYTE: {
            npy_byte value = *(npy_byte *)valueptr;
            if (value >= 0) {
                *is_small_unsigned = 1;
                return NPY_UBYTE;
            }
            break;
        }
        case NPY_USHORT: {
            npy_ushort value = *(npy_ushort *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }

            if (value <= NPY_MAX_SHORT) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_SHORT: {
            npy_short value = *(npy_short *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_USHORT, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_ULONG:
#endif
        case NPY_UINT: {
            npy_uint value = *(npy_uint *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }

            if (value <= NPY_MAX_INT) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_LONG:
#endif
        case NPY_INT: {
            npy_int value = *(npy_int *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_UINT, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            break;
        }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
        case NPY_ULONG: {
            npy_ulong value = *(npy_ulong *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            else if (value <= NPY_MAX_UINT) {
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }

            if (value <= NPY_MAX_LONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_LONG: {
            npy_long value = *(npy_long *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONG, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
            break;
        }
#endif
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_ULONG:
#endif
        case NPY_ULONGLONG: {
            npy_ulonglong value = *(npy_ulonglong *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            else if (value <= NPY_MAX_UINT) {
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            else if (value <= NPY_MAX_ULONG) {
                if (value <= NPY_MAX_LONG) {
                    *is_small_unsigned = 1;
                }
                return NPY_ULONG;
            }
#endif

            if (value <= NPY_MAX_LONGLONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_LONG:
#endif
        case NPY_LONGLONG: {
            npy_longlong value = *(npy_longlong *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONGLONG, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            else if (value >= NPY_MIN_LONG) {
                return NPY_LONG;
            }
#endif
            break;
        }
        /*
         * Float types aren't allowed to be demoted to integer types,
         * but precision loss is allowed.
         */
        case NPY_HALF: {
            return NPY_HALF;
        }
        case NPY_FLOAT: {
            float value = *(float *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            break;
        }
        case NPY_DOUBLE: {
            double value = *(double *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            else if (value > -3.4e38 && value < 3.4e38) {
                return NPY_FLOAT;
            }
            break;
        }
        case NPY_LONGDOUBLE: {
            npy_longdouble value = *(npy_longdouble *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            else if (value > -3.4e38 && value < 3.4e38) {
                return NPY_FLOAT;
            }
            else if (value > -1.7e308 && value < 1.7e308) {
                return NPY_DOUBLE;
            }
            break;
        }
        /*
         * The code to demote complex to float is disabled for now,
         * as forcing complex by adding 0j is probably desirable.
         */
        case NPY_CFLOAT: {
            /*
            npy_cfloat value = *(npy_cfloat *)valueptr;
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_FLOAT, is_small_unsigned);
            }
            */
            break;
        }
        case NPY_CDOUBLE: {
            npy_cdouble value = *(npy_cdouble *)valueptr;
            /*
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_DOUBLE, is_small_unsigned);
            }
            */
            if (value.real > -3.4e38 && value.real < 3.4e38 &&
                     value.imag > -3.4e38 && value.imag < 3.4e38) {
                return NPY_CFLOAT;
            }
            break;
        }
        case NPY_CLONGDOUBLE: {
            npy_clongdouble value = *(npy_clongdouble *)valueptr;
            /*
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_LONGDOUBLE, is_small_unsigned);
            }
            */
            if (value.real > -3.4e38 && value.real < 3.4e38 &&
                     value.imag > -3.4e38 && value.imag < 3.4e38) {
                return NPY_CFLOAT;
            }
            else if (value.real > -1.7e308 && value.real < 1.7e308 &&
                     value.imag > -1.7e308 && value.imag < 1.7e308) {
                return NPY_CDOUBLE;
            }
            break;
        }
    }

    return type_num;
}


NPY_NO_EXPORT HPy
HPyArray_MinScalarType_internal(HPyContext *ctx, HPy h_arr, int *is_small_unsigned)
{
    PyArrayObject *arr = PyArrayObject_AsStruct(ctx, h_arr);
    HPy h_dtype = HPyArray_DESCR(ctx, h_arr, arr);
    PyArray_Descr *dtype = PyArray_Descr_AsStruct(ctx, h_dtype);
    *is_small_unsigned = 0;
    /*
     * If the array isn't a numeric scalar, just return the array's dtype.
     */
    if (PyArray_NDIM(arr) > 0 || !PyTypeNum_ISNUMBER(dtype->type_num)) {
        return h_dtype;
    }
    else {
        char *data = PyArray_BYTES(arr);
        int swap = !PyArray_ISNBO(dtype->byteorder);
        /* An aligned memory buffer large enough to hold any type */
        npy_longlong value[4];
        dtype->f->copyswap(&value, data, swap, NULL);

        HPy ret = HPyArray_DescrFromType(ctx,
                        min_scalar_type_num((char *)&value,
                                dtype->type_num, is_small_unsigned));
        HPy_Close(ctx, h_dtype);
        return ret;

    }
}

/*NUMPY_API
 * If arr is a scalar (has 0 dimensions) with a built-in number data type,
 * finds the smallest type size/kind which can still represent its data.
 * Otherwise, returns the array's data type.
 *
 */
NPY_NO_EXPORT HPy
PyArray_MinScalarType(PyArrayObject *arr)
{
    HPyContext *ctx = npy_get_context();
    HPy h_arr = HPy_FromPyObject(ctx, (PyObject *)arr);
    HPy ret = HPyArray_MinScalarType(ctx, h_arr);
    HPy_Close(ctx, h_arr);
    return ret;
}

/*HPY_NUMPY_API
 * If arr is a scalar (has 0 dimensions) with a built-in number data type,
 * finds the smallest type size/kind which can still represent its data.
 * Otherwise, returns the array's data type.
 *
 */
NPY_NO_EXPORT HPy
HPyArray_MinScalarType(HPyContext *ctx, HPy h_arr)
{
    int is_small_unsigned;
    return HPyArray_MinScalarType_internal(ctx, h_arr, &is_small_unsigned);
}

/*
 * Provides an ordering for the dtype 'kind' character codes, to help
 * determine when to use the min_scalar_type function. This groups
 * 'kind' into boolean, integer, floating point, and everything else.
 */
static int
dtype_kind_to_simplified_ordering(char kind)
{
    switch (kind) {
        /* Boolean kind */
        case 'b':
            return 0;
        /* Unsigned int kind */
        case 'u':
        /* Signed int kind */
        case 'i':
            return 1;
        /* Float kind */
        case 'f':
        /* Complex kind */
        case 'c':
            return 2;
        /* Anything else */
        default:
            return 3;
    }
}


/*
 * Determine if there is a mix of scalars and arrays/dtypes.
 * If this is the case, the scalars should be handled as the minimum type
 * capable of holding the value when the maximum "category" of the scalars
 * surpasses the maximum "category" of the arrays/dtypes.
 * If the scalars are of a lower or same category as the arrays, they may be
 * demoted to a lower type within their category (the lowest type they can
 * be cast to safely according to scalar casting rules).
 *
 * If any new style dtype is involved (non-legacy), always returns 0.
 */
NPY_NO_EXPORT int
should_use_min_scalar(npy_intp narrs, PyArrayObject **arr,
                      npy_intp ndtypes, PyArray_Descr **dtypes)
{
    HPyContext *ctx = npy_get_context();
    HPy *h_arr = HPy_FromPyObjectArray(ctx, (PyObject **)arr, (Py_ssize_t)narrs);
    HPy *h_dtypes = HPy_FromPyObjectArray(ctx, (PyObject **)dtypes, (Py_ssize_t)ndtypes);
    int res = hshould_use_min_scalar(ctx, narrs, h_arr, ndtypes, h_dtypes);
    HPy_CloseAndFreeArray(ctx, h_arr, (HPy_ssize_t)narrs);
    HPy_CloseAndFreeArray(ctx, h_dtypes, (HPy_ssize_t)ndtypes);
    return res;
}

NPY_NO_EXPORT int
hshould_use_min_scalar(HPyContext *ctx, npy_intp narrs, HPy *arr,
                      npy_intp ndtypes, HPy *dtypes)
{
    int use_min_scalar = 0;
    HPy descr;
    HPy descr_type;

    if (narrs > 0) {
        int all_scalars;
        int max_scalar_kind = -1;
        int max_array_kind = -1;

        all_scalars = (ndtypes > 0) ? 0 : 1;

        /* Compute the maximum "kinds" and whether everything is scalar */
        for (npy_intp i = 0; i < narrs; ++i) {
            descr = HPyArray_GetDescr(ctx, arr[i]);
            descr_type = HNPY_DTYPE(ctx, descr);

            if (!NPY_DT_is_legacy(PyArray_DTypeMeta_AsStruct(ctx, descr_type))) {
                HPy_Close(ctx, descr);
                HPy_Close(ctx, descr_type);
                return 0;
            }
            if (HPyArray_GetNDim(ctx, arr[i]) == 0) {
                int kind = dtype_kind_to_simplified_ordering(
                                    PyArray_Descr_AsStruct(ctx, descr)->kind);
                if (kind > max_scalar_kind) {
                    max_scalar_kind = kind;
                }
            }
            else {
                int kind = dtype_kind_to_simplified_ordering(
                                    PyArray_Descr_AsStruct(ctx, descr)->kind);
                if (kind > max_array_kind) {
                    max_array_kind = kind;
                }
                all_scalars = 0;
            }
            HPy_Close(ctx, descr);
            HPy_Close(ctx, descr_type);
        }
        /*
         * If the max scalar kind is bigger than the max array kind,
         * finish computing the max array kind
         */
        for (npy_intp i = 0; i < ndtypes; ++i) {
            descr_type = HNPY_DTYPE(ctx, dtypes[i]);
            int is_legacy = NPY_DT_is_legacy(PyArray_DTypeMeta_AsStruct(ctx, descr_type));
            HPy_Close(ctx, descr_type);
            if (!is_legacy) {
                return 0;
            }
            int kind = dtype_kind_to_simplified_ordering(
                    PyArray_Descr_AsStruct(ctx, dtypes[i])->kind);
            if (kind > max_array_kind) {
                max_array_kind = kind;
            }
        }

        /* Indicate whether to use the min_scalar_type function */
        if (!all_scalars && max_array_kind >= max_scalar_kind) {
            use_min_scalar = 1;
        }
    }
    return use_min_scalar;
}


/*
 * Utility function used only in PyArray_ResultType for value-based logic.
 * See that function for the meaning and contents of the parameters.
 */
static HPy // PyArray_Descr *
hpy_get_descr_from_cast_or_value(HPyContext *ctx,
        npy_intp i,
        HPy arrs[], /* PyArrayObject * */
        npy_intp ndtypes,
        HPy descriptor, /* PyArray_Descr * */
        HPy common_dtype) /* PyArray_DTypeMeta * */
{
    HPy curr;
    if (NPY_LIKELY(i < ndtypes ||
            !(PyArray_FLAGS(PyArrayObject_AsStruct(ctx, arrs[i-ndtypes])) & _NPY_ARRAY_WAS_PYSCALAR))) {
        curr = HPyArray_CastDescrToDType(ctx, descriptor, common_dtype);
    }
    else {
        /*
         * Unlike `PyArray_CastToDTypeAndPromoteDescriptors`, deal with
         * plain Python values "graciously". This recovers the original
         * value the long route, but it should almost never happen...
         */
        HPy h_tmp = HPyArray_GETITEM(ctx, arrs[i-ndtypes],
                                        PyArray_BYTES(PyArrayObject_AsStruct(ctx, arrs[i-ndtypes])));
        if (HPy_IsNull(h_tmp)) {
            return HPy_NULL;
        }
        curr = HNPY_DT_CALL_discover_descr_from_pyobject(ctx, common_dtype, h_tmp);
        HPy_Close(ctx, h_tmp);
    }
    return curr;
}

/*NUMPY_API
 *
 * Produces the result type of a bunch of inputs, using the same rules
 * as `np.result_type`.
 *
 * NOTE: This function is expected to through a transitional period or
 *       change behaviour.  DTypes should always be strictly enforced for
 *       0-D arrays, while "weak DTypes" will be used to represent Python
 *       integers, floats, and complex in all cases.
 *       (Within this function, these are currently flagged on the array
 *       object to work through `np.result_type`, this may change.)
 *
 *       Until a time where this transition is complete, we probably cannot
 *       add new "weak DTypes" or allow users to create their own.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_ResultType(
        npy_intp narrs, PyArrayObject *arrs[],
        npy_intp ndtypes, PyArray_Descr *descrs[])
{
    HPyContext *ctx = npy_get_context();
    HPy *h_arrs = HPy_FromPyObjectArray(ctx, arrs, narrs);
    HPy *h_descrs = HPy_FromPyObjectArray(ctx, descrs, ndtypes);
    HPy h_result = HPyArray_ResultType(ctx,
            narrs, h_arrs, ndtypes, h_descrs);
    PyArray_Descr *result = (PyArray_Descr *)HPy_AsPyObject(ctx, h_result);
    HPy_Close(ctx, h_result);
    HPy_CloseAndFreeArray(ctx, h_descrs, ndtypes);
    HPy_CloseAndFreeArray(ctx, h_arrs, narrs);
    return result;
}

/*HPY_NUMPY_API
 *
 * Produces the result type of a bunch of inputs, using the same rules
 * as `np.result_type`.
 *
 * NOTE: This function is expected to through a transitional period or
 *       change behaviour.  DTypes should always be strictly enforced for
 *       0-D arrays, while "weak DTypes" will be used to represent Python
 *       integers, floats, and complex in all cases.
 *       (Within this function, these are currently flagged on the array
 *       object to work through `np.result_type`, this may change.)
 *
 *       Until a time where this transition is complete, we probably cannot
 *       add new "weak DTypes" or allow users to create their own.
 */
NPY_NO_EXPORT HPy
HPyArray_ResultType(HPyContext *ctx,
        npy_intp narrs, HPy /* (PyArrayObject *) */ arrs[],
        npy_intp ndtypes, HPy /* (PyArray_Descr *) */ descrs[])
{
    HPy result = HPy_NULL; /* (PyArray_Descr *) */

    if (narrs + ndtypes <= 1) {
        /* If the input is a single value, skip promotion. */
        if (narrs == 1) {
            result = HPyArray_DTYPE(ctx, arrs[0]);
        }
        else if (ndtypes == 1) {
            result = HPy_Dup(ctx, descrs[0]);
        }
        else {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "no arrays or types available to calculate result type");
            return HPy_NULL;
        }
        HPy_SETREF(ctx, result, hensure_dtype_nbo(ctx, result));
        return result;
    }

    HPy *info_on_heap = NULL;
    HPy _info_on_stack[NPY_MAXARGS * 2];
    HPy *all_DTypes; /* (PyArray_DTypeMeta *) */
    HPy *all_descriptors; /* (PyArray_Descr *) */

    if (narrs + ndtypes > NPY_MAXARGS) {
        // TODO HPY LABS PORT: PyMem_Malloc
        // info_on_heap = PyMem_Malloc(2 * (narrs+ndtypes) * sizeof(PyObject *));
        info_on_heap = (HPy *)malloc(2 * (narrs+ndtypes) * sizeof(HPy));
        if (info_on_heap == NULL) {
            HPyErr_NoMemory(ctx);
            return HPy_NULL;
        }
        all_DTypes = info_on_heap;
        all_descriptors = (info_on_heap + narrs + ndtypes);
    }
    else {
        all_DTypes = _info_on_stack;
        all_descriptors = (_info_on_stack + narrs + ndtypes);
    }

    /* Copy all dtypes into a single array defining non-value-based behaviour */
    for (npy_intp i=0; i < ndtypes; i++) {
        all_DTypes[i] = HNPY_DTYPE(ctx, descrs[i]); /* new ref */
         // Py_INCREF(all_DTypes[i_all]);
        all_descriptors[i] = HPy_Dup(ctx, descrs[i]);
    }

    int at_least_one_scalar = 0;
    int all_pyscalar = ndtypes == 0;
    for (npy_intp i=0, i_all=ndtypes; i < narrs; i++, i_all++) {
        PyArrayObject *arrs_i_data = PyArrayObject_AsStruct(ctx, arrs[i]);
        /* Array descr is also the correct "default" for scalars: */
        if (PyArray_NDIM(arrs_i_data) == 0) {
            at_least_one_scalar = 1;
        }

        if (!(PyArray_FLAGS(arrs_i_data) & _NPY_ARRAY_WAS_PYSCALAR)) {
            /* This was not a scalar with an abstract DType */
            all_descriptors[i_all] = HPyArray_DTYPE(ctx, arrs[i]);
            all_DTypes[i_all] = HNPY_DTYPE(ctx, all_descriptors[i_all]); /* new ref */
            // Py_INCREF(all_DTypes[i_all]);
            all_pyscalar = 0;
            continue;
        }

        /*
         * The original was a Python scalar with an abstract DType.
         * In a future world, this type of code may need to work on the
         * DType level first and discover those from the original value.
         * But, right now we limit the logic to int, float, and complex
         * and do it here to allow for a transition without losing all of
         * our remaining sanity.
         */
        if (PyArray_ISFLOAT(arrs_i_data)) {
            all_DTypes[i_all] = HPyGlobal_Load(ctx, HPyArray_PyFloatAbstractDType);
        }
        else if (PyArray_ISCOMPLEX(arrs_i_data)) {
            all_DTypes[i_all] = HPyGlobal_Load(ctx, HPyArray_PyComplexAbstractDType);
        }
        else {
            /* N.B.: Could even be an object dtype here for large ints */
            all_DTypes[i_all] = HPyGlobal_Load(ctx, HPyArray_PyIntAbstractDType);
        }
        // Py_INCREF(all_DTypes[i_all]);
        /*
         * Leave the descriptor empty, if we need it, we will have to go
         * to more extreme lengths unfortunately.
         */
        all_descriptors[i_all] = HPy_NULL;
    }

    HPy common_dtype = HPyArray_PromoteDTypeSequence(ctx, 
            narrs+ndtypes, all_DTypes);
    for (npy_intp i=0; i < narrs+ndtypes; i++) {
        HPy_Close(ctx, all_DTypes[i]);
    }
    if (HPy_IsNull(common_dtype)) {
        goto error;
    }

    PyArray_DTypeMeta *common_dtype_data = PyArray_DTypeMeta_AsStruct(ctx, common_dtype);
    if (NPY_DT_is_abstract(common_dtype_data)) {
        /* (ab)use default descriptor to define a default */
        HPy tmp_descr = HNPY_DT_CALL_default_descr(ctx, common_dtype, common_dtype_data);
        if (HPy_IsNull(tmp_descr)) {
            goto error;
        }
        HPy_SETREF(ctx, common_dtype, HNPY_DTYPE(ctx, tmp_descr));
        HPy_Close(ctx, tmp_descr);
    }

    /*
     * NOTE: Code duplicates `PyArray_CastToDTypeAndPromoteDescriptors`, but
     *       supports special handling of the abstract values.
     */
    if (!NPY_DT_is_parametric(common_dtype_data)) {
        /* Note that this "fast" path loses all metadata */
        result = HNPY_DT_CALL_default_descr(ctx, common_dtype, common_dtype_data);
    }
    else {
        result = hpy_get_descr_from_cast_or_value(ctx,
                    0, arrs, ndtypes, all_descriptors[0], common_dtype);
        if (HPy_IsNull(result)) {
            goto error;
        }

        for (npy_intp i = 1; i < ndtypes+narrs; i++) {
            HPy curr = hpy_get_descr_from_cast_or_value(ctx,
                    i, arrs, ndtypes, all_descriptors[i], common_dtype);
            if (HPy_IsNull(curr)) {
                goto error;
            }
            PyObject *py_result = HPy_AsPyObject(ctx, result);
            PyObject *py_curr = HPy_AsPyObject(ctx, curr);
            CAPI_WARN("HPyArray_ResultType: calling HNPY_DT_SLOTS(ctx, common_dtype)->common_instance");
            py_result = HNPY_DT_SLOTS(ctx, common_dtype)->common_instance(py_result, py_curr);
            HPy_SETREF(ctx, result, HPy_FromPyObject(ctx, py_result));
            Py_DECREF(py_curr);
            HPy_Close(ctx, curr);
            if (HPy_IsNull(result)) {
                goto error;
            }
            Py_DECREF(py_result);
        }
    }

    /*
     * Unfortunately, when 0-D "scalar" arrays are involved and mixed, we
     * have to use the value-based logic.  The intention is to move away from
     * the complex logic arising from it.  We thus fall back to the legacy
     * version here.
     * It may be possible to micro-optimize this to skip some of the above
     * logic when this path is necessary.
     */
    if (at_least_one_scalar && !all_pyscalar && PyArray_Descr_AsStruct(ctx, result)->type_num < NPY_NTYPES) {
        HPy legacy_result = HPyArray_LegacyResultType(ctx,
                narrs, arrs, ndtypes, descrs);
        if (HPy_IsNull(legacy_result)) {
            /*
             * Going from error to success should not really happen, but is
             * probably OK if it does.
             */
            goto error;
        }
        /* Return the old "legacy" result (could warn here if different) */
        HPy_SETREF(ctx, result, legacy_result);
    }

    for(int i=0; i < narrs+ndtypes; i++) {
        HPy_Close(ctx, all_descriptors[i]);
    }

    HPy_Close(ctx, common_dtype);
    // TODO HPY LABS PORT: PyMem_Free
    MEM_FREE(info_on_heap);
    return result;

  error:
    for(int i=0; i < narrs+ndtypes; i++) {
        HPy_Close(ctx, all_descriptors[i]);
    }
    HPy_Close(ctx, result);
    HPy_Close(ctx, common_dtype);
    // TODO HPY LABS PORT: PyMem_Free
    MEM_FREE(info_on_heap);
    return HPy_NULL;
}


/*
 * Produces the result type of a bunch of inputs, using the UFunc
 * type promotion rules. Use this function when you have a set of
 * input arrays, and need to determine an output array dtype.
 *
 * If all the inputs are scalars (have 0 dimensions) or the maximum "kind"
 * of the scalars is greater than the maximum "kind" of the arrays, does
 * a regular type promotion.
 *
 * Otherwise, does a type promotion on the MinScalarType
 * of all the inputs.  Data types passed directly are treated as array
 * types.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_LegacyResultType(
        npy_intp narrs, PyArrayObject **arr,
        npy_intp ndtypes, PyArray_Descr **dtypes)
{
    HPyContext *ctx = npy_get_context();
    HPy *h_arr = HPy_FromPyObjectArray(ctx, (PyObject **)arr, (Py_ssize_t)narrs);
    HPy *h_dtypes = HPy_FromPyObjectArray(ctx, (PyObject **)dtypes, (Py_ssize_t)ndtypes);
    HPy h_res = HPyArray_LegacyResultType(ctx, narrs, h_arr, ndtypes, h_dtypes);
    PyArray_Descr *res = (PyArray_Descr *)HPy_AsPyObject(ctx, h_res);
    HPy_CloseAndFreeArray(ctx, h_arr, (HPy_ssize_t)narrs);
    HPy_CloseAndFreeArray(ctx, h_dtypes, (HPy_ssize_t)ndtypes);
    HPy_Close(ctx, h_res);
    return res;
}

NPY_NO_EXPORT HPy
HPyArray_LegacyResultType(HPyContext *ctx,
        npy_intp narrs, HPy *arr,
        npy_intp ndtypes, HPy *dtypes)
{
    npy_intp i;

    /* If there's just one type, pass it through */
    if (narrs + ndtypes == 1) {
        HPy ret = HPy_NULL;
        if (narrs == 1) {
            ret = HPyArray_DESCR(ctx, arr[0], PyArrayObject_AsStruct(ctx, arr[0]));
        }
        else {
            ret = HPy_Dup(ctx, dtypes[0]);
        }
        return ret;
    }

    int use_min_scalar = hshould_use_min_scalar(ctx, narrs, arr, ndtypes, dtypes);

    /* Loop through all the types, promoting them */
    if (!use_min_scalar) {
        HPy ret;

        /* Build a single array of all the dtypes */
        HPy *all_dtypes = (HPy *)malloc(
            sizeof(all_dtypes) * (narrs + ndtypes));
        if (all_dtypes == NULL) {
            HPyErr_NoMemory(ctx);
            return HPy_NULL;
        }
        for (i = 0; i < narrs; ++i) {
            all_dtypes[i] = HPyArray_DESCR(ctx, arr[i], PyArrayObject_AsStruct(ctx, arr[i]));
        }
        for (i = 0; i < ndtypes; ++i) {
            all_dtypes[narrs + i] = dtypes[i];
        }
        ret = HPyArray_PromoteTypeSequence(ctx, all_dtypes, narrs + ndtypes);
        for (i = 0; i < narrs; ++i) {
            HPy_Close(ctx, all_dtypes[i]);
        }
        PyArray_free(all_dtypes);
        return ret;
    }
    else {
        int ret_is_small_unsigned = 0;
        HPy ret = HPy_NULL;

        for (i = 0; i < narrs; ++i) {
            int tmp_is_small_unsigned;
            HPy tmp = HPyArray_MinScalarType_internal(ctx,
                arr[i], &tmp_is_small_unsigned);
            if (HPy_IsNull(tmp)) {
                HPy_Close(ctx, ret);
                return HPy_NULL;
            }
            /* Combine it with the existing type */
            if (HPy_IsNull(ret)) {
                ret = tmp;
                ret_is_small_unsigned = tmp_is_small_unsigned;
            }
            else {
                HPy tmpret = hpy_promote_types(ctx,
                    tmp, ret, tmp_is_small_unsigned, ret_is_small_unsigned);
                HPy_Close(ctx, tmp);
                HPy_Close(ctx, ret);
                ret = tmpret;
                if (HPy_IsNull(ret)) {
                    return HPy_NULL;
                }

                ret_is_small_unsigned = tmp_is_small_unsigned &&
                                        ret_is_small_unsigned;
            }
        }

        for (i = 0; i < ndtypes; ++i) {
            HPy tmp = dtypes[i];
            /* Combine it with the existing type */
            if (HPy_IsNull(ret)) {
                ret = HPy_Dup(ctx, tmp);
            }
            else {
                HPy tmpret = hpy_promote_types(ctx,
                    tmp, ret, 0, ret_is_small_unsigned);
                HPy_Close(ctx, ret);
                ret = tmpret;
                if (HPy_IsNull(ret)) {
                    return HPy_NULL;
                }
            }
        }
        /* None of the above loops ran */
        if (HPy_IsNull(ret)) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "no arrays or types available to calculate result type");
        }

        return ret;
    }
}

/**
 * Promotion of descriptors (of arbitrary DType) to their correctly
 * promoted instances of the given DType.
 * I.e. the given DType could be a string, which then finds the correct
 * string length, given all `descrs`.
 *
 * @param ndescrs number of descriptors to cast and find the common instance.
 *        At least one must be passed in.
 * @param descrs The descriptors to work with.
 * @param DType The DType of the desired output descriptor.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_CastToDTypeAndPromoteDescriptors(
        npy_intp ndescr, PyArray_Descr *descrs[], PyArray_DTypeMeta *DType)
{
    assert(ndescr > 0);

    PyArray_Descr *result = PyArray_CastDescrToDType(descrs[0], DType);
    if (result == NULL || ndescr == 1) {
        return result;
    }
    if (!NPY_DT_is_parametric(DType)) {
        /* Note that this "fast" path loses all metadata */
        Py_DECREF(result);
        return NPY_DT_CALL_default_descr(DType);
    }

    for (npy_intp i = 1; i < ndescr; i++) {
        PyArray_Descr *curr = PyArray_CastDescrToDType(descrs[i], DType);
        if (curr == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        Py_SETREF(result, NPY_DT_SLOTS(DType)->common_instance(result, curr));
        Py_DECREF(curr);
        if (result == NULL) {
            return NULL;
        }
    }
    return result;
}


/*NUMPY_API
 * Is the typenum valid?
 */
NPY_NO_EXPORT int
PyArray_ValidType(int type)
{
    PyArray_Descr *descr;
    int res=NPY_TRUE;

    descr = PyArray_DescrFromType(type);
    if (descr == NULL) {
        res = NPY_FALSE;
    }
    Py_DECREF(descr);
    return res;
}

/* Backward compatibility only */
/* In both Zero and One

***You must free the memory once you are done with it
using PyDataMem_FREE(ptr) or you create a memory leak***

If arr is an Object array you are getting a
BORROWED reference to Zero or One.
Do not DECREF.
Please INCREF if you will be hanging on to it.

The memory for the ptr still must be freed in any case;
*/

static int
_check_object_rec(PyArray_Descr *descr)
{
    if (PyDataType_HASFIELDS(descr) && PyDataType_REFCHK(descr)) {
        PyErr_SetString(PyExc_TypeError, "Not supported for this data-type.");
        return -1;
    }
    return 0;
}

/*NUMPY_API
  Get pointer to zero of correct type for array.
*/
NPY_NO_EXPORT char *
PyArray_Zero(PyArrayObject *arr)
{
    char *zeroval;
    int ret, storeflags;
    static PyObject * zero_obj = NULL;

    if (_check_object_rec(PyArray_DESCR(arr)) < 0) {
        return NULL;
    }
    zeroval = PyDataMem_NEW(PyArray_DESCR(arr)->elsize);
    if (zeroval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (zero_obj == NULL) {
        zero_obj = PyLong_FromLong((long) 0);
        if (zero_obj == NULL) {
            return NULL;
        }
    }
    if (PyArray_ISOBJECT(arr)) {
        /* XXX this is dangerous, the caller probably is not
           aware that zeroval is actually a static PyObject*
           In the best case they will only use it as-is, but
           if they simply memcpy it into a ndarray without using
           setitem(), refcount errors will occur
        */
        memcpy(zeroval, &zero_obj, sizeof(PyObject *));
        return zeroval;
    }
    storeflags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_BEHAVED);
    ret = PyArray_SETITEM(arr, zeroval, zero_obj);
    ((PyArrayObject_fields *)arr)->flags = storeflags;
    if (ret < 0) {
        PyDataMem_FREE(zeroval);
        return NULL;
    }
    return zeroval;
}

/*NUMPY_API
  Get pointer to one of correct type for array
*/
NPY_NO_EXPORT char *
PyArray_One(PyArrayObject *arr)
{
    char *oneval;
    int ret, storeflags;
    static PyObject * one_obj = NULL;

    if (_check_object_rec(PyArray_DESCR(arr)) < 0) {
        return NULL;
    }
    oneval = PyDataMem_NEW(PyArray_DESCR(arr)->elsize);
    if (oneval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (one_obj == NULL) {
        one_obj = PyLong_FromLong((long) 1);
        if (one_obj == NULL) {
            return NULL;
        }
    }
    if (PyArray_ISOBJECT(arr)) {
        /* XXX this is dangerous, the caller probably is not
           aware that oneval is actually a static PyObject*
           In the best case they will only use it as-is, but
           if they simply memcpy it into a ndarray without using
           setitem(), refcount errors will occur
        */
        memcpy(oneval, &one_obj, sizeof(PyObject *));
        return oneval;
    }

    storeflags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_BEHAVED);
    ret = PyArray_SETITEM(arr, oneval, one_obj);
    ((PyArrayObject_fields *)arr)->flags = storeflags;
    if (ret < 0) {
        PyDataMem_FREE(oneval);
        return NULL;
    }
    return oneval;
}

/* End deprecated */

/*NUMPY_API
 * Return the typecode of the array a Python object would be converted to
 *
 * Returns the type number the result should have, or NPY_NOTYPE on error.
 */
NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type)
{
    PyArray_Descr *dtype = NULL;
    int ret;

    if (minimum_type != NPY_NOTYPE && minimum_type >= 0) {
        dtype = PyArray_DescrFromType(minimum_type);
        if (dtype == NULL) {
            return NPY_NOTYPE;
        }
    }
    if (PyArray_DTypeFromObject(op, NPY_MAXDIMS, &dtype) < 0) {
        return NPY_NOTYPE;
    }

    if (dtype == NULL) {
        ret = NPY_DEFAULT_TYPE;
    }
    else if (!NPY_DT_is_legacy(NPY_DTYPE(dtype))) {
        /*
         * TODO: If we keep all type number style API working, by defining
         *       type numbers always. We may be able to allow this again.
         */
        PyErr_Format(PyExc_TypeError,
                "This function currently only supports native NumPy dtypes "
                "and old-style user dtypes, but the dtype was %S.\n"
                "(The function may need to be updated to support arbitrary"
                "user dtypes.)",
                dtype);
        ret = NPY_NOTYPE;
    }
    else {
        ret = dtype->type_num;
    }

    Py_XDECREF(dtype);

    return ret;
}

/* Raises error when len(op) == 0 */

/*NUMPY_API
 *
 * This function is only used in one place within NumPy and should
 * generally be avoided. It is provided mainly for backward compatibility.
 *
 * The user of the function has to free the returned array with PyDataMem_FREE.
 */
NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn)
{
    int i, n;
    PyArray_Descr *common_descr = NULL;
    PyArrayObject **mps = NULL;

    *retn = n = PySequence_Length(op);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "0-length sequence.");
    }
    if (PyErr_Occurred()) {
        *retn = 0;
        return NULL;
    }
    mps = (PyArrayObject **)PyDataMem_NEW(n*sizeof(PyArrayObject *));
    if (mps == NULL) {
        *retn = 0;
        return (void*)PyErr_NoMemory();
    }

    if (PyArray_Check(op)) {
        for (i = 0; i < n; i++) {
            mps[i] = (PyArrayObject *) array_item_asarray((PyArrayObject *)op, i);
        }
        if (!PyArray_ISCARRAY((PyArrayObject *)op)) {
            for (i = 0; i < n; i++) {
                PyObject *obj;
                obj = PyArray_NewCopy(mps[i], NPY_CORDER);
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)obj;
            }
        }
        return mps;
    }

    for (i = 0; i < n; i++) {
        mps[i] = NULL;
    }

    for (i = 0; i < n; i++) {
        /* Convert everything to an array, this could be optimized away */
        PyObject *tmp = PySequence_GetItem(op, i);
        if (tmp == NULL) {
            goto fail;
        }

        mps[i] = (PyArrayObject *)PyArray_FROM_O(tmp);
        Py_DECREF(tmp);
        if (mps[i] == NULL) {
            goto fail;
        }
    }

    common_descr = PyArray_ResultType(n, mps, 0, NULL);
    if (common_descr == NULL) {
        goto fail;
    }

    /* Make sure all arrays are contiguous and have the correct dtype. */
    for (i = 0; i < n; i++) {
        int flags = NPY_ARRAY_CARRAY;
        PyArrayObject *tmp = mps[i];

        Py_INCREF(common_descr);
        mps[i] = (PyArrayObject *)PyArray_FromArray(tmp, common_descr, flags);
        Py_DECREF(tmp);
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    Py_DECREF(common_descr);
    return mps;

 fail:
    Py_XDECREF(common_descr);
    *retn = 0;
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return NULL;
}


/**
 * Private function to add a casting implementation by unwrapping a bound
 * array method.
 *
 * @param meth
 * @return 0 on success -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AddCastingImplementation(PyBoundArrayMethodObject *meth)
{
    HPyContext *ctx = npy_get_context();
    HPy h_meth = HPy_FromPyObject(ctx, (PyObject *)meth);
    int res = HPyArray_AddCastingImplementation(ctx, h_meth);
    HPy_Close(ctx, h_meth);
    return res;
}

NPY_NO_EXPORT int
HPyArray_AddCastingImplementation(HPyContext *ctx, HPy bmeth)
{
    PyBoundArrayMethodObject *bmeth_data = PyBoundArrayMethodObject_AsStruct(ctx, bmeth);
    HPy h_method = HPyField_Load(ctx, bmeth, bmeth_data->method);
    PyArrayMethodObject *method_data = PyArrayMethodObject_AsStruct(ctx, h_method);
    if (method_data->nin != 1 || method_data->nout != 1) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "A cast must have one input and one output.");
        return -1;
    }
    HPy h_castingimpls = HPy_NULL;
    HPy h_dtype = HPyField_Load(ctx, bmeth, bmeth_data->dtypes[0]);
    HPy h_dtype1 = HPyField_Load(ctx, bmeth, bmeth_data->dtypes[1]);
    if (HPy_Is(ctx, h_dtype, h_dtype1)) {
        /*
         * The method casting between instances of the same dtype is special,
         * since it is common, it is stored explicitly (currently) and must
         * obey additional constraints to ensure convenient casting.
         */
        if (!(method_data->flags & NPY_METH_SUPPORTS_UNALIGNED)) {
            HPyErr_Format_p(ctx, ctx->h_TypeError,
                    "A cast where input and output DType (class) are identical "
                    "must currently support unaligned data. (method: %s)",
                    method_data->name);
            goto fail;
        }
        if (!HPyField_IsNull(HNPY_DT_SLOTS(ctx, h_dtype)->within_dtype_castingimpl)) {
            // TODO HPY LABS PORT: PyErr_Format
            // HPyErr_SetString(ctx, ctx->h_TypeError,
            //        "A cast was already added for %S -> %S. (method: %s)",
            //        meth->dtypes[0], meth->dtypes[1], meth->method->name);
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "A cast was already added for %S -> %S. (method: %s)");
            goto fail;
        }

        HPyField_Store(ctx, h_dtype, &HNPY_DT_SLOTS(ctx, h_dtype)->within_dtype_castingimpl, h_method);
        goto success;
    }
    h_castingimpls = HPY_DTYPE_SLOTS_CASTINGIMPL0(ctx, h_dtype);
    int c = HPy_Contains(ctx, h_castingimpls, h_dtype1);
    if (c) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                 "A cast was already added for %S -> %S. (method: %s)");
        // PyErr_Format(PyExc_RuntimeError,
        //         "A cast was already added for %S -> %S. (method: %s)",
        //         meth->dtypes[0], meth->dtypes[1], meth->method->name);
        goto fail;
    }
    if (HPy_SetItem(ctx, h_castingimpls, h_dtype1, h_method) < 0) {
        goto fail;
    }

success:
    HPy_Close(ctx, h_method);
    HPy_Close(ctx, h_castingimpls);
    HPy_Close(ctx, h_dtype);
    HPy_Close(ctx, h_dtype1);
    return 0;

fail:
    HPy_Close(ctx, h_method);
    HPy_Close(ctx, h_castingimpls);
    HPy_Close(ctx, h_dtype);
    HPy_Close(ctx, h_dtype1);
    return -1;
}

/**
 * Add a new casting implementation using a PyArrayMethod_Spec.
 *
 * @param spec
 * @param private If private, allow slots not publicly exposed.
 * @return 0 on success -1 on failure
 */
NPY_NO_EXPORT int
PyArray_AddCastingImplementation_FromSpec(PyArrayMethod_Spec *spec, int private)
{
    /* Create a bound method, unbind and store it */
    PyBoundArrayMethodObject *meth = PyArrayMethod_FromSpec_int(spec, private);
    if (meth == NULL) {
        return -1;
    }
    int res = PyArray_AddCastingImplementation(meth);
    Py_DECREF(meth);
    if (res < 0) {
        return -1;
    }
    return 0;
}

NPY_NO_EXPORT int
HPyArray_AddCastingImplementation_FromSpec(HPyContext *ctx, PyArrayMethod_Spec *spec, int private)
{
    /* Create a bound method, unbind and store it */
    HPy meth = HPyArrayMethod_FromSpec_int(ctx, spec, private);
    if (HPy_IsNull(meth)) {
        return -1;
    }
    int res = HPyArray_AddCastingImplementation(ctx, meth);
    HPy_Close(ctx, meth);
    if (res < 0) {
        return -1;
    }
    return 0;
}


NPY_NO_EXPORT NPY_CASTING
legacy_same_dtype_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), /* PyArrayMethodObject *method */
        HPy NPY_UNUSED(dtypes[2]), /* PyArray_DTypeMeta *dtypes[2] */
        HPy given_descrs[2], /* PyArray_Descr *given_descrs[2] */
        HPy loop_descrs[2], /* PyArray_Descr *output_descrs[2] */
        npy_intp *view_offset)
{
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);

    if (HPy_IsNull(given_descrs[1])) {
        loop_descrs[1] = hensure_dtype_nbo(ctx, loop_descrs[0]);
        if (HPy_IsNull(loop_descrs[1])) {
            HPy_Close(ctx, loop_descrs[0]);
            return -1;
        }
    }
    else {
        loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);
    }

    /* this function only makes sense for non-flexible legacy dtypes: */
    assert(PyArray_Descr_AsStruct(ctx, loop_descrs[0])->elsize ==
            PyArray_Descr_AsStruct(ctx, loop_descrs[1])->elsize);

    /*
     * Legacy dtypes (except datetime) only have byte-order and elsize as
     * storage parameters.
     */
    if (PyDataType_ISNOTSWAPPED(PyArray_Descr_AsStruct(ctx, loop_descrs[0])) ==
                PyDataType_ISNOTSWAPPED(PyArray_Descr_AsStruct(ctx, loop_descrs[1]))) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }
    return NPY_EQUIV_CASTING;
}

NPY_NO_EXPORT int
legacy_cast_get_strided_loop(
        HPyContext *hctx,
        HPyArrayMethod_Context *context,
        int aligned, int move_references, npy_intp *strides,
        HPyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    HPy *descrs = context->descriptors; /* (PyArray_Descr **) */
    int out_needs_api = 0;

    PyArrayMethodObject *method_data = PyArrayMethodObject_AsStruct(hctx, context->method);
    *flags = method_data->flags & NPY_METH_RUNTIME_FLAGS;

    if (get_wrapped_legacy_cast_function(hctx,
            aligned, strides[0], strides[1], 
            descrs[0], PyArray_Descr_AsStruct(hctx, descrs[0]),
            descrs[1], PyArray_Descr_AsStruct(hctx, descrs[1]),
            move_references, out_loop, out_transferdata, &out_needs_api, 0) < 0) {
        return -1;
    }
    if (!out_needs_api) {
        *flags &= ~NPY_METH_REQUIRES_PYAPI;
    }
    return 0;
}


/*
 * Simple dtype resolver for casting between two different (non-parametric)
 * (legacy) dtypes.
 */
NPY_NO_EXPORT NPY_CASTING
simple_cast_resolve_descriptors(
        HPyContext *ctx,
        HPy self,
        HPy dtypes[2],
        HPy given_descrs[2],
        HPy loop_descrs[2],
        npy_intp *view_offset)
{
    /*
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
     */
    PyArray_DTypeMeta *dtypes_1 = PyArray_DTypeMeta_AsStruct(ctx, dtypes[1]);
    assert(NPY_DT_is_legacy(PyArray_DTypeMeta_AsStruct(ctx, dtypes[0]))
            && NPY_DT_is_legacy(dtypes_1));

    loop_descrs[0] = hensure_dtype_nbo(ctx, given_descrs[0]);
    if (HPy_IsNull(loop_descrs[0])) {
        return -1;
    }
    if (!HPy_IsNull(given_descrs[1])) {
        loop_descrs[1] = hensure_dtype_nbo(ctx, given_descrs[1]);
        if (HPy_IsNull(loop_descrs[1])) {
            HPy_Close(ctx, loop_descrs[0]);
            return -1;
        }
    }
    else {
        loop_descrs[1] = HNPY_DT_CALL_default_descr(ctx, dtypes[1], dtypes_1);
    }

    PyArrayMethodObject *data = PyArrayMethodObject_AsStruct(ctx, self);
    if (data->casting != NPY_NO_CASTING) {
        return data->casting;
    }
    if (PyDataType_ISNOTSWAPPED(PyArray_Descr_AsStruct(ctx, loop_descrs[0])) ==
            PyDataType_ISNOTSWAPPED(PyArray_Descr_AsStruct(ctx, loop_descrs[1]))) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }
    return NPY_EQUIV_CASTING;
}


NPY_NO_EXPORT int
get_byteswap_loop(
        HPyContext *hctx,
        HPyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), npy_intp *strides,
        HPyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    HPy *h_descrs = context->descriptors; /* (PyArray_Descr **) */
    PyArray_Descr *descrs[2] = {
            PyArray_Descr_AsStruct(hctx, h_descrs[0]),
            PyArray_Descr_AsStruct(hctx, h_descrs[1])
    };
    assert(descrs[0]->kind == descrs[1]->kind);
    assert(descrs[0]->elsize == descrs[1]->elsize);
    int itemsize = descrs[0]->elsize;
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_transferdata = NULL;
    if (descrs[0]->kind == 'c') {
        /*
         * TODO: we have an issue with complex, since the below loops
         *       use the itemsize, the complex alignment would be too small.
         *       Using aligned = 0, might cause slow downs in some cases.
         */
        aligned = 0;
    }

    if (PyDataType_ISNOTSWAPPED(descrs[0]) ==
            PyDataType_ISNOTSWAPPED(descrs[1])) {
        *out_loop = PyArray_GetStridedCopyFn(
                aligned, strides[0], strides[1], itemsize);
    }
    else if (!PyTypeNum_ISCOMPLEX(descrs[0]->type_num)) {
        *out_loop = PyArray_GetStridedCopySwapFn(
                aligned, strides[0], strides[1], itemsize);
    }
    else {
        *out_loop = PyArray_GetStridedCopySwapPairFn(
                aligned, strides[0], strides[1], itemsize);
    }
    if (*out_loop == NULL) {
        return -1;
    }
    return 0;
}


NPY_NO_EXPORT int
complex_to_noncomplex_get_loop(
        HPyContext *ctx,
        HPyArrayMethod_Context *context,
        int aligned, int move_references, npy_intp *strides,
        HPyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // TODO HPY LABS PORT: use HPyGlobal
    static HPy cls = HPy_NULL;
    int ret;
    npy_hpy_cache_import(ctx, "numpy.core", "ComplexWarning", &cls);
    if (HPy_IsNull(cls)) {
        return -1;
    }
    ret = HPyErr_WarnEx(ctx, cls,
            "Casting complex values to real discards "
            "the imaginary part", 1);
    if (ret < 0) {
        return -1;
    }
    return npy_default_get_strided_loop(ctx,
            context, aligned, move_references, strides,
            out_loop, out_transferdata, flags);
}


static int
add_numeric_cast(HPyContext *ctx, HPy h_from, HPy h_to)
{
    PyArray_DTypeMeta *from = PyArray_DTypeMeta_AsStruct(ctx, h_from);
    PyArray_DTypeMeta *to = PyArray_DTypeMeta_AsStruct(ctx, h_to);

    PyType_Slot slots[7];
    HPy dtypes[2] = {h_from, h_to}; /* PyArray_DTypeMeta *dtypes[2] */
    PyArrayMethod_Spec spec = {
            .name = "numeric_cast",
            .nin = 1,
            .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .slots = slots,
            .dtypes = dtypes,
    };

    HPy h_from_singleton = HPyField_Load(ctx, h_from, from->singleton);
    HPy h_to_singleton = HPyField_Load(ctx, h_to, to->singleton);

    PyArray_Descr *h_from_singleton_data = PyArray_Descr_AsStruct(ctx, h_from_singleton);
    PyArray_Descr *h_to_singleton_data = PyArray_Descr_AsStruct(ctx, h_to_singleton);

    npy_intp from_itemsize = h_from_singleton_data->elsize;
    npy_intp to_itemsize = h_to_singleton_data->elsize;

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &simple_cast_resolve_descriptors;
    /* Fetch the optimized loops (2<<10 is a non-contiguous stride) */
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = PyArray_GetStridedNumericCastFn(
            1, 2<<10, 2<<10, from->type_num, to->type_num);
    slots[2].slot = NPY_METH_contiguous_loop;
    slots[2].pfunc = PyArray_GetStridedNumericCastFn(
            1, from_itemsize, to_itemsize, from->type_num, to->type_num);
    slots[3].slot = NPY_METH_unaligned_strided_loop;
    slots[3].pfunc = PyArray_GetStridedNumericCastFn(
            0, 2<<10, 2<<10, from->type_num, to->type_num);
    slots[4].slot = NPY_METH_unaligned_contiguous_loop;
    slots[4].pfunc = PyArray_GetStridedNumericCastFn(
            0, from_itemsize, to_itemsize, from->type_num, to->type_num);
    if (PyTypeNum_ISCOMPLEX(from->type_num) &&
            !PyTypeNum_ISCOMPLEX(to->type_num) &&
            !PyTypeNum_ISBOOL(to->type_num)) {
        /*
         * The get_loop function must also give a ComplexWarning. We could
         * consider moving this warning into the inner-loop at some point
         * for simplicity (this requires ensuring it is only emitted once).
         */
        slots[5].slot = NPY_METH_get_loop;
        slots[5].pfunc = &complex_to_noncomplex_get_loop;
        slots[6].slot = 0;
        slots[6].pfunc = NULL;
    }
    else {
        /* Use the default get loop function. */
        slots[5].slot = 0;
        slots[5].pfunc = NULL;
    }

    assert(slots[1].pfunc && slots[2].pfunc && slots[3].pfunc && slots[4].pfunc);

    /* Find the correct casting level, and special case no-cast */
    if (h_from_singleton_data->kind == h_to_singleton_data->kind
            && from_itemsize == to_itemsize) {
        spec.casting = NPY_EQUIV_CASTING;

        /* When there is no casting (equivalent C-types) use byteswap loops */
        slots[0].slot = NPY_METH_resolve_descriptors;
        slots[0].pfunc = &legacy_same_dtype_resolve_descriptors;
        slots[1].slot = NPY_METH_get_loop;
        slots[1].pfunc = &get_byteswap_loop;
        slots[2].slot = 0;
        slots[2].pfunc = NULL;

        spec.name = "numeric_copy_or_byteswap";
        spec.flags |= NPY_METH_NO_FLOATINGPOINT_ERRORS;
    }
    else if (_npy_can_cast_safely_table[from->type_num][to->type_num]) {
        spec.casting = NPY_SAFE_CASTING;
    }
    else if (dtype_kind_to_ordering(h_from_singleton_data->kind) <=
             dtype_kind_to_ordering(h_to_singleton_data->kind)) {
        spec.casting = NPY_SAME_KIND_CASTING;
    }
    else {
        spec.casting = NPY_UNSAFE_CASTING;
    }

    HPy_Close(ctx, h_from_singleton);
    HPy_Close(ctx, h_to_singleton);

    /* Create a bound method, unbind and store it */
    int res = HPyArray_AddCastingImplementation_FromSpec(ctx, &spec, 1);

    return res;
}


/*
 * This registers the castingimpl for all casts between numeric types.
 * Eventually, this function should likely be defined as part of a .c.src
 * file to remove `PyArray_GetStridedNumericCastFn` entirely.
 */
static int
PyArray_InitializeNumericCasts(HPyContext *ctx)
{
    for (int from = 0; from < NPY_NTYPES; from++) {
        if (!PyTypeNum_ISNUMBER(from) && from != NPY_BOOL) {
            continue;
        }
        HPy from_dt = HPyArray_DTypeFromTypeNum(ctx, from);

        for (int to = 0; to < NPY_NTYPES; to++) {
            if (!PyTypeNum_ISNUMBER(to) && to != NPY_BOOL) {
                continue;
            }
            HPy to_dt = HPyArray_DTypeFromTypeNum(ctx, to);
            int res = add_numeric_cast(ctx, from_dt, to_dt);
            HPy_Close(ctx, to_dt);
            if (res < 0) {
                HPy_Close(ctx, from_dt);
                return -1;
            }
        }
        HPy_Close(ctx, from_dt);
    }
    return 0;
}


static int
cast_to_string_resolve_descriptors(
        HPyContext *ctx,
        HPy self, // PyArrayMethodObject *
        HPy dtypes[2], // PyArray_DTypeMeta *
        HPy given_descrs[2], // PyArray_Descr *
        HPy loop_descrs[2], // PyArray_Descr *
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * NOTE: The following code used to be part of PyArray_AdaptFlexibleDType
     *
     * Get a string-size estimate of the input. These
     * are generallly the size needed, rounded up to
     * a multiple of eight.
     */
    npy_intp size = -1;
    PyArray_Descr *given_descrs_0 = PyArray_Descr_AsStruct(ctx, given_descrs[0]);
    switch (given_descrs_0->type_num) {
        case NPY_BOOL:
        case NPY_UBYTE:
        case NPY_BYTE:
        case NPY_USHORT:
        case NPY_SHORT:
        case NPY_UINT:
        case NPY_INT:
        case NPY_ULONG:
        case NPY_LONG:
        case NPY_ULONGLONG:
        case NPY_LONGLONG:
            assert(given_descrs_0->elsize <= 8);
            assert(given_descrs_0->elsize > 0);
            if (given_descrs_0->kind == 'b') {
                /* 5 chars needed for cast to 'True' or 'False' */
                size = 5;
            }
            else if (given_descrs_0->kind == 'u') {
                size = REQUIRED_STR_LEN[given_descrs_0->elsize];
            }
            else if (given_descrs_0->kind == 'i') {
                /* Add character for sign symbol */
                size = REQUIRED_STR_LEN[given_descrs_0->elsize] + 1;
            }
            break;
        case NPY_HALF:
        case NPY_FLOAT:
        case NPY_DOUBLE:
            size = 32;
            break;
        case NPY_LONGDOUBLE:
            size = 48;
            break;
        case NPY_CFLOAT:
        case NPY_CDOUBLE:
            size = 2 * 32;
            break;
        case NPY_CLONGDOUBLE:
            size = 2 * 48;
            break;
        case NPY_STRING:
        case NPY_VOID:
            size = given_descrs_0->elsize;
            break;
        case NPY_UNICODE:
            size = given_descrs_0->elsize / 4;
            break;
        default:
            PyErr_SetString(PyExc_SystemError,
                    "Impossible cast to string path requested.");
            return -1;
    }
    PyArray_DTypeMeta *dtypes_1 = PyArray_DTypeMeta_AsStruct(ctx, dtypes[1]);
    if (dtypes_1->type_num == NPY_UNICODE) {
        size *= 4;
    }

    PyArray_Descr *loop_descrs_1;
    if (HPy_IsNull(given_descrs[1])) {
        loop_descrs[1] = HPyArray_DescrNewFromType(ctx, dtypes_1->type_num);
        if (HPy_IsNull(loop_descrs[1])) {
            return -1;
        }
        loop_descrs_1 = PyArray_Descr_AsStruct(ctx, loop_descrs[1]);
        loop_descrs_1->elsize = size;
    }
    else {
        /* The legacy loop can handle mismatching itemsizes */
        loop_descrs[1] = hensure_dtype_nbo(ctx, given_descrs[1]);
        if (HPy_IsNull(loop_descrs[1])) {
            return -1;
        }
        loop_descrs_1 = PyArray_Descr_AsStruct(ctx, loop_descrs[1]);
    }

    /* Set the input one as well (late for easier error management) */
    loop_descrs[0] = hensure_dtype_nbo(ctx, given_descrs[0]);
    if (HPy_IsNull(loop_descrs[0])) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes_0 = PyArray_DTypeMeta_AsStruct(ctx, dtypes[0]);
    if (PyArrayMethodObject_AsStruct(ctx, self)->casting == NPY_UNSAFE_CASTING) {
        assert(dtypes_0->type_num == NPY_UNICODE &&
               dtypes_1->type_num == NPY_STRING);
        return NPY_UNSAFE_CASTING;
    }

    if (loop_descrs_1->elsize >= size) {
        return NPY_SAFE_CASTING;
    }
    return NPY_SAME_KIND_CASTING;
}


static int
add_other_to_and_from_string_cast(
        HPyContext *ctx, HPy h_string, HPy h_other,
        PyArray_DTypeMeta *other)
{
    if (HPy_Is(ctx, h_string, h_other)) {
        return 0;
    }

    /* Casting from string, is always a simple legacy-style cast */
    if (other->type_num != NPY_STRING && other->type_num != NPY_UNICODE) {
        if (HPyArray_AddLegacyWrapping_CastingImpl(
                ctx, h_string, h_other, NPY_UNSAFE_CASTING) < 0) {
            return -1;
        }
    }
    /*
     * Casting to strings, is almost the same, but requires a custom resolver
     * to define the correct string length. Right now we use a generic function
     * for this.
     */
    HPy dtypes[2] = {h_other, h_string}; /* PyArray_DTypeMeta *dtypes[2] */
    PyType_Slot slots[] = {
            {NPY_METH_get_loop, &legacy_cast_get_strided_loop},
            {NPY_METH_resolve_descriptors, &cast_to_string_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
        .name = "legacy_cast_to_string",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_REQUIRES_PYAPI,
        .dtypes = dtypes,
        .slots = slots,
    };
    /* Almost everything can be same-kind cast to string (except unicode) */
    if (other->type_num != NPY_UNICODE) {
        spec.casting = NPY_SAME_KIND_CASTING;  /* same-kind if too short */
    }
    else {
        spec.casting = NPY_UNSAFE_CASTING;
    }

    int res = HPyArray_AddCastingImplementation_FromSpec(ctx, &spec, 1);
    return res;
}


NPY_NO_EXPORT NPY_CASTING
string_to_string_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), // PyArrayMethodObject *
        HPy NPY_UNUSED(dtypes[2]), // PyArray_DTypeMeta *
        HPy given_descrs[2], // PyArray_Descr *
        HPy loop_descrs[2], // PyArray_Descr *
        npy_intp *view_offset)
{
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);

    if (HPy_IsNull(given_descrs[1])) {
        loop_descrs[1] = hensure_dtype_nbo(ctx, loop_descrs[0]);
        if (HPy_IsNull(loop_descrs[1])) {
            return -1;
        }
    }
    else {
        loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);
    }

    PyArray_Descr *loop_descrs_0 = PyArray_Descr_AsStruct(ctx, loop_descrs[0]);
    PyArray_Descr *loop_descrs_1 = PyArray_Descr_AsStruct(ctx, loop_descrs[1]);

    if (loop_descrs_0->elsize < loop_descrs_1->elsize) {
        /* New string is longer: safe but cannot be a view */
        return NPY_SAFE_CASTING;
    }
    else {
        /* New string fits into old: if the byte-order matches can be a view */
        int not_swapped = (PyDataType_ISNOTSWAPPED(loop_descrs_0)
                           == PyDataType_ISNOTSWAPPED(loop_descrs_1));
        if (not_swapped) {
            *view_offset = 0;
        }

        if (loop_descrs_0->elsize > loop_descrs_1->elsize) {
            return NPY_SAME_KIND_CASTING;
        }
        /* The strings have the same length: */
        if (not_swapped) {
            return NPY_NO_CASTING;
        }
        else {
            return NPY_EQUIV_CASTING;
        }
    }
}


NPY_NO_EXPORT int
string_to_string_get_loop(
        HPyContext *hctx,
        HPyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        HPyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    int unicode_swap = 0;
    HPy *h_descrs = context->descriptors; /* (PyArray_Descr **) */
    PyArray_Descr *descrs[2] = {
            PyArray_Descr_AsStruct(hctx, h_descrs[0]),
            PyArray_Descr_AsStruct(hctx, h_descrs[1])
    };

    assert(NPY_DTYPE(descrs[0]) == NPY_DTYPE(descrs[1]));
    *flags = PyArrayMethodObject_AsStruct(hctx, context->method)->flags & NPY_METH_RUNTIME_FLAGS;
    if (descrs[0]->type_num == NPY_UNICODE) {
        if (PyDataType_ISNOTSWAPPED(descrs[0]) !=
                PyDataType_ISNOTSWAPPED(descrs[1])) {
            unicode_swap = 1;
        }
    }

    if (PyArray_GetStridedZeroPadCopyFn(
            aligned, unicode_swap, strides[0], strides[1],
            descrs[0]->elsize, descrs[1]->elsize,
            out_loop, out_transferdata) == NPY_FAIL) {
        return -1;
    }
    return 0;
}


/*
 * Add string casts. Right now all string casts are just legacy-wrapped ones
 * (except string<->string and unicode<->unicode), but they do require
 * custom type resolution for the string length.
 *
 * A bit like `object`, it could make sense to define a simpler protocol for
 * string casts, however, we also need to remember that the itemsize of the
 * output has to be found.
 */
static int
PyArray_InitializeStringCasts(HPyContext *ctx)
{
    int result = -1;
    HPy h_string = HPyArray_DTypeFromTypeNum(ctx, NPY_STRING);
    HPy h_unicode = HPyArray_DTypeFromTypeNum(ctx, NPY_UNICODE);
    HPy h_other_dt = HPy_NULL;
    PyArray_DTypeMeta *other_dt = NULL;

    /* Add most casts as legacy ones */
    for (int other = 0; other < NPY_NTYPES; other++) {
        if (PyTypeNum_ISDATETIME(other) || other == NPY_VOID ||
                other == NPY_OBJECT) {
            continue;
        }
        h_other_dt = HPyArray_DTypeFromTypeNum(ctx, other);
        other_dt = PyArray_DTypeMeta_AsStruct(ctx, h_other_dt);

        /* The functions skip string == other_dt or unicode == other_dt */
        if (add_other_to_and_from_string_cast(ctx, h_string, h_other_dt, other_dt) < 0) {
            goto finish;
        }
        if (add_other_to_and_from_string_cast(ctx, h_unicode, h_other_dt, other_dt) < 0) {
            goto finish;
        }

        HPy_Close(ctx, h_other_dt);
        h_other_dt = HPy_NULL;
    }

    /* string<->string and unicode<->unicode have their own specialized casts */
    HPy dtypes[2]; /* PyArray_DTypeMeta *dtypes[2] */
    PyType_Slot slots[] = {
            {NPY_METH_get_loop, &string_to_string_get_loop},
            {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "string_to_string_cast",
            .casting = NPY_UNSAFE_CASTING,
            .nin = 1,
            .nout = 1,
            .flags = (NPY_METH_REQUIRES_PYAPI |
                      NPY_METH_NO_FLOATINGPOINT_ERRORS |
                      NPY_METH_SUPPORTS_UNALIGNED),
            .dtypes = dtypes,
            .slots = slots,
    };

    dtypes[0] = h_string;
    dtypes[1] = h_string;
    if (HPyArray_AddCastingImplementation_FromSpec(ctx, &spec, 1) < 0) {
        goto finish;
    }

    dtypes[0] = h_unicode;
    dtypes[1] = h_unicode;
    if (HPyArray_AddCastingImplementation_FromSpec(ctx, &spec, 1) < 0) {
        goto finish;
    }

    result = 0;
  finish:
    HPy_Close(ctx, h_string);
    HPy_Close(ctx, h_unicode);
    HPy_Close(ctx, h_other_dt);
    return result;
}


/*
 * Small helper function to handle the case of `arr.astype(dtype="V")`.
 * When the output descriptor is not passed, we always use `V<itemsize>`
 * of the other dtype.
 */
static NPY_CASTING
hpy_cast_to_void_dtype_class(HPyContext *ctx,
        HPy /* PyArray_Descr ** */ *given_descrs, 
        HPy /* PyArray_Descr ** */ *loop_descrs,
        npy_intp *view_offset)
{
    /* `dtype="V"` means unstructured currently (compare final path) */
    loop_descrs[1] = HPyArray_DescrNewFromType(ctx, NPY_VOID);
    if (HPy_IsNull(loop_descrs[1])) {
        return -1;
    }
    PyArray_Descr *loop_descrs_1 = PyArray_Descr_AsStruct(ctx, loop_descrs[1]);
    PyArray_Descr *given_descrs_0 = PyArray_Descr_AsStruct(ctx, given_descrs[0]);
    loop_descrs_1->elsize = given_descrs_0->elsize;
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);
    PyArray_Descr *loop_descrs_0 = given_descrs_0; // save a function call

    *view_offset = 0;
    if (loop_descrs_0->type_num == NPY_VOID &&
            loop_descrs_0->subarray == NULL && HPyField_IsNull(loop_descrs_1->names)) {
        return NPY_NO_CASTING;
    }
    return NPY_SAFE_CASTING;
}


static NPY_CASTING
nonstructured_to_structured_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), /* (PyArrayMethodObject *) */
        HPy NPY_UNUSED(dtypes[2]), /* (PyArray_DTypeMeta *) */
        HPy given_descrs[2], /* (PyArray_Descr *) */
        HPy loop_descrs[2], /* (PyArray_Descr *) */
        npy_intp *view_offset)
{
    NPY_CASTING casting;

    if (HPy_IsNull(given_descrs[1])) {
        return hpy_cast_to_void_dtype_class(ctx, given_descrs, loop_descrs, view_offset);
    }
    PyArray_Descr *given_descrs_0 = PyArray_Descr_AsStruct(ctx, given_descrs[0]);
    PyArray_Descr *given_descrs_1 = PyArray_Descr_AsStruct(ctx, given_descrs[1]);
    if (given_descrs_1->subarray != NULL) {
        /*
            * We currently consider this at most a safe cast. It would be
            * possible to allow a view if the field has exactly one element.
            */
        casting = NPY_SAFE_CASTING;
        npy_intp sub_view_offset = NPY_MIN_INTP;
        /* Subarray dtype */
        CAPI_WARN("using PyArray_Descr->PyArray_ArrayDescr->base (PyArray_Descr *)");
        HPy given_descrs_1_subarray_base = HPy_FromPyObject(ctx, (PyObject *)given_descrs_1->subarray->base);
        NPY_CASTING base_casting = HPyArray_GetCastInfo(ctx,
                given_descrs[0], given_descrs_1_subarray_base, HPy_NULL,
                &sub_view_offset);
        if (base_casting < 0) {
            return -1;
        }
        if (given_descrs_1->elsize == given_descrs_1->subarray->base->elsize) {
            /* A single field, view is OK if sub-view is */
            *view_offset = sub_view_offset;
        }
        casting = PyArray_MinCastSafety(casting, base_casting);
    }
    else if (!HPyField_IsNull(given_descrs_1->names)) {
        /* Structured dtype */
        HPy given_descrs_1_names = HPyField_Load(ctx, given_descrs[1], given_descrs_1->names);
        HPy_ssize_t given_descrs_1_names_len = HPy_Length(ctx, given_descrs_1_names);
        if (given_descrs_1_names_len == 0) {
            /* TODO: This retained behaviour, but likely should be changed. */
            casting = NPY_UNSAFE_CASTING;
        }
        else {
            /* Considered at most unsafe casting (but this could be changed) */
            casting = NPY_UNSAFE_CASTING;

            CAPI_WARN("using PyArray_Descr->fields (PyObject *)");
            HPy fields = HPy_FromPyObject(ctx, given_descrs_1->fields);
            HPy keys = HPyDict_Keys(ctx, fields);
            HPy_ssize_t keys_len = HPy_Length(ctx, keys);
            for (HPy_ssize_t i = 0; i < keys_len; i++) {
                HPy key = HPy_GetItem_i(ctx, keys, i);
                HPy h_tuple = HPy_GetItem(ctx, fields, key);
                HPy_Close(ctx, key);
                HPy field_descr = HPy_GetItem_i(ctx, h_tuple, 0); // PyArray_Descr *
                npy_intp field_view_off = NPY_MIN_INTP;
                NPY_CASTING field_casting = HPyArray_GetCastInfo(ctx,
                        given_descrs[0], field_descr, HPy_NULL, &field_view_off);
                HPy_Close(ctx, field_descr);
                casting = PyArray_MinCastSafety(casting, field_casting);
                if (casting < 0) {
                    HPy_Close(ctx, h_tuple);
                    HPy_Close(ctx, fields);
                    HPy_Close(ctx, keys);
                    return -1;
                }
                if (field_view_off != NPY_MIN_INTP) {
                    HPy item = HPy_GetItem_i(ctx, h_tuple, 1);
                    npy_intp to_off = HPyLong_AsSsize_t(ctx, item);
                    HPy_Close(ctx, item);
                    if (error_converting(to_off)) {
                        HPy_Close(ctx, h_tuple);
                        HPy_Close(ctx, fields);
                        HPy_Close(ctx, keys);
                        return -1;
                    }
                    *view_offset = field_view_off - to_off;
                }
                HPy_Close(ctx, h_tuple);
            }
            HPy_Close(ctx, fields);
            HPy_Close(ctx, keys);
            if (given_descrs_1_names_len != 1) {
                /*
                    * Assume that a view is impossible when there is more than one
                    * field.  (Fields could overlap, but that seems weird...)
                    */
                *view_offset = NPY_MIN_INTP;
            }
        }
    }
    else {
        /* Plain void type. This behaves much like a "view" */
        if (given_descrs_0->elsize == given_descrs_1->elsize &&
                !PyDataType_REFCHK(given_descrs_0)) {
            /*
                * A simple view, at the moment considered "safe" (the refcheck is
                * probably not necessary, but more future proof)
                */
            *view_offset = 0;
            casting = NPY_SAFE_CASTING;
        }
        else if (given_descrs_0->elsize <= given_descrs_1->elsize) {
            casting = NPY_SAFE_CASTING;
        }
        else {
            casting = NPY_UNSAFE_CASTING;
            /* new elsize is smaller so a view is OK (reject refs for now) */
            if (!PyDataType_REFCHK(given_descrs_0)) {
                *view_offset = 0;
            }
        }
    }

    /* Void dtypes always do the full cast. */
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);
    loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);

    return casting;
}


int hpy_give_bad_field_error(HPyContext *ctx, HPy key)
{
    if (!HPyErr_Occurred(ctx)) {
        // PyErr_Format(PyExc_RuntimeError,
        //         "Invalid or missing field %R, this should be impossible "
        //         "and indicates a NumPy bug.", key);
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "Invalid or missing field %R, this should be impossible "
                "and indicates a NumPy bug.");
    }
    return -1;
}


static int
nonstructured_to_structured_get_loop(
        HPyContext *ctx,
        HPyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        HPyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArray_Descr *descr_0 = PyArray_Descr_AsStruct(ctx, context->descriptors[0]);
    PyArray_Descr *descr_1 = PyArray_Descr_AsStruct(ctx, context->descriptors[1]);
    if (!HPyField_IsNull(descr_1->names)) {
        int needs_api = 0;
        if (get_fields_transfer_function(ctx,
                aligned, strides[0], strides[1],
                context->descriptors[0], descr_0,
                context->descriptors[1], descr_1,
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else if (descr_1->subarray != NULL) {
        int needs_api = 0;
        if (get_subarray_transfer_function(ctx,
                aligned, strides[0], strides[1],
                context->descriptors[0], descr_0,
                context->descriptors[1], descr_1,
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else {
        /*
         * TODO: This could be a simple zero padded cast, adding a decref
         *       in case of `move_references`. But for now use legacy casts
         *       (which is the behaviour at least up to 1.20).
         */
        int needs_api = 0;
        if (get_wrapped_legacy_cast_function(ctx,
                1, strides[0], strides[1],
                context->descriptors[0], descr_0,
                context->descriptors[1], descr_1,
                move_references, out_loop, out_transferdata,
                &needs_api, 1) < 0) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    return 0;
}


static HPy
HPyArray_GetGenericToVoidCastingImpl(HPyContext *ctx)
{
    static HPyGlobal method;
    static int hg_method_is_set = 0;

    HPy h_method = hg_method_is_set ? HPyGlobal_Load(ctx, method) : HPy_NULL;
    if (!HPy_IsNull(h_method)) {
        // Py_INCREF(method);
        return h_method;
    }
    PyArrayMethodObject *method_data;
    HPy arrmeth_type = HPyGlobal_Load(ctx, HPyArrayMethod_Type);
    h_method = HPy_New(ctx, arrmeth_type, &method_data);
    if (HPy_IsNull(h_method)) {
        return HPyErr_NoMemory(ctx);
    }

    method_data->name = "any_to_void_cast";
    method_data->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method_data->casting = -1;
    method_data->resolve_descriptors = &nonstructured_to_structured_resolve_descriptors;
    method_data->get_strided_loop = &nonstructured_to_structured_get_loop;
    method_data->nin = 1;
    method_data->nout = 1;

    HPyGlobal_Store(ctx, &method, h_method);
    hg_method_is_set = 1;
    return h_method;
}


static NPY_CASTING
structured_to_nonstructured_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), /* (PyArrayMethodObject *) */
        HPy dtypes[2], /* (PyArray_DTypeMeta *) */
        HPy given_descrs[2], /* (PyArray_Descr *) */
        HPy loop_descrs[2], /* (PyArray_Descr *) */
        npy_intp *view_offset)
{
    HPy base_descr; // PyArray_Descr *
    /* The structured part may allow a view (and have its own offset): */
    npy_intp struct_view_offset = NPY_MIN_INTP;

    PyArray_Descr *given_descrs_0 = PyArray_Descr_AsStruct(ctx, given_descrs[0]);
    if (given_descrs_0->subarray != NULL) {
        CAPI_WARN("using PyArray_Descr.(_arr_descr)subarray.base");
        base_descr = HPy_FromPyObject(ctx, (PyObject *)given_descrs_0->subarray->base);
        /* A view is possible if the subarray has exactly one element: */
        if (given_descrs_0->elsize == given_descrs_0->subarray->base->elsize) {
            struct_view_offset = 0;
        }
    }
    else if (!HPyField_IsNull(given_descrs_0->names)) {
        HPy names = HPyField_Load(ctx, given_descrs[0], given_descrs_0->names);
        if (HPy_Length(ctx, names) != 1) {
            HPy_Close(ctx, names);
            /* Only allow casting a single field */
            return -1;
        }
        HPy key = HPy_GetItem_i(ctx, names, 0);
        HPy_Close(ctx, names);
        CAPI_WARN("using PyArray_Descr.fields (PyObject *)");
        HPy given_descrs_0_fields = HPy_FromPyObject(ctx, given_descrs_0->fields);
        HPy base_tup = HPy_GetItem(ctx, given_descrs_0_fields, key);
        HPy_Close(ctx, given_descrs_0_fields);
        base_descr = HPy_GetItem_i(ctx, base_tup, 0);
        HPy item = HPy_GetItem_i(ctx, base_tup, 1);
        struct_view_offset = HPyLong_AsSsize_t(ctx, item);
        HPy_Close(ctx, item);
        if (error_converting(struct_view_offset)) {
            return -1;
        }
    }
    else {
        /*
         * unstructured voids are considered unsafe casts and defined, albeit,
         * at this time they go back to legacy behaviour using getitem/setitem.
         */
        base_descr = HPy_NULL;
        struct_view_offset = 0;
    }

    /*
     * The cast is always considered unsafe, so the PyArray_GetCastInfo
     * result currently only matters for the view_offset.
     */
    npy_intp base_view_offset = NPY_MIN_INTP;
    if (!HPy_IsNull(base_descr) && HPyArray_GetCastInfo(ctx,
            base_descr, given_descrs[1], dtypes[1], &base_view_offset) < 0) {
        if (given_descrs_0->subarray != NULL) {
            HPy_Close(ctx, base_descr);
        }
        return -1;
    }
    if (given_descrs_0->subarray != NULL) {
        HPy_Close(ctx, base_descr);
    }
    if (base_view_offset != NPY_MIN_INTP
            && struct_view_offset != NPY_MIN_INTP) {
        *view_offset = base_view_offset + struct_view_offset;
    }

    /* Void dtypes always do the full cast. */
    if (HPy_IsNull(given_descrs[1])) {
        PyArray_DTypeMeta *dtypes_1 = PyArray_DTypeMeta_AsStruct(ctx, dtypes[1]);
        loop_descrs[1] = HNPY_DT_CALL_default_descr(ctx, dtypes[1], dtypes_1);
        if (HPy_IsNull(loop_descrs[1])) {
            return -1;
        }
        /*
         * Special case strings here, it should be useless (and only actually
         * work for empty arrays).  Possibly this should simply raise for
         * all parametric DTypes.
         */
        PyArray_Descr *loop_descrs_1 = PyArray_Descr_AsStruct(ctx, loop_descrs[1]);
        if (dtypes_1->type_num == NPY_STRING) {
            loop_descrs_1->elsize = given_descrs_0->elsize;
        }
        else if (dtypes_1->type_num == NPY_UNICODE) {
            loop_descrs_1->elsize = given_descrs_0->elsize * 4;
        }
    }
    else {
        loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);
    }
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);

    return NPY_UNSAFE_CASTING;
}


static int
structured_to_nonstructured_get_loop(
        HPyContext *ctx,
        HPyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        HPyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArray_Descr *descr_0 = PyArray_Descr_AsStruct(ctx, context->descriptors[0]);
    PyArray_Descr *descr_1 = PyArray_Descr_AsStruct(ctx, context->descriptors[1]);
    if (!HPyField_IsNull(descr_0->names)) {
        int needs_api = 0;
        if (get_fields_transfer_function(ctx,
                aligned, strides[0], strides[1],
                context->descriptors[0], descr_0,
                context->descriptors[1], descr_1,
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else if (descr_0->subarray != NULL) {
        int needs_api = 0;
        if (get_subarray_transfer_function(ctx,
                aligned, strides[0], strides[1],
                context->descriptors[0], descr_0,
                context->descriptors[1], descr_1,
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else {
        /*
            * In general this is currently defined through legacy behaviour via
            * scalars, and should likely just not be allowed.
            */
        int needs_api = 0;
        if (get_wrapped_legacy_cast_function(ctx,
                aligned, strides[0], strides[1],
                context->descriptors[0], descr_0, 
                context->descriptors[1], descr_1,
                move_references, out_loop, out_transferdata,
                &needs_api, 1) < 0) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    return 0;
}


static HPy
HPyArray_GetVoidToGenericCastingImpl(HPyContext *ctx)
{
    static HPyGlobal method;
    static int hg_method_is_set = 0;

    HPy h_method = hg_method_is_set ? HPyGlobal_Load(ctx, method) : HPy_NULL;
    if (!HPy_IsNull(h_method)) {
        // Py_INCREF(method);
        return h_method;
    }
    PyArrayMethodObject *method_data;
    HPy arrmeth_type = HPyGlobal_Load(ctx, HPyArrayMethod_Type);
    h_method = HPy_New(ctx, arrmeth_type, &method_data);
    if (HPy_IsNull(h_method)) {
        return HPyErr_NoMemory(ctx);
    }

    method_data->name = "void_to_any_cast";
    method_data->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method_data->casting = -1;
    method_data->resolve_descriptors = &structured_to_nonstructured_resolve_descriptors;
    method_data->get_strided_loop = &structured_to_nonstructured_get_loop;
    method_data->nin = 1;
    method_data->nout = 1;

    HPyGlobal_Store(ctx, &method, h_method);
    hg_method_is_set = 1;
    return h_method;
}


/*
 * Find the correct field casting safety.  See the TODO note below, including
 * in 1.20 (and later) this was based on field names rather than field order
 * which it should be using.
 *
 * NOTE: In theory it would be possible to cache the all the field casting
 *       implementations on the dtype, to avoid duplicate work.
 */
static NPY_CASTING
hpy_can_cast_fields_safety(HPyContext *ctx,
        HPy /* PyArray_Descr * */ from, 
        HPy /* PyArray_Descr * */ to, npy_intp *view_offset)
{
    NPY_CASTING casting = NPY_UNSAFE_CASTING;
    PyArray_Descr *from_data = PyArray_Descr_AsStruct(ctx, from);
    PyArray_Descr *to_data = PyArray_Descr_AsStruct(ctx, to);
    HPy names = HPyField_Load(ctx, from, from_data->names);
    HPy to_names = HPyField_Load(ctx, to, to_data->names);
    Py_ssize_t field_count = HPy_Length(ctx, names);
    if (field_count != HPy_Length(ctx, to_names)) {
        /* TODO: This should be rejected! */
        // NPY_UNSAFE_CASTING
        goto finish;
    }

    casting = NPY_NO_CASTING;
    *view_offset = 0;  /* if there are no fields, a view is OK. */
    for (Py_ssize_t i = 0; i < field_count; i++) {
        npy_intp field_view_off = NPY_MIN_INTP;
        HPy from_key = HPy_GetItem_i(ctx, names, i);
        CAPI_WARN("using PyArray_Descr.fields (PyObject *)");
        HPy fields = HPy_FromPyObject(ctx, from_data->fields);
        HPy from_tup = HPyDict_GetItemWithError(ctx, fields, from_key);
        if (HPy_IsNull(from_tup)) {
            casting = (NPY_CASTING) hpy_give_bad_field_error(ctx, from_key);
            goto finish;
        }
        HPy from_base = HPy_GetItem_i(ctx, from_tup, 0); // PyArray_Descr *

        /*
         * TODO: This should use to_key (order), compare gh-15509 by
         *       by Allan Haldane.  And raise an error on failure.
         *       (Fixing that may also requires fixing/changing promotion.)
         */
        CAPI_WARN("using PyArray_Descr.fields (PyObject *)");
        HPy to_fields = HPy_FromPyObject(ctx, to_data->fields);
        HPy to_tup = HPy_GetItem(ctx, to_fields, from_key);
        if (HPy_IsNull(to_tup)) {
            casting = NPY_UNSAFE_CASTING;
            goto finish;
        }
        HPy to_base = HPy_GetItem_i(ctx, to_tup, 0); // PyArray_Descr *

        NPY_CASTING field_casting = HPyArray_GetCastInfo(ctx,
                from_base, to_base, HPy_NULL, &field_view_off);
        if (field_casting < 0) {
            casting = _NPY_ERROR_OCCURRED_IN_CAST;
            goto finish;

        }
        casting = PyArray_MinCastSafety(casting, field_casting);

        /* Adjust the "view offset" by the field offsets: */
        if (field_view_off != NPY_MIN_INTP) {
            HPy to_item = HPy_GetItem_i(ctx, to_tup, 1);
            npy_intp to_off = HPyLong_AsSsize_t(ctx, to_item);
            HPy_Close(ctx, to_item);
            if (error_converting(to_off)) {
                casting = _NPY_ERROR_OCCURRED_IN_CAST;
                goto finish;
            }
            HPy from_item = HPy_GetItem_i(ctx, from_tup, 1);
            npy_intp from_off = HPyLong_AsSsize_t(ctx, from_item);
            HPy_Close(ctx, from_item);
            if (error_converting(from_off)) {
                casting = _NPY_ERROR_OCCURRED_IN_CAST;
                goto finish;
            }
            field_view_off = field_view_off - to_off + from_off;
        }

        /*
         * If there is one field, use its field offset.  After that propagate
         * the view offset if they match and set to "invalid" if not.
         */
        if (i == 0) {
            *view_offset = field_view_off;
        }
        else if (*view_offset != field_view_off) {
            *view_offset = NPY_MIN_INTP;
        }
    }
    if (*view_offset != 0) {
        /* If the calculated `view_offset` is not 0, it can only be "equiv" */
        casting = PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
        goto finish;
    }

    /*
     * If the itemsize (includes padding at the end), fields, or names
     * do not match, this cannot be a view and also not a "no" cast
     * (identical dtypes).
     * It may be possible that this can be relaxed in some cases.
     */
    if (from_data->elsize != to_data->elsize) {
        /*
         * The itemsize may mismatch even if all fields and formats match
         * (due to additional padding).
         */
        casting = PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
        goto finish;
    }

    CAPI_WARN("using PyArray_Descr.fields (PyObject *)");
    int cmp = PyObject_RichCompareBool(from_data->fields, to_data->fields, Py_EQ);
    if (cmp != 1) {
        if (cmp == -1) {
            HPyErr_Clear(ctx);
        }
        casting = PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
        goto finish;
    }
    cmp = HPy_RichCompareBool(ctx, names, to_names, HPy_EQ);
    if (cmp != 1) {
        if (cmp == -1) {
            HPyErr_Clear(ctx);
        }
        casting = PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
        // fall through to 'finish'
    }
finish:
    HPy_Close(ctx, names);
    HPy_Close(ctx, to_names);
    return casting;
}


static NPY_CASTING
void_to_void_resolve_descriptors(
        HPyContext *ctx,
        HPy self, /* (PyArrayMethodObject *) */
        HPy dtypes[2], /* (PyArray_DTypeMeta *) */
        HPy given_descrs[2], /* (PyArray_Descr *) */
        HPy loop_descrs[2], /* (PyArray_Descr *) */
        npy_intp *view_offset)
{
    NPY_CASTING casting;

    if (HPy_IsNull(given_descrs[1])) {
        /* This is weird, since it doesn't return the original descr, but... */
        return hpy_cast_to_void_dtype_class(ctx, given_descrs, loop_descrs, view_offset);
    }

    PyArray_Descr *given_descrs_0 = PyArray_Descr_AsStruct(ctx, given_descrs[0]);
    PyArray_Descr *given_descrs_1 = PyArray_Descr_AsStruct(ctx, given_descrs[1]);
    if (!HPyField_IsNull(given_descrs_0->names) && !HPyField_IsNull(given_descrs_1->names)) {
        /* From structured to structured, need to check fields */
        casting = hpy_can_cast_fields_safety(ctx,
                given_descrs[0], given_descrs[1], view_offset);
    }
    else if (!HPyField_IsNull(given_descrs_0->names)) {
        return structured_to_nonstructured_resolve_descriptors(ctx,
                self, dtypes, given_descrs, loop_descrs, view_offset);
    }
    else if (!HPyField_IsNull(given_descrs_1->names)) {
        return nonstructured_to_structured_resolve_descriptors(ctx,
                self, dtypes, given_descrs, loop_descrs, view_offset);
    }
    else if (given_descrs_0->subarray == NULL &&
                given_descrs_1->subarray == NULL) {
        /* Both are plain void dtypes */
        if (given_descrs_0->elsize == given_descrs_1->elsize) {
            casting = NPY_NO_CASTING;
            *view_offset = 0;
        }
        else if (given_descrs_0->elsize < given_descrs_1->elsize) {
            casting = NPY_SAFE_CASTING;
        }
        else {
            casting = NPY_SAME_KIND_CASTING;
            *view_offset = 0;
        }
    }
    else {
        /*
            * At this point, one of the dtypes must be a subarray dtype, the
            * other is definitely not a structured one.
            */
        PyArray_ArrayDescr *from_sub = given_descrs_0->subarray;
        PyArray_ArrayDescr *to_sub = given_descrs_1->subarray;
        assert(from_sub || to_sub);

        /* If the shapes do not match, this is at most an unsafe cast */
        casting = NPY_UNSAFE_CASTING;
        /*
            * We can use a view in two cases:
            * 1. The shapes and elsizes matches, so any view offset applies to
            *    each element of the subarray identically.
            *    (in practice this probably implies the `view_offset` will be 0)
            * 2. There is exactly one element and the subarray has no effect
            *    (can be tested by checking if the itemsizes of the base matches)
            */
        npy_bool subarray_layout_supports_view = NPY_FALSE;
        if (from_sub && to_sub) {
            CAPI_WARN("using PyArray_ArrayDescr.shape");
            int res = PyObject_RichCompareBool(from_sub->shape, to_sub->shape, Py_EQ);
            if (res < 0) {
                return -1;
            }
            else if (res) {
                /* Both are subarrays and the shape matches, could be no cast */
                casting = NPY_NO_CASTING;
                /* May be a view if there is one element or elsizes match */
                if (from_sub->base->elsize == to_sub->base->elsize
                        || given_descrs_0->elsize == from_sub->base->elsize) {
                    subarray_layout_supports_view = NPY_TRUE;
                }
            }
        }
        else if (from_sub) {
            /* May use a view if "from" has only a single element: */
            if (given_descrs_0->elsize == from_sub->base->elsize) {
                subarray_layout_supports_view = NPY_TRUE;
            }
        }
        else {
            /* May use a view if "from" has only a single element: */
            if (given_descrs_1->elsize == to_sub->base->elsize) {
                subarray_layout_supports_view = NPY_TRUE;
            }
        }

        HPy from_base;
        if (from_sub == NULL) {
            from_base = given_descrs[0];
        } else {
            CAPI_WARN("using PyArray_ArrayDescr->base (PyArray_Descr*)");
            from_base = HPy_FromPyObject(ctx, (PyObject *)from_sub->base); 
        }
        HPy to_base;
        if (to_sub == NULL) {
            to_base = given_descrs[1];
        } else {
            CAPI_WARN("using PyArray_ArrayDescr->base (PyArray_Descr*)");
            to_base = HPy_FromPyObject(ctx, (PyObject *)to_sub->base);
        }
        /* An offset for  */
        NPY_CASTING field_casting = HPyArray_GetCastInfo(ctx,
                from_base, to_base, HPy_NULL, view_offset);
        if (from_sub != NULL) {
            HPy_Close(ctx, from_base);
        }
        if (to_sub != NULL) {
            HPy_Close(ctx, to_base);
        }
        if (!subarray_layout_supports_view) {
            *view_offset = NPY_MIN_INTP;
        }
        if (field_casting < 0) {
            return -1;
        }
        casting = PyArray_MinCastSafety(casting, field_casting);
    }

    /* Void dtypes always do the full cast. */
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);
    loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);

    return casting;
}


NPY_NO_EXPORT int
void_to_void_get_loop(
        HPyContext *ctx,
        HPyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        HPyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    HPy *h_descrs = context->descriptors; /* (PyArray_Descr **) */
    PyArray_Descr *descr_0 = PyArray_Descr_AsStruct(ctx, h_descrs[0]);
    PyArray_Descr *descr_1 = PyArray_Descr_AsStruct(ctx, h_descrs[1]);
    PyArray_Descr *descrs[2] = {
            descr_0,
            descr_1,
    };
    if (!HPyField_IsNull(descr_0->names) ||
            !HPyField_IsNull(descrs[1]->names)) {
        int needs_api = 0;
        if (get_fields_transfer_function(ctx,
                aligned, strides[0], strides[1],
                h_descrs[0], descr_0, h_descrs[1], descr_1,
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else if (descrs[0]->subarray != NULL ||
             descrs[1]->subarray != NULL) {
        int needs_api = 0;
        if (get_subarray_transfer_function(ctx,
                aligned, strides[0], strides[1],
                h_descrs[0], descr_0, h_descrs[1], descr_1,
                move_references, out_loop, out_transferdata,
                &needs_api) == NPY_FAIL) {
            return -1;
        }
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    else {
        /*
         * This is a string-like copy of the two bytes (zero padding if
         * necessary)
         */
        if (PyArray_GetStridedZeroPadCopyFn(
                0, 0, strides[0], strides[1],
                descrs[0]->elsize, descrs[1]->elsize,
                out_loop, out_transferdata) == NPY_FAIL) {
            return -1;
        }
        *flags = 0;
    }
    return 0;
}


/*
 * This initializes the void to void cast. Voids include structured dtypes,
 * which means that they can cast from and to any other dtype and, in that
 * sense, are special (similar to Object).
 */
static int
PyArray_InitializeVoidToVoidCast(HPyContext *ctx)
{
    HPy Void = HPyArray_DTypeFromTypeNum(ctx, NPY_VOID);
    HPy dtypes[2] = {Void, Void};
    PyType_Slot slots[] = {
            {NPY_METH_get_loop, &void_to_void_get_loop},
            {NPY_METH_resolve_descriptors, &void_to_void_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "void_to_void_cast",
            .casting = -1,  /* may not cast at all */
            .nin = 1,
            .nout = 1,
            .flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    int res = HPyArray_AddCastingImplementation_FromSpec(ctx, &spec, 1);
    HPy_Close(ctx, Void);
    return res;
}


/*
 * Implement object to any casting implementation. Casting from object may
 * require inspecting of all array elements (for parametric dtypes), and
 * the resolver will thus reject all parametric dtypes if the out dtype
 * is not provided.
 */
static NPY_CASTING
object_to_any_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), /* (PyArrayMethodObject *) */
        HPy dtypes[2], /* (PyArray_DTypeMeta *) */
        HPy given_descrs[2], /* (PyArray_Descr *) */
        HPy loop_descrs[2], /* (PyArray_Descr *) */
        npy_intp *NPY_UNUSED(view_offset))
{
    if (HPy_IsNull(given_descrs[1])) {
        /*
         * This should not really be called, since object -> parametric casts
         * require inspecting the object array. Allow legacy ones, the path
         * here is that e.g. "M8" input is considered to be the DType class,
         * and by allowing it here, we go back to the "M8" instance.
         */
        PyArray_DTypeMeta *dtypes_1 = PyArray_DTypeMeta_AsStruct(ctx, dtypes[1]);
        if (NPY_DT_is_parametric(dtypes_1)) {
            // PyErr_Format(PyExc_TypeError,
            //         "casting from object to the parametric DType %S requires "
            //         "the specified output dtype instance. "
            //         "This may be a NumPy issue, since the correct instance "
            //         "should be discovered automatically, however.", dtypes[1]);
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "casting from object to the parametric DType %S requires "
                    "the specified output dtype instance. "
                    "This may be a NumPy issue, since the correct instance "
                    "should be discovered automatically, however.");
            return -1;
        }
        loop_descrs[1] = HNPY_DT_CALL_default_descr(ctx, dtypes[1], dtypes_1);
        if (HPy_IsNull(loop_descrs[1])) {
            return -1;
        }
    }
    else {
        loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);
    }

    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);
    return NPY_UNSAFE_CASTING;
}


/*
 * Casting to object is special since it is generic to all input dtypes.
 */
static HPy
HPyArray_GetObjectToGenericCastingImpl(HPyContext *ctx)
{
    static HPyGlobal method;
    static int hg_method_is_set = 0;

    HPy h_method = hg_method_is_set ? HPyGlobal_Load(ctx, method) : HPy_NULL;
    if (!HPy_IsNull(h_method)) {
        // Py_INCREF(method);
        return h_method;
    }
    PyArrayMethodObject *method_data;
    HPy arrmeth_type = HPyGlobal_Load(ctx, HPyArrayMethod_Type);
    h_method = HPy_New(ctx, arrmeth_type, &method_data);
    if (HPy_IsNull(h_method)) {
        return HPyErr_NoMemory(ctx);
    }

    method_data->nin = 1;
    method_data->nout = 1;
    method_data->name = "object_to_any_cast";
    method_data->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method_data->casting = NPY_UNSAFE_CASTING;
    method_data->resolve_descriptors = &object_to_any_resolve_descriptors;
    method_data->get_strided_loop = &object_to_any_get_loop;

    HPyGlobal_Store(ctx, &method, h_method);
    hg_method_is_set = 1;
    return h_method;
}



/* Any object is simple (could even use the default) */
static NPY_CASTING
any_to_object_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), /* (PyArrayMethodObject *) */
        HPy dtypes[2], /* (PyArray_DTypeMeta *) */
        HPy given_descrs[2], /* (PyArray_Descr *) */
        HPy loop_descrs[2], /* (PyArray_Descr *) */
        npy_intp *NPY_UNUSED(view_offset))
{
    if (HPy_IsNull(given_descrs[1])) {
        PyArray_DTypeMeta *dtypes_1 = PyArray_DTypeMeta_AsStruct(ctx, dtypes[1]);
        loop_descrs[1] = HNPY_DT_CALL_default_descr(ctx, dtypes[1], dtypes_1);
        if (HPy_IsNull(loop_descrs[1])) {
            return -1;
        }
    }
    else {
        loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);
    }

    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);
    return NPY_SAFE_CASTING;
}


/*
 * Casting to object is special since it is generic to all input dtypes.
 */
static HPy
HPyArray_GetGenericToObjectCastingImpl(HPyContext *ctx)
{
    static HPyGlobal method;
    static int hg_method_is_set = 0;

    HPy h_method = hg_method_is_set ? HPyGlobal_Load(ctx, method) : HPy_NULL;
    if (!HPy_IsNull(h_method)) {
        // Py_INCREF(method);
        return h_method;
    }
    PyArrayMethodObject *method_data;
    HPy arrmeth_type = HPyGlobal_Load(ctx, HPyArrayMethod_Type);
    h_method = HPy_New(ctx, arrmeth_type, &method_data);
    if (HPy_IsNull(h_method)) {
        return HPyErr_NoMemory(ctx);
    }

    method_data->nin = 1;
    method_data->nout = 1;
    method_data->name = "any_to_object_cast";
    method_data->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method_data->casting = NPY_SAFE_CASTING;
    method_data->resolve_descriptors = &any_to_object_resolve_descriptors;
    method_data->get_strided_loop = &any_to_object_get_loop;

    HPyGlobal_Store(ctx, &method, h_method);
    hg_method_is_set = 1;
    return h_method;
}


/*
 * Casts within the object dtype is always just a plain copy/view.
 * For that reason, this function might remain unimplemented.
 */
static int
object_to_object_get_loop(
        HPyContext *hctx,
        HPyArrayMethod_Context *NPY_UNUSED(context),
        int NPY_UNUSED(aligned), int move_references,
        const npy_intp *NPY_UNUSED(strides),
        HPyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS;
    if (move_references) {
        *out_loop = &_strided_to_strided_move_references;
        *out_transferdata = NULL;
    }
    else {
        *out_loop = &_strided_to_strided_copy_references;
        *out_transferdata = NULL;
    }
    return 0;
}


static int
PyArray_InitializeObjectToObjectCast(HPyContext *ctx)
{
    HPy h_Object = HPyArray_DTypeFromTypeNum(ctx, NPY_OBJECT);
    HPy dtypes[2] = {h_Object, h_Object}; /* PyArray_DTypeMeta *dtypes[2] */
    PyType_Slot slots[] = {
            {NPY_METH_get_loop, &object_to_object_get_loop},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "object_to_object_cast",
            .casting = NPY_NO_CASTING,
            .nin = 1,
            .nout = 1,
            .flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    int res = HPyArray_AddCastingImplementation_FromSpec(ctx, &spec, 1);
    HPy_Close(ctx, h_Object);
    return res;
}


NPY_NO_EXPORT int
PyArray_InitializeCasts(HPyContext *ctx)
{
    if (PyArray_InitializeNumericCasts(ctx) < 0) {
        return -1;
    }
    if (PyArray_InitializeStringCasts(ctx) < 0) {
        return -1;
    }
    if (PyArray_InitializeVoidToVoidCast(ctx) < 0) {
        return -1;
    }
    if (PyArray_InitializeObjectToObjectCast(ctx) < 0) {
        return -1;
    }
    /* Datetime casts are defined in datetime.c */
    if (PyArray_InitializeDatetimeCasts(ctx) < 0) {
        return -1;
    }
    return 0;
}

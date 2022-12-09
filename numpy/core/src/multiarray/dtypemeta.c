/* Array Descr Object */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include "npy_pycompat.h"

#include "common.h"
#include "dtypemeta.h"
#include "_datetime.h"
#include "array_coercion.h"
#include "scalartypes.h"
#include "convert_datatype.h"
#include "usertypes.h"
#include "descriptor.h"

#include <assert.h>

HPyDef_SLOT(dtypemeta_init, HPy_tp_init)
static int
dtypemeta_init_impl(HPyContext *ctx, HPy NPY_UNUSED(self),
        HPy *NPY_UNUSED(args), HPy_ssize_t NPY_UNUSED(n), HPy NPY_UNUSED(kw))
{
    HPyErr_SetString(ctx, ctx->h_TypeError,
            "Preliminary-API: Cannot __init__ DType class.");
    return -1;
}

/**
 * tp_is_gc slot of Python types. This is implemented only for documentation
 * purposes to indicate and document the subtleties involved.
 *
 * Python Type objects are either statically created (typical C-Extension type)
 * or HeapTypes (typically created in Python).
 * HeapTypes have the Py_TPFLAGS_HEAPTYPE flag and are garbage collected.
 * Our DTypeMeta instances (`np.dtype` and its subclasses) *may* be HeapTypes
 * if the Py_TPFLAGS_HEAPTYPE flag is set (they are created from Python).
 * They are not for legacy DTypes or np.dtype itself.
 *
 * @param self
 * @return nonzero if the object is garbage collected
 */
static NPY_INLINE int
dtypemeta_is_gc(PyObject *dtype_class)
{
    return PyType_Type.tp_is_gc(dtype_class);
}

HPyDef_SLOT(DTypeMeta_traverse, HPy_tp_traverse)
static int DTypeMeta_traverse_impl(void *self_p, HPyFunc_visitproc visit, void *arg) {
    /*
     * We have to traverse the base class (if it is a HeapType).
     * PyType_Type will handle this logic for us.
     * This function is currently not used, but will probably be necessary
     * in the future when we implement HeapTypes (python/dynamically
     * defined types). It should be revised at that time.
     */
    // TODO HPY LABS PORT: enable assertion
    // assert(!NPY_DT_is_legacy(type) && (PyTypeObject *)type != &PyArrayDescr_Type);
    PyArray_DTypeMeta *self = (PyArray_DTypeMeta*) self_p;
    HPy_VISIT(&self->singleton);
    HPy_VISIT(&self->scalar_type);
    NPY_DType_Slots *slots = NPY_DT_SLOTS(self);
    if (slots) {
        HPy_VISIT(&slots->castingimpls);
        HPy_VISIT(&slots->within_dtype_castingimpl);
    }
    // return PyType_Type.tp_traverse((PyObject *)type, visit, arg);
    return 0;
}

HPyDef_SLOT(DTypeMeta_destroy, HPy_tp_destroy)
static void DTypeMeta_destroy_impl(void *self) {
    PyArray_DTypeMeta *data = (PyArray_DTypeMeta *) self;
    PyMem_Free(data->dt_slots);
}



HPyDef_SLOT(legacy_dtype_default_new, HPy_tp_new)
static HPy
legacy_dtype_default_new_impl(HPyContext *ctx, HPy h_self,
        HPy *args, HPy_ssize_t nargs, HPy kwargs)
{
    /* TODO: This should allow endianness and possibly metadata */
    PyArray_DTypeMeta *self = PyArray_DTypeMeta_AsStruct(ctx, h_self);
    if (NPY_DT_is_parametric(self)) {
        /* reject parametric ones since we would need to get unit, etc. info */
        // TODO HPY LABS PORT: PyErr_Format
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Preliminary-API: Flexible/Parametric legacy DType '%S' can "
                "only be instantiated using `np.dtype(...)`");
        return HPy_NULL;
    }

    if (nargs != 0 ||
                (!HPy_IsNull(kwargs) && HPy_Length(ctx, kwargs))) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "currently only the no-argument instantiation is supported; "
                "use `np.dtype` instead.");
        return HPy_NULL;
    }
    return HPyField_Load(ctx, h_self, self->singleton);
}


//static PyArray_Descr *
//nonparametric_discover_descr_from_pyobject(
//        PyArray_DTypeMeta *cls, PyObject *obj)
static HPy
nonparametric_discover_descr_from_pyobject(
        HPyContext *ctx, HPy cls, HPy obj)
{
    /* If the object is of the correct scalar type return our singleton */
    PyArray_DTypeMeta *cls_data = PyArray_DTypeMeta_AsStruct(ctx, cls);
    assert(!NPY_DT_is_parametric(cls_data));
    return HPyField_Load(ctx, cls, cls_data->singleton);
}


static PyArray_Descr *
string_discover_descr_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj)
{
    npy_intp itemsize = -1;
    if (PyBytes_Check(obj)) {
        itemsize = PyBytes_Size(obj);
    }
    else if (PyUnicode_Check(obj)) {
        itemsize = PyUnicode_GetLength(obj);
    }
    if (itemsize != -1) {
        if (cls->type_num == NPY_UNICODE) {
            itemsize *= 4;
        }
        if (itemsize > NPY_MAX_INT) {
            PyErr_SetString(PyExc_TypeError,
                    "string to large to store inside array.");
        }
        PyArray_Descr *res = PyArray_DescrNewFromType(cls->type_num);
        if (res == NULL) {
            return NULL;
        }
        res->elsize = (int)itemsize;
        return res;
    }
    return PyArray_DTypeFromObjectStringDiscovery(obj, NULL, cls->type_num);
}


static PyArray_Descr *
void_discover_descr_from_pyobject(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    if (PyArray_IsScalar(obj, Void)) {
        PyVoidScalarObject *void_obj = (PyVoidScalarObject *)obj;
        Py_INCREF(void_obj->descr);
        return void_obj->descr;
    }
    if (PyBytes_Check(obj)) {
        PyArray_Descr *descr = PyArray_DescrNewFromType(NPY_VOID);
        if (descr == NULL) {
            return NULL;
        }
        Py_ssize_t itemsize = PyBytes_Size(obj);
        if (itemsize > NPY_MAX_INT) {
            PyErr_SetString(PyExc_TypeError,
                    "byte-like to large to store inside array.");
            Py_DECREF(descr);
            return NULL;
        }
        descr->elsize = (int)itemsize;
        return descr;
    }
    PyErr_Format(PyExc_TypeError,
            "A bytes-like object is required, not '%s'", Py_TYPE(obj)->tp_name);
    return NULL;
}


static PyArray_Descr *
discover_datetime_and_timedelta_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj) {
    if (PyArray_IsScalar(obj, Datetime) ||
            PyArray_IsScalar(obj, Timedelta)) {
        PyArray_DatetimeMetaData *meta;
        PyArray_Descr *descr = PyArray_DescrFromScalar(obj);
        meta = get_datetime_metadata_from_dtype(descr);
        if (meta == NULL) {
            return NULL;
        }
        PyArray_Descr *new_descr = create_datetime_dtype(cls->type_num, meta);
        Py_DECREF(descr);
        return new_descr;
    }
    else {
        return find_object_datetime_type(obj, cls->type_num);
    }
}


static HPy
nonparametric_default_descr(HPyContext *ctx, HPy cls)
{
    PyArray_DTypeMeta *cls_data = PyArray_DTypeMeta_AsStruct(ctx, cls);
    return HPyField_Load(ctx, cls, cls_data->singleton);
}


/* Ensure a copy of the singleton (just in case we do adapt it somewhere) */
static HPy
datetime_and_timedelta_default_descr(HPyContext *ctx, HPy cls)
{
    PyArray_DTypeMeta *cls_data = PyArray_DTypeMeta_AsStruct(ctx, cls);
    HPy h_singleton = HPyField_Load(ctx, cls, cls_data->singleton);
    HPy res = HPyArray_DescrNew(ctx, h_singleton);
    HPy_Close(ctx, h_singleton);
    return res;
}


static HPy
void_default_descr(HPyContext *ctx, HPy cls)
{
    PyArray_DTypeMeta *cls_data = PyArray_DTypeMeta_AsStruct(ctx, cls);
    HPy res = HPyField_Load(ctx, cls, cls_data->singleton);
    if (HPy_IsNull(res)) {
        return HPy_NULL;
    }
    /*
     * The legacy behaviour for `np.array([], dtype="V")` is to use "V8".
     * This is because `[]` uses `float64` as dtype, and then that is used
     * for the size of the requested void.
     */
    PyArray_Descr_AsStruct(ctx, res)->elsize = 8;
    return res;
}

static HPy
string_and_unicode_default_descr(HPyContext *ctx, HPy cls)
{
    PyArray_DTypeMeta *cls_data = PyArray_DTypeMeta_AsStruct(ctx, cls);
    HPy res;
    PyArray_Descr *py_res;

    CAPI_WARN("string_and_unicode_default_descr: call to PyArray_DescrNewFromType");
    py_res = PyArray_DescrNewFromType(cls_data->type_num);
    if (py_res == NULL) {
        return HPy_NULL;
    }
    py_res->elsize = 1;
    if (cls_data->type_num == NPY_UNICODE) {
        py_res->elsize *= 4;
    }
    res = HPy_FromPyObject(ctx, (PyObject *)py_res);
    Py_DECREF(py_res);
    return res;
}


static PyArray_Descr *
string_unicode_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    if (descr1->elsize >= descr2->elsize) {
        return ensure_dtype_nbo(descr1);
    }
    else {
        return ensure_dtype_nbo(descr2);
    }
}

static HPy // PyArray_Descr *
hpy_string_unicode_common_instance(HPyContext *ctx,
            HPy /* PyArray_Descr * */ descr1, HPy /* PyArray_Descr * */ descr2)
{
    PyArray_Descr *descr1_struct = PyArray_Descr_AsStruct(ctx, descr1);
    PyArray_Descr *descr2_struct = PyArray_Descr_AsStruct(ctx, descr2);
    if (descr1_struct->elsize >= descr2_struct->elsize) {
        return hensure_dtype_nbo_with_struct(ctx, descr1, descr1_struct);
    }
    else {
        return hensure_dtype_nbo_with_struct(ctx, descr2, descr2_struct);
    }
}


static PyArray_Descr *
void_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    /*
     * We currently do not support promotion of void types unless they
     * are equivalent.
     */
    if (!PyArray_CanCastTypeTo(descr1, descr2, NPY_EQUIV_CASTING)) {
        if (descr1->subarray == NULL && HPyField_IsNull(descr1->names) &&
                descr2->subarray == NULL && HPyField_IsNull(descr2->names)) {
            PyErr_SetString(PyExc_TypeError,
                    "Invalid type promotion with void datatypes of different "
                    "lengths. Use the `np.bytes_` datatype instead to pad the "
                    "shorter value with trailing zero bytes.");
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "invalid type promotion with structured datatype(s).");
        }
        return NULL;
    }
    Py_INCREF(descr1);
    return descr1;
}

static HPy // PyArray_Descr *
hpy_void_common_instance(HPyContext *ctx,
            HPy /* PyArray_Descr * */ descr1, HPy /* PyArray_Descr * */ descr2)
{
    /*
     * We currently do not support promotion of void types unless they
     * are equivalent.
     */
    PyArray_Descr *descr1_struct = PyArray_Descr_AsStruct(ctx, descr1);
    PyArray_Descr *descr2_struct = PyArray_Descr_AsStruct(ctx, descr2);
    if (!HPyArray_CanCastTypeTo(ctx, descr1, descr2, NPY_EQUIV_CASTING)) {
        if (descr1_struct->subarray == NULL && HPyField_IsNull(descr1_struct->names) &&
                descr2_struct->subarray == NULL && HPyField_IsNull(descr2_struct->names)) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "Invalid type promotion with void datatypes of different "
                    "lengths. Use the `np.bytes_` datatype instead to pad the "
                    "shorter value with trailing zero bytes.");
        }
        else {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "invalid type promotion with structured datatype(s).");
        }
        return HPy_NULL;
    }
    HPy_Close(ctx, descr1);
    return descr1;
}

NPY_NO_EXPORT int
python_builtins_are_known_scalar_types(HPyContext *ctx,
        HPy NPY_UNUSED(cls), HPy pytype)
{
    /*
     * Always accept the common Python types, this ensures that we do not
     * convert pyfloat->float64->integers. Subclasses are hopefully rejected
     * as being discovered.
     * This is necessary only for python scalar classes which we discover
     * as valid DTypes.
     */
    if (HPy_Is(ctx, pytype, ctx->h_FloatType)) {
        return 1;
    }
    if (HPy_Is(ctx, pytype, ctx->h_LongType)) {
        return 1;
    }
    if (HPy_Is(ctx, pytype, ctx->h_BoolType)) {
        return 1;
    }
    if (HPy_Is(ctx, pytype, ctx->h_ComplexType)) {
        return 1;
    }
    if (HPy_Is(ctx, pytype, ctx->h_UnicodeType)) {
        return 1;
    }
    if (HPy_Is(ctx, pytype, ctx->h_BytesType)) {
        return 1;
    }
    return 0;
}


static int
signed_integers_is_known_scalar_types(HPyContext *ctx,
        HPy cls, HPy pytype)
{
    if (python_builtins_are_known_scalar_types(ctx, cls, pytype)) {
        return 1;
    }
    /* Convert our scalars (raise on too large unsigned and NaN, etc.) */
    HPy h_generic_type = HPyGlobal_Load(ctx, HPyGenericArrType_Type);
    int res = HPyType_IsSubtype(ctx, pytype, h_generic_type);
    HPy_Close(ctx, h_generic_type);
    return res;
}


static int
datetime_known_scalar_types(HPyContext *ctx, HPy cls, HPy pytype)
{
    if (python_builtins_are_known_scalar_types(ctx, cls, pytype)) {
        return 1;
    }
    /*
     * To be able to identify the descriptor from e.g. any string, datetime
     * must take charge. Otherwise we would attempt casting which does not
     * truly support this. Only object arrays are special cased in this way.
     */
    return (HPyType_IsSubtype(ctx, pytype, ctx->h_BytesType) ||
            HPyType_IsSubtype(ctx, pytype, ctx->h_UnicodeType));
}


static int
string_known_scalar_types(HPyContext *ctx, HPy cls, HPy pytype) {
    if (python_builtins_are_known_scalar_types(ctx, cls, pytype)) {
        return 1;
    }

    HPy h_datetime_type = HPyGlobal_Load(ctx, HPyDatetimeArrType_Type);
    int res = HPyType_IsSubtype(ctx, pytype, h_datetime_type);
    HPy_Close(ctx, h_datetime_type);
    if (res) {
        /*
         * TODO: This should likely be deprecated or otherwise resolved.
         *       Deprecation has to occur in `String->setitem` unfortunately.
         *
         * Datetime currently do not cast to shorter strings, but string
         * coercion for arbitrary values uses `str(obj)[:len]` so it works.
         * This means `np.array(np.datetime64("2020-01-01"), "U9")`
         * and `np.array(np.datetime64("2020-01-01")).astype("U9")` behave
         * differently.
         */
        return 1;
    }
    return 0;
}


/*
 * The following set of functions define the common dtype operator for
 * the builtin types.
 */
static PyArray_DTypeMeta *
default_builtin_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    assert(cls->type_num < NPY_NTYPES);
    if (!NPY_DT_is_legacy(other) || other->type_num > cls->type_num) {
        /*
         * Let the more generic (larger type number) DType handle this
         * (note that half is after all others, which works out here.)
         */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }

    /*
     * Note: The use of the promotion table should probably be revised at
     *       some point. It may be most useful to remove it entirely and then
     *       consider adding a fast path/cache `PyArray_CommonDType()` itself.
     */
    int common_num = _npy_type_promotion_table[cls->type_num][other->type_num];
    if (common_num < 0) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    return PyArray_DTypeFromTypeNum(common_num);
}

static HPy // PyArray_DTypeMeta *
hpy_default_builtin_common_dtype(HPyContext *ctx, HPy /* PyArray_DTypeMeta * */ cls, 
                                            HPy /* PyArray_DTypeMeta * */ other)
{
    PyArray_DTypeMeta *cls_struct = PyArray_DTypeMeta_AsStruct(ctx, cls);
    assert(cls_struct->type_num < NPY_NTYPES);
    PyArray_DTypeMeta *other_struct = PyArray_DTypeMeta_AsStruct(ctx, other);
    if (!NPY_DT_is_legacy(other_struct) || other_struct->type_num > cls_struct->type_num) {
        /*
         * Let the more generic (larger type number) DType handle this
         * (note that half is after all others, which works out here.)
         */
        // Py_INCREF(Py_NotImplemented);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }

    /*
     * Note: The use of the promotion table should probably be revised at
     *       some point. It may be most useful to remove it entirely and then
     *       consider adding a fast path/cache `PyArray_CommonDType()` itself.
     */
    int common_num = _npy_type_promotion_table[cls_struct->type_num][other_struct->type_num];
    if (common_num < 0) {
        // Py_INCREF(Py_NotImplemented);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    return HPyArray_DTypeFromTypeNum(ctx, common_num);
}


static PyArray_DTypeMeta *
string_unicode_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    assert(cls->type_num < NPY_NTYPES && cls != other);
    if (!NPY_DT_is_legacy(other) || (!PyTypeNum_ISNUMBER(other->type_num) &&
            /* Not numeric so defer unless cls is unicode and other is string */
            !(cls->type_num == NPY_UNICODE && other->type_num == NPY_STRING))) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    /*
     * The builtin types are ordered by complexity (aside from object) here.
     * Arguably, we should not consider numbers and strings "common", but
     * we currently do.
     */
    Py_INCREF(cls);
    return cls;
}

static HPy // PyArray_DTypeMeta *
hpy_string_unicode_common_dtype(HPyContext *ctx, HPy /* PyArray_DTypeMeta * */ cls, 
                                            HPy /* PyArray_DTypeMeta * */ other)
{
    PyArray_DTypeMeta *other_struct = PyArray_DTypeMeta_AsStruct(ctx, other);
    PyArray_DTypeMeta *cls_struct = PyArray_DTypeMeta_AsStruct(ctx, cls);
    assert(cls_struct->type_num < NPY_NTYPES && !HPy_Is(ctx, cls, other));
    if (!NPY_DT_is_legacy(other_struct) || (!PyTypeNum_ISNUMBER(other_struct->type_num) &&
            /* Not numeric so defer unless cls is unicode and other is string */
            !(cls_struct->type_num == NPY_UNICODE && other_struct->type_num == NPY_STRING))) {
        // Py_INCREF(Py_NotImplemented);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    /*
     * The builtin types are ordered by complexity (aside from object) here.
     * Arguably, we should not consider numbers and strings "common", but
     * we currently do.
     */
    // Py_INCREF(cls);
    return HPy_Dup(ctx, cls);
}

static PyArray_DTypeMeta *
datetime_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (cls->type_num == NPY_DATETIME && other->type_num == NPY_TIMEDELTA) {
        /*
         * TODO: We actually currently do allow promotion here. This is
         *       currently relied on within `np.add(datetime, timedelta)`,
         *       while for concatenation the cast step will fail.
         */
        Py_INCREF(cls);
        return cls;
    }
    return default_builtin_common_dtype(cls, other);
}

static HPy // PyArray_DTypeMeta *
hpy_datetime_common_dtype(HPyContext *ctx, HPy /* PyArray_DTypeMeta * */ cls, 
                                            HPy /* PyArray_DTypeMeta * */ other)
{
    PyArray_DTypeMeta *other_struct = PyArray_DTypeMeta_AsStruct(ctx, other);
    PyArray_DTypeMeta *cls_struct = PyArray_DTypeMeta_AsStruct(ctx, cls);
    if (cls_struct->type_num == NPY_DATETIME && other_struct->type_num == NPY_TIMEDELTA) {
        /*
         * TODO: We actually currently do allow promotion here. This is
         *       currently relied on within `np.add(datetime, timedelta)`,
         *       while for concatenation the cast step will fail.
         */
        // Py_INCREF(cls);
        return HPy_Dup(ctx, cls);
    }
    return hpy_default_builtin_common_dtype(ctx, cls, other);
}



static PyArray_DTypeMeta *
object_common_dtype(
        PyArray_DTypeMeta *cls, PyArray_DTypeMeta *NPY_UNUSED(other))
{
    /*
     * The object DType is special in that it can represent everything,
     * including all potential user DTypes.
     * One reason to defer (or error) here might be if the other DType
     * does not support scalars so that e.g. `arr1d[0]` returns a 0-D array
     * and `arr.astype(object)` would fail. But object casts are special.
     */
    Py_INCREF(cls);
    return cls;
}

static HPy // PyArray_DTypeMeta *
hpy_object_common_dtype(HPyContext *ctx, HPy /* PyArray_DTypeMeta * */ cls, 
                                            HPy /* PyArray_DTypeMeta * */ other)
{
    /*
     * The object DType is special in that it can represent everything,
     * including all potential user DTypes.
     * One reason to defer (or error) here might be if the other DType
     * does not support scalars so that e.g. `arr1d[0]` returns a 0-D array
     * and `arr.astype(object)` would fail. But object casts are special.
     */
    // Py_INCREF(cls);
    return HPy_Dup(ctx, cls);
}

static HPyDef *new_dtype_legacy_defines[] = {
    &legacy_dtype_default_new,
    NULL
};
static HPyType_Spec New_PyArrayDescr_spec_prototype = {
    .basicsize = sizeof(PyArray_Descr),
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_BASETYPE,
    .defines = new_dtype_legacy_defines,
    .builtin_shape = SHAPE(PyArray_DTypeMeta),
};


NPY_NO_EXPORT PyArray_Descr *
default_descr_function_trampoline(PyArray_DTypeMeta *cls)
{
    HPyContext *ctx = npy_get_context();
    HPy h_cls = HPy_FromPyObject(ctx, (PyObject *)cls);
    HPy h_res = hdtypemeta_call_default_descr(ctx, h_cls);
    PyArray_Descr *res = (PyArray_Descr *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_Close(ctx, h_cls);
    return res;
}

NPY_NO_EXPORT PyArray_Descr *
discover_descr_from_pyobject_function_trampoline(PyArray_DTypeMeta *cls, PyObject *obj)
{
    HPyContext *ctx = npy_get_context();
    HPy h_cls = HPy_FromPyObject(ctx, (PyObject *)cls);
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPy h_res = HNPY_DT_CALL_discover_descr_from_pyobject(ctx, h_cls, h_obj);
    PyArray_Descr *res = (PyArray_Descr *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_Close(ctx, h_obj);
    HPy_Close(ctx, h_cls);
    return res;
}

NPY_NO_EXPORT HPy
hdiscover_descr_from_pyobject_function_trampoline(HPyContext *ctx, HPy cls, HPy obj)
{
    CAPI_WARN("hdiscover_descr_from_pyobject_function_trampoline: calling to legacy function");
    PyArray_DTypeMeta *py_cls = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, cls);
    PyObject *py_obj = HPy_AsPyObject(ctx, obj);
    PyArray_Descr *py_res = NPY_DT_CALL_discover_descr_from_pyobject(py_cls, py_obj);
    HPy res = HPy_FromPyObject(ctx, (PyObject *)py_res);
    Py_XDECREF(py_res);
    Py_XDECREF(py_obj);
    Py_XDECREF(py_cls);
    return res;
}

/**
 * This function takes a PyArray_Descr and replaces its base class with
 * a newly created dtype subclass (DTypeMeta instances).
 * There are some subtleties that need to be remembered when doing this,
 * first for the class objects itself it could be either a HeapType or not.
 * Since we are defining the DType from C, we will not make it a HeapType,
 * thus making it identical to a typical *static* type (except that we
 * malloc it). We could do it the other way, but there seems no reason to
 * do so.
 *
 * The DType instances (the actual dtypes or descriptors), are based on
 * prototypes which are passed in. These should not be garbage collected
 * and thus Py_TPFLAGS_HAVE_GC is not set. (We could allow this, but than
 * would have to allocate a new object, since the GC needs information before
 * the actual struct).
 *
 * The above is the reason why we should works exactly like we would for a
 * static type here.
 * Otherwise, we blurry the lines between C-defined extension classes
 * and Python subclasses. e.g. `class MyInt(int): pass` is very different
 * from our `class Float64(np.dtype): pass`, because the latter should not
 * be a HeapType and its instances should be exact PyArray_Descr structs.
 *
 * @param descr The descriptor that should be wrapped.
 * @param name The name for the DType, if NULL the type character is used.
 *
 * @returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(HPyContext *ctx, HPy h_descr, PyArray_Descr *descr)
{
    int result = -1;
    HPy h_typeobj = HPy_NULL;
    HPy h_PyArrayDescr_Type = HPy_NULL;
    HPy h_PyArrayDTypeMeta_Type = HPy_NULL;
    HPy h_new_dtype_type = HPy_NULL;

    HPy descr_type = HPy_Type(ctx, h_descr);
    HPy array_descr_type = HPyGlobal_Load(ctx, HPyArrayDescr_Type);
    int has_type_set = HPy_Is(ctx, descr_type, array_descr_type);
    HPy_Close(ctx, array_descr_type);
    if (!has_type_set) {
        /* Accept if the type was filled in from an existing builtin dtype */
        for (int i = 0; i < NPY_NTYPES; i++) {
            HPy builtin = HPyArray_DescrFromType(ctx, i);
            has_type_set = HPy_Is(ctx, descr_type, HPy_Type(ctx, builtin));
            HPy_Close(ctx, builtin);
            if (has_type_set) {
                break;
            }
        }
    }
    HPy_Close(ctx, descr_type);

    if (!has_type_set) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "During creation/wrapping of legacy DType, the original class "
                "was not of PyArrayDescr_Type (it is replaced in this step). "
                "The extension creating a custom DType for type %S must be "
                "modified to ensure `Py_TYPE(descr) == &PyArrayDescr_Type` or "
                "that of an existing dtype (with the assumption it is just "
                "copied over and can be replaced)."
                /*, descr->typeobj, Py_TYPE(descr)*/);
        goto cleanup;
    }

    /*
     * Note: we have no intention of freeing the memory again since this
     * behaves identically to static type definition (see comment above).
     * This is seems cleaner for the legacy API, in the new API both static
     * and heap types are possible (some difficulty arises from the fact that
     * these are instances of DTypeMeta and not type).
     * In particular our own DTypes can be true static declarations.
     * However, this function remains necessary for legacy user dtypes.
     */

    h_typeobj = HPyField_Load(ctx, h_descr, descr->typeobj);
    const char *scalar_name = HPyType_GetName(ctx, h_typeobj);

    /*
     * We have to take only the name, and ignore the module to get
     * a reasonable __name__, since static types are limited in this regard
     * (this is not ideal, but not a big issue in practice).
     * This is what Python does to print __name__ for static types.
     */
    const char *dot = strrchr(scalar_name, '.');
    if (dot) {
        scalar_name = dot + 1;
    }
    Py_ssize_t name_length = strlen(scalar_name) + 14;

    char *tp_name = PyMem_Malloc(name_length);
    if (tp_name == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    snprintf(tp_name, name_length, "numpy.dtype[%s]", scalar_name);

    NPY_DType_Slots *dt_slots = PyMem_Malloc(sizeof(NPY_DType_Slots));
    if (dt_slots == NULL) {
        PyMem_Free(tp_name);
        return -1;
    }
    memset(dt_slots, '\0', sizeof(NPY_DType_Slots));

    // We memcpy from static prototype so that whole HPyType_Spec is initialized
    HPyType_Spec New_PyArrayDescr_spec;
    memcpy(&New_PyArrayDescr_spec, &New_PyArrayDescr_spec_prototype, sizeof(New_PyArrayDescr_spec));
    New_PyArrayDescr_spec.name = tp_name;

    h_PyArrayDescr_Type = HPyGlobal_Load(ctx, HPyArrayDescr_Type);
    h_PyArrayDTypeMeta_Type = HPyGlobal_Load(ctx, HPyArrayDTypeMeta_Type);

    HPyType_SpecParam new_dtype_params[] = {
        { HPyType_SpecParam_Base, h_PyArrayDescr_Type},
        // HPY STEVE TODO: do I need to specify the metaclass if the base already has it as its metaclass?
        { HPyType_SpecParam_Metaclass, h_PyArrayDTypeMeta_Type },
        { 0 }
    };

    h_new_dtype_type = HPyType_FromSpec(ctx, &New_PyArrayDescr_spec, new_dtype_params);
    if (HPy_IsNull(h_new_dtype_type)) {
        goto cleanup;
    }

    PyArray_DTypeMeta *new_dtype_data = PyArray_DTypeMeta_AsStruct(ctx, h_new_dtype_type);
    new_dtype_data->dt_slots = dt_slots;

    HPy h_castingimpls = HPyDict_New(ctx);
    if (HPy_IsNull(h_castingimpls)) {
        goto cleanup;
    }
    HPyField_Store(ctx, h_new_dtype_type, &dt_slots->castingimpls, h_castingimpls);
    HPy_Close(ctx, h_castingimpls);

    /*
     * Fill DTypeMeta information that varies between DTypes, any variable
     * type information would need to be set before PyType_Ready().
     */
    HPyField_Store(ctx, h_new_dtype_type, &new_dtype_data->singleton, h_descr);
    HPyField_Store(ctx, h_new_dtype_type, &new_dtype_data->scalar_type, h_typeobj);
    new_dtype_data->type_num = descr->type_num;
    new_dtype_data->flags = NPY_DT_LEGACY;
    dt_slots->f = *(descr->f);

    /* Set default functions (correct for most dtypes, override below) */
    dt_slots->default_descr = nonparametric_default_descr;
    dt_slots->discover_descr_from_pyobject = discover_descr_from_pyobject_function_trampoline;
    dt_slots->hdiscover_descr_from_pyobject = (
            nonparametric_discover_descr_from_pyobject);
    dt_slots->is_known_scalar_type = python_builtins_are_known_scalar_types;
    dt_slots->common_dtype = default_builtin_common_dtype;
    dt_slots->hpy_common_dtype = hpy_default_builtin_common_dtype;
    dt_slots->common_instance = NULL;

    if (PyTypeNum_ISSIGNED(new_dtype_data->type_num)) {
        /* Convert our scalars (raise on too large unsigned and NaN, etc.) */
        dt_slots->is_known_scalar_type = signed_integers_is_known_scalar_types;
    }

    if (PyTypeNum_ISUSERDEF(descr->type_num)) {
        dt_slots->common_dtype = legacy_userdtype_common_dtype_function;
        dt_slots->hpy_common_dtype = hpy_legacy_userdtype_common_dtype_function;
    }
    else if (descr->type_num == NPY_OBJECT) {
        dt_slots->common_dtype = object_common_dtype;
        dt_slots->hpy_common_dtype = hpy_object_common_dtype;
    }
    else if (PyTypeNum_ISDATETIME(descr->type_num)) {
        /* Datetimes are flexible, but were not considered previously */
        new_dtype_data->flags |= NPY_DT_PARAMETRIC;
        dt_slots->default_descr = datetime_and_timedelta_default_descr;
        dt_slots->discover_descr_from_pyobject = (
                discover_datetime_and_timedelta_from_pyobject);
        dt_slots->hdiscover_descr_from_pyobject = (
                hdiscover_descr_from_pyobject_function_trampoline);
        dt_slots->hpy_common_dtype = hpy_datetime_common_dtype;
        dt_slots->hpy_common_dtype = hpy_datetime_common_dtype;
        dt_slots->common_instance = datetime_type_promotion;
        dt_slots->hpy_common_instance = hpy_datetime_type_promotion;
        if (descr->type_num == NPY_DATETIME) {
            dt_slots->is_known_scalar_type = datetime_known_scalar_types;
        }
    }
    else if (PyTypeNum_ISFLEXIBLE(descr->type_num)) {
        new_dtype_data->flags |= NPY_DT_PARAMETRIC;
        if (descr->type_num == NPY_VOID) {
            dt_slots->default_descr = void_default_descr;
            dt_slots->discover_descr_from_pyobject = (
                    void_discover_descr_from_pyobject);
            dt_slots->hdiscover_descr_from_pyobject = (
                    hdiscover_descr_from_pyobject_function_trampoline);
            dt_slots->common_instance = void_common_instance;
            dt_slots->hpy_common_instance = hpy_void_common_instance;
        }
        else {
            dt_slots->default_descr = string_and_unicode_default_descr;
            dt_slots->is_known_scalar_type = string_known_scalar_types;
            dt_slots->discover_descr_from_pyobject = (
                    string_discover_descr_from_pyobject);
            dt_slots->hdiscover_descr_from_pyobject = (
                    hdiscover_descr_from_pyobject_function_trampoline);
            dt_slots->common_dtype = string_unicode_common_dtype;
            dt_slots->hpy_common_dtype = hpy_string_unicode_common_dtype;
            dt_slots->common_instance = string_unicode_common_instance;
            dt_slots->hpy_common_instance = hpy_string_unicode_common_instance;
        }
    }

    HPy descr_typeobj = HPyField_Load(ctx, h_descr, descr->typeobj);
    if (_PyArray_MapPyTypeToDType(ctx, h_new_dtype_type, descr_typeobj,
            PyTypeNum_ISUSERDEF(new_dtype_data->type_num)) < 0) {
        HPy_Close(ctx, descr_typeobj);
        goto cleanup;
    }
    HPy_Close(ctx, descr_typeobj);

    /* Finally, replace the current class of the descr */
    // Re HPy_SetType: in longer term, this can be refactored: it seems that
    // this whole function is here to support some legacy API, which we can keep
    // in C API, and to initialize singleton descriptors like BOOL_Descr, which
    // we can initialize with the right type already to avoid setting it ex-post
    HPy_SetType(ctx, h_descr, h_new_dtype_type);
    result = 0;

cleanup:
    HPy_Close(ctx, h_typeobj);
    HPy_Close(ctx, h_PyArrayDescr_Type);
    HPy_Close(ctx, h_PyArrayDTypeMeta_Type);
    HPy_Close(ctx, h_new_dtype_type);
    return result;
}


/*
 * Simple exposed information, defined for each DType (class).
 */

HPyDef_GET(dtypemeta_get_abstract, "_abstract")
static HPy
dtypemeta_get_abstract_get(HPyContext *ctx, HPy self, void *ptr) {
    return HPyBool_FromLong(ctx, NPY_DT_is_abstract(PyArray_DTypeMeta_AsStruct(ctx, self)));
}

HPyDef_GET(dtypemeta_get_parametric, "_parametric")
static HPy
dtypemeta_get_parametric_get(HPyContext *ctx, HPy self, void *ptr) {
    return HPyBool_FromLong(ctx, NPY_DT_is_parametric(PyArray_DTypeMeta_AsStruct(ctx, self)));
}

HPyDef_MEMBER(dtypemeta_member_type, "type", HPyMember_OBJECT, offsetof(PyArray_DTypeMeta, scalar_type), .readonly=1)

// TODO HPY LABS PORT: global set in module init:
NPY_NO_EXPORT PyTypeObject *PyArrayDTypeMeta_Type;
NPY_NO_EXPORT HPyGlobal HPyArrayDTypeMeta_Type;

/* NOTE: Originally, DTypeMeta had a 'tp_new' slot (that would just prevent
   subclassing by throwing an error) but metaclasses must not define a custom
   constructor. */
NPY_NO_EXPORT HPyDef *PyArrayDTypeMeta_Type_defines[] = {
    &dtypemeta_init,
    &DTypeMeta_traverse,
    &DTypeMeta_destroy,
    &dtypemeta_get_abstract,
    &dtypemeta_get_parametric,
    &dtypemeta_member_type,
    0,
};

NPY_NO_EXPORT HPyType_Spec PyArrayDTypeMeta_Type_spec = {
    .name = "numpy._DTypeMeta",
    .basicsize = sizeof(PyArray_DTypeMeta),
    /* Types are garbage collected (see dtypemeta_is_gc documentation) */
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_HAVE_GC,
    .doc = "Preliminary NumPy API: The Type of NumPy DTypes (metaclass)",
    .builtin_shape = SHAPE(PyArray_DTypeMeta),
    .defines = PyArrayDTypeMeta_Type_defines,
};

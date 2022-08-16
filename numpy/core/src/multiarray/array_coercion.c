#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/npy_3kcompat.h"

#include "lowlevel_strided_loops.h"
#include "numpy/arrayobject.h"

#include "descriptor.h"
#include "convert_datatype.h"
#include "common_dtype.h"
#include "dtypemeta.h"

#include "array_coercion.h"
#include "ctors.h"
#include "common.h"
#include "_datetime.h"
#include "npy_import.h"
#include "scalarapi.h"


/*
 * This file defines helpers for some of the ctors.c functions which
 * create an array from Python sequences and types.
 * When creating an array with ``np.array(...)`` we have to do two main things:
 *
 * 1. Find the exact shape of the resulting array
 * 2. Find the correct dtype of the resulting array.
 *
 * In most cases these two things are can be done in a single processing step.
 * There are in principle three different calls that should be distinguished:
 *
 * 1. The user calls ``np.array(..., dtype=np.dtype("<f8"))``
 * 2. The user calls ``np.array(..., dtype="S")``
 * 3. The user calls ``np.array(...)``
 *
 * In the first case, in principle only the shape needs to be found. In the
 * second case, the DType class (e.g. string) is already known but the DType
 * instance (e.g. length of the string) has to be found.
 * In the last case the DType class needs to be found as well. Note that
 * it is not necessary to find the DType class of the entire array, but
 * the DType class needs to be found for each element before the actual
 * dtype instance can be found.
 *
 * Further, there are a few other things to keep in mind when coercing arrays:
 *
 *   * For UFunc promotion, Python scalars need to be handled specially to
 *     allow value based casting.  This requires python complex/float to
 *     have their own DTypes.
 *   * It is necessary to decide whether or not a sequence is an element.
 *     For example tuples are considered elements for structured dtypes, but
 *     otherwise are considered sequences.
 *     This means that if a dtype is given (either as a class or instance),
 *     it can effect the dimension discovery part.
 *     For the "special" NumPy types structured void and "c" (single character)
 *     this is special cased.  For future user-types, this is currently
 *     handled by providing calling an `is_known_scalar` method.  This method
 *     currently ensures that Python numerical types are handled quickly.
 *
 * In the initial version of this implementation, it is assumed that dtype
 * discovery can be implemented sufficiently fast.  That is, it is not
 * necessary to create fast paths that only find the correct shape e.g. when
 * ``dtype=np.dtype("f8")`` is given.
 *
 * The code here avoid multiple conversion of array-like objects (including
 * sequences). These objects are cached after conversion, which will require
 * additional memory, but can drastically speed up coercion from array like
 * objects.
 */


/*
 * For finding a DType quickly from a type, it is easiest to have a
 * a mapping of pytype -> DType.
 * TODO: This mapping means that it is currently impossible to delete a
 *       pair of pytype <-> DType.  To resolve this, it is necessary to
 *       weakly reference the pytype. As long as the pytype is alive, we
 *       want to be able to use `np.array([pytype()])`.
 *       It should be possible to retrofit this without too much trouble
 *       (all type objects support weak references).
 */
NPY_NO_EXPORT HPyGlobal _global_pytype_to_type_dict;

static inline PyObject *get_global_pytype_to_type_dict() {
    HPyContext *ctx = npy_get_context();
    HPy h = HPyGlobal_Load(ctx, _global_pytype_to_type_dict);
    PyObject *res = HPy_AsPyObject(ctx, h);
    HPy_Close(ctx, h);
    Py_DECREF(res); // simulate a borrowed ref for ease of using this where borrowed ref was assumed
    return res;
}


/* Enum to track or signal some things during dtype and shape discovery */
enum _dtype_discovery_flags {
    FOUND_RAGGED_ARRAY = 1 << 0,
    GAVE_SUBCLASS_WARNING = 1 << 1,
    PROMOTION_FAILED = 1 << 2,
    DISCOVER_STRINGS_AS_SEQUENCES = 1 << 3,
    DISCOVER_TUPLES_AS_ELEMENTS = 1 << 4,
    MAX_DIMS_WAS_REACHED = 1 << 5,
    DESCRIPTOR_WAS_SET = 1 << 6,
};


/**
 * Adds known sequence types to the global type dictionary, note that when
 * a DType is passed in, this lookup may be ignored.
 *
 * @return -1 on error 0 on success
 */
NPY_NO_EXPORT int
init_global_pytype_to_type_dict(HPyContext *ctx)
{
    int res = 0;
    HPy d = HPyDict_New(ctx);

    /* Add the basic Python sequence types */
    res = HPy_SetItem(ctx, d, ctx->h_ListType, ctx->h_None);
    if (res < 0) {
        goto cleanup;
    }

    res = HPy_SetItem(ctx, d, ctx->h_TupleType, ctx->h_None);
    if (res < 0) {
        goto cleanup;
    }
    /* NumPy Arrays are not handled as scalars */
    HPy arr_type = HPyGlobal_Load(ctx, HPyArray_Type);
    res = HPy_SetItem(ctx, d, arr_type, ctx->h_None);
    HPy_Close(ctx, arr_type);
    if (res < 0) {
        goto cleanup;
    }

    HPyGlobal_Store(ctx, &_global_pytype_to_type_dict, d);
cleanup:
    HPy_Close(ctx, d);
    return res;
}


/**
 * Add a new mapping from a python type to the DType class. For a user
 * defined legacy dtype, this function does nothing unless the pytype
 * subclass from `np.generic`.
 *
 * This assumes that the DType class is guaranteed to hold on the
 * python type (this assumption is guaranteed).
 * This functionality supersedes ``_typenum_fromtypeobj``.
 *
 * @param DType DType to map the python type to
 * @param pytype Python type to map from
 * @param userdef Whether or not it is user defined. We ensure that user
 *        defined scalars subclass from our scalars (for now).
 */
NPY_NO_EXPORT int
_PyArray_MapPyTypeToDType(
        HPyContext *ctx, /*PyArray_DTypeMeta*/ HPy h_DType, /*PyTypeObject*/ HPy h_pytype, npy_bool userdef)
{
    int res;
    HPy d = HPy_NULL;
    HPy generic_arr_type = HPyGlobal_Load(ctx, HPyGenericArrType_Type);

    // TODO HPY LABS PORT: PyObject_IsSubclass
    if (userdef && !HPyType_IsSubtype(ctx, h_pytype, generic_arr_type)) {
        /*
         * We expect that user dtypes (for now) will subclass some numpy
         * scalar class to allow automatic discovery.
         */
        PyArray_DTypeMeta *meta_data = PyArray_DTypeMeta_AsStruct(ctx, h_DType);
        if (NPY_DT_is_legacy(meta_data)) {
            /*
             * For legacy user dtypes, discovery relied on subclassing, but
             * arbitrary type objects are supported, so do nothing.
             */
            res = 0;
            goto cleanup;
        }
        /*
         * We currently enforce that user DTypes subclass from `np.generic`
         * (this should become a `np.generic` base class and may be lifted
         * entirely).
         */
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "currently it is only possible to register a DType "
                "for scalars deriving from `np.generic`, got '%S'."
                /*, (PyObject *)pytype*/);
        res = -1;
        goto cleanup;
    }

    /* Create the global dictionary if it does not exist */
    // Initialized eagerly in HPy
    // if (NPY_UNLIKELY(_global_pytype_to_type_dict == NULL)) {
    //     _global_pytype_to_type_dict = PyDict_New();
    //     if (_global_pytype_to_type_dict == NULL) {
    //         return -1;
    //     }
    //     if (_prime_global_pytype_to_type_dict() < 0) {
    //         return -1;
    //     }
    // }

    d = HPyGlobal_Load(ctx, _global_pytype_to_type_dict);
    res = HPy_Contains(ctx, d, h_pytype);
    if (res < 0) {
        goto cleanup;
    }
    else if (res) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "Can only map one python type to DType.");
        res = -1;
        goto cleanup;
    }

    res = HPy_SetItem(ctx, d, h_pytype, h_DType);

cleanup:
    HPy_Close(ctx, generic_arr_type);
    HPy_Close(ctx, d);
    return res;
}


/**
 * Lookup the DType for a registered known python scalar type.
 *
 * @param pytype Python Type to look up
 * @return DType, None if it a known non-scalar, or NULL if an unknown object.
 */
static NPY_INLINE HPy /* (PyArray_DTypeMeta *) */
npy_discover_dtype_from_pytype(HPyContext *ctx, HPy /* (PyTypeObject *) */ pytype)
{
    HPy DType;

    if (HPyGlobal_Is(ctx, pytype, HPyArray_Type)) {
        return HPy_Dup(ctx, ctx->h_None);
    }

    HPy pytype_to_type_dict = HPyGlobal_Load(ctx, _global_pytype_to_type_dict);
    DType = HPyDict_GetItem(ctx, pytype_to_type_dict, pytype);
    if (HPy_IsNull(DType)) {
        /* the python type is not known */
        return HPy_NULL;
    }

    if (HPy_Is(ctx, DType, ctx->h_None)) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    assert(HPyGlobal_TypeCheck(ctx, DType, HPyArrayDTypeMeta_Type));
    return DType;
}

static NPY_INLINE HPy /* (PyArray_DTypeMeta *) */
hdiscover_dtype_from_pyobject(
        HPyContext *ctx,
        HPy obj, enum _dtype_discovery_flags *flags,
        HPy /* (PyArray_DTypeMeta *) */ fixed_DType)
{
    HPy res;
    HPy obj_type = HPy_Type(ctx, obj);
    if (!HPy_IsNull(fixed_DType)) {
        /*
         * Let the given DType handle the discovery.  This is when the
         * scalar-type matches exactly, or the DType signals that it can
         * handle the scalar-type.  (Even if it cannot handle here it may be
         * asked to attempt to do so later, if no other matching DType exists.)
         */
        PyArray_DTypeMeta *fixed_DType_data = PyArray_DTypeMeta_AsStruct(ctx, fixed_DType);
        HPy scalar_type = HPyField_Load(ctx, fixed_DType, fixed_DType_data->scalar_type);
        int is_scalar_type = HPy_Is(ctx, obj_type, scalar_type);
        if (is_scalar_type ||
                HNPY_DT_CALL_is_known_scalar_type(ctx, fixed_DType, fixed_DType_data, obj_type)) {
            res = HPy_Dup(ctx, fixed_DType);
            goto cleanup;
        }
    }

    HPy DType = npy_discover_dtype_from_pytype(ctx, obj_type);
    if (!HPy_IsNull(DType)) {
        res = DType;
        goto cleanup;
    }
    /*
     * At this point we have not found a clear mapping, but mainly for
     * backward compatibility we have to make some further attempts at
     * interpreting the input as a known scalar type.
     */
    HPy legacy_descr; /* (PyArray_Descr *) */
    if (HPyArray_IsScalar(ctx, obj, Generic)) {
        legacy_descr = HPyArray_DescrFromScalar(ctx, obj);
        if (HPy_IsNull(legacy_descr)) {
            return HPy_NULL;
        }
    }
    else if (flags == NULL) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    else if (HPyBytes_Check(ctx, obj)) {
        legacy_descr = HPyArray_DescrFromType(ctx, NPY_BYTE);
    }
    else if (HPyUnicode_Check(ctx, obj)) {
        legacy_descr = HPyArray_DescrFromType(ctx, NPY_UNICODE);
    }
    else {
        legacy_descr = _array_find_python_scalar_type(ctx, obj);
    }

    if (!HPy_IsNull(legacy_descr)) {
        DType = HNPY_DTYPE(ctx, legacy_descr);
        HPy_Close(ctx, legacy_descr);
        /* TODO: Enable warning about subclass handling */
        if ((0) && !((*flags) & GAVE_SUBCLASS_WARNING)) {
            if (HPY_DEPRECATE_FUTUREWARNING(ctx,
                    "in the future NumPy will not automatically find the "
                    "dtype for subclasses of scalars known to NumPy (i.e. "
                    "python types). Use the appropriate `dtype=...` to create "
                    "this array. This will use the `object` dtype or raise "
                    "an error in the future.") < 0) {
                return HPy_NULL;
            }
            *flags |= GAVE_SUBCLASS_WARNING;
        }
        return DType;
    }
    return HPy_Dup(ctx, ctx->h_None);

cleanup:
    HPy_Close(ctx, obj_type);
    return res;
}

/**
 * Find the correct DType class for the given python type. If flags is NULL
 * this is not used to discover a dtype, but only for conversion to an
 * existing dtype. In that case the Python (not NumPy) scalar subclass
 * checks are skipped.
 *
 * @param obj The python object, mainly type(pyobj) is used, the object
 *        is passed to reuse existing code at this time only.
 * @param flags Flags used to know if warnings were already given. If
 *        flags is NULL, this is not
 * @param fixed_DType if not NULL, will be checked first for whether or not
 *        it can/wants to handle the (possible) scalar value.
 * @return New reference to either a DType class, Py_None, or NULL on error.
 */
static NPY_INLINE PyArray_DTypeMeta *
discover_dtype_from_pyobject(
        PyObject *obj, enum _dtype_discovery_flags *flags,
        PyArray_DTypeMeta *fixed_DType)
{
    HPyContext *ctx = npy_get_context();
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPy h_fixed_DType = HPy_FromPyObject(ctx, (PyObject *)fixed_DType);
    HPy h_res = hdiscover_dtype_from_pyobject(ctx, h_obj, flags, h_fixed_DType);
    PyArray_DTypeMeta *res = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_Close(ctx, h_fixed_DType);
    HPy_Close(ctx, h_obj);
    return res;
}

/**
 * Discover the correct descriptor from a known DType class and scalar.
 * If the fixed DType can discover a dtype instance/descr all is fine,
 * if it cannot and DType is used instead, a cast will have to be tried.
 *
 * @param fixed_DType A user provided fixed DType, can be NULL
 * @param DType A discovered DType (by discover_dtype_from_pyobject);
 *        this can be identical to `fixed_DType`, if it obj is a
 *        known scalar. Can be `NULL` indicating no known type.
 * @param obj The Python scalar object. At the time of calling this function
 *        it must be known that `obj` should represent a scalar.
 */
static NPY_INLINE PyArray_Descr *
find_scalar_descriptor(
        PyArray_DTypeMeta *fixed_DType, PyArray_DTypeMeta *DType,
        PyObject *obj)
{
    PyArray_Descr *descr;

    if (DType == NULL && fixed_DType == NULL) {
        /* No known DType and no fixed one means we go to object. */
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    else if (DType == NULL) {
        /*
         * If no DType is known/found, give the fixed give one a second
         * chance.  This allows for example string, to call `str(obj)` to
         * figure out the length for arbitrary objects.
         */
        descr = NPY_DT_CALL_discover_descr_from_pyobject(fixed_DType, obj);
    }
    else {
        descr = NPY_DT_CALL_discover_descr_from_pyobject(DType, obj);
    }
    if (descr == NULL) {
        return NULL;
    }
    if (fixed_DType == NULL) {
        return descr;
    }

    Py_SETREF(descr, PyArray_CastDescrToDType(descr, fixed_DType));
    return descr;
}

static NPY_INLINE HPy
hpy_find_scalar_descriptor(HPyContext *ctx,
        HPy /* (PyArray_DTypeMeta *) */ fixed_DType, HPy /* (PyArray_DTypeMeta *) */ DType,
        HPy obj)
{
    HPy descr; /* (PyArray_Descr *) */

    if (HPy_IsNull(DType) && HPy_IsNull(fixed_DType)) {
        /* No known DType and no fixed one means we go to object. */
        return HPyArray_DescrFromType(ctx, NPY_OBJECT);
    }
    else if (HPy_IsNull(DType)) {
        /*
         * If no DType is known/found, give the fixed give one a second
         * chance.  This allows for example string, to call `str(obj)` to
         * figure out the length for arbitrary objects.
         */
        descr = HNPY_DT_CALL_discover_descr_from_pyobject(ctx, fixed_DType, obj);
    }
    else {
        descr = HNPY_DT_CALL_discover_descr_from_pyobject(ctx, DType, obj);
    }
    if (HPy_IsNull(descr)) {
        return HPy_NULL;
    }
    if (HPy_IsNull(fixed_DType)) {
        return descr;
    }

    HPy_SETREF(ctx, descr, HPyArray_CastDescrToDType(ctx, descr, fixed_DType));
    return descr;
}


/**
 * Assign a single element in an array from a python value.
 *
 * The dtypes SETITEM should only be trusted to generally do the right
 * thing if something is known to be a scalar *and* is of a python type known
 * to the DType (which should include all basic Python math types), but in
 * general a cast may be necessary.
 * This function handles the cast, which is for example hit when assigning
 * a float128 to complex128.
 *
 * At this time, this function does not support arrays (historically we
 * mainly supported arrays through `__float__()`, etc.). Such support should
 * possibly be added (although when called from `PyArray_AssignFromCache`
 * the input cannot be an array).
 * Note that this is also problematic for some array-likes, such as
 * `astropy.units.Quantity` and `np.ma.masked`.  These are used to us calling
 * `__float__`/`__int__` for 0-D instances in many cases.
 * Eventually, we may want to define this as wrong: They must use DTypes
 * instead of (only) subclasses.  Until then, here as well as in
 * `PyArray_AssignFromCache` (which already does this), we need to special
 * case 0-D array-likes to behave like arbitrary (unknown!) Python objects.
 *
 * @param descr
 * @param item
 * @param value
 * @return 0 on success -1 on failure.
 */
/*
 * TODO: This function should possibly be public API.
 */
NPY_NO_EXPORT int
PyArray_Pack(PyArray_Descr *descr, char *item, PyObject *value)
{
    HPyContext *ctx = npy_get_context();
    HPy h_descr = HPy_FromPyObject(ctx, (PyObject *)descr);
    HPy h_value = HPy_FromPyObject(ctx, value);
    int res = HPyArray_Pack(ctx, h_descr, item, h_value);
    HPy_Close(ctx, h_value);
    HPy_Close(ctx, h_descr);
    return res;
}

NPY_NO_EXPORT HPyGlobal g_dummy_arr;
static int g_dummy_arr_initialized;

NPY_NO_EXPORT int
HPyArray_Pack(HPyContext *ctx, HPy /* (PyArray_Descr *) */ descr, char *item, HPy value)
{
    HPy dummy_arr;
    if (!g_dummy_arr_initialized) {
        dummy_arr = dummy_array_new(ctx,
                HPy_NULL, NPY_ARRAY_WRITEABLE, HPy_NULL);
        if (HPy_IsNull(dummy_arr)) {
            return -1;
        }
        HPyGlobal_Store(ctx, &g_dummy_arr, dummy_arr);
        g_dummy_arr_initialized = 1;
    } else {
        dummy_arr = HPyGlobal_Load(ctx, g_dummy_arr); /* (PyArrayObject *) */
    }
    PyArrayObject *dummy_arr_data = PyArrayObject_AsStruct(ctx, dummy_arr);

    PyArray_Descr *descr_data = PyArray_Descr_AsStruct(ctx, descr);
    if (NPY_UNLIKELY(descr_data->type_num == NPY_OBJECT)) {
        /*
         * We always have store objects directly, casting will lose some
         * type information. Any other dtype discards the type information.
         * TODO: For a Categorical[object] this path may be necessary?
         * NOTE: OBJECT_setitem doesn't care about the NPY_ARRAY_ALIGNED flag
         */
        _hpy_set_descr(ctx, dummy_arr, dummy_arr_data, descr);
        int result = descr_data->f->setitem(ctx, value, item, dummy_arr);
        _hpy_set_descr(ctx, dummy_arr, dummy_arr_data, HPy_NULL);
        return result;
    }

    /* discover_dtype_from_pyobject includes a check for is_known_scalar_type */
    HPy descr_type = HNPY_DTYPE(ctx, descr);
    HPy DType = hdiscover_dtype_from_pyobject(
            ctx, value, NULL, descr_type); /* (PyArray_DTypeMeta *) */
    if (HPy_IsNull(DType)) {
        HPy_Close(ctx, descr_type);
        return -1;
    }
    if (HPy_Is(ctx, DType, descr_type) || HPy_Is(ctx, DType, ctx->h_None)) {
        HPy_Close(ctx, descr_type);
        /* We can set the element directly (or at least will try to) */
        HPy_Close(ctx, DType);
        _hpy_set_descr(ctx, dummy_arr, dummy_arr_data, descr);
        if (npy_is_aligned(item, descr_data->alignment)) {
            PyArray_ENABLEFLAGS(dummy_arr_data, NPY_ARRAY_ALIGNED);
        }
        else {
            PyArray_CLEARFLAGS(dummy_arr_data, NPY_ARRAY_ALIGNED);
        }
        int result = descr_data->f->setitem(ctx, value, item, dummy_arr);
        _hpy_set_descr(ctx, dummy_arr, dummy_arr_data, HPy_NULL);
        return result;
    }
    HPy tmp_descr; /* (PyArray_Descr *) */
    tmp_descr = HNPY_DT_CALL_discover_descr_from_pyobject(ctx, DType, value);
    HPy_Close(ctx, DType);
    if (HPy_IsNull(tmp_descr)) {
        return -1;
    }

    PyArray_Descr *tmp_descr_data = PyArray_Descr_AsStruct(ctx, tmp_descr);
    // TODO HPY LABS PORT: PyObject_Malloc
    // char *data = PyObject_Malloc(tmp_descr_data->elsize);
    char *data = malloc(tmp_descr_data->elsize);
    if (data == NULL) {
        HPyErr_NoMemory(ctx);
        HPy_Close(ctx, tmp_descr);
        return -1;
    }
    if (PyDataType_FLAGCHK(tmp_descr_data, NPY_NEEDS_INIT)) {
        memset(data, 0, tmp_descr_data->elsize);
    }
    _hpy_set_descr(ctx, dummy_arr, dummy_arr_data, tmp_descr);
    if (npy_is_aligned(item, tmp_descr_data->alignment)) {
        PyArray_ENABLEFLAGS(dummy_arr_data, NPY_ARRAY_ALIGNED);
    }
    else {
        PyArray_CLEARFLAGS(dummy_arr_data, NPY_ARRAY_ALIGNED);
    }
    if (descr_data->f->setitem(ctx, value, item, dummy_arr) < 0) {
        // TODO HPY LABS PORT: PyObject_Free
        // PyObject_Free(data);
        free(data);
        _hpy_set_descr(ctx, dummy_arr, dummy_arr_data, HPy_NULL);
        HPy_Close(ctx, tmp_descr);
        return -1;
    }
    _hpy_set_descr(ctx, dummy_arr, dummy_arr_data, HPy_NULL);
    if (PyDataType_REFCHK(tmp_descr_data)) {
        /* We could probably use move-references above */
        PyArray_Item_INCREF(data, (PyArray_Descr *)HPy_AsPyObject(ctx, tmp_descr));
    }

    int res = 0;
    int needs_api = 0;
    HNPY_cast_info cast_info;
    if (HPyArray_GetDTypeTransferFunction(ctx,
            0, 0, 0, tmp_descr, descr, 0, &cast_info,
            &needs_api) == NPY_FAIL) {
        res = -1;
        goto finish;
    }
    char *args[2] = {data, item};
    const npy_intp strides[2] = {0, 0};
    const npy_intp length = 1;
    if (cast_info.func(ctx, &cast_info.context,
            args, &length, strides, cast_info.auxdata) < 0) {
        res = -1;
    }
    HNPY_cast_info_xfree(ctx, &cast_info);

  finish:
    if (PyDataType_REFCHK(tmp_descr_data)) {
        /* We could probably use move-references above */
        PyArray_Item_XDECREF(data, (PyArray_Descr *)HPy_AsPyObject(ctx, tmp_descr));
    }
    // TODO HPY LABS PORT: PyObject_Free
    // PyObject_Free(data);
    free(data);
    HPy_Close(ctx, tmp_descr);
    return res;
}

//NPY_NO_EXPORT int
//HPyArray_Pack(HPyContext *ctx, HPy /* (PyArray_Descr *) */ descr, char *item, HPy value)
//{
//    CAPI_WARN("HPyArray_Pack: call to PyArray_Pack");
//    PyArray_Descr *py_descr = (PyArray_Descr *)HPy_AsPyObject(ctx, descr);
//    PyObject *py_value = HPy_AsPyObject(ctx, value);
//    int res = PyArray_Pack(py_descr, item, py_value);
//    Py_DECREF(py_value);
//    Py_DECREF(py_descr);
//    return res;
//}


static int
update_shape(int curr_ndim, int *max_ndim,
             npy_intp out_shape[NPY_MAXDIMS], int new_ndim,
             const npy_intp new_shape[NPY_MAXDIMS], npy_bool sequence,
             enum _dtype_discovery_flags *flags)
{
    int success = 0;  /* unsuccessful if array is ragged */
    const npy_bool max_dims_reached = *flags & MAX_DIMS_WAS_REACHED;

    if (curr_ndim + new_ndim > *max_ndim) {
        success = -1;
        /* Only update/check as many dims as possible, max_ndim is unchanged */
        new_ndim = *max_ndim - curr_ndim;
    }
    else if (!sequence && (*max_ndim != curr_ndim + new_ndim)) {
        /*
         * Sequences do not update max_ndim, otherwise shrink and check.
         * This is depth first, so if it is already set, `out_shape` is filled.
         */
        *max_ndim = curr_ndim + new_ndim;
        /* If a shape was already set, this is also ragged */
        if (max_dims_reached) {
            success = -1;
        }
    }
    for (int i = 0; i < new_ndim; i++) {
        npy_intp curr_dim = out_shape[curr_ndim + i];
        npy_intp new_dim = new_shape[i];

        if (!max_dims_reached) {
            out_shape[curr_ndim + i] = new_dim;
        }
        else if (new_dim != curr_dim) {
            /* The array is ragged, and this dimension is unusable already */
            success = -1;
            if (!sequence) {
                /* Remove dimensions that we cannot use: */
                *max_ndim -= new_ndim - i;
            }
            else {
                assert(i == 0);
                /* max_ndim is usually not updated for sequences, so set now: */
                *max_ndim = curr_ndim;
            }
            break;
        }
    }
    if (!sequence) {
        *flags |= MAX_DIMS_WAS_REACHED;
    }
    return success;
}


#define COERCION_CACHE_CACHE_SIZE 5
static int _coercion_cache_num = 0;
static coercion_cache_obj *_coercion_cache_cache[COERCION_CACHE_CACHE_SIZE];

static NPY_INLINE int
hnpy_new_coercion_cache(HPyContext *ctx,
        HPy converted_obj, HPy arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr, int ndim)
{
    coercion_cache_obj *cache;
    if (_coercion_cache_num > 0) {
        _coercion_cache_num--;
        cache = _coercion_cache_cache[_coercion_cache_num];
    }
    else {
        cache = PyMem_Malloc(sizeof(coercion_cache_obj));
    }
    if (cache == NULL) {
        HPy_Close(ctx, arr_or_sequence);
        PyErr_NoMemory();
        return -1;
    }
    cache->converted_obj = HPy_Dup(ctx, converted_obj);
    cache->arr_or_sequence = HPy_Dup(ctx, arr_or_sequence);
    cache->sequence = sequence;
    cache->depth = ndim;
    cache->next = NULL;
    **next_ptr = cache;
    *next_ptr = &(cache->next);
    return 0;
}

/*
 * Steals a reference to the object.
 */
static NPY_INLINE int
npy_new_coercion_cache(
        PyObject *converted_obj, PyObject *arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr, int ndim)
{
    HPyContext *ctx = npy_get_context();
    HPy h_converted_obj = HPy_FromPyObject(ctx, converted_obj);
    Py_XDECREF(converted_obj); /* the object is stolen */
    HPy h_arr_or_sequence = HPy_FromPyObject(ctx, arr_or_sequence);
    Py_XDECREF(arr_or_sequence); /* the object is stolen */
    int res = hnpy_new_coercion_cache(ctx, h_converted_obj, h_arr_or_sequence, sequence, next_ptr, ndim);
    HPy_Close(ctx, h_converted_obj);
    HPy_Close(ctx, h_arr_or_sequence);
    return res;
}

/**
 * Unlink coercion cache item.
 *
 * @param current
 * @return next coercion cache object (or NULL)
 */
NPY_NO_EXPORT coercion_cache_obj *
hnpy_unlink_coercion_cache(HPyContext *ctx, coercion_cache_obj *current)
{
    coercion_cache_obj *next = current->next;
    HPy_Close(ctx, current->arr_or_sequence);
    if (_coercion_cache_num < COERCION_CACHE_CACHE_SIZE) {
        _coercion_cache_cache[_coercion_cache_num] = current;
        _coercion_cache_num++;
    }
    else {
        PyMem_Free(current);
    }
    return next;
}

NPY_NO_EXPORT coercion_cache_obj *
npy_unlink_coercion_cache(coercion_cache_obj *current)
{
    return hnpy_unlink_coercion_cache(npy_get_context(), current);
}

NPY_NO_EXPORT void
hnpy_free_coercion_cache(HPyContext *ctx, coercion_cache_obj *next) {
    /* We only need to check from the last used cache pos */
    while (next != NULL) {
        next = hnpy_unlink_coercion_cache(ctx, next);
    }
}

NPY_NO_EXPORT void
npy_free_coercion_cache(coercion_cache_obj *next) {
    hnpy_free_coercion_cache(npy_get_context(), next);
}

#undef COERCION_CACHE_CACHE_SIZE

/**
 * Do the promotion step and possible casting. This function should
 * never be called if a descriptor was requested. In that case the output
 * dtype is not of importance, so we must not risk promotion errors.
 *
 * @param out_descr The current descriptor.
 * @param descr The newly found descriptor to promote with
 * @param fixed_DType The user provided (fixed) DType or NULL
 * @param flags dtype discover flags to signal failed promotion.
 * @return -1 on error, 0 on success.
 */
static NPY_INLINE int
handle_promotion(PyArray_Descr **out_descr, PyArray_Descr *descr,
        PyArray_DTypeMeta *fixed_DType, enum _dtype_discovery_flags *flags)
{
    assert(!(*flags & DESCRIPTOR_WAS_SET));

    if (*out_descr == NULL) {
        Py_INCREF(descr);
        *out_descr = descr;
        return 0;
    }
    PyArray_Descr *new_descr = PyArray_PromoteTypes(descr, *out_descr);
    if (NPY_UNLIKELY(new_descr == NULL)) {
        if (fixed_DType != NULL || PyErr_ExceptionMatches(PyExc_FutureWarning)) {
            /*
             * If a DType is fixed, promotion must not fail. Do not catch
             * FutureWarning (raised for string+numeric promotions). We could
             * only catch TypeError here or even always raise the error.
             */
            return -1;
        }
        PyErr_Clear();
        *flags |= PROMOTION_FAILED;
        /* Continue with object, since we may need the dimensionality */
        new_descr = PyArray_DescrFromType(NPY_OBJECT);
    }
    Py_SETREF(*out_descr, new_descr);
    return 0;
}

static NPY_INLINE int
hpy_handle_promotion(HPyContext *ctx, HPy /* (PyArray_Descr **) */ *out_descr, HPy /* (PyArray_Descr *) */ descr,
        HPy /* (PyArray_DTypeMeta *) */ fixed_DType, enum _dtype_discovery_flags *flags)
{
    assert(!(*flags & DESCRIPTOR_WAS_SET));

    if (HPy_IsNull(*out_descr)) {
        *out_descr = HPy_Dup(ctx, descr);
        return 0;
    }
    HPy new_descr = HPyArray_PromoteTypes(ctx, descr, *out_descr);

    if (NPY_UNLIKELY(HPy_IsNull(new_descr))) {
        if (!HPy_IsNull(fixed_DType) || HPyErr_ExceptionMatches(ctx, ctx->h_FutureWarning)) {
            /*
             * If a DType is fixed, promotion must not fail. Do not catch
             * FutureWarning (raised for string+numeric promotions). We could
             * only catch TypeError here or even always raise the error.
             */
            return -1;
        }
        HPyErr_Clear(ctx);
        *flags |= PROMOTION_FAILED;
        /* Continue with object, since we may need the dimensionality */
        new_descr = HPyArray_DescrFromType(ctx, NPY_OBJECT);
    }
    HPy_SETREF(ctx, *out_descr, new_descr);
    return 0;
}


/**
 * Handle a leave node (known scalar) during dtype and shape discovery.
 *
 * @param obj The python object or nested sequence to convert
 * @param curr_dims The current number of dimensions (depth in the recursion)
 * @param max_dims The maximum number of dimensions.
 * @param out_shape The discovered output shape, will be filled
 * @param fixed_DType The user provided (fixed) DType or NULL
 * @param flags used signal that this is a ragged array, used internally and
 *        can be expanded if necessary.
 * @param DType the DType class that should be used, or NULL, if not provided.
 *
 * @return 0 on success -1 on error
 */
static NPY_INLINE int
handle_scalar(
        PyObject *obj, int curr_dims, int *max_dims,
        PyArray_Descr **out_descr, npy_intp *out_shape,
        PyArray_DTypeMeta *fixed_DType,
        enum _dtype_discovery_flags *flags, PyArray_DTypeMeta *DType)
{
    PyArray_Descr *descr;

    if (update_shape(curr_dims, max_dims, out_shape,
            0, NULL, NPY_FALSE, flags) < 0) {
        *flags |= FOUND_RAGGED_ARRAY;
        return *max_dims;
    }
    if (*flags & DESCRIPTOR_WAS_SET) {
        /* no need to do any promotion */
        return *max_dims;
    }
    /* This is a scalar, so find the descriptor */
    descr = find_scalar_descriptor(fixed_DType, DType, obj);
    if (descr == NULL) {
        return -1;
    }
    if (handle_promotion(out_descr, descr, fixed_DType, flags) < 0) {
        Py_DECREF(descr);
        return -1;
    }
    Py_DECREF(descr);
    return *max_dims;
}

static NPY_INLINE int
hpy_handle_scalar(HPyContext *ctx,
        HPy obj, int curr_dims, int *max_dims,
        HPy /* (PyArray_Descr **) */ *out_descr, npy_intp *out_shape,
        HPy /* (PyArray_DTypeMeta *) */ fixed_DType,
        enum _dtype_discovery_flags *flags, HPy /* PyArray_DTypeMeta *) */ DType)
{
    HPy descr; /* (PyArray_Descr *) */

    if (update_shape(curr_dims, max_dims, out_shape,
            0, NULL, NPY_FALSE, flags) < 0) {
        *flags |= FOUND_RAGGED_ARRAY;
        return *max_dims;
    }
    if (*flags & DESCRIPTOR_WAS_SET) {
        /* no need to do any promotion */
        return *max_dims;
    }
    /* This is a scalar, so find the descriptor */
    descr = hpy_find_scalar_descriptor(ctx, fixed_DType, DType, obj);
    if (HPy_IsNull(descr)) {
        return -1;
    }
    if (hpy_handle_promotion(ctx, out_descr, descr, fixed_DType, flags) < 0) {
        HPy_Close(ctx, descr);
        return -1;
    }
    HPy_Close(ctx, descr);
    return *max_dims;
}


/**
 * Return the correct descriptor given an array object and a DType class.
 *
 * This is identical to casting the arrays descriptor/dtype to the new
 * DType class
 *
 * @param arr The array object.
 * @param DType The DType class to cast to (or NULL for convenience)
 * @param out_descr The output descriptor will set. The result can be NULL
 *        when the array is of object dtype and has no elements.
 *
 * @return -1 on failure, 0 on success.
 */
static int
find_descriptor_from_array(
        PyArrayObject *arr, PyArray_DTypeMeta *DType, PyArray_Descr **out_descr)
{
    enum _dtype_discovery_flags flags = 0;
    *out_descr = NULL;

    if (DType == NULL) {
        *out_descr = PyArray_DESCR(arr);
        Py_INCREF(*out_descr);
        return 0;
    }

    if (NPY_UNLIKELY(NPY_DT_is_parametric(DType) && PyArray_ISOBJECT(arr))) {
        /*
         * We have one special case, if (and only if) the input array is of
         * object DType and the dtype is not fixed already but parametric.
         * Then, we allow inspection of all elements, treating them as
         * elements. We do this recursively, so nested 0-D arrays can work,
         * but nested higher dimensional arrays will lead to an error.
         */
        assert(DType->type_num != NPY_OBJECT);  /* not parametric */

        PyArrayIterObject *iter;
        iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
        if (iter == NULL) {
            return -1;
        }
        while (iter->index < iter->size) {
            PyArray_DTypeMeta *item_DType;
            /*
             * Note: If the array contains typed objects we may need to use
             *       the dtype to use casting for finding the correct instance.
             */
            PyObject *elem = PyArray_GETITEM(arr, iter->dataptr);
            if (elem == NULL) {
                Py_DECREF(iter);
                return -1;
            }
            item_DType = discover_dtype_from_pyobject(elem, &flags, DType);
            if (item_DType == NULL) {
                Py_DECREF(iter);
                Py_DECREF(elem);
                return -1;
            }
            if (item_DType == (PyArray_DTypeMeta *)Py_None) {
                Py_SETREF(item_DType, NULL);
            }
            int flat_max_dims = 0;
            if (handle_scalar(elem, 0, &flat_max_dims, out_descr,
                    NULL, DType, &flags, item_DType) < 0) {
                Py_DECREF(iter);
                Py_DECREF(elem);
                Py_XDECREF(*out_descr);
                Py_XDECREF(item_DType);
                return -1;
            }
            Py_XDECREF(item_DType);
            Py_DECREF(elem);
            PyArray_ITER_NEXT(iter);
        }
        Py_DECREF(iter);
    }
    else if (NPY_UNLIKELY(DType->type_num == NPY_DATETIME) &&
                PyArray_ISSTRING(arr)) {
        /*
         * TODO: This branch should be deprecated IMO, the workaround is
         *       to cast to the object to a string array. Although a specific
         *       function (if there is even any need) would be better.
         *       This is value based casting!
         * Unless of course we actually want to support this kind of thing
         * in general (not just for object dtype)...
         */
        PyArray_DatetimeMetaData meta;
        meta.base = NPY_FR_GENERIC;
        meta.num = 1;

        if (find_string_array_datetime64_type(arr, &meta) < 0) {
            return -1;
        }
        else {
            *out_descr = create_datetime_dtype(NPY_DATETIME, &meta);
            if (*out_descr == NULL) {
                return -1;
            }
        }
    }
    else {
        /*
         * If this is not an object array figure out the dtype cast,
         * or simply use the returned DType.
         */
        *out_descr = PyArray_CastDescrToDType(PyArray_DESCR(arr), DType);
        if (*out_descr == NULL) {
            return -1;
        }
    }
    return 0;
}

/**
 * Return the correct descriptor given an array object and a DType class.
 *
 * This is identical to casting the arrays descriptor/dtype to the new
 * DType class
 *
 * @param arr The array object.
 * @param DType The DType class to cast to (or NULL for convenience)
 * @param out_descr The output descriptor will set. The result can be NULL
 *        when the array is of object dtype and has no elements.
 *
 * @return -1 on failure, 0 on success.
 */
static int
h_find_descriptor_from_array(HPyContext *ctx, HPy h_arr, HPy h_DType, HPy *h_out_descr)
{
    *h_out_descr = HPy_NULL;

    if (HPy_IsNull(h_DType)) {
        *h_out_descr = HPyArray_GetDescr(ctx, h_arr);
        return 0;
    }

    PyArrayObject *arr = (PyArrayObject *)HPy_AsPyObject(ctx, h_arr);
    PyArray_DTypeMeta *DType = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_DType);
    PyArray_Descr *out_descr = NULL;
    int ret = find_descriptor_from_array(arr, DType, &out_descr);
    *h_out_descr = HPy_FromPyObject(ctx, (PyObject *)out_descr);
    return ret;
}

/**
 * Given a dtype or DType object, find the correct descriptor to cast the
 * array to.
 *
 * This function is identical to normal casting using only the dtype, however,
 * it supports inspecting the elements when the array has object dtype
 * (and the given datatype describes a parametric DType class).
 *
 * @param arr
 * @param dtype A dtype instance or class.
 * @return A concrete dtype instance or NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptDescriptorToArray(PyArrayObject *arr, PyObject *dtype)
{
    /* If the requested dtype is flexible, adapt it */
    PyArray_Descr *new_dtype;
    PyArray_DTypeMeta *new_DType;
    int res;

    res = PyArray_ExtractDTypeAndDescriptor((PyObject *)dtype,
            &new_dtype, &new_DType);
    if (res < 0) {
        return NULL;
    }
    if (new_dtype == NULL) {
        res = find_descriptor_from_array(arr, new_DType, &new_dtype);
        if (res < 0) {
            Py_DECREF(new_DType);
            return NULL;
        }
        if (new_dtype == NULL) {
            /* This is an object array but contained no elements, use default */
            new_dtype = NPY_DT_CALL_default_descr(new_DType);
        }
    }
    Py_DECREF(new_DType);
    return new_dtype;
}

/**
 * Given a dtype or DType object, find the correct descriptor to cast the
 * array to.
 *
 * This function is identical to normal casting using only the dtype, however,
 * it supports inspecting the elements when the array has object dtype
 * (and the given datatype describes a parametric DType class).
 *
 * @param arr
 * @param dtype A dtype instance or class.
 * @return A concrete dtype instance or HPy_NULL
 */
NPY_NO_EXPORT HPy
HPyArray_AdaptDescriptorToArray(HPyContext *ctx, HPy arr, HPy dtype)
{
    /* If the requested dtype is flexible, adapt it */
    HPy new_dtype; /* PyArray_Descr *new_dtype */
    HPy new_DType; /* PyArray_DTypeMeta *new_DType */
    int res;

    res = HPyArray_ExtractDTypeAndDescriptor(ctx, dtype,
            &new_dtype, &new_DType);
    if (res < 0) {
        return HPy_NULL;
    }
    if (HPy_IsNull(new_dtype)) {
        res = h_find_descriptor_from_array(ctx, arr, new_DType, &new_dtype);
        if (res < 0) {
            HPy_Close(ctx, new_DType);
            return HPy_NULL;
        }
        if (HPy_IsNull(new_dtype)) {
            /* This is an object array but contained no elements, use default */
            PyArray_DTypeMeta *new_DType_data = PyArray_DTypeMeta_AsStruct(ctx, new_DType);
            new_DType = HNPY_DT_CALL_default_descr(ctx, new_DType, new_DType_data);
        }
    }
    HPy_Close(ctx, new_DType);
    return new_dtype;
}


/**
 * Recursion helper for `PyArray_DiscoverDTypeAndShape`.  See its
 * documentation for additional details.
 *
 * @param obj The current (possibly nested) object
 * @param curr_dims The current depth, i.e. initially 0 and increasing.
 * @param max_dims Maximum number of dimensions, modified during discovery.
 * @param out_descr dtype instance (or NULL) to promoted and update.
 * @param out_shape The current shape (updated)
 * @param coercion_cache_tail_ptr The tail of the linked list of coercion
 *        cache objects, which hold on to converted sequences and arrays.
 *        This is a pointer to the `->next` slot of the previous cache so
 *        that we can append a new cache object (and update this pointer).
 *        (Initially it is a pointer to the user-provided head pointer).
 * @param fixed_DType User provided fixed DType class
 * @param flags Discovery flags (reporting and behaviour flags, see def.)
 * @param never_copy Specifies if a copy is allowed during array creation.
 * @return The updated number of maximum dimensions (i.e. scalars will set
 *         this to the current dimensions).
 */
NPY_NO_EXPORT int
HPyArray_DiscoverDTypeAndShape_Recursive(
        HPyContext *ctx,
        HPy obj, int curr_dims, int max_dims, HPy /* (PyArray_Descr**) */ *out_descr,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj ***coercion_cache_tail_ptr,
        HPy /* (PyArray_DTypeMeta *) */ fixed_DType, enum _dtype_discovery_flags *flags,
        int never_copy)
{
    HPy arr = HPy_NULL; /* (PyArrayObject *) */
    HPy seq;
    PyObject *py_obj;
    PyObject *py_seq;
    /*
     * The first step is to find the DType class if it was not provided,
     * alternatively we have to find out that this is not a scalar at all
     * (which could fail and lead us to `object` dtype).
     */
    HPy DType = HPy_NULL; /* (PyArray_DTypeMeta *) */

    if (NPY_UNLIKELY(*flags & DISCOVER_STRINGS_AS_SEQUENCES)) {
        /*
         * We currently support that bytes/strings are considered sequences,
         * if the dtype is np.dtype('c'), this should be deprecated probably,
         * but requires hacks right now.
         */
        if (HPyBytes_Check(ctx, obj) && HPy_Length(ctx, obj) != 1) {
            goto force_sequence_due_to_char_dtype;
        }
        else if (HPyUnicode_Check(ctx, obj) && HPy_Length(ctx, obj) != 1) {
            goto force_sequence_due_to_char_dtype;
        }
    }

    /* If this is a known scalar, find the corresponding DType class */
    DType = hdiscover_dtype_from_pyobject(ctx, obj, flags, fixed_DType);
    if (HPy_IsNull(DType)) {
        return -1;
    }
    else if (HPy_Is(ctx, DType, ctx->h_None)) {
        HPy_Close(ctx, DType);
    }
    else {
        max_dims = hpy_handle_scalar(ctx,
                obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                flags, DType);
        HPy_Close(ctx, DType);
        return max_dims;
    }

    /*
     * At this point we expect to find either a sequence, or an array-like.
     * Although it is still possible that this fails and we have to use
     * `object`.
     */
    if (HPyArray_Check(ctx, obj)) {
        arr = HPy_Dup(ctx, obj);
    }
    else {
        HPy requested_descr = HPy_NULL;
        if (*flags & DESCRIPTOR_WAS_SET) {
            /* __array__ may be passed the requested descriptor if provided */
            requested_descr = *out_descr;
        }
        arr = _hpy_array_from_array_like(ctx, obj,
                requested_descr, 0, HPy_NULL, never_copy);
        if (HPy_IsNull(arr)) {
            return -1;
        }
        else if (HPy_Is(ctx, arr, ctx->h_NotImplemented)) {
            HPy_Close(ctx, arr);
            arr = HPy_NULL;
        }
        else if (curr_dims > 0 && curr_dims != max_dims) {
            PyArrayObject *arr_data = PyArrayObject_AsStruct(ctx, arr);
            /*
             * Deprecated 2020-12-09, NumPy 1.20
             *
             * See https://github.com/numpy/numpy/issues/17965
             * Shapely had objects which are not sequences but did export
             * the array-interface (and so are arguably array-like).
             * Previously numpy would not use array-like information during
             * shape discovery, so that it ended up acting as if this was
             * an (unknown) scalar but with the specified dtype.
             * Thus we ignore "scalars" here, as the value stored in the
             * array should be acceptable.
             */
            if (PyArray_NDIM(arr_data) > 0 && NPY_UNLIKELY(!HPySequence_Check(ctx, obj))) {
                // TODO HPY LABS PORT: PyErr_WarnFormat
                if (HPyErr_WarnEx(ctx, ctx->h_FutureWarning,
                        "The input object of type '%s' is an array-like "
                        "implementing one of the corresponding protocols "
                        "(`__array__`, `__array_interface__` or "
                        "`__array_struct__`); but not a sequence (or 0-D). "
                        "In the future, this object will be coerced as if it "
                        "was first converted using `np.array(obj)`. "
                        "To retain the old behaviour, you have to either "
                        "modify the type '%s', or assign to an empty array "
                        "created with `np.empty(correct_shape, dtype=object)`.", 1) < 0) {
                        // Py_TYPE(obj)->tp_name, Py_TYPE(obj)->tp_name) < 0) {
                    HPy_Close(ctx, arr);
                    return -1;
                }
                /*
                 * Strangely enough, even though we threw away the result here,
                 * we did use it during descriptor discovery, so promote it:
                 */
                if (update_shape(curr_dims, &max_dims, out_shape,
                        0, NULL, NPY_FALSE, flags) < 0) {
                    *flags |= FOUND_RAGGED_ARRAY;
                    HPy_Close(ctx, arr);
                    return max_dims;
                }
                if (!(*flags & DESCRIPTOR_WAS_SET)) {
                    HPy arr_descr = HPyArray_GetDescr(ctx, arr);
                    if (hpy_handle_promotion(ctx, out_descr, arr_descr, fixed_DType, flags) < 0) {
                        HPy_Close(ctx, arr);
                        return -1;
                    }
                }
                HPy_Close(ctx, arr);
                return max_dims;
            }
        }
    }
    if (!HPy_IsNull(arr)) {
        /*
         * This is an array object which will be added to the cache, keeps
         * the reference to the array alive (takes ownership).
         */
        if (hnpy_new_coercion_cache(ctx, obj, arr,
                0, coercion_cache_tail_ptr, curr_dims) < 0) {
            return -1;
        }

        PyArrayObject *arr_data = PyArrayObject_AsStruct(ctx, arr);
        if (curr_dims == 0) {
            /*
             * Special case for reverse broadcasting, ignore max_dims if this
             * is a single array-like object; needed for PyArray_CopyObject.
             */
            memcpy(out_shape, PyArray_SHAPE(arr_data),
                   PyArray_NDIM(arr_data) * sizeof(npy_intp));
            max_dims = PyArray_NDIM(arr_data);
        }
        else if (update_shape(curr_dims, &max_dims, out_shape,
                PyArray_NDIM(arr_data), PyArray_SHAPE(arr_data), NPY_FALSE, flags) < 0) {
            *flags |= FOUND_RAGGED_ARRAY;
            return max_dims;
        }

        if (*flags & DESCRIPTOR_WAS_SET) {
            return max_dims;
        }
        /*
         * For arrays we may not just need to cast the dtype to the user
         * provided fixed_DType. If this is an object array, the elements
         * may need to be inspected individually.
         * Note, this finds the descriptor of the array first and only then
         * promotes here (different associativity).
         */
        HPy cast_descr; /* (PyArray_Descr *) */
        if (h_find_descriptor_from_array(ctx, arr, fixed_DType, &cast_descr) < 0) {
            return -1;
        }
        if (HPy_IsNull(cast_descr)) {
            /* object array with no elements, no need to promote/adjust. */
            return max_dims;
        }
        if (hpy_handle_promotion(ctx, out_descr, cast_descr, fixed_DType, flags) < 0) {
            HPy_Close(ctx, cast_descr);
            return -1;
        }
        HPy_Close(ctx, cast_descr);
        return max_dims;
    }

    /*
     * The last step is to assume the input should be handled as a sequence
     * and to handle it recursively. That is, unless we have hit the
     * dimension limit.
     */
    npy_bool is_sequence = HPySequence_Check(ctx, obj);
    if (is_sequence) {
        is_sequence = HPy_Length(ctx, obj) >= 0;
        if (NPY_UNLIKELY(!is_sequence)) {
            /* NOTE: This should likely just raise all errors */
            if (HPyErr_ExceptionMatches(ctx, ctx->h_RecursionError) ||
                    HPyErr_ExceptionMatches(ctx, ctx->h_MemoryError)) {
                /*
                 * Consider these unrecoverable errors, continuing execution
                 * might crash the interpreter.
                 */
                return -1;
            }
            HPyErr_Clear(ctx);
        }
    }
    if (NPY_UNLIKELY(*flags & DISCOVER_TUPLES_AS_ELEMENTS) &&
            HPyTuple_Check(ctx, obj)) {
        is_sequence = NPY_FALSE;
    }
    if (curr_dims == max_dims || !is_sequence) {
        /* Clear any PySequence_Size error which would corrupts further calls */
        max_dims = hpy_handle_scalar(ctx,
                obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                flags, HPy_NULL);
        if (is_sequence) {
            /* Flag as ragged or too deep array */
            *flags |= FOUND_RAGGED_ARRAY;
        }
        return max_dims;
    }
    /* If we stop supporting bytes/str subclasses, more may be required here: */
    assert(!HPyBytes_Check(ctx, obj) && !HPyUnicode_Check(ctx, obj));

  force_sequence_due_to_char_dtype:

    // TODO HPY LABS PORT: PySequence_Fast
    /* Ensure we have a sequence (required for PyPy) */
    py_obj = HPy_AsPyObject(ctx, obj);
    py_seq = PySequence_Fast(py_obj, "Could not convert object to sequence");
    if (py_seq == NULL) {
        /*
            * Specifically do not fail on things that look like a dictionary,
            * instead treat them as scalar.
            */
        if (HPyErr_ExceptionMatches(ctx, ctx->h_KeyError)) {
            HPyErr_Clear(ctx);
            max_dims = hpy_handle_scalar(ctx,
                    obj, curr_dims, &max_dims, out_descr, out_shape, fixed_DType,
                    flags, HPy_NULL);
            return max_dims;
        }
        return -1;
    } else {
        seq = HPy_FromPyObject(ctx, py_seq);
        Py_DECREF(py_obj);
        Py_DECREF(py_seq);
    }
    seq = obj;
    if (hnpy_new_coercion_cache(ctx, obj, seq, 1, coercion_cache_tail_ptr, curr_dims) < 0) {
        return -1;
    }

    npy_intp size;
    HPy *seq_arr = NULL;
    if (!HPySequence_Check(ctx, seq)) { // PySequence_Fast
        npy_intp size = PySequence_Fast_GET_SIZE(py_seq);
        PyObject **objects = PySequence_Fast_ITEMS(py_seq);
        seq_arr = HPy_FromPyObjectArray(ctx, objects, size);
    } else {
        size = HPy_Length(ctx, seq);
    }

    if (update_shape(curr_dims, &max_dims,
                     out_shape, 1, &size, NPY_TRUE, flags) < 0) {
        /* But do update, if there this is a ragged case */
        *flags |= FOUND_RAGGED_ARRAY;
        return max_dims;
    }
    if (size == 0) {
        /* If the sequence is empty, this must be the last dimension */
        *flags |= MAX_DIMS_WAS_REACHED;
        return curr_dims + 1;
    }

    // TODO HPY LABS PORT: PyErr_CheckSignals
    /* Allow keyboard interrupts. See gh issue 18117. */
    // if (PyErr_CheckSignals() < 0) {
    //     return -1;
    // }

    /* Recursive call for each sequence item */
    for (HPy_ssize_t i = 0; i < size; i++) {
        HPy item;
        if (seq_arr == NULL) {
            item = HPy_GetItem_i(ctx, seq, i);
        } else {
            item = seq_arr[i];
        }
        if (HPy_IsNull(item)) {
            return -1;
        }
        max_dims = HPyArray_DiscoverDTypeAndShape_Recursive(ctx,
                item, curr_dims + 1, max_dims,
                out_descr, out_shape, coercion_cache_tail_ptr, fixed_DType,
                flags, never_copy);
        HPy_Close(ctx, item);

        if (max_dims < 0) {
            return -1;
        }
    }
    return max_dims;
}

NPY_NO_EXPORT int
PyArray_DiscoverDTypeAndShape(
        PyObject *obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *fixed_DType, PyArray_Descr *requested_descr,
        PyArray_Descr **out_descr, int never_copy)
{
    HPyContext *ctx = npy_get_context();
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPy h_fixed_DType = HPy_FromPyObject(ctx, (PyObject *)fixed_DType);
    HPy h_requested_descr = HPy_FromPyObject(ctx, (PyObject *)requested_descr);
    HPy h_out_descr = HPy_FromPyObject(ctx, (PyObject *)*out_descr);

    int res = HPyArray_DiscoverDTypeAndShape(ctx, h_obj, max_dims, out_shape,
            coercion_cache, h_fixed_DType, h_requested_descr, &h_out_descr,
            never_copy);

    Py_XSETREF(*out_descr, (PyArray_Descr *)HPy_AsPyObject(ctx, h_out_descr));
    HPy_Close(ctx, h_out_descr);
    HPy_Close(ctx, h_requested_descr);
    HPy_Close(ctx, h_fixed_DType);
    HPy_Close(ctx, h_obj);
    return res;
}

/**
 * Finds the DType and shape of an arbitrary nested sequence. This is the
 * general purpose function to find the parameters of the array (but not
 * the array itself) as returned by `np.array()`
 *
 * Note: Before considering to make part of this public, we should consider
 *       whether things such as `out_descr != NULL` should be supported in
 *       a public API.
 *
 * @param obj Scalar or nested sequences.
 * @param max_dims Maximum number of dimensions (after this scalars are forced)
 * @param out_shape Will be filled with the output shape (more than the actual
 *        shape may be written).
 * @param coercion_cache NULL initialized reference to a cache pointer.
 *        May be set to the first coercion_cache, and has to be freed using
 *        npy_free_coercion_cache.
 *        This should be stored in a thread-safe manner (i.e. function static)
 *        and is designed to be consumed by `PyArray_AssignFromCache`.
 *        If not consumed, must be freed using `npy_free_coercion_cache`.
 * @param fixed_DType A user provided fixed DType class.
 * @param requested_descr A user provided fixed descriptor. This is always
 *        returned as the discovered descriptor, but currently only used
 *        for the ``__array__`` protocol.
 * @param out_descr Set to the discovered output descriptor. This may be
 *        non NULL but only when fixed_DType/requested_descr are not given.
 *        If non NULL, it is the first dtype being promoted and used if there
 *        are no elements.
 *        The result may be unchanged (remain NULL) when converting a
 *        sequence with no elements. In this case it is callers responsibility
 *        to choose a default.
 * @param never_copy Specifies that a copy is not allowed.
 * @return dimensions of the discovered object or -1 on error.
 *         WARNING: If (and only if) the output is a single array, the ndim
 *         returned _can_ exceed the maximum allowed number of dimensions.
 *         It might be nice to deprecate this? But it allows things such as
 *         `arr1d[...] = np.array([[1,2,3,4]])`
 */
NPY_NO_EXPORT int
HPyArray_DiscoverDTypeAndShape(
        HPyContext *ctx,
        HPy obj, int max_dims,
        npy_intp out_shape[NPY_MAXDIMS],
        coercion_cache_obj **coercion_cache,
        HPy /* (PyArray_DTypeMeta *) */ fixed_DType, HPy /* (PyArray_Descr *) */ requested_descr,
        HPy /* (PyArray_Descr **) */ *out_descr, int never_copy)
{
    coercion_cache_obj **coercion_cache_head = coercion_cache;
    *coercion_cache = NULL;
    enum _dtype_discovery_flags flags = 0;

    /*
     * Support a passed in descriptor (but only if nothing was specified).
     */
    assert(HPy_IsNull(*out_descr) || HPy_IsNull(fixed_DType));
    /* Validate input of requested descriptor and DType */
    if (!HPy_IsNull(fixed_DType)) {
        assert(HPyGlobal_TypeCheck(ctx,
                fixed_DType, HPyArrayDTypeMeta_Type));
    }

    if (!HPy_IsNull(requested_descr)) {
        assert(HPy_Is(ctx, fixed_DType, HNPY_DTYPE(ctx, requested_descr)));
        /* The output descriptor must be the input. */
        *out_descr = HPy_Dup(ctx, requested_descr);
        flags |= DESCRIPTOR_WAS_SET;
    }

    /*
     * Call the recursive function, the setup for this may need expanding
     * to handle caching better.
     */

    /* Legacy discovery flags */
    if (!HPy_IsNull(requested_descr)) {
        PyArray_Descr *requested_descr_data = PyArray_Descr_AsStruct(ctx, requested_descr);
        if (requested_descr_data->type_num == NPY_STRING &&
                requested_descr_data->type == 'c') {
            /* Character dtype variation of string (should be deprecated...) */
            flags |= DISCOVER_STRINGS_AS_SEQUENCES;
        }
        else if (requested_descr_data->type_num == NPY_VOID &&
                    (!HPyField_IsNull(requested_descr_data->names) || requested_descr_data->subarray))  {
            /* Void is a chimera, in that it may or may not be structured... */
            flags |= DISCOVER_TUPLES_AS_ELEMENTS;
        }
    }

    int ndim = HPyArray_DiscoverDTypeAndShape_Recursive(ctx,
            obj, 0, max_dims, out_descr, out_shape, &coercion_cache,
            fixed_DType, &flags, never_copy);
    if (ndim < 0) {
        goto fail;
    }

    if (NPY_UNLIKELY(flags & FOUND_RAGGED_ARRAY)) {
        /*
         * If max-dims was reached and the dimensions reduced, this is ragged.
         * Otherwise, we merely reached the maximum dimensions, which is
         * slightly different. This happens for example for `[1, [2, 3]]`
         * where the maximum dimensions is 1, but then a sequence found.
         *
         * In this case we need to inform the user and clean out the cache
         * since it may be too deep.
         */

        /* Handle reaching the maximum depth differently: */
        int too_deep = ndim == max_dims;

        if (HPy_IsNull(fixed_DType)) {
            /* This is discovered as object, but deprecated */
            // TODO LABS HPY PORT: should be HPyGlobal
            static HPy visibleDeprecationWarning = HPy_NULL;
            npy_hpy_cache_import(ctx,
                    "numpy", "VisibleDeprecationWarning",
                    &visibleDeprecationWarning);
            if (HPy_IsNull(visibleDeprecationWarning)) {
                goto fail;
            }
            if (!too_deep) {
                /* NumPy 1.19, 2019-11-01 */
                if (HPyErr_WarnEx(ctx, visibleDeprecationWarning,
                        "Creating an ndarray from ragged nested sequences (which "
                        "is a list-or-tuple of lists-or-tuples-or ndarrays with "
                        "different lengths or shapes) is deprecated. If you "
                        "meant to do this, you must specify 'dtype=object' "
                        "when creating the ndarray.", 1) < 0) {
                    goto fail;
                }
            }
            else {
                /* NumPy 1.20, 2020-05-08 */
                /* Note, max_dims should normally always be NPY_MAXDIMS here */
                if (HPyErr_WarnEx(ctx, visibleDeprecationWarning,
                        "Creating an ndarray from nested sequences exceeding "
                        "the maximum number of dimensions of %d is deprecated. "
                        "If you mean to do this, you must specify "
                        "'dtype=object' when creating the ndarray.", 1) < 0) {
                    goto fail;
                }
            }
            /* Ensure that ragged arrays always return object dtype */
            HPy_SETREF(ctx, *out_descr, HPyArray_DescrFromType(ctx, NPY_OBJECT));
        }
        else if (PyArray_DTypeMeta_AsStruct(ctx, fixed_DType)->type_num != NPY_OBJECT) {
            /* Only object DType supports ragged cases unify error */

            /*
             * We used to let certain ragged arrays pass if they also
             * support e.g. conversion using `float(arr)`, which currently
             * works for arrays with only one element.
             * Thus we catch at least most of such cases here and give a
             * DeprecationWarning instead of an error.
             * Note that some of these will actually error later on when
             * attempting to do the actual assign.
             */
            int deprecate_single_element_ragged = 0;
            coercion_cache_obj *current = *coercion_cache_head;
            while (current != NULL) {
                if (current->sequence) {
                    if (current->depth == ndim) {
                        /*
                         * Assume that only array-likes will allow the deprecated
                         * behaviour
                         */
                        deprecate_single_element_ragged = 0;
                        break;
                    }
                    /* check next converted sequence/array-like */
                    current = current->next;
                    continue;
                }
                //PyArrayObject *arr = (PyArrayObject *)HPy_AsPyObject(npy_get_context(), current->arr_or_sequence);
                HPy arr = current->arr_or_sequence;
                PyArrayObject *arr_data = PyArrayObject_AsStruct(ctx, current->arr_or_sequence);
                assert(PyArray_NDIM(arr_data) + current->depth >= ndim);
                if (PyArray_NDIM(arr_data) != ndim - current->depth) {
                    /* This array is not compatible with the final shape */
                    if (HPyArray_SIZE(arr_data) != 1) {
                        HPy_Close(ctx, arr);
                        deprecate_single_element_ragged = 0;
                        break;
                    } else {
                        HPy_Close(ctx, arr);
                    }
                    deprecate_single_element_ragged = 1;
                }
                HPy_Close(ctx, arr);
                current = current->next;
            }

            if (deprecate_single_element_ragged) {
                /* Deprecated 2020-07-24, NumPy 1.20 */
                if (HPY_DEPRECATE(ctx,
                        "setting an array element with a sequence. "
                        "This was supported in some cases where the elements "
                        "are arrays with a single element. For example "
                        "`np.array([1, np.array([2])], dtype=int)`. "
                        "In the future this will raise the same ValueError as "
                        "`np.array([1, [2]], dtype=int)`.") < 0) {
                    goto fail;
                }
            }
            else if (!too_deep) {
                HPyErr_Format_p(ctx, ctx->h_ValueError,
                        "setting an array element with a sequence. The "
                        "requested array has an inhomogeneous shape after "
                        "%d dimensions. The detected shape was "
                        "?? + inhomogeneous part.",
                        ndim);
                // TODO HPY LABS PORT: PyErr_Format
                // PyObject *shape = PyArray_IntTupleFromIntp(ndim, out_shape);
                // PyErr_Format(PyExc_ValueError,
                //        "setting an array element with a sequence. The "
                //        "requested array has an inhomogeneous shape after "
                //        "%d dimensions. The detected shape was "
                //        "%R + inhomogeneous part.",
                //        ndim, shape);
                // Py_DECREF(shape);
                goto fail;
            }
            else {
                HPyErr_Format_p(ctx, ctx->h_ValueError,
                        "setting an array element with a sequence. The "
                        "requested array would exceed the maximum number of "
                        "dimension of %d.",
                        max_dims);
                goto fail;
            }
        }

        /*
         * If the array is ragged, the cache may be too deep, so clean it.
         * The cache is left at the same depth as the array though.
         */
        coercion_cache_obj **next_ptr = coercion_cache_head;
        coercion_cache_obj *current = *coercion_cache_head;  /* item to check */
        while (current != NULL) {
            if (current->depth > ndim) {
                /* delete "next" cache item and advanced it (unlike later) */
                current = hnpy_unlink_coercion_cache(ctx, current);
                continue;
            }
            /* advance both prev and next, and set prev->next to new item */
            *next_ptr = current;
            next_ptr = &(current->next);
            current = current->next;
        }
        *next_ptr = NULL;
    }
    /* We could check here for max-ndims being reached as well */

    if (!HPy_IsNull(requested_descr)) {
        /* descriptor was provided, we did not accidentally change it */
        assert(HPy_Is(ctx, *out_descr, requested_descr));
    }
    else if (NPY_UNLIKELY(HPy_IsNull(*out_descr))) {
        /*
         * When the object contained no elements (sequence of length zero),
         * the no descriptor may have been found. When a DType was requested
         * we use it to define the output dtype.
         * Otherwise, out_descr will remain NULL and the caller has to set
         * the correct default.
         */
        if (!HPy_IsNull(fixed_DType)) {
            PyArray_DTypeMeta *fixed_DType_data = PyArray_DTypeMeta_AsStruct(ctx, fixed_DType);
            *out_descr = HNPY_DT_CALL_default_descr(ctx, fixed_DType, fixed_DType_data);
            if (HPy_IsNull(*out_descr)) {
                goto fail;
            }
        }
    }
    return ndim;

  fail:
    npy_free_coercion_cache(*coercion_cache_head);
    *coercion_cache_head = NULL;
    HPy_SETREF(ctx, *out_descr, HPy_NULL);
    return -1;
}



/**
 * Check the descriptor is a legacy "flexible" DType instance, this is
 * an instance which is (normally) not attached to an array, such as a string
 * of length 0 or a datetime with no unit.
 * These should be largely deprecated, and represent only the DType class
 * for most `dtype` parameters.
 *
 * TODO: This function should eventually receive a deprecation warning and
 *       be removed.
 *
 * @param descr
 * @return 1 if this is not a concrete dtype instance 0 otherwise
 */
static int
h_descr_is_legacy_parametric_instance(HPyContext *ctx, HPy descr)
{
    PyArray_Descr *descr_data = PyArray_Descr_AsStruct(ctx, descr);
    if (PyDataType_ISUNSIZED(descr_data)) {
        return 1;
    }
    /* Flexible descr with generic time unit (which can be adapted) */
    if (PyDataType_ISDATETIME(descr_data)) {
        PyArray_DatetimeMetaData *meta;
        meta = h_get_datetime_metadata_from_dtype(ctx, descr_data);
        if (meta->base == NPY_FR_GENERIC) {
            return 1;
        }
    }
    return 0;
}


/**
 * Given either a DType instance or class, (or legacy flexible instance),
 * ands sets output dtype instance and DType class. Both results may be
 * NULL, but if `out_descr` is set `out_DType` will always be the
 * corresponding class.
 *
 * @param dtype
 * @param out_descr
 * @param out_DType
 * @return 0 on success -1 on failure
 */
NPY_NO_EXPORT int
PyArray_ExtractDTypeAndDescriptor(PyObject *dtype,
        PyArray_Descr **out_descr, PyArray_DTypeMeta **out_DType)
{
    HPyContext *ctx = npy_get_context();
    HPy h_out_DType, h_out_descr;
    HPy h_dtype;
    int res = 0;

    *out_DType = NULL;
    *out_descr = NULL;

    if (dtype != NULL) {
        h_dtype = HPy_FromPyObject(ctx, dtype);

        res = HPyArray_ExtractDTypeAndDescriptor(ctx, h_dtype, &h_out_descr, &h_out_DType);

        *out_descr = (PyArray_Descr *)HPy_AsPyObject(ctx, h_out_descr);
        *out_DType = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_out_DType);

        HPy_Close(ctx, h_dtype);
        HPy_Close(ctx, h_out_descr);
        HPy_Close(ctx, h_out_DType);
    }
    return res;
}

NPY_NO_EXPORT int
HPyArray_ExtractDTypeAndDescriptor(HPyContext *ctx, HPy dtype,
        HPy *out_descr, HPy *out_DType)
{
    *out_DType = HPy_NULL;
    *out_descr = HPy_NULL;
    int res = 0;

    if (!HPy_IsNull(dtype)) {
        HPy h_PyArrayDTypeMeta_Type = HPyGlobal_Load(ctx, HPyArrayDTypeMeta_Type);
        if (HPy_TypeCheck(ctx, dtype, h_PyArrayDTypeMeta_Type)) {
            assert(!HPyGlobal_Is(ctx, dtype, HPyArrayDescr_Type));  /* not np.dtype */
            *out_DType = HPy_Dup(ctx, dtype);
        }
        else if (HPy_TypeCheck(ctx, HPy_Type(ctx, dtype),
                    h_PyArrayDTypeMeta_Type)) {
            *out_DType = HNPY_DTYPE(ctx, dtype);
            if (!h_descr_is_legacy_parametric_instance(ctx, dtype)) {
                *out_descr = HPy_Dup(ctx, dtype);
            }
        }
        else {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "dtype parameter must be a DType instance or class.");
            res = -1;
        }
        HPy_Close(ctx, h_PyArrayDTypeMeta_Type);
    }
    return res;
}


/*
 * Python API function to expose the dtype+shape discovery functionality
 * directly.
 */
NPY_NO_EXPORT PyObject *
_discover_array_parameters(PyObject *NPY_UNUSED(self),
                           PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"obj", "dtype", NULL};

    PyObject *obj;
    PyObject *dtype = NULL;
    PyArray_Descr *fixed_descriptor = NULL;
    PyArray_DTypeMeta *fixed_DType = NULL;
    npy_intp shape[NPY_MAXDIMS];

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|O:_discover_array_parameters", kwlist,
            &obj, &dtype)) {
        return NULL;
    }

    if (PyArray_ExtractDTypeAndDescriptor(dtype,
            &fixed_descriptor, &fixed_DType) < 0) {
        return NULL;
    }

    coercion_cache_obj *coercion_cache = NULL;
    PyObject *out_dtype = NULL;
    int ndim = PyArray_DiscoverDTypeAndShape(
            obj, NPY_MAXDIMS, shape,
            &coercion_cache,
            fixed_DType, fixed_descriptor, (PyArray_Descr **)&out_dtype, 0);
    Py_XDECREF(fixed_DType);
    Py_XDECREF(fixed_descriptor);
    if (ndim < 0) {
        return NULL;
    }
    npy_free_coercion_cache(coercion_cache);
    if (out_dtype == NULL) {
        /* Empty sequence, report this as None. */
        out_dtype = Py_None;
        Py_INCREF(Py_None);
    }

    PyObject *shape_tuple = PyArray_IntTupleFromIntp(ndim, shape);
    if (shape_tuple == NULL) {
        return NULL;
    }

    PyObject *res = PyTuple_Pack(2, (PyObject *)out_dtype, shape_tuple);
    Py_DECREF(out_dtype);
    Py_DECREF(shape_tuple);
    return res;
}

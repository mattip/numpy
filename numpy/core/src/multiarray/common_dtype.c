#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/npy_common.h"
#include "numpy/arrayobject.h"

#include "common_dtype.h"
#include "dtypemeta.h"
#include "abstractdtypes.h"

#include "hpy_utils.h"

/*
 * This file defines all logic necessary for generic "common dtype"
 * operations.  This is unfortunately surprisingly complicated to get right
 * due to the value based logic NumPy uses and the fact that NumPy has
 * no clear (non-transitive) type promotion hierarchy.
 * Unlike most languages `int32 + float32 -> float64` instead of `float32`.
 * The other complicated thing is value-based-promotion, which means that
 * in many cases a Python 1, may end up as an `int8` or `uint8`.
 *
 * This file implements the necessary logic so that `np.result_type(...)`
 * can give the correct result for any order of inputs and can further
 * generalize to user DTypes.
 */


/**
 * This function defines the common DType operator.
 *
 * Note that the common DType will not be "object" (unless one of the dtypes
 * is object), even though object can technically represent all values
 * correctly.
 *
 * TODO: Before exposure, we should review the return value (e.g. no error
 *       when no common DType is found).
 *
 * @param dtype1 DType class to find the common type for.
 * @param dtype2 Second DType class.
 * @return The common DType or NULL with an error set
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2)
{
    if (dtype1 == dtype2) {
        Py_INCREF(dtype1);
        return dtype1;
    }

    PyArray_DTypeMeta *common_dtype;

    common_dtype = NPY_DT_CALL_common_dtype(dtype1, dtype2);
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(common_dtype);
        common_dtype = NPY_DT_CALL_common_dtype(dtype2, dtype1);
    }
    if (common_dtype == NULL) {
        return NULL;
    }
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(Py_NotImplemented);
        PyErr_Format(PyExc_TypeError,
                "The DTypes %S and %S do not have a common DType. "
                "For example they cannot be stored in a single array unless "
                "the dtype is `object`.", dtype1, dtype2);
        return NULL;
    }
    return common_dtype;
}


/**
 * This function takes a list of dtypes and "reduces" them (in a sense,
 * it finds the maximal dtype). Note that "maximum" here is defined by
 * knowledge (or category or domain). A user DType must always "know"
 * about all NumPy dtypes, floats "know" about integers, integers "know"
 * about unsigned integers.
 *
 *           c
 *          / \
 *         a   \    <-- The actual promote(a, b) may be c or unknown.
 *        / \   \
 *       a   b   c
 *
 * The reduction is done "pairwise". In the above `a.__common_dtype__(b)`
 * has a result (so `a` knows more) and `a.__common_dtype__(c)` returns
 * NotImplemented (so `c` knows more).  You may notice that the result
 * `res = a.__common_dtype__(b)` is not important.  We could try to use it
 * to remove the whole branch if `res is c` or by checking if
 * `c.__common_dtype(res) is c`.
 * Right now, we only clear initial elements in the most simple case where
 * `a.__common_dtype(b) is a` (and thus `b` cannot alter the end-result).
 * Clearing means, we do not have to worry about them later.
 *
 * There is one further subtlety. If we have an abstract DType and a
 * non-abstract one, we "prioritize" the non-abstract DType here.
 * In this sense "prioritizing" means that we use:
 *       abstract.__common_dtype__(other)
 * If both return NotImplemented (which is acceptable and even expected in
 * this case, see later) then `other` will be considered to know more.
 *
 * The reason why this may be acceptable for abstract DTypes, is that
 * the value-dependent abstract DTypes may provide default fall-backs.
 * The priority inversion effectively means that abstract DTypes are ordered
 * just below their concrete counterparts.
 * (This fall-back is convenient but not perfect, it can lead to
 * non-minimal promotions: e.g. `np.uint24 + 2**20 -> int32`. And such
 * cases may also be possible in some mixed type scenarios; they can be
 * avoided by defining the promotion explicitly in the user DType.)
 *
 * @param length Number of DTypes
 * @param dtypes
 */
static /* PyArray_DTypeMeta * */ HPy
hreduce_dtypes_to_most_knowledgeable(HPyContext *ctx,
        npy_intp length, /* PyArray_DTypeMeta ** */ HPy *dtypes)
{
    assert(length >= 2);
    npy_intp half = length / 2;

    HPy res = HPy_NULL;

    for (npy_intp low = 0; low < half; low++) {
        npy_intp high = length - 1 - low;
        if (HPy_Is(ctx, dtypes[high], dtypes[low])) {
            HPy_SETREF(ctx, res, HPy_Dup(ctx, dtypes[low]));
        }
        else {
            if (HNPY_DT_is_abstract(ctx, dtypes[high])) {
                /*
                 * Priority inversion, start with abstract, because if it
                 * returns `other`, we can let other pass instead.
                 */
                HPy tmp = dtypes[low];
                dtypes[low] = dtypes[high];
                dtypes[high] = tmp;
            }

            CAPI_WARN("hreduce_dtypes_to_most_knowledgeable: NPY_DT_CALL_common_dtype");
            PyArray_DTypeMeta *py_dtypes_high = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, dtypes[high]);
            PyArray_DTypeMeta *py_dtypes_low = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, dtypes[low]);
            PyArray_DTypeMeta *py_res = NPY_DT_CALL_common_dtype(py_dtypes_low, py_dtypes_high);
            res = HPy_FromPyObject(ctx, (PyObject *)py_res);
            Py_XDECREF(py_dtypes_high);
            Py_XDECREF(py_dtypes_low);
            Py_XDECREF(py_res);
            if (HPy_IsNull(res)) {
                return HPy_NULL;
            }
        }

        if (HPy_Is(ctx, res, ctx->h_NotImplemented)) {
            HPy tmp = dtypes[low];
            dtypes[low] = dtypes[high];
            dtypes[high] = tmp;
        }
        if (HPy_Is(ctx, res, dtypes[low])) {
            /* `dtypes[high]` cannot influence the final result, so clear: */
            dtypes[high] = HPy_NULL;
        }
    }

    if (length == 2) {
        return res;
    }
    HPy_Close(ctx, res);
    return hreduce_dtypes_to_most_knowledgeable(ctx, length - half, dtypes);
}


/**
 * Promotes a list of DTypes with each other in a way that should guarantee
 * stable results even when changing the order.
 *
 * In general this approach always works as long as the most generic dtype
 * is either strictly larger, or compatible with all other dtypes.
 * For example promoting float16 with any other float, integer, or unsigned
 * integer again gives a floating point number. And any floating point number
 * promotes in the "same way" as `float16`.
 * If a user inserts more than one type into the NumPy type hierarchy, this
 * can break. Given:
 *     uint24 + int32 -> int48  # Promotes to a *new* dtype!
 *
 * The following becomes problematic (order does not matter):
 *         uint24 +      int16  +           uint32  -> int64
 *    <==      (uint24 + int16) + (uint24 + uint32) -> int64
 *    <==                int32  +           uint32  -> int64
 *
 * It is impossible to achieve an `int48` result in the above.
 *
 * This is probably only resolvable by asking `uint24` to take over the
 * whole reduction step; which we currently do not do.
 * (It may be possible to notice the last up-cast and implement use something
 * like: `uint24.nextafter(int32).__common_dtype__(uint32)`, but that seems
 * even harder to grasp.)
 *
 * Note that a case where two dtypes are mixed (and know nothing about each
 * other) will always generate an error:
 *     uint24 + int48 + int64 -> Error
 *
 * Even though `int64` is a safe solution, since `uint24 + int64 -> int64` and
 * `int48 + int64 -> int64` and `int64` and there cannot be a smaller solution.
 *
 * //TODO: Maybe this function should allow not setting an error?
 *
 * @param length Number of dtypes (and values) must be at least 1
 * @param dtypes The concrete or abstract DTypes to promote
 * @return NULL or the promoted DType.
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_PromoteDTypeSequence(
        npy_intp length, PyArray_DTypeMeta **dtypes_in)
{
    HPyContext *ctx = npy_get_context();
    HPy *h_dtypes_in = HPy_FromPyObjectArray(ctx, (PyObject **) dtypes_in, (Py_ssize_t)length);
    HPy h_ret = HPyArray_PromoteDTypeSequence(ctx, length, h_dtypes_in);
    PyArray_DTypeMeta *ret = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_ret);
    HPy_Close(ctx, h_ret);
    HPy_CloseAndFreeArray(ctx, h_dtypes_in, (HPy_ssize_t)length);
    return ret;
}

NPY_NO_EXPORT HPy
HPyArray_PromoteDTypeSequence(HPyContext *ctx,
        npy_intp length, HPy *dtypes_in)
{
    if (length == 1) {
        return HPy_Dup(ctx, dtypes_in[0]);
    }
    HPy result = HPy_NULL;

    /* Copy dtypes so that we can reorder them (only allocate when many) */
    HPy *_scratch_stack[NPY_MAXARGS];
    HPy*_scratch_heap = NULL;
    HPy *dtypes = _scratch_stack;

    if (length > NPY_MAXARGS) {
        _scratch_heap = malloc(length * sizeof(HPy));
        if (_scratch_heap == NULL) {
            HPyErr_NoMemory(ctx);
            return HPy_NULL;
        }
        dtypes = _scratch_heap;
    }

    memcpy(dtypes, dtypes_in, length * sizeof(HPy));

    /*
     * `result` is the last promotion result, which can usually be reused if
     * it is not NotImplemneted.
     * The passed in dtypes are partially sorted (and cleared, when clearly
     * not relevant anymore).
     * `dtypes[0]` will be the most knowledgeable (highest category) which
     * we consider the "main_dtype" here.
     */
    result = hreduce_dtypes_to_most_knowledgeable(ctx, length, dtypes);
    if (HPy_IsNull(result)) {
        goto finish;
    }
    HPy main_dtype = dtypes[0];

    npy_intp reduce_start = 1;
    if (HPy_Is(ctx, result, ctx->h_NotImplemented)) {
        HPy_SETREF(ctx, result, HPy_NULL);
    }
    else {
        /* (new) first value is already taken care of in `result` */
        reduce_start = 2;
    }
    /*
     * At this point, we have only looked at every DType at most once.
     * The `main_dtype` must know all others (or it will be a failure) and
     * all dtypes returned by its `common_dtype` must be guaranteed to succeed
     * promotion with one another.
     * It is the job of the "main DType" to ensure that at this point order
     * is irrelevant.
     * If this turns out to be a limitation, this "reduction" will have to
     * become a default version and we have to allow DTypes to override it.
     */
    HPy prev = HPy_NULL;
    for (npy_intp i = reduce_start; i < length; i++) {
        if (HPy_IsNull(dtypes[i]) || HPy_Is(ctx, dtypes[i], prev)) {
            continue;
        }
        /*
         * "Promote" the current dtype with the main one (which should be
         * a higher category). We assume that the result is not in a lower
         * category.
         */
        CAPI_WARN("HPyArray_PromoteDTypeSequence: call to NPY_DT_CALL_common_dtype");
        PyArray_DTypeMeta *py_main_dtype = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, main_dtype);
        PyArray_DTypeMeta *py_dtypes_i = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, dtypes[i]);
        PyArray_DTypeMeta *promotion = NPY_DT_CALL_common_dtype(
                py_main_dtype, py_dtypes_i);
        HPy h_promotion = HPy_FromPyObject(ctx, (PyObject *)promotion);
        Py_DECREF(py_main_dtype);
        Py_DECREF(py_dtypes_i);

        if (HPy_IsNull(h_promotion)) {
            HPy_SETREF(ctx, result, HPy_NULL);
            goto finish;
        }
        else if (HPy_Is(ctx, h_promotion, ctx->h_NotImplemented)) {
            Py_DECREF(promotion);
            HPy_Close(ctx, h_promotion);
            HPy_SETREF(ctx, result, HPy_NULL);
            HPyTupleBuilder dtypes_in_tuple = HPyTupleBuilder_New(ctx, length);
            if (HPyTupleBuilder_IsNull(dtypes_in_tuple)) {
                goto finish;
            }
            for (HPy_ssize_t l=0; l < length; l++) {
                HPyTupleBuilder_Set(ctx, dtypes_in_tuple, l, HPy_Dup(ctx, dtypes_in[l]));
            }
            // HPy h_dtypes_in_tuple = HPyTupleBuilder_Build(ctx, dtypes_in_tuple);
            // HPY TODO: PyErr_Format(PyExc_TypeError,
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "The DType %S could not be promoted by %S. This means that "
                    "no common DType exists for the given inputs. "
                    "For example they cannot be stored in a single array unless "
                    "the dtype is `object`. The full list of DTypes is: %S"); //,
                    // dtypes[i], main_dtype, h_dtypes_in_tuple);
            HPyTupleBuilder_Cancel(ctx, dtypes_in_tuple);
            goto finish;
        }
        if (HPy_IsNull(result)) {
            result = h_promotion;
            continue;
        }

        /*
         * The above promoted, now "reduce" with the current result; note that
         * in the typical cases we expect this step to be a no-op.
         */
        CAPI_WARN("HPyArray_PromoteDTypeSequence: call to PyArray_CommonDType");
        PyArray_DTypeMeta *py_common = PyArray_CommonDType(HPy_AsPyObject(ctx, result), promotion);
        HPy_SETREF(ctx, result, HPy_FromPyObject(ctx, py_common));
        Py_DECREF(promotion);
        HPy_Close(ctx, h_promotion);
        if (HPy_IsNull(result)) {
            goto finish;
        }
    }

  finish:
    if (_scratch_heap) {
        free(_scratch_heap);
    }
    return result;
}

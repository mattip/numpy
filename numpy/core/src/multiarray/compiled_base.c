#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "numpy/npy_math.h"
#include "npy_config.h"
#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "lowlevel_strided_loops.h" /* for npy_bswap8 */
#include "alloc.h"
#include "ctors.h"
#include "common.h"
#include "simd/simd.h"

#include <string.h>

typedef enum {
    PACK_ORDER_LITTLE = 0,
    PACK_ORDER_BIG
} PACK_ORDER;

/*
 * Returns -1 if the array is monotonic decreasing,
 * +1 if the array is monotonic increasing,
 * and 0 if the array is not monotonic.
 */
static int
check_array_monotonic(const double *a, npy_intp lena)
{
    npy_intp i;
    double next;
    double last;

    if (lena == 0) {
        /* all bin edges hold the same value */
        return 1;
    }
    last = a[0];

    /* Skip repeated values at the beginning of the array */
    for (i = 1; (i < lena) && (a[i] == last); i++);

    if (i == lena) {
        /* all bin edges hold the same value */
        return 1;
    }

    next = a[i];
    if (last < next) {
        /* Possibly monotonic increasing */
        for (i += 1; i < lena; i++) {
            last = next;
            next = a[i];
            if (last > next) {
                return 0;
            }
        }
        return 1;
    }
    else {
        /* last > next, possibly monotonic decreasing */
        for (i += 1; i < lena; i++) {
            last = next;
            next = a[i];
            if (last < next) {
                return 0;
            }
        }
        return -1;
    }
}

/* Find the minimum and maximum of an integer array */
static void
minmax(const npy_intp *data, npy_intp data_len, npy_intp *mn, npy_intp *mx)
{
    npy_intp min = *data;
    npy_intp max = *data;

    while (--data_len) {
        const npy_intp val = *(++data);
        if (val < min) {
            min = val;
        }
        else if (val > max) {
            max = val;
        }
    }

    *mn = min;
    *mx = max;
}

/*
 * arr_bincount is registered as bincount.
 *
 * bincount accepts one, two or three arguments. The first is an array of
 * non-negative integers The second, if present, is an array of weights,
 * which must be promotable to double. Call these arguments list and
 * weight. Both must be one-dimensional with len(weight) == len(list). If
 * weight is not present then bincount(list)[i] is the number of occurrences
 * of i in list.  If weight is present then bincount(self,list, weight)[i]
 * is the sum of all weight[j] where list [j] == i.  Self is not used.
 * The third argument, if present, is a minimum length desired for the
 * output array.
 */
HPyDef_METH(arr_bincount, "bincount", HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
arr_bincount_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), const HPy *args, size_t nargs, HPy kwnames)
{
    HPy list = HPy_NULL, weight = ctx->h_None, mlength = HPy_NULL;
    HPy lst = HPy_NULL, ans = HPy_NULL, wts = HPy_NULL; // PyArrayObject *
    npy_intp *numbers, *ians, len, mx, mn, ans_size;
    npy_intp minlength = 0;
    npy_intp i;
    double *weights , *dans;
    static const char *kwlist[] = {"list", "weights", "minlength", NULL};

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwnames, "O|OO:bincount",
                kwlist, &list, &weight, &mlength)) {
        return HPy_NULL;
    }

    HPy intp_descr = HPyArray_DescrFromType(ctx, NPY_INTP);
    lst = HPyArray_ContiguousFromAny(ctx, list, intp_descr, 1, 1);
    HPy_Close(ctx, intp_descr);
    if (HPy_IsNull(lst)) {
        goto fail;
    }
    PyArrayObject *lst_struct = PyArrayObject_AsStruct(ctx, lst);
    len = PyArray_SIZE(lst_struct);

    /*
     * This if/else if can be removed by changing the argspec to O|On above,
     * once we retire the deprecation
     */
    if (HPy_Is(ctx, mlength, ctx->h_None)) {
        /* NumPy 1.14, 2017-06-01 */
        if (HPY_DEPRECATE(ctx, "0 should be passed as minlength instead of None; "
                      "this will error in future.") < 0) {
            goto fail;
        }
    }
    else if (!HPy_IsNull(mlength)) {
        minlength = HPyArray_PyIntAsIntp(ctx, mlength);
        if (hpy_error_converting(ctx, minlength)) {
            goto fail;
        }
    }

    if (minlength < 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "'minlength' must not be negative");
        goto fail;
    }

    /* handle empty list */
    if (len == 0) {
        intp_descr = HPyArray_DescrFromType(ctx, NPY_INTP);
        ans = HPyArray_ZEROS(ctx, 1, &minlength, intp_descr, 0);
        HPy_Close(ctx, intp_descr);
        if (HPy_IsNull(ans)){
            goto fail;
        }
        HPy_Close(ctx, lst);
        return ans;
    }

    numbers = (npy_intp *)PyArray_DATA(lst_struct);
    minmax(numbers, len, &mn, &mx);
    if (mn < 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "'list' argument must have no negative elements");
        goto fail;
    }
    ans_size = mx + 1;
    if (!HPy_Is(ctx, mlength, ctx->h_None)) {
        if (ans_size < minlength) {
            ans_size = minlength;
        }
    }
    if (HPy_Is(ctx, weight, ctx->h_None)) {
        HPy intp_descr = HPyArray_DescrFromType(ctx, NPY_INTP);
        ans = HPyArray_ZEROS(ctx, 1, &ans_size, intp_descr, 0);
        HPy_Close(ctx, intp_descr);
        if (HPy_IsNull(ans)) {
            goto fail;
        }
        ians = (npy_intp *)PyArray_DATA(PyArrayObject_AsStruct(ctx, ans));
        HPY_NPY_BEGIN_ALLOW_THREADS(ctx);
        for (i = 0; i < len; i++)
            ians[numbers[i]] += 1;
        HPY_NPY_END_ALLOW_THREADS(ctx);
        HPy_Close(ctx, lst);
    }
    else {
        HPy double_descr = HPyArray_DescrFromType(ctx, NPY_DOUBLE);
        wts = HPyArray_ContiguousFromAny(ctx,
                                                weight, double_descr, 1, 1);
        if (HPy_IsNull(wts)) {
            HPy_Close(ctx, double_descr);
            goto fail;
        }
        PyArrayObject *wts_struct = PyArrayObject_AsStruct(ctx, wts);
        weights = (double *)PyArray_DATA(wts_struct);
        if (PyArray_SIZE(wts_struct) != len) {
            HPy_Close(ctx, double_descr);
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "The weights and list don't have the same length.");
            goto fail;
        }

        ans = HPyArray_ZEROS(ctx, 1, &ans_size, double_descr, 0);
        HPy_Close(ctx, double_descr);
        if (HPy_IsNull(ans)) {
            goto fail;
        }
        dans = (double *)PyArray_DATA(PyArrayObject_AsStruct(ctx, ans));
        HPY_NPY_BEGIN_ALLOW_THREADS(ctx);
        for (i = 0; i < len; i++) {
            dans[numbers[i]] += weights[i];
        }
        HPY_NPY_END_ALLOW_THREADS(ctx);
        HPy_Close(ctx, lst);
        HPy_Close(ctx, wts);
    }
    HPyTracker_Close(ctx, ht);
    return ans;

fail:
    HPyTracker_Close(ctx, ht);
    HPy_Close(ctx, lst);
    HPy_Close(ctx, wts);
    HPy_Close(ctx, ans);
    return HPy_NULL;
}

/* Internal function to expose check_array_monotonic to python */
HPyDef_METH(_monotonicity, "_monotonicity", HPyFunc_KEYWORDS)
static HPy
_monotonicity_impl(HPyContext *ctx, HPy NPY_UNUSED(self), const HPy *args, size_t nargs, HPy kwnames)
{
    static const char *kwlist[] = {"x", NULL};
    HPy obj_x = HPy_NULL;
    HPy arr_x = HPy_NULL; // PyArrayObject *
    long monotonic;
    npy_intp len_x;
    HPY_NPY_BEGIN_THREADS_DEF(ctx);

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwnames, "O|_monotonicity", kwlist,
                                     &obj_x)) {
        return HPy_NULL;
    }

    /*
     * TODO:
     *  `x` could be strided, needs change to check_array_monotonic
     *  `x` is forced to double for this check
     */
    HPy type_descr = HPyArray_DescrFromType(ctx, NPY_DOUBLE);
    arr_x = HPyArray_FROMANY(ctx,
        obj_x, type_descr, 1, 1, NPY_ARRAY_CARRAY_RO);
    HPyTracker_Close(ctx, ht);
    HPy_Close(ctx, type_descr);
    if (!HPy_IsNull(arr_x)) {
        return HPy_NULL;
    }
    PyArrayObject *arr_x_struct = PyArrayObject_AsStruct(ctx, arr_x);

    len_x = HPyArray_SIZE(arr_x_struct);
    HPY_NPY_BEGIN_THREADS_THRESHOLDED(ctx, len_x)
    monotonic = check_array_monotonic(
        (const double *)PyArray_DATA(arr_x_struct), len_x);
    HPY_NPY_END_THREADS(ctx)
    HPy_Close(ctx, arr_x);

    return HPyLong_FromLong(ctx, monotonic);
}

/*
 * Returns input array with values inserted sequentially into places
 * indicated by the mask
 */
HPyDef_METH(_insert, "_insert", HPyFunc_KEYWORDS)
static HPy
_insert_impl(HPyContext *ctx, HPy NPY_UNUSED(self), const HPy *args, size_t nargs, HPy kwdict)
{
    char *src, *dest;
    npy_bool *mask_data;
    HPy dtype; // PyArray_Descr *
    HPyArray_CopySwapFunc *copyswap;
    HPy array0, mask0, values0;
    HPy array, mask, values; // PyArrayObject *
    npy_intp i, j, chunk, nm, ni, nv;

    static const char *kwlist[] = {"input", "mask", "vals", NULL};
    HPY_NPY_BEGIN_THREADS_DEF;
    values = mask = HPy_NULL;

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwdict, "OOO:place", kwlist,
                &array0, &mask0, &values0)) {
        return HPy_NULL;
    }

    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    HPy array0_type = HPy_Type(ctx, array0);
    if (!HPy_Is(ctx, array0_type, array_type)) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "..");
        // TODO
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    array = HPyArray_FromArray(ctx, array0, PyArrayObject_AsStruct(ctx, array0), HPy_NULL, NULL,
                                    NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (HPy_IsNull(array)) {
        goto fail;
    }
    PyArrayObject *array_struct = PyArrayObject_AsStruct(ctx, array);

    ni = PyArray_SIZE(array_struct);
    dest = PyArray_DATA(array_struct);
    HPy array_descr = HPyArray_DESCR(ctx, array, array_struct);
    PyArray_Descr *array_descr_struct = PyArray_Descr_AsStruct(ctx, array_descr);
    chunk = array_descr_struct->elsize;
    HPy type_descr = HPyArray_DescrFromType(ctx, NPY_BOOL);
    mask = HPyArray_FROM_OTF(ctx, mask0, type_descr,
                                NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST);
    if (HPy_IsNull(mask)) {
        goto fail;
    }
    PyArrayObject *mask_struct = PyArrayObject_AsStruct(ctx, mask);

    nm = PyArray_SIZE(mask_struct);
    if (nm != ni) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "place: mask and data must be "
                        "the same size");
        goto fail;
    }

    mask_data = PyArray_DATA(mask_struct);
    dtype = array_descr;
    // Py_INCREF(dtype);

    values = HPyArray_FromAny(ctx, values0, dtype,
                                    0, 0, NPY_ARRAY_CARRAY, HPy_NULL);
    if (HPy_IsNull(values)) {
        goto fail;
    }
    PyArrayObject *values_struct = PyArrayObject_AsStruct(ctx, values);

    nv = PyArray_SIZE(values_struct); /* zero if null array */
    if (nv <= 0) {
        npy_bool allFalse = 1;
        i = 0;

        while (allFalse && i < ni) {
            if (mask_data[i]) {
                allFalse = 0;
            } else {
                i++;
            }
        }
        if (!allFalse) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Cannot insert from an empty array!");
            goto fail;
        } else {
            HPy_Close(ctx, values);
            HPy_Close(ctx, mask);
            HPyArray_ResolveWritebackIfCopy(ctx, array);
            HPy_Close(ctx, array);
            HPyTracker_Close(ctx, ht);
            return HPy_Dup(ctx, ctx->h_None);
        }
    }

    src = PyArray_DATA(values_struct);
    j = 0;

    copyswap = array_descr_struct->f->copyswap;
    HPY_NPY_BEGIN_THREADS_DESCR(ctx, array_descr_struct);
    for (i = 0; i < ni; i++) {
        if (mask_data[i]) {
            if (j >= nv) {
                j = 0;
            }

            copyswap(ctx, dest + i*chunk, src + j*chunk, 0, array_descr);
            j++;
        }
    }
    HPY_NPY_END_THREADS(ctx);

    HPy_Close(ctx, values);
    HPy_Close(ctx, mask);
    HPyArray_ResolveWritebackIfCopy(ctx, array);
    HPy_Close(ctx, array);
    HPyTracker_Close(ctx, ht);
    return HPy_Dup(ctx, ctx->h_None);

 fail:
    HPy_Close(ctx, mask);
    HPyArray_ResolveWritebackIfCopy(ctx, array);
    HPy_Close(ctx, array);
    HPy_Close(ctx, values);
    HPyTracker_Close(ctx, ht);
    return HPy_NULL;
}

#define LIKELY_IN_CACHE_SIZE 8

#ifdef __INTEL_COMPILER
#pragma intel optimization_level 0
#endif
static NPY_INLINE npy_intp
_linear_search(const npy_double key, const npy_double *arr, const npy_intp len, const npy_intp i0)
{
    npy_intp i;

    for (i = i0; i < len && key >= arr[i]; i++);
    return i - 1;
}

/** @brief find index of a sorted array such that arr[i] <= key < arr[i + 1].
 *
 * If an starting index guess is in-range, the array values around this
 * index are first checked.  This allows for repeated calls for well-ordered
 * keys (a very common case) to use the previous index as a very good guess.
 *
 * If the guess value is not useful, bisection of the array is used to
 * find the index.  If there is no such index, the return values are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @param guess initial guess of index
 * @return index
 */
static npy_intp
binary_search_with_guess(const npy_double key, const npy_double *arr,
                         npy_intp len, npy_intp guess)
{
    npy_intp imin = 0;
    npy_intp imax = len;

    /* Handle keys outside of the arr range first */
    if (key > arr[len - 1]) {
        return len;
    }
    else if (key < arr[0]) {
        return -1;
    }

    /*
     * If len <= 4 use linear search.
     * From above we know key >= arr[0] when we start.
     */
    if (len <= 4) {
        return _linear_search(key, arr, len, 1);
    }

    if (guess > len - 3) {
        guess = len - 3;
    }
    if (guess < 1)  {
        guess = 1;
    }

    /* check most likely values: guess - 1, guess, guess + 1 */
    if (key < arr[guess]) {
        if (key < arr[guess - 1]) {
            imax = guess - 1;
            /* last attempt to restrict search to items in cache */
            if (guess > LIKELY_IN_CACHE_SIZE &&
                        key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
                imin = guess - LIKELY_IN_CACHE_SIZE;
            }
        }
        else {
            /* key >= arr[guess - 1] */
            return guess - 1;
        }
    }
    else {
        /* key >= arr[guess] */
        if (key < arr[guess + 1]) {
            return guess;
        }
        else {
            /* key >= arr[guess + 1] */
            if (key < arr[guess + 2]) {
                return guess + 1;
            }
            else {
                /* key >= arr[guess + 2] */
                imin = guess + 2;
                /* last attempt to restrict search to items in cache */
                if (guess < len - LIKELY_IN_CACHE_SIZE - 1 &&
                            key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
                    imax = guess + LIKELY_IN_CACHE_SIZE;
                }
            }
        }
    }

    /* finally, find index by bisection */
    while (imin < imax) {
        const npy_intp imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }
    return imin - 1;
}

#undef LIKELY_IN_CACHE_SIZE

HPyDef_METH(arr_interp, "interp", HPyFunc_KEYWORDS)
static HPy
arr_interp_impl(HPyContext *ctx, HPy NPY_UNUSED(self), const HPy *args, size_t nargs, HPy kwdict)
{

    HPy fp, xp, x;
    HPy left = HPy_NULL, right = HPy_NULL;
    HPy afp = HPy_NULL, axp = HPy_NULL, ax = HPy_NULL, af = HPy_NULL; // PyArrayObject *
    npy_intp i, lenx, lenxp;
    npy_double lval, rval;
    const npy_double *dy, *dx, *dz;
    npy_double *dres, *slopes = NULL;

    static const char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    HPY_NPY_BEGIN_THREADS_DEF;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwdict, "OOO|OO:interp", kwlist,
                                     &x, &xp, &fp, &left, &right)) {
        return HPy_NULL;
    }

    HPy double_descr = HPyArray_DescrFromType(ctx, NPY_DOUBLE);
    afp = HPyArray_ContiguousFromAny(ctx, fp, double_descr, 1, 1);
    if (HPy_IsNull(afp)) {
        HPy_Close(ctx, double_descr);
        return HPy_NULL;
    }
    axp = HPyArray_ContiguousFromAny(ctx, xp, double_descr, 1, 1);
    if (HPy_IsNull(axp)) {
        HPy_Close(ctx, double_descr);
        goto fail;
    }
    ax = HPyArray_ContiguousFromAny(ctx, x, double_descr, 0, 0);
    HPy_Close(ctx, double_descr);
    if (HPy_IsNull(ax)) {
        goto fail;
    }
    PyArrayObject *axp_struct = PyArrayObject_AsStruct(ctx, axp);
    lenxp = PyArray_SIZE(axp_struct);
    if (lenxp == 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    PyArrayObject *afp_struct = PyArrayObject_AsStruct(ctx, afp);
    if (PyArray_SIZE(afp_struct) != lenxp) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    PyArrayObject *ax_struct = PyArrayObject_AsStruct(ctx, ax);
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    af = HPyArray_SimpleNew(ctx, array_type, PyArray_NDIM(ax_struct),
                                            PyArray_DIMS(ax_struct), NPY_DOUBLE);
    HPy_Close(ctx, array_type);
    if (HPy_IsNull(af)) {
        goto fail;
    }
    lenx = PyArray_SIZE(ax_struct);

    dy = (const npy_double *)PyArray_DATA(afp_struct);
    dx = (const npy_double *)PyArray_DATA(axp_struct);
    dz = (const npy_double *)PyArray_DATA(ax_struct);
    dres = (npy_double *)PyArray_DATA(PyArrayObject_AsStruct(ctx, af));
    /* Get left and right fill values. */
    if (HPy_IsNull(left) || HPy_Is(ctx, left, ctx->h_None)) {
        lval = dy[0];
    }
    else {
        lval = HPyFloat_AsDouble(ctx, left);
        if (hpy_error_converting(ctx, lval)) {
            goto fail;
        }
    }
    if (HPy_IsNull(right) || HPy_Is(ctx, right, ctx->h_None)) {
        rval = dy[lenxp - 1];
    }
    else {
        rval = HPyFloat_AsDouble(ctx, right);
        if (hpy_error_converting(ctx, rval)) {
            goto fail;
        }
    }

    /* binary_search_with_guess needs at least a 3 item long array */
    if (lenxp == 1) {
        const npy_double xp_val = dx[0];
        const npy_double fp_val = dy[0];

        HPY_NPY_BEGIN_THREADS_THRESHOLDED(ctx, lenx);
        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];
            dres[i] = (x_val < xp_val) ? lval :
                                         ((x_val > xp_val) ? rval : fp_val);
        }
        HPY_NPY_END_THREADS(ctx);
    }
    else {
        npy_intp j = 0;

        /* only pre-calculate slopes if there are relatively few of them. */
        if (lenxp <= lenx) {
            slopes = PyArray_malloc((lenxp - 1) * sizeof(npy_double));
            if (slopes == NULL) {
                HPyErr_NoMemory(ctx);
                goto fail;
            }
        }

        HPY_NPY_BEGIN_THREADS(ctx);

        if (slopes != NULL) {
            for (i = 0; i < lenxp - 1; ++i) {
                slopes[i] = (dy[i+1] - dy[i]) / (dx[i+1] - dx[i]);
            }
        }

        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];

            if (npy_isnan(x_val)) {
                dres[i] = x_val;
                continue;
            }

            j = binary_search_with_guess(x_val, dx, lenxp, j);
            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (dx[j] == x_val) {
                /* Avoid potential non-finite interpolation */
                dres[i] = dy[j];
            }
            else {
                const npy_double slope =
                        (slopes != NULL) ? slopes[j] :
                        (dy[j+1] - dy[j]) / (dx[j+1] - dx[j]);

                /* If we get nan in one direction, try the other */
                dres[i] = slope*(x_val - dx[j]) + dy[j];
                if (NPY_UNLIKELY(npy_isnan(dres[i]))) {
                    dres[i] = slope*(x_val - dx[j+1]) + dy[j+1];
                    if (NPY_UNLIKELY(npy_isnan(dres[i])) && dy[j] == dy[j+1]) {
                        dres[i] = dy[j];
                    }
                }
            }
        }

        HPY_NPY_END_THREADS(ctx);
    }

    PyArray_free(slopes);
    HPy_Close(ctx, afp);
    HPy_Close(ctx, axp);
    HPy_Close(ctx, ax);
    HPy ret = HPyArray_Return(ctx, af);
    HPy_Close(ctx, af);
    return ret;

fail:
    HPy_Close(ctx, afp);
    HPy_Close(ctx, axp);
    HPy_Close(ctx, ax);
    HPy_Close(ctx, af);
    return HPy_NULL;
}

/* As for arr_interp but for complex fp values */
HPyDef_METH(arr_interp_complex, "interp_complex", HPyFunc_KEYWORDS)
static HPy
arr_interp_complex_impl(HPyContext *ctx, HPy NPY_UNUSED(self), const HPy *args, size_t nargs, HPy kwdict)
{

    HPy fp, xp, x;
    HPy left = HPy_NULL, right = HPy_NULL;
    HPy afp = HPy_NULL, axp = HPy_NULL, ax = HPy_NULL, af = HPy_NULL; // PyArrayObject *
    npy_intp i, lenx, lenxp;

    const npy_double *dx, *dz;
    const npy_cdouble *dy;
    npy_cdouble lval, rval;
    npy_cdouble *dres, *slopes = NULL;

    static const char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    HPY_NPY_BEGIN_THREADS_DEF;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwdict, "OOO|OO:interp_complex",
                                     kwlist, &x, &xp, &fp, &left, &right)) {
        return HPy_NULL;
    }

    HPy cdouble_descr = HPyArray_DescrFromType(ctx, NPY_CDOUBLE);
    afp = HPyArray_ContiguousFromAny(ctx, fp, cdouble_descr, 1, 1);
    HPy_Close(ctx, cdouble_descr);
    if (HPy_IsNull(afp)) {
        return HPy_NULL;
    }

    HPy double_descr = HPyArray_DescrFromType(ctx, NPY_DOUBLE);
    axp = HPyArray_ContiguousFromAny(ctx, xp, double_descr, 1, 1);
    if (HPy_IsNull(axp)) {
        HPy_Close(ctx, double_descr);
        goto fail;
    }
    ax = HPyArray_ContiguousFromAny(ctx, x, double_descr, 0, 0);
    HPy_Close(ctx, double_descr);
    if (HPy_IsNull(ax)) {
        goto fail;
    }
    PyArrayObject *axp_struct = PyArrayObject_AsStruct(ctx, axp);
    lenxp = PyArray_SIZE(axp_struct);
    if (lenxp == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    PyArrayObject *afp_struct = PyArrayObject_AsStruct(ctx, afp);
    if (PyArray_SIZE(afp_struct) != lenxp) {
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    PyArrayObject *ax_struct = PyArrayObject_AsStruct(ctx, ax);
    lenx = PyArray_SIZE(ax_struct);
    dx = (const npy_double *)PyArray_DATA(axp_struct);
    dz = (const npy_double *)PyArray_DATA(ax_struct);

    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    af = HPyArray_SimpleNew(ctx, array_type, PyArray_NDIM(ax_struct),
                                            PyArray_DIMS(ax_struct), NPY_CDOUBLE);
    HPy_Close(ctx, array_type);
    if (HPy_IsNull(af)) {
        goto fail;
    }

    dy = (const npy_cdouble *)PyArray_DATA(afp_struct);
    dres = (npy_cdouble *)PyArray_DATA(PyArrayObject_AsStruct(ctx, af));
    /* Get left and right fill values. */
    if (HPy_IsNull(left) || HPy_Is(ctx, left, ctx->h_None)) {
        lval = dy[0];
    }
    else {
        PyObject *py_left = HPy_AsPyObject(ctx, left);
        CAPI_WARN("missing PyComplex_RealAsDouble");
        lval.real = PyComplex_RealAsDouble(py_left);
        if (error_converting(lval.real)) {
            Py_DECREF(py_left);
            goto fail;
        }
        lval.imag = PyComplex_ImagAsDouble(py_left);
        Py_DECREF(py_left);
        if (error_converting(lval.imag)) {
            goto fail;
        }
    }

    if (HPy_IsNull(right) || HPy_Is(ctx, right, ctx->h_None)) {
        rval = dy[lenxp - 1];
    }
    else {
        CAPI_WARN("missing PyComplex_RealAsDouble");
        PyObject *py_right = HPy_AsPyObject(ctx, right);
        rval.real = PyComplex_RealAsDouble(py_right);
        if (error_converting(rval.real)) {
            Py_DECREF(py_right);
            goto fail;
        }
        rval.imag = PyComplex_ImagAsDouble(py_right);
        Py_DECREF(py_right);
        if (error_converting(rval.imag)) {
            goto fail;
        }
    }

    /* binary_search_with_guess needs at least a 3 item long array */
    if (lenxp == 1) {
        const npy_double xp_val = dx[0];
        const npy_cdouble fp_val = dy[0];

        HPY_NPY_BEGIN_THREADS_THRESHOLDED(ctx, lenx);
        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];
            dres[i] = (x_val < xp_val) ? lval :
              ((x_val > xp_val) ? rval : fp_val);
        }
        HPY_NPY_END_THREADS(ctx);
    }
    else {
        npy_intp j = 0;

        /* only pre-calculate slopes if there are relatively few of them. */
        if (lenxp <= lenx) {
            slopes = PyArray_malloc((lenxp - 1) * sizeof(npy_cdouble));
            if (slopes == NULL) {
                PyErr_NoMemory();
                goto fail;
            }
        }

        HPY_NPY_BEGIN_THREADS(ctx);

        if (slopes != NULL) {
            for (i = 0; i < lenxp - 1; ++i) {
                const double inv_dx = 1.0 / (dx[i+1] - dx[i]);
                slopes[i].real = (dy[i+1].real - dy[i].real) * inv_dx;
                slopes[i].imag = (dy[i+1].imag - dy[i].imag) * inv_dx;
            }
        }

        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];

            if (npy_isnan(x_val)) {
                dres[i].real = x_val;
                dres[i].imag = 0.0;
                continue;
            }

            j = binary_search_with_guess(x_val, dx, lenxp, j);
            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (dx[j] == x_val) {
                /* Avoid potential non-finite interpolation */
                dres[i] = dy[j];
            }
            else {
                npy_cdouble slope;
                if (slopes != NULL) {
                    slope = slopes[j];
                }
                else {
                    const npy_double inv_dx = 1.0 / (dx[j+1] - dx[j]);
                    slope.real = (dy[j+1].real - dy[j].real) * inv_dx;
                    slope.imag = (dy[j+1].imag - dy[j].imag) * inv_dx;
                }

                /* If we get nan in one direction, try the other */
                dres[i].real = slope.real*(x_val - dx[j]) + dy[j].real;
                if (NPY_UNLIKELY(npy_isnan(dres[i].real))) {
                    dres[i].real = slope.real*(x_val - dx[j+1]) + dy[j+1].real;
                    if (NPY_UNLIKELY(npy_isnan(dres[i].real)) &&
                            dy[j].real == dy[j+1].real) {
                        dres[i].real = dy[j].real;
                    }
                }
                dres[i].imag = slope.imag*(x_val - dx[j]) + dy[j].imag;
                if (NPY_UNLIKELY(npy_isnan(dres[i].imag))) {
                    dres[i].imag = slope.imag*(x_val - dx[j+1]) + dy[j+1].imag;
                    if (NPY_UNLIKELY(npy_isnan(dres[i].imag)) &&
                            dy[j].imag == dy[j+1].imag) {
                        dres[i].imag = dy[j].imag;
                    }
                }
            }
        }

        HPY_NPY_END_THREADS(ctx);
    }
    PyArray_free(slopes);

    HPy_Close(ctx, afp);
    HPy_Close(ctx, axp);
    HPy_Close(ctx, ax);
    HPy ret = HPyArray_Return(ctx, af);
    HPy_Close(ctx, af);
    return ret;

fail:
    HPy_Close(ctx, afp);
    HPy_Close(ctx, axp);
    HPy_Close(ctx, ax);
    HPy_Close(ctx, af);
    return HPy_NULL;
}

static const char *EMPTY_SEQUENCE_ERR_MSG = "indices must be integral: the provided " \
    "empty sequence was inferred as float. Wrap it with " \
    "'np.array(indices, dtype=np.intp)'";

static const char *NON_INTEGRAL_ERROR_MSG = "only int indices permitted";

/* Convert obj to an ndarray with integer dtype or fail */
static HPy // PyArrayObject *
hpy_astype_anyint(HPyContext *ctx, HPy obj) {
    HPy ret; // PyArrayObject *

    if (!HPyArray_Check(ctx, obj)) {
        /* prefer int dtype */
        HPy dtype_guess = HPy_NULL; // PyArray_Descr *
        if (HPyArray_DTypeFromObject(ctx, obj, NPY_MAXDIMS, &dtype_guess) < 0) {
            return HPy_NULL;
        }
        if (HPy_IsNull(dtype_guess)) {
            if (HPySequence_Check(ctx, obj) && HPy_Length(ctx, obj) == 0) {
                HPyErr_SetString(ctx, ctx->h_TypeError, EMPTY_SEQUENCE_ERR_MSG);
            }
            return HPy_NULL;
        }
        ret = HPyArray_FromAny(ctx, obj, dtype_guess, 0, 0, 0, HPy_NULL);
        if (HPy_IsNull(ret)) {
            return HPy_NULL;
        }
    }
    else {
        ret = HPy_Dup(ctx, obj);
        // Py_INCREF(ret);
    }

    if (!(HPyArray_ISINTEGER(ctx, ret) || HPyArray_ISBOOL(ctx, ret))) {
        /* ensure dtype is int-based */
        HPyErr_SetString(ctx, ctx->h_TypeError, NON_INTEGRAL_ERROR_MSG);
        HPy_Close(ctx, ret);
        return HPy_NULL;
    }

    return ret;
}

/*
 * Converts a Python sequence into 'count' PyArrayObjects
 *
 * seq         - Input Python object, usually a tuple but any sequence works.
 *               Must have integral content.
 * paramname   - The name of the parameter that produced 'seq'.
 * count       - How many arrays there should be (errors if it doesn't match).
 * op          - Where the arrays are placed.
 */
static int hpy_int_sequence_to_arrays(HPyContext *ctx, HPy seq,
                              char *paramname,
                              int count,
                              HPy *op
                              )
{
    int i;

    if (!HPySequence_Check(ctx, seq) || HPy_Length(ctx, seq) != count) {
        HPyErr_Format_p(ctx, ctx->h_ValueError,
                "parameter %s must be a sequence of length %d",
                paramname, count);
        return -1;
    }

    for (i = 0; i < count; ++i) {
        HPy item = HPy_GetItem_i(ctx, seq, i);
        if (HPy_IsNull(item)) {
            goto fail;
        }
        op[i] = hpy_astype_anyint(ctx, item);
        HPy_Close(ctx, item);
        if (HPy_IsNull(op[i])) {
            goto fail;
        }
    }

    return 0;

fail:
    while (--i >= 0) {
        HPy_Close(ctx, op[i]);
        op[i] = HPy_NULL;
    }
    return -1;
}

/* Inner loop for ravel_multi_index */
static int
ravel_multi_index_loop(HPyContext *ctx, int ravel_ndim, npy_intp *ravel_dims,
                        npy_intp *ravel_strides,
                        npy_intp count,
                        NPY_CLIPMODE *modes,
                        char **coords, npy_intp *coords_strides)
{
    int i;
    char invalid;
    npy_intp j, m;

    /*
     * Check for 0-dimensional axes unless there is nothing to do.
     * An empty array/shape cannot be indexed at all.
     */
    if (count != 0) {
        for (i = 0; i < ravel_ndim; ++i) {
            if (ravel_dims[i] == 0) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "cannot unravel if shape has zero entries (is empty).");
                return NPY_FAIL;
            }
        }
    }

    NPY_BEGIN_ALLOW_THREADS;
    invalid = 0;
    while (count--) {
        npy_intp raveled = 0;
        for (i = 0; i < ravel_ndim; ++i) {
            m = ravel_dims[i];
            j = *(npy_intp *)coords[i];
            switch (modes[i]) {
                case NPY_RAISE:
                    if (j < 0 || j >= m) {
                        invalid = 1;
                        goto end_while;
                    }
                    break;
                case NPY_WRAP:
                    if (j < 0) {
                        j += m;
                        if (j < 0) {
                            j = j % m;
                            if (j != 0) {
                                j += m;
                            }
                        }
                    }
                    else if (j >= m) {
                        j -= m;
                        if (j >= m) {
                            j = j % m;
                        }
                    }
                    break;
                case NPY_CLIP:
                    if (j < 0) {
                        j = 0;
                    }
                    else if (j >= m) {
                        j = m - 1;
                    }
                    break;

            }
            raveled += j * ravel_strides[i];

            coords[i] += coords_strides[i];
        }
        *(npy_intp *)coords[ravel_ndim] = raveled;
        coords[ravel_ndim] += coords_strides[ravel_ndim];
    }
end_while:
    NPY_END_ALLOW_THREADS;
    if (invalid) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
              "invalid entry in coordinates array");
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/* ravel_multi_index implementation - see add_newdocs.py */
HPyDef_METH(arr_ravel_multi_index, "ravel_multi_index", HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
arr_ravel_multi_index_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), const HPy *args, size_t nargs, HPy kwnames)
{
    int i;
    HPy mode0 = HPy_NULL, coords0 = HPy_NULL;
    HPy ret = HPy_NULL; // PyArrayObject *
    PyArray_Dims dimensions={0,0};
    npy_intp s, ravel_strides[NPY_MAXDIMS];
    NPY_ORDER order = NPY_CORDER;
    NPY_CLIPMODE modes[NPY_MAXDIMS];

    HPy op[NPY_MAXARGS]; // PyArrayObject *
    HPy dtype[NPY_MAXARGS]; // PyArray_Descr *
    npy_uint32 op_flags[NPY_MAXARGS];

    NpyIter *iter = NULL;

    static const char *kwlist[] = {"multi_index", "dims", "mode", "order", NULL};

    memset(op, 0, sizeof(op));
    dtype[0] = HPy_NULL;

    HPyTracker ht;
    HPy h_dimensions = HPy_NULL, h_order = HPy_NULL;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwnames,
                        "OO|OO:ravel_multi_index", kwlist,
                     &coords0,
                     &h_dimensions,
                     &mode0,
                     &h_order)) {
        return HPy_NULL;
    }
    if (HPyArray_IntpConverter(ctx, h_dimensions, &dimensions) != NPY_SUCCEED ||
            HPyArray_OrderConverter(ctx, h_order, &order) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "ravel_multi_index: TODO");
        goto fail;
    }

    if (dimensions.len+1 > NPY_MAXARGS) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                    "too many dimensions passed to ravel_multi_index");
        goto fail;
    }

    if (!HPyArray_ConvertClipmodeSequence(ctx, mode0, modes, dimensions.len)) {
       goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            s = 1;
            for (i = dimensions.len-1; i >= 0; --i) {
                ravel_strides[i] = s;
                if (npy_mul_with_overflow_intp(&s, s, dimensions.ptr[i])) {
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                        "invalid dims: array size defined by dims is larger "
                        "than the maximum possible size.");
                    goto fail;
                }
            }
            break;
        case NPY_FORTRANORDER:
            s = 1;
            for (i = 0; i < dimensions.len; ++i) {
                ravel_strides[i] = s;
                if (npy_mul_with_overflow_intp(&s, s, dimensions.ptr[i])) {
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                        "invalid dims: array size defined by dims is larger "
                        "than the maximum possible size.");
                    goto fail;
                }
            }
            break;
        default:
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    /* Get the multi_index into op */
    if (hpy_int_sequence_to_arrays(ctx, coords0, "multi_index", dimensions.len, op) < 0) {
        goto fail;
    }

    for (i = 0; i < dimensions.len; ++i) {
        op_flags[i] = NPY_ITER_READONLY|
                      NPY_ITER_ALIGNED;
    }
    op_flags[dimensions.len] = NPY_ITER_WRITEONLY|
                               NPY_ITER_ALIGNED|
                               NPY_ITER_ALLOCATE;
    dtype[0] = HPyArray_DescrFromType(ctx, NPY_INTP);
    for (i = 1; i <= dimensions.len; ++i) {
        dtype[i] = dtype[0];
    }

    iter = HNpyIter_MultiNew(ctx, dimensions.len+1, op, NPY_ITER_BUFFERED|
                                                  NPY_ITER_EXTERNAL_LOOP|
                                                  NPY_ITER_ZEROSIZE_OK,
                                                  NPY_KEEPORDER,
                                                  NPY_SAME_KIND_CASTING,
                                                  op_flags, dtype);
    if (iter == NULL) {
        goto fail;
    }

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strides;
        npy_intp *countptr;

        iternext = HNpyIter_GetIterNext(ctx, iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strides = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            if (ravel_multi_index_loop(ctx, dimensions.len, dimensions.ptr,
                        ravel_strides, *countptr, modes,
                        dataptr, strides) != NPY_SUCCEED) {
                goto fail;
            }
        } while(iternext(ctx, iter));
    }

    ret = HPy_Dup(ctx, HNpyIter_GetOperandArray(iter)[dimensions.len]);
    // Py_INCREF(ret);

    HPy_Close(ctx, dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        HPy_Close(ctx, op[i]);
    }
    npy_free_cache_dim_obj(dimensions);
    HNpyIter_Deallocate(ctx, iter);
    HPy r = ret;
    ret = HPyArray_Return(ctx, r);
    HPy_Close(ctx, r);
    HPyTracker_Close(ctx, ht);
    return ret;

fail:
    HPyTracker_Close(ctx, ht);
    HPy_Close(ctx, dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        HPy_Close(ctx, op[i]);
    }
    npy_free_cache_dim_obj(dimensions);
    HNpyIter_Deallocate(ctx, iter);
    return HPy_NULL;
}


/*
 * Inner loop for unravel_index
 * order must be NPY_CORDER or NPY_FORTRANORDER
 */
static int
unravel_index_loop(HPyContext *ctx, int unravel_ndim, npy_intp const *unravel_dims,
                   npy_intp unravel_size, npy_intp count,
                   char *indices, npy_intp indices_stride,
                   npy_intp *coords, NPY_ORDER order)
{
    int i, idx;
    int idx_start = (order == NPY_CORDER) ? unravel_ndim - 1: 0;
    int idx_step = (order == NPY_CORDER) ? -1 : 1;
    char invalid = 0;
    npy_intp val = 0;

    HPY_NPY_BEGIN_ALLOW_THREADS(ctx);
    /* NPY_KEEPORDER or NPY_ANYORDER have no meaning in this setting */
    assert(order == NPY_CORDER || order == NPY_FORTRANORDER);
    while (count--) {
        val = *(npy_intp *)indices;
        if (val < 0 || val >= unravel_size) {
            invalid = 1;
            break;
        }
        idx = idx_start;
        for (i = 0; i < unravel_ndim; ++i) {
            /*
             * Using a local seems to enable single-divide optimization
             * but only if the / precedes the %
             */
            npy_intp tmp = val / unravel_dims[idx];
            coords[idx] = val % unravel_dims[idx];
            val = tmp;
            idx += idx_step;
        }
        coords += unravel_ndim;
        indices += indices_stride;
    }
    HPY_NPY_END_ALLOW_THREADS(ctx);
    if (invalid) {
        HPyErr_Format_p(ctx, ctx->h_ValueError,
            "index %" NPY_INTP_FMT " is out of bounds for array with size "
            "%" NPY_INTP_FMT,
            val, unravel_size
        );
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/* unravel_index implementation - see add_newdocs.py */
HPyDef_METH(arr_unravel_index, "unravel_index", HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
arr_unravel_index_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), const HPy *args, size_t nargs, HPy kwnames)
{
    HPy indices0 = HPy_NULL;
    HPyTupleBuilder ret_tuple;
    HPy ret_arr = HPy_NULL; // PyArrayObject *
    HPy indices = HPy_NULL; // PyArrayObject *
    HPy dtype = HPy_NULL; // PyArray_Descr *
    HPy array_type = HPy_NULL;
    PyArray_Dims dimensions = {0, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_intp unravel_size;

    NpyIter *iter = NULL;
    int i, ret_ndim;
    npy_intp ret_dims[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];

    static const char *kwlist[] = {"indices", "shape", "order", NULL};

    HPy h_dimensions = HPy_NULL, h_order = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwnames, "OO|O:unravel_index",
                    kwlist,
                    &indices0,
                    &h_dimensions,
                    &h_order)) {
        goto fail;
    }
    if (HPyArray_IntpConverter(ctx, h_dimensions, &dimensions) != NPY_SUCCEED ||
            HPyArray_OrderConverter(ctx, h_order, &order) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "unravel_index: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    unravel_size = PyArray_OverflowMultiplyList(dimensions.ptr, dimensions.len);
    if (unravel_size == -1) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "dimensions are too large; arrays and shapes with "
                        "a total size greater than 'intp' are not supported.");
        goto fail;
    }

    indices = hpy_astype_anyint(ctx, indices0);
    if (HPy_IsNull(indices)) {
        goto fail;
    }

    dtype = HPyArray_DescrFromType(ctx, NPY_INTP);
    if (HPy_IsNull(dtype)) {
        goto fail;
    }

    iter = HNpyIter_New(ctx, indices, NPY_ITER_READONLY|
                                NPY_ITER_ALIGNED|
                                NPY_ITER_BUFFERED|
                                NPY_ITER_ZEROSIZE_OK|
                                NPY_ITER_DONT_NEGATE_STRIDES|
                                NPY_ITER_MULTI_INDEX,
                                NPY_KEEPORDER, NPY_SAME_KIND_CASTING,
                                dtype);
    if (iter == NULL) {
        goto fail;
    }

    /*
     * Create the return array with a layout compatible with the indices
     * and with a dimension added to the end for the multi-index
     */
    PyArrayObject *indices_struct = PyArrayObject_AsStruct(ctx, indices);
    ret_ndim = PyArray_NDIM(indices_struct) + 1;
    if (NpyIter_GetShape(iter, ret_dims) != NPY_SUCCEED) {
        goto fail;
    }
    ret_dims[ret_ndim-1] = dimensions.len;
    if (HNpyIter_CreateCompatibleStrides(ctx, iter,
                dimensions.len*sizeof(npy_intp), ret_strides) != NPY_SUCCEED) {
        goto fail;
    }
    ret_strides[ret_ndim-1] = sizeof(npy_intp);

    /* Remove the multi-index and inner loop */
    if (HNpyIter_RemoveMultiIndex(ctx, iter) != NPY_SUCCEED) {
        goto fail;
    }
    if (HNpyIter_EnableExternalLoop(ctx, iter) != NPY_SUCCEED) {
        goto fail;
    }

    array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    ret_arr = HPyArray_NewFromDescr(ctx, array_type, dtype,
                            ret_ndim, ret_dims, ret_strides, NULL, 0, HPy_NULL);
    dtype = HPy_NULL;
    if (HPy_IsNull(ret_arr)) {
        goto fail;
    }

    PyArrayObject *ret_arr_struct = PyArrayObject_AsStruct(ctx, ret_arr);

    if (order != NPY_CORDER && order != NPY_FORTRANORDER) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "only 'C' or 'F' order is permitted");
        goto fail;
    }
    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strides;
        npy_intp *countptr, count;
        npy_intp *coordsptr = (npy_intp *)PyArray_DATA(ret_arr_struct);

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strides = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            count = *countptr;
            if (unravel_index_loop(ctx, dimensions.len, dimensions.ptr,
                                   unravel_size, count, *dataptr, *strides,
                                   coordsptr, order) != NPY_SUCCEED) {
                goto fail;
            }
            coordsptr += count * dimensions.len;
        } while (iternext(npy_get_context(), iter));
    }


    if (dimensions.len == 0 && PyArray_NDIM(indices_struct) != 0) {
        /*
         * There's no index meaning "take the only element 10 times"
         * on a zero-d array, so we have no choice but to error. (See gh-580)
         *
         * Do this check after iterating, so we give a better error message
         * for invalid indices.
         */
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "multiple indices are not supported for 0d arrays");
        goto fail;
    }

    /* Now make a tuple of views, one per index */
    ret_tuple = HPyTupleBuilder_New(ctx, dimensions.len);
    if (HPyTupleBuilder_IsNull(ret_tuple)) {
        goto fail;
    }
    
    HPy intp_descr = HPyArray_DescrFromType(ctx, NPY_INTP);
    for (i = 0; i < dimensions.len; ++i) {
        HPy view;

        view = HPyArray_NewFromDescrAndBase(ctx,
                array_type, intp_descr,
                ret_ndim - 1, ret_dims, ret_strides,
                PyArray_BYTES(ret_arr_struct) + i*sizeof(npy_intp),
                NPY_ARRAY_WRITEABLE, HPy_NULL, ret_arr);
        if (HPy_IsNull(view)) {
            HPyTupleBuilder_Cancel(ctx, ret_tuple);
            goto fail;
        }
        HPy ret_view = HPyArray_Return(ctx, view);
        HPyTupleBuilder_Set(ctx, ret_tuple, i, ret_view);
    }

    HPy_Close(ctx, array_type);
    HPy_Close(ctx, ret_arr);
    HPy_Close(ctx, indices);
    npy_free_cache_dim_obj(dimensions);
    HNpyIter_Deallocate(ctx, iter);

    return HPyTupleBuilder_Build(ctx, ret_tuple);

fail:
    // HPy_Close(ctx, ret_tuple);
    HPy_Close(ctx, array_type);
    HPy_Close(ctx, ret_arr);
    HPy_Close(ctx, dtype);
    HPy_Close(ctx, indices);
    npy_free_cache_dim_obj(dimensions);
    HNpyIter_Deallocate(ctx, iter);
    return HPy_NULL;
}

// TODO

HPyDef_METH(hpy_add_docstring, "add_docstring", HPyFunc_VARARGS)
static HPy
hpy_add_docstring_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), const HPy *args, size_t nargs)
{
    // HPy PORT: we will ignore adding docs and behave as `Py_OptimizeFlag > 1` path
    return HPy_Dup(ctx, ctx->h_None);
}

/* Can only be called if doc is currently NULL */
NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *obj;
    PyObject *str;
    #if !defined(PYPY_VERSION_NUM) || PYPY_VERSION_NUM > 0x07030300
    const char *docstr;
    #else
    char *docstr;
    #endif
    static char *msg = "already has a different docstring";

    /* Don't add docstrings */
    if (Py_OptimizeFlag > 1) {
        Py_RETURN_NONE;
    }

    if (!PyArg_ParseTuple(args, "OO!:add_docstring", &obj, &PyUnicode_Type, &str)) {
        return NULL;
    }

    docstr = PyUnicode_AsUTF8(str);
    if (docstr == NULL) {
        return NULL;
    }

#define _ADDDOC(doc, name)                                              \
        if (!(doc)) {                                                   \
            doc = docstr;                                               \
            Py_INCREF(str);  /* hold on to string (leaks reference) */  \
        }                                                               \
        else if (strcmp(doc, docstr) != 0) {                            \
            PyErr_Format(PyExc_RuntimeError, "%s method %s", name, msg); \
            return NULL;                                                \
        }

    if (Py_TYPE(obj) == &PyCFunction_Type) {
        PyCFunctionObject *new = (PyCFunctionObject *)obj;
        _ADDDOC(new->m_ml->ml_doc, new->m_ml->ml_name);
    }
    else if (PyObject_TypeCheck(obj, &PyType_Type)) {
        /*
         * We add it to both `tp_doc` and `__doc__` here.  Note that in theory
         * `tp_doc` extracts the signature line, but we currently do not use
         * it.  It may make sense to only add it as `__doc__` and
         * `__text_signature__` to the dict in the future.
         * The dictionary path is only necessary for heaptypes (currently not
         * used) and metaclasses.
         * If `__doc__` as stored in `tp_dict` is None, we assume this was
         * filled in by `PyType_Ready()` and should also be replaced.
         */
        PyTypeObject *new = (PyTypeObject *)obj;
        _ADDDOC(new->tp_doc, new->tp_name);
        if (new->tp_dict != NULL && PyDict_CheckExact(new->tp_dict) &&
                PyDict_GetItemString(new->tp_dict, "__doc__") == Py_None) {
            /* Warning: Modifying `tp_dict` is not generally safe! */
            if (PyDict_SetItemString(new->tp_dict, "__doc__", str) < 0) {
                return NULL;
            }
        }
    }
    else if (Py_TYPE(obj) == &PyMemberDescr_Type) {
        PyMemberDescrObject *new = (PyMemberDescrObject *)obj;
        _ADDDOC(new->d_member->doc, new->d_member->name);
    }
    else if (Py_TYPE(obj) == &PyGetSetDescr_Type) {
        PyGetSetDescrObject *new = (PyGetSetDescrObject *)obj;
        _ADDDOC(new->d_getset->doc, new->d_getset->name);
    }
    else if (Py_TYPE(obj) == &PyMethodDescr_Type) {
        PyMethodDescrObject *new = (PyMethodDescrObject *)obj;
        _ADDDOC(new->d_method->ml_doc, new->d_method->ml_name);
    }
    else {
        PyObject *doc_attr;

        doc_attr = PyObject_GetAttrString(obj, "__doc__");
        if (doc_attr != NULL && doc_attr != Py_None &&
                (PyUnicode_Compare(doc_attr, str) != 0)) {
            Py_DECREF(doc_attr);
            if (PyErr_Occurred()) {
                /* error during PyUnicode_Compare */
                return NULL;
            }
            PyErr_Format(PyExc_RuntimeError, "object %s", msg);
            return NULL;
        }
        Py_XDECREF(doc_attr);

        if (PyObject_SetAttrString(obj, "__doc__", str) < 0) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot set a docstring for that object");
            return NULL;
        }
        Py_RETURN_NONE;
    }

#undef _ADDDOC

    Py_RETURN_NONE;
}

/*
 * This function packs boolean values in the input array into the bits of a
 * byte array. Truth values are determined as usual: 0 is false, everything
 * else is true.
 */
static NPY_GCC_OPT_3 NPY_INLINE void
pack_inner(const char *inptr,
           npy_intp element_size,   /* in bytes */
           npy_intp n_in,
           npy_intp in_stride,
           char *outptr,
           npy_intp n_out,
           npy_intp out_stride,
           PACK_ORDER order)
{
    /*
     * Loop through the elements of inptr.
     * Determine whether or not it is nonzero.
     *  Yes: set corresponding bit (and adjust build value)
     *  No:  move on
     * Every 8th value, set the value of build and increment the outptr
     */
    npy_intp index = 0;
    int remain = n_in % 8;              /* uneven bits */

#if NPY_SIMD
    if (in_stride == 1 && element_size == 1 && n_out > 2) {
        npyv_u8 v_zero = npyv_zero_u8();
        /* don't handle non-full 8-byte remainder */
        npy_intp vn_out = n_out - (remain ? 1 : 0);
        const int vstep = npyv_nlanes_u64;
        const int vstepx4 = vstep * 4;
        const int isAligned = npy_is_aligned(outptr, sizeof(npy_uint64));
        vn_out -= (vn_out & (vstep - 1));
        for (; index <= vn_out - vstepx4; index += vstepx4, inptr += npyv_nlanes_u8 * 4) {
            npyv_u8 v0 = npyv_load_u8((const npy_uint8*)inptr);
            npyv_u8 v1 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 1);
            npyv_u8 v2 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 2);
            npyv_u8 v3 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 3);
            if (order == PACK_ORDER_BIG) {
                v0 = npyv_rev64_u8(v0);
                v1 = npyv_rev64_u8(v1);
                v2 = npyv_rev64_u8(v2);
                v3 = npyv_rev64_u8(v3);
            }
            npy_uint64 bb[4];
            bb[0] = npyv_tobits_b8(npyv_cmpneq_u8(v0, v_zero));
            bb[1] = npyv_tobits_b8(npyv_cmpneq_u8(v1, v_zero));
            bb[2] = npyv_tobits_b8(npyv_cmpneq_u8(v2, v_zero));
            bb[3] = npyv_tobits_b8(npyv_cmpneq_u8(v3, v_zero));
            if(out_stride == 1 && 
                (!NPY_ALIGNMENT_REQUIRED || isAligned)) {
                npy_uint64 *ptr64 = (npy_uint64*)outptr;
            #if NPY_SIMD_WIDTH == 16
                npy_uint64 bcomp = bb[0] | (bb[1] << 16) | (bb[2] << 32) | (bb[3] << 48);
                ptr64[0] = bcomp;
            #elif NPY_SIMD_WIDTH == 32
                ptr64[0] = bb[0] | (bb[1] << 32);
                ptr64[1] = bb[2] | (bb[3] << 32);
            #else
                ptr64[0] = bb[0]; ptr64[1] = bb[1];
                ptr64[2] = bb[2]; ptr64[3] = bb[3];
            #endif
                outptr += vstepx4;
            } else {
                for(int i = 0; i < 4; i++) {
                    for (int j = 0; j < vstep; j++) {
                        memcpy(outptr, (char*)&bb[i] + j, 1);
                        outptr += out_stride;
                    }
                }
            }
        }
        for (; index < vn_out; index += vstep, inptr += npyv_nlanes_u8) {
            npyv_u8 va = npyv_load_u8((const npy_uint8*)inptr);
            if (order == PACK_ORDER_BIG) {
                va = npyv_rev64_u8(va);
            }
            npy_uint64 bb = npyv_tobits_b8(npyv_cmpneq_u8(va, v_zero));
            for (int i = 0; i < vstep; ++i) {
                memcpy(outptr, (char*)&bb + i, 1);
                outptr += out_stride;
            }
        }
    }
#endif

    if (remain == 0) {                  /* assumes n_in > 0 */
        remain = 8;
    }
    /* Don't reset index. Just handle remainder of above block */
    for (; index < n_out; index++) {
        unsigned char build = 0;
        int maxi = (index == n_out - 1) ? remain : 8;
        if (order == PACK_ORDER_BIG) {
            for (int i = 0; i < maxi; i++) {
                build <<= 1;
                for (npy_intp j = 0; j < element_size; j++) {
                    build |= (inptr[j] != 0);
                }
                inptr += in_stride;
            }
            if (index == n_out - 1) {
                build <<= 8 - remain;
            }
        }
        else
        {
            for (int i = 0; i < maxi; i++) {
                build >>= 1;
                for (npy_intp j = 0; j < element_size; j++) {
                    build |= (inptr[j] != 0) ? 128 : 0;
                }
                inptr += in_stride;
            }
            if (index == n_out - 1) {
                build >>= 8 - remain;
            }
        }
        *outptr = (char)build;
        outptr += out_stride;
    }
}

static HPy
pack_bits(HPyContext *ctx, HPy input, int axis, char order)
{
    HPy inp; // PyArrayObject *
    HPy new = HPy_NULL; // PyArrayObject *
    HPy out = HPy_NULL; // PyArrayObject *
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    HPY_NPY_BEGIN_THREADS_DEF;

    inp = HPyArray_FROM_O(ctx, input);

    if (HPy_IsNull(inp)) {
        return HPy_NULL;
    }
    if (!HPyArray_ISBOOL(ctx, inp) && !HPyArray_ISINTEGER(ctx, inp)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Expected an input array of integer or boolean data type");
        HPy_Close(ctx, inp);
        goto fail;
    }

    new = HPyArray_CheckAxis(ctx, inp, &axis, 0);
    // Py_DECREF(inp);
    if (HPy_IsNull(new)) {
        return HPy_NULL;
    }

    PyArrayObject *new_struct = PyArrayObject_AsStruct(ctx, new);
    if (PyArray_NDIM(new_struct) == 0) {
        char *optr, *iptr;

        HPy ubyte_descr = HPyArray_DescrFromType(ctx, NPY_UBYTE);
        HPy new_type = HPy_Type(ctx, new);
        out = HPyArray_NewFromDescr(ctx,
                new_type, ubyte_descr,
                0, NULL, NULL, NULL,
                0, HPy_NULL);
        HPy_Close(ctx, ubyte_descr);
        HPy_Close(ctx, new_type);
        if (HPy_IsNull(out)) {
            goto fail;
        }
        optr = PyArray_DATA(PyArrayObject_AsStruct(ctx, out));
        iptr = PyArray_DATA(new_struct);
        *optr = 0;
        for (i = 0; i < PyArray_ITEMSIZE(new_struct); i++) {
            if (*iptr != 0) {
                *optr = 1;
                break;
            }
            iptr++;
        }
        goto finish;
    }


    /* Setup output shape */
    for (i = 0; i < PyArray_NDIM(new_struct); i++) {
        outdims[i] = PyArray_DIM(new_struct, i);
    }

    /*
     * Divide axis dimension by 8
     * 8 -> 1, 9 -> 2, 16 -> 2, 17 -> 3 etc..
     */
    outdims[axis] = ((outdims[axis] - 1) >> 3) + 1;

    /* Create output array */
    HPy ubyte_descr = HPyArray_DescrFromType(ctx, NPY_UBYTE);
    HPy new_type = HPy_Type(ctx, new);
    out = HPyArray_NewFromDescr(ctx,
            new_type, ubyte_descr,
            PyArray_NDIM(new_struct), outdims, NULL, NULL,
            PyArray_ISFORTRAN(new_struct), HPy_NULL);
    HPy_Close(ctx, ubyte_descr);
    HPy_Close(ctx, new_type);
    if (HPy_IsNull(out)) {
        goto fail;
    }
    /* Setup iterators to iterate over all but given axis */
    it = (PyArrayIterObject *)HPyArray_IterAllButAxis(ctx, new, &axis);
    ot = (PyArrayIterObject *)HPyArray_IterAllButAxis(ctx, out, &axis);
    if (it == NULL || ot == NULL) {
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }
    const PACK_ORDER ordere = order == 'b' ? PACK_ORDER_BIG : PACK_ORDER_LITTLE;
    PyArrayObject *out_struct = PyArrayObject_AsStruct(ctx, out);
    HPY_NPY_BEGIN_THREADS_THRESHOLDED(ctx, PyArray_DIM(out_struct, axis));
    while (PyArray_ITER_NOTDONE(it)) {
        pack_inner(PyArray_ITER_DATA(it), PyArray_ITEMSIZE(new_struct),
                   PyArray_DIM(new_struct, axis), PyArray_STRIDE(new_struct, axis),
                   PyArray_ITER_DATA(ot), PyArray_DIM(out_struct, axis),
                   PyArray_STRIDE(out_struct, axis), ordere);
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    HPY_NPY_END_THREADS(ctx);

    Py_DECREF(it);
    Py_DECREF(ot);

finish:
    HPy_Close(ctx, new);
    return out;

fail:
    HPy_Close(ctx, new);
    HPy_Close(ctx, out);
    return HPy_NULL;
}

static HPy
unpack_bits(HPyContext *ctx, HPy input, int axis, HPy count_obj, char order)
{
    static int unpack_init = 0;
    /*
     * lookuptable for bitorder big as it has been around longer
     * bitorder little is handled via byteswapping in the loop
     */
    static union {
        npy_uint8  bytes[8];
        npy_uint64 uint64;
    } unpack_lookup_big[256];
    HPy inp; // PyArrayObject *
    HPy new = HPy_NULL; // PyArrayObject *
    HPy out = HPy_NULL; // PyArrayObject *
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    npy_intp count, in_n, in_tail, out_pad, in_stride, out_stride;
    HPY_NPY_BEGIN_THREADS_DEF;

    inp = HPyArray_FROM_O(ctx, input);

    if (HPy_IsNull(inp)) {
        return HPy_NULL;
    }
    if (HPyArray_GetType(ctx, inp) != NPY_UBYTE) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Expected an input array of unsigned byte data type");
        HPy_Close(ctx, inp);
        goto fail;
    }

    new = HPyArray_CheckAxis(ctx, inp, &axis, 0);
    HPy_Close(ctx, inp);
    if (HPy_IsNull(new)) {
        return HPy_NULL;
    }

    PyArrayObject *new_struct = PyArrayObject_AsStruct(ctx, new);
    if (PyArray_NDIM(new_struct) == 0) {
        /* Handle 0-d array by converting it to a 1-d array */
        HPy temp; // PyArrayObject *
        PyArray_Dims newdim = {NULL, 1};
        npy_intp shape = 1;

        newdim.ptr = &shape;
        temp = HPyArray_Newshape(ctx, new, new_struct, &newdim, NPY_CORDER);
        HPy_Close(ctx, new);
        if (HPy_IsNull(temp)) {
            return HPy_NULL;
        }
        new = temp;
        new_struct = PyArrayObject_AsStruct(ctx, new);
    }

    /* Setup output shape */
    for (i = 0; i < PyArray_NDIM(new_struct); i++) {
        outdims[i] = PyArray_DIM(new_struct, i);
    }

    /* Multiply axis dimension by 8 */
    outdims[axis] *= 8;
    if (!HPy_Is(ctx, count_obj, ctx->h_None)) {
        count = HPyArray_PyIntAsIntp(ctx, count_obj);
        if (hpy_error_converting(ctx, count)) {
            goto fail;
        }
        if (count < 0) {
            outdims[axis] += count;
            if (outdims[axis] < 0) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                             "-count larger than number of elements");
                goto fail;
            }
        }
        else {
            outdims[axis] = count;
        }
    }

    /* Create output array */
    HPy ubyte_descr = HPyArray_DescrFromType(ctx, NPY_UBYTE);
    HPy new_type = HPy_Type(ctx, new);
    out = HPyArray_NewFromDescr(ctx,
            new_type, ubyte_descr,
            PyArray_NDIM(new_struct), outdims, NULL, NULL,
            PyArray_ISFORTRAN(new_struct), HPy_NULL);
    HPy_Close(ctx, ubyte_descr);
    HPy_Close(ctx, new_type);
    if (HPy_IsNull(out)) {
        goto fail;
    }

    /* Setup iterators to iterate over all but given axis */
    it = (PyArrayIterObject *)HPyArray_IterAllButAxis(ctx, new, &axis);
    ot = (PyArrayIterObject *)HPyArray_IterAllButAxis(ctx, out, &axis);
    if (it == NULL || ot == NULL) {
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }

    /*
     * setup lookup table under GIL, 256 8 byte blocks representing 8 bits
     * expanded to 1/0 bytes
     */
    if (unpack_init == 0) {
        npy_intp j;
        for (j=0; j < 256; j++) {
            npy_intp k;
            for (k=0; k < 8; k++) {
                npy_uint8 v = (j & (1 << k)) == (1 << k);
                unpack_lookup_big[j].bytes[7 - k] = v;
            }
        }
        unpack_init = 1;
    }

    count = PyArray_DIM(new_struct, axis) * 8;
    if (outdims[axis] > count) {
        in_n = count / 8;
        in_tail = 0;
        out_pad = outdims[axis] - count;
    }
    else {
        in_n = outdims[axis] / 8;
        in_tail = outdims[axis] % 8;
        out_pad = 0;
    }

    in_stride = PyArray_STRIDE(new_struct, axis);
    out_stride = PyArray_STRIDE(PyArrayObject_AsStruct(ctx, out), axis);

    HPY_NPY_BEGIN_THREADS_THRESHOLDED(ctx, HPyArray_Size(ctx, out) / 8);

    while (PyArray_ITER_NOTDONE(it)) {
        npy_intp index;
        unsigned const char *inptr = PyArray_ITER_DATA(it);
        char *outptr = PyArray_ITER_DATA(ot);

        if (out_stride == 1) {
            /* for unity stride we can just copy out of the lookup table */
            if (order == 'b') {
                for (index = 0; index < in_n; index++) {
                    npy_uint64 v = unpack_lookup_big[*inptr].uint64;
                    memcpy(outptr, &v, 8);
                    outptr += 8;
                    inptr += in_stride;
                }
            }
            else {
                for (index = 0; index < in_n; index++) {
                    npy_uint64 v = unpack_lookup_big[*inptr].uint64;
                    if (order != 'b') {
                        v = npy_bswap8(v);
                    }
                    memcpy(outptr, &v, 8);
                    outptr += 8;
                    inptr += in_stride;
                }
            }
            /* Clean up the tail portion */
            if (in_tail) {
                npy_uint64 v = unpack_lookup_big[*inptr].uint64;
                if (order != 'b') {
                    v = npy_bswap8(v);
                }
                memcpy(outptr, &v, in_tail);
            }
            /* Add padding */
            else if (out_pad) {
                memset(outptr, 0, out_pad);
            }
        }
        else {
            if (order == 'b') {
                for (index = 0; index < in_n; index++) {
                    for (i = 0; i < 8; i++) {
                        *outptr = ((*inptr & (128 >> i)) != 0);
                        outptr += out_stride;
                    }
                    inptr += in_stride;
                }
                /* Clean up the tail portion */
                for (i = 0; i < in_tail; i++) {
                    *outptr = ((*inptr & (128 >> i)) != 0);
                    outptr += out_stride;
                }
            }
            else {
                for (index = 0; index < in_n; index++) {
                    for (i = 0; i < 8; i++) {
                        *outptr = ((*inptr & (1 << i)) != 0);
                        outptr += out_stride;
                    }
                    inptr += in_stride;
                }
                /* Clean up the tail portion */
                for (i = 0; i < in_tail; i++) {
                    *outptr = ((*inptr & (1 << i)) != 0);
                    outptr += out_stride;
                }
            }
            /* Add padding */
            for (index = 0; index < out_pad; index++) {
                *outptr = 0;
                outptr += out_stride;
            }
        }

        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    HPY_NPY_END_THREADS(ctx);

    Py_DECREF(it);
    Py_DECREF(ot);

    HPy_Close(ctx, new);
    return out;

fail:
    HPy_Close(ctx, new);
    HPy_Close(ctx, out);
    return HPy_NULL;
}

HPyDef_METH(io_pack, "packbits", HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
io_pack_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), const HPy *args, size_t nargs, HPy kwnames)
{
    HPy obj;
    int axis = NPY_MAXDIMS;
    static const char *kwlist[] = {"in", "axis", "bitorder", NULL};
    char c = 'b';
    const char * order_str = NULL;

    HPy h_axis = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwnames, "O|O&s:pack" , kwlist,
                &obj, &h_axis, &order_str)) {
        return HPy_NULL;
    }
    if (HPyArray_AxisConverter(ctx, h_axis, &axis) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "pack: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    if (order_str != NULL) {
        if (strncmp(order_str, "little", 6) == 0)
            c = 'l';
        else if (strncmp(order_str, "big", 3) == 0)
            c = 'b';
        else {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "'order' must be either 'little' or 'big'");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
    }
    HPy ret = pack_bits(ctx, obj, axis, c);
    HPyTracker_Close(ctx, ht);
    return ret;
}

HPyDef_METH(io_unpack, "unpackbits", HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
io_unpack_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), const HPy *args, size_t nargs, HPy kwnames)
{
    HPy obj;
    int axis = NPY_MAXDIMS;
    HPy count = ctx->h_None;
    static const char *kwlist[] = {"in", "axis", "count", "bitorder", NULL};
    const char * c = NULL;

    HPy h_axis = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwnames, "O|O&Os:unpack" , kwlist,
                &obj,  &h_axis, &count, &c)) {
        return HPy_NULL;
    }
    if (HPyArray_AxisConverter(ctx, h_axis, &axis) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "unpack: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    if (c == NULL) {
        c = "b";
    }
    if (c[0] != 'l' && c[0] != 'b') {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                    "'order' must begin with 'l' or 'b'");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    HPy ret = unpack_bits(ctx, obj, axis, count, c[0]);
    HPyTracker_Close(ctx, ht);
    return ret;
}

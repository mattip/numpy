#ifndef NUMPY_CORE_SRC_MULTIARRAY_HPY_UTILS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_HPY_UTILS_H_

#include "numpy/ndarraytypes.h"

/* declare alloca() */
#if defined(_MSC_VER)
# include <malloc.h>   /* for alloca() */
#else
# include <stdint.h>
# if (defined (__SVR4) && defined (__sun)) || defined(_AIX) || defined(__hpux)
#  include <alloca.h>
# endif
#endif

#ifdef CPYTHON_DEBUG
    #define OBJECT_MALLOC PyObject_Malloc
    #define OBJECT_FREE PyObject_Free
    #define MEM_MALLOC PyMem_Malloc
    #define MEM_CALLOC PyMem_Calloc
    #define MEM_CALLOC PyMem_RawCalloc
    #define MEM_FREE PyMem_Free
#else
    #define OBJECT_MALLOC malloc
    #define OBJECT_FREE free
    #define MEM_MALLOC malloc
    #define MEM_CALLOC calloc
    #define MEM_RAWCALLOC calloc
    #define MEM_FREE free
#endif

#if defined(_MSC_VER) && defined(__cplusplus) // MSVC C4576
#  define _htconv(h) {h}
#else
#  define _htconv(h) ((HPyTracker){h})
#endif

#define HPyTracker_NULL _htconv(0)
#define HPyTracker_IsNull(f) ((f)._i == 0)

// #define GRAALVM_PYTHON 1

#define HPy_SETREF(ctx, op, op2) \
    do {                         \
        HPy _h_tmp = (op);       \
        (op) = (op2);            \
        HPy_Close(ctx, _h_tmp);  \
    } while (0)

/* Set an error with a format string; it will use 'vsnprintf' for formatting. */
NPY_NO_EXPORT void
HPyErr_Format_p(HPyContext *ctx, HPy h_type, const char *fmt, ...);

NPY_NO_EXPORT HPy
HPyUnicode_FromFormat_p(HPyContext *ctx, const char *fmt, ...);

NPY_NO_EXPORT HPy
HPyUnicode_Concat_t(HPyContext *ctx, HPy s1, HPy s2);

NPY_NO_EXPORT int
HPyGlobal_Is(HPyContext *ctx, HPy obj, HPyGlobal expected);

NPY_NO_EXPORT int
HPyGlobal_TypeCheck(HPyContext *ctx, HPy obj, HPyGlobal type);

/**
 * Converts the vectorcall calling convention (i.e. the argument array contains
 * 'nargs' positional arguments and after that, the keyword argument values) to
 * HPy's calling convention (i.e. positional arguments in an array and keywords
 * in a dict). It returns a the keywords dict.
 */
NPY_NO_EXPORT HPy
HPyFastcallToDict(HPyContext *ctx, HPy *args, HPy_ssize_t nargs, HPy kwnames);

static NPY_INLINE HPy *
HPy_FromPyObjectArray(HPyContext *ctx, PyObject **arr, Py_ssize_t n)
{
    if (!arr)
        return NULL;
    HPy *h_arr = PyMem_RawCalloc(n, sizeof(HPy));
    Py_ssize_t i;
    for (i = 0; i < n; i++) {
        h_arr[i] = HPy_FromPyObject(ctx, arr[i]);
    }
    return h_arr;
}

static NPY_INLINE void
HPy_CloseArray(HPyContext *ctx, HPy *h_arr, HPy_ssize_t n)
{
    if (!h_arr)
        return;
    HPy_ssize_t i;
    for (i = 0; i < n; i++) {
        HPy_Close(ctx, h_arr[i]);
    }
}

static NPY_INLINE void
HPy_CloseAndFreeArray(HPyContext *ctx, HPy *h_arr, HPy_ssize_t n)
{
    if (!h_arr)
        return;
    HPy_CloseArray(ctx, h_arr, n);
    PyMem_RawFree(h_arr);
}

static NPY_INLINE void
HPy_CloseAndFreeFieldArray(HPyContext *ctx, HPy obj, HPyField *h_arr, HPy_ssize_t n)
{
    if (!h_arr)
        return;
    HPy_ssize_t i;
    for (i = 0; i < n; i++) {
        HPyField_Store(ctx, obj, &h_arr[i], HPy_NULL);
    }
    PyMem_RawFree(h_arr);
}

static NPY_INLINE PyObject **
HPy_AsPyObjectArray(HPyContext *ctx, HPy *h_arr, HPy_ssize_t n)
{
    if (!h_arr)
        return NULL;
    PyObject **arr = PyMem_RawCalloc(n, sizeof(PyObject *));
    HPy_ssize_t i;
    for (i = 0; i < n; i++) {
        arr[i] = HPy_AsPyObject(ctx, h_arr[i]);
    }
    return arr;
}

static NPY_INLINE void
HPy_DecrefAndFreeArray(HPyContext *ctx, PyObject **arr, Py_ssize_t n)
{
    if (!arr)
        return;
    Py_ssize_t i;
    for (i = 0; i < n; i++) {
        Py_XDECREF(arr[i]);
    }
    PyMem_RawFree(arr);
}

static NPY_INLINE int
HPyBool_Check(HPyContext *ctx, HPy obj)
{
    return HPy_TypeCheck(ctx, obj, ctx->h_BoolType);
}

static NPY_INLINE int
HPyLong_Check(HPyContext *ctx, HPy obj)
{
    return HPy_TypeCheck(ctx, obj, ctx->h_LongType);
}

static inline PyObject *HPyGlobal_LoadPyObj(HPyGlobal g) {
    HPyContext *ctx = npy_get_context();
    HPy h = HPyGlobal_Load(ctx, g);
    PyObject *res = HPy_AsPyObject(ctx, h);
    HPy_Close(ctx, h);
    return res;
}

static inline PyObject *HPyField_LoadPyObj(PyObject *owner, HPyField f) {
    HPyContext *ctx = npy_get_context();
    HPy h_owner = HPy_FromPyObject(ctx, owner);
    HPy h = HPyField_Load(ctx, h_owner, f);
    PyObject *res = HPy_AsPyObject(ctx, h);
    HPy_Close(ctx, h);
    HPy_Close(ctx, h_owner);
    return res;
}

static inline void HPyField_StorePyObj(PyObject *owner, HPyField *f, PyObject *value) {
    HPyContext *ctx = npy_get_context();
    HPy h_owner = HPy_FromPyObject(ctx, owner);
    HPy h_value = HPy_FromPyObject(ctx, value);
    HPyField_Store(ctx, h_owner, f, h_value);
    HPy_Close(ctx, h_value);
    HPy_Close(ctx, h_owner);
}

static inline int
HPyTuple_CheckExact(HPyContext *ctx, HPy h)
{
    HPy type = HPy_Type(ctx, h);
    int res = HPy_Is(ctx, type, ctx->h_TupleType);
    HPy_Close(ctx, type);
    return res;
}

static inline HPy HPyDict_GetItemWithError(HPyContext *ctx, HPy d, HPy k)
{
    HPy res = HPy_GetItem(ctx, d, k);
    if (HPy_IsNull(res) && HPyErr_Occurred(ctx)) {
        // PyDict_GetItemWithError supresses KeyErrors when the key is not present
        if (HPyErr_ExceptionMatches(ctx, ctx->h_KeyError)) {
            HPyErr_Clear(ctx);
        }
    }
    return res;
}

static inline int HPyDict_GetItemWithError_IsNull(HPyContext *ctx, HPy d, HPy k) {
    HPy item = HPyDict_GetItemWithError(ctx, d, k);
    int ret = HPy_IsNull(item);
    HPy_Close(ctx, item);
    return ret;
}

static NPY_INLINE HPy *
HPy_TupleToArray(HPyContext *ctx, HPy tuple, HPy_ssize_t *n)
{
    *n = HPy_Length(ctx, tuple);
    HPy *h_arr = MEM_RAWCALLOC(*n, sizeof(HPy));
    if (!h_arr)
        return NULL;
    HPy_ssize_t i;
    for (i = 0; i < *n; i++) {
        h_arr[i] = HPy_GetItem_i(ctx, tuple, i);
    }
    return h_arr;
}

static NPY_INLINE HPy
HPySequence_Tuple(HPyContext *ctx, HPy seq) {
    HPy_ssize_t len = HPy_Length(ctx, seq);
    HPyTupleBuilder tb = HPyTupleBuilder_New(ctx, len);
    for (HPy_ssize_t i = 0; i < len; i++) {
        HPy item = HPy_GetItem_i(ctx, seq, i);
        HPyTupleBuilder_Set(ctx, tb, i, item);
        HPy_Close(ctx, item);
    }
    return HPyTupleBuilder_Build(ctx, tb);
}

/*
    XXX: not sure if we can support this
*/
NPY_NO_EXPORT HPy
HPyLong_FromVoidPtr(HPyContext *ctx, void *p);


NPY_NO_EXPORT int
HPyFloat_CheckExact(HPyContext *ctx, HPy obj);

NPY_NO_EXPORT int
HPyComplex_CheckExact(HPyContext *ctx, HPy obj);

NPY_NO_EXPORT HPy_ssize_t
HPyNumber_AsSsize_t(HPyContext *ctx, HPy item, HPy err);

// Common patterns helper functions

/*
 * returns:
 * 1 if success
 * 0 if error occurred
 */
static NPY_INLINE int
HPy_ExtractDictItems_OiO(HPyContext *ctx, HPy value, HPy *v1, int *v2, HPy *v3) {

    // HPy_ssize_t value_len;
    // HPy *value_arr = HPy_TupleToArray(ctx, value, &value_len);
    // if (!HPyArg_Parse(ctx, NULL, value_arr, value_len, "Oi|O", &v1, &v2, &v3)) {
    //     HPy_CloseAndFreeArray(ctx, value_arr, value_len);
    //     return;
    // }

    // process the items wihtout extra calls and preprocess
    HPy_ssize_t len = HPy_Length(ctx, value);
    if (len < 2) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
            "required positional argument missing");
        return 0;
    } else if (len > 3) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
            "mismatched args (too many arguments for fmt)");
        return 0;
    }

    HPy h_v2 = HPy_GetItem_i(ctx, value, 1);
    long v = HPyLong_AsLong(ctx, h_v2);
    if (v == -1 && HPyErr_Occurred(ctx)) {
        return 0;
    }
    if (v > INT_MAX) {
        HPyErr_SetString(ctx, ctx->h_OverflowError,
            "signed integer is greater than maximum");
        return 0;
    }
    if (v < INT_MIN) {
        HPyErr_SetString(ctx, ctx->h_OverflowError,
            "signed integer is less than minimum");
        return 0;
    }
    *v2 = (int)v;
    *v1 = HPy_GetItem_i(ctx, value, 0);
    if (v3) {
        *v3 = (len > 2) ? HPy_GetItem_i(ctx, value, 2) : HPy_NULL;
    }
    return 1;
}

static NPY_INLINE double
HPyComplex_RealAsDouble(HPyContext *ctx, HPy obj) {
    HPy real = HPy_GetAttr_s(ctx, obj, "real");
    double val = HPyFloat_AsDouble(ctx, real);
    HPy_Close(ctx, real);
    return val;
}

static NPY_INLINE double
HPyComplex_ImagAsDouble(HPyContext *ctx, HPy obj) {
    HPy imag = HPy_GetAttr_s(ctx, obj, "imag");
    double val = HPyFloat_AsDouble(ctx, imag);
    HPy_Close(ctx, imag);
    return val;
}

static NPY_INLINE HPy
HPySequence_Fast(HPyContext *ctx, HPy v, const char *m)
{
    if (HPy_IsNull(v)) {
        if (!HPyErr_Occurred(ctx))
            HPyErr_SetString(ctx, ctx->h_SystemError,
                            "null argument to internal routine");
        return HPy_NULL;
    }
    HPy v_type = HPy_Type(ctx, v);
    if (HPy_Is(ctx, v_type, ctx->h_TupleType) || HPy_Is(ctx, v_type, ctx->h_ListType)) {
        HPy_Close(ctx, v_type);
        return HPy_Dup(ctx, v);
    }
    HPy_Close(ctx, v_type);

    CAPI_WARN("missing PyObject_GetIter & PySequence_List");
    PyObject *py_v = HPy_AsPyObject(ctx, v);
    PyObject *it = PyObject_GetIter(py_v);
    if (it == NULL) {
        Py_DECREF(py_v);
        if (HPyErr_ExceptionMatches(ctx, ctx->h_TypeError))
            HPyErr_SetString(ctx, ctx->h_TypeError, m);
        return HPy_NULL;
    }

    PyObject *l = PySequence_List(it);
    HPy res = HPy_FromPyObject(ctx, l);
    Py_XDECREF(py_v);
    Py_XDECREF(it);
    Py_XDECREF(l);

    return res;
}

/**
 * Delegates to HPy_DelItem but just to keep track of where 'PyDict_DelItem'
 * was used.
 */
static NPY_INLINE int
HPyDict_DelItem(HPyContext *ctx, HPy mp, HPy key)
{
    return HPy_DelItem(ctx, mp, key);
}

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_HPY_UTILS_H_ */

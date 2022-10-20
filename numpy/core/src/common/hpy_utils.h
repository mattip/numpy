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
    #define MEM_FREE PyMem_Free
#else
    #define OBJECT_MALLOC malloc
    #define OBJECT_FREE free
    #define MEM_MALLOC malloc
    #define MEM_CALLOC calloc
    #define MEM_FREE free
#endif

#ifdef GRAALVM_PYTHON
#define HPyMem_RawCalloc calloc
#else
#define HPyMem_RawCalloc PyMem_RawCalloc
#endif

#define HPy_SETREF(ctx, op, op2) \
    do {                         \
        HPy _h_tmp = (op);       \
        (op) = (op2);            \
        HPy_Close(ctx, _h_tmp);  \
    } while (0)

/* Set an error with a format string; it will use 'vsnprintf' for formatting. */
NPY_NO_EXPORT void
HPyErr_Format_p(HPyContext *ctx, HPy h_type, const char *fmt, ...);

NPY_NO_EXPORT int
HPyGlobal_Is(HPyContext *ctx, HPy obj, HPyGlobal expected);

NPY_NO_EXPORT int
HPyGlobal_TypeCheck(HPyContext *ctx, HPy obj, HPyGlobal type);

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

static NPY_INLINE HPy *
HPy_TupleToArray(HPyContext *ctx, HPy tuple, HPy_ssize_t *n)
{
    *n = HPy_Length(ctx, tuple);
    HPy *h_arr = HPyMem_RawCalloc(*n, sizeof(HPy));
    if (!h_arr)
        return NULL;
    HPy_ssize_t i;
    for (i = 0; i < *n; i++) {
        h_arr[i] = HPy_GetItem_i(ctx, tuple, i);
    }
    return h_arr;
}


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
        HPy_Close(ctx, value);
        return 0;
    } else if (len > 3) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
            "mismatched args (too many arguments for fmt)");
        HPy_Close(ctx, value);
        return 0;
    }

    HPy h_v2 = HPy_GetItem_i(ctx, value, 1);
    long v = HPyLong_AsLong(ctx, h_v2);
    if (v == -1 && HPyErr_Occurred(ctx)) {
        HPy_Close(ctx, value);
        return 0;
    }
    if (v > INT_MAX) {
        HPyErr_SetString(ctx, ctx->h_OverflowError,
            "signed integer is greater than maximum");
        HPy_Close(ctx, value);
        return 0;
    }
    if (v < INT_MIN) {
        HPyErr_SetString(ctx, ctx->h_OverflowError,
            "signed integer is less than minimum");
        HPy_Close(ctx, value);
        return 0;
    }
    *v2 = (int)v;
    *v1 = HPy_GetItem_i(ctx, value, 0);
    if (v3) {
        *v3 = (len > 2) ? HPy_GetItem_i(ctx, value, 2) : HPy_NULL;
    }
    HPy_Close(ctx, value);
    return 1;
}

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_HPY_UTILS_H_ */

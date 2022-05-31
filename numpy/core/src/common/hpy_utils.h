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
HPy_CloseAndFreeArray(HPyContext *ctx, HPy *h_arr, HPy_ssize_t n)
{
    if (!h_arr)
        return;
    HPy_ssize_t i;
    for (i = 0; i < n; i++) {
        HPy_Close(ctx, h_arr[i]);
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

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_HPY_UTILS_H_ */

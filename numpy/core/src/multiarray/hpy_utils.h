#ifndef NUMPY_CORE_SRC_MULTIARRAY_HPY_UTILS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_HPY_UTILS_H_

#include "numpy/ndarraytypes.h"

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

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_HPY_UTILS_H_ */

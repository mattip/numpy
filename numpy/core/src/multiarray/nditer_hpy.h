/*
 * This is a PRIVATE INTERNAL NumPy header, intended to be used *ONLY*
 * by the iterator implementation code. All other internal NumPy code
 * should use the exposed iterator API.
 */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_NDITER_HPY_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NDITER_HPY_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"

/* Numpy NpyIter HPy API */
NPY_NO_EXPORT int
HNpyIter_Reset(HPyContext *ctx, NpyIter *iter, char **errmsg);
NPY_NO_EXPORT HPy *
HNpyIter_GetDescrArray(NpyIter *iter);
NPY_NO_EXPORT int
HNpyIter_ResetBasePointers(HPyContext *ctx, NpyIter *iter, char **baseptrs, char **errmsg);
NPY_NO_EXPORT void
HNpyIter_GetInnerFixedStrideArray(HPyContext *ctx, NpyIter *iter, npy_intp *out_strides);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NDITER_HPY_H_ */

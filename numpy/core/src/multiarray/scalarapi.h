/*
 * This is a PRIVATE INTERNAL NumPy header, intended to be used *ONLY*
 * by the iterator implementation code. All other internal NumPy code
 * should use the exposed iterator API.
 */
#ifndef NPY_SCALARAPI_H
#define NPY_SCALARAPI_H

#include "numpy/arrayobject.h"

NPY_NO_EXPORT HPy
HPyArray_DescrFromTypeObject(HPyContext *ctx, HPy type);

#endif  /* NPY_SCALARAPI_H */

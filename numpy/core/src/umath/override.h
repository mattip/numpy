#ifndef _NPY_UMATH_OVERRIDE_H
#define _NPY_UMATH_OVERRIDE_H

#include "npy_config.h"
#include "numpy/ufuncobject.h"

NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
        PyObject *in_args, PyObject *out_args,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject **result);

NPY_NO_EXPORT int
HPyUFunc_CheckOverride(HPyContext *ctx, HPy ufunc, char *method,
        HPy in_args, HPy out_args,
        HPy const *args, HPy_ssize_t len_args, HPy kw,
        HPy *result);

#endif

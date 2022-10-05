#ifndef NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_

#include "numpy/ndarraytypes.h"

NPY_NO_EXPORT PyObject *
arr_bincount(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_interp(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_interp_complex(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_ravel_multi_index(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_unravel_index(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
io_pack(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
io_unpack(PyObject *, PyObject *, PyObject *);

extern NPY_NO_EXPORT HPyDef hpy_add_docstring;
extern NPY_NO_EXPORT HPyDef _insert;
extern NPY_NO_EXPORT HPyDef _monotonicity;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_ */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_

#include "numpy/ndarraytypes.h"

NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *, PyObject *);

extern NPY_NO_EXPORT HPyDef hpy_add_docstring;
extern NPY_NO_EXPORT HPyDef _insert;
extern NPY_NO_EXPORT HPyDef _monotonicity;
extern NPY_NO_EXPORT HPyDef arr_bincount;
extern NPY_NO_EXPORT HPyDef arr_ravel_multi_index;
extern NPY_NO_EXPORT HPyDef arr_unravel_index;
extern NPY_NO_EXPORT HPyDef io_pack;
extern NPY_NO_EXPORT HPyDef io_unpack;
extern NPY_NO_EXPORT HPyDef arr_interp;
extern NPY_NO_EXPORT HPyDef arr_interp_complex;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_COMPILED_BASE_H_ */

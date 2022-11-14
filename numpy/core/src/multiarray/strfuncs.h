#ifndef NUMPY_CORE_SRC_MULTIARRAY_STRFUNCS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_STRFUNCS_H_

NPY_NO_EXPORT void
PyArray_SetStringFunction(PyObject *op, int repr);

extern NPY_NO_EXPORT HPyDef array_repr_slot;

NPY_NO_EXPORT PyObject *
array_str(PyArrayObject *self);

NPY_NO_EXPORT PyObject *
array_format(PyArrayObject *self, PyObject *args);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_STRFUNCS_H_ */

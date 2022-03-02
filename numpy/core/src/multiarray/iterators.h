#ifndef NUMPY_CORE_SRC_MULTIARRAY_ITERATORS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ITERATORS_H_

extern NPY_NO_EXPORT HPyType_Spec PyArrayIter_Type_Spec;

NPY_NO_EXPORT PyObject
*iter_subscript(PyArrayIterObject *, PyObject *);

NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *, PyObject *, PyObject *);

NPY_NO_EXPORT void
PyArray_RawIterBaseInit(PyArrayIterObject *it, PyArrayObject *ao);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ITERATORS_H_ */

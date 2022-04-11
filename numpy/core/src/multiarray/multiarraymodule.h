#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

extern NPY_NO_EXPORT PyTypeObject* _PyArray_Type_p;

NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_array_finalize;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_implementation;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_axis1;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_axis2;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_like;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_ma_str_numpy;

NPY_NO_EXPORT unsigned char
HPyArray_EquivTypes(HPyContext *ctx, /*PyArray_Descr*/HPy type1, /*PyArray_Descr*/HPy type2);

NPY_NO_EXPORT double
HPyArray_GetPriority(HPyContext *ctx, HPy obj, double default_);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

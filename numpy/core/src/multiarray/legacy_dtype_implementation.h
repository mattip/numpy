#ifndef NUMPY_CORE_SRC_MULTIARRAY_LEGACY_DTYPE_IMPLEMENTATION_H_
#define NUMPY_CORE_SRC_MULTIARRAY_LEGACY_DTYPE_IMPLEMENTATION_H_

NPY_NO_EXPORT npy_bool
PyArray_LegacyCanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting);
NPY_NO_EXPORT npy_bool
HPyArray_LegacyCanCastTypeTo(HPyContext *ctx, 
        HPy /* PyArray_Descr * */ from, 
        HPy /* PyArray_Descr * */ to,
        NPY_CASTING casting);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_LEGACY_DTYPE_IMPLEMENTATION_H_ */

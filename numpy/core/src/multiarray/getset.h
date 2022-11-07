#ifndef NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_
#define NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_

extern NPY_NO_EXPORT PyGetSetDef array_getsetlist[];

extern HPyDef array_shape;
extern HPyDef array_descr_dtype;
extern HPyDef array_ndim_get;
extern HPyDef array_strides;
extern HPyDef array_priority_get;
extern HPyDef array_flags_get;
extern HPyDef array_ctypes_get;
extern HPyDef array_transpose_get;
extern HPyDef array_base_get;
extern HPyDef array_itemsize_get;
extern HPyDef array_size_get;
extern HPyDef array_nbytes_get;
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_ */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_
#define NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_

extern NPY_NO_EXPORT PyGetSetDef array_getsetlist[];

extern HPyDef array_shape;
extern HPyDef array_descr_dtype;
extern HPyDef array_ndim;
extern HPyDef array_strides;
extern HPyDef array_priority;
extern HPyDef array_flags;
extern HPyDef array_ctypes;
extern HPyDef array_T;
extern HPyDef array_base;
extern HPyDef array_itemsize;
extern HPyDef array_size;
extern HPyDef array_nbytes;
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_GETSET_H_ */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

extern NPY_NO_EXPORT PyTypeObject* _PyArray_Type_p;

extern NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_array_wrap;
extern NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_array_finalize;
extern NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_implementation;
extern NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_axis1;
extern NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_axis2;
extern NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_like;
extern NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_numpy;

NPY_NO_EXPORT unsigned char
HPyArray_EquivTypes(HPyContext *ctx, /*PyArray_Descr*/HPy type1, /*PyArray_Descr*/HPy type2);

NPY_NO_EXPORT double
HPyArray_GetPriority(HPyContext *ctx, HPy obj, double default_);

NPY_NO_EXPORT HPy
HPyArray_Where(HPyContext *ctx, HPy condition, HPy x, HPy y);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */

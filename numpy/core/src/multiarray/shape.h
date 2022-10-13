#ifndef NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_

/*
 * Creates a sorted stride perm matching the KEEPORDER behavior
 * of the NpyIter object. Because this operates based on multiple
 * input strides, the 'stride' member of the npy_stride_sort_item
 * would be useless and we simply argsort a list of indices instead.
 *
 * The caller should have already validated that 'ndim' matches for
 * every array in the arrays list.
 */
NPY_NO_EXPORT void
PyArray_CreateMultiSortedStridePerm(int narrays, PyArrayObject **arrays,
                        int ndim, int *out_strideperm);

NPY_NO_EXPORT void
HPyArray_CreateMultiSortedStridePerm(HPyContext *ctx, int narrays, 
                        HPy /* PyArrayObject ** */ *arrays,
                        int ndim, int *out_strideperm);

/*
 * Just like PyArray_Squeeze, but allows the caller to select
 * a subset of the size-one dimensions to squeeze out.
 */
NPY_NO_EXPORT PyObject *
PyArray_SqueezeSelected(PyArrayObject *self, npy_bool *axis_flags);

NPY_NO_EXPORT HPy
HPyArray_Ravel(HPyContext *ctx, /*PyArrayObject*/ HPy h_arr, NPY_ORDER order);

NPY_NO_EXPORT HPy
HPyArray_Transpose(HPyContext *ctx, HPy h_ap, PyArrayObject *ap, PyArray_Dims *permute);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_SHAPE_H_ */

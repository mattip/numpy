#ifndef NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_

extern NPY_NO_EXPORT HPyDef NpyIter_NestedIters;

NPY_NO_EXPORT HPy
NpyIter_NestedIters_impl(HPyContext *ctx, HPy NPY_UNUSED(self),
        HPy *args, HPy_ssize_t len_args, HPy kwds);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NDITER_PYWRAP_H_ */

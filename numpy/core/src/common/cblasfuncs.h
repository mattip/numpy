#ifndef NUMPY_CORE_SRC_COMMON_CBLASFUNCS_H_
#define NUMPY_CORE_SRC_COMMON_CBLASFUNCS_H_

NPY_NO_EXPORT PyObject *
cblas_matrixproduct(int, PyArrayObject *, PyArrayObject *, PyArrayObject *);

NPY_NO_EXPORT HPy
hpy_cblas_matrixproduct(HPyContext *ctx, int typenum, 
                    HPy /* PyArrayObject * */ ap1, PyArrayObject *ap1_struct,
                    HPy /* PyArrayObject * */ ap2, PyArrayObject *ap2_struct,
                    HPy /* PyArrayObject * */ out, PyArrayObject *out_struct);

#endif  /* NUMPY_CORE_SRC_COMMON_CBLASFUNCS_H_ */

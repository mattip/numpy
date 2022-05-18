#ifndef SCALAR_API_H
#define SCALAR_API_H

// Ad-hoc header so that we do not need to define new APIs for HPy porting
// Temporaty solution

NPY_NO_EXPORT HPy 
HPyArray_DescrFromTypeObject(HPyContext *ctx, HPy type);

NPY_NO_EXPORT int
HPyArray_CheckTypeAnyScalarExact(HPyContext *ctx, HPy type);

NPY_NO_EXPORT int
HPyArray_CheckAnyScalarExact(HPyContext *ctx, HPy obj);

NPY_NO_EXPORT HPy
HPyArray_Return(HPyContext *ctx, HPy mp);

NPY_NO_EXPORT HPy
HPyArray_DescrFromScalar(HPyContext *ctx, HPy sc);

NPY_NO_EXPORT HPy
HPyArray_Scalar(HPyContext *ctx, void *data, /*PyArray_Descr*/ HPy h_descr, HPy base, PyArrayObject *base_struct);

NPY_NO_EXPORT HPy
HPyArray_FromScalar(HPyContext *ctx, HPy h_scalar, /*PyArray_Descr*/ HPy h_outcode);

#endif // SCALAR_API_H

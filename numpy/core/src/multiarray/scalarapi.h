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

#endif // SCALAR_API_H

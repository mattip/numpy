#ifndef SCALAR_API_H
#define SCALAR_API_H

// Ad-hoc header so that we do not need to define new APIs for HPy porting
// Temporaty solution

NPY_NO_EXPORT HPy 
HPyArray_DescrFromTypeObject(HPyContext *ctx, HPy type);

#endif // SCALAR_API_H
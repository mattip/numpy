#ifndef FLAGSOBJECT_H
#define FLAGSOBJECT_H

NPY_NO_EXPORT void
HPyArray_UpdateFlags(HPyContext *ctx, HPy h_ret, PyArrayObject *ret, int flagmask);

#endif // FLAGSOBJECT_H
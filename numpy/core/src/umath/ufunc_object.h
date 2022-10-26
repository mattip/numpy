#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

#include <numpy/ufuncobject.h>
#include <hpy.h> // TODO HPY LABS PORT: move to numpy/npy_common.h

extern NPY_NO_EXPORT HPyGlobal HPyUFunc_Type;
extern NPY_NO_EXPORT HPyType_Spec PyUFunc_Type_Spec;


NPY_NO_EXPORT const char*
ufunc_get_name_cstr(PyUFuncObject *ufunc);

NPY_NO_EXPORT HPy
HPyUFunc_FromFuncAndDataAndSignatureAndIdentity(HPyContext *ctx, PyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     const char *name, const char *doc,
                                     const int unused, const char *signature,
                                     HPy identity_value);

/* strings from umathmodule.c that are interned on umath import */
NPY_VISIBILITY_HIDDEN extern HPyGlobal npy_hpy_um_str_array_prepare;
NPY_VISIBILITY_HIDDEN extern HPyGlobal npy_hpy_um_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern HPyGlobal npy_hpy_um_str_pyvals_name;

extern NPY_NO_EXPORT HPyDef ufunc_geterr;
extern NPY_NO_EXPORT HPyDef ufunc_seterr;

#endif

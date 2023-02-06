#ifndef NUMPY_CORE_SRC_MULTIARRAY_USERTYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_USERTYPES_H_

#include "array_method.h"

extern NPY_NO_EXPORT HPyGlobal *userdescrs;

NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f);

NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar);

NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_Descr *descr);

NPY_NO_EXPORT int
HPyArray_RegisterDataType(HPyContext* ctx, HPy h_descr);

NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc);

NPY_NO_EXPORT PyArray_DTypeMeta *
legacy_userdtype_common_dtype_function(
        PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other);

NPY_NO_EXPORT HPy // PyArray_DTypeMeta *
hpy_legacy_userdtype_common_dtype_function(HPyContext *ctx,
        HPy /* PyArray_DTypeMeta * */ cls, HPy /* PyArray_DTypeMeta * */ other);

NPY_NO_EXPORT int
PyArray_AddLegacyWrapping_CastingImpl(
        PyArray_DTypeMeta *from, PyArray_DTypeMeta *to, NPY_CASTING casting);

NPY_NO_EXPORT int
HPyArray_AddLegacyWrapping_CastingImpl(HPyContext *ctx,
        HPy h_from, HPy h_to, NPY_CASTING casting);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_USERTYPES_H_ */

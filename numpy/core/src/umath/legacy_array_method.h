#ifndef _NPY_LEGACY_ARRAY_METHOD_H
#define _NPY_LEGACY_ARRAY_METHOD_H

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "array_method.h"


NPY_NO_EXPORT PyArrayMethodObject *
PyArray_NewLegacyWrappingArrayMethod(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *signature[]);

NPY_NO_EXPORT HPy
HPyArray_NewLegacyWrappingArrayMethod(HPyContext *ctx, PyUFuncObject *ufunc,
        HPy signature[]);


/*
 * The following two symbols are in the header so that other places can use
 * them to probe for special cases (or whether an ArrayMethod is a "legacy"
 * one).
 */
NPY_NO_EXPORT int
get_wrapped_legacy_ufunc_loop(HPyContext *ctx, HPyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *NPY_UNUSED(strides),
        HPyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT NPY_CASTING
wrapped_legacy_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(method), /* (struct PyArrayMethodObject_tag *) */
        HPy *NPY_UNUSED(dtypes), /* PyArray_DTypeMeta **dtypes */
        HPy *NPY_UNUSED(given_descrs), /* PyArray_Descr **given_descrs */
        HPy *NPY_UNUSED(loop_descrs), /* PyArray_Descr **loop_descrs */
        npy_intp *NPY_UNUSED(view_offset));


#endif  /*_NPY_LEGACY_ARRAY_METHOD_H */

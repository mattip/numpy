#ifndef NUMPY_CORE_SRC_MULTIARRAY_TYPEINFO_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TYPEINFO_H_

#define PY_SSIZE_T_CLEAN
#include <hpy.h>
#include "npy_config.h"

NPY_VISIBILITY_HIDDEN int
typeinfo_init_structsequences(HPyContext *ctx, HPy multiarray_dict);

NPY_VISIBILITY_HIDDEN HPy
PyArray_typeinfo(HPyContext *ctx,
    char typechar, int typenum, int nbits, int align,
    HPy type_obj);

NPY_VISIBILITY_HIDDEN HPy
PyArray_typeinforanged(HPyContext *ctx,
    char typechar, int typenum, int nbits, int align,
    HPy max, HPy min, HPy type_obj);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TYPEINFO_H_ */

#ifndef NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_
#define NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_

#include "__umath_generated.c"
#include "__ufunc_api.c"

extern NPY_NO_EXPORT HPyDef get_sfloat_dtype;

extern NPY_NO_EXPORT HPyDef add_newdoc_ufunc;
extern NPY_NO_EXPORT HPyDef frompyfunc;
int initumath(HPyContext *ctx, HPy m, HPy module_dict);

#endif  /* NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_ */

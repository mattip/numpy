#ifndef NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_
#define NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_

#include "__umath_generated.c"
#include "__ufunc_api.c"

NPY_NO_EXPORT PyObject *
get_sfloat_dtype(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(args));

PyObject * add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args);
PyObject * ufunc_frompyfunc(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *NPY_UNUSED(kwds));
int initumath(HPyContext *ctx, PyObject *m, HPy module_dict);

#endif  /* NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_ */

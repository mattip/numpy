#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_

NPY_NO_EXPORT PyObject *
array_implement_array_function(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args);

NPY_NO_EXPORT PyObject *
array__get_implementing_args(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args);

NPY_NO_EXPORT PyObject *
array_implement_c_array_function_creation(
        const char *function_name, PyObject *like,
        PyObject *args, PyObject *kwargs,
        PyObject *const *fast_args, Py_ssize_t len_args, PyObject *kwnames);

NPY_NO_EXPORT HPy
hpy_array_implement_c_array_function_creation(HPyContext *ctx,
    const char *function_name, HPy like,
    HPy args, HPy kwargs,
    HPy *fast_args, Py_ssize_t len_args, HPy kwnames);

NPY_NO_EXPORT PyObject *
array_function_method_impl(PyObject *func, PyObject *types, PyObject *args,
                           PyObject *kwargs);


extern NPY_NO_EXPORT HPyDef implement_array_function;
extern NPY_NO_EXPORT HPyDef _get_implementing_args;
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYFUNCTION_OVERRIDE_H_ */

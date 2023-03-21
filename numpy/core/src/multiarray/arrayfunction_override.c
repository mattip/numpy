#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "npy_pycompat.h"
#include "get_attr_string.h"
#include "npy_import.h"
#include "multiarraymodule.h"

#include "hpy_utils.h"


/* Return the ndarray.__array_function__ method. */
static PyObject *
get_ndarray_array_function(void)
{
    PyObject* method = PyObject_GetAttrString((PyObject *)&PyArray_Type,
                                              "__array_function__");
    assert(method != NULL);
    return method;
}

static HPy
hpy_get_ndarray_array_function(HPyContext *ctx)
{
    HPy h_array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    HPy method = HPy_GetAttr_s(ctx, h_array_type,
                                              "__array_function__");
    HPy_Close(ctx, h_array_type);
    assert(!HPy_IsNull(method));
    return method;
}


/*
 * Get an object's __array_function__ method in the fastest way possible.
 * Never raises an exception. Returns NULL if the method doesn't exist.
 */
static PyObject *
get_array_function(PyObject *obj)
{
    static PyObject *ndarray_array_function = NULL;

    if (ndarray_array_function == NULL) {
        ndarray_array_function = get_ndarray_array_function();
    }

    /* Fast return for ndarray */
    if (PyArray_CheckExact(obj)) {
        Py_INCREF(ndarray_array_function);
        return ndarray_array_function;
    }

    PyObject *array_function = PyArray_LookupSpecial(obj, "__array_function__");
    if (array_function == NULL && PyErr_Occurred()) {
        PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
    }

    return array_function;
}

NPY_NO_EXPORT HPyGlobal hg_ndarray_array_function;
static bool hg_ndarray_array_function_initialized = false;

static HPy
hpy_get_array_function(HPyContext *ctx, HPy obj)
{
    HPy h_ndarray_array_function;
    /* Fast return for ndarray */
    if (HPyArray_CheckExact(ctx, obj)) {
        if (hg_ndarray_array_function_initialized) {
            h_ndarray_array_function = HPyGlobal_Load(ctx, hg_ndarray_array_function);
        } else {
            h_ndarray_array_function = hpy_get_ndarray_array_function(ctx);
            HPyGlobal_Store(ctx, &hg_ndarray_array_function, h_ndarray_array_function);
            hg_ndarray_array_function_initialized = true;
        }
        return h_ndarray_array_function;
    }
    HPy obj_type = HPy_Type(ctx, obj);
    HPy array_function = HPyArray_LookupSpecial_OnType(ctx, obj_type, "__array_function__");
    if (HPy_IsNull(array_function) && HPyErr_Occurred(ctx)) {
        HPyErr_Clear(ctx); /* TODO[gh-14801]: propagate crashes during attribute access? */
    }

    return array_function;
}


/*
 * Like list.insert(), but for C arrays of PyObject*. Skips error checking.
 */
static void
pyobject_array_insert(PyObject **array, int length, int index, PyObject *item)
{
    for (int j = length; j > index; j--) {
        array[j] = array[j - 1];
    }
    array[index] = item;
}

static void
hpy_array_insert(HPy *array, int length, int index, HPy item)
{
    for (int j = length; j > index; j--) {
        array[j] = array[j - 1];
    }
    array[index] = item;
}


/*
 * Collects arguments with __array_function__ and their corresponding methods
 * in the order in which they should be tried (i.e., skipping redundant types).
 * `relevant_args` is expected to have been produced by PySequence_Fast.
 * Returns the number of arguments, or -1 on failure. 
 */
static int
get_implementing_args_and_methods(PyObject *relevant_args,
                                  PyObject **implementing_args,
                                  PyObject **methods)
{
    int num_implementing_args = 0;

    PyObject **items = PySequence_Fast_ITEMS(relevant_args);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(relevant_args);

    for (Py_ssize_t i = 0; i < length; i++) {
        int new_class = 1;
        PyObject *argument = items[i];

        /* Have we seen this type before? */
        for (int j = 0; j < num_implementing_args; j++) {
            if (Py_TYPE(argument) == Py_TYPE(implementing_args[j])) {
                new_class = 0;
                break;
            }
        }
        if (new_class) {
            PyObject *method = get_array_function(argument);

            if (method != NULL) {
                int arg_index;

                if (num_implementing_args >= NPY_MAXARGS) {
                    PyErr_Format(
                        PyExc_TypeError,
                        "maximum number (%d) of distinct argument types " \
                        "implementing __array_function__ exceeded",
                        NPY_MAXARGS);
                    Py_DECREF(method);
                    goto fail;
                }

                /* "subclasses before superclasses, otherwise left to right" */
                arg_index = num_implementing_args;
                for (int j = 0; j < num_implementing_args; j++) {
                    PyObject *other_type;
                    other_type = (PyObject *)Py_TYPE(implementing_args[j]);
                    if (PyObject_IsInstance(argument, other_type)) {
                        arg_index = j;
                        break;
                    }
                }
                Py_INCREF(argument);
                pyobject_array_insert(implementing_args, num_implementing_args,
                                      arg_index, argument);
                pyobject_array_insert(methods, num_implementing_args,
                                      arg_index, method);
                ++num_implementing_args;
            }
        }
    }
    return num_implementing_args;

fail:
    for (int j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(methods[j]);
    }
    return -1;
}

static int
hpy_get_implementing_args_and_methods(HPyContext *ctx, HPy relevant_args,
                                  HPy *implementing_args,
                                  HPy *methods)
{
    int num_implementing_args = 0;

    HPy_ssize_t length = HPy_Length(ctx, relevant_args);

    for (HPy_ssize_t i = 0; i < length; i++) {
        int new_class = 1;
        HPy argument = HPy_GetItem_i(ctx, relevant_args, i);

        /* Have we seen this type before? */
        HPy argument_type = HPy_Type(ctx, argument);
        for (int j = 0; j < num_implementing_args; j++) {
            HPy implementing_arg_type = HPy_Type(ctx, implementing_args[j]);
            int is_equal = HPy_Is(ctx, argument_type, implementing_arg_type);
            HPy_Close(ctx, implementing_arg_type);
            if (is_equal) {
                new_class = 0;
                break;
            }
        }
        HPy_Close(ctx, argument_type);
        if (new_class) {
            HPy method = hpy_get_array_function(ctx, argument);

            if (!HPy_IsNull(method)) {
                int arg_index;

                if (num_implementing_args >= NPY_MAXARGS) {
                    // PyErr_Format(
                    //     PyExc_TypeError,
                    //     "maximum number (%d) of distinct argument types " \
                    //     "implementing __array_function__ exceeded",
                    //     NPY_MAXARGS);
                    HPyErr_SetString(ctx,
                        ctx->h_TypeError,
                        "maximum number (%d) of distinct argument types " \
                        "implementing __array_function__ exceeded");
                    HPy_Close(ctx, method);
                    goto fail;
                }

                /* "subclasses before superclasses, otherwise left to right" */
                arg_index = num_implementing_args;
                for (int j = 0; j < num_implementing_args; j++) {
                    HPy other_type = HPy_Type(ctx, implementing_args[j]);
                    int is_instance = HPy_IsInstance(ctx, argument, other_type);
                    HPy_Close(ctx, other_type);
                    if (is_instance) {
                        arg_index = j;
                        break;
                    }
                }
                hpy_array_insert(implementing_args, num_implementing_args,
                                      arg_index, HPy_Dup(ctx, argument));
                // HPy_Close(ctx, argument);
                hpy_array_insert(methods, num_implementing_args,
                                      arg_index, method);
                ++num_implementing_args;
            }
        }
    }
    return num_implementing_args;

fail:
    for (int j = 0; j < num_implementing_args; j++) {
        HPy_Close(ctx, implementing_args[j]);
        HPy_Close(ctx, methods[j]);
    }
    return -1;
}

/*
 * Is this object ndarray.__array_function__?
 */
static int
is_default_array_function(PyObject *obj)
{
    static PyObject *ndarray_array_function = NULL;

    if (ndarray_array_function == NULL) {
        ndarray_array_function = get_ndarray_array_function();
    }
    return obj == ndarray_array_function;
}

static int
hpy_is_default_array_function(HPyContext *ctx, HPy obj)
{
    HPy h_ndarray_array_function = HPyGlobal_Load(ctx, hg_ndarray_array_function);
    if (HPy_IsNull(h_ndarray_array_function)) {
        h_ndarray_array_function = hpy_get_ndarray_array_function(ctx);
        HPyGlobal_Store(ctx, &hg_ndarray_array_function, h_ndarray_array_function);
    }
    return HPy_Is(ctx, obj, h_ndarray_array_function);
}

/*
 * Core implementation of ndarray.__array_function__. This is exposed
 * separately so we can avoid the overhead of a Python method call from
 * within `implement_array_function`.
 */
NPY_NO_EXPORT PyObject *
array_function_method_impl(PyObject *func, PyObject *types, PyObject *args,
                           PyObject *kwargs)
{
    PyObject **items = PySequence_Fast_ITEMS(types);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(types);

    for (Py_ssize_t j = 0; j < length; j++) {
        int is_subclass = PyObject_IsSubclass(
            items[j], (PyObject *)&PyArray_Type);
        if (is_subclass == -1) {
            return NULL;
        }
        if (!is_subclass) {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
    }

    PyObject *tmp = HPyGlobal_LoadPyObj(npy_ma_str_implementation);
    PyObject *implementation = PyObject_GetAttr(func, tmp);
    Py_XDECREF(tmp);
    if (implementation == NULL) {
        return NULL;
    }
    PyObject *result = PyObject_Call(implementation, args, kwargs);
    Py_DECREF(implementation);
    return result;
}

NPY_NO_EXPORT HPy
hpy_array_function_method_impl(HPyContext *ctx, HPy func, HPy types, HPy args,
                           HPy kwargs)
{
    HPy_ssize_t length = HPy_Length(ctx, types);

    HPy h_array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    PyObject *py_array_type = HPy_AsPyObject(ctx, h_array_type);
    HPy_Close(ctx, h_array_type);
    for (HPy_ssize_t j = 0; j < length; j++) {
        HPy item = HPy_GetItem_i(ctx, types, j);
        PyObject *py_item = HPy_AsPyObject(ctx, item);
        CAPI_WARN("calling PyObject_IsSubclass");
        int is_subclass = PyObject_IsSubclass(
            py_item, py_array_type);
        HPy_Close(ctx, item);
        Py_DECREF(py_item);
        if (is_subclass == -1) {
            Py_DECREF(py_array_type);
            return HPy_NULL;
        }
        if (!is_subclass) {
            return HPy_Dup(ctx, ctx->h_NotImplemented);
        }
    }

    HPy tmp = HPyGlobal_Load(ctx, npy_ma_str_implementation);
    HPy implementation = HPy_GetAttr(ctx, func, tmp);
    HPy_Close(ctx, tmp);
    if (HPy_IsNull(implementation)) {
        return HPy_NULL;
    }
    HPy result = HPy_CallTupleDict(ctx, implementation, args, kwargs);
    HPy_Close(ctx, implementation);
    return result;
}


/*
 * Calls __array_function__ on the provided argument, with a fast-path for
 * ndarray.
 */
static PyObject *
call_array_function(PyObject* argument, PyObject* method,
                    PyObject* public_api, PyObject* types,
                    PyObject* args, PyObject* kwargs)
{
    if (is_default_array_function(method)) {
        return array_function_method_impl(public_api, types, args, kwargs);
    }
    else {
        return PyObject_CallFunctionObjArgs(
            method, argument, public_api, types, args, kwargs, NULL);
    }
}

static HPy
hpy_call_array_function(HPyContext *ctx, HPy argument, HPy method,
                    HPy public_api, HPy types,
                    HPy args, HPy kwargs)
{
    if (hpy_is_default_array_function(ctx, method)) {
        return hpy_array_function_method_impl(ctx, public_api, types, args, kwargs);
    }
    else {
        HPy args_tuple = HPyTuple_Pack(ctx, 5, argument, public_api, types, args, kwargs);
        HPy ret = HPy_CallTupleDict(ctx, method, args_tuple, HPy_NULL);
        HPy_Close(ctx, args_tuple);
        return ret;
    }
}


/**
 * Internal handler for the array-function dispatching. The helper returns
 * either the result, or NotImplemented (as a borrowed reference).
 *
 * @param public_api The public API symbol used for dispatching
 * @param relevant_args Arguments which may implement __array_function__
 * @param args Original arguments
 * @param kwargs Original keyword arguments
 *
 * @returns The result of the dispatched version, or a borrowed reference
 *          to NotImplemented to indicate the default implementation should
 *          be used.
 */
static PyObject *
array_implement_array_function_internal(
    PyObject *public_api, PyObject *relevant_args,
    PyObject *args, PyObject *kwargs)
{
    PyObject *implementing_args[NPY_MAXARGS];
    PyObject *array_function_methods[NPY_MAXARGS];
    PyObject *types = NULL;

    PyObject *result = NULL;

    static PyObject *errmsg_formatter = NULL;

    relevant_args = PySequence_Fast(
        relevant_args,
        "dispatcher for __array_function__ did not return an iterable");
    if (relevant_args == NULL) {
        return NULL;
    }

    /* Collect __array_function__ implementations */
    int num_implementing_args = get_implementing_args_and_methods(
        relevant_args, implementing_args, array_function_methods);
    if (num_implementing_args == -1) {
        goto cleanup;
    }

    /*
     * Handle the typical case of no overrides. This is merely an optimization
     * if some arguments are ndarray objects, but is also necessary if no
     * arguments implement __array_function__ at all (e.g., if they are all
     * built-in types).
     */
    int any_overrides = 0;
    for (int j = 0; j < num_implementing_args; j++) {
        if (!is_default_array_function(array_function_methods[j])) {
            any_overrides = 1;
            break;
        }
    }
    if (!any_overrides) {
        /*
         * When the default implementation should be called, return
         * `Py_NotImplemented` to indicate this.
         */
        result = Py_NotImplemented;
        goto cleanup;
    }

    /*
     * Create a Python object for types.
     * We use a tuple, because it's the fastest Python collection to create
     * and has the bonus of being immutable.
     */
    types = PyTuple_New(num_implementing_args);
    if (types == NULL) {
        goto cleanup;
    }
    for (int j = 0; j < num_implementing_args; j++) {
        PyObject *arg_type = (PyObject *)Py_TYPE(implementing_args[j]);
        Py_INCREF(arg_type);
        PyTuple_SET_ITEM(types, j, arg_type);
    }

    /* Call __array_function__ methods */
    for (int j = 0; j < num_implementing_args; j++) {
        PyObject *argument = implementing_args[j];
        PyObject *method = array_function_methods[j];

        /*
         * We use `public_api` instead of `implementation` here so
         * __array_function__ implementations can do equality/identity
         * comparisons.
         */
        result = call_array_function(
            argument, method, public_api, types, args, kwargs);

        if (result == Py_NotImplemented) {
            /* Try the next one */
            Py_DECREF(result);
            result = NULL;
        }
        else {
            /* Either a good result, or an exception was raised. */
            goto cleanup;
        }
    }

    /* No acceptable override found, raise TypeError. */
    npy_cache_import("numpy.core._internal",
                     "array_function_errmsg_formatter",
                     &errmsg_formatter);
    if (errmsg_formatter != NULL) {
        PyObject *errmsg = PyObject_CallFunctionObjArgs(
            errmsg_formatter, public_api, types, NULL);
        if (errmsg != NULL) {
            PyErr_SetObject(PyExc_TypeError, errmsg);
            Py_DECREF(errmsg);
        }
    }

cleanup:
    for (int j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(array_function_methods[j]);
    }
    Py_XDECREF(types);
    Py_DECREF(relevant_args);
    return result;
}

static HPy
hpy_array_implement_array_function_internal(HPyContext *ctx,
    HPy public_api, HPy relevant_args,
    HPy args, HPy kwargs)
{
    HPy implementing_args[NPY_MAXARGS];
    HPy array_function_methods[NPY_MAXARGS];
    HPy types = HPy_NULL;

    HPy result = HPy_NULL;

    // static PyObject *errmsg_formatter = NULL;
    HPy rel_arg_seq = HPySequence_Fast(ctx, relevant_args,
                "dispatcher for __array_function__ did not return an iterable");
    if (HPy_IsNull(rel_arg_seq)) {
        return HPy_NULL;
    }
    // relevant_args = PySequence_Fast(
    //     relevant_args,
    //     "dispatcher for __array_function__ did not return an iterable");
    // if (relevant_args == NULL) {
    //     return NULL;
    // }

    /* Collect __array_function__ implementations */
    int num_implementing_args = hpy_get_implementing_args_and_methods(ctx,
        rel_arg_seq, implementing_args, array_function_methods);
    HPy_Close(ctx, rel_arg_seq);
    if (num_implementing_args == -1) {
        goto cleanup;
    }

    /*
     * Handle the typical case of no overrides. This is merely an optimization
     * if some arguments are ndarray objects, but is also necessary if no
     * arguments implement __array_function__ at all (e.g., if they are all
     * built-in types).
     */
    int any_overrides = 0;
    for (int j = 0; j < num_implementing_args; j++) {
        if (!hpy_is_default_array_function(ctx, array_function_methods[j])) {
            any_overrides = 1;
            break;
        }
    }
    if (!any_overrides) {
        /*
         * When the default implementation should be called, return
         * `Py_NotImplemented` to indicate this.
         */
        result = ctx->h_NotImplemented;
        goto cleanup;
    }

    /*
     * Create a Python object for types.
     * We use a tuple, because it's the fastest Python collection to create
     * and has the bonus of being immutable.
     */
    HPyTupleBuilder tb_types = HPyTupleBuilder_New(ctx, num_implementing_args);
    if (HPyTupleBuilder_IsNull(tb_types)) {
        goto cleanup;
    }
    for (int j = 0; j < num_implementing_args; j++) {
        HPy arg_type = HPy_Type(ctx, implementing_args[j]);
        HPyTupleBuilder_Set(ctx, tb_types, j, arg_type);
        HPy_Close(ctx, arg_type);
    }

    types = HPyTupleBuilder_Build(ctx, tb_types);

    /* Call __array_function__ methods */
    for (int j = 0; j < num_implementing_args; j++) {
        HPy argument = implementing_args[j];
        HPy method = array_function_methods[j];

        /*
         * We use `public_api` instead of `implementation` here so
         * __array_function__ implementations can do equality/identity
         * comparisons.
         */
        result = hpy_call_array_function(ctx,
            argument, method, public_api, types, args, kwargs);

        if (HPy_Is(ctx, result, ctx->h_NotImplemented)) {
            /* Try the next one */
            HPy_Close(ctx, result);
            result = HPy_NULL;
        }
        else {
            /* Either a good result, or an exception was raised. */
            goto cleanup;
        }
    }

    /* No acceptable override found, raise TypeError. */
    npy_hpy_cache_import(ctx, "numpy.core._internal",
                     "array_function_errmsg_formatter",
                     NULL);
    CAPI_WARN("missing errmsg_formatter");
    // if (errmsg_formatter != NULL) {
    //     PyObject *errmsg = PyObject_CallFunctionObjArgs(
    //         errmsg_formatter, public_api, types, NULL);
    //     if (errmsg != NULL) {
    //         PyErr_SetObject(PyExc_TypeError, errmsg);
    //         Py_DECREF(errmsg);
    //     }
    // }

cleanup:
    for (int j = 0; j < num_implementing_args; j++) {
        HPy_Close(ctx, implementing_args[j]);
        HPy_Close(ctx, array_function_methods[j]);
    }
    HPy_Close(ctx, types);
    return result;
}

/*
 * Implements the __array_function__ protocol for a Python function, as described in
 * in NEP-18. See numpy.core.overrides for a full docstring.
 */
HPyDef_METH(implement_array_function, "implement_array_function", HPyFunc_VARARGS)
static HPy
implement_array_function_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs)
{
    // if (!PyArg_UnpackTuple(
    //         positional_args, "implement_array_function", 5, 5,
    //         &implementation, &public_api, &relevant_args, &args, &kwargs)) {
    //     return NULL;
    // }
    if (nargs != 5) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "implement_array_function.. TODO");
        return HPy_NULL;
    }
    HPy implementation = args[0];
    HPy public_api = args[1];
    HPy relevant_args = args[2];
    HPy h_args = args[3];
    HPy kwargs = args[4];
    /*
     * Remove `like=` kwarg, which is NumPy-exclusive and thus not present
     * in downstream libraries. If `like=` is specified but doesn't
     * implement `__array_function__`, raise a `TypeError`.
     */
    HPy tmp = HPyGlobal_Load(ctx, npy_ma_str_axis1);
    if (!HPy_IsNull(kwargs) && HPy_Contains(ctx, kwargs, tmp)) {
        HPy like_arg = HPy_GetItem(ctx, kwargs, tmp);
        if (!HPy_IsNull(like_arg)) {
            HPy tmp_has_override = hpy_get_array_function(ctx, like_arg);
            if (HPy_IsNull(tmp_has_override)) {
                HPy_Close(ctx, tmp);
                HPyErr_SetString(ctx, ctx->h_TypeError,
                        "The `like` argument must be an array-like that "
                        "implements the `__array_function__` protocol.");
                return HPy_NULL;
            }
            HPy_Close(ctx, tmp_has_override);
            HPyDict_DelItem(ctx, kwargs, tmp);
        }
    }
    HPy_Close(ctx, tmp);

    HPy res = hpy_array_implement_array_function_internal(ctx,
        public_api, relevant_args, h_args, kwargs);

    if (HPy_Is(ctx, res, ctx->h_NotImplemented)) {
        return HPy_CallTupleDict(ctx, implementation, h_args, kwargs);
    }
    return res;
}

/*
 * Implements the __array_function__ protocol for C array creation functions
 * only. Added as an extension to NEP-18 in an effort to bring NEP-35 to
 * life with minimal dispatch overhead.
 *
 * The caller must ensure that `like != NULL`.
 */
NPY_NO_EXPORT PyObject *
array_implement_c_array_function_creation(
    const char *function_name, PyObject *like,
    PyObject *args, PyObject *kwargs,
    PyObject *const *fast_args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *relevant_args = NULL;
    PyObject *numpy_module = NULL;
    PyObject *public_api = NULL;
    PyObject *result = NULL;

    /* If `like` doesn't implement `__array_function__`, raise a `TypeError` */
    PyObject *tmp_has_override = get_array_function(like);
    if (tmp_has_override == NULL) {
        return PyErr_Format(PyExc_TypeError,
                "The `like` argument must be an array-like that "
                "implements the `__array_function__` protocol.");
    }
    Py_DECREF(tmp_has_override);

    if (fast_args != NULL) {
        /*
         * Convert from vectorcall convention, since the protocol requires
         * the normal convention.  We have to do this late to ensure the
         * normal path where NotImplemented is returned is fast.
         */
        assert(args == NULL);
        assert(kwargs == NULL);
        args = PyTuple_New(len_args);
        if (args == NULL) {
            return NULL;
        }
        for (Py_ssize_t i = 0; i < len_args; i++) {
            Py_INCREF(fast_args[i]);
            PyTuple_SET_ITEM(args, i, fast_args[i]);
        }
        if (kwnames != NULL) {
            kwargs = PyDict_New();
            if (kwargs == NULL) {
                Py_DECREF(args);
                return NULL;
            }
            Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
            for (Py_ssize_t i = 0; i < nkwargs; i++) {
                PyObject *key = PyTuple_GET_ITEM(kwnames, i);
                PyObject *value = fast_args[i+len_args];
                if (PyDict_SetItem(kwargs, key, value) < 0) {
                    Py_DECREF(args);
                    Py_DECREF(kwargs);
                    return NULL;
                }
            }
        }
    }

    relevant_args = PyTuple_Pack(1, like);
    if (relevant_args == NULL) {
        goto finish;
    }
    /* The like argument must be present in the keyword arguments, remove it */
    PyObject *tmp = HPyGlobal_LoadPyObj(npy_ma_str_like);
    if (PyDict_DelItem(kwargs, tmp) < 0) {
        Py_XDECREF(tmp);
        goto finish;
    }
    Py_XDECREF(tmp);

    tmp = HPyGlobal_LoadPyObj(npy_ma_str_numpy);
    numpy_module = PyImport_Import(tmp);
    Py_XDECREF(tmp);
    if (numpy_module == NULL) {
        goto finish;
    }

    public_api = PyObject_GetAttrString(numpy_module, function_name);
    Py_DECREF(numpy_module);
    if (public_api == NULL) {
        goto finish;
    }
    if (!PyCallable_Check(public_api)) {
        PyErr_Format(PyExc_RuntimeError,
                "numpy.%s is not callable.", function_name);
        goto finish;
    }

    result = array_implement_array_function_internal(
            public_api, relevant_args, args, kwargs);

  finish:
    if (kwnames != NULL) {
        /* args and kwargs were converted from vectorcall convention */
        Py_XDECREF(args);
        Py_XDECREF(kwargs);
    }
    Py_XDECREF(relevant_args);
    Py_XDECREF(public_api);
    return result;
}

NPY_NO_EXPORT HPy
hpy_array_implement_c_array_function_creation(HPyContext *ctx,
    const char *function_name, HPy like,
    const HPy *fast_args, size_t len_args, HPy kwnames)
{
    HPy relevant_args = HPy_NULL;
    PyObject *numpy_module = NULL;
    HPy public_api = HPy_NULL;
    HPy result = HPy_NULL;

    /* If `like` doesn't implement `__array_function__`, raise a `TypeError` */
    HPy tmp_has_override = hpy_get_array_function(ctx, like);
    if (HPy_IsNull(tmp_has_override)) {
        return HPyErr_SetString(ctx, ctx->h_TypeError,
                "The `like` argument must be an array-like that "
                "implements the `__array_function__` protocol.");
    }
    HPy_Close(ctx, tmp_has_override);

    HPy args, kwargs;
    if (fast_args != NULL) {
        /*
         * Convert from vectorcall convention, since the protocol requires
         * the normal convention.  We have to do this late to ensure the
         * normal path where NotImplemented is returned is fast.
         */
        if (!HPyHelpers_PackArgsAndKeywords(ctx, fast_args, len_args, kwnames, &args, &kwargs))
            goto finish;
    } else {
        args = HPy_NULL;
        kwargs = HPy_NULL;
    }

    relevant_args = HPyTuple_Pack(ctx, 1, like);
    if (HPy_IsNull(relevant_args)) {
        goto finish;
    }
    /* The like argument must be present in the keyword arguments, remove it */
    CAPI_WARN("calling PyImport_Import & PyDict_DelItem");
    PyObject *py_kwargs = HPy_AsPyObject(ctx, kwargs);
    PyObject *tmp = HPyGlobal_LoadPyObj(npy_ma_str_like);
    if (PyDict_DelItem(py_kwargs, tmp) < 0) {
        Py_DECREF(py_kwargs);
        Py_XDECREF(tmp);
        goto finish;
    }
    Py_DECREF(py_kwargs);
    Py_XDECREF(tmp);

    tmp = HPyGlobal_LoadPyObj(npy_ma_str_numpy);
    numpy_module = PyImport_Import(tmp);
    Py_XDECREF(tmp);
    if (numpy_module == NULL) {
        goto finish;
    }
    HPy h_numpy_module = HPy_FromPyObject(ctx, numpy_module);

    public_api = HPy_GetAttr_s(ctx, h_numpy_module, function_name);
    HPy_Close(ctx, h_numpy_module);
    if (HPy_IsNull(public_api)) {
        goto finish;
    }
    if (!HPyCallable_Check(ctx, public_api)) {
        // PyErr_Format(PyExc_RuntimeError,
        //         "numpy.%s is not callable.", function_name);
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "numpy.%s is not callable.");
        goto finish;
    }

    result = hpy_array_implement_array_function_internal(ctx, 
            public_api, relevant_args, args, kwargs);

  finish:
    // HPy: see comments above
    // if (!HPy_IsNull(kwnames)) {
    //     /* args and kwargs were converted from vectorcall convention */
    //     HPy_Close(ctx, args);
    //     HPy_Close(ctx, kwargs);
    // }
    HPy_Close(ctx, relevant_args);
    HPy_Close(ctx, public_api);
    return result;
}

/*
 * Python wrapper for get_implementing_args_and_methods, for testing purposes.
 */
HPyDef_METH(_get_implementing_args, "_get_implementing_args", HPyFunc_VARARGS)
NPY_NO_EXPORT HPy
_get_implementing_args_impl(
    HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs)
{
    HPy relevant_args;
    HPy implementing_args[NPY_MAXARGS];
    HPy array_function_methods[NPY_MAXARGS];
    HPy result = HPy_NULL;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O:array__get_implementing_args",
                          &relevant_args)) {
        return HPy_NULL;
    }

    HPy rel_arg_seq = HPySequence_Fast(ctx, relevant_args,
                "dispatcher for __array_function__ did not return an iterable");
    if (HPy_IsNull(rel_arg_seq)) {
        return HPy_NULL;
    }
    // relevant_args = PySequence_Fast(
    //     relevant_args,
    //     "dispatcher for __array_function__ did not return an iterable");
    // if (relevant_args == NULL) {
    //     return NULL;
    // }

    int num_implementing_args = hpy_get_implementing_args_and_methods(ctx,
        rel_arg_seq, implementing_args, array_function_methods);
    HPy_Close(ctx, rel_arg_seq);
    if (num_implementing_args == -1) {
        goto cleanup;
    }

    /* create a Python object for implementing_args */
    HPyListBuilder resultlist = HPyListBuilder_New(ctx, num_implementing_args);
    if (HPyListBuilder_IsNull(resultlist)) {
        goto cleanup;
    }
    for (int j = 0; j < num_implementing_args; j++) {
        // Py_INCREF(implementing_args[j]); we are incrementing as we set.
        HPyListBuilder_Set(ctx, resultlist, j, implementing_args[j]);
    }
    result = HPyListBuilder_Build(ctx, resultlist);

cleanup:
    for (int j = 0; j < num_implementing_args; j++) {
        HPy_Close(ctx, implementing_args[j]);
        HPy_Close(ctx, array_function_methods[j]);
    }
    return result;
}

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY

#include "npy_pycompat.h"
#include "numpy/ufuncobject.h"
#include "npy_import.h"

#include "override.h"
#include "ufunc_override.h"
#include "hpy_utils.h"


/*
 * For each positional argument and each argument in a possible "out"
 * keyword, look for overrides of the standard ufunc behaviour, i.e.,
 * non-default __array_ufunc__ methods.
 *
 * Returns the number of overrides, setting corresponding objects
 * in PyObject array ``with_override`` and the corresponding
 * __array_ufunc__ methods in ``methods`` (both using new references).
 *
 * Only the first override for a given class is returned.
 *
 * Returns -1 on failure.
 */
static int
hget_array_ufunc_overrides(HPyContext *ctx, HPy in_args, HPy out_args,
                          HPy *with_override, HPy *methods)
{
    int i;
    int num_override_args = 0;
    int narg, nout;

    narg = (int)HPy_Length(ctx, in_args);
    /* It is valid for out_args to be NULL: */
    nout = (!HPy_IsNull(out_args)) ? (int)HPy_Length(ctx, out_args) : 0;

    for (i = 0; i < narg + nout; ++i) {
        HPy obj = HPy_NULL;
        int j;
        int new_class = 1;

        if (i < narg) {
            obj = HPy_GetItem_i(ctx, in_args, i);
        }
        else {
            obj = HPy_GetItem_i(ctx, out_args, i - narg);
        }
        /*
         * Have we seen this class before?  If so, ignore.
         */
        for (j = 0; j < num_override_args; j++) {
            HPy obj_type = HPy_Type(ctx, obj);
            HPy wo_type = HPy_Type(ctx, with_override[j]);
            new_class = !HPy_Is(ctx, obj_type, wo_type);
            HPy_Close(ctx, obj_type);
            HPy_Close(ctx, wo_type);
            if (!new_class) {
                HPy_Close(ctx, obj);
                break;
            }
        }
        if (new_class) {
            /*
             * Now see if the object provides an __array_ufunc__. However, we should
             * ignore the base ndarray.__ufunc__, so we skip any ndarray as well as
             * any ndarray subclass instances that did not override __array_ufunc__.
             */
            HPy method = HPyUFuncOverride_GetNonDefaultArrayUfunc(ctx, obj);
            if (HPy_IsNull(method)) {
                HPy_Close(ctx, obj);
                continue;
            }
            if (HPy_Is(ctx, method, ctx->h_None)) {
                // TODO HPY LABS PORT: PyErr_Format
                // PyErr_Format(ctx, ctx->h_TypeError,
                //             "operand '%.200s' does not support ufuncs "
                //             "(__array_ufunc__=None)",
                //             obj->ob_type->tp_name);
                HPyErr_SetString(ctx, ctx->h_TypeError,
                             "operand '%.200s' does not support ufuncs "
                             "(__array_ufunc__=None)");
                HPy_Close(ctx, method);
                HPy_Close(ctx, obj);
                goto fail;
            }
            with_override[num_override_args] = obj;
            methods[num_override_args] = method;
            ++num_override_args;
        }
    }
    return num_override_args;

fail:
    for (i = 0; i < num_override_args; i++) {
        HPy_Close(ctx, with_override[i]);
        HPy_Close(ctx, methods[i]);
    }
    return -1;
}


/*
 * Build a dictionary from the keyword arguments, but replace out with the
 * normalized version (and always pass it even if it was passed by position).
 */
static int
initialize_normal_kwds(HPyContext *ctx, HPy out_args,
        const HPy *args, HPy_ssize_t len_args, HPy kwnames,
        HPy normal_kwds)
{
    if (!HPy_IsNull(kwnames)) {
        for (Py_ssize_t i = 0; i < HPy_Length(ctx, kwnames); i++) {
            HPy item = HPy_GetItem_i(ctx, kwnames, i);
            if (HPy_SetItem(ctx, normal_kwds,
                    item, args[i + len_args]) < 0) {
                HPy_Close(ctx, item);
                return -1;
            }
            HPy_Close(ctx, item);
        }
    }
    static HPyGlobal hg_out_str;
    static int is_out_str_set = 0;
    HPy out_str = HPyGlobal_Load(ctx, hg_out_str);
    if (!is_out_str_set || HPy_IsNull(out_str)) {
        out_str = HPyUnicode_InternFromString(ctx, "out");
        if (HPy_IsNull(out_str)) {
            return -1;
        }
        HPyGlobal_Store(ctx, &hg_out_str, out_str);
        is_out_str_set = 1;
    }

    if (!HPy_IsNull(out_args)) {
        /* Replace `out` argument with the normalized version */
        int res = HPy_SetItem(ctx, normal_kwds, out_str, out_args);
        if (res < 0) {
            return -1;
        }
    }
    else {
        /* Ensure that `out` is not present. */
        int res = HPy_Contains(ctx, normal_kwds, out_str);
        if (res < 0) {
            return -1;
        }
        if (res) {
            CAPI_WARN("missing PyDict_DelItem");
            PyObject *py_normal_kwds = HPy_AsPyObject(ctx, normal_kwds);
            PyObject *py_out_str = HPy_AsPyObject(ctx, out_str);
            int res = PyDict_DelItem(py_normal_kwds, py_out_str);
            Py_DECREF(py_normal_kwds);
            Py_DECREF(py_out_str);
            return res;
        }
    }
    return 0;
}

/*
 * ufunc() and ufunc.outer() accept 'sig' or 'signature'.  We guarantee
 * that it is passed as 'signature' by renaming 'sig' if present.
 * Note that we have already validated that only one of them was passed
 * before checking for overrides.
 */
static int
normalize_signature_keyword(HPyContext *ctx, HPy normal_kwds)
{
    /* If the keywords include `sig` rename to `signature`. */
    HPy sig = HPyUnicode_FromString(ctx, "sig");
    HPy obj = HPyDict_GetItemWithError(ctx, normal_kwds, sig);
    if (HPy_IsNull(obj) && HPyErr_Occurred(ctx)) {
        return -1;
    }
    if (!HPy_IsNull(obj)) {
        if (HPy_SetItem_s(ctx, normal_kwds, "signature", obj) < 0) {
            HPy_Close(ctx, sig);
            HPy_Close(ctx, obj);
            return -1;
        }
        HPy_Close(ctx, sig);
        HPy_Close(ctx, obj);
        CAPI_WARN("missing PyDict_DelItemString");
        PyObject *py_normal_kwds = HPy_AsPyObject(ctx, normal_kwds);
        if (PyDict_DelItemString(py_normal_kwds, "sig") < 0) {
            Py_DECREF(py_normal_kwds);
            return -1;
        }
        Py_DECREF(py_normal_kwds);
    }
    return 0;
}


static int
copy_positional_args_to_kwargs(HPyContext *ctx, const char **keywords,
        const HPy *args, HPy_ssize_t len_args,
        HPy normal_kwds)
{
    for (HPy_ssize_t i = 0; i < len_args; i++) {
        if (keywords[i] == NULL) {
            /* keyword argument is either input or output and not set here */
            continue;
        }
        if (NPY_UNLIKELY(i == 5)) {
            /*
             * This is only relevant for reduce, which is the only one with
             * 5 keyword arguments.
             */
            static HPyGlobal hg_NoValue;
            static int is_NoValue_set = 0;
            HPy NoValue = HPy_NULL;
            if (is_NoValue_set) {
                NoValue = HPyGlobal_Load(ctx, hg_NoValue);
            }
            assert(strcmp(keywords[i], "initial") == 0);
            npy_hpy_cache_import(ctx, "numpy", "_NoValue", &NoValue);
            if (!is_NoValue_set) {
                HPyGlobal_Store(ctx, &hg_NoValue, NoValue);
            }
            if (HPy_Is(ctx, args[i], NoValue)) {
                continue;
            }
        }

        int res = HPy_SetItem_s(ctx, normal_kwds, keywords[i], args[i]);
        if (res < 0) {
            return -1;
        }
    }
    return 0;
}

/*
 * Check a set of args for the `__array_ufunc__` method.  If more than one of
 * the input arguments implements `__array_ufunc__`, they are tried in the
 * order: subclasses before superclasses, otherwise left to right. The first
 * (non-None) routine returning something other than `NotImplemented`
 * determines the result. If all of the `__array_ufunc__` operations return
 * `NotImplemented` (or are None), a `TypeError` is raised.
 *
 * Returns 0 on success and 1 on exception. On success, *result contains the
 * result of the operation, if any. If *result is NULL, there is no override.
 */
NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
        PyObject *in_args, PyObject *out_args,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject **result)
{
    HPyContext *ctx = npy_get_context();
    HPy h_ufunc = HPy_FromPyObject(ctx, (PyObject *)ufunc);
    HPy h_in_args = HPy_FromPyObject(ctx, in_args);
    HPy h_out_args = HPy_FromPyObject(ctx, out_args);
    HPy const *h_args = (HPy const *)HPy_FromPyObjectArray(ctx, (PyObject **)args, len_args);
    HPy h_kwnames = HPy_FromPyObject(ctx, kwnames);
    HPy h_result;

    int res = HPyUFunc_CheckOverride(ctx, h_ufunc, method, h_in_args, h_out_args, h_args, (HPy_ssize_t)len_args, h_kwnames, &h_result);
    *result = HPy_AsPyObject(ctx, h_result);

    HPy_Close(ctx, h_result);
    HPy_Close(ctx, h_kwnames);
    HPy_CloseAndFreeArray(ctx, (HPy *)h_args, (HPy_ssize_t)len_args);
    HPy_Close(ctx, h_out_args);
    HPy_Close(ctx, h_in_args);
    HPy_Close(ctx, h_ufunc);

    return res;
}

NPY_NO_EXPORT int
HPyUFunc_CheckOverride(HPyContext *ctx, HPy ufunc, char *method,
        HPy in_args, HPy out_args,
        HPy const *args, HPy_ssize_t len_args, HPy kwnames,
        HPy *result)
{
    int status;

    int num_override_args;
    HPy with_override[NPY_MAXARGS];
    HPy array_ufunc_methods[NPY_MAXARGS];

    HPy method_name = HPy_NULL;
    HPy normal_kwds = HPy_NULL;

    HPy override_args = HPy_NULL;

    /*
     * Check inputs for overrides
     */

    num_override_args = hget_array_ufunc_overrides(ctx,
           in_args, out_args, with_override, array_ufunc_methods);
    if (num_override_args == -1) {
        // goto fail;
        return -1;
    }
    /* No overrides, bail out.*/
    if (num_override_args == 0) {
        *result = HPy_NULL;
        return 0;
    }

    /*
     * Normalize ufunc arguments, note that any input and output arguments
     * have already been stored in `in_args` and `out_args`.
     */
    normal_kwds = HPyDict_New(ctx);
    if (HPy_IsNull(normal_kwds)) {
        goto fail;
    }
    if (initialize_normal_kwds(ctx, out_args,
            args, len_args, kwnames, normal_kwds) < 0) {
        goto fail;
    }

    /*
        * Reduce-like methods can pass keyword arguments also by position,
        * in which case the additional positional arguments have to be copied
        * into the keyword argument dictionary. The `__call__` and `__outer__`
        * method have to normalize sig and signature.
        */

    /* ufunc.__call__ */
    if (strcmp(method, "__call__") == 0) {
        status = normalize_signature_keyword(ctx, normal_kwds);
    }
    /* ufunc.reduce */
    else if (strcmp(method, "reduce") == 0) {
        static const char *keywords[] = {
                NULL, "axis", "dtype", NULL, "keepdims",
                "initial", "where"};
        status = copy_positional_args_to_kwargs(ctx, keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.accumulate */
    else if (strcmp(method, "accumulate") == 0) {
        static const char *keywords[] = {
                NULL, "axis", "dtype", NULL};
        status = copy_positional_args_to_kwargs(ctx, keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.reduceat */
    else if (strcmp(method, "reduceat") == 0) {
        static const char *keywords[] = {
                NULL, NULL, "axis", "dtype", NULL};
        status = copy_positional_args_to_kwargs(ctx, keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.outer (identical to call) */
    else if (strcmp(method, "outer") == 0) {
        status = normalize_signature_keyword(ctx, normal_kwds);
    }
    /* ufunc.at */
    else if (strcmp(method, "at") == 0) {
        status = 0;
    }
    /* unknown method */
    else {
        // PyErr_Format(PyExc_TypeError,
        //                 "Internal Numpy error: unknown ufunc method '%s' in call "
        //                 "to PyUFunc_CheckOverride", method);
        HPyErr_SetString(ctx, ctx->h_TypeError,
                        "Internal Numpy error: unknown ufunc method '%s' in call "
                        "to PyUFunc_CheckOverride");
        status = -1;
    }
    if (status != 0) {
        goto fail;
    }

    method_name = HPyUnicode_FromString(ctx, method);
    if (HPy_IsNull(method_name)) {
        goto fail;
    }

    int len = (int)HPy_Length(ctx, in_args);

    /* Call __array_ufunc__ functions in correct order */
    while (1) {
        HPy override_obj;
        HPy override_array_ufunc;

        override_obj = HPy_NULL;
        *result = HPy_NULL;

        /* Choose an overriding argument */
        for (int i = 0; i < num_override_args; i++) {
            override_obj = with_override[i];
            if (HPy_IsNull(override_obj)) {
                continue;
            }

            /* Check for sub-types to the right of obj. */
            for (int j = i + 1; j < num_override_args; j++) {
                HPy other_obj = with_override[j];
                if (!HPy_IsNull(other_obj)) {
                    HPy other_obj_type = HPy_Type(ctx, other_obj);
                    HPy override_obj_type = HPy_Type(ctx, override_obj);
                    int is_equal = HPy_Is(ctx, other_obj_type, override_obj_type);
                    HPy_Close(ctx, other_obj_type);
                    HPy_Close(ctx, override_obj_type);
                    if (!is_equal) {
                        CAPI_WARN("missing PyObject_IsInstance");
                        PyObject *py_other_obj = HPy_AsPyObject(ctx, other_obj);
                        PyObject *py_override_obj = HPy_AsPyObject(ctx, override_obj);
                        int is_instance = PyObject_IsInstance(py_other_obj,
                                        (PyObject *)Py_TYPE(py_override_obj));
                        Py_DECREF(py_other_obj);
                        Py_DECREF(py_override_obj);
                        if (is_instance) {
                            override_obj = HPy_NULL;
                            break;
                        }
                    }
                }
            }

            /* override_obj had no subtypes to the right. */
            if (!HPy_IsNull(override_obj)) {
                override_array_ufunc = array_ufunc_methods[i];
                /* We won't call this one again (references decref'd below) */
                with_override[i] = HPy_NULL;
                array_ufunc_methods[i] = HPy_NULL;
                break;
            }
        }
        /*
            * Set override arguments for each call since the tuple must
            * not be mutated after use in PyPy
            * We increase all references since SET_ITEM steals
            * them and they will be DECREF'd when the tuple is deleted.
            */
        HPyTupleBuilder tb_override_args = HPyTupleBuilder_New(ctx, len + 3);
        if (HPyTupleBuilder_IsNull(tb_override_args)) {
            goto fail;
        }
        // Py_INCREF(ufunc);
        HPyTupleBuilder_Set(ctx, tb_override_args, 1, ufunc);
        // Py_INCREF(method_name);
        HPyTupleBuilder_Set(ctx, tb_override_args, 2, method_name);
        for (int i = 0; i < len; i++) {
            HPy item = HPy_GetItem_i(ctx, in_args, i);

            // Py_INCREF(item);
            HPyTupleBuilder_Set(ctx, tb_override_args, i + 3, item);
        }

        /* Check if there is a method left to call */
        if (HPy_IsNull(override_obj)) {
            /* No acceptable override found. */
            static HPyGlobal hg_errmsg_formatter;
            static int is_errmsg_formatter_set = 0;
            HPy errmsg_formatter;
            HPy errmsg;
            if (is_errmsg_formatter_set) {
                errmsg_formatter = HPyGlobal_Load(ctx, hg_errmsg_formatter);
            }
            npy_hpy_cache_import(ctx, "numpy.core._internal",
                                "array_ufunc_errmsg_formatter",
                                &errmsg_formatter);

            if (!HPy_IsNull(errmsg_formatter)) {
                if(!is_errmsg_formatter_set) {
                    HPyGlobal_Store(ctx, &hg_errmsg_formatter, errmsg_formatter);
                    is_errmsg_formatter_set = 1;
                }
                /* All tuple items must be set before use */
                // Py_INCREF(Py_None);
                HPyTupleBuilder_Set(ctx, tb_override_args, 0, ctx->h_None);
                override_args = HPyTupleBuilder_Build(ctx, tb_override_args);
                errmsg = HPy_CallTupleDict(ctx, errmsg_formatter, override_args,
                                        normal_kwds);
                if (!HPy_IsNull(errmsg)) {
                    HPyErr_SetObject(ctx, ctx->h_TypeError, errmsg);
                    HPy_Close(ctx, errmsg);
                }
            } else {
                HPyTupleBuilder_Cancel(ctx, tb_override_args);
            }
            HPy_Close(ctx, override_args);
            goto fail;
        }

        /*
            * Set the self argument of our unbound method.
            * This also steals the reference, so no need to DECREF after.
            */
        HPyTupleBuilder_Set(ctx, tb_override_args, 0, override_obj);
        /* Call the method */
        override_args = HPyTupleBuilder_Build(ctx, tb_override_args);
        *result = HPy_CallTupleDict(ctx,
            override_array_ufunc, override_args, normal_kwds);
        HPy_Close(ctx, override_array_ufunc);
        HPy_Close(ctx, override_args);
        if (HPy_IsNull(*result)) {
            /* Exception occurred */
            goto fail;
        }
        else if (HPy_Is(ctx, *result, ctx->h_NotImplemented)) {
            /* Try the next one */
            HPy_Close(ctx, *result);
            continue;
        }
        else {
            /* Good result. */
            break;
        }
    }
    status = 0;
    /* Override found, return it. */
    goto cleanup;
    fail:
    status = -1;
    cleanup:
    for (int i = 0; i < num_override_args; i++) {
        HPy_Close(ctx, with_override[i]);
        HPy_Close(ctx, array_ufunc_methods[i]);
    }
    HPy_Close(ctx, method_name);
    HPy_Close(ctx, normal_kwds);
    return status;
}

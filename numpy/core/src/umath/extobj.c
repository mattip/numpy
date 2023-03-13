#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "npy_config.h"

#include "npy_pycompat.h"

#include "extobj.h"
#include "numpy/ufuncobject.h"

#include "ufunc_object.h"  /* for npy_hpy_um_str_pyvals_name */
#include "common.h"

#if USE_USE_DEFAULTS==1
static int PyUFunc_NUM_NODEFAULTS = 0;

/*
 * This is a strategy to buy a little speed up and avoid the dictionary
 * look-up in the default case.  It should work in the presence of
 * threads.  If it is deemed too complicated or it doesn't actually work
 * it could be taken out.
 */
NPY_NO_EXPORT int
ufunc_update_use_defaults(void)
{
    PyObject *errobj = NULL;
    int errmask, bufsize;
    int res;

    PyUFunc_NUM_NODEFAULTS += 1;
    res = PyUFunc_GetPyValues("test", &bufsize, &errmask, &errobj);
    PyUFunc_NUM_NODEFAULTS -= 1;
    if (res < 0) {
        Py_XDECREF(errobj);
        return -1;
    }
    if ((errmask != UFUNC_ERR_DEFAULT) || (bufsize != NPY_BUFSIZE)
            || (PyTuple_GET_ITEM(errobj, 1) != Py_None)) {
        PyUFunc_NUM_NODEFAULTS += 1;
    }
    else if (PyUFunc_NUM_NODEFAULTS > 0) {
        PyUFunc_NUM_NODEFAULTS -= 1;
    }
    Py_XDECREF(errobj);
    return 0;
}
#endif

/*
 * fpstatus is the ufunc_formatted hardware status
 * errmask is the handling mask specified by the user.
 * errobj is a Python object with (string, callable object or None)
 * or NULL
 */

/*
 * 2. for each of the flags
 * determine whether to ignore, warn, raise error, or call Python function.
 * If ignore, do nothing
 * If warn, print a warning and continue
 * If raise return an error
 * If call, call a user-defined function with string
 */

NPY_NO_EXPORT int
_error_handler(int method, PyObject *errobj, char *errtype, int retstatus, int *first)
{
    PyObject *pyfunc, *ret, *args;
    char *name = PyBytes_AS_STRING(PyTuple_GET_ITEM(errobj,0));
    char msg[100];

    NPY_ALLOW_C_API_DEF

    /* don't need C API for a simple ignore */
    if (method == UFUNC_ERR_IGNORE) {
        return 0;
    }

    /* don't need C API for a simple print */
    if (method == UFUNC_ERR_PRINT) {
        if (*first) {
            fprintf(stderr, "Warning: %s encountered in %s\n", errtype, name);
            *first = 0;
        }
        return 0;
    }

    NPY_ALLOW_C_API;
    switch(method) {
    case UFUNC_ERR_WARN:
        PyOS_snprintf(msg, sizeof(msg), "%s encountered in %s", errtype, name);
        if (PyErr_Warn(PyExc_RuntimeWarning, msg) < 0) {
            goto fail;
        }
        break;
    case UFUNC_ERR_RAISE:
        PyErr_Format(PyExc_FloatingPointError, "%s encountered in %s",
                errtype, name);
        goto fail;
    case UFUNC_ERR_CALL:
        pyfunc = PyTuple_GET_ITEM(errobj, 1);
        if (pyfunc == Py_None) {
            PyErr_Format(PyExc_NameError,
                    "python callback specified for %s (in " \
                    " %s) but no function found.",
                    errtype, name);
            goto fail;
        }
        args = Py_BuildValue("NN", PyUnicode_FromString(errtype),
                PyLong_FromLong((long) retstatus));
        if (args == NULL) {
            goto fail;
        }
        ret = PyObject_CallObject(pyfunc, args);
        Py_DECREF(args);
        if (ret == NULL) {
            goto fail;
        }
        Py_DECREF(ret);
        break;
    case UFUNC_ERR_LOG:
        if (first) {
            *first = 0;
            pyfunc = PyTuple_GET_ITEM(errobj, 1);
            if (pyfunc == Py_None) {
                PyErr_Format(PyExc_NameError,
                        "log specified for %s (in %s) but no " \
                        "object with write method found.",
                        errtype, name);
                goto fail;
            }
            PyOS_snprintf(msg, sizeof(msg),
                    "Warning: %s encountered in %s\n", errtype, name);
            ret = PyObject_CallMethod(pyfunc, "write", "s", msg);
            if (ret == NULL) {
                goto fail;
            }
            Py_DECREF(ret);
        }
        break;
    }
    NPY_DISABLE_C_API;
    return 0;

fail:
    NPY_DISABLE_C_API;
    return -1;
}



NPY_NO_EXPORT PyObject *
get_global_ext_obj(void)
{
    PyObject *thedict;
    PyObject *ref = NULL;

#if USE_USE_DEFAULTS==1
    if (PyUFunc_NUM_NODEFAULTS != 0) {
#endif
        CAPI_WARN("Using PyThreadState_GetDict/PyEval_GetBuiltins");
        thedict = PyThreadState_GetDict();
        if (thedict == NULL) {
            thedict = PyEval_GetBuiltins();
        }
        HPy s = HPyGlobal_Load(npy_get_context(), npy_hpy_um_str_pyvals_name);
        ref = PyDict_GetItemWithError(thedict, HPy_AsPyObject(npy_get_context(), s));
        HPy_Close(npy_get_context(), s);
#if USE_USE_DEFAULTS==1
    }
#endif

    return ref;
}

NPY_NO_EXPORT HPy
hpy_get_global_ext_obj(HPyContext *ctx)
{
    PyObject *res = get_global_ext_obj();
    if (res != NULL) {
        HPy h_res = HPy_FromPyObject(ctx, res);
        Py_DECREF(res);
        return h_res;
    }
    return HPy_NULL;
}


/*
 * Extracts some values from the global pyvals tuple.
 * all destinations may be NULL, in which case they are not retrieved
 * ref - should hold the global tuple
 * name - is the name of the ufunc (ufuncobj->name)
 *
 * bufsize - receives the buffer size to use
 * errmask - receives the bitmask for error handling
 * errobj - receives the python object to call with the error,
 *          if an error handling method is 'call'
 */
NPY_NO_EXPORT int
_extract_pyvals(PyObject *ref, const char *name, int *bufsize,
                int *errmask, PyObject **errobj)
{
    PyObject *retval;

    /* default errobj case, skips dictionary lookup */
    if (ref == NULL) {
        if (errmask) {
            *errmask = UFUNC_ERR_DEFAULT;
        }
        if (errobj) {
            *errobj = Py_BuildValue("NO", PyBytes_FromString(name), Py_None);
        }
        if (bufsize) {
            *bufsize = NPY_BUFSIZE;
        }
        return 0;
    }

    if (!PyList_Check(ref) || (PyList_GET_SIZE(ref)!=3)) {
        PyErr_Format(PyExc_TypeError,
                "%s must be a length 3 list.", UFUNC_PYVALS_NAME);
        return -1;
    }

    if (bufsize != NULL) {
        *bufsize = PyLong_AsLong(PyList_GET_ITEM(ref, 0));
        if (error_converting(*bufsize)) {
            return -1;
        }
        if ((*bufsize < NPY_MIN_BUFSIZE) ||
                (*bufsize > NPY_MAX_BUFSIZE) ||
                (*bufsize % 16 != 0)) {
            PyErr_Format(PyExc_ValueError,
                    "buffer size (%d) is not in range "
                    "(%"NPY_INTP_FMT" - %"NPY_INTP_FMT") or not a multiple of 16",
                    *bufsize, (npy_intp) NPY_MIN_BUFSIZE,
                    (npy_intp) NPY_MAX_BUFSIZE);
            return -1;
        }
    }

    if (errmask != NULL) {
        *errmask = PyLong_AsLong(PyList_GET_ITEM(ref, 1));
        if (*errmask < 0) {
            if (PyErr_Occurred()) {
                return -1;
            }
            PyErr_Format(PyExc_ValueError,
                         "invalid error mask (%d)",
                         *errmask);
            return -1;
        }
    }

    if (errobj != NULL) {
        *errobj = NULL;
        retval = PyList_GET_ITEM(ref, 2);
        if (retval != Py_None && !PyCallable_Check(retval)) {
            PyObject *temp;
            temp = PyObject_GetAttrString(retval, "write");
            if (temp == NULL || !PyCallable_Check(temp)) {
                PyErr_SetString(PyExc_TypeError,
                                "python object must be callable or have " \
                                "a callable write method");
                Py_XDECREF(temp);
                return -1;
            }
            Py_DECREF(temp);
        }

        *errobj = Py_BuildValue("NO", PyBytes_FromString(name), retval);
        if (*errobj == NULL) {
            return -1;
        }
    }
    return 0;
}

NPY_NO_EXPORT int
_hpy_extract_pyvals(HPyContext *ctx, HPy h_ref, const char *name, int *bufsize,
                int *errmask, HPy *errobj)
{
    /* default errobj case, skips dictionary lookup */
    if (HPy_IsNull(h_ref)) {
        if (errmask) {
            *errmask = UFUNC_ERR_DEFAULT;
        }
        if (errobj) {
            HPy h_name = HPyBytes_FromString(ctx, name);
            *errobj = HPy_BuildValue(ctx, "OO", h_name, ctx->h_None);
            HPy_Close(ctx, h_name);
        }
        if (bufsize) {
            *bufsize = NPY_BUFSIZE;
        }
        return 0;
    }

    CAPI_WARN("calling _extract_pyvals");
    PyObject *py_ref = HPy_AsPyObject(ctx, h_ref);
    int result;
    if (errobj) {
        PyObject *py_errobj = NULL;
        result = _extract_pyvals(py_ref, name, bufsize, errmask, &py_errobj);
        *errobj = HPy_FromPyObject(ctx, py_errobj);
        Py_XDECREF(py_errobj);
    } else {
        result = _extract_pyvals(py_ref, name, bufsize, errmask, NULL);
    }
    Py_DECREF(py_ref);
    return result;
}

/*
 * check the floating point status
 *  - errmask: mask of status to check
 *  - extobj: ufunc pyvals object
 *            may be null, in which case the thread global one is fetched
 *  - ufunc_name: name of ufunc
 */
NPY_NO_EXPORT int
_check_ufunc_fperr(int errmask, PyObject *extobj, const char *ufunc_name) {
    HPyContext *ctx = npy_get_context();
    HPy h_extobj = HPy_FromPyObject(ctx, extobj);
    int res = _hpy_check_ufunc_fperr(ctx, errmask, h_extobj, ufunc_name);
    HPy_Close(ctx, h_extobj);
    return res;
}

NPY_NO_EXPORT int
_hpy_check_ufunc_fperr(HPyContext *ctx, int errmask, HPy extobj, const char *ufunc_name) {
    int fperr;
    HPy errobj = HPy_NULL;
    int ret;
    int first = 1;
    int close_extobj = 0;

    if (!errmask) {
        return 0;
    }
    fperr = npy_get_floatstatus_barrier((char*)HPy_AsVoidP(extobj));
    if (!fperr) {
        return 0;
    }

    /* Get error object globals */
    if (HPy_IsNull(extobj)) {
        extobj = hpy_get_global_ext_obj(ctx);
        if (HPy_IsNull(extobj) && HPyErr_Occurred(ctx)) {
            return -1;
        }
        close_extobj = 1;
    }
    int r = _hpy_extract_pyvals(ctx, extobj, ufunc_name, NULL, NULL, &errobj);
    if (close_extobj) {
        HPy_Close(ctx, extobj);
    }
    if (r < 0) {
        HPy_Close(ctx, errobj);
        return -1;
    }

    CAPI_WARN("_hpy_check_ufunc_fperr");
    PyObject *py_errobj = HPy_IsNull(errobj) ? NULL : HPy_AsPyObject(ctx, errobj);
    ret = PyUFunc_handlefperr(errmask, py_errobj, fperr, &first);
    // HPy_Close(ctx, errobj);
    Py_XDECREF(py_errobj);

    return ret;
}


NPY_NO_EXPORT int
_get_bufsize_errmask(PyObject * extobj, const char *ufunc_name,
                     int *buffersize, int *errormask)
{
    CAPI_WARN("_get_bufsize_errmask (there is a port of this already)");
    /* Get the buffersize and errormask */
    if (extobj == NULL) {
        extobj = get_global_ext_obj();
        if (extobj == NULL && PyErr_Occurred()) {
            return -1;
        }
    }
    if (_extract_pyvals(extobj, ufunc_name,
                        buffersize, errormask, NULL) < 0) {
        return -1;
    }

    return 0;
}

NPY_NO_EXPORT int
_hpy_get_bufsize_errmask(HPyContext *ctx, HPy extobj, const char *ufunc_name,
                     int *buffersize, int *errormask)
{
    /* Get the buffersize and errormask */
    if (HPy_IsNull(extobj)) {
        extobj = hpy_get_global_ext_obj(ctx);
        if (HPy_IsNull(extobj) && HPyErr_Occurred(ctx)) {
            return -1;
        }
    }
    if (_hpy_extract_pyvals(ctx, extobj, ufunc_name,
                        buffersize, errormask, NULL) < 0) {
        return -1;
    }

    return 0;
}

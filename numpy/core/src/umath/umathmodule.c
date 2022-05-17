/* -*- c -*- */
/* vim:syntax=c */

/*
 * _UMATHMODULE IS needed in __ufunc_api.h, included from numpy/ufuncobject.h.
 * This is a mess and it would be nice to fix it. It has nothing to do with
 * __ufunc_api.c
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "npy_config.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "abstract.h"

#include "numpy/npy_math.h"
#include "number.h"
#include "dispatching.h"

// Added for HPy port:
#include "hpy_utils.h"

static PyUFuncGenericFunction pyfunc_functions[] = {PyUFunc_On_Om};

static int
object_ufunc_type_resolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int i, nop = ufunc->nin + ufunc->nout;

    out_dtypes[0] = PyArray_DescrFromType(NPY_OBJECT);
    if (out_dtypes[0] == NULL) {
        return -1;
    }

    for (i = 1; i < nop; ++i) {
        Py_INCREF(out_dtypes[0]);
        out_dtypes[i] = out_dtypes[0];
    }

    return 0;
}

static int
object_ufunc_loop_selector(HPyContext *ctx,
                                HPy /* (PyUFuncObject *) */ ufunc,
                                HPy /* (PyArray_Descr **) */ *dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata,
                                int *out_needs_api)
{
    PyUFuncObject *ufunc_data = PyUFuncObject_AsStruct(ctx, ufunc);
    *out_innerloop = ufunc_data->functions[0];
    *out_innerloopdata = (ufunc_data->data == NULL) ? NULL : ufunc_data->data[0];
    *out_needs_api = 1;

    return 0;
}

PyObject *
ufunc_frompyfunc(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds) {
    PyObject *function, *pyname = NULL;
    int nin, nout, i, nargs;
    PyUFunc_PyFuncData *fdata;
    PyUFuncObject *self;
    const char *fname = NULL;
    char *str, *types, *doc;
    Py_ssize_t fname_len = -1;
    void * ptr, **data;
    int offset[2];
    PyObject *identity = NULL;  /* note: not the same semantics as Py_None */
    static char *kwlist[] = {"", "nin", "nout", "identity", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii|$O:frompyfunc", kwlist,
                &function, &nin, &nout, &identity)) {
        return NULL;
    }
    if (!PyCallable_Check(function)) {
        PyErr_SetString(PyExc_TypeError, "function must be callable");
        return NULL;
    }

    nargs = nin + nout;

    pyname = PyObject_GetAttrString(function, "__name__");
    if (pyname) {
        fname = PyUnicode_AsUTF8AndSize(pyname, &fname_len);
    }
    if (fname == NULL) {
        PyErr_Clear();
        fname = "?";
        fname_len = 1;
    }

    /*
     * ptr will be assigned to self->ptr, holds a pointer for enough memory for
     * self->data[0] (fdata)
     * self->data
     * self->name
     * self->types
     *
     * To be safest, all of these need their memory aligned on void * pointers
     * Therefore, we may need to allocate extra space.
     */
    offset[0] = sizeof(PyUFunc_PyFuncData);
    i = (sizeof(PyUFunc_PyFuncData) % sizeof(void *));
    if (i) {
        offset[0] += (sizeof(void *) - i);
    }
    offset[1] = nargs;
    i = (nargs % sizeof(void *));
    if (i) {
        offset[1] += (sizeof(void *)-i);
    }
    ptr = PyArray_malloc(offset[0] + offset[1] + sizeof(void *) +
                            (fname_len + 14));
    if (ptr == NULL) {
        Py_XDECREF(pyname);
        return PyErr_NoMemory();
    }
    fdata = (PyUFunc_PyFuncData *)(ptr);
    fdata->callable = function;
    fdata->nin = nin;
    fdata->nout = nout;

    data = (void **)(((char *)ptr) + offset[0]);
    data[0] = (void *)fdata;
    types = (char *)data + sizeof(void *);
    for (i = 0; i < nargs; i++) {
        types[i] = NPY_OBJECT;
    }
    str = types + offset[1];
    memcpy(str, fname, fname_len);
    memcpy(str+fname_len, " (vectorized)", 14);
    Py_XDECREF(pyname);

    /* Do a better job someday */
    doc = "dynamic ufunc based on a python function";

    self = (PyUFuncObject *)PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
            (PyUFuncGenericFunction *)pyfunc_functions, data,
            types, /* ntypes */ 1, nin, nout, identity ? PyUFunc_IdentityValue : PyUFunc_None,
            str, doc, /* unused */ 0, NULL, identity);

    if (self == NULL) {
        PyArray_free(ptr);
        return NULL;
    }
    // Py_INCREF(function);
    // self->obj = function;
    HPyField_StorePyObj((PyObject *)self, &self->obj, function);
    self->ptr = ptr;

    self->type_resolver = &object_ufunc_type_resolver;
    self->legacy_inner_loop_selector = &object_ufunc_loop_selector;
    PyObject_GC_Track(self);

    return (PyObject *)self;
}

/* docstring in numpy.add_newdocs.py */
PyObject *
add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyUFuncObject *ufunc;
    PyObject *str;
    if (!PyArg_ParseTuple(args, "O!O!:_add_newdoc_ufunc", &PyUFunc_Type, &ufunc,
                                        &PyUnicode_Type, &str)) {
        return NULL;
    }
    if (ufunc->doc != NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot change docstring of ufunc with non-NULL docstring");
        return NULL;
    }

    PyObject *tmp = PyUnicode_AsUTF8String(str);
    if (tmp == NULL) {
        return NULL;
    }
    char *docstr = PyBytes_AS_STRING(tmp);

    /*
     * This introduces a memory leak, as the memory allocated for the doc
     * will not be freed even if the ufunc itself is deleted. In practice
     * this should not be a problem since the user would have to
     * repeatedly create, document, and throw away ufuncs.
     */
    char *newdocstr = malloc(strlen(docstr) + 1);
    if (!newdocstr) {
        Py_DECREF(tmp);
        return PyErr_NoMemory();
    }
    strcpy(newdocstr, docstr);
    ufunc->doc = newdocstr;

    Py_DECREF(tmp);
    Py_RETURN_NONE;
}


/*
 *****************************************************************************
 **                            SETUP UFUNCS                                 **
 *****************************************************************************
 */

NPY_VISIBILITY_HIDDEN HPyGlobal npy_hpy_um_str_array_prepare;
NPY_VISIBILITY_HIDDEN HPyGlobal npy_hpy_um_str_array_wrap;
NPY_VISIBILITY_HIDDEN HPyGlobal npy_hpy_um_str_pyvals_name;

/* intern some strings used in ufuncs, returns 0 on success */
static int
intern_strings(HPyContext *ctx)
{
    HPy h__array_prepare__ = HPyUnicode_InternFromString(ctx, "__array_prepare__");
    if (HPy_IsNull(h__array_prepare__)) {
        return -1;
    }
    HPyGlobal_Store(ctx, &npy_hpy_um_str_array_prepare, h__array_prepare__);
    HPy_Close(ctx, h__array_prepare__);
    HPy h__array_wrap__ = HPyUnicode_InternFromString(ctx, "__array_wrap__");
    if (HPy_IsNull(h__array_wrap__)) {
        return -1;
    }
    HPyGlobal_Store(ctx, &npy_hpy_um_str_array_wrap, h__array_wrap__);
    HPy_Close(ctx, h__array_wrap__);

    HPy h_pyvals_name = HPyUnicode_InternFromString(ctx, UFUNC_PYVALS_NAME);
    if (HPy_IsNull(h_pyvals_name)) {
        return -1;
    }
    HPyGlobal_Store(ctx, &npy_hpy_um_str_pyvals_name, h_pyvals_name);
    HPy_Close(ctx, h_pyvals_name);
    return 0;
}

/* Setup the umath part of the module */

int initumath(HPyContext *ctx, HPy m, HPy d)
{
    int UFUNC_FLOATING_POINT_SUPPORT = 1;

#ifdef NO_UFUNC_FLOATING_POINT_SUPPORT
    UFUNC_FLOATING_POINT_SUPPORT = 0;
#endif

    /* Add some symbolic constants to the module */
    HPy s;
    HPy_SetItem_s(ctx, d, "pi", s = HPyFloat_FromDouble(ctx, NPY_PI));
    HPy_Close(ctx, s);
    HPy_SetItem_s(ctx, d, "e", s = HPyFloat_FromDouble(ctx, NPY_E));
    HPy_Close(ctx, s);
    HPy_SetItem_s(ctx, d, "euler_gamma", s = HPyFloat_FromDouble(ctx, NPY_EULER));
    HPy_Close(ctx, s);

#define ADDCONST(str) \
    HPy_SetItem_s(ctx, d, #str, s = HPyLong_FromLong(ctx, UFUNC_##str)); \
    HPy_Close(ctx, s)
#define ADDSCONST(str) \
    HPy_SetItem_s(ctx, d, "UFUNC_" #str, s = HPyUnicode_FromString(ctx, UFUNC_##str)); \
    HPy_Close(ctx, s);

    ADDCONST(ERR_IGNORE);
    ADDCONST(ERR_WARN);
    ADDCONST(ERR_CALL);
    ADDCONST(ERR_RAISE);
    ADDCONST(ERR_PRINT);
    ADDCONST(ERR_LOG);
    ADDCONST(ERR_DEFAULT);

    ADDCONST(SHIFT_DIVIDEBYZERO);
    ADDCONST(SHIFT_OVERFLOW);
    ADDCONST(SHIFT_UNDERFLOW);
    ADDCONST(SHIFT_INVALID);

    ADDCONST(FPE_DIVIDEBYZERO);
    ADDCONST(FPE_OVERFLOW);
    ADDCONST(FPE_UNDERFLOW);
    ADDCONST(FPE_INVALID);

    ADDCONST(FLOATING_POINT_SUPPORT);

    ADDSCONST(PYVALS_NAME);

#undef ADDCONST
#undef ADDSCONST

    HPy_SetItem_s(ctx, d, "UFUNC_BUFSIZE_DEFAULT", s = HPyLong_FromLong(ctx, (long)NPY_BUFSIZE));
    HPy_Close(ctx, s);

#define ADDFCONST(name, value) \
    HPy_SetItem_s(ctx, d, #name, s = HPyFloat_FromDouble(ctx, (double) (value))); \
    HPy_Close(ctx, s);

    ADDFCONST(PINF, NPY_INFINITY);
    ADDFCONST(NINF, -NPY_INFINITY);
    ADDFCONST(PZERO, NPY_PZERO);
    ADDFCONST(NZERO, NPY_NZERO);
    ADDFCONST(NAN, NPY_NAN);

#undef ADDFCONST

    s = HPy_GetItem_s(ctx, d, "divide");
    HPy_SetItem_s(ctx, d, "true_divide", s);
    HPy_Close(ctx, s);

    s = HPy_GetItem_s(ctx, d, "conjugate");
    HPy s2 = HPy_GetItem_s(ctx, d, "remainder");
    /* Setup the array object's numerical structures with appropriate
       ufuncs in d*/
    _PyArray_SetNumericOps(ctx, d);

    HPy_SetItem_s(ctx, d, "conj", s);
    HPy_SetItem_s(ctx, d, "mod", s2);

    if (intern_strings(ctx) < 0) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
           "cannot intern umath strings while initializing _multiarray_umath.");
        return -1;
    }

    /*
     * Set up promoters for logical functions
     * TODO: This should probably be done at a better place, or even in the
     *       code generator directly.
     */
    // TODO HPY LABS PORT: original used PyDict_GetItemWithError, is HPy_GetItem suposed to be equivalent?
    s = HPy_GetItem_s(ctx, d, "logical_and");
    if (HPy_IsNull(s)) {
        return -1;
    }

    // Calls to install_logical_ufunc_promoter, install_logical_ufunc_promoter,
    // and install_logical_ufunc_promoter removed, because they do not seem necessary
    // for the HPy example
    // PyObject *py_s;
    // *py_s = HPy_AsPyObject(ctx, s);
    // CAPI_WARN("Leaving to install_logical_ufunc_promoter");
    // if (install_logical_ufunc_promoter(py_s) < 0) {
    //      return -1;
    // }
    // Py_DECREF(py_s);
    // HPy_Close(ctx, s);

    s = HPy_GetItem_s(ctx, d, "logical_or");
    if (HPy_IsNull(s)) {
        return -1;
    }

    // py_s = HPy_AsPyObject(ctx, s);
    // if (install_logical_ufunc_promoter(py_s) < 0) {
    //     return -1;
    // }
    // Py_DECREF(py_s);
    // HPy_Close(ctx, s);

    s = HPy_GetItem_s(ctx, d, "logical_xor");
    if (HPy_IsNull(s)) {
        return -1;
    }

    // py_s = HPy_AsPyObject(ctx, s);
    // if (install_logical_ufunc_promoter(py_s) < 0) {
    //     return -1;
    // }
    // Py_DECREF(py_s);
    // HPy_Close(ctx, s);

    return 0;
}

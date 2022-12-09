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
#include "ufunc_object.h" // HPy ufunc ported functions
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
h_object_ufunc_type_resolver(HPyContext *ctx,
                                HPy ufunc, // PyUFuncObject *
                                NPY_CASTING casting,
                                HPy *operands, //PyArrayObject **
                                HPy type_tup, // PyObject *
                                HPy *out_dtypes) // PyArray_Descr **
{
    PyUFuncObject *ufunc_data = PyUFuncObject_AsStruct(ctx, ufunc);
    int i, nop = ufunc_data->nin + ufunc_data->nout;

    out_dtypes[0] = HPyArray_DescrFromType(ctx, NPY_OBJECT);
    if (HPy_IsNull(out_dtypes[0])) {
        return -1;
    }

    for (i = 1; i < nop; ++i) {
        out_dtypes[i] = HPy_Dup(ctx, out_dtypes[0]);
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

HPyDef_METH(frompyfunc, "frompyfunc", HPyFunc_KEYWORDS)
HPy
frompyfunc_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t args_len, HPy kwds) {
    HPy function, pyname = HPy_NULL;
    int nin, nout, i, nargs;
    PyUFunc_PyFuncData *fdata;
    HPy self; // PyUFuncObject *
    const char *fname = NULL;
    char *str, *types, *doc;
    HPy_ssize_t fname_len = -1;
    void * ptr, **data;
    int offset[2];
    HPy identity = HPy_NULL;  /* note: not the same semantics as Py_None */
    static const char *kwlist[] = {"", "nin", "nout", "identity", NULL};

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, args_len, kwds, "Oii|$O:frompyfunc", kwlist,
                &function, &nin, &nout, &identity)) {
        return HPy_NULL;
    }
    if (!HPyCallable_Check(ctx, function)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "function must be callable");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    nargs = nin + nout;

    pyname = HPy_GetAttr_s(ctx, function, "__name__");
    if (!HPy_IsNull(pyname)) {
        fname = HPyUnicode_AsUTF8AndSize(ctx, pyname, &fname_len);
    }
    if (fname == NULL) {
        HPyErr_Clear(ctx);
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
        HPy_Close(ctx, pyname);
        return HPyErr_NoMemory(ctx);
    }
    fdata = (PyUFunc_PyFuncData *)(ptr);
    fdata->callable = HPy_AsPyObject(ctx, function);
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
    HPy_Close(ctx, pyname);

    /* Do a better job someday */
    doc = "dynamic ufunc based on a python function";

    self = HPyUFunc_FromFuncAndDataAndSignatureAndIdentity(ctx,
            (PyUFuncGenericFunction *)pyfunc_functions, data,
            types, /* ntypes */ 1, nin, nout, !HPy_IsNull(identity) ? PyUFunc_IdentityValue : PyUFunc_None,
            str, doc, /* unused */ 0, NULL, identity);

    if (HPy_IsNull(self)) {
        PyArray_free(ptr);
        return HPy_NULL;
    }
    // Py_INCREF(function);
    // self->obj = function;
    PyUFuncObject *self_struct = PyUFuncObject_AsStruct(ctx, self);
    HPyField_Store(ctx, self, &self_struct->obj, function);
    self_struct->ptr = ptr;

    self_struct->hpy_type_resolver = &h_object_ufunc_type_resolver;
    self_struct->type_resolver = &object_ufunc_type_resolver;
    self_struct->legacy_inner_loop_selector = &object_ufunc_loop_selector;
    PyObject *py_self = HPy_AsPyObject(ctx, self);
    CAPI_WARN("missing PyObject_GC_Track");
    PyObject_GC_Track(py_self);
    Py_DECREF(py_self);

    return self;
}

/* docstring in numpy.add_newdocs.py */
HPyDef_METH(add_newdoc_ufunc, "_add_newdoc_ufunc", HPyFunc_VARARGS)
HPy
add_newdoc_ufunc_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs)
{
    HPy ufunc; // PyUFuncObject *
    HPy str;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O!O!:_add_newdoc_ufunc", 
                        &ufunc, &str)) {
        return HPy_NULL;
    }
    HPy ufunc_type = HPyGlobal_Load(ctx, HPyUFunc_Type);
    if (!HPyType_IsSubtype(ctx, ufunc, ufunc_type) || 
            !HPyType_IsSubtype(ctx, str, ctx->h_UnicodeType)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "_add_newdoc_ufunc: argument error TODO");
        HPy_Close(ctx, ufunc_type);
        return HPy_NULL;
    }
    HPy_Close(ctx, ufunc_type);
    PyUFuncObject *ufunc_struct = PyUFuncObject_AsStruct(ctx, ufunc);
    if (ufunc_struct->doc != NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot change docstring of ufunc with non-NULL docstring");
        return HPy_NULL;
    }

    HPy tmp = HPyUnicode_AsUTF8String(ctx, str);
    if (HPy_IsNull(tmp)) {
        return HPy_NULL;
    }
    const char *docstr = HPyBytes_AS_STRING(ctx, tmp);

    /*
     * This introduces a memory leak, as the memory allocated for the doc
     * will not be freed even if the ufunc itself is deleted. In practice
     * this should not be a problem since the user would have to
     * repeatedly create, document, and throw away ufuncs.
     */
    char *newdocstr = malloc(strlen(docstr) + 1);
    if (!newdocstr) {
        HPy_Close(ctx, tmp);
        return HPyErr_NoMemory(ctx);
    }
    strcpy(newdocstr, docstr);
    ufunc_struct->doc = newdocstr;

    HPy_Close(ctx, tmp);
    return HPy_Dup(ctx, ctx->h_None);
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

    if (install_logical_ufunc_promoter(ctx, s) < 0) {
         return -1;
    }
    HPy_Close(ctx, s);

    s = HPy_GetItem_s(ctx, d, "logical_or");
    if (HPy_IsNull(s)) {
        return -1;
    }

    if (install_logical_ufunc_promoter(ctx, s) < 0) {
         return -1;
    }
    HPy_Close(ctx, s);

    s = HPy_GetItem_s(ctx, d, "logical_xor");
    if (HPy_IsNull(s)) {
        return -1;
    }

    if (install_logical_ufunc_promoter(ctx, s) < 0) {
         return -1;
    }
    HPy_Close(ctx, s);

    return 0;
}

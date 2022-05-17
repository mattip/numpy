/*
 * This file defines most of the machinery in order to wrap legacy style
 * ufunc loops into new style arraymethods.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"

#include "convert_datatype.h"
#include "array_method.h"
#include "dtype_transfer.h"
#include "legacy_array_method.h"
#include "dtypemeta.h"
#include "hpy_utils.h"
#include "ufunc_object.h"


typedef struct {
    NpyAuxData base;
    /* The legacy loop and additional user data: */
    PyUFuncGenericFunction loop;
    void *user_data;
    /* Whether to check for PyErr_Occurred(), must require GIL if used */
    int pyerr_check;
} legacy_array_method_auxdata;


/* Use a free list, since we should normally only need one at a time */
#define NPY_LOOP_DATA_CACHE_SIZE 5
static int loop_data_num_cached = 0;
static  legacy_array_method_auxdata *loop_data_cache[NPY_LOOP_DATA_CACHE_SIZE];


static void
legacy_array_method_auxdata_free(NpyAuxData *data)
{
    if (loop_data_num_cached < NPY_LOOP_DATA_CACHE_SIZE) {
        loop_data_cache[loop_data_num_cached] = (
                (legacy_array_method_auxdata *)data);
        loop_data_num_cached++;
    }
    else {
        PyMem_Free(data);
    }
}

#undef NPY_LOOP_DATA_CACHE_SIZE


NpyAuxData *
get_new_loop_data(
        PyUFuncGenericFunction loop, void *user_data, int pyerr_check)
{
    legacy_array_method_auxdata *data;
    if (NPY_LIKELY(loop_data_num_cached > 0)) {
        loop_data_num_cached--;
        data = loop_data_cache[loop_data_num_cached];
    }
    else {
        data = PyMem_Malloc(sizeof(legacy_array_method_auxdata));
        if (data == NULL) {
            return NULL;
        }
        data->base.free = legacy_array_method_auxdata_free;
        data->base.clone = NULL;  /* no need for cloning (at least for now) */
    }
    data->loop = loop;
    data->user_data = user_data;
    data->pyerr_check = pyerr_check;
    return (NpyAuxData *)data;
}


/*
 * This is a thin wrapper around the legacy loop signature.
 */
static int
generic_wrapped_legacy_loop(HPyContext *ctx, HPyArrayMethod_Context *NPY_UNUSED(context),
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    legacy_array_method_auxdata *ldata = (legacy_array_method_auxdata *)auxdata;

    ldata->loop((char **)data, dimensions, strides, ldata->user_data);
    if (ldata->pyerr_check && HPyErr_Occurred(ctx)) {
        return -1;
    }
    return 0;
}


/*
 * Signal that the old type-resolution function must be used to resolve
 * the descriptors (mainly/only used for datetimes due to the unit).
 *
 * ArrayMethod's are expected to implement this, but it is too tricky
 * to support properly.  So we simply set an error that should never be seen.
 */
NPY_NO_EXPORT NPY_CASTING
wrapped_legacy_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(method), /* (struct PyArrayMethodObject_tag *) */
        HPy *NPY_UNUSED(dtypes), /* PyArray_DTypeMeta **dtypes */
        HPy *NPY_UNUSED(given_descrs), /* PyArray_Descr **given_descrs */
        HPy *NPY_UNUSED(loop_descrs), /* PyArray_Descr **loop_descrs */
        npy_intp *NPY_UNUSED(view_offset))
{
    HPyErr_SetString(ctx, ctx->h_RuntimeError,
            "cannot use legacy wrapping ArrayMethod without calling the ufunc "
            "itself.  If this error is hit, the solution will be to port the "
            "legacy ufunc loop implementation to the new API.");
    return -1;
}

/*
 * Much the same as the default type resolver, but tries a bit harder to
 * preserve metadata.
 */
static NPY_CASTING
simple_legacy_resolve_descriptors(
        HPyContext *ctx,
        HPy method, /* PyArrayMethodObject *method */
        HPy *dtypes, /* PyArray_DTypeMeta **dtypes */
        HPy *given_descrs, /* PyArray_Descr **given_descrs */
        HPy *output_descrs, /* PyArray_Descr **output_descrs */
        npy_intp *view_offset)
{
    PyArrayMethodObject *method_data = PyArrayMethodObject_AsStruct(ctx, method);
    int i = 0;
    int nin = method_data->nin;
    int nout = method_data->nout;

    if (nin == 2 && nout == 1 && !HPy_IsNull(given_descrs[2])
            && HPy_Is(ctx, dtypes[0], dtypes[2])) {
        /*
         * Could be a reduction, which requires `descr[0] is descr[2]`
         * (identity) at least currently. This is because `op[0] is op[2]`.
         * (If the output descriptor is not passed, the below works.)
         */
        output_descrs[2] = hensure_dtype_nbo(ctx, given_descrs[2]);
        if (HPy_IsNull(output_descrs[2])) {
            HPy_Close(ctx, output_descrs[2]);
            return -1;
        }
        output_descrs[0] = HPy_Dup(ctx, output_descrs[2]);
        if (HPy_Is(ctx, dtypes[1], dtypes[2])) {
            /* Same for the second one (accumulation is stricter) */
            output_descrs[1] = HPy_Dup(ctx, output_descrs[2]);
        }
        else {
            output_descrs[1] = hensure_dtype_nbo(ctx, given_descrs[1]);
            if (HPy_IsNull(output_descrs[1])) {
                i = 2;
                goto fail;
            }
        }
        return NPY_NO_CASTING;
    }

    for (; i < nin + nout; i++) {
        if (!HPy_IsNull(given_descrs[i])) {
            output_descrs[i] = hensure_dtype_nbo(ctx, given_descrs[i]);
        }
        else if (HPy_Is(ctx, dtypes[i], dtypes[0]) && i > 0) {
            /* Preserve metadata from the first operand if same dtype */
            output_descrs[i] = HPy_Dup(ctx, output_descrs[0]);
        }
        else {
            output_descrs[i] = hdtypemeta_call_default_descr(ctx, dtypes[i]);
        }
        if (HPy_IsNull(output_descrs[i])) {
            goto fail;
        }
    }

    return NPY_NO_CASTING;

  fail:
    for (; i >= 0; i--) {
        HPy_Close(ctx, output_descrs[i]);
        output_descrs[i] = HPy_NULL;
    }
    return -1;
}


/*
 * This function grabs the legacy inner-loop.  If this turns out to be slow
 * we could probably cache it (with some care).
 */
NPY_NO_EXPORT int
get_wrapped_legacy_ufunc_loop(HPyContext *ctx, HPyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *NPY_UNUSED(strides),
        HPyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    assert(aligned);
    assert(!move_references);


    PyArrayMethodObject *method_data = PyArrayMethodObject_AsStruct(ctx, context->method);
    if (HPy_IsNull(context->caller) ||
            !HPyGlobal_TypeCheck(ctx, context->caller, HPyUFunc_Type)) {
        HPyErr_Format_p(ctx, ctx->h_RuntimeError,
                "cannot call %s without its ufunc as caller context.",
                method_data->name);
        return -1;
    }

    PyUFuncObject *ufunc = PyUFuncObject_AsStruct(ctx, context->caller);
    void *user_data;
    int needs_api = 0;

    PyUFuncGenericFunction loop = NULL;
    /* Note that `needs_api` is not reliable (it was in fact unused normally) */
    if (ufunc->legacy_inner_loop_selector(ctx, context->caller,
            context->descriptors, &loop, &user_data, &needs_api) < 0) {
        return -1;
    }
    *flags = method_data->flags & NPY_METH_RUNTIME_FLAGS;
    if (needs_api) {
        *flags |= NPY_METH_REQUIRES_PYAPI;
    }

    *out_loop = &generic_wrapped_legacy_loop;
    *out_transferdata = get_new_loop_data(
            loop, user_data, (*flags & NPY_METH_REQUIRES_PYAPI) != 0);
    if (*out_transferdata == NULL) {
        HPyErr_NoMemory(ctx);
        return -1;
    }
    return 0;
}


/*
 * Get the unbound ArrayMethod which wraps the instances of the ufunc.
 * Note that this function stores the result on the ufunc and then only
 * returns the same one.
 */
NPY_NO_EXPORT PyArrayMethodObject *
PyArray_NewLegacyWrappingArrayMethod(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *signature[])
{
    HPyContext *ctx = npy_get_context();
    int nargs = ufunc->nin+ufunc->nout;
    HPy *h_signature = HPy_FromPyObjectArray(ctx, (PyObject **)signature, nargs);
    HPy h_res = HPyArray_NewLegacyWrappingArrayMethod(ctx, ufunc, h_signature);
    PyArrayMethodObject *res = (PyArrayMethodObject *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_CloseAndFreeArray(ctx, h_signature, nargs);
    return res;
}

NPY_NO_EXPORT HPy
HPyArray_NewLegacyWrappingArrayMethod(HPyContext *ctx, PyUFuncObject *ufunc,
        HPy signature[])
{
    // PyArray_DTypeMeta *signature[]
#define DATA(ctx, h) PyArray_DTypeMeta_AsStruct(ctx, h)
    char method_name[101];
    const char *name = ufunc->name ? ufunc->name : "<unknown>";
    snprintf(method_name, 100, "legacy_ufunc_wrapper_for_%s", name);

    /*
     * Assume that we require the Python API when any of the (legacy) dtypes
     * flags it.
     */
    int any_output_flexible = 0;
    NPY_ARRAYMETHOD_FLAGS flags = 0;
    if (ufunc->nargs == 3 &&
            DATA(ctx, signature[0])->type_num == NPY_BOOL &&
            DATA(ctx, signature[1])->type_num == NPY_BOOL &&
            DATA(ctx, signature[2])->type_num == NPY_BOOL && (
                strcmp(ufunc->name, "logical_or") == 0 ||
                strcmp(ufunc->name, "logical_and") == 0 ||
                strcmp(ufunc->name, "logical_xor") == 0)) {
        /*
         * This is a logical ufunc, and the `??->?` loop`. It is always OK
         * to cast any input to bool, because that cast is defined by
         * truthiness.
         * This allows to ensure two things:
         * 1. `np.all`/`np.any` know that force casting the input is OK
         *    (they must do this since there are no `?l->?`, etc. loops)
         * 2. The logical functions automatically work for any DType
         *    implementing a cast to boolean.
         */
        flags = _NPY_METH_FORCE_CAST_INPUTS;
    }

    for (int i = 0; i < ufunc->nin+ufunc->nout; i++) {
        HPy h_singleton = hdtypemeta_get_singleton(ctx, signature[i]);
        if (PyArray_Descr_AsStruct(ctx, h_singleton)->flags & (
                NPY_ITEM_REFCOUNT | NPY_ITEM_IS_POINTER | NPY_NEEDS_PYAPI)) {
            flags |= NPY_METH_REQUIRES_PYAPI;
        }
        HPy_Close(ctx, h_singleton);
        if (NPY_DT_is_parametric(DATA(ctx, signature[i]))) {
            any_output_flexible = 1;
        }
    }

    PyType_Slot slots[3] = {
        {NPY_METH_get_loop, &get_wrapped_legacy_ufunc_loop},
        {NPY_METH_resolve_descriptors, &simple_legacy_resolve_descriptors},
        {0, NULL},
    };
    if (any_output_flexible) {
        /* We cannot use the default descriptor resolver. */
        slots[1].pfunc = &wrapped_legacy_resolve_descriptors;
    }

    PyArrayMethod_Spec spec = {
        .name = method_name,
        .nin = ufunc->nin,
        .nout = ufunc->nout,
        .dtypes = signature,
        .flags = flags,
        .slots = slots,
        .casting = NPY_NO_CASTING,
    };

    HPy bound_res = HPyArrayMethod_FromSpec_int(ctx, &spec, 1);
    if (HPy_IsNull(bound_res)) {
        return HPy_NULL;
    }
    PyBoundArrayMethodObject *bound_res_data = PyBoundArrayMethodObject_AsStruct(ctx, bound_res);
    // PyArrayMethodObject *res
    HPy res = HPyField_Load(ctx, bound_res, bound_res_data->method);
    HPy_Close(ctx, bound_res);
    return res;
}

/*
 * This file implements a basic scaled float64 DType.  The reason is to have
 * a simple parametric DType for testing.  It is not meant to be a useful
 * DType by itself, but due to the scaling factor has similar properties as
 * a Unit DType.
 *
 * The code here should be seen as a work in progress.  Some choices are made
 * to test certain code paths, but that does not mean that they must not
 * be modified.
 *
 * NOTE: The tests were initially written using private API and ABI, ideally
 *       they should be replaced/modified with versions using public API.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

#include "array_method.h"
#include "common.h"
#include "numpy/npy_math.h"
#include "convert_datatype.h"
#include "dtypemeta.h"
#include "dispatching.h"


typedef struct {
    PyArray_Descr base;
    double scaling;
} PyArray_SFloatDescr;

HPyType_LEGACY_HELPERS(PyArray_SFloatDescr)

// HPy: moved to "array_method.h" to be registered in multiarraymodule.c
// static HPyGlobal HPyArray_SFloatDType;
// static HPyGlobal SFloatSingleton;
static PyArray_DTypeMeta *PyArray_SFloatDType;


static int
sfloat_is_known_scalar_type(HPyContext *ctx,
        HPy /* PyArray_DTypeMeta * */ NPY_UNUSED(cls), HPy type)
{
    /* Accept only floats (some others may work due to normal casting) */
    if (HPy_Is(ctx, type, ctx->h_FloatType)) {
        return 1;
    }
    return 0;
}


static HPy
sfloat_default_descr(HPyContext *ctx, HPy NPY_UNUSED(cls))
{
    return HPyGlobal_Load(ctx, SFloatSingleton);
}

static HPy
sfloat_discover_from_hpy(HPyContext *ctx, HPy cls, HPy NPY_UNUSED(obj))
{
    return sfloat_default_descr(ctx, cls);
}

static PyArray_Descr *
sfloat_discover_from_pyobject(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    HPyContext *ctx = npy_get_context();
    HPy h_cls = HPy_FromPyObject(ctx, (PyObject*)cls);
    HPy h_res = sfloat_default_descr(ctx, h_cls);
    PyArray_Descr *res = (PyArray_Descr *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_cls);
    HPy_Close(ctx, h_res);
    return res;
}


static PyArray_DTypeMeta *
sfloat_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num == NPY_DOUBLE) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}

static HPy // PyArray_DTypeMeta *
hpy_sfloat_common_dtype(HPyContext *ctx, HPy /* PyArray_DTypeMeta * */ cls, 
                                            HPy /* PyArray_DTypeMeta * */ other)
{
    PyArray_DTypeMeta *other_struct = PyArray_DTypeMeta_AsStruct(ctx, other);
    if (NPY_DT_is_legacy(other_struct) && other_struct->type_num == NPY_DOUBLE) {
        // Py_INCREF(cls);
        return HPy_Dup(ctx, cls);
    }
    // Py_INCREF(Py_NotImplemented);
    return HPy_Dup(ctx, ctx->h_NotImplemented);
}


static PyArray_Descr *
sfloat_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    PyArray_SFloatDescr *sf1 = (PyArray_SFloatDescr *)descr1;
    PyArray_SFloatDescr *sf2 = (PyArray_SFloatDescr *)descr2;
    /* We make the choice of using the larger scaling */
    if (sf1->scaling >= sf2->scaling) {
        Py_INCREF(descr1);
        return descr1;
    }
    Py_INCREF(descr2);
    return descr2;
}

static HPy
hpy_sfloat_common_instance(HPyContext *ctx, HPy descr1, HPy descr2)
{

    PyArray_SFloatDescr *sf1 = PyArray_SFloatDescr_AsStruct(ctx, descr1);
    PyArray_SFloatDescr *sf2 = PyArray_SFloatDescr_AsStruct(ctx, descr2);
    /* We make the choice of using the larger scaling */
    if (sf1->scaling >= sf2->scaling) {
        return HPy_Dup(ctx, descr1);
    }
    return HPy_Dup(ctx, descr2);
}


/*
 * Implement minimal getitem and setitem to make this DType mostly(?) safe to
 * expose in Python.
 * TODO: This should not use the old-style API, but the new-style is missing!
*/

static HPy
sfloat_getitem(HPyContext *ctx, char *data, HPy h_arr, PyArrayObject *arr)
{
    HPy h_descr = HPyArray_DESCR(ctx, h_arr, arr);
    CAPI_WARN("sfloat_getitem: using PyArray_SFloatDescr");
    PyArray_SFloatDescr *descr = (PyArray_SFloatDescr *)HPy_AsPyObject(ctx, h_descr);
    HPy_Close(ctx, h_descr);
    double value;
    memcpy(&value, data, sizeof(double));
    double scaling = descr->scaling;
    Py_DECREF(descr);
    return HPyFloat_FromDouble(ctx, value * scaling);
}


static int
sfloat_setitem(HPyContext *ctx, HPy obj, char *data, HPy h_arr)
{
    HPy obj_type = HPy_Type(ctx, obj);
    if (!HPy_Is(ctx, obj_type, ctx->h_FloatType)) {
        HPy_Close(ctx, obj_type);
        HPyErr_SetString(ctx, ctx->h_NotImplementedError,
                "Currently only accepts floats");
        return -1;
    }
    HPy_Close(ctx, obj_type);

    HPy h_descr = HPyArray_GetDescr(ctx, h_arr);
    CAPI_WARN("sfloat_setitem: using PyArray_SFloatDescr");
    PyArray_SFloatDescr *descr = (PyArray_SFloatDescr *)HPy_AsPyObject(ctx, h_descr);
    HPy_Close(ctx, h_descr);
    double value = HPyFloat_AsDouble(ctx, obj);
    value /= descr->scaling;
    Py_DECREF(descr);

    memcpy(data, &value, sizeof(double));
    return 0;
}


/* Special DType methods and the descr->f slot storage */
NPY_DType_Slots sfloat_slots = {
    .default_descr = &sfloat_default_descr,
    .discover_descr_from_pyobject = &sfloat_discover_from_pyobject,
    .hdiscover_descr_from_pyobject = &sfloat_discover_from_hpy,
    .is_known_scalar_type = &sfloat_is_known_scalar_type,
    .common_dtype = &sfloat_common_dtype,
    .hpy_common_dtype = &hpy_sfloat_common_dtype,
    .common_instance = &sfloat_common_instance,
    .hpy_common_instance = &hpy_sfloat_common_instance,
    .f = {
        .getitem = (PyArray_GetItemFunc *)&sfloat_getitem,
        .setitem = (PyArray_SetItemFunc *)&sfloat_setitem,
    }
};

static HPy // PyArray_Descr *
sfloat_scaled_copy(HPyContext *ctx, HPy /* PyArray_SFloatDescr * */ h_self, double factor) {
    PyArray_SFloatDescr *new;
    HPy h_PyArray_SFloatDType = HPyGlobal_Load(ctx, HPyArray_SFloatDType);
    HPy h_new = HPy_New(ctx, h_PyArray_SFloatDType, &new);
    HPy_Close(ctx, h_PyArray_SFloatDType);
    if (HPy_IsNull(h_new)) {
        return HPy_NULL;
    }
    PyArray_SFloatDescr *self = PyArray_SFloatDescr_AsStruct(ctx, h_self);
    /* Don't copy PyObject_HEAD part */
    memcpy((char *)new + sizeof(PyObject),
            (char *)self + sizeof(PyObject),
            sizeof(PyArray_SFloatDescr) - sizeof(PyObject));

    new->scaling = new->scaling * factor;
    return h_new;
}


HPyDef_METH(python_sfloat_scaled_copy, "scaled_by", HPyFunc_O)
HPy
python_sfloat_scaled_copy_impl(HPyContext *ctx, HPy /* PyArray_SFloatDescr * */ h_self, HPy arg)
{
    HPy arg_type = HPy_Type(ctx, arg);
    if (!HPy_Is(ctx, arg_type, ctx->h_FloatType)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Scaling factor must be a python float.");
        return HPy_NULL;
    }
    double factor = HPyFloat_AsDouble(ctx, arg);

    return sfloat_scaled_copy(ctx, h_self, factor);
}


HPyDef_METH(sfloat_get_scaling, "get_scaling", HPyFunc_NOARGS)
static HPy
sfloat_get_scaling_impl(HPyContext *ctx, HPy /* PyArray_SFloatDescr * */ h_self)
{
    PyArray_SFloatDescr *self = PyArray_SFloatDescr_AsStruct(ctx, h_self);
    return HPyFloat_FromDouble(ctx, self->scaling);
}

HPyDef_SLOT(sfloat_new, sfloat_new_impl, HPy_tp_new)
static HPy
sfloat_new_impl(HPyContext *ctx, HPy NPY_UNUSED(cls), HPy *args_h,
                          HPy_ssize_t nargs, HPy kwds)
{
    double scaling = 1.;
    static const char *kwargs_strs[] = {"scaling", NULL};


    if (!HPyArg_ParseKeywords(ctx, NULL,
            args_h, nargs, kwds, "|d:_ScaledFloatTestDType", kwargs_strs, &scaling)) {
        return HPy_NULL;
    }
    HPy h_SFloatSingleton = HPyGlobal_Load(ctx, SFloatSingleton);
    if (scaling == 1.) {
        return h_SFloatSingleton;
    }
    HPy ret = sfloat_scaled_copy(ctx, h_SFloatSingleton, scaling);
    HPy_Close(ctx, h_SFloatSingleton);
    return ret;
}


HPyDef_SLOT(sfloat_repr, sfloat_repr_impl, HPy_tp_repr)
static HPy
sfloat_repr_impl(HPyContext *ctx, HPy /* PyArray_SFloatDescr * */ h_self)
{
    PyArray_SFloatDescr *self = PyArray_SFloatDescr_AsStruct(ctx, h_self);
    HPy scaling = HPyFloat_FromDouble(ctx, self->scaling);
    if (HPy_IsNull(scaling)) {
        return HPy_NULL;
    }
    CAPI_WARN("missing PyUnicode_FromFormat");
    PyObject *py_scaling = HPy_AsPyObject(ctx, scaling);
    HPy_Close(ctx, scaling);
    PyObject *res = PyUnicode_FromFormat(
            "_ScaledFloatTestDType(scaling=%R)", py_scaling);
    HPy h_res = HPy_FromPyObject(ctx, res);
    Py_DECREF(py_scaling);
    Py_DECREF(res);
    return h_res;
}

static PyObject *
sfloat_str(PyArray_SFloatDescr *self)
{
    HPyContext *ctx = npy_get_context(); 
    HPy h_self = HPy_FromPyObject(ctx, self);
    HPy h_ret = sfloat_repr_impl(ctx, h_self);
    PyObject *ret = HPy_AsPyObject(ctx, h_ret);
    HPy_Close(ctx, h_self);
    HPy_Close(ctx, h_ret);
    return ret;
}
/*
 * Implement some casts.
 */

/*
 * It would make more sense to test this early on, but this allows testing
 * error returns.
 */
static int
check_factor(HPyContext *ctx, double factor) {
    if (npy_isfinite(factor) && factor != 0.) {
        return 0;
    }
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    HPyErr_SetString(ctx, ctx->h_TypeError,
            "error raised inside the core-loop: non-finite factor!");
    NPY_DISABLE_C_API;
    return -1;
}


static int
cast_sfloat_to_sfloat_unaligned(HPyContext *ctx, HPyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /* could also be moved into auxdata: */
    double factor = PyArray_SFloatDescr_AsStruct(ctx, context->descriptors[0])->scaling;
    factor /= PyArray_SFloatDescr_AsStruct(ctx, context->descriptors[1])->scaling;
    if (check_factor(ctx, factor) < 0) {
        return -1;
    }

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++) {
        double tmp;
        memcpy(&tmp, in, sizeof(double));
        tmp *= factor;
        memcpy(out, &tmp, sizeof(double));

        in += strides[0];
        out += strides[1];
    }
    return 0;
}


static int
cast_sfloat_to_sfloat_aligned(HPyContext *ctx, HPyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /* could also be moved into auxdata: */
    double factor = PyArray_SFloatDescr_AsStruct(ctx, context->descriptors[0])->scaling;
    factor /= PyArray_SFloatDescr_AsStruct(ctx, context->descriptors[1])->scaling;
    if (check_factor(ctx, factor) < 0) {
        return -1;
    }

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++) {
        *(double *)out = *(double *)in * factor;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}


static NPY_CASTING
sfloat_to_sfloat_resolve_descriptors(
            HPyContext *ctx,
            HPy NPY_UNUSED(self), // PyArrayMethodObject *
            HPy NPY_UNUSED(dtypes[2]), // PyArray_DTypeMeta *
            HPy given_descrs[2], // PyArray_Descr *
            HPy loop_descrs[2], // PyArray_Descr *
            npy_intp *view_offset)
{
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);

    if (HPy_IsNull(given_descrs[1])) {
        loop_descrs[1] = HPy_Dup(ctx, given_descrs[0]);
    }
    else {
        loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);
    }

    PyArray_Descr *loop_descrs_0 = PyArray_Descr_AsStruct(ctx, loop_descrs[0]);
    PyArray_Descr *loop_descrs_1 = PyArray_Descr_AsStruct(ctx, loop_descrs[1]);
    if (((PyArray_SFloatDescr *)loop_descrs_0)->scaling
            == ((PyArray_SFloatDescr *)loop_descrs_1)->scaling) {
        /* same scaling is just a view */
        *view_offset = 0;
        return NPY_NO_CASTING;
    }
    else if (-((PyArray_SFloatDescr *)loop_descrs_0)->scaling
             == ((PyArray_SFloatDescr *)loop_descrs_1)->scaling) {
        /* changing the sign does not lose precision */
        return NPY_EQUIV_CASTING;
    }
    /* Technically, this is not a safe cast, since over/underflows can occur */
    return NPY_SAME_KIND_CASTING;
}


/*
 * Casting to and from doubles.
 *
 * To keep things interesting, we ONLY define the trivial cast with a factor
 * of 1.  All other casts have to be handled by the sfloat to sfloat cast.
 *
 * The casting machinery should optimize this step away normally, since we
 * flag the this is a view.
 */
static int
cast_float_to_from_sfloat(HPyContext *ctx, HPyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++) {
        *(double *)out = *(double *)in;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}


static NPY_CASTING
float_to_from_sfloat_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), // PyArrayMethodObject *
        HPy dtypes[2], // PyArray_DTypeMeta *
        HPy NPY_UNUSED(given_descrs[2]), // PyArray_Descr *
        HPy loop_descrs[2], // PyArray_Descr *
        npy_intp *view_offset)
{
    PyArray_DTypeMeta *dtypes_0 = PyArray_DTypeMeta_AsStruct(ctx, dtypes[0]);
    loop_descrs[0] = HNPY_DT_CALL_default_descr(ctx, dtypes[0], dtypes_0);
    if (HPy_IsNull(loop_descrs[0])) {
        return -1;
    }
    PyArray_DTypeMeta *dtypes_1 = PyArray_DTypeMeta_AsStruct(ctx, dtypes[1]);
    loop_descrs[1] = HNPY_DT_CALL_default_descr(ctx, dtypes[1], dtypes_1);
    if (HPy_IsNull(loop_descrs[1])) {
        return -1;
    }
    *view_offset = 0;
    return NPY_NO_CASTING;
}


/*
 * Cast to boolean (for testing the logical functions a bit better).
 */
static int
cast_sfloat_to_bool(HPyContext *ctx, HPyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++) {
        *(npy_bool *)out = *(double *)in != 0;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

static NPY_CASTING
sfloat_to_bool_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), // PyArrayMethodObject *
        HPy NPY_UNUSED(dtypes[2]), // PyArray_DTypeMeta *
        HPy given_descrs[2], // PyArray_Descr *
        HPy loop_descrs[2], // PyArray_Descr *
        npy_intp *NPY_UNUSED(view_offset))
{
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);
    if (HPy_IsNull(loop_descrs[0])) {
        return -1;
    }
    loop_descrs[1] = HPyArray_DescrFromType(ctx, NPY_BOOL);  /* cannot fail */
    return NPY_UNSAFE_CASTING;
}


static int
init_casts(HPyContext *ctx, HPy PyArray_SFloatDType_arg)
{
    // PyArray_DTypeMeta *dtypes[2] = {&PyArray_SFloatDType, &PyArray_SFloatDType};
    HPy h_PyArray_SFloatDType = HPy_Dup(ctx, PyArray_SFloatDType_arg);
    HPy dtypes[2] = {
            h_PyArray_SFloatDType, 
            h_PyArray_SFloatDType, 
    };
    PyType_Slot slots[4] = {{0, NULL}};
    PyArrayMethod_Spec spec = {
        .name = "sfloat_to_sfloat_cast",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = dtypes,
        .slots = slots,
        /* minimal guaranteed casting */
        .casting = NPY_SAME_KIND_CASTING,
    };

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &sfloat_to_sfloat_resolve_descriptors;

    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &cast_sfloat_to_sfloat_aligned;

    slots[2].slot = NPY_METH_unaligned_strided_loop;
    slots[2].pfunc = &cast_sfloat_to_sfloat_unaligned;

    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        HPy_Close(ctx, dtypes[0]);  /* immortal anyway */
        return -1;
    }

    spec.name = "float_to_sfloat_cast";
    /* Technically, it is just a copy currently so this is fine: */
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    HPy double_DType = HPyArray_DTypeFromTypeNum(ctx, NPY_DOUBLE); // PyArray_DTypeMeta *
    // HPy_Close(ctx, double_DType);  /* immortal anyway */
    dtypes[0] = double_DType;

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &float_to_from_sfloat_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &cast_float_to_from_sfloat;
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        HPy_Close(ctx, dtypes[0]);
        HPy_Close(ctx, dtypes[1]);
        return -1;
    }

    spec.name = "sfloat_to_float_cast";
    dtypes[0] = h_PyArray_SFloatDType;
    dtypes[1] = double_DType;

    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        HPy_Close(ctx, dtypes[0]);
        HPy_Close(ctx, dtypes[1]);
        return -1;
    }
    HPy_Close(ctx, dtypes[1]);

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &sfloat_to_bool_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &cast_sfloat_to_bool;
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    spec.name = "sfloat_to_bool_cast";
    dtypes[0] = h_PyArray_SFloatDType;
    dtypes[1] = HPyArray_DTypeFromTypeNum(ctx, NPY_BOOL);

    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        HPy_Close(ctx, dtypes[0]);
        HPy_Close(ctx, dtypes[1]);
        return -1;
    }
    HPy_Close(ctx, dtypes[0]);
    HPy_Close(ctx, dtypes[1]);

    return 0;
}


/*
 * We also wish to test very simple ufunc functionality.  So create two
 * ufunc loops:
 * 1. Multiplication, which can multiply the factors and work with that.
 * 2. Addition, which needs to use the common instance, and runs into
 *    cast safety subtleties since we will implement it without an additional
 *    cast.
 */
static int
multiply_sfloats(HPyContext *ctx, HPyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    for (npy_intp i = 0; i < N; i++) {
        *(double *)out = *(double *)in1 * *(double *)in2;
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}


static NPY_CASTING
multiply_sfloats_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), // PyArrayMethodObject *
        HPy NPY_UNUSED(dtypes[3]), // PyArray_DTypeMeta *
        HPy given_descrs[3], // PyArray_Descr *
        HPy loop_descrs[3], // PyArray_Descr *
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * Multiply the scaling for the result.  If the result was passed in we
     * simply ignore it and let the casting machinery fix it up here.
     */
    PyArray_SFloatDescr *given_descrs_1 = PyArray_SFloatDescr_AsStruct(ctx, given_descrs[1]);
    double factor = given_descrs_1->scaling;
    loop_descrs[2] = sfloat_scaled_copy(ctx, given_descrs[0], factor);
    if (HPy_IsNull(loop_descrs[2])) {
        return -1;
    }
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);
    loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);
    return NPY_NO_CASTING;
}


/*
 * Unlike the multiplication implementation above, this loops deals with
 * scaling (casting) internally.  This allows to test some different paths.
 */
static int
add_sfloats(
    HPyContext *ctx, HPyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    double fin1 = PyArray_SFloatDescr_AsStruct(ctx, context->descriptors[0])->scaling;
    double fin2 = PyArray_SFloatDescr_AsStruct(ctx, context->descriptors[1])->scaling;
    double fout = PyArray_SFloatDescr_AsStruct(ctx, context->descriptors[2])->scaling;

    double fact1 = fin1 / fout;
    double fact2 = fin2 / fout;
    if (check_factor(ctx, fact1) < 0) {
        return -1;
    }
    if (check_factor(ctx, fact2) < 0) {
        return -1;
    }

    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    for (npy_intp i = 0; i < N; i++) {
        *(double *)out = (*(double *)in1 * fact1) + (*(double *)in2 * fact2);
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}


static NPY_CASTING
add_sfloats_resolve_descriptors(
        HPyContext *ctx,
        HPy NPY_UNUSED(self), // PyArrayMethodObject *
        HPy NPY_UNUSED(dtypes[3]), // PyArray_DTypeMeta *
        HPy given_descrs[3], // PyArray_Descr *
        HPy loop_descrs[3], // PyArray_Descr *
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * Here we accept an output descriptor (the inner loop can deal with it),
     * if none is given, we use the "common instance":
     */
    if (HPy_IsNull(given_descrs[2])) {
        loop_descrs[2] = hpy_sfloat_common_instance(ctx,
                given_descrs[0], given_descrs[1]);
        if (HPy_IsNull(loop_descrs[2])) {
            return -1;
        }
    }
    else {
        loop_descrs[2] = HPy_Dup(ctx, given_descrs[2]);
    }
    loop_descrs[0] = HPy_Dup(ctx, given_descrs[0]);
    loop_descrs[1] = HPy_Dup(ctx, given_descrs[1]);

    /* If the factors mismatch, we do implicit casting inside the ufunc! */
    double fin1 = PyArray_SFloatDescr_AsStruct(ctx, loop_descrs[0])->scaling;
    double fin2 = PyArray_SFloatDescr_AsStruct(ctx, loop_descrs[1])->scaling;
    double fout = PyArray_SFloatDescr_AsStruct(ctx, loop_descrs[2])->scaling;

    if (fin1 == fout && fin2 == fout) {
        return NPY_NO_CASTING;
    }
    if (npy_fabs(fin1) == npy_fabs(fout) && npy_fabs(fin2) == npy_fabs(fout)) {
        return NPY_EQUIV_CASTING;
    }
    return NPY_SAME_KIND_CASTING;
}


static int
add_loop(const char *ufunc_name,
        PyArray_DTypeMeta *dtypes[3], PyObject *meth_or_promoter)
{
    PyObject *mod = PyImport_ImportModule("numpy");
    if (mod == NULL) {
        return -1;
    }
    PyObject *ufunc = PyObject_GetAttrString(mod, ufunc_name);
    Py_DECREF(mod);
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        Py_DECREF(ufunc);
        PyErr_Format(PyExc_TypeError,
                "numpy.%s was not a ufunc!", ufunc_name);
        return -1;
    }
    PyObject *dtype_tup = PyArray_TupleFromItems(3, (PyObject **)dtypes, 1);
    if (dtype_tup == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }
    PyObject *info = PyTuple_Pack(2, dtype_tup, meth_or_promoter);
    Py_DECREF(dtype_tup);
    if (info == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }
    int res = PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
    Py_DECREF(ufunc);
    Py_DECREF(info);
    return res;
}

#include "ufunc_object.h" // remove once HPyUFunc_Type in 'ufuncobject.h'

static int
hpy_add_loop(HPyContext *ctx, const char *ufunc_name,
        HPy /* PyArray_DTypeMeta * */dtypes[3], HPy meth_or_promoter)
{
    HPy mod = HPyImport_ImportModule(ctx, "numpy");
    if (HPy_IsNull(mod)) {
        return -1;
    }
    HPy ufunc = HPy_GetAttr_s(ctx, mod, ufunc_name);
    HPy_Close(ctx, mod);
    HPy ufunc_type = HPyGlobal_Load(ctx, HPyUFunc_Type);
    if (!HPy_TypeCheck(ctx, ufunc, ufunc_type)) {
        HPy_Close(ctx, ufunc);
        HPy_Close(ctx, ufunc_type);
        PyErr_Format(PyExc_TypeError,
                "numpy.%s was not a ufunc!", ufunc_name);
        return -1;
    }
    HPy_Close(ctx, ufunc_type);
    HPy dtype_tup = HPyArray_TupleFromItems(ctx, 3, dtypes, 1);
    if (HPy_IsNull(dtype_tup)) {
        HPy_Close(ctx, ufunc);
        return -1;
    }
    HPy info = HPyTuple_Pack(ctx, 2, dtype_tup, meth_or_promoter);
    HPy_Close(ctx, dtype_tup);
    if (HPy_IsNull(info)) {
        HPy_Close(ctx, ufunc);
        return -1;
    }
    int res = HPyUFunc_AddLoop(ctx, ufunc, info, 0);
    HPy_Close(ctx, ufunc);
    HPy_Close(ctx, info);
    return res;
}



/*
 * We add some very basic promoters to allow multiplying normal and scaled
 */
static int
hpy_promote_to_sfloat(HPyContext *ctx, HPy /* PyUFuncObject * */ NPY_UNUSED(ufunc),
        HPy /* PyArray_DTypeMeta * */ const NPY_UNUSED(dtypes[3]),
        HPy /* PyArray_DTypeMeta * */ const signature[3],
        HPy /* PyArray_DTypeMeta * */ new_dtypes[3])
{
    HPy sfloatDType = HPyGlobal_Load(ctx, HPyArray_SFloatDType);
    for (int i = 0; i < 3; i++) {
        new_dtypes[i] = HPy_Dup(ctx, !HPy_IsNull(signature[i]) ? signature[i] : sfloatDType);
    }
    HPy_Close(ctx, sfloatDType);
    return 0;
}


/*
 * Add new ufunc loops (this is somewhat clumsy as of writing it, but should
 * get less so with the introduction of public API).
 */
static int
init_ufuncs(HPyContext *ctx) {
    // PyArray_DTypeMeta *dtypes[3] = {
    //         &PyArray_SFloatDType, &PyArray_SFloatDType, &PyArray_SFloatDType};
    HPy h_PyArray_SFloatDType = HPyGlobal_Load(ctx, HPyArray_SFloatDType);
    HPy dtypes[3] = {
            h_PyArray_SFloatDType, 
            h_PyArray_SFloatDType, 
            h_PyArray_SFloatDType
    };
    PyType_Slot slots[3] = {{0, NULL}};
    PyArrayMethod_Spec spec = {
        .nin = 2,
        .nout =1,
        .dtypes = dtypes,
        .slots = slots,
    };
    spec.name = "sfloat_multiply";
    spec.casting = NPY_NO_CASTING;

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &multiply_sfloats_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &multiply_sfloats;
    HPy h_bmeth = HPyArrayMethod_FromSpec_int(ctx, &spec, 0);
    if (HPy_IsNull(h_bmeth)) {
        return -1;
    }
    PyBoundArrayMethodObject *bmeth = PyBoundArrayMethodObject_AsStruct(ctx, h_bmeth);
    int res = -1;
    HPy h_bmeth_dtypes[] = {
        HPyField_Load(ctx, h_bmeth, bmeth->dtypes[0]),
        HPyField_Load(ctx, h_bmeth, bmeth->dtypes[1]),
        HPyField_Load(ctx, h_bmeth, bmeth->dtypes[2]),
    };
    HPy h_bmeth_method = HPyField_Load(ctx, h_bmeth, bmeth->method);
    res = hpy_add_loop(ctx, "multiply", h_bmeth_dtypes, h_bmeth_method);
    HPy_Close(ctx, h_bmeth_dtypes[0]);
    HPy_Close(ctx, h_bmeth_dtypes[1]);
    HPy_Close(ctx, h_bmeth_dtypes[2]);
    HPy_Close(ctx, h_bmeth_method);
    HPy_Close(ctx, h_bmeth);
    if (res < 0) {
        return -1;
    }

    spec.name = "sfloat_add";
    spec.casting = NPY_SAME_KIND_CASTING;

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &add_sfloats_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &add_sfloats;
    h_bmeth = HPyArrayMethod_FromSpec_int(ctx, &spec, 0);
    bmeth = PyBoundArrayMethodObject_AsStruct(ctx, h_bmeth);
    if (HPy_IsNull(h_bmeth)) {
        return -1;
    }
    h_bmeth_dtypes[0] = HPyField_Load(ctx, h_bmeth, bmeth->dtypes[0]),
    h_bmeth_dtypes[1] = HPyField_Load(ctx, h_bmeth, bmeth->dtypes[1]),
    h_bmeth_dtypes[2] = HPyField_Load(ctx, h_bmeth, bmeth->dtypes[2]),
    h_bmeth_method = HPyField_Load(ctx, h_bmeth, bmeth->method);
    res = hpy_add_loop(ctx, "add",
            h_bmeth_dtypes, h_bmeth_method);
    HPy_Close(ctx, h_bmeth_dtypes[0]);
    HPy_Close(ctx, h_bmeth_dtypes[1]);
    HPy_Close(ctx, h_bmeth_dtypes[2]);
    HPy_Close(ctx, h_bmeth_method);
    HPy_Close(ctx, h_bmeth);
    if (res < 0) {
        return -1;
    }

    /*
     * Add a promoter for both directions of multiply with double.
     */
    HPy double_DType = HPyArray_DTypeFromTypeNum(ctx, NPY_DOUBLE);
    // Py_DECREF(double_DType);  /* immortal anyway */

    HPy promoter_dtypes[3] = { // PyArray_DTypeMeta *
            h_PyArray_SFloatDType, double_DType, HPy_NULL};

    HPy promoter = HPyCapsule_New(ctx,
            &hpy_promote_to_sfloat, "numpy._ufunc_promoter", NULL);
    if (HPy_IsNull(promoter)) {
        return -1;
    }
    res = hpy_add_loop(ctx, "multiply", promoter_dtypes, promoter);
    if (res < 0) {
        HPy_Close(ctx, promoter);
        HPy_Close(ctx, double_DType);
        HPy_Close(ctx, h_PyArray_SFloatDType);
        return -1;
    }
    promoter_dtypes[0] = double_DType;
    promoter_dtypes[1] = h_PyArray_SFloatDType;
    res = hpy_add_loop(ctx, "multiply", promoter_dtypes, promoter);
    HPy_Close(ctx, promoter);
    HPy_Close(ctx, double_DType);
    HPy_Close(ctx, h_PyArray_SFloatDType);
    if (res < 0) {
        return -1;
    }

    return 0;
}


static HPyDef *sfloat_defines[] = {
    &sfloat_new,
    &sfloat_repr,

    // methods:
    &python_sfloat_scaled_copy,
    &sfloat_get_scaling,
    NULL,
};

static PyType_Slot sfloat_slots_legacy[] = {
    // HPy TODO: add HPy_tp_str
    {Py_tp_str, sfloat_str},
    {0},
};

/*
 * Python entry point, exported via `umathmodule.h` and `multiarraymodule.c`.
 * TODO: Should be moved when the necessary API is not internal anymore.
 */
HPyDef_METH(get_sfloat_dtype, "_get_sfloat_dtype", HPyFunc_NOARGS)
NPY_NO_EXPORT HPy
get_sfloat_dtype_impl(HPyContext *ctx, HPy NPY_UNUSED(mod))
{
    /* Allow calling the function multiple times. */
    static npy_bool initialized = NPY_FALSE;

    if (initialized) {
        return HPyGlobal_Load(ctx, HPyArray_SFloatDType);
    }

    // PyArray_SFloatDType.super.ht_type.tp_base = &PyArrayDescr_Type;
    HPy h_PyArrayDescr_Type = HPyGlobal_Load(ctx, HPyArrayDescr_Type);

    static HPyType_Spec PyArray_SFloatDType_spec = {
        .name = "numpy._ScaledFloatTestDType",
        .basicsize = sizeof(PyArray_SFloatDescr),
        .flags = HPy_TPFLAGS_DEFAULT,
        .defines = sfloat_defines,
        .legacy_slots = sfloat_slots_legacy,
        .legacy = true,
    };

    HPy h_PyArrayDTypeMeta_Type = HPyGlobal_Load(ctx, HPyArrayDTypeMeta_Type);
    HPyType_SpecParam dtypemeta_params[] = {
        { HPyType_SpecParam_Base, h_PyArrayDescr_Type },
        { HPyType_SpecParam_Metaclass, h_PyArrayDTypeMeta_Type },
        { 0 }
    };

    HPy h_PyArray_SFloatDType = HPyType_FromSpec(ctx, 
                            &PyArray_SFloatDType_spec, dtypemeta_params);
    HPy_Close(ctx, h_PyArrayDescr_Type);
    HPy_Close(ctx, h_PyArrayDTypeMeta_Type);
    if (HPy_IsNull(h_PyArray_SFloatDType)) {
        return HPy_NULL;
    }
    PyArray_DTypeMeta *h_PyArray_SFloatDType_data = 
                            PyArray_DTypeMeta_AsStruct(ctx, h_PyArray_SFloatDType);
    h_PyArray_SFloatDType_data->type_num = -1;
    h_PyArray_SFloatDType_data->scalar_type = HPyField_NULL;
    h_PyArray_SFloatDType_data->flags = NPY_DT_PARAMETRIC;
    h_PyArray_SFloatDType_data->dt_slots = &sfloat_slots;
    HPy h_castingimpls = HPyDict_New(ctx);
    if (HPy_IsNull(h_castingimpls)) {
        HPy_Close(ctx, h_PyArray_SFloatDType);
        return HPy_NULL;
    }
    HPyField_Store(ctx, h_PyArray_SFloatDType, 
                        &NPY_DT_SLOTS(h_PyArray_SFloatDType_data)->castingimpls, h_castingimpls);
    HPy_Close(ctx, h_castingimpls);
    HPyGlobal_Store(ctx, &HPyArray_SFloatDType, h_PyArray_SFloatDType);
    PyArray_SFloatDType = (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_PyArray_SFloatDType);
    PyArray_SFloatDescr *h_SFloatSingleton_data;
    HPy h_SFloatSingleton = HPy_New(ctx, h_PyArray_SFloatDType, &h_SFloatSingleton_data);
    h_SFloatSingleton_data->base.elsize = sizeof(double);
    h_SFloatSingleton_data->base.alignment = _ALIGN(double);
    h_SFloatSingleton_data->base.flags = NPY_USE_GETITEM|NPY_USE_SETITEM;
    h_SFloatSingleton_data->base.type_num = -1;
    h_SFloatSingleton_data->base.f = &sfloat_slots.f;
    h_SFloatSingleton_data->base.byteorder = '|';  /* do not bother with byte-swapping... */
    h_SFloatSingleton_data->scaling = 1;
    PyObject *py_SFloatSingleton = HPy_AsPyObject(ctx, h_SFloatSingleton);
    PyObject *py_PyArray_SFloatDType = HPy_AsPyObject(ctx, h_PyArray_SFloatDType);
    // CAPI_WARN("missing PyObject_Init");
    // PyObject *o = PyObject_Init(
    //         (PyObject *)py_SFloatSingleton, (PyTypeObject *)py_PyArray_SFloatDType);
    HPyGlobal_Store(ctx, &SFloatSingleton, h_SFloatSingleton);
    Py_DECREF(py_SFloatSingleton);
    Py_DECREF(py_PyArray_SFloatDType);
    HPy_Close(ctx, h_SFloatSingleton);
    // if (o == NULL) {
    //     return HPy_NULL;
    // }

    if (init_casts(ctx, h_PyArray_SFloatDType) < 0) {
        return HPy_NULL;
    }

    if (init_ufuncs(ctx) < 0) {
        return HPy_NULL;
    }

    initialized = NPY_TRUE;
    return h_PyArray_SFloatDType;
}

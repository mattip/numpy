/*
 * This file implements an abstraction layer for "Array methods", which
 * work with a specific DType class input and provide low-level C function
 * pointers to do fast operations on the given input functions.
 * It thus adds an abstraction layer around individual ufunc loops.
 *
 * Unlike methods, a ArrayMethod can have multiple inputs and outputs.
 * This has some serious implication for garbage collection, and as far
 * as I (@seberg) understands, it is not possible to always guarantee correct
 * cyclic garbage collection of dynamically created DTypes with methods.
 * The keyword (or rather the solution) for this seems to be an "ephemeron"
 * which I believe should allow correct garbage collection but seems
 * not implemented in Python at this time.
 * The vast majority of use-cases will not require correct garbage collection.
 * Some use cases may require the user to be careful.
 *
 * Generally there are two main ways to solve this issue:
 *
 * 1. A method with a single input (or inputs of all the same DTypes) can
 *    be "owned" by that DType (it becomes unusable when the DType is deleted).
 *    This holds especially for all casts, which must have a defined output
 *    DType and must hold on to it strongly.
 * 2. A method which can infer the output DType(s) from the input types does
 *    not need to keep the output type alive. (It can use NULL for the type,
 *    or an abstract base class which is known to be persistent.)
 *    It is then sufficient for a ufunc (or other owner) to only hold a
 *    weak reference to the input DTypes.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <npy_pycompat.h>
#include "arrayobject.h"
#include "array_method.h"
#include "dtypemeta.h"
#include "common_dtype.h"
#include "convert_datatype.h"
#include "common.h"


/*
 * The default descriptor resolution function.  The logic is as follows:
 *
 * 1. The output is ensured to be canonical (currently native byte order),
 *    if it is of the correct DType.
 * 2. If any DType is was not defined, it is replaced by the common DType
 *    of all inputs. (If that common DType is parametric, this is an error.)
 *
 * We could allow setting the output descriptors specifically to simplify
 * this step.
 *
 * Note that the default version will indicate that the cast can be done
 * as using `arr.view(new_dtype)` if the default cast-safety is
 * set to "no-cast".  This default function cannot be used if a view may
 * be sufficient for casting but the cast is not always "no-cast".
 */
static NPY_CASTING
default_resolve_descriptors(
        HPyContext *ctx,
        HPy method, /* (struct PyArrayMethodObject_tag *) */
        HPy *dtypes, /* PyArray_DTypeMeta **dtypes */
        HPy *input_descrs, /* PyArray_Descr **given_descrs */
        HPy *output_descrs, /* PyArray_Descr **loop_descrs */
        npy_intp *view_offset)
{
    PyArrayMethodObject *method_data = PyArrayMethodObject_AsStruct(ctx, method);
    int nin = method_data->nin;
    int nout = method_data->nout;

    for (int i = 0; i < nin + nout; i++) {
        HPy dtype = dtypes[i]; /* (PyArray_DTypeMeta *) */
        if (!HPy_IsNull(input_descrs[i])) {
            output_descrs[i] = hensure_dtype_nbo(ctx, input_descrs[i]);
        }
        else {
            output_descrs[i] = hdtypemeta_call_default_descr(ctx, dtype);
        }
        if (NPY_UNLIKELY(HPy_IsNull(output_descrs[i]))) {
            goto fail;
        }
    }
    /*
     * If we relax the requirement for specifying all `dtypes` (e.g. allow
     * abstract ones or unspecified outputs).  We can use the common-dtype
     * operation to provide a default here.
     */
    if (method_data->casting == NPY_NO_CASTING) {
        /*
         * By (current) definition no-casting should imply viewable.  This
         * is currently indicated for example for object to object cast.
         */
        *view_offset = 0;
    }
    return method_data->casting;

  fail:
    for (int i = 0; i < nin + nout; i++) {
        HPy_Close(ctx, output_descrs[i]);
    }
    return -1;
}


NPY_INLINE static int
is_contiguous(HPyContext *ctx,
        npy_intp const *strides, HPy const *descriptors, int nargs)
{
    for (int i = 0; i < nargs; i++) {
        PyArray_Descr *descriptor_data = PyArray_Descr_AsStruct(ctx, descriptors[i]);
        if (strides[i] != descriptor_data->elsize) {
            return 0;
        }
    }
    return 1;
}


/**
 * The default method to fetch the correct loop for a cast or ufunc
 * (at the time of writing only casts).
 * Note that the default function provided here will only indicate that a cast
 * can be done as a view (i.e., arr.view(new_dtype)) when this is trivially
 * true, i.e., for cast safety "no-cast". It will not recognize view as an
 * option for other casts (e.g., viewing '>i8' as '>i4' with an offset of 4).
 *
 * @param hctx the HPy context
 * @param context
 * @param aligned
 * @param move_references UNUSED.
 * @param strides
 * @param descriptors
 * @param out_loop
 * @param out_transferdata
 * @param flags
 * @return 0 on success -1 on failure.
 */
NPY_NO_EXPORT int
npy_default_get_strided_loop(
        HPyContext *hctx,
        HPyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        HPyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    HPy *descrs = context->descriptors; /* (PyArray_Descr **) */
    HPy method = context->method; /* (PyArrayMethodObject *) */
    PyArrayMethodObject *meth_data = PyArrayMethodObject_AsStruct(hctx, method);
    *flags = meth_data->flags & NPY_METH_RUNTIME_FLAGS;
    *out_transferdata = NULL;

    int nargs = meth_data->nin + meth_data->nout;
    if (aligned) {
        if (meth_data->contiguous_loop == NULL ||
                !is_contiguous(hctx, strides, descrs, nargs)) {
            *out_loop = meth_data->strided_loop;
            return 0;
        }
        *out_loop = meth_data->contiguous_loop;
    }
    else {
        if (meth_data->unaligned_contiguous_loop == NULL ||
                !is_contiguous(hctx, strides, descrs, nargs)) {
            *out_loop = meth_data->unaligned_strided_loop;
            return 0;
        }
        *out_loop = meth_data->unaligned_contiguous_loop;
    }
    return 0;
}


/**
 * Validate that the input is usable to create a new ArrayMethod.
 *
 * @param spec
 * @return 0 on success -1 on error.
 */
static int
validate_spec(HPyContext *ctx, PyArrayMethod_Spec *spec)
{
    int nargs = spec->nin + spec->nout;
    /* Check the passed spec for invalid fields/values */
    if (spec->nin < 0 || spec->nout < 0 || nargs > NPY_MAXARGS) {
        HPyErr_Format_p(ctx, ctx->h_ValueError,
                "ArrayMethod inputs and outputs must be greater zero and"
                "not exceed %d. (method: %s)", NPY_MAXARGS, spec->name);
        return -1;
    }
    switch (spec->casting) {
        case NPY_NO_CASTING:
        case NPY_EQUIV_CASTING:
        case NPY_SAFE_CASTING:
        case NPY_SAME_KIND_CASTING:
        case NPY_UNSAFE_CASTING:
            break;
        default:
            if (spec->casting != -1) {
                HPyErr_Format_p(ctx, ctx->h_TypeError,
                        "ArrayMethod has invalid casting `%d`. (method: %s)",
                        spec->casting, spec->name);
                return -1;
            }
    }

    HPy dtypemeta = HPyGlobal_Load(ctx, HPyArrayDTypeMeta_Type);
    for (int i = 0; i < nargs; i++) {
        /*
         * Note that we could allow for output dtypes to not be specified
         * (the array-method would have to make sure to support this).
         * We could even allow for some dtypes to be abstract.
         * For now, assume that this is better handled in a promotion step.
         * One problem with providing all DTypes is the definite need to
         * hold references.  We probably, eventually, have to implement
         * traversal and trust the GC to deal with it.
         */
        if (HPy_IsNull(spec->dtypes[i])) {
            HPyErr_Format_p(ctx, ctx->h_TypeError,
                    "ArrayMethod must provide all input and output DTypes. "
                    "(method: %s)", spec->name);
            HPy_Close(ctx, dtypemeta);
            return -1;
        }
        if (!HPy_TypeCheck(ctx, spec->dtypes[i], dtypemeta)) {
            // TODO HPY LABS PORT: PyErr_Format
            HPyErr_Format_p(ctx, ctx->h_TypeError,
                    "ArrayMethod provided object XX is not a DType."
                    "(method: %s)", spec->name);
            // PyErr_Format(PyExc_TypeError,
            //         "ArrayMethod provided object %R is not a DType."
            //         "(method: %s)", spec->dtypes[i], spec->name);
            HPy_Close(ctx, dtypemeta);
            return -1;
        }
        if (NPY_DT_is_abstract(PyArray_DTypeMeta_AsStruct(ctx, spec->dtypes[i]))) {
            // TODO HPY LABS PORT: PyErr_Format
            HPyErr_Format_p(ctx, ctx->h_TypeError,
                    "abstract DType XX are currently not supported."
                    "(method: %s)", spec->name);
            // PyErr_Format(PyExc_TypeError,
            //         "abstract DType %S are currently not supported."
            //         "(method: %s)", spec->dtypes[i], spec->name);
            HPy_Close(ctx, dtypemeta);
            return -1;
        }
    }
    HPy_Close(ctx, dtypemeta);
    return 0;
}


/**
 * Initialize a new BoundArrayMethodObject from slots.  Slots which are
 * not provided may be filled with defaults.
 *
 * @param res The new PyBoundArrayMethodObject to be filled.
 * @param spec The specification list passed by the user.
 * @param private Private flag to limit certain slots to use in NumPy.
 * @return -1 on error 0 on success
 */
static int
fill_arraymethod_from_slots(HPyContext *ctx,
        HPy res, PyBoundArrayMethodObject *data, PyArrayMethod_Spec *spec,
        HPy h_meth, PyArrayMethodObject *meth,
        int private)
{
    /* Set the defaults */
    meth->get_strided_loop = &npy_default_get_strided_loop;
    meth->resolve_descriptors = &default_resolve_descriptors;

    /* Fill in the slots passed by the user */
    /*
     * TODO: This is reasonable for now, but it would be nice to find a
     *       shorter solution, and add some additional error checking (e.g.
     *       the same slot used twice). Python uses an array of slot offsets.
     */
    for (PyType_Slot *slot = &spec->slots[0]; slot->slot != 0; slot++) {
        switch (slot->slot) {
            case NPY_METH_resolve_descriptors:
                meth->resolve_descriptors = slot->pfunc;
                continue;
            case NPY_METH_get_loop:
                /*
                 * NOTE: get_loop is considered "unstable" in the public API,
                 *       I do not like the signature, and the `move_references`
                 *       parameter must NOT be used.
                 *       (as in: we should not worry about changing it, but of
                 *       course that would not break it immediately.)
                 */
                /* Only allow override for private functions initially */
                meth->get_strided_loop = slot->pfunc;
                continue;
            /* "Typical" loops, supported used by the default `get_loop` */
            case NPY_METH_strided_loop:
                meth->strided_loop = slot->pfunc;
                continue;
            case NPY_METH_contiguous_loop:
                meth->contiguous_loop = slot->pfunc;
                continue;
            case NPY_METH_unaligned_strided_loop:
                meth->unaligned_strided_loop = slot->pfunc;
                continue;
            case NPY_METH_unaligned_contiguous_loop:
                meth->unaligned_contiguous_loop = slot->pfunc;
                continue;
            default:
                break;
        }
        HPyErr_Format_p(ctx, ctx->h_RuntimeError,
                "invalid slot number %d to ArrayMethod: %s",
                slot->slot, spec->name);
        return -1;
    }

    /* Check whether the slots are valid: */
    if (meth->resolve_descriptors == &default_resolve_descriptors) {
        if (spec->casting == -1) {
            HPyErr_Format_p(ctx, ctx->h_TypeError,
                    "Cannot set casting to -1 (invalid) when not providing "
                    "the default `resolve_descriptors` function. "
                    "(method: %s)", spec->name);
            return -1;
        }
        for (int i = 0; i < meth->nin + meth->nout; i++) {
            if (HPyField_IsNull(data->dtypes[i])) {
                if (i < meth->nin) {
                    HPyErr_Format_p(ctx, ctx->h_TypeError,
                            "All input DTypes must be specified when using "
                            "the default `resolve_descriptors` function. "
                            "(method: %s)", spec->name);
                    return -1;
                }
                else if (meth->nin == 0) {
                    HPyErr_Format_p(ctx, ctx->h_TypeError,
                            "Must specify output DTypes or use custom "
                            "`resolve_descriptors` when there are no inputs. "
                            "(method: %s)", spec->name);
                    return -1;
                }
            }
            if (i >= meth->nin) {
                HPy h_descr = HPyField_Load(ctx, res, data->dtypes[i]);
                if (NPY_DT_is_parametric(PyArray_DTypeMeta_AsStruct(ctx, h_descr))) {
                    HPy_Close(ctx, h_descr);
                    HPyErr_Format_p(ctx, ctx->h_TypeError,
                            "must provide a `resolve_descriptors` function if any "
                            "output DType is parametric. (method: %s)",
                            spec->name);
                    return -1;
                }
                HPy_Close(ctx, h_descr);
            }
        }
    }
    if (meth->get_strided_loop != &npy_default_get_strided_loop) {
        /* Do not check the actual loop fields. */
        return 0;
    }

    /* Check whether the provided loops make sense. */
    if (meth->strided_loop == NULL) {
        HPyErr_Format_p(ctx, ctx->h_TypeError,
                "Must provide a strided inner loop function. (method: %s)",
                spec->name);
        return -1;
    }
    if (meth->contiguous_loop == NULL) {
        meth->contiguous_loop = meth->strided_loop;
    }
    if (meth->unaligned_contiguous_loop != NULL &&
            meth->unaligned_strided_loop == NULL) {
        HPyErr_Format_p(ctx, ctx->h_TypeError,
                "Must provide unaligned strided inner loop when providing "
                "a contiguous version. (method: %s)", spec->name);
        return -1;
    }
    if ((meth->unaligned_strided_loop == NULL) !=
            !(meth->flags & NPY_METH_SUPPORTS_UNALIGNED)) {
        HPyErr_Format_p(ctx, ctx->h_TypeError,
                "Must provide unaligned strided inner loop when providing "
                "a contiguous version. (method: %s)", spec->name);
        return -1;
    }

    return 0;
}


/*
 * Public version of `PyArrayMethod_FromSpec_int` (see below).
 *
 * TODO: Error paths will probably need to be improved before a release into
 *       the non-experimental public API.
 */
NPY_NO_EXPORT PyObject *
PyArrayMethod_FromSpec(PyArrayMethod_Spec *spec)
{
    HPyContext *ctx = npy_get_context();
    for (int i = 0; i < spec->nin + spec->nout; i++) {
        if (!HPyGlobal_TypeCheck(ctx, spec->dtypes[i], HPyArrayDTypeMeta_Type)) {
            HPyErr_SetString(ctx, ctx->h_RuntimeError,
                    "ArrayMethod spec contained a non DType.");
            return NULL;
        }
    }
    return (PyObject *)PyArrayMethod_FromSpec_int(spec, 0);
}

NPY_NO_EXPORT HPy
HPyArrayMethod_FromSpec(HPyContext *ctx, HPy spec, PyArrayMethod_Spec *spec_data)
{
    for (int i = 0; i < spec_data->nin + spec_data->nout; i++) {
        if (!HPyGlobal_TypeCheck(ctx, spec_data->dtypes[i], HPyArrayDTypeMeta_Type)) {
            HPyErr_SetString(ctx, ctx->h_RuntimeError,
                    "ArrayMethod spec contained a non DType.");
            return HPy_NULL;
        }
    }
    return HPyArrayMethod_FromSpec_int(ctx, spec_data, 0);
}


/**
 * Create a new ArrayMethod (internal version).
 *
 * @param name A name for the individual method, may be NULL.
 * @param spec A filled context object to pass generic information about
 *        the method (such as usually needing the API, and the DTypes).
 *        Unused fields must be NULL.
 * @param slots Slots with the correct pair of IDs and (function) pointers.
 * @param private Some slots are currently considered private, if not true,
 *        these will be rejected.
 *
 * @returns A new (bound) ArrayMethod object.
 */
NPY_NO_EXPORT PyBoundArrayMethodObject *
PyArrayMethod_FromSpec_int(PyArrayMethod_Spec *spec, int private)
{
    HPyContext *ctx = npy_get_context();
    HPy h_res = HPyArrayMethod_FromSpec_int(ctx, spec, private);
    PyBoundArrayMethodObject *res = (PyBoundArrayMethodObject *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    return res;
}

//NPY_NO_EXPORT PyBoundArrayMethodObject *
//PyArrayMethod_FromSpec_int(PyArrayMethod_Spec *spec, int private)
NPY_NO_EXPORT HPy
HPyArrayMethod_FromSpec_int(HPyContext *ctx, PyArrayMethod_Spec *spec, int private)
{
    int nargs = spec->nin + spec->nout;

    if (spec->name == NULL) {
        spec->name = "<unknown>";
    }

    if (validate_spec(ctx, spec) < 0) {
        return HPy_NULL;
    }

    HPy bound_array_method_type = HPyGlobal_Load(ctx, HPyBoundArrayMethod_Type);

    PyBoundArrayMethodObject *data;
    HPy res = HPy_New(ctx, bound_array_method_type, &data);
    HPy_Close(ctx, bound_array_method_type);
    if (HPy_IsNull(res)) {
        return HPy_NULL;
    }
    data->method = HPyField_NULL;
    data->nargs = nargs;
    // TODO: HPY LABS PORT: PyMem_Malloc
    data->dtypes = calloc(nargs, sizeof(HPy));
    if (data->dtypes == NULL) {
        HPy_Close(ctx, res);
        HPyErr_NoMemory(ctx);
        return HPy_NULL;
    }
    for (int i = 0; i < nargs ; i++) {
        if (!HPy_IsNull(spec->dtypes[i])) {
            HPyField_Store(ctx, res, &data->dtypes[i], spec->dtypes[i]);
        } else {
            data->dtypes[i] = HPyField_NULL;
        }
    }
    HPy array_method_type = HPyGlobal_Load(ctx, HPyArrayMethod_Type);
    PyArrayMethodObject *method_data;
    HPy method = HPy_New(ctx, array_method_type, &method_data);
    HPy_Close(ctx, array_method_type);
    if (HPy_IsNull(method)) {
        HPy_Close(ctx, res);
        HPyErr_NoMemory(ctx);
        return HPy_NULL;
    }
//    memset((char *)(res->method) + sizeof(PyObject), 0,
//           sizeof(PyArrayMethodObject) - sizeof(PyObject));

    method_data->nin = spec->nin;
    method_data->nout = spec->nout;
    method_data->flags = spec->flags;
    method_data->casting = spec->casting;
    if (fill_arraymethod_from_slots(ctx, res, data, spec, method,
            method_data, private) < 0) {
        HPy_Close(ctx, res);
        HPy_Close(ctx, method);
        return HPy_NULL;
    }

    Py_ssize_t length = strlen(spec->name);
    // TODO HPY LABS PORT: PyMem_Malloc
    // res->method->name = PyMem_Malloc(length + 1);
    method_data->name = malloc(length + 1);
    if (method_data->name == NULL) {
        HPy_Close(ctx, res);
        HPy_Close(ctx, method);
        HPyErr_NoMemory(ctx);
        return HPy_NULL;
    }
    strcpy(method_data->name, spec->name);
    HPyField_Store(ctx, res, &data->method, method);
    HPy_Close(ctx, method);

    return res;
}


HPyDef_SLOT(arraymethod_dealloc, arraymethod_dealloc_impl, HPy_tp_destroy)
static void
arraymethod_dealloc_impl(void *data)
{
    PyArrayMethodObject *meth = (PyArrayMethodObject *)data;
    // PyMem_Free(meth->name);
    free(meth->name);
    free(meth->wrapped_dtypes);
}

HPyDef_SLOT(arraymethod_traverse, arraymethod_traverse_impl, HPy_tp_traverse)
static int
arraymethod_traverse_impl(void *self, HPyFunc_visitproc visit, void *arg)
{
    PyArrayMethodObject *meth = (PyArrayMethodObject *)self;
    HPy_VISIT(&meth->wrapped_meth);
    if (!HPyField_IsNull(meth->wrapped_meth)) {
        assert(meth->wrapped_dtypes);
        for (int i = 0; i < meth->nin + meth->nout; i++) {
            HPy_VISIT((meth->wrapped_dtypes + i));
        }
    }
    return 0;
}

NPY_NO_EXPORT HPyDef *PyArrayMethod_Type_defines[] = {
        &arraymethod_dealloc,
        &arraymethod_traverse,
        NULL
};

// TODO HPY LABS PORT: for legacy compat; eventually remove !
NPY_NO_EXPORT PyTypeObject *PyArrayMethod_Type;
NPY_NO_EXPORT HPyGlobal HPyArrayMethod_Type;

NPY_NO_EXPORT HPyType_Spec PyArrayMethod_Type_Spec = {
    .name = "numpy._ArrayMethod",
    .basicsize = sizeof(PyArrayMethodObject),
    .flags = HPy_TPFLAGS_DEFAULT,
    .defines = PyArrayMethod_Type_defines,
    .legacy = 1,
};


HPyDef_SLOT(boundarraymethod_repr, boundarraymethod_repr_impl, HPy_tp_repr)
static HPy
boundarraymethod_repr_impl(HPyContext *ctx, HPy self)
{
    PyBoundArrayMethodObject *data = PyBoundArrayMethodObject_AsStruct(ctx, self);
    int nargs = data->nargs;
    HPy dtypes = HPyArray_TupleFromFields(ctx,
            nargs, self, data->dtypes, 0);
    if (HPy_IsNull(dtypes)) {
        return HPy_NULL;
    }
    // TODO HPY LABS PORT: PyUnicode_FromFormat
    // HPy repr = HPyUnicode_FromString(
    //                     "<np._BoundArrayMethod `%s` for dtypes %S>",
    //                     self->method->name, dtypes);
    HPy repr = HPyUnicode_FromString(ctx,
                        "<np._BoundArrayMethod `%s` for dtypes %S>");
    HPy_Close(ctx, dtypes);
    return repr;
}


HPyDef_SLOT(boundarraymethod_traverse, boundarraymethod_traverse_impl, HPy_tp_traverse)
static int
boundarraymethod_traverse_impl(void *object, HPyFunc_visitproc visit, void *arg)
{
    PyBoundArrayMethodObject *meth = (PyBoundArrayMethodObject *) object;
    int nargs = meth->nargs;
    for (int i = 0; i < nargs; i++) {
        HPy_VISIT(&meth->dtypes[i]);
    }
    HPy_VISIT(&meth->method);
    return 0;
}

HPyDef_SLOT(boundarraymethod_destroy, boundarraymethod_destroy_impl, HPy_tp_destroy)
static void
boundarraymethod_destroy_impl(void *object)
{
    PyBoundArrayMethodObject *meth = (PyBoundArrayMethodObject *) object;
    // TODO: HPY LABS PORT: PyMem_Free
    free(meth->dtypes);
}

/*
 * Calls resolve_descriptors() and returns the casting level, the resolved
 * descriptors as a tuple, and a possible view-offset (integer or None).
 * If the operation is impossible returns (-1, None, None).
 * May raise an error, but usually should not.
 * The function validates the casting attribute compared to the returned
 * casting level.
 *
 * TODO: This function is not public API, and certain code paths will need
 *       changes and especially testing if they were to be made public.
 */
//    {"_resolve_descriptors", (PyCFunction)boundarraymethod__resolve_descripors,
//     METH_O, "Resolve the given dtypes."},
HPyDef_METH(boundarraymethod__resolve_descripors, "_resolve_descriptors", boundarraymethod__resolve_descripors_impl, HPyFunc_O, .doc="Resolve the given dtypes.")
static HPy
boundarraymethod__resolve_descripors_impl(HPyContext *ctx,
        HPy self, HPy descr_tuple)
{
    // PyBoundArrayMethodObject *self, PyObject *descr_tuple)
    // TODO HPY LABS PORT
    hpy_abort_not_implemented("boundarraymethod__resolve_descripors");
    return HPy_NULL;
//    int nin = self->method->nin;
//    int nout = self->method->nout;
//
//    PyArray_Descr *given_descrs[NPY_MAXARGS];
//    PyArray_Descr *loop_descrs[NPY_MAXARGS];
//
//    if (!PyTuple_CheckExact(descr_tuple) ||
//            PyTuple_Size(descr_tuple) != nin + nout) {
//        PyErr_Format(PyExc_TypeError,
//                "_resolve_descriptors() takes exactly one tuple with as many "
//                "elements as the method takes arguments (%d+%d).", nin, nout);
//        return NULL;
//    }
//
//    for (int i = 0; i < nin + nout; i++) {
//        PyObject *tmp = PyTuple_GetItem(descr_tuple, i);
//        if (tmp == NULL) {
//            return NULL;
//        }
//        else if (tmp == Py_None) {
//            if (i < nin) {
//                PyErr_SetString(PyExc_TypeError,
//                        "only output dtypes may be omitted (set to None).");
//                return NULL;
//            }
//            given_descrs[i] = NULL;
//        }
//        else if (PyArray_DescrCheck(tmp)) {
//            if (Py_TYPE(tmp) != (PyTypeObject *)self->dtypes[i]) {
//                PyErr_Format(PyExc_TypeError,
//                        "input dtype %S was not an exact instance of the bound "
//                        "DType class %S.", tmp, self->dtypes[i]);
//                return NULL;
//            }
//            given_descrs[i] = (PyArray_Descr *)tmp;
//        }
//        else {
//            PyErr_SetString(PyExc_TypeError,
//                    "dtype tuple can only contain dtype instances or None.");
//            return NULL;
//        }
//    }
//
//    npy_intp view_offset = NPY_MIN_INTP;
////    NPY_CASTING casting = self->method->resolve_descriptors(
////            self->method, self->dtypes, given_descrs, loop_descrs, &view_offset);
//    NPY_CASTING casting = NPY_NO_CASTING;
//
//    if (casting < 0 && PyErr_Occurred()) {
//        return NULL;
//    }
//    else if (casting < 0) {
//        return Py_BuildValue("iO", casting, Py_None, Py_None);
//    }
//
//    PyObject *result_tuple = PyTuple_New(nin + nout);
//    if (result_tuple == NULL) {
//        return NULL;
//    }
//    for (int i = 0; i < nin + nout; i++) {
//        /* transfer ownership to the tuple. */
//        PyTuple_SET_ITEM(result_tuple, i, (PyObject *)loop_descrs[i]);
//    }
//
//    PyObject *view_offset_obj;
//    if (view_offset == NPY_MIN_INTP) {
//        Py_INCREF(Py_None);
//        view_offset_obj = Py_None;
//    }
//    else {
//        view_offset_obj = PyLong_FromSsize_t(view_offset);
//        if (view_offset_obj == NULL) {
//            Py_DECREF(result_tuple);
//            return NULL;
//        }
//    }
//
//    /*
//     * The casting flags should be the most generic casting level.
//     * If no input is parametric, it must match exactly.
//     *
//     * (Note that these checks are only debugging checks.)
//     */
//    int parametric = 0;
//    for (int i = 0; i < nin + nout; i++) {
//        if (NPY_DT_is_parametric(self->dtypes[i])) {
//            parametric = 1;
//            break;
//        }
//    }
//    if (self->method->casting != -1) {
//        NPY_CASTING cast = casting;
//        if (self->method->casting !=
//                PyArray_MinCastSafety(cast, self->method->casting)) {
//            PyErr_Format(PyExc_RuntimeError,
//                    "resolve_descriptors cast level did not match stored one. "
//                    "(set level is %d, got %d for method %s)",
//                    self->method->casting, cast, self->method->name);
//            Py_DECREF(result_tuple);
//            Py_DECREF(view_offset_obj);
//            return NULL;
//        }
//        if (!parametric) {
//            /*
//             * Non-parametric can only mismatch if it switches from equiv to no
//             * (e.g. due to byteorder changes).
//             */
//            if (cast != self->method->casting &&
//                    self->method->casting != NPY_EQUIV_CASTING) {
//                PyErr_Format(PyExc_RuntimeError,
//                        "resolve_descriptors cast level changed even though "
//                        "the cast is non-parametric where the only possible "
//                        "change should be from equivalent to no casting. "
//                        "(set level is %d, got %d for method %s)",
//                        self->method->casting, cast, self->method->name);
//                Py_DECREF(result_tuple);
//                Py_DECREF(view_offset_obj);
//                return NULL;
//            }
//        }
//    }
//
//    return Py_BuildValue("iNN", casting, result_tuple, view_offset_obj);
}


/*
 * TODO: This function is not public API, and certain code paths will need
 *       changes and especially testing if they were to be made public.
 */
HPyDef_METH(boundarraymethod__simple_strided_call, "", boundarraymethod__simple_strided_call_impl, HPyFunc_O, .doc="call on 1-d inputs and pre-allocated outputs (single call).")
static HPy
boundarraymethod__simple_strided_call_impl(HPyContext *ctx,
        HPy h_self, // PyBoundArrayMethodObject *self
        HPy h_arr_tuple)
{
    HPy arrays[NPY_MAXARGS]; // PyArrayObject *
    HPy descrs[NPY_MAXARGS]; // PyArray_Descr *
    HPy out_descrs[NPY_MAXARGS]; // PyArray_Descr *
    Py_ssize_t length = -1;
    int aligned = 1;
    char *args[NPY_MAXARGS];
    npy_intp strides[NPY_MAXARGS];
    PyBoundArrayMethodObject *self_data = PyBoundArrayMethodObject_AsStruct(ctx, h_self);
    HPy h_method = HPyField_Load(ctx, h_self, self_data->method);
    PyArrayMethodObject *method = PyArrayMethodObject_AsStruct(ctx, h_method);
    int nin = method->nin;
    int nout = method->nout;

    if (!HPyTuple_CheckExact(ctx, h_arr_tuple) ||
            HPy_Length(ctx, h_arr_tuple) != nin + nout) {
        // PyErr_Format(PyExc_TypeError,
        //         "_simple_strided_call() takes exactly one tuple with as many "
        //         "arrays as the method takes arguments (%d+%d).", nin, nout);
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "_simple_strided_call() takes exactly one tuple with as many "
                "arrays as the method takes arguments.");
        return HPy_NULL;
    }

    HPy *dtypes_arr = (HPy *)malloc((nin + nout) * sizeof(HPy));
    for (int i = 0; i < nin + nout; i++) {
        HPy tmp = HPy_GetItem_i(ctx, h_arr_tuple, i);
        if (HPy_IsNull(tmp)) {
            free(dtypes_arr);
            return HPy_NULL;
        }
        else if (!HPyArray_CheckExact(ctx, tmp)) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "All inputs must be NumPy arrays.");
            free(dtypes_arr);
            return HPy_NULL;
        }
        arrays[i] = tmp;
        PyArrayObject *arrays_i_data = PyArrayObject_AsStruct(ctx, arrays[i]);
        descrs[i] = HPyArray_DESCR(ctx, arrays[i], arrays_i_data);

        /* Check that the input is compatible with a simple method call. */
        dtypes_arr[i] = HPyField_Load(ctx, h_self, self_data->dtypes[i]);
        PyObject *dtype_i = HPy_AsPyObject(ctx, dtypes_arr[i]);
        HPy descrs_i_type = HPy_Type(ctx, descrs[i]);
        if (HPy_Is(ctx, descrs_i_type, dtypes_arr[i])) {
            HPy_Close(ctx, descrs_i_type);
            // PyErr_Format(PyExc_TypeError,
            //         "input dtype %S was not an exact instance of the bound "
            //         "DType class %S.", descrs[i], self_data->dtypes[i]);
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "input dtype %S was not an exact instance of the bound "
                    "DType class %S.");
            free(dtypes_arr);
            return HPy_NULL;
        }
        HPy_Close(ctx, descrs_i_type);
        if (PyArray_NDIM(arrays_i_data) != 1) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "All arrays must be one dimensional.");
            free(dtypes_arr);
            return HPy_NULL;
        }
        if (i == 0) {
            length = HPyArray_SIZE(arrays_i_data);
        }
        else if (HPyArray_SIZE(arrays_i_data) != length) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "All arrays must have the same length.");
            free(dtypes_arr);
            return HPy_NULL;
        }
        if (i >= nin) {
            if (HPyArray_FailUnlessWriteableWithStruct(ctx,
                    arrays[i], arrays_i_data, "_simple_strided_call() output") < 0) {
                free(dtypes_arr);
                return HPy_NULL;
            }
        }

        args[i] = PyArray_BYTES(arrays_i_data);
        strides[i] = PyArray_STRIDES(arrays_i_data)[0];
        /* TODO: We may need to distinguish aligned and itemsize-aligned */
        aligned &= PyArray_ISALIGNED(arrays_i_data);
    }
    if (!aligned && !(method->flags & NPY_METH_SUPPORTS_UNALIGNED)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "method does not support unaligned input.");
        free(dtypes_arr);
        return HPy_NULL;
    }

    npy_intp view_offset = NPY_MIN_INTP;

    NPY_CASTING casting = method->resolve_descriptors(ctx,
            h_method, dtypes_arr, descrs, out_descrs, &view_offset);

    free(dtypes_arr);
    if (casting < 0) {
        CAPI_WARN("boundarraymethod__simple_strided_call_impl: PyErr_Fetch & npy_PyErr_ChainExceptions");
        PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
        PyErr_Fetch(&err_type, &err_value, &err_traceback);
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "cannot perform method call with the given dtypes.");
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        return HPy_NULL;
    }

    int dtypes_were_adapted = 0;
    for (int i = 0; i < nin + nout; i++) {
        /* NOTE: This check is probably much stricter than necessary... */
        dtypes_were_adapted |= !HPy_Is(ctx, descrs[i], out_descrs[i]);
        HPy_Close(ctx, out_descrs[i]);
    }
    if (dtypes_were_adapted) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "_simple_strided_call(): requires dtypes to not require a cast "
                "(must match exactly with `_resolve_descriptors()`).");
        return HPy_NULL;
    }

    HPy *h_descrs = HPy_FromPyObjectArray(ctx, descrs, NPY_MAXARGS);
    HPyArrayMethod_Context context = {
            .caller = HPy_NULL,
            .method = h_method,
            .descriptors = h_descrs,
    };
    HPyArrayMethod_StridedLoop *strided_loop = NULL;
    NpyAuxData *loop_data = NULL;
    NPY_ARRAYMETHOD_FLAGS flags = 0;

    if (method->get_strided_loop(ctx,
            &context, aligned, 0, strides,
            &strided_loop, &loop_data, &flags) < 0) {
        return HPy_NULL;
    }

    /*
        * TODO: Add floating point error checks if requested and
        *       possibly release GIL if allowed by the flags.
        */
    int res = strided_loop(ctx, &context, args, &length, strides, loop_data);
    if (loop_data != NULL) {
        loop_data->free(loop_data);
    }
    if (res < 0) {
        return HPy_NULL;
    }
    return HPy_Dup(ctx, ctx->h_None);
}


/*
 * Support for masked inner-strided loops.  Masked inner-strided loops are
 * only used in the ufunc machinery.  So this special cases them.
 * In the future it probably makes sense to create an::
 *
 *     Arraymethod->get_masked_strided_loop()
 *
 * Function which this can wrap instead.
 */
typedef struct {
    NpyAuxData base;
    HPyArrayMethod_StridedLoop *unmasked_stridedloop;
    NpyAuxData *unmasked_auxdata;
    int nargs;
    char *dataptrs[];
} _masked_stridedloop_data;


static void
_masked_stridedloop_data_free(NpyAuxData *auxdata)
{
    _masked_stridedloop_data *data = (_masked_stridedloop_data *)auxdata;
    NPY_AUXDATA_FREE(data->unmasked_auxdata);
    PyMem_Free(data);
}


/*
 * This function wraps a regular unmasked strided-loop as a
 * masked strided-loop, only calling the function for elements
 * where the mask is True.
 *
 * TODO: Reductions also use this code to implement masked reductions.
 *       Before consolidating them, reductions had a special case for
 *       broadcasts: when the mask stride was 0 the code does not check all
 *       elements as `npy_memchr` currently does.
 *       It may be worthwhile to add such an optimization again if broadcasted
 *       masks are common enough.
 */
static int
generic_masked_strided_loop(HPyContext *hctx,
        HPyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions,
        const npy_intp *strides, NpyAuxData *_auxdata)
{
    _masked_stridedloop_data *auxdata = (_masked_stridedloop_data *)_auxdata;
    int nargs = auxdata->nargs;
    HPyArrayMethod_StridedLoop *strided_loop = auxdata->unmasked_stridedloop;
    NpyAuxData *strided_loop_auxdata = auxdata->unmasked_auxdata;

    char **dataptrs = auxdata->dataptrs;
    memcpy(dataptrs, data, nargs * sizeof(char *));
    char *mask = data[nargs];
    npy_intp mask_stride = strides[nargs];

    npy_intp N = dimensions[0];
    /* Process the data as runs of unmasked values */
    do {
        Py_ssize_t subloopsize;

        /* Skip masked values */
        mask = npy_memchr(mask, 0, mask_stride, N, &subloopsize, 1);
        for (int i = 0; i < nargs; i++) {
            dataptrs[i] += subloopsize * strides[i];
        }
        N -= subloopsize;

        /* Process unmasked values */
        mask = npy_memchr(mask, 0, mask_stride, N, &subloopsize, 0);
        int res = strided_loop(hctx, context,
                dataptrs, &subloopsize, strides, strided_loop_auxdata);
        if (res != 0) {
            return res;
        }
        for (int i = 0; i < nargs; i++) {
            dataptrs[i] += subloopsize * strides[i];
        }
        N -= subloopsize;
    } while (N > 0);

    return 0;
}


/*
 * Fetches a strided-loop function that supports a boolean mask as additional
 * (last) operand to the strided-loop.  It is otherwise largely identical to
 * the `get_loop` method which it wraps.
 * This is the core implementation for the ufunc `where=...` keyword argument.
 *
 * NOTE: This function does not support `move_references` or inner dimensions.
 */
NPY_NO_EXPORT int
PyArrayMethod_GetMaskedStridedLoop(
        PyArrayMethod_Context *context,
        int aligned, npy_intp *fixed_strides,
        HPyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    return HPyArrayMethod_GetMaskedStridedLoop(npy_get_context(), 
                                            context, aligned, fixed_strides, 
                                            out_loop, out_transferdata, flags);
}

NPY_NO_EXPORT int
HPyArrayMethod_GetMaskedStridedLoop(
        HPyContext *hctx,
        HPyArrayMethod_Context *context,
        int aligned, npy_intp *fixed_strides,
        HPyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    _masked_stridedloop_data *data;
    PyArrayMethodObject *method_data = PyArrayMethodObject_AsStruct(hctx, context->method);
    int nargs = method_data->nin + method_data->nout;

    /* Add working memory for the data pointers, to modify them in-place */
    // TODO HPY LABS PORT: PyMem_Malloc
    data = malloc(sizeof(_masked_stridedloop_data) +
                        sizeof(char *) * nargs);
    if (data == NULL) {
        HPyErr_NoMemory(hctx);
        return -1;
    }
    data->base.free = _masked_stridedloop_data_free;
    data->base.clone = NULL;  /* not currently used */
    data->unmasked_stridedloop = NULL;
    data->nargs = nargs;

    int res = method_data->get_strided_loop(hctx, context,
            aligned, 0, fixed_strides,
            &data->unmasked_stridedloop, &data->unmasked_auxdata, flags);
    if (res < 0) {
        // TODO HPY LABS PORT: PyMem_Free
        // PyMem_Free(data);
        free(data);
        return -1;
    }
    *out_transferdata = (NpyAuxData *)data;
    *out_loop = generic_masked_strided_loop;
    return 0;
}

#define PARAM_DTYPES(p, n) (p)
#define PARAM_IN_DESCRS(p, n) ((p)+(n))
#define PARAM_OUT_DESCRS(p, n) ((p)+(n)*2)

NPY_NO_EXPORT NPY_CASTING
resolve_descriptors_trampoline(
        h_resolve_descriptors_function target,
        struct PyArrayMethodObject_tag *method,
        PyArray_DTypeMeta **dtypes,
        PyArray_Descr **given_descrs,
        PyArray_Descr **loop_descrs,
        npy_intp *view_offset)
{
    HPyContext *hctx = npy_get_context();
    int i, n = method->nin + method->nout;
    HPy *params = (HPy *)alloca(n*3*sizeof(HPy));
    for (i=0; i < n; i++) {
        PARAM_DTYPES(params, n)[i] = HPy_FromPyObject(hctx, (PyObject *)dtypes[i]);
        PARAM_IN_DESCRS(params, n)[i] = HPy_FromPyObject(hctx, (PyObject *)given_descrs[i]);
        PARAM_OUT_DESCRS(params, n)[i] = HPy_NULL;
    }
    HPy h_meth = HPy_FromPyObject(hctx, (PyObject *)method);
    NPY_CASTING res = target(hctx, h_meth, PARAM_DTYPES(params, n),
            PARAM_IN_DESCRS(params, n), PARAM_OUT_DESCRS(params, n),
            view_offset);
    HPy_Close(hctx, h_meth);

    for (i=0; i < n; i++) {
        HPy_Close(hctx, PARAM_DTYPES(params, n)[i]);
        HPy_Close(hctx, PARAM_IN_DESCRS(params, n)[i]);
        loop_descrs[i] = (PyArray_Descr *) HPy_AsPyObject(hctx, PARAM_OUT_DESCRS(params, n)[i]);
        HPy_Close(hctx, PARAM_OUT_DESCRS(params, n)[i]);
    }
    return res;
}
#undef PARAM_DTYPES
#undef PARAM_IN_DESCRS
#undef PARAM_OUT_DESCRS


HPyDef_GET(boundarraymethod__supports_unaligned, "_supports_unaligned",
        boundarraymethod__supports_unaligned_impl,
        .doc = "whether the method supports unaligned inputs/outputs.")
static HPy
boundarraymethod__supports_unaligned_impl(HPyContext *ctx, HPy self, void *NPY_UNUSED(ptr))
{
    PyBoundArrayMethodObject *data = PyBoundArrayMethodObject_AsStruct(ctx, self);
    HPy h_method = HPyField_Load(ctx, self, data->method);
    HPy res = HPyBool_FromLong(ctx,
            PyArrayMethodObject_AsStruct(ctx, h_method)->flags & NPY_METH_SUPPORTS_UNALIGNED);
    HPy_Close(ctx, h_method);
    return res;
}

static HPyDef *boundarraymethod_defines[] = {
        &boundarraymethod_traverse,
        &boundarraymethod_destroy,
        &boundarraymethod__supports_unaligned,
        &boundarraymethod_repr,
        &boundarraymethod__resolve_descripors,
        &boundarraymethod__simple_strided_call,
        NULL
};

//NPY_NO_EXPORT PyTypeObject PyBoundArrayMethod_Type = {
//    PyVarObject_HEAD_INIT(NULL, 0)
//    .tp_name = "numpy._BoundArrayMethod",
//    .tp_basicsize = sizeof(PyBoundArrayMethodObject),
//    .tp_flags = Py_TPFLAGS_DEFAULT,
//    .tp_repr = (reprfunc)boundarraymethod_repr,
//    .tp_dealloc = boundarraymethod_dealloc,
//    .tp_methods = boundarraymethod_methods,
//    .tp_getset = boundarraymethods_getters,
//};
NPY_NO_EXPORT PyTypeObject *PyBoundArrayMethod_Type;
NPY_NO_EXPORT HPyGlobal HPyBoundArrayMethod_Type;

NPY_NO_EXPORT HPyType_Spec PyBoundArrayMethod_Type_Spec = {
    .name = "numpy._BoundArrayMethod",
    .basicsize = sizeof(PyBoundArrayMethodObject),
    .flags = HPy_TPFLAGS_DEFAULT,
    .defines = boundarraymethod_defines,
};

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlpack/dlpack.h>

#include "numpy/arrayobject.h"
#include "common/npy_argparse.h"

#include "common/dlpack/dlpack.h"
#include "common/npy_dlpack.h"

static void
array_dlpack_deleter(DLManagedTensor *self)
{
    PyArrayObject *array = (PyArrayObject *)self->manager_ctx;
    // This will also free the strides as it's one allocation.
    PyMem_Free(self->dl_tensor.shape);
    PyMem_Free(self);
    Py_XDECREF(array);
}

/* This is exactly as mandated by dlpack */
static void dlpack_capsule_deleter(PyObject *self) {
    if (PyCapsule_IsValid(self, NPY_DLPACK_USED_CAPSULE_NAME)) {
        return;
    }

    /* an exception may be in-flight, we must save it in case we create another one */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);

    DLManagedTensor *managed =
        (DLManagedTensor *)PyCapsule_GetPointer(self, NPY_DLPACK_CAPSULE_NAME);
    if (managed == NULL) {
        PyErr_WriteUnraisable(self);
        goto done;
    }
    /*
     *  the spec says the deleter can be NULL if there is no way for the caller
     * to provide a reasonable destructor.
     */
    if (managed->deleter) {
        managed->deleter(managed);
        /* TODO: is the deleter allowed to set a python exception? */
        assert(!PyErr_Occurred());
    }

done:
    PyErr_Restore(type, value, traceback);
}

/* used internally, almost identical to dlpack_capsule_deleter() */
static void array_dlpack_internal_capsule_deleter(HPyContext *ctx, HPy self)
{
    /* an exception may be in-flight, we must save it in case we create another one */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);

    DLManagedTensor *managed =
        (DLManagedTensor *)HPyCapsule_GetPointer(ctx, self, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
    if (managed == NULL) {
        HPyErr_WriteUnraisable(ctx, self);
        goto done;
    }
    /*
     *  the spec says the deleter can be NULL if there is no way for the caller
     * to provide a reasonable destructor.
     */
    if (managed->deleter) {
        managed->deleter(managed);
        /* TODO: is the deleter allowed to set a python exception? */
        assert(!HPyErr_Occurred(ctx));
    }

done:
    PyErr_Restore(type, value, traceback);
}

static DLDevice
hpy_array_get_dl_device(HPyContext *ctx, HPy /* PyArrayObject * */ self) {
    DLDevice ret;
    ret.device_type = kDLCPU;
    ret.device_id = 0;
    PyArrayObject *self_struct = PyArrayObject_AsStruct(ctx, self);
    HPy base = HPyArray_BASE(ctx, self, self_struct);
    // The outer if is due to the fact that NumPy arrays are on the CPU
    // by default (if not created from DLPack).
    if (HPyCapsule_IsValid(ctx, base, NPY_DLPACK_INTERNAL_CAPSULE_NAME)) {
        DLManagedTensor *managed = HPyCapsule_GetPointer(ctx,
                base, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
        HPy_Close(ctx, base);
        if (managed == NULL) {
            return ret;
        }
        return managed->dl_tensor.device;
    }
    HPy_Close(ctx, base);
    return ret;
}

// This function cannot return NULL, but it can fail,
// So call PyErr_Occurred to check if it failed after
// calling it.
static DLDevice
array_get_dl_device(PyArrayObject *self) {
    HPyContext *ctx = npy_get_context();
    HPy h_self = HPy_FromPyObject(ctx, self);
    DLDevice ret = hpy_array_get_dl_device(ctx, h_self);
    HPy_Close(ctx, h_self);
    return ret;
    // DLDevice ret;
    // ret.device_type = kDLCPU;
    // ret.device_id = 0;
    // PyObject *base = PyArray_BASE(self);
    // // The outer if is due to the fact that NumPy arrays are on the CPU
    // // by default (if not created from DLPack).
    // if (PyCapsule_IsValid(base, NPY_DLPACK_INTERNAL_CAPSULE_NAME)) {
    //     DLManagedTensor *managed = PyCapsule_GetPointer(
    //             base, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
    //     if (managed == NULL) {
    //         return ret;
    //     }
    //     return managed->dl_tensor.device;
    // }
    // return ret;
}


PyObject *
array_dlpack(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *stream = Py_None;
    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("__dlpack__", args, len_args, kwnames,
            "$stream", NULL, &stream, NULL, NULL, NULL)) {
        return NULL;
    }

    if (stream != Py_None) {
        PyErr_SetString(PyExc_RuntimeError, "NumPy only supports "
                "stream=None.");
        return NULL;
    }

    if ( !(PyArray_FLAGS(self) & NPY_ARRAY_WRITEABLE)) {
        PyErr_SetString(PyExc_TypeError, "NumPy currently only supports "
                "dlpack for writeable arrays");
        return NULL;
    }

    npy_intp itemsize = PyArray_ITEMSIZE(self);
    int ndim = PyArray_NDIM(self);
    npy_intp *strides = PyArray_STRIDES(self);
    npy_intp *shape = PyArray_SHAPE(self);

    if (!PyArray_IS_C_CONTIGUOUS(self) && PyArray_SIZE(self) != 1) {
        for (int i = 0; i < ndim; ++i) {
            if (strides[i] % itemsize != 0) {
                PyErr_SetString(PyExc_RuntimeError,
                        "DLPack only supports strides which are a multiple of "
                        "itemsize.");
                return NULL;
            }
        }
    }

    DLDataType managed_dtype;
    PyArray_Descr *dtype = PyArray_DESCR(self);

    if (PyDataType_ISBYTESWAPPED(dtype)) {
        PyErr_SetString(PyExc_TypeError, "DLPack only supports native "
                    "byte swapping.");
            return NULL;
    }

    managed_dtype.bits = 8 * itemsize;
    managed_dtype.lanes = 1;

    if (PyDataType_ISSIGNED(dtype)) {
        managed_dtype.code = kDLInt;
    }
    else if (PyDataType_ISUNSIGNED(dtype)) {
        managed_dtype.code = kDLUInt;
    }
    else if (PyDataType_ISFLOAT(dtype)) {
        // We can't be sure that the dtype is
        // IEEE or padded.
        if (itemsize > 8) {
            PyErr_SetString(PyExc_TypeError, "DLPack only supports IEEE "
                    "floating point types without padding.");
            return NULL;
        }
        managed_dtype.code = kDLFloat;
    }
    else if (PyDataType_ISCOMPLEX(dtype)) {
        // We can't be sure that the dtype is
        // IEEE or padded.
        if (itemsize > 16) {
            PyErr_SetString(PyExc_TypeError, "DLPack only supports IEEE "
                    "complex point types without padding.");
            return NULL;
        }
        managed_dtype.code = kDLComplex;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "DLPack only supports signed/unsigned integers, float "
                        "and complex dtypes.");
        return NULL;
    }

    DLDevice device = array_get_dl_device(self);
    if (PyErr_Occurred()) {
        return NULL;
    }

    DLManagedTensor *managed = PyMem_Malloc(sizeof(DLManagedTensor));
    if (managed == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    /*
     * Note: the `dlpack.h` header suggests/standardizes that `data` must be
     * 256-byte aligned.  We ignore this intentionally, because `__dlpack__`
     * standardizes that `byte_offset` must be 0 (for now) to not break pytorch:
     * https://github.com/data-apis/array-api/issues/293#issuecomment-964111413
     *
     * We further assume that exporting fully unaligned data is OK even without
     * `byte_offset` since the standard does not reject it.
     * Presumably, pytorch will support importing `byte_offset != 0` and NumPy
     * can choose to use it starting about 2023.  At that point, it may be
     * that NumPy MUST use `byte_offset` to adhere to the standard (as
     * specified in the header)!
     */
    managed->dl_tensor.data = PyArray_DATA(self);
    managed->dl_tensor.byte_offset = 0;
    managed->dl_tensor.device = device;
    managed->dl_tensor.dtype = managed_dtype;

    int64_t *managed_shape_strides = PyMem_Malloc(sizeof(int64_t) * ndim * 2);
    if (managed_shape_strides == NULL) {
        PyErr_NoMemory();
        PyMem_Free(managed);
        return NULL;
    }

    int64_t *managed_shape = managed_shape_strides;
    int64_t *managed_strides = managed_shape_strides + ndim;
    for (int i = 0; i < ndim; ++i) {
        managed_shape[i] = shape[i];
        // Strides in DLPack are items; in NumPy are bytes.
        managed_strides[i] = strides[i] / itemsize;
    }

    managed->dl_tensor.ndim = ndim;
    managed->dl_tensor.shape = managed_shape;
    managed->dl_tensor.strides = NULL;
    if (PyArray_SIZE(self) != 1 && !PyArray_IS_C_CONTIGUOUS(self)) {
        managed->dl_tensor.strides = managed_strides;
    }
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = self;
    managed->deleter = array_dlpack_deleter;

    PyObject *capsule = PyCapsule_New(managed, NPY_DLPACK_CAPSULE_NAME,
            dlpack_capsule_deleter);
    if (capsule == NULL) {
        PyMem_Free(managed);
        PyMem_Free(managed_shape_strides);
        return NULL;
    }

    // the capsule holds a reference
    Py_INCREF(self);
    return capsule;
}

PyObject *
array_dlpack_device(PyArrayObject *self, PyObject *NPY_UNUSED(args))
{
    DLDevice device = array_get_dl_device(self);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("ii", device.device_type, device.device_id);
}

HPyDef_METH(_from_dlpack, "_from_dlpack", HPyFunc_O)
NPY_NO_EXPORT HPy
_from_dlpack_impl(HPyContext *ctx, HPy NPY_UNUSED(self), HPy obj) {
    HPy obj_type = HPy_Type(ctx, obj);
    HPy __dlpack__ = HPy_GetAttr_s(ctx, obj_type, "__dlpack__");
    HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
    HPy args = HPyTuple_Pack(ctx, 1, obj);
    HPy capsule = HPy_CallTupleDict(ctx, __dlpack__, args, HPy_NULL);
    HPy_Close(ctx, obj_type);
    HPy_Close(ctx, __dlpack__);
    HPy_Close(ctx, args);
    // HPy capsule = PyObject_CallMethod((PyObject *)obj->ob_type,
    //         "__dlpack__", "O", obj);
    if (HPy_IsNull(capsule)) {
        return HPy_NULL;
    }

    DLManagedTensor *managed =
        (DLManagedTensor *)HPyCapsule_GetPointer(ctx, capsule,
        NPY_DLPACK_CAPSULE_NAME);

    if (managed == NULL) {
        HPy_Close(ctx, capsule);
        return HPy_NULL;
    }

    const int ndim = managed->dl_tensor.ndim;
    if (ndim > NPY_MAXDIMS) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "maxdims of DLPack tensor is higher than the supported "
                "maxdims.");
        HPy_Close(ctx, capsule);
        return HPy_NULL;
    }

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    if (device_type != kDLCPU &&
            device_type != kDLCUDAHost &&
            device_type != kDLROCMHost &&
            device_type != kDLCUDAManaged) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "Unsupported device in DLTensor.");
        HPy_Close(ctx, capsule);
        return HPy_NULL;
    }

    if (managed->dl_tensor.dtype.lanes != 1) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "Unsupported lanes in DLTensor dtype.");
        HPy_Close(ctx, capsule);
        return HPy_NULL;
    }

    int typenum = -1;
    const uint8_t bits = managed->dl_tensor.dtype.bits;
    const npy_intp itemsize = bits / 8;
    switch (managed->dl_tensor.dtype.code) {
    case kDLInt:
        switch (bits)
        {
            case 8: typenum = NPY_INT8; break;
            case 16: typenum = NPY_INT16; break;
            case 32: typenum = NPY_INT32; break;
            case 64: typenum = NPY_INT64; break;
        }
        break;
    case kDLUInt:
        switch (bits)
        {
            case 8: typenum = NPY_UINT8; break;
            case 16: typenum = NPY_UINT16; break;
            case 32: typenum = NPY_UINT32; break;
            case 64: typenum = NPY_UINT64; break;
        }
        break;
    case kDLFloat:
        switch (bits)
        {
            case 16: typenum = NPY_FLOAT16; break;
            case 32: typenum = NPY_FLOAT32; break;
            case 64: typenum = NPY_FLOAT64; break;
        }
        break;
    case kDLComplex:
        switch (bits)
        {
            case 64: typenum = NPY_COMPLEX64; break;
            case 128: typenum = NPY_COMPLEX128; break;
        }
        break;
    }

    if (typenum == -1) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "Unsupported dtype in DLTensor.");
        HPy_Close(ctx, capsule);
        return HPy_NULL;
    }

    npy_intp shape[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];

    for (int i = 0; i < ndim; ++i) {
        shape[i] = managed->dl_tensor.shape[i];
        // DLPack has elements as stride units, NumPy has bytes.
        if (managed->dl_tensor.strides != NULL) {
            strides[i] = managed->dl_tensor.strides[i] * itemsize;
        }
    }

    char *data = (char *)managed->dl_tensor.data +
            managed->dl_tensor.byte_offset;

    HPy descr = HPyArray_DescrFromType(ctx, typenum); // PyArray_Descr *
    if (HPy_IsNull(descr)) {
        HPy_Close(ctx, capsule);
        return HPy_NULL;
    }

    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    HPy ret = HPyArray_NewFromDescr(ctx, array_type, descr, ndim, shape,
            managed->dl_tensor.strides != NULL ? strides : NULL, data, 0, HPy_NULL);
    HPy_Close(ctx, array_type);
    if (HPy_IsNull(ret)) {
        HPy_Close(ctx, capsule);
        return HPy_NULL;
    }

    HPy new_capsule = HPyCapsule_New(ctx, managed,
            NPY_DLPACK_INTERNAL_CAPSULE_NAME,
            array_dlpack_internal_capsule_deleter);
    if (HPy_IsNull(new_capsule)) {
        HPy_Close(ctx, capsule);
        HPy_Close(ctx, ret);
        return HPy_NULL;
    }

    PyArrayObject *ret_struct = PyArrayObject_AsStruct(ctx, ret);
    if (HPyArray_SetBaseObject(ctx, ret, ret_struct, new_capsule) < 0) {
        HPy_Close(ctx, capsule);
        HPy_Close(ctx, ret);
        return HPy_NULL;
    }

    if (HPyCapsule_SetName(ctx, capsule, NPY_DLPACK_USED_CAPSULE_NAME) < 0) {
        HPy_Close(ctx, capsule);
        HPy_Close(ctx, ret);
        return HPy_NULL;
    }

    HPy_Close(ctx, capsule);
    return ret;
}



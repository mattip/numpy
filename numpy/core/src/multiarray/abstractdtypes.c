#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "abstractdtypes.h"
#include "array_coercion.h"
#include "common.h"
#include "arraytypes.h"


//static NPY_INLINE PyArray_Descr *
//int_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
static NPY_INLINE HPy
int_default_descriptor(HPyContext *ctx, HPy NPY_UNUSED(cls))
{
    return HPyArray_DescrFromType(ctx, NPY_LONG);
}

static HPy /* PyArray_Descr * */
hpy_discover_descriptor_from_pyint(HPyContext *ctx,
        HPy /* (PyArray_DTypeMeta *) */ NPY_UNUSED(cls), HPy obj)
{
    assert(HPyLong_Check(ctx, obj));
    /*
     * We check whether long is good enough. If not, check longlong and
     * unsigned long before falling back to `object`.
     */
    long long value = HPyLong_AsLongLong(ctx, obj);
    if (hpy_error_converting(ctx, value)) {
        HPyErr_Clear(ctx);
    }
    else {
        if (NPY_MIN_LONG <= value && value <= NPY_MAX_LONG) {
            return HPyArray_DescrFromType(ctx, NPY_LONG);
        }
        return HPyArray_DescrFromType(ctx, NPY_LONGLONG);
    }

    unsigned long long uvalue = HPyLong_AsUnsignedLongLong(ctx, obj);
    if (uvalue == (unsigned long long)-1 && PyErr_Occurred()){
        HPyErr_Clear(ctx);
    }
    else {
        return HPyArray_DescrFromType(ctx, NPY_ULONGLONG);
    }

    return HPyArray_DescrFromType(ctx, NPY_OBJECT);
}


static NPY_INLINE HPy
float_default_descriptor(HPyContext *ctx, HPy NPY_UNUSED(cls))
{
    return HPyArray_DescrFromType(ctx, NPY_DOUBLE);
}


static PyArray_Descr*
discover_descriptor_from_pyfloat(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyFloat_CheckExact(obj));
    return PyArray_DescrFromType(NPY_DOUBLE);
}

static NPY_INLINE HPy
complex_default_descriptor(HPyContext *ctx, HPy NPY_UNUSED(cls))
{
    return HPyArray_DescrFromType(ctx, NPY_CDOUBLE);
}

static PyArray_Descr*
discover_descriptor_from_pycomplex(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyComplex_CheckExact(obj));
    return PyArray_DescrFromType(NPY_COMPLEX128);
}

/*
 * TODO: These abstract DTypes also carry the dual role of representing
 *       `Floating`, `Complex`, and `Integer` (both signed and unsigned).
 *       They will have to be renamed and exposed in that capacity.
 */
NPY_NO_EXPORT HPyType_Spec HPyArray_PyIntAbstractDType_spec = {
    .name = "numpy._IntegerAbstractDType",
    .basicsize = sizeof(PyArray_Descr),
    .flags = Py_TPFLAGS_DEFAULT,
};

NPY_NO_EXPORT HPyType_Spec HPyArray_PyFloatAbstractDType_spec = {
    .name = "numpy._FloatAbstractDType",
    .basicsize = sizeof(PyArray_Descr),
    .flags = HPy_TPFLAGS_DEFAULT,
};

NPY_NO_EXPORT HPyType_Spec HPyArray_PyComplexAbstractDType_spec = {
    .name = "numpy._ComplexAbstractDType",
    .basicsize = sizeof(PyArray_Descr),
    .flags = HPy_TPFLAGS_DEFAULT,
};

// "forward" declarations:
NPY_DType_Slots pyintabstractdtype_slots;
NPY_DType_Slots pyfloatabstractdtype_slots;
NPY_DType_Slots pycomplexabstractdtype_slots;

NPY_NO_EXPORT int
initialize_and_map_pytypes_to_dtypes(HPyContext *ctx)
{
    int result = -1;
    HPy h_PyArray_PyIntAbstractDType = HPy_NULL;
    HPy h_PyArray_PyFloatAbstractDType = HPy_NULL;
    HPy h_PyArray_PyComplexAbstractDType = HPy_NULL;
    HPy dtype = HPy_NULL;

    HPy h_PyArrayDescr_Type = HPyGlobal_Load(ctx, HPyArrayDescr_Type);
    HPy h_PyArrayDTypeMeta_Type = HPyGlobal_Load(ctx, HPyArrayDTypeMeta_Type);
    HPyType_SpecParam abstract_dtype_params[] = {
        {HPyType_SpecParam_Base, h_PyArrayDescr_Type},
        { HPyType_SpecParam_Metaclass, h_PyArrayDTypeMeta_Type },
        { 0 },
    };

    h_PyArray_PyIntAbstractDType = HPyType_FromSpec(ctx, &HPyArray_PyIntAbstractDType_spec, abstract_dtype_params);
    if (HPy_IsNull(h_PyArray_PyIntAbstractDType)) {
        goto cleanup;
    }
    PyArray_DTypeMeta *int_abstract_dtype_data = PyArray_DTypeMeta_AsStruct(ctx, h_PyArray_PyIntAbstractDType);
    int_abstract_dtype_data->dt_slots = &pyintabstractdtype_slots;
    int_abstract_dtype_data->flags = NPY_DT_ABSTRACT;
    HPyField_Store(ctx, h_PyArray_PyIntAbstractDType, &int_abstract_dtype_data->scalar_type, ctx->h_LongType);
    HPyGlobal_Store(ctx, &HPyArray_PyIntAbstractDType, h_PyArray_PyIntAbstractDType);

    h_PyArray_PyFloatAbstractDType = HPyType_FromSpec(ctx, &HPyArray_PyFloatAbstractDType_spec, abstract_dtype_params);
    if (HPy_IsNull(h_PyArray_PyFloatAbstractDType)) {
        goto cleanup;
    }
    PyArray_DTypeMeta *float_abstract_dtype_data = PyArray_DTypeMeta_AsStruct(ctx, h_PyArray_PyFloatAbstractDType);
    float_abstract_dtype_data->dt_slots = &pyfloatabstractdtype_slots;
    float_abstract_dtype_data->flags = NPY_DT_ABSTRACT;
    HPyField_Store(ctx, h_PyArray_PyFloatAbstractDType, &float_abstract_dtype_data->scalar_type, ctx->h_FloatType);
    HPyGlobal_Store(ctx, &HPyArray_PyFloatAbstractDType, h_PyArray_PyFloatAbstractDType);

    h_PyArray_PyComplexAbstractDType = HPyType_FromSpec(ctx, &HPyArray_PyComplexAbstractDType_spec, abstract_dtype_params);
    if (HPy_IsNull(h_PyArray_PyComplexAbstractDType)) {
        goto cleanup;
    }
    PyArray_DTypeMeta *complex_abstract_dtype_data = PyArray_DTypeMeta_AsStruct(ctx, h_PyArray_PyComplexAbstractDType);
    complex_abstract_dtype_data->dt_slots = &pycomplexabstractdtype_slots;
    complex_abstract_dtype_data->flags = NPY_DT_ABSTRACT;
    HPyField_Store(ctx, h_PyArray_PyComplexAbstractDType, &complex_abstract_dtype_data->scalar_type, ctx->h_ComplexType);
    HPyGlobal_Store(ctx, &HPyArray_PyComplexAbstractDType, h_PyArray_PyComplexAbstractDType);

    /* Register the new DTypes for discovery */
    if (_PyArray_MapPyTypeToDType(
            ctx, h_PyArray_PyIntAbstractDType, ctx->h_LongType, NPY_FALSE) < 0) {
        goto cleanup;
    }
    if (_PyArray_MapPyTypeToDType(
            ctx, h_PyArray_PyFloatAbstractDType, ctx->h_FloatType, NPY_FALSE) < 0) {
        goto cleanup;
    }
    if (_PyArray_MapPyTypeToDType(
            ctx, h_PyArray_PyComplexAbstractDType, ctx->h_ComplexType, NPY_FALSE) < 0) {
        goto cleanup;
    }

    /*
     * Map str, bytes, and bool, for which we do not need abstract versions
     * to the NumPy DTypes. This is done here using the `is_known_scalar_type`
     * function.
     * TODO: The `is_known_scalar_type` function is considered preliminary,
     *       the same could be achieved e.g. with additional abstract DTypes.
     */
    HPy tmp_descr;
    #define SET_DTYPE_FROM_NUM(num)                           \
        HPy_Close(ctx, dtype);                                \
        tmp_descr = HPyArray_DescrFromType(ctx, NPY_UNICODE); \
        dtype = HNPY_DTYPE(ctx, tmp_descr);                   \
        HPy_Close(ctx, tmp_descr);

    SET_DTYPE_FROM_NUM(NPY_UNICODE);
    if (_PyArray_MapPyTypeToDType(ctx, dtype, ctx->h_UnicodeType, NPY_FALSE) < 0) {
        goto cleanup;
    }

    SET_DTYPE_FROM_NUM(NPY_STRING);
    if (_PyArray_MapPyTypeToDType(ctx, dtype, ctx->h_BytesType, NPY_FALSE) < 0) {
        goto cleanup;
    }

    SET_DTYPE_FROM_NUM(NPY_BOOL);
    if (_PyArray_MapPyTypeToDType(ctx, dtype, ctx->h_BoolType, NPY_FALSE) < 0) {
        goto cleanup;
    }

    result = 0;
cleanup:
    HPy_Close(ctx, h_PyArray_PyIntAbstractDType);
    HPy_Close(ctx, h_PyArray_PyFloatAbstractDType);
    HPy_Close(ctx, h_PyArray_PyComplexAbstractDType);
    HPy_Close(ctx, dtype);
    HPy_Close(ctx, h_PyArrayDescr_Type);
    HPy_Close(ctx, h_PyArrayDTypeMeta_Type);
    return result;
}


/*
 * The following functions define the "common DType" for the abstract dtypes.
 *
 * Note that the logic with respect to the "higher" dtypes such as floats
 * could likely be more logically defined for them, but since NumPy dtypes
 * largely "know" each other, that is not necessary.
 */
static PyArray_DTypeMeta *
int_common_dtype(PyArray_DTypeMeta *NPY_UNUSED(cls), PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES) {
        if (other->type_num == NPY_BOOL) {
            /* Use the default integer for bools: */
            return PyArray_DTypeFromTypeNum(NPY_LONG);
        }
        else if (PyTypeNum_ISNUMBER(other->type_num) ||
                 other->type_num == NPY_TIMEDELTA) {
            /* All other numeric types (ant timedelta) are preserved: */
            Py_INCREF(other);
            return other;
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        return PyArray_DTypeFromTypeNum(NPY_UINT8);
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
float_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES) {
        if (other->type_num == NPY_BOOL || PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return PyArray_DTypeFromTypeNum(NPY_DOUBLE);
        }
        else if (PyTypeNum_ISNUMBER(other->type_num)) {
            /* All other numeric types (float+complex) are preserved: */
            Py_INCREF(other);
            return other;
        }
    }
    else if (other == PyArray_PyIntAbstractDType) {
        Py_INCREF(cls);
        return cls;
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        return PyArray_DTypeFromTypeNum(NPY_HALF);
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
complex_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES) {
        if (other->type_num == NPY_BOOL ||
                PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return PyArray_DTypeFromTypeNum(NPY_CDOUBLE);
        }
        else if (PyTypeNum_ISFLOAT(other->type_num)) {
            /*
             * For floats we choose the equivalent precision complex, although
             * there is no CHALF, so half also goes to CFLOAT.
             */
            if (other->type_num == NPY_HALF || other->type_num == NPY_FLOAT) {
                return PyArray_DTypeFromTypeNum(NPY_CFLOAT);
            }
            if (other->type_num == NPY_DOUBLE) {
                return PyArray_DTypeFromTypeNum(NPY_CDOUBLE);
            }
            assert(other->type_num == NPY_LONGDOUBLE);
            return PyArray_DTypeFromTypeNum(NPY_CLONGDOUBLE);
        }
        else if (PyTypeNum_ISCOMPLEX(other->type_num)) {
            /* All other numeric types are preserved: */
            Py_INCREF(other);
            return other;
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        return PyArray_DTypeFromTypeNum(NPY_CFLOAT);
    }
    else if (other == PyArray_PyIntAbstractDType ||
             other == PyArray_PyFloatAbstractDType) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}

/*
 * TODO: These abstract DTypes also carry the dual role of representing
 *       `Floating`, `Complex`, and `Integer` (both signed and unsigned).
 *       They will have to be renamed and exposed in that capacity.
 */
// HPY TODO: eventually get rid of those PyObject* compatible global variables
// The classes are now initialized as heap types, we had to change them to pointers
NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_PyIntAbstractDType;
NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_PyFloatAbstractDType;
NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_PyComplexAbstractDType;
NPY_NO_EXPORT HPyGlobal HPyArray_PyIntAbstractDType;
NPY_NO_EXPORT HPyGlobal HPyArray_PyFloatAbstractDType;
NPY_NO_EXPORT HPyGlobal HPyArray_PyComplexAbstractDType;

NPY_DType_Slots pyintabstractdtype_slots = {
    .default_descr = int_default_descriptor,
    .discover_descr_from_pyobject = discover_descr_from_pyobject_function_trampoline,
    .hdiscover_descr_from_pyobject = hpy_discover_descriptor_from_pyint,
    .common_dtype = int_common_dtype,
};

NPY_DType_Slots pyfloatabstractdtype_slots = {
    .default_descr = float_default_descriptor,
    .discover_descr_from_pyobject = discover_descriptor_from_pyfloat,
    .hdiscover_descr_from_pyobject = hdiscover_descr_from_pyobject_function_trampoline,
    .common_dtype = float_common_dtype,
};

NPY_DType_Slots pycomplexabstractdtype_slots = {
    .default_descr = complex_default_descriptor,
    .discover_descr_from_pyobject = discover_descriptor_from_pycomplex,
    .hdiscover_descr_from_pyobject = hdiscover_descr_from_pyobject_function_trampoline,
    .common_dtype = complex_common_dtype,
};

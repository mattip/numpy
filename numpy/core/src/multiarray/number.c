#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "common.h"
#include "number.h"
#include "temp_elide.h"

#include "binop_override.h"
#include "ufunc_override.h"
#include "abstractdtypes.h"
#include "common_dtype.h"
#include "convert_datatype.h"

// Added for HPy:
#include "arraytypes.h"

/*************************************************************************
 ****************   Implement Number Protocol ****************************
 *************************************************************************/

NPY_NO_EXPORT NumericOps n_ops; /* NB: static objects initialized to zero */
NPY_NO_EXPORT HPyNumericOps hpy_n_ops; /* NB: static objects initialized to zero */

/*
 * Dictionary can contain any of the numeric operations, by name.
 * Those not present will not be changed
 */

/* FIXME - macro contains a return */
// HPY note: we make the assumption that the items are set only once,
// it seems that this could be called multiple times via API PyArray_SetNumericOps, 
// but that API is deprecated in numpy and removed here in HPy numpy
#define SET(op)   temp = HPy_GetItem_s(ctx, dict, #op); \
    if (HPy_IsNull(temp) && HPyErr_Occurred(ctx)) { \
        return -1; \
    } \
    else if (!HPy_IsNull(temp)) { \
        if (!(HPyCallable_Check(ctx, temp))) { \
            return -1; \
        } \
        HPyGlobal_Store(ctx, &hpy_n_ops.op, temp); \
    }

NPY_NO_EXPORT int
_PyArray_SetNumericOps(HPyContext *ctx, HPy dict)
{
    HPy temp = HPy_NULL;
    SET(add);
    SET(subtract);
    SET(multiply);
    SET(divide);
    SET(remainder);
    SET(divmod);
    SET(power);
    SET(square);
    SET(reciprocal);
    SET(_ones_like);
    SET(sqrt);
    SET(cbrt);
    SET(negative);
    SET(positive);
    SET(absolute);
    SET(invert);
    SET(left_shift);
    SET(right_shift);
    SET(bitwise_and);
    SET(bitwise_or);
    SET(bitwise_xor);
    SET(less);
    SET(less_equal);
    SET(equal);
    SET(not_equal);
    SET(greater);
    SET(greater_equal);
    SET(floor_divide);
    SET(true_divide);
    SET(logical_or);
    SET(logical_and);
    SET(floor);
    SET(ceil);
    SET(maximum);
    SET(minimum);
    SET(rint);
    SET(conjugate);
    SET(matmul);
    SET(clip);
    return 0;
}

/*HPY_NUMPY_API
 *Set internal structure with number functions that all arrays will use
 */
NPY_NO_EXPORT int
HPyArray_SetNumericOps(HPyContext *ctx, HPy dict)
{
    /* 2018-09-09, 1.16 */
    if (HPY_DEPRECATE(ctx, "PyArray_SetNumericOps is deprecated. Use "
        "PyUFunc_ReplaceLoopBySignature to replace ufunc inner loop functions "
        "instead.") < 0) {
        return -1;
    }
    return _PyArray_SetNumericOps(ctx, dict);
}

/*NUMPY_API
 *Set internal structure with number functions that all arrays will use
 */
NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict)
{
    HPyContext *ctx = npy_get_context();
    HPy h_dict = HPy_FromPyObject(ctx, dict);
    int ret = HPyArray_SetNumericOps(ctx, h_dict);
    HPy_Close(ctx, h_dict);
    return ret;
}

/* Note - macro contains goto */
#define GET(op) if (!HPy_IsNull(HN_OPS_GET(ctx, op)) &&                                         \
                    (HPy_SetItem_s(ctx, dict, #op, HN_OPS_GET(ctx, op))==-1))    \
        goto fail;

NPY_NO_EXPORT HPy
_PyArray_GetNumericOps(HPyContext *ctx)
{
    HPy dict;
    if (HPy_IsNull(dict = HPyDict_New(ctx)))
        return HPy_NULL;
    GET(add);
    GET(subtract);
    GET(multiply);
    GET(divide);
    GET(remainder);
    GET(divmod);
    GET(power);
    GET(square);
    GET(reciprocal);
    GET(_ones_like);
    GET(sqrt);
    GET(negative);
    GET(positive);
    GET(absolute);
    GET(invert);
    GET(left_shift);
    GET(right_shift);
    GET(bitwise_and);
    GET(bitwise_or);
    GET(bitwise_xor);
    GET(less);
    GET(less_equal);
    GET(equal);
    GET(not_equal);
    GET(greater);
    GET(greater_equal);
    GET(floor_divide);
    GET(true_divide);
    GET(logical_or);
    GET(logical_and);
    GET(floor);
    GET(ceil);
    GET(maximum);
    GET(minimum);
    GET(rint);
    GET(conjugate);
    GET(matmul);
    GET(clip);
    return dict;

 fail:
    HPy_Close(ctx, dict);
    return HPy_NULL;
}

/*HPY_NUMPY_API
  Get dictionary showing number functions that all arrays will use
*/
NPY_NO_EXPORT HPy
HPyArray_GetNumericOps(HPyContext *ctx)
{
    /* 2018-09-09, 1.16 */
    if (HPY_DEPRECATE(ctx, "PyArray_GetNumericOps is deprecated.") < 0) {
        return HPy_NULL;
    }
    return _PyArray_GetNumericOps(ctx);
}


/*NUMPY_API
  Get dictionary showing number functions that all arrays will use
*/
NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void)
{
    HPyContext *ctx = npy_get_context();
    HPy dict = HPyArray_GetNumericOps(ctx);
    PyObject *ret = HPy_AsPyObject(ctx, dict);
    HPy_Close(ctx, dict);
    return ret;
}

static PyObject *
_get_keywords(int rtype, PyArrayObject *out)
{
    PyObject *kwds = NULL;
    if (rtype != NPY_NOTYPE || out != NULL) {
        kwds = PyDict_New();
        if (rtype != NPY_NOTYPE) {
            PyArray_Descr *descr;
            descr = PyArray_DescrFromType(rtype);
            if (descr) {
                PyDict_SetItemString(kwds, "dtype", (PyObject *)descr);
                Py_DECREF(descr);
            }
        }
        if (out != NULL) {
            PyDict_SetItemString(kwds, "out", (PyObject *)out);
        }
    }
    return kwds;
}

static HPy
_hpy_get_keywords(HPyContext *ctx, int rtype, HPy /* PyArrayObject * */ out)
{
    HPy kwds = HPy_NULL;
    if (rtype != NPY_NOTYPE || !HPy_IsNull(out)) {
        kwds = HPyDict_New(ctx);
        if (rtype != NPY_NOTYPE) {
            HPy descr; // PyArray_Descr *
            descr = HPyArray_DescrFromType(ctx, rtype);
            if (!HPy_IsNull(descr)) {
                HPy_SetItem_s(ctx, kwds, "dtype", descr);
                HPy_Close(ctx, descr);
            }
        }
        if (!HPy_IsNull(out)) {
            HPy_SetItem_s(ctx, kwds, "out", out);
        }
    }
    return kwds;
}

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;

    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "reduce");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}

NPY_NO_EXPORT HPy
HPyArray_GenericReduceFunction(HPyContext *ctx, 
                                HPy /* PyArrayObject * */ m1, 
                                HPy op, int axis, int rtype, 
                                HPy /* PyArrayObject * */ out)
{
    HPy args, ret = HPy_NULL, meth;
    HPy kwds;

    HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
    args = HPy_BuildValue(ctx, "(Oi)", m1, axis);
    kwds = _hpy_get_keywords(ctx, rtype, out);
    meth = HPy_GetAttr_s(ctx, op, "reduce");
    if (!HPy_IsNull(meth) && HPyCallable_Check(ctx, meth)) {
        ret = HPy_CallTupleDict(ctx, meth, args, kwds);
    }
    HPy_Close(ctx, args);
    HPy_Close(ctx, meth);
    HPy_Close(ctx, kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;

    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "accumulate");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyObject *m1, PyObject *m2, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, m2, NULL);
}

static inline HPy
__HPyArray_GenericHelper(HPyContext *ctx, HPyGlobal op, size_t n_args, HPy args[]) {

    HPy callable = HPyGlobal_Load(ctx, op);
    HPy res = HPy_Call(ctx, callable, args, n_args, HPy_NULL);
    HPy_Close(ctx, callable);
    return res;
}

#define _HPyArray_GenericHelper(ctx, op, n, ...) \
    (__HPyArray_GenericHelper(ctx, op, n, (HPy[]){ __VA_ARGS__ }))

NPY_NO_EXPORT HPy
HPyArray_GenericBinaryFunction(HPyContext *ctx, HPy m1, HPy m2, HPyGlobal op)
{
    return _HPyArray_GenericHelper(ctx, op, 2, m1, m2);
}

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, NULL);
}

NPY_NO_EXPORT HPy
HPyArray_GenericUnaryFunction(HPyContext *ctx, HPy m1, HPyGlobal op)
{
    return _HPyArray_GenericHelper(ctx, op, 1, m1);
}


static PyObject *
PyArray_GenericInplaceBinaryFunction(PyArrayObject *m1,
                                     PyObject *m2, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, m2, m1, NULL);
}

static HPy
HPyArray_GenericInplaceBinaryFunction(HPyContext *ctx, HPy m1,
                                     HPy m2, HPyGlobal op)
{
    return _HPyArray_GenericHelper(ctx, op, 3, m1, m2, m1);
}

static PyObject *
PyArray_GenericInplaceUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, m1, NULL);
}

static HPy
HPyArray_GenericInplaceUnaryFunction(HPyContext *ctx, HPy m1, HPyGlobal op)
{
    return _HPyArray_GenericHelper(ctx, op, 2, m1, m1);
}



HPyDef_SLOT(array_add, HPy_nb_add)
NPY_NO_EXPORT HPy
array_add_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_add);
    // We cannot support this hack on HPy
    // if (try_binary_elide(m1, m2, &array_inplace_add, &res, 1)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, hpy_n_ops.add);
}

HPyDef_SLOT(array_subtract, HPy_nb_subtract);
NPY_NO_EXPORT HPy
array_subtract_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_subtract);
    // We cannot support this hack on HPy:
    // if (try_binary_elide(m1, m2, &array_inplace_subtract, &res, 0)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, hpy_n_ops.subtract);
}

HPyDef_SLOT(array_multiply_slot, HPy_nb_multiply);
static HPy
array_multiply_slot_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    PyObject *res;

    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx,m1, m2, &array_multiply_slot);
    // if (try_binary_elide(m1, m2, &array_inplace_multiply, &res, 1)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, HPY_N_OPS(multiply));
}

HPyDef_SLOT(array_remainder_slot, HPy_nb_remainder);
static HPy
array_remainder_slot_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx,m1, m2, &array_remainder_slot);
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, HPY_N_OPS(remainder));
}

HPyDef_SLOT(array_divmod_slot, HPy_nb_divmod);
static HPy
array_divmod_slot_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx,m1, m2, &array_divmod_slot);
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, HPY_N_OPS(divmod));
}

/* Need this to be version dependent on account of the slot check */
HPyDef_SLOT(array_matrix_multiply_slot, HPy_nb_matrix_multiply);
static HPy
array_matrix_multiply_slot_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx,m1, m2, &array_matrix_multiply_slot);
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, HPY_N_OPS(matmul));
}

HPyDef_SLOT(array_inplace_matrix_multiply_slot, HPy_nb_inplace_matrix_multiply);
static HPy
array_inplace_matrix_multiply_slot_impl(HPyContext *ctx,
        HPy /* PyArrayObject * */ NPY_UNUSED(m1), HPy NPY_UNUSED(m2))
{
    HPyErr_SetString(ctx, ctx->h_TypeError,
                    "In-place matrix multiplication is not (yet) supported. "
                    "Use 'a = a @ b' instead of 'a @= b'.");
    return HPy_NULL;
}

/*
 * Determine if object is a scalar and if so, convert the object
 * to a double and place it in the out_exponent argument
 * and return the "scalar kind" as a result.   If the object is
 * not a scalar (or if there are other error conditions)
 * return NPY_NOSCALAR, and out_exponent is undefined.
 */
static NPY_SCALARKIND
is_scalar_with_conversion(HPyContext *ctx, HPy h_o2, double* out_exponent)
{
    PyObject *temp;
    const int optimize_fpexps = 1;

    if (HPy_TypeCheck(ctx, h_o2, ctx->h_LongType)) {
        long tmp = HPyLong_AsLong(ctx, h_o2);
        if (error_converting(tmp)) {
            HPyErr_Clear(ctx);
            return NPY_NOSCALAR;
        }
        *out_exponent = (double)tmp;
        return NPY_INTPOS_SCALAR;
    }

    if (optimize_fpexps && HPy_TypeCheck(ctx, h_o2, ctx->h_FloatType)) {
        *out_exponent = HPyFloat_AsDouble(ctx, h_o2);
        return NPY_FLOAT_SCALAR;
    }

    CAPI_WARN("is_scalar_with_conversion: array checks");
    PyArrayObject *o2 = PyArrayObject_AsStruct(ctx, h_o2);
    if (HPyArray_Check(ctx, h_o2)) {
        if ((PyArray_NDIM(o2) == 0) &&
                ((HPyArray_ISINTEGER(ctx, h_o2) ||
                 (optimize_fpexps && HPyArray_ISFLOAT(ctx, h_o2))))) {
            // TODO: add HPy_AsFloat?
            hpy_abort_not_implemented("Py_TYPE(o2)->tp_as_number->nb_float(o2)");
            // temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
            // if (temp == NULL) {
            //     return NPY_NOSCALAR;
            // }
            // *out_exponent = PyFloat_AsDouble(o2);
            // Py_DECREF(temp);
            // if (PyArray_ISINTEGER((PyArrayObject *)o2)) {
            //     return NPY_INTPOS_SCALAR;
            // }
            // else { /* ISFLOAT */
            //     return NPY_FLOAT_SCALAR;
            // }
        }
    }
    // CURRENT PROBLEM: PyArray_IsScalar checks PyIntegerArrType_Type
    // we do not seem to initialize that type??? And definitely we do not
    // expose HPyGlobals for it
    // NOTE: do partial porting approach, this looked small, but it isn't!
    else if (PyArray_IsScalar(o2, Integer) ||
                (optimize_fpexps && PyArray_IsScalar(o2, Floating))) {
        temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
        if (temp == NULL) {
            return NPY_NOSCALAR;
        }
        *out_exponent = PyFloat_AsDouble(o2);
        Py_DECREF(temp);

        if (PyArray_IsScalar(o2, Integer)) {
                return NPY_INTPOS_SCALAR;
        }
        else { /* IsScalar(o2, Floating) */
            return NPY_FLOAT_SCALAR;
        }
    }
    else if (PyIndex_Check((PyObject*) o2)) {
        PyObject* value = PyNumber_Index(o2);
        Py_ssize_t val;
        if (value == NULL) {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            return NPY_NOSCALAR;
        }
        val = PyLong_AsSsize_t(value);
        Py_DECREF(value);
        if (error_converting(val)) {
            PyErr_Clear();
            return NPY_NOSCALAR;
        }
        *out_exponent = (double) val;
        return NPY_INTPOS_SCALAR;
    }
    return NPY_NOSCALAR;
}

/*
 * optimize float array or complex array to a scalar power
 * returns 0 on success, -1 if no optimization is possible
 * the result is in value (can be NULL if an error occurred)
 */
static int
fast_scalar_power(HPyContext *ctx, HPy h_o1, HPy h_o2, int inplace,
                  HPy *value)
{
    double exponent;
    NPY_SCALARKIND kind;   /* NPY_NOSCALAR is not scalar */

    if (!HPyArray_Check(ctx, h_o1)) {
        /* no fast operation found */
        return -1;
    }
    PyArrayObject *o1 = PyArrayObject_AsStruct(ctx, h_o1);
    HPy h_o1_descr = HPyArray_DESCR(ctx, h_o1, o1);
    PyArray_Descr *o1_descr = PyArray_Descr_AsStruct(ctx, h_o1_descr);
    if (HPyArrayDescr_ISOBJECT(o1_descr)) {
        /* no fast operation found */
        return -1;
    }
    
    if ((kind=is_scalar_with_conversion(ctx, h_o2, &exponent))>0) {
        PyArrayObject *a1 = (PyArrayObject *)o1;
        HPyGlobal fastop;
        if (HPyArrayDescr_ISFLOAT(o1_descr) || HPyArrayDescr_ISCOMPLEX(o1_descr)) {
            if (exponent == 1.0) {
                fastop = hpy_n_ops.positive;
            }
            else if (exponent == -1.0) {
                fastop = hpy_n_ops.reciprocal;
            }
            else if (exponent ==  0.0) {
                fastop = hpy_n_ops._ones_like;
            }
            else if (exponent ==  0.5) {
                fastop = hpy_n_ops.sqrt;
            }
            else if (exponent ==  2.0) {
                fastop = hpy_n_ops.square;
            }
            else {
                return -1;
            }

            if (inplace || can_elide_temp_unary(a1)) {
                *value = HPyArray_GenericInplaceUnaryFunction(ctx, h_o1, fastop);
            }
            else {
                *value = HPyArray_GenericUnaryFunction(ctx, h_o1, fastop);
            }
            return 0;
        }
        /* Because this is called with all arrays, we need to
         *  change the output if the kind of the scalar is different
         *  than that of the input and inplace is not on ---
         *  (thus, the input should be up-cast)
         */
        else if (exponent == 2.0) {
            fastop = hpy_n_ops.square;
            if (inplace) {
                *value = HPyArray_GenericInplaceUnaryFunction(ctx, h_o1, fastop);
            }
            else {
                /* We only special-case the FLOAT_SCALAR and integer types */
                if (kind == NPY_FLOAT_SCALAR && HPyArrayDescr_ISINTEGER(o1_descr)) {
                    HPy h_dtype = HPyArray_DescrFromType(ctx, NPY_DOUBLE);
                    PyArray_Descr *dtype = PyArray_Descr_AsStruct(ctx, h_dtype);
                    CAPI_WARN("pow with exponent 2.0, FLOAT_SCALAR and integer types needs PyArray_CastToType");
                    a1 = (PyArrayObject *)PyArray_CastToType(a1, dtype,
                            PyArray_ISFORTRAN(a1));
                    if (a1 != NULL) {
                        /* cast always creates a new array */
                        *value = HPyArray_GenericInplaceUnaryFunction(ctx, h_o1, fastop);
                    }
                }
                else {
                    *value = HPyArray_GenericUnaryFunction(ctx, h_o1, fastop);
                }
            }
            return 0;
        }
    }
    /* no fast operation found */
    return -1;
}

HPyDef_SLOT(array_power, HPy_nb_power);
static HPy array_power_impl(HPyContext *ctx, HPy a1, HPy o2, HPy modulo)
{
    HPy value = HPy_NULL;

    if (!HPy_Is(ctx, modulo, ctx->h_None)) {
        /* modular exponentiation is not implemented (gh-8804) */
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }

    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, a1, o2, &array_power);
    if (fast_scalar_power(ctx, a1, o2, 0, &value) != 0) {
        value = HPyArray_GenericBinaryFunction(ctx, a1, o2, hpy_n_ops.power);
    }
    return value;
}

HPyDef_SLOT(array_positive_slot, HPy_nb_positive);
static HPy
array_positive_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1)
{
    /*
     * For backwards compatibility, where + just implied a copy,
     * we cannot just call N_OPS_GET(positive).  Instead, we do the following
     * 1. Try N_OPS_GET(positive)
     * 2. If we get an exception, check whether __array_ufunc__ is
     *    overridden; if so, we live in the future and we allow the
     *    TypeError to be passed on.
     * 3. If not, give a deprecation warning and return a copy.
     */
    HPy value;
    // if (can_elide_temp_unary(m1)) {
    //     value = PyArray_GenericInplaceUnaryFunction(m1, N_OPS_GET(positive));
    // }
    // else {
        value = HPyArray_GenericUnaryFunction(ctx, m1, hpy_n_ops.positive);
    // }
    if (HPy_IsNull(value)) {
        /*
         * We first fetch the error, as it needs to be clear to check
         * for the override.  When the deprecation is removed,
         * this whole stanza can be deleted.
         */
        CAPI_WARN("mssing PyErr_Fetch & PyErr_Restore");
        PyObject *exc, *val, *tb;
        PyErr_Fetch(&exc, &val, &tb);
        if (HPyUFunc_HasOverride(ctx, m1)) {
            PyErr_Restore(exc, val, tb);
            return HPy_NULL;
        }
        Py_XDECREF(exc);
        Py_XDECREF(val);
        Py_XDECREF(tb);

        /* 2018-06-28, 1.16.0 */
        if (HPY_DEPRECATE(ctx, "Applying '+' to a non-numerical array is "
                      "ill-defined. Returning a copy, but in the future "
                      "this will error.") < 0) {
            return HPy_NULL;
        }
        HPy v = HPyArray_Copy(ctx, m1);
        value = HPyArray_Return(ctx, v);
        HPy_Close(ctx, v);
    }
    return value;
}

HPyDef_SLOT(array_negative_slot, HPy_nb_negative);
static HPy
array_negative_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1)
{
    // if (can_elide_temp_unary(m1)) {
    //     return PyArray_GenericInplaceUnaryFunction(m1, N_OPS_GET(negative));
    // }
    return HPyArray_GenericUnaryFunction(ctx, m1, hpy_n_ops.negative);
}

HPyDef_SLOT(array_absolute_slot, HPy_nb_absolute);
static HPy
array_absolute_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1)
{
    // if (can_elide_temp_unary(m1) && !PyArray_ISCOMPLEX(m1)) {
    //     return PyArray_GenericInplaceUnaryFunction(m1, N_OPS_GET(absolute));
    // }
    return HPyArray_GenericUnaryFunction(ctx, m1, hpy_n_ops.absolute);
}

HPyDef_SLOT(array_invert_slot, HPy_nb_invert);
NPY_NO_EXPORT HPy
array_invert_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1)
{
    // if (can_elide_temp_unary(m1)) {
    //     return PyArray_GenericInplaceUnaryFunction(m1, N_OPS_GET(invert));
    // }
    return HPyArray_GenericUnaryFunction(ctx, m1, hpy_n_ops.invert);
}

HPyDef_SLOT(array_left_shift_slot, HPy_nb_lshift);
NPY_NO_EXPORT HPy
array_left_shift_slot_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    // PyObject *res;

    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_left_shift_slot);
    // if (try_binary_elide(m1, m2, &array_inplace_left_shift, &res, 0)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, HPY_N_OPS(left_shift));
}

HPyDef_SLOT(array_right_shift_slot, HPy_nb_rshift);
NPY_NO_EXPORT HPy
array_right_shift_slot_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    // PyObject *res;

    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_right_shift_slot);
    // if (try_binary_elide(m1, m2, &array_inplace_right_shift, &res, 0)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, HPY_N_OPS(right_shift));
}

HPyDef_SLOT(array_bitwise_and, HPy_nb_and)
NPY_NO_EXPORT HPy
array_bitwise_and_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_bitwise_and);
    // This hack is too much for HPy
    // if (try_binary_elide(m1, m2, &array_inplace_bitwise_and, &res, 1)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, hpy_n_ops.bitwise_and);
}


HPyDef_SLOT(array_bitwise_or, HPy_nb_or)
NPY_NO_EXPORT HPy
array_bitwise_or_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_bitwise_or);
    // This hack is too much for HPy
    // if (try_binary_elide(m1, m2, &array_inplace_bitwise_and, &res, 1)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, hpy_n_ops.bitwise_or);
}

HPyDef_SLOT(array_bitwise_xor, HPy_nb_xor)
NPY_NO_EXPORT HPy
array_bitwise_xor_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_bitwise_xor);
    // This hack is too much for HPy
    // if (try_binary_elide(m1, m2, &array_inplace_bitwise_and, &res, 1)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, hpy_n_ops.bitwise_xor);
}

HPyDef_SLOT(array_inplace_add, HPy_nb_inplace_add)
HPy array_inplace_add_impl(HPyContext *ctx, /*PyArrayObject*/HPy m1, /*PyObject*/HPy m2)
{    
    // If m2's nb_inplace_add != array_inplace_add => return NotImplemented
    HPY_INPLACE_GIVE_UP_IF_NEEDED(
            ctx, m1, m2, &array_inplace_add);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, hpy_n_ops.add);
}

HPyDef_SLOT(array_inplace_subtract_slot, HPy_nb_inplace_subtract);
static HPy
array_inplace_subtract_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_subtract_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, HPY_N_OPS(subtract));
}

HPyDef_SLOT(array_inplace_multiply_slot, HPy_nb_inplace_multiply);
NPY_NO_EXPORT HPy
array_inplace_multiply_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_multiply_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, HPY_N_OPS(multiply));
}

HPyDef_SLOT(array_inplace_remainder_slot, HPy_nb_inplace_remainder);
NPY_NO_EXPORT HPy
array_inplace_remainder_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_remainder_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, HPY_N_OPS(remainder));
}

HPyDef_SLOT(array_inplace_power_slot, HPy_nb_inplace_power);
NPY_NO_EXPORT HPy
array_inplace_power_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ a1, HPy o2, HPy NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    HPy value = HPy_NULL;

    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            a1, o2, &array_inplace_power_slot);
    // TODO HPY LABS PORT
    // if (fast_scalar_power((PyObject *)a1, o2, 1, &value) != 0) {
        value = HPyArray_GenericInplaceBinaryFunction(ctx, a1, o2, HPY_N_OPS(power));
    // }
    return value;
}

HPyDef_SLOT(array_inplace_left_shift_slot, HPy_nb_inplace_lshift);
NPY_NO_EXPORT HPy
array_inplace_left_shift_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_left_shift_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, HPY_N_OPS(left_shift));
}

HPyDef_SLOT(array_inplace_right_shift_slot, HPy_nb_inplace_rshift);
NPY_NO_EXPORT HPy
array_inplace_right_shift_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_right_shift_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, HPY_N_OPS(right_shift));
}

HPyDef_SLOT(array_inplace_bitwise_and_slot, HPy_nb_inplace_and);
NPY_NO_EXPORT HPy
array_inplace_bitwise_and_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_bitwise_and_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, HPY_N_OPS(bitwise_and));
}

HPyDef_SLOT(array_inplace_bitwise_or_slot, HPy_nb_inplace_or);
NPY_NO_EXPORT HPy
array_inplace_bitwise_or_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_bitwise_or_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, HPY_N_OPS(bitwise_or));
}

HPyDef_SLOT(array_inplace_bitwise_xor_slot, HPy_nb_inplace_xor);
NPY_NO_EXPORT HPy
array_inplace_bitwise_xor_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_bitwise_xor_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, HPY_N_OPS(bitwise_xor));
}

HPyDef_SLOT(array_floor_divide_slot, HPy_nb_floor_divide);
NPY_NO_EXPORT HPy
array_floor_divide_slot_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    // PyObject *res;

    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_floor_divide_slot);
    // if (try_binary_elide(m1, m2, &array_inplace_floor_divide, &res, 0)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, HPY_N_OPS(floor_divide));
}

HPyDef_SLOT(array_true_divide, HPy_nb_true_divide);
NPY_NO_EXPORT HPy
array_true_divide_impl(HPyContext *ctx, HPy m1, HPy m2)
{
    // PyObject *res;

    HPY_BINOP_GIVE_UP_IF_NEEDED(ctx, m1, m2, &array_true_divide);
    // We cannot support this hack on HPy:
    // if (PyArray_CheckExact(m1) &&
    //         (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) &&
    //         try_binary_elide(m1, m2, &array_inplace_true_divide, &res, 0)) {
    //     return res;
    // }
    return HPyArray_GenericBinaryFunction(ctx, m1, m2, hpy_n_ops.true_divide);
}

HPyDef_SLOT(array_inplace_floor_divide_slot, HPy_nb_inplace_floor_divide);
NPY_NO_EXPORT HPy
array_inplace_floor_divide_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_floor_divide_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2,
                                                HPY_N_OPS(floor_divide));
}

HPyDef_SLOT(array_inplace_true_divide_slot, HPy_nb_inplace_true_divide);
NPY_NO_EXPORT HPy
array_inplace_true_divide_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ m1, HPy m2)
{
    HPY_INPLACE_GIVE_UP_IF_NEEDED(ctx,
            m1, m2, &array_inplace_true_divide_slot);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2,
                                                HPY_N_OPS(true_divide));
}


HPyDef_SLOT(_array_nonzero_slot, HPy_nb_bool);
NPY_NO_EXPORT int
_array_nonzero_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ mp)
{
    npy_intp n;
    PyArrayObject *mp_struct = PyArrayObject_AsStruct(ctx, mp);
    n = PyArray_SIZE(mp_struct);
    if (n == 1) {
        int res;
        HPy mp_descr = HPyArray_DESCR(ctx, mp, mp_struct);
        CAPI_WARN("missing Py_EnterRecursiveCall");
        if (Py_EnterRecursiveCall(" while converting array to bool")) {
            return -1;
        }
        CAPI_WARN("calling f->nonzero");
        PyObject *py_mp = HPy_AsPyObject(ctx, mp);
        res = PyArray_Descr_AsStruct(ctx,mp_descr)->f->nonzero(PyArray_DATA(mp_struct), py_mp);
        /* nonzero has no way to indicate an error, but one can occur */
        if (HPyErr_Occurred(ctx)) {
            res = -1;
        }
        Py_LeaveRecursiveCall();
        return res;
    }
    else if (n == 0) {
        /* 2017-09-25, 1.14 */
        if (HPY_DEPRECATE(ctx, "The truth value of an empty array is ambiguous. "
                      "Returning False, but in future this will result in an error. "
                      "Use `array.size > 0` to check that an array is not empty.") < 0) {
            return -1;
        }
        return 0;
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "The truth value of an array "
                        "with more than one element is ambiguous. "
                        "Use a.any() or a.all()");
        return -1;
    }
}

/*
 * Convert the array to a scalar if allowed, and apply the builtin function
 * to it. The where argument is passed onto Py_EnterRecursiveCall when the
 * array contains python objects.
 */
NPY_NO_EXPORT HPy
array_scalar_forward(HPyContext *ctx, /*PyArrayObject*/ HPy h_v,
                     HPy (*builtin_func)(HPyContext *, HPy),
                     const char *where)
{
    PyArrayObject *v = PyArrayObject_AsStruct(ctx, h_v);
    if (HPyArray_SIZE(v) != 1) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "only size-1 arrays can be"\
                        " converted to Python scalars");
        return HPy_NULL;
    }

    HPy h_descr = HPyArray_DESCR(ctx, h_v, v);
    PyArray_Descr *descr = PyArray_Descr_AsStruct(ctx, h_descr);
    HPy h_scalar = HPyArray_DESCR_GETITEM(ctx, descr, h_v, v, PyArray_DATA(v));
    HPy_Close(ctx, h_descr);
    if (HPy_IsNull(h_scalar)) {
        return HPy_NULL;
    }

    /* Need to guard against recursion if our array holds references */
    if (PyDataType_REFCHK(descr)) {
        hpy_abort_not_implemented("arrays with references...");
        // PyObject *res;
        // if (Py_EnterRecursiveCall(where) != 0) {
        //     Py_DECREF(scalar);
        //     return NULL;
        // }
        // res = builtin_func(scalar);
        // Py_DECREF(scalar);
        // Py_LeaveRecursiveCall();
        // return res;
    }
    else {
        HPy res = builtin_func(ctx, h_scalar);
        HPy_Close(ctx, h_scalar);
        return res;
    }
}


HPyDef_SLOT(array_float, HPy_nb_float)
NPY_NO_EXPORT HPy
array_float_impl(HPyContext *ctx, /*PyArrayObject*/ HPy h_v)
{
    return array_scalar_forward(ctx, h_v, &HPy_Float, " in ndarray.__float__");
}

HPyDef_SLOT(array_int, HPy_nb_int)
NPY_NO_EXPORT HPy
array_int_impl(HPyContext *ctx, HPy h_v)
{
    return array_scalar_forward(ctx, h_v, &HPy_Long, " in ndarray.__int__");
}

HPyDef_SLOT(array_index_slot, HPy_nb_index);
NPY_NO_EXPORT HPy
array_index_slot_impl(HPyContext *ctx, HPy /* PyArrayObject * */ v)
{
    PyArrayObject *v_struct = PyArrayObject_AsStruct(ctx, v);
    if (!PyArray_ISINTEGER(v_struct) || PyArray_NDIM(v_struct) != 0) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
            "only integer scalar arrays can be converted to a scalar index");
        return HPy_NULL;
    }
    return HPyArray_GETITEM(ctx, v, PyArray_DATA(v_struct));
}



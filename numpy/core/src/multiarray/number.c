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

/*NUMPY_API
 *Set internal structure with number functions that all arrays will use
 */
NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict)
{
    /* 2018-09-09, 1.16 */
    if (DEPRECATE("PyArray_SetNumericOps is deprecated. Use "
        "PyUFunc_ReplaceLoopBySignature to replace ufunc inner loop functions "
        "instead.") < 0) {
        return -1;
    }
    hpy_abort_not_implemented("PyArray_SetNumericOps");
}

/* Note - macro contains goto */
#define GET(op) if (N_OPS_GET(op) &&                                         \
                    (PyDict_SetItemString(dict, #op, N_OPS_GET(op))==-1))    \
        goto fail;

NPY_NO_EXPORT PyObject *
_PyArray_GetNumericOps(void)
{
    PyObject *dict;
    if ((dict = PyDict_New())==NULL)
        return NULL;
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
    Py_DECREF(dict);
    return NULL;
}

/*NUMPY_API
  Get dictionary showing number functions that all arrays will use
*/
NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void)
{
    /* 2018-09-09, 1.16 */
    if (DEPRECATE("PyArray_GetNumericOps is deprecated.") < 0) {
        return NULL;
    }
    return _PyArray_GetNumericOps();
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

NPY_NO_EXPORT HPy
HPyArray_GenericBinaryFunction(HPyContext *ctx, HPy m1, HPy m2, HPyGlobal op)
{
    HPy args = HPyTuple_Pack(ctx, 2, m1, m2);
    HPy callable = HPyGlobal_Load(ctx, op);
    HPy res = HPy_CallTupleDict(ctx, callable, args, HPy_NULL);
    HPy_Close(ctx, args);
    HPy_Close(ctx, callable);
    return res;
}

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, NULL);
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
    HPy args = HPyTuple_Pack(ctx, 3, m1, m2, m1);
    HPy callable = HPyGlobal_Load(ctx, op);
    HPy res = HPy_CallTupleDict(ctx, callable, args, HPy_NULL);
    HPy_Close(ctx, args);
    HPy_Close(ctx, callable);
    return res;
}

static PyObject *
PyArray_GenericInplaceUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, m1, NULL);
}

NPY_NO_EXPORT PyObject *
array_add(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    hpy_abort_not_implemented("array_add...");
    // BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_add, array_add);
    // if (try_binary_elide(m1, m2, &array_inplace_add, &res, 1)) {
    //     return res;
    // }
    // return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(add));
}

NPY_NO_EXPORT PyObject *
array_subtract(PyObject *m1, PyObject *m2)
{
    CAPI_WARN("array_subtract");
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_subtract, array_subtract);
    if (try_binary_elide(m1, m2, &array_inplace_subtract, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(subtract));
}

NPY_NO_EXPORT PyObject *
array_multiply(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_multiply, array_multiply);
    if (try_binary_elide(m1, m2, &array_inplace_multiply, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(multiply));
}

NPY_NO_EXPORT PyObject *
array_remainder(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_remainder, array_remainder);
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(remainder));
}

NPY_NO_EXPORT PyObject *
array_divmod(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_divmod, array_divmod);
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(divmod));
}

/* Need this to be version dependent on account of the slot check */
NPY_NO_EXPORT PyObject *
array_matrix_multiply(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_matrix_multiply, array_matrix_multiply);
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(matmul));
}

NPY_NO_EXPORT PyObject *
array_inplace_matrix_multiply(
        PyArrayObject *NPY_UNUSED(m1), PyObject *NPY_UNUSED(m2))
{
    PyErr_SetString(PyExc_TypeError,
                    "In-place matrix multiplication is not (yet) supported. "
                    "Use 'a = a @ b' instead of 'a @= b'.");
    return NULL;
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

    if (!HPyArray_Check(ctx, h_o1) ||
        HPyArray_ISOBJECT(ctx, h_o1)) {
        /* no fast operation found */
        return -1;
    }
    PyArrayObject *o1 = PyArrayObject_AsStruct(ctx, h_o1);
    if ((kind=is_scalar_with_conversion(ctx, h_o2, &exponent))>0) {
        CAPI_WARN("fast_scalar_power: is_scalar_with_conversion branch");
        PyArrayObject *a1 = (PyArrayObject *)o1;
        PyObject *fastop = NULL;
        if (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) {
            if (exponent == 1.0) {
                fastop = N_OPS_GET(positive);
            }
            else if (exponent == -1.0) {
                fastop = N_OPS_GET(reciprocal);
            }
            else if (exponent ==  0.0) {
                fastop = N_OPS_GET(_ones_like);
            }
            else if (exponent ==  0.5) {
                fastop = N_OPS_GET(sqrt);
            }
            else if (exponent ==  2.0) {
                fastop = N_OPS_GET(square);
            }
            else {
                return -1;
            }

            if (inplace || can_elide_temp_unary(a1)) {
                PyObject *tmp = PyArray_GenericInplaceUnaryFunction(a1, fastop);
                *value = HPy_FromPyObject(ctx, tmp);
                Py_DECREF(tmp);
            }
            else {
                PyObject *tmp = PyArray_GenericUnaryFunction(a1, fastop);
                *value = HPy_FromPyObject(ctx, tmp);
                Py_DECREF(tmp);
            }
            return 0;
        }
        /* Because this is called with all arrays, we need to
         *  change the output if the kind of the scalar is different
         *  than that of the input and inplace is not on ---
         *  (thus, the input should be up-cast)
         */
        else if (exponent == 2.0) {
            fastop = N_OPS_GET(square);
            if (inplace) {
                PyObject *tmp = PyArray_GenericInplaceUnaryFunction(a1, fastop);
                *value = HPy_FromPyObject(ctx, tmp);
                Py_DECREF(tmp);
            }
            else {
                /* We only special-case the FLOAT_SCALAR and integer types */
                if (kind == NPY_FLOAT_SCALAR && PyArray_ISINTEGER(a1)) {
                    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_DOUBLE);
                    a1 = (PyArrayObject *)PyArray_CastToType(a1, dtype,
                            PyArray_ISFORTRAN(a1));
                    if (a1 != NULL) {
                        /* cast always creates a new array */
                        PyObject *tmp = PyArray_GenericInplaceUnaryFunction(a1, fastop);
                        *value = HPy_FromPyObject(ctx, tmp);
                        Py_DECREF(tmp);
                        Py_DECREF(a1);
                    }
                }
                else {
                    PyObject *tmp = PyArray_GenericUnaryFunction(a1, fastop);
                    *value = HPy_FromPyObject(ctx, tmp);
                    Py_DECREF(tmp);
                }
            }
            return 0;
        }
    }
    /* no fast operation found */
    return -1;
}

HPyDef_SLOT(array_power, array_power_impl, HPy_nb_power);
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

NPY_NO_EXPORT PyObject *
array_positive(PyArrayObject *m1)
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
    PyObject *value;
    if (can_elide_temp_unary(m1)) {
        value = PyArray_GenericInplaceUnaryFunction(m1, N_OPS_GET(positive));
    }
    else {
        value = PyArray_GenericUnaryFunction(m1, N_OPS_GET(positive));
    }
    if (value == NULL) {
        /*
         * We first fetch the error, as it needs to be clear to check
         * for the override.  When the deprecation is removed,
         * this whole stanza can be deleted.
         */
        PyObject *exc, *val, *tb;
        PyErr_Fetch(&exc, &val, &tb);
        if (PyUFunc_HasOverride((PyObject *)m1)) {
            PyErr_Restore(exc, val, tb);
            return NULL;
        }
        Py_XDECREF(exc);
        Py_XDECREF(val);
        Py_XDECREF(tb);

        /* 2018-06-28, 1.16.0 */
        if (DEPRECATE("Applying '+' to a non-numerical array is "
                      "ill-defined. Returning a copy, but in the future "
                      "this will error.") < 0) {
            return NULL;
        }
        value = PyArray_Return((PyArrayObject *)PyArray_Copy(m1));
    }
    return value;
}

NPY_NO_EXPORT PyObject *
array_negative(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, N_OPS_GET(negative));
    }
    return PyArray_GenericUnaryFunction(m1, N_OPS_GET(negative));
}

NPY_NO_EXPORT PyObject *
array_absolute(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1) && !PyArray_ISCOMPLEX(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, N_OPS_GET(absolute));
    }
    return PyArray_GenericUnaryFunction(m1, N_OPS_GET(absolute));
}

NPY_NO_EXPORT PyObject *
array_invert(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, N_OPS_GET(invert));
    }
    return PyArray_GenericUnaryFunction(m1, N_OPS_GET(invert));
}

NPY_NO_EXPORT PyObject *
array_left_shift(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_lshift, array_left_shift);
    if (try_binary_elide(m1, m2, &array_inplace_left_shift, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(left_shift));
}

NPY_NO_EXPORT PyObject *
array_right_shift(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_rshift, array_right_shift);
    if (try_binary_elide(m1, m2, &array_inplace_right_shift, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(right_shift));
}

NPY_NO_EXPORT PyObject *
array_bitwise_and(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_and, array_bitwise_and);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_and, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(bitwise_and));
}

NPY_NO_EXPORT PyObject *
array_bitwise_or(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_or, array_bitwise_or);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_or, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(bitwise_or));
}

NPY_NO_EXPORT PyObject *
array_bitwise_xor(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_xor, array_bitwise_xor);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_xor, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(bitwise_xor));
}

HPyDef_SLOT(array_inplace_add, array_inplace_add_impl, HPy_nb_inplace_add)
HPy array_inplace_add_impl(HPyContext *ctx, /*PyArrayObject*/HPy m1, /*PyObject*/HPy m2)
{    
    // If m2's nb_inplace_add != array_inplace_add => return NotImplemented
    HPY_INPLACE_GIVE_UP_IF_NEEDED(
            ctx, m1, m2, &array_inplace_add);
    return HPyArray_GenericInplaceBinaryFunction(ctx, m1, m2, hpy_n_ops.add);
}

NPY_NO_EXPORT PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_subtract, array_inplace_subtract);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, N_OPS_GET(subtract));
}

NPY_NO_EXPORT PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_multiply, array_inplace_multiply);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, N_OPS_GET(multiply));
}

NPY_NO_EXPORT PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_remainder, array_inplace_remainder);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, N_OPS_GET(remainder));
}

NPY_NO_EXPORT PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value = NULL;

    INPLACE_GIVE_UP_IF_NEEDED(
            a1, o2, nb_inplace_power, array_inplace_power);
    // TODO HPY LABS PORT
    // if (fast_scalar_power((PyObject *)a1, o2, 1, &value) != 0) {
        value = PyArray_GenericInplaceBinaryFunction(a1, o2, N_OPS_GET(power));
    // }
    return value;
}

NPY_NO_EXPORT PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_lshift, array_inplace_left_shift);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, N_OPS_GET(left_shift));
}

NPY_NO_EXPORT PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_rshift, array_inplace_right_shift);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, N_OPS_GET(right_shift));
}

NPY_NO_EXPORT PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_and, array_inplace_bitwise_and);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, N_OPS_GET(bitwise_and));
}

NPY_NO_EXPORT PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_or, array_inplace_bitwise_or);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, N_OPS_GET(bitwise_or));
}

NPY_NO_EXPORT PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_xor, array_inplace_bitwise_xor);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, N_OPS_GET(bitwise_xor));
}

NPY_NO_EXPORT PyObject *
array_floor_divide(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_floor_divide, array_floor_divide);
    if (try_binary_elide(m1, m2, &array_inplace_floor_divide, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(floor_divide));
}

NPY_NO_EXPORT PyObject *
array_true_divide(PyObject *m1, PyObject *m2)
{
    CAPI_WARN("array_true_divide");
    PyObject *res;
    PyArrayObject *a1 = (PyArrayObject *)m1;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_true_divide, array_true_divide);
    if (PyArray_CheckExact(m1) &&
            (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) &&
            try_binary_elide(m1, m2, &array_inplace_true_divide, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, N_OPS_GET(true_divide));
}

NPY_NO_EXPORT PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_floor_divide, array_inplace_floor_divide);
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                N_OPS_GET(floor_divide));
}

NPY_NO_EXPORT PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_true_divide, array_inplace_true_divide);
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                N_OPS_GET(true_divide));
}


NPY_NO_EXPORT int
_array_nonzero(PyArrayObject *mp)
{
    npy_intp n;

    n = PyArray_SIZE(mp);
    if (n == 1) {
        int res;
        if (Py_EnterRecursiveCall(" while converting array to bool")) {
            return -1;
        }
        res = PyArray_DESCR(mp)->f->nonzero(PyArray_DATA(mp), mp);
        /* nonzero has no way to indicate an error, but one can occur */
        if (PyErr_Occurred()) {
            res = -1;
        }
        Py_LeaveRecursiveCall();
        return res;
    }
    else if (n == 0) {
        /* 2017-09-25, 1.14 */
        if (DEPRECATE("The truth value of an empty array is ambiguous. "
                      "Returning False, but in future this will result in an error. "
                      "Use `array.size > 0` to check that an array is not empty.") < 0) {
            return -1;
        }
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
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
NPY_NO_EXPORT PyObject *
array_scalar_forward(PyArrayObject *v,
                     PyObject *(*builtin_func)(PyObject *),
                     const char *where)
{
    PyObject *scalar;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only size-1 arrays can be"\
                        " converted to Python scalars");
        return NULL;
    }

    scalar = PyArray_GETITEM(v, PyArray_DATA(v));
    if (scalar == NULL) {
        return NULL;
    }

    /* Need to guard against recursion if our array holds references */
    if (PyDataType_REFCHK(PyArray_DESCR(v))) {
        PyObject *res;
        if (Py_EnterRecursiveCall(where) != 0) {
            Py_DECREF(scalar);
            return NULL;
        }
        res = builtin_func(scalar);
        Py_DECREF(scalar);
        Py_LeaveRecursiveCall();
        return res;
    }
    else {
        PyObject *res;
        res = builtin_func(scalar);
        Py_DECREF(scalar);
        return res;
    }
}


NPY_NO_EXPORT PyObject *
array_float(PyArrayObject *v)
{
    return array_scalar_forward(v, &PyNumber_Float, " in ndarray.__float__");
}

NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v)
{
    return array_scalar_forward(v, &PyNumber_Long, " in ndarray.__int__");
}

NPY_NO_EXPORT PyObject *
array_index(PyArrayObject *v)
{
    if (!PyArray_ISINTEGER(v) || PyArray_NDIM(v) != 0) {
        PyErr_SetString(PyExc_TypeError,
            "only integer scalar arrays can be converted to a scalar index");
        return NULL;
    }
    return PyArray_GETITEM(v, PyArray_DATA(v));
}



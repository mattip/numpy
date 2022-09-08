#ifndef NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_

typedef struct {
    PyObject *add;
    PyObject *subtract;
    PyObject *multiply;
    PyObject *divide;
    PyObject *remainder;
    PyObject *divmod;
    PyObject *power;
    PyObject *square;
    PyObject *reciprocal;
    PyObject *_ones_like;
    PyObject *sqrt;
    PyObject *cbrt;
    PyObject *negative;
    PyObject *positive;
    PyObject *absolute;
    PyObject *invert;
    PyObject *left_shift;
    PyObject *right_shift;
    PyObject *bitwise_and;
    PyObject *bitwise_xor;
    PyObject *bitwise_or;
    PyObject *less;
    PyObject *less_equal;
    PyObject *equal;
    PyObject *not_equal;
    PyObject *greater;
    PyObject *greater_equal;
    PyObject *floor_divide;
    PyObject *true_divide;
    PyObject *logical_or;
    PyObject *logical_and;
    PyObject *floor;
    PyObject *ceil;
    PyObject *maximum;
    PyObject *minimum;
    PyObject *rint;
    PyObject *conjugate;
    PyObject *matmul;
    PyObject *clip;
} NumericOps;

typedef struct {
    HPyGlobal add;
    HPyGlobal subtract;
    HPyGlobal multiply;
    HPyGlobal divide;
    HPyGlobal remainder;
    HPyGlobal divmod;
    HPyGlobal power;
    HPyGlobal square;
    HPyGlobal reciprocal;
    HPyGlobal _ones_like;
    HPyGlobal sqrt;
    HPyGlobal cbrt;
    HPyGlobal negative;
    HPyGlobal positive;
    HPyGlobal absolute;
    HPyGlobal invert;
    HPyGlobal left_shift;
    HPyGlobal right_shift;
    HPyGlobal bitwise_and;
    HPyGlobal bitwise_xor;
    HPyGlobal bitwise_or;
    HPyGlobal less;
    HPyGlobal less_equal;
    HPyGlobal equal;
    HPyGlobal not_equal;
    HPyGlobal greater;
    HPyGlobal greater_equal;
    HPyGlobal floor_divide;
    HPyGlobal true_divide;
    HPyGlobal logical_or;
    HPyGlobal logical_and;
    HPyGlobal floor;
    HPyGlobal ceil;
    HPyGlobal maximum;
    HPyGlobal minimum;
    HPyGlobal rint;
    HPyGlobal conjugate;
    HPyGlobal matmul;
    HPyGlobal clip;
} HPyNumericOps;

extern NPY_NO_EXPORT NumericOps n_ops;
extern NPY_NO_EXPORT HPyNumericOps hpy_n_ops;

static inline PyObject *_n_ops_get(PyObject **pyobj, HPyGlobal h_global) {
    if (*pyobj == NULL) {
        HPyContext *ctx = npy_get_context();
        *pyobj = HPy_AsPyObject(ctx, HPyGlobal_Load(ctx, h_global));
    }
    return *pyobj;
}

#define N_OPS_GET(name)     _n_ops_get(&n_ops.name, hpy_n_ops.name)

NPY_NO_EXPORT extern HPyDef array_add;

NPY_NO_EXPORT extern HPyDef array_subtract;

NPY_NO_EXPORT PyObject *
array_multiply(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_remainder(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_divmod(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_matrix_multiply(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_matrix_multiply(
        PyArrayObject *NPY_UNUSED(m1), PyObject *NPY_UNUSED(m2));

NPY_NO_EXPORT extern HPyDef array_power;

NPY_NO_EXPORT PyObject *
array_positive(PyArrayObject *m1);

NPY_NO_EXPORT PyObject *
array_negative(PyArrayObject *m1);

NPY_NO_EXPORT PyObject *
array_absolute(PyArrayObject *m1);

NPY_NO_EXPORT PyObject *
array_invert(PyArrayObject *m1);

NPY_NO_EXPORT PyObject *
array_left_shift(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_right_shift(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT extern HPyDef array_bitwise_and;

NPY_NO_EXPORT extern HPyDef array_bitwise_or;

NPY_NO_EXPORT extern HPyDef array_bitwise_xor;

NPY_NO_EXPORT extern HPyDef array_inplace_add;

NPY_NO_EXPORT PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo));

NPY_NO_EXPORT PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_floor_divide(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT extern HPyDef array_true_divide;

NPY_NO_EXPORT PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT int
_array_nonzero(PyArrayObject *mp);

NPY_NO_EXPORT extern HPyDef array_float;

NPY_NO_EXPORT extern HPyDef array_int;

NPY_NO_EXPORT PyObject *
array_index(PyArrayObject *v);

NPY_NO_EXPORT int
_PyArray_SetNumericOps(HPyContext *ctx, HPy dict);

NPY_NO_EXPORT PyObject *
_PyArray_GetNumericOps(void);

NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyObject *m1, PyObject *m2, PyObject *op);

NPY_NO_EXPORT HPy
HPyArray_GenericBinaryFunction(HPyContext *ctx, HPy m1, HPy m2, HPyGlobal op);

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out);

NPY_NO_EXPORT HPy
HPyArray_GenericReduceFunction(HPyContext *ctx, 
                                HPy /* PyArrayObject * */ m1, 
                                HPy op, int axis, int rtype, 
                                HPy /* PyArrayObject * */ out);
                                
NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_ */

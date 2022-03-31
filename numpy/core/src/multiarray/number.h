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

extern NPY_NO_EXPORT NumericOps n_ops;

NPY_NO_EXPORT PyObject *
array_add(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_subtract(PyObject *m1, PyObject *m2);

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

NPY_NO_EXPORT PyObject *
array_power(PyObject *a1, PyObject *o2, PyObject *modulo);

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

NPY_NO_EXPORT PyObject *
array_bitwise_and(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_bitwise_or(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_bitwise_xor(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT HPyDef array_inplace_add;

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

NPY_NO_EXPORT PyObject *
array_true_divide(PyObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2);

NPY_NO_EXPORT int
_array_nonzero(PyArrayObject *mp);

NPY_NO_EXPORT PyObject *
array_float(PyArrayObject *v);

NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v);

NPY_NO_EXPORT PyObject *
array_index(PyArrayObject *v);

NPY_NO_EXPORT int
_PyArray_SetNumericOps(PyObject *dict);

NPY_NO_EXPORT PyObject *
_PyArray_GetNumericOps(void);

NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyObject *m1, PyObject *m2, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out);

NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NUMBER_H_ */

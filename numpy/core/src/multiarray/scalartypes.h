#ifndef NUMPY_CORE_SRC_MULTIARRAY_SCALARTYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_SCALARTYPES_H_

/* Internal look-up tables */
extern NPY_NO_EXPORT unsigned char
_npy_can_cast_safely_table[NPY_NTYPES][NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_scalar_kinds_table[NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_type_promotion_table[NPY_NTYPES][NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_smallest_type_of_kind_table[NPY_NSCALARKINDS];
extern NPY_NO_EXPORT signed char
_npy_next_larger_type_table[NPY_NTYPES];

NPY_NO_EXPORT void
initialize_casting_tables(void);

NPY_NO_EXPORT void
initialize_numeric_types(void);

NPY_NO_EXPORT void
gentype_struct_free(PyObject *ptr);

NPY_NO_EXPORT int
is_anyscalar_exact(PyObject *obj);

NPY_NO_EXPORT int
hpy_is_anyscalar_exact(HPyContext *ctx, HPy obj);

NPY_NO_EXPORT int
_typenum_fromtypeobj(PyObject *type, int user);

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr);

NPY_NO_EXPORT int
init_scalartypes_basetypes(HPyContext *ctx);

NPY_NO_EXPORT void *
hpy_scalar_value(HPyContext *ctx, HPy scalar, PyArray_Descr *descr);

NPY_NO_EXPORT int
_hpy_typenum_fromtypeobj(HPyContext *ctx, HPy type, int user);

NPY_NO_EXPORT int
hpy_is_anyscalar_exact(HPyContext *ctx, HPy obj);

NPY_NO_EXPORT int
hpy_is_type_anyscalar_exact(HPyContext *ctx, HPy type);

NPY_NO_EXPORT extern HPyType_Spec PyGenericArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyBoolArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyNumberArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyIntegerArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PySignedIntegerArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyUnsignedIntegerArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyInexactArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyFloatingArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyComplexFloatingArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyFlexibleArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyCharacterArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyByteArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyShortArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyIntArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyLongArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyLongLongArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyUByteArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyUShortArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyUIntArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyULongArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyULongLongArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyFloatArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyDoubleArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyLongDoubleArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyCFloatArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyCDoubleArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyCLongDoubleArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyObjectArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyStringArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyUnicodeArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyVoidArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyTimeIntegerArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyDatetimeArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyTimedeltaArrType_Type_spec;
NPY_NO_EXPORT extern HPyType_Spec PyHalfArrType_Type_spec;


NPY_NO_EXPORT extern PyTypeObject *_PyGenericArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyBoolArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyNumberArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyIntegerArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PySignedIntegerArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyUnsignedIntegerArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyInexactArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyFloatingArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyComplexFloatingArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyFlexibleArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyCharacterArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyByteArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyShortArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyIntArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyLongArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyLongLongArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyUByteArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyUShortArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyUIntArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyULongArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyULongLongArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyFloatArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyDoubleArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyLongDoubleArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyCFloatArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyCDoubleArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyCLongDoubleArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyObjectArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyStringArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyUnicodeArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyVoidArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyTimeIntegerArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyDatetimeArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyTimedeltaArrType_Type_p;
NPY_NO_EXPORT extern PyTypeObject *_PyHalfArrType_Type_p;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_SCALARTYPES_H_ */

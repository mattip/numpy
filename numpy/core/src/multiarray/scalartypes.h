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

NPY_NO_EXPORT int
_hpy_typenum_fromtypeobj(HPyContext *ctx, HPy type, int user);

NPY_NO_EXPORT int
hpy_is_anyscalar_exact(HPyContext *ctx, HPy obj);

NPY_NO_EXPORT int
hpy_is_type_anyscalar_exact(HPyContext *ctx, HPy type);

NPY_NO_EXPORT HPyType_Spec PyGenericArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyBoolArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyNumberArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyIntegerArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PySignedIntegerArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyUnsignedIntegerArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyInexactArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyFloatingArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyComplexFloatingArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyFlexibleArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyCharacterArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyByteArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyShortArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyIntArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyLongArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyLongLongArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyUByteArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyUShortArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyUIntArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyULongArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyULongLongArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyFloatArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyDoubleArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyLongDoubleArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyCFloatArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyCDoubleArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyCLongDoubleArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyObjectArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyStringArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyUnicodeArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyVoidArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyTimeIntegerArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyDatetimeArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyTimedeltaArrType_Type_spec;
NPY_NO_EXPORT HPyType_Spec PyHalfArrType_Type_spec;


NPY_NO_EXPORT PyTypeObject *_PyGenericArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyBoolArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyNumberArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyIntegerArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PySignedIntegerArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyUnsignedIntegerArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyInexactArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyFloatingArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyComplexFloatingArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyFlexibleArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyCharacterArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyByteArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyShortArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyIntArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyLongArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyLongLongArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyUByteArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyUShortArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyUIntArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyULongArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyULongLongArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyFloatArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyDoubleArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyLongDoubleArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyCFloatArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyCDoubleArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyCLongDoubleArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyObjectArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyStringArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyUnicodeArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyVoidArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyTimeIntegerArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyDatetimeArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyTimedeltaArrType_Type_p;
NPY_NO_EXPORT PyTypeObject *_PyHalfArrType_Type_p;

extern NPY_NO_EXPORT HPyGlobal HPyGenericArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyGenericArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyBoolArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyNumberArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyIntegerArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPySignedIntegerArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyUnsignedIntegerArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyInexactArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyFloatingArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyComplexFloatingArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyFlexibleArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyCharacterArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyByteArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyShortArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyIntArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyLongArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyLongLongArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyUByteArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyUShortArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyUIntArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyULongArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyULongLongArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyFloatArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyDoubleArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyLongDoubleArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyCFloatArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyCDoubleArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyCLongDoubleArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyObjectArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyStringArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyUnicodeArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyVoidArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyDatetimeArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyTimedeltaArrType_Type;
extern NPY_NO_EXPORT HPyGlobal HPyHalfArrType_Type;

extern NPY_NO_EXPORT HPyGlobal _HPyArrayScalar_BoolValues[2];

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_SCALARTYPES_H_ */

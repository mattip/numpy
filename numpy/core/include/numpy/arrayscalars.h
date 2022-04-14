#ifndef NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_

#ifndef _MULTIARRAYMODULE
typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;
HPyType_LEGACY_HELPERS(PyBoolScalarObject);
#endif


typedef struct {
        PyObject_HEAD
        signed char obval;
} PyByteScalarObject;
HPyType_LEGACY_HELPERS(PyByteScalarObject);


typedef struct {
        PyObject_HEAD
        short obval;
} PyShortScalarObject;
HPyType_LEGACY_HELPERS(PyShortScalarObject);


typedef struct {
        PyObject_HEAD
        int obval;
} PyIntScalarObject;
HPyType_LEGACY_HELPERS(PyIntScalarObject);


typedef struct {
        PyObject_HEAD
        long obval;
} PyLongScalarObject;
HPyType_LEGACY_HELPERS(PyLongScalarObject);


typedef struct {
        PyObject_HEAD
        npy_longlong obval;
} PyLongLongScalarObject;
HPyType_LEGACY_HELPERS(PyLongLongScalarObject);


typedef struct {
        PyObject_HEAD
        unsigned char obval;
} PyUByteScalarObject;
HPyType_LEGACY_HELPERS(PyUByteScalarObject);


typedef struct {
        PyObject_HEAD
        unsigned short obval;
} PyUShortScalarObject;
HPyType_LEGACY_HELPERS(PyUShortScalarObject);


typedef struct {
        PyObject_HEAD
        unsigned int obval;
} PyUIntScalarObject;
HPyType_LEGACY_HELPERS(PyUIntScalarObject);


typedef struct {
        PyObject_HEAD
        unsigned long obval;
} PyULongScalarObject;
HPyType_LEGACY_HELPERS(PyULongScalarObject);


typedef struct {
        PyObject_HEAD
        npy_ulonglong obval;
} PyULongLongScalarObject;
HPyType_LEGACY_HELPERS(PyULongLongScalarObject);


typedef struct {
        PyObject_HEAD
        npy_half obval;
} PyHalfScalarObject;
HPyType_LEGACY_HELPERS(PyHalfScalarObject);


typedef struct {
        PyObject_HEAD
        float obval;
} PyFloatScalarObject;
HPyType_LEGACY_HELPERS(PyFloatScalarObject);


typedef struct {
        PyObject_HEAD
        double obval;
} PyDoubleScalarObject;
HPyType_LEGACY_HELPERS(PyDoubleScalarObject);


typedef struct {
        PyObject_HEAD
        npy_longdouble obval;
} PyLongDoubleScalarObject;
HPyType_LEGACY_HELPERS(PyLongDoubleScalarObject);


typedef struct {
        PyObject_HEAD
        npy_cfloat obval;
} PyCFloatScalarObject;
HPyType_LEGACY_HELPERS(PyCFloatScalarObject);


typedef struct {
        PyObject_HEAD
        npy_cdouble obval;
} PyCDoubleScalarObject;
HPyType_LEGACY_HELPERS(PyCDoubleScalarObject);


typedef struct {
        PyObject_HEAD
        npy_clongdouble obval;
} PyCLongDoubleScalarObject;
HPyType_LEGACY_HELPERS(PyCLongDoubleScalarObject);

typedef struct {
        PyObject_HEAD
        PyObject * obval;
} PyObjectScalarObject;
HPyType_LEGACY_HELPERS(PyObjectScalarObject);

typedef struct {
        PyObject_HEAD
        npy_datetime obval;
        PyArray_DatetimeMetaData obmeta;
} PyDatetimeScalarObject;
HPyType_LEGACY_HELPERS(PyDatetimeScalarObject);

typedef struct {
        PyObject_HEAD
        npy_timedelta obval;
        PyArray_DatetimeMetaData obmeta;
} PyTimedeltaScalarObject;
HPyType_LEGACY_HELPERS(PyTimedeltaScalarObject);


typedef struct {
        PyObject_HEAD
        char obval;
} PyScalarObject;
HPyType_LEGACY_HELPERS(PyScalarObject);

#define PyStringScalarObject PyBytesObject
typedef struct {
        /* note that the PyObject_HEAD macro lives right here */
        PyUnicodeObject base;
        Py_UCS4 *obval;
        char *buffer_fmt;
} PyUnicodeScalarObject;
HPyType_LEGACY_HELPERS(PyUnicodeScalarObject);


typedef struct {
        PyObject_VAR_HEAD
        char *obval;
        PyArray_Descr *descr;
        int flags;
        PyObject *base;
        void *_buffer_info;  /* private buffer info, tagged to allow warning */
} PyVoidScalarObject;
HPyType_LEGACY_HELPERS(PyVoidScalarObject);

/* Macros
     Py<Cls><bitsize>ScalarObject
     Py<Cls><bitsize>ArrType_Type
   are defined in ndarrayobject.h
*/

#define PyArrayScalar_False ((PyObject *)((_PyArrayScalar_BoolValues[0])))
#define PyArrayScalar_True ((PyObject *)((_PyArrayScalar_BoolValues[1])))
#define PyArrayScalar_FromLong(i) \
        ((PyObject *)((_PyArrayScalar_BoolValues[((i)!=0)])))
#define PyArrayScalar_RETURN_BOOL_FROM_LONG(i)                  \
        return Py_INCREF(PyArrayScalar_FromLong(i)), \
                PyArrayScalar_FromLong(i)
#define PyArrayScalar_RETURN_FALSE              \
        return Py_INCREF(PyArrayScalar_False),  \
                PyArrayScalar_False
#define PyArrayScalar_RETURN_TRUE               \
        return Py_INCREF(PyArrayScalar_True),   \
                PyArrayScalar_True

#define PyArrayScalar_New(cls) \
        Py##cls##ArrType_Type.tp_alloc(&Py##cls##ArrType_Type, 0)
#define PyArrayScalar_VAL(obj, cls)             \
        ((Py##cls##ScalarObject *)obj)->obval
#define PyArrayScalar_ASSIGN(obj, cls, val) \
        PyArrayScalar_VAL(obj, cls) = val

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_ */

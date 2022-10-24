/*
 * DON'T INCLUDE THIS DIRECTLY.
 */
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NDARRAYOBJECT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NDARRAYOBJECT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include "ndarraytypes.h"

/* Includes the "function" C-API -- these are all stored in a
   list of pointers --- one for each file
   The two lists are concatenated into one in multiarray.

   They are available as import_array()
*/

#include "__multiarray_api.h"


/* C-API that requires previous API to be defined */

#define PyArray_DescrCheck(op) PyObject_TypeCheck(op, &PyArrayDescr_Type)

static NPY_INLINE int
HPyArray_DescrCheck(HPyContext *ctx, HPy op)
{
    HPy array_descr_type = HPyGlobal_Load(ctx, HPyArrayDescr_Type);
    int res = HPy_TypeCheck(ctx, op, array_descr_type);
    HPy_Close(ctx, array_descr_type);
    return res;
}

#define PyArray_Check(op) PyObject_TypeCheck(op, &PyArray_Type)
#define PyArray_CheckExact(op) (((PyObject*)(op))->ob_type == &PyArray_Type)

static NPY_INLINE int
HPyArray_Check(HPyContext *ctx, HPy op)
{
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    int res = HPy_TypeCheck(ctx, op, array_type);
    HPy_Close(ctx, array_type);
    return res;
}

static NPY_INLINE int
HPyArray_CheckExactWithType(HPyContext *ctx, HPy op, HPy array_type) {
    HPy op_type = HPy_Type(ctx, op);
    int ret = HPy_Is(ctx, op_type, array_type);
    HPy_Close(ctx, op_type);
    return ret;
}

static NPY_INLINE int
HPyArray_CheckExact(HPyContext *ctx, HPy op)
{
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    int res = HPyArray_CheckExactWithType(ctx, op, array_type);
    HPy_Close(ctx, array_type);
    return res;
}

static NPY_INLINE int
HPyArray_IsPythonScalar(HPyContext *ctx, HPy op)
{
    return HPy_TypeCheck(ctx, op, ctx->h_FloatType) ||
           HPy_TypeCheck(ctx, op, ctx->h_ComplexType) ||
           HPy_TypeCheck(ctx, op, ctx->h_LongType) ||
           HPy_TypeCheck(ctx, op, ctx->h_BoolType) ||
           HPyBytes_Check(ctx, op) ||
           HPyUnicode_Check(ctx, op);
}

#define PyArray_HasArrayInterfaceType(op, type, context, out)                 \
        ((((out)=PyArray_FromStructInterface(op)) != Py_NotImplemented) ||    \
         (((out)=PyArray_FromInterface(op)) != Py_NotImplemented) ||          \
         (((out)=PyArray_FromArrayAttr(op, type, context)) !=                 \
          Py_NotImplemented))

#define PyArray_HasArrayInterface(op, out)                                    \
        PyArray_HasArrayInterfaceType(op, NULL, NULL, out)

#define PyArray_IsZeroDim(op) (PyArray_Check(op) && \
                               (PyArray_NDIM((PyArrayObject *)op) == 0))

#define HPyArray_IsZeroDim(ctx, op) (HPyArray_Check(ctx, op) && \
                               (HPyArray_GetNDim(ctx, op) == 0))

#define PyArray_IsScalar(obj, cls)                                            \
        (PyObject_TypeCheck(obj, &Py##cls##ArrType_Type))

#define HPyArray_IsScalar(ctx, obj, cls)                                      \
        (HPyGlobal_TypeCheck(ctx, obj, HPy##cls##ArrType_Type))

#define PyArray_CheckScalar(m) (PyArray_IsScalar(m, Generic) ||               \
                                PyArray_IsZeroDim(m))
#define PyArray_IsPythonNumber(obj)                                           \
        (PyFloat_Check(obj) || PyComplex_Check(obj) ||                        \
         PyLong_Check(obj) || PyBool_Check(obj))
#define PyArray_IsIntegerScalar(obj) (PyLong_Check(obj)                       \
              || PyArray_IsScalar((obj), Integer))
#define PyArray_IsPythonScalar(obj)                                           \
        (PyArray_IsPythonNumber(obj) || PyBytes_Check(obj) ||                 \
         PyUnicode_Check(obj))

#define PyArray_IsAnyScalar(obj)                                              \
        (PyArray_IsScalar(obj, Generic) || PyArray_IsPythonScalar(obj))

#define HPyArray_IsAnyScalar(ctx, obj)                                        \
        (HPyArray_IsScalar(ctx, obj, Generic) ||                              \
         HPyArray_IsPythonScalar(ctx, obj))

#define PyArray_CheckAnyScalar(obj) (PyArray_IsPythonScalar(obj) ||           \
                                     PyArray_CheckScalar(obj))


#define PyArray_GETCONTIGUOUS(m) (PyArray_ISCONTIGUOUS(m) ?                   \
                                  Py_INCREF(m), (m) :                         \
                                  (PyArrayObject *)(PyArray_Copy(m)))

#define PyArray_SAMESHAPE(a1,a2) ((PyArray_NDIM(a1) == PyArray_NDIM(a2)) &&   \
                                  PyArray_CompareLists(PyArray_DIMS(a1),      \
                                                       PyArray_DIMS(a2),      \
                                                       PyArray_NDIM(a1)))

#define PyArray_SIZE(m) PyArray_MultiplyList(PyArray_DIMS(m), PyArray_NDIM(m))
#define HPyArray_SIZE(m) PyArray_MultiplyList(PyArray_DIMS(m), PyArray_NDIM(m))
#define PyArray_NBYTES(m) (PyArray_ITEMSIZE(m) * PyArray_SIZE(m))
#define HPyArray_NBYTES(ctx, m, m_struct) (HPyArray_ITEMSIZE(ctx, m, m_struct) * PyArray_SIZE(m_struct))
#define PyArray_FROM_O(m) PyArray_FromAny(m, NULL, 0, 0, 0, NULL)
#define HPyArray_FROM_O(ctx, m) HPyArray_FromAny(ctx, m, HPy_NULL, 0, 0, 0, HPy_NULL)

#define PyArray_FROM_OF(m,flags) PyArray_CheckFromAny(m, NULL, 0, 0, flags,   \
                                                      NULL)
#define HPyArray_FROM_OF(ctx, m,flags) HPyArray_CheckFromAny(ctx, m, HPy_NULL, 0, 0, flags,   \
                                                      HPy_NULL)

#define PyArray_FROM_OT(m,type) PyArray_FromAny(m,                            \
                                PyArray_DescrFromType(type), 0, 0, 0, NULL)

#define PyArray_FROM_OTF(m, type, flags) \
        PyArray_FromAny(m, PyArray_DescrFromType(type), 0, 0, \
                        (((flags) & NPY_ARRAY_ENSURECOPY) ? \
                         ((flags) | NPY_ARRAY_DEFAULT) : (flags)), NULL)

// pass value of HPyArray_DescrFromType(ctx, type)
#define HPyArray_FROM_OTF(ctx, m, type_descr, flags) \
        HPyArray_FromAny(ctx, m, type_descr, 0, 0, \
                        (((flags) & NPY_ARRAY_ENSURECOPY) ? \
                         ((flags) | NPY_ARRAY_DEFAULT) : (flags)), HPy_NULL)

// pass value of HPyArray_DescrFromType(ctx, type)
#define HPyArray_FROMANY(ctx, m, type, min, max, flags) \
        HPyArray_FromAny(ctx, m, type_descr, min, max, \
                        (((flags) & NPY_ARRAY_ENSURECOPY) ? \
                         (flags) | NPY_ARRAY_DEFAULT : (flags)), HPy_NULL)

#define PyArray_FROMANY(m, type, min, max, flags) \
        PyArray_FromAny(m, PyArray_DescrFromType(type), min, max, \
                        (((flags) & NPY_ARRAY_ENSURECOPY) ? \
                         (flags) | NPY_ARRAY_DEFAULT : (flags)), NULL)

#define PyArray_ZEROS(m, dims, type, is_f_order) \
        PyArray_Zeros(m, dims, PyArray_DescrFromType(type), is_f_order)

#define PyArray_EMPTY(m, dims, type, is_f_order) \
        PyArray_Empty(m, dims, PyArray_DescrFromType(type), is_f_order)

#define PyArray_FILLWBYTE(obj, val) memset(PyArray_DATA(obj), val, \
                                           PyArray_NBYTES(obj))
#ifndef PYPY_VERSION
#define PyArray_REFCOUNT(obj) (((PyObject *)(obj))->ob_refcnt)
#define NPY_REFCOUNT PyArray_REFCOUNT
#endif
#define NPY_MAX_ELSIZE (2 * NPY_SIZEOF_LONGDOUBLE)

#define PyArray_ContiguousFromAny(op, type, min_depth, max_depth) \
        PyArray_FromAny(op, PyArray_DescrFromType(type), min_depth, \
                              max_depth, NPY_ARRAY_DEFAULT, NULL)

#define HPyArray_ContiguousFromAny(ctx, op, type, min_depth, max_depth) \
        HPyArray_FromAny(ctx, op, HPyArray_DescrFromType(ctx, type), min_depth, \
                              max_depth, NPY_ARRAY_DEFAULT, HPy_NULL)

#define PyArray_EquivArrTypes(a1, a2) \
        PyArray_EquivTypes(PyArray_DESCR(a1), PyArray_DESCR(a2))

#define PyArray_EquivByteorders(b1, b2) \
        (((b1) == (b2)) || (PyArray_ISNBO(b1) == PyArray_ISNBO(b2)))

#define PyArray_SimpleNew(nd, dims, typenum) \
        PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, NULL, 0, 0, NULL)

#define PyArray_SimpleNewFromData(nd, dims, typenum, data) \
        PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, \
                    data, 0, NPY_ARRAY_CARRAY, NULL)

#define PyArray_SimpleNewFromDescr(nd, dims, descr) \
        PyArray_NewFromDescr(&PyArray_Type, descr, nd, dims, \
                             NULL, NULL, 0, NULL)

#define HPyArray_SimpleNewFromDescr(ctx, array_type, nd, dims, descr) \
        HPyArray_NewFromDescr(ctx, array_type, descr, nd, dims, \
                             NULL, NULL, 0, HPy_NULL)

#define PyArray_ToScalar(data, arr) \
        PyArray_Scalar(data, PyArray_DESCR(arr), (PyObject *)arr)

#define HPyArray_ToScalar(ctx, data, arr, arr_struct) \
        HPyArray_Scalar(ctx, data, HPyArray_DESCR(ctx, arr, arr_struct), arr, arr_struct)


/* These might be faster without the dereferencing of obj
   going on inside -- of course an optimizing compiler should
   inline the constants inside a for loop making it a moot point
*/

#define PyArray_GETPTR1(obj, i) ((void *)(PyArray_BYTES(obj) + \
                                         (i)*PyArray_STRIDES(obj)[0]))

#define PyArray_GETPTR2(obj, i, j) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1]))

#define PyArray_GETPTR3(obj, i, j, k) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2]))

#define PyArray_GETPTR4(obj, i, j, k, l) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3]))

static NPY_INLINE void
PyArray_DiscardWritebackIfCopy(PyArrayObject *arr)
{
    if (arr) {
        HPyContext *ctx = npy_get_context();
        HPy h_arr = HPy_FromPyObject(ctx, (PyObject*)arr);
        HPy h_base = HPyArray_GetBase(ctx, h_arr);
        if (HPy_IsNull(h_base)) {
            HPy_Close(ctx, h_arr);
            return;
        }
        int flags = HPyArray_FLAGS(ctx, h_arr);
        if (flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            HPyArray_ENABLEFLAGS(ctx, h_base, NPY_ARRAY_WRITEABLE);
            HPyArray_SetBase(ctx, h_arr, HPy_NULL);
            HPyArray_CLEARFLAGS(ctx, h_arr, NPY_ARRAY_WRITEBACKIFCOPY);
        }
        HPy_Close(ctx, h_base);
        HPy_Close(ctx, h_arr);
    }
}

static NPY_INLINE void
HPyArray_DiscardWritebackIfCopy(HPyContext *ctx, HPy h_arr)
{
    if (!HPy_IsNull(h_arr)) {
        HPy h_base = HPyArray_GetBase(ctx, h_arr);
        if (HPy_IsNull(h_base)) {
            HPy_Close(ctx, h_arr);
            return;
        }
        int flags = HPyArray_FLAGS(ctx, h_arr);
        if (flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            HPyArray_ENABLEFLAGS(ctx, h_base, NPY_ARRAY_WRITEABLE);
            HPyArray_SetBase(ctx, h_arr, HPy_NULL);
            HPyArray_CLEARFLAGS(ctx, h_arr, NPY_ARRAY_WRITEBACKIFCOPY);
        }
        HPy_Close(ctx, h_base);
        HPy_Close(ctx, h_arr);
    }
}

#define PyArray_DESCR_REPLACE(descr) do { \
                PyArray_Descr *_new_; \
                _new_ = PyArray_DescrNew(descr); \
                Py_XDECREF(descr); \
                descr = _new_; \
        } while(0)

#define HPyArray_DESCR_REPLACE(ctx, descr) do { \
                HPy _new_; \
                _new_ = HPyArray_DescrNew(ctx, descr); \
                HPy_Close(ctx, descr); \
                descr = _new_; \
        } while(0)

/* Copy should always return contiguous array */
#define PyArray_Copy(obj) PyArray_NewCopy(obj, NPY_CORDER)

#define PyArray_FromObject(op, type, min_depth, max_depth) \
        PyArray_FromAny(op, PyArray_DescrFromType(type), min_depth, \
                              max_depth, NPY_ARRAY_BEHAVED | \
                                         NPY_ARRAY_ENSUREARRAY, NULL)

/*
 * Macro HPyArray_FromObject was turned into an inline function and was in
 * 'ctors.h'.
 */
static NPY_INLINE HPy
HPyArray_FromObject(
        HPyContext *ctx, HPy op, int type, int min_depth, int max_depth)
{
    HPy descr = HPyArray_DescrFromType(ctx, type);
    HPy res = HPyArray_FromAny(ctx, op, descr, min_depth, max_depth,
            NPY_ARRAY_BEHAVED | NPY_ARRAY_ENSUREARRAY, HPy_NULL);
    /*
     * HPyArray_FromAny does not steal reference to 'descr' like
     * PyArray_FromAny.
     */
    HPy_Close(ctx, descr);
    return res;
}

#define PyArray_ContiguousFromObject(op, type, min_depth, max_depth) \
        PyArray_FromAny(op, PyArray_DescrFromType(type), min_depth, \
                              max_depth, NPY_ARRAY_DEFAULT | \
                                         NPY_ARRAY_ENSUREARRAY, NULL)

#define PyArray_CopyFromObject(op, type, min_depth, max_depth) \
        PyArray_FromAny(op, PyArray_DescrFromType(type), min_depth, \
                        max_depth, NPY_ARRAY_ENSURECOPY | \
                                   NPY_ARRAY_DEFAULT | \
                                   NPY_ARRAY_ENSUREARRAY, NULL)

#define PyArray_Cast(mp, type_num)                                            \
        PyArray_CastToType(mp, PyArray_DescrFromType(type_num), 0)

#define PyArray_Take(ap, items, axis)                                         \
        PyArray_TakeFrom(ap, items, axis, NULL, NPY_RAISE)

#define PyArray_Put(ap, items, values)                                        \
        PyArray_PutTo(ap, items, values, NPY_RAISE)

/* Compatibility with old Numeric stuff -- don't use in new code */

#define PyArray_FromDimsAndData(nd, d, type, data)                            \
        PyArray_FromDimsAndDataAndDescr(nd, d, PyArray_DescrFromType(type),   \
                                        data)


/*
   Check to see if this key in the dictionary is the "title"
   entry of the tuple (i.e. a duplicate dictionary entry in the fields
   dict).
*/

static NPY_INLINE int
NPY_TITLE_KEY_check(PyObject *key, PyObject *value)
{
    PyObject *title;
    if (PyTuple_Size(value) != 3) {
        return 0;
    }
    title = PyTuple_GetItem(value, 2);
    if (key == title) {
        return 1;
    }
#ifdef PYPY_VERSION
    /*
     * On PyPy, dictionary keys do not always preserve object identity.
     * Fall back to comparison by value.
     */
    if (PyUnicode_Check(title) && PyUnicode_Check(key)) {
        return PyUnicode_Compare(title, key) == 0 ? 1 : 0;
    }
#endif
    return 0;
}

static NPY_INLINE int
HNPY_TITLE_KEY_check(HPyContext *ctx, HPy key, HPy value)
{
    HPy title;
    if (HPy_Length(ctx, value) != 3) {
        return 0;
    }
    title = HPy_GetItem_i(ctx, value, 2);
    if (HPy_Is(ctx, key, title)) {
        return 1;
    }
#ifdef PYPY_VERSION
    /*
     * On PyPy, dictionary keys do not always preserve object identity.
     * Fall back to comparison by value.
     */
    hpy_abort_not_implemented("PyPy unimplemeted path 'PyUnicode_Compare' missing");
//     if (PyUnicode_Check(title) && PyUnicode_Check(key)) {
//         return PyUnicode_Compare(title, key) == 0 ? 1 : 0;
//     }
#endif
    return 0;
}


/* Macro, for backward compat with "if NPY_TITLE_KEY(key, value) { ..." */
#define NPY_TITLE_KEY(key, value) (NPY_TITLE_KEY_check((key), (value)))
#define HNPY_TITLE_KEY(ctx, key, value) (HNPY_TITLE_KEY_check(ctx, (key), (value)))

#define DEPRECATE(msg) PyErr_WarnEx(PyExc_DeprecationWarning,msg,1)
#define HPY_DEPRECATE(ctx, msg) HPyErr_WarnEx(ctx,ctx->h_DeprecationWarning,msg,1)
#define DEPRECATE_FUTUREWARNING(msg) PyErr_WarnEx(PyExc_FutureWarning,msg,1)
#define HPY_DEPRECATE_FUTUREWARNING(ctx, msg) HPyErr_WarnEx(ctx, ctx->h_FutureWarning,msg,1)

#ifdef __cplusplus
}
#endif


#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NDARRAYOBJECT_H_ */

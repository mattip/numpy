#ifndef NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_

#include "hpy.h"
#include "array_coercion.h"

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj);

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescrAndBase(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base);

NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr_int(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base, int zeroed,
        int allow_emptystring);

NPY_NO_EXPORT HPy
HPyArray_NewFromDescr_int(
        HPyContext *ctx,
        HPy h_subtype, HPy h_descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, HPy h_obj, HPy h_base, int zeroed,
        int allow_emptystring);

NPY_NO_EXPORT HPy
HPyArray_NewFromDescr(
        HPyContext *ctx, HPy subtype, HPy descr,
        int nd, npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, HPy obj);

NPY_NO_EXPORT PyObject *
PyArray_NewLikeArrayWithShape(
        PyArrayObject *prototype, NPY_ORDER order,
        PyArray_Descr *dtype, int ndim, npy_intp const *dims, int subok);

NPY_NO_EXPORT PyObject *
PyArray_New(
        PyTypeObject *, int nd, npy_intp const *,
        int, npy_intp const*, void *, int, int, PyObject *);

NPY_NO_EXPORT PyObject *
_array_from_array_like(PyObject *op,
        PyArray_Descr *requested_dtype, npy_bool writeable, PyObject *context,
        int never_copy);

NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context);

NPY_NO_EXPORT HPy
HPyArray_FromAny(HPyContext *ctx, HPy h_op, HPy h_newtype, int min_depth,
                int max_depth, int flags, HPy h_context);

NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context);

NPY_NO_EXPORT PyObject *
PyArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int flags);

NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input);

NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *input);

NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr_int(
        PyObject *op, PyArray_Descr *descr, int never_copy);

NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode,
                      PyObject *context);

NPY_NO_EXPORT PyObject *
PyArray_EnsureArray(PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_EnsureAnyArray(PyObject *op);

NPY_NO_EXPORT int
PyArray_MoveInto(PyArrayObject *dest, PyArrayObject *src);

NPY_NO_EXPORT int
PyArray_CopyAnyInto(PyArrayObject *dest, PyArrayObject *src);

NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags);

NPY_NO_EXPORT HPy
HPyArray_CheckAxis(HPyContext *ctx, HPy h_arr, int *axis, int flags);

/* TODO: Put the order parameter in PyArray_CopyAnyInto and remove this */
NPY_NO_EXPORT int
PyArray_CopyAsFlat(PyArrayObject *dst, PyArrayObject *src,
                                NPY_ORDER order);

/* FIXME: remove those from here */
NPY_NO_EXPORT void
_array_fill_strides(npy_intp *strides, npy_intp const *dims, int nd, size_t itemsize,
                    int inflag, int *objflags);

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize);

NPY_NO_EXPORT void
_strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size);

NPY_NO_EXPORT void
copy_and_swap(void *dst, void *src, int itemsize, npy_intp numitems,
              npy_intp srcstrides, int swap);

NPY_NO_EXPORT void
byte_swap_vector(void *p, npy_intp n, int size);

/*
 * Calls arr_of_subclass.__array_wrap__(towrap), in order to make 'towrap'
 * have the same ndarray subclass as 'arr_of_subclass'.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_SubclassWrap(PyArrayObject *arr_of_subclass, PyArrayObject *towrap);


NPY_NO_EXPORT HPy
HPyArray_CheckFromAny(HPyContext *ctx, HPy op, HPy descr, int min_depth,
                     int max_depth, int requires, HPy context);

NPY_NO_EXPORT HPy
HPyArray_NewLikeArray(HPyContext *ctx, HPy prototype, NPY_ORDER order,
                     HPy dtype, int subok);

NPY_NO_EXPORT int
HPyArray_AssignFromCache(HPyContext *ctx, HPy self, coercion_cache_obj *cache);

/*
 * This function was originally a macro in 'ndarrayobject.h' but it needs the
 * declaration for 'HPyArray_FromAny'.
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

NPY_NO_EXPORT int
HPyArray_CopyInto(HPyContext *ctx, HPy dst, HPy src);

NPY_NO_EXPORT HPy
HPyArray_NewFromDescrAndBase(HPyContext *ctx,
        /*PyTypeObject*/ HPy subtype, /*PyArray_Descr*/ HPy descr,
        int nd, npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, HPy obj, HPy base);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CTORS_H_ */

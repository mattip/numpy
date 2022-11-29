#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "common.h"

#include "abstractdtypes.h"
#include "usertypes.h"

#include "npy_buffer.h"

#include "get_attr_string.h"
#include "mem_overlap.h"
#include "array_coercion.h"
#include "ctors.h"

/*
 * The casting to use for implicit assignment operations resulting from
 * in-place operations (like +=) and out= arguments. (Notice that this
 * variable is misnamed, but it's part of the public API so I'm not sure we
 * can just change it. Maybe someone should try and see if anyone notices.
 */
/*
 * In numpy 1.6 and earlier, this was NPY_UNSAFE_CASTING. In a future
 * release, it will become NPY_SAME_KIND_CASTING.  Right now, during the
 * transitional period, we continue to follow the NPY_UNSAFE_CASTING rules (to
 * avoid breaking people's code), but we also check for whether the cast would
 * be allowed under the NPY_SAME_KIND_CASTING rules, and if not we issue a
 * warning (that people's code will be broken in a future release.)
 */

NPY_NO_EXPORT NPY_CASTING NPY_DEFAULT_ASSIGN_CASTING = NPY_SAME_KIND_CASTING;


NPY_NO_EXPORT HPy /* (PyArray_Descr *) */
_array_find_python_scalar_type(HPyContext *ctx, HPy op)
{
    if (HPy_TypeCheck(ctx, op, ctx->h_FloatType)) {
        return HPyArray_DescrFromType(ctx, NPY_DOUBLE);
    }
    else if (HPy_TypeCheck(ctx, op, ctx->h_ComplexType)) {
        return HPyArray_DescrFromType(ctx, NPY_CDOUBLE);
    }
    else if (HPy_TypeCheck(ctx, op, ctx->h_LongType)) {
        // TODO HPY LABS PORT
        CAPI_WARN("_array_find_python_scalar_type: loading type from legacy global");
        HPy int_abstract_dtype = HPyGlobal_Load(ctx, HPyArray_PyIntAbstractDType);
        HPy res = HNPY_DT_CALL_discover_descr_from_pyobject(
                ctx, int_abstract_dtype, op);
        HPy_Close(ctx, int_abstract_dtype);
        return res;
    }
    return HPy_NULL;
}


/*
 * Get a suitable string dtype by calling `__str__`.
 * For `np.bytes_`, this assumes an ASCII encoding.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DTypeFromObjectStringDiscovery(
        PyObject *obj, PyArray_Descr *last_dtype, int string_type)
{
    int itemsize;

    if (string_type == NPY_STRING) {
        PyObject *temp = PyObject_Str(obj);
        if (temp == NULL) {
            return NULL;
        }
        /* assume that when we do the encoding elsewhere we'll use ASCII */
        itemsize = PyUnicode_GetLength(temp);
        Py_DECREF(temp);
        if (itemsize < 0) {
            return NULL;
        }
    }
    else if (string_type == NPY_UNICODE) {
        PyObject *temp = PyObject_Str(obj);
        if (temp == NULL) {
            return NULL;
        }
        itemsize = PyUnicode_GetLength(temp);
        Py_DECREF(temp);
        if (itemsize < 0) {
            return NULL;
        }
        itemsize *= 4;  /* convert UCS4 codepoints to bytes */
    }
    else {
        return NULL;
    }
    if (last_dtype != NULL &&
        last_dtype->type_num == string_type &&
        last_dtype->elsize >= itemsize) {
        Py_INCREF(last_dtype);
        return last_dtype;
    }
    PyArray_Descr *dtype = PyArray_DescrNewFromType(string_type);
    if (dtype == NULL) {
        return NULL;
    }
    dtype->elsize = itemsize;
    return dtype;
}


/*
 * This function is now identical to the new PyArray_DiscoverDTypeAndShape
 * but only returns the dtype. It should in most cases be slowly phased out.
 * (Which may need some refactoring to PyArray_FromAny to make it simpler)
 */
NPY_NO_EXPORT int
PyArray_DTypeFromObject(PyObject *obj, int maxdims, PyArray_Descr **out_dtype)
{
    coercion_cache_obj *cache = NULL;
    npy_intp shape[NPY_MAXDIMS];
    int ndim;

    ndim = PyArray_DiscoverDTypeAndShape(
            obj, maxdims, shape, &cache, NULL, NULL, out_dtype, 0);
    if (ndim < 0) {
        return -1;
    }
    npy_free_coercion_cache(cache);
    return 0;
}

NPY_NO_EXPORT int
HPyArray_DTypeFromObject(HPyContext *ctx, HPy obj, int maxdims, 
                            HPy /* PyArray_Descr ** */ *out_dtype) {
    coercion_cache_obj *cache = NULL;
    npy_intp shape[NPY_MAXDIMS];
    int ndim;

    ndim = HPyArray_DiscoverDTypeAndShape(ctx,
            obj, maxdims, shape, &cache, HPy_NULL, HPy_NULL, out_dtype, 0);
    if (ndim < 0) {
        return -1;
    }
    hnpy_free_coercion_cache(ctx, cache);
    return 0;
}

NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, npy_intp i)
{
    npy_intp dim0;

    if (PyArray_NDIM(mp) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed");
        return NULL;
    }
    dim0 = PyArray_DIMS(mp)[0];
    if (check_and_adjust_index(&i, dim0, 0, NULL) < 0)
        return NULL;
    if (i == 0) {
        return PyArray_DATA(mp);
    }
    return PyArray_BYTES(mp)+i*PyArray_STRIDES(mp)[0];
}

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret)
{
    if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
        PyObject *zero = PyLong_FromLong(0);
        PyArray_FillObjectArray(ret, zero);
        Py_DECREF(zero);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return -1;
        }
    }
    else {
        npy_intp n = PyArray_NBYTES(ret);
        memset(PyArray_DATA(ret), 0, n);
    }
    return 0;
}

NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap)
{
    PyObject *base = PyArray_BASE(ap);
    Py_buffer view;

    /*
     * C-data wrapping arrays may not own their data while not having a base;
     * WRITEBACKIFCOPY arrays have a base, but do own their data.
     */
    if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
        /*
         * This is somewhat unsafe for directly wrapped non-writable C-arrays,
         * which do not know whether the memory area is writable or not and
         * do not own their data (but have no base).
         * It would be better if this returned PyArray_ISWRITEABLE(ap).
         * Since it is hard to deprecate, this is deprecated only on the Python
         * side, but not on in PyArray_UpdateFlags.
         */
        return NPY_TRUE;
    }

    /*
     * Get to the final base object.
     * If it is a writeable array, then return True if we can
     * find an array object or a writeable buffer object as
     * the final base object.
     */
    while (PyArray_Check(base)) {
        ap = (PyArrayObject *)base;
        base = PyArray_BASE(ap);

        if (PyArray_ISWRITEABLE(ap)) {
            /*
             * If any base is writeable, it must be OK to switch, note that
             * bases are typically collapsed to always point to the most
             * general one.
             */
            return NPY_TRUE;
        }

        if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
            /* there is no further base to test the writeable flag for */
            return NPY_FALSE;
        }
        assert(!PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA));
    }

    if (PyObject_GetBuffer(base, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        PyErr_Clear();
        return NPY_FALSE;
    }
    PyBuffer_Release(&view);
    return NPY_TRUE;
}


/**
 * Convert an array shape to a string such as "(1, 2)".
 *
 * @param Dimensionality of the shape
 * @param npy_intp pointer to shape array
 * @param String to append after the shape `(1, 2)%s`.
 *
 * @return Python unicode string
 */
NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp const *vals, char *ending)
{
    npy_intp i;

    /*
     * Negative dimension indicates "newaxis", which can
     * be discarded for printing if it's a leading dimension.
     * Find the first non-"newaxis" dimension.
     */
    for (i = 0; i < n && vals[i] < 0; i++);

    if (i == n) {
        return PyUnicode_FromFormat("()%s", ending);
    }

    PyObject *ret = PyUnicode_FromFormat("%" NPY_INTP_FMT, vals[i++]);
    if (ret == NULL) {
        return NULL;
    }
    for (; i < n; ++i) {
        PyObject *tmp;

        if (vals[i] < 0) {
            tmp = PyUnicode_FromString(",newaxis");
        }
        else {
            tmp = PyUnicode_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        Py_SETREF(ret, PyUnicode_Concat(ret, tmp));
        Py_DECREF(tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    if (i == 1) {
        Py_SETREF(ret, PyUnicode_FromFormat("(%S,)%s", ret, ending));
    }
    else {
        Py_SETREF(ret, PyUnicode_FromFormat("(%S)%s", ret, ending));
    }
    return ret;
}


NPY_NO_EXPORT void
dot_alignment_error(PyArrayObject *a, int i, PyArrayObject *b, int j)
{
    PyObject *errmsg = NULL, *format = NULL, *fmt_args = NULL,
             *i_obj = NULL, *j_obj = NULL,
             *shape1 = NULL, *shape2 = NULL,
             *shape1_i = NULL, *shape2_j = NULL;

    format = PyUnicode_FromString("shapes %s and %s not aligned:"
                                  " %d (dim %d) != %d (dim %d)");

    shape1 = convert_shape_to_string(PyArray_NDIM(a), PyArray_DIMS(a), "");
    shape2 = convert_shape_to_string(PyArray_NDIM(b), PyArray_DIMS(b), "");

    i_obj = PyLong_FromLong(i);
    j_obj = PyLong_FromLong(j);

    shape1_i = PyLong_FromSsize_t(PyArray_DIM(a, i));
    shape2_j = PyLong_FromSsize_t(PyArray_DIM(b, j));

    if (!format || !shape1 || !shape2 || !i_obj || !j_obj ||
            !shape1_i || !shape2_j) {
        goto end;
    }

    fmt_args = PyTuple_Pack(6, shape1, shape2,
                            shape1_i, i_obj, shape2_j, j_obj);
    if (fmt_args == NULL) {
        goto end;
    }

    errmsg = PyUnicode_Format(format, fmt_args);
    if (errmsg != NULL) {
        PyErr_SetObject(PyExc_ValueError, errmsg);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "shapes are not aligned");
    }

end:
    Py_XDECREF(errmsg);
    Py_XDECREF(fmt_args);
    Py_XDECREF(format);
    Py_XDECREF(i_obj);
    Py_XDECREF(j_obj);
    Py_XDECREF(shape1);
    Py_XDECREF(shape2);
    Py_XDECREF(shape1_i);
    Py_XDECREF(shape2_j);
}

NPY_NO_EXPORT void
hpy_dot_alignment_error(HPyContext *ctx, PyArrayObject *a, int i, PyArrayObject *b, int j)
{
    PyObject *errmsg = NULL;
    HPy format = HPy_NULL, fmt_args = HPy_NULL,
        i_obj = HPy_NULL, j_obj = HPy_NULL,
        shape1 = HPy_NULL, shape2 = HPy_NULL,
        shape1_i = HPy_NULL, shape2_j = HPy_NULL;

    format = HPyUnicode_FromString(ctx, "shapes %s and %s not aligned:"
                                  " %d (dim %d) != %d (dim %d)");

    CAPI_WARN("missing PyUnicode_Format");    
    PyObject *py_shape1 = convert_shape_to_string(PyArray_NDIM(a), PyArray_DIMS(a), "");
    PyObject *py_shape2 = convert_shape_to_string(PyArray_NDIM(b), PyArray_DIMS(b), "");
    shape1 = HPy_FromPyObject(ctx, py_shape1);
    Py_DECREF(py_shape1);
    shape2 = HPy_FromPyObject(ctx, py_shape2);
    Py_DECREF(py_shape2);
    i_obj = HPyLong_FromLong(ctx, i);
    j_obj = HPyLong_FromLong(ctx, j);

    shape1_i = HPyLong_FromSsize_t(ctx, PyArray_DIM(a, i));
    shape2_j = HPyLong_FromSsize_t(ctx, PyArray_DIM(b, j));

    if (HPy_IsNull(format) || HPy_IsNull(shape1) || HPy_IsNull(shape2) ||
            HPy_IsNull(i_obj) || HPy_IsNull(j_obj) ||HPy_IsNull(shape1_i) ||
            HPy_IsNull(shape2_j)) {
        goto end;
    }

    fmt_args = HPyTuple_Pack(ctx, 6, shape1, shape2,
                            shape1_i, i_obj, shape2_j, j_obj);
    if (HPy_IsNull(fmt_args)) {
        goto end;
    }
    PyObject *py_format = HPy_AsPyObject(ctx, format);
    PyObject *py_fmt_args = HPy_AsPyObject(ctx, fmt_args);
    errmsg = PyUnicode_Format(py_format, py_fmt_args);
    Py_DECREF(py_format);
    Py_DECREF(py_fmt_args);
    if (errmsg != NULL) {
        HPy h_errmsg = HPy_FromPyObject(ctx, errmsg);
        Py_DECREF(errmsg);
        HPyErr_SetObject(ctx, ctx->h_ValueError, h_errmsg);
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError, "shapes are not aligned");
    }

end:
    Py_XDECREF(errmsg);
    HPy_Close(ctx, fmt_args);
    HPy_Close(ctx, format);
    HPy_Close(ctx, i_obj);
    HPy_Close(ctx, j_obj);
    HPy_Close(ctx, shape1);
    HPy_Close(ctx, shape2);
    HPy_Close(ctx, shape1_i);
    HPy_Close(ctx, shape2_j);
}


NPY_NO_EXPORT HPy
dummy_array_new(HPyContext *ctx, HPy /* (PyArray_Descr *) */ descr, npy_intp flags, HPy base)
{
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);

    // PyArrayObject_fields *fa = (PyArrayObject_fields *)PyArray_Type.tp_alloc(&PyArray_Type, 0);
    PyArrayObject_fields *data = NULL;
    HPy fa = HPy_New(ctx, array_type, &data);
    if (HPy_IsNull(fa)) {
        return HPy_NULL;
    }
    data->f_descr = HPyField_NULL;
    if (!HPy_IsNull(descr)) {
        _hpy_set_descr(ctx, fa, data, descr);
    }
    data->flags = flags;
    if (!HPy_IsNull(base)) {
        HPyArray_SetBase(ctx, fa, base);
    }
    return fa;
}

/*
 * Define a dummy array with only the information required by
 * dtype member functions such as descr->f->getitem:
 *   1. The descr, the main field interesting here.
 *   2. The flags, which are needed for alignment.
 *   3. The base is set to orig (or its base), which is used in the subarray
 *      case of VOID_getitem.
 *
 */
NPY_NO_EXPORT PyArrayObject *
get_tmp_array(PyArrayObject *orig)
{
    PyArray_Descr *dtype = PyArray_DESCR(orig);
    Py_INCREF(dtype);
    npy_intp shape = 1;
    PyObject *ret = PyArray_NewFromDescr_int(
            &PyArray_Type, dtype, 1,
            &shape, NULL, NULL,
            PyArray_FLAGS(orig), NULL, (PyObject *)orig, 0, 1);
    return (PyArrayObject *)ret;
}

/**
 * unpack tuple of dtype->fields (descr, offset, title[not-needed])
 *
 * @param "value" should be the tuple.
 *
 * @return "descr" will be set to the field's dtype
 * @return "offset" will be set to the field's offset
 *
 * returns -1 on failure, 0 on success.
 */
NPY_NO_EXPORT int
_unpack_field(PyObject *value, PyArray_Descr **descr, npy_intp *offset)
{
    PyObject * off;
    if (PyTuple_GET_SIZE(value) < 2) {
        return -1;
    }
    *descr = (PyArray_Descr *)PyTuple_GET_ITEM(value, 0);
    off  = PyTuple_GET_ITEM(value, 1);

    if (PyLong_Check(off)) {
        *offset = PyLong_AsSsize_t(off);
    }
    else {
        PyErr_SetString(PyExc_IndexError, "can't convert offset");
        return -1;
    }

    return 0;
}

NPY_NO_EXPORT int
_hunpack_field(HPyContext *ctx, 
                    HPy value, 
                    HPy *descr, // PyArray_Descr **
                    npy_intp *offset)
{
    HPy off;
    if (HPy_Length(ctx, value) < 2) {
        return -1;
    }
    *descr = HPy_GetItem_i(ctx, value, 0);
    off  = HPy_GetItem_i(ctx, value, 1);

    if (HPyLong_Check(ctx, off)) {
        *offset = HPyLong_AsSsize_t(ctx, off);
    }
    else {
        HPyErr_SetString(ctx, ctx->h_IndexError, "can't convert offset");
        HPy_Close(ctx, *descr);
        HPy_Close(ctx, off);
        return -1;
    }

    return 0;
}

/*
 * check whether arrays with datatype dtype might have object fields. This will
 * only happen for structured dtypes (which may have hidden objects even if the
 * HASOBJECT flag is false), object dtypes, or subarray dtypes whose base type
 * is either of these.
 */
NPY_NO_EXPORT int
_may_have_objects(PyArray_Descr *dtype)
{
    PyArray_Descr *base = dtype;
    if (PyDataType_HASSUBARRAY(dtype)) {
        base = dtype->subarray->base;
    }

    return (PyDataType_HASFIELDS(base) ||
            PyDataType_FLAGCHK(base, NPY_ITEM_HASOBJECT) );
}

/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 *
 * If `out` is non-NULL, memory overlap is checked with ap1 and ap2, and an
 * updateifcopy temporary array may be returned. If `result` is non-NULL, the
 * output array to be returned (`out` if non-NULL and the newly allocated array
 * otherwise) is incref'd and put to *result.
 */
NPY_NO_EXPORT PyArrayObject *
new_array_for_sum(PyArrayObject *ap1, PyArrayObject *ap2, PyArrayObject* out,
                  int nd, npy_intp dimensions[], int typenum, PyArrayObject **result)
{
    PyArrayObject *out_buf;

    if (out) {
        int d;

        /* verify that out is usable */
        if (PyArray_NDIM(out) != nd ||
            PyArray_TYPE(out) != typenum ||
            !PyArray_ISCARRAY(out)) {
            PyErr_SetString(PyExc_ValueError,
                "output array is not acceptable (must have the right datatype, "
                "number of dimensions, and be a C-Array)");
            return 0;
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != PyArray_DIM(out, d)) {
                PyErr_SetString(PyExc_ValueError,
                    "output array has wrong dimensions");
                return 0;
            }
        }

        /* check for memory overlap */
        if (!(solve_may_share_memory(out, ap1, 1) == 0 &&
              solve_may_share_memory(out, ap2, 1) == 0)) {
            /* allocate temporary output array */
            out_buf = (PyArrayObject *)PyArray_NewLikeArray(out, NPY_CORDER,
                                                            NULL, 0);
            if (out_buf == NULL) {
                return NULL;
            }

            /* set copy-back */
            Py_INCREF(out);
            if (PyArray_SetWritebackIfCopyBase(out_buf, out) < 0) {
                Py_DECREF(out);
                Py_DECREF(out_buf);
                return NULL;
            }
        }
        else {
            Py_INCREF(out);
            out_buf = out;
        }

        if (result) {
            Py_INCREF(out);
            *result = out;
        }

        return out_buf;
    }
    else {
        PyTypeObject *subtype;
        double prior1, prior2;
        /*
         * Need to choose an output array that can hold a sum
         * -- use priority to determine which subtype.
         */
        if (Py_TYPE(ap2) != Py_TYPE(ap1)) {
            prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
            prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
            subtype = (prior2 > prior1 ? Py_TYPE(ap2) : Py_TYPE(ap1));
        }
        else {
            prior1 = prior2 = 0.0;
            subtype = Py_TYPE(ap1);
        }

        out_buf = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
                                               typenum, NULL, NULL, 0, 0,
                                               (PyObject *)
                                               (prior2 > prior1 ? ap2 : ap1));

        if (out_buf != NULL && result) {
            Py_INCREF(out_buf);
            *result = out_buf;
        }

        return out_buf;
    }
}

/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 *
 * If `out` is non-NULL, memory overlap is checked with ap1 and ap2, and an
 * updateifcopy temporary array may be returned. If `result` is non-NULL, the
 * output array to be returned (`out` if non-NULL and the newly allocated array
 * otherwise) is incref'd and put to *result.
 */
NPY_NO_EXPORT HPy // PyArrayObject *
hpy_new_array_for_sum(HPyContext *ctx, 
                        HPy ap1, PyArrayObject *ap1_struct,
                        HPy ap2, PyArrayObject *ap2_struct,
                        HPy out, PyArrayObject* out_struct,
                        int nd, npy_intp dimensions[], int typenum, 
                        HPy /* PyArrayObject ** */ *result)
{
    HPy out_buf; // PyArrayObject *

    if (!HPy_IsNull(out)) {
        int d;

        /* verify that out is usable */
        if (PyArray_NDIM(out_struct) != nd ||
            PyArray_TYPE(out_struct) != typenum ||
            !PyArray_ISCARRAY(out_struct)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                "output array is not acceptable (must have the right datatype, "
                "number of dimensions, and be a C-Array)");
            return HPy_NULL;
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != PyArray_DIM(out_struct, d)) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                    "output array has wrong dimensions");
                return HPy_NULL;
            }
        }

        /* check for memory overlap */
        if (!(solve_may_share_memory(out_struct, ap1_struct, 1) == 0 &&
              solve_may_share_memory(out_struct, ap2_struct, 1) == 0)) {
            /* allocate temporary output array */
            out_buf = HPyArray_NewLikeArray(ctx, out, NPY_CORDER,
                                                            HPy_NULL, 0);
            if (HPy_IsNull(out_buf)) {
                return HPy_NULL;
            }
            PyArrayObject *out_buf_struct = PyArrayObject_AsStruct(ctx, out_buf);
            /* set copy-back */
            // Py_INCREF(out);
            if (HPyArray_SetWritebackIfCopyBase(ctx, out_buf, out_buf_struct, out, out_struct) < 0) {
                // HPy_Close(ctx, out);
                HPy_Close(ctx, out_buf);
                return HPy_NULL;
            }
        }
        else {
            // Py_INCREF(out);
            out_buf = HPy_Dup(ctx, out);
        }

        if (result) {
            // Py_INCREF(out);
            *result = HPy_Dup(ctx, out);
        }

        return out_buf;
    }
    else {
        HPy subtype;
        double prior1, prior2;
        /*
         * Need to choose an output array that can hold a sum
         * -- use priority to determine which subtype.
         */
        HPy ap1_type = HPy_Type(ctx, ap1);
        HPy ap2_type = HPy_Type(ctx, ap2);
        if (!HPy_Is(ctx, ap1_type, ap2_type)) {
            prior2 = HPyArray_GetPriority(ctx, ap2, 0.0);
            prior1 = HPyArray_GetPriority(ctx, ap1, 0.0);
            subtype = (prior2 > prior1 ? ap2_type : ap1_type);
        }
        else {
            prior1 = prior2 = 0.0;
            subtype = ap1_type;
        }

        out_buf = HPyArray_New(ctx, subtype, nd, dimensions,
                                               typenum, NULL, NULL, 0, 0,
                                               (prior2 > prior1 ? ap2 : ap1));

        if (!HPy_IsNull(out_buf) && result) {
            // Py_INCREF(out_buf);
            *result = HPy_Dup(ctx, out_buf);
        }

        return out_buf;
    }
}

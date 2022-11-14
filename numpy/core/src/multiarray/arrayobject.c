/*
  Provide multidimensional arrays as a basic object type in python.

  Based on Original Numeric implementation
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  with contributions from many Numeric Python developers 1995-2004

  Heavily modified in 2005 with inspiration from Numarray

  by

  Travis Oliphant,  oliphant@ee.byu.edu
  Brigham Young University


maintainer email:  oliphant.travis@ieee.org

  Numarray design (which provided guidance) by
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include "hpy.h"

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "common.h"

#include "number.h"
#include "usertypes.h"
#include "arraytypes.h"
#include "scalartypes.h"
#include "arrayobject.h"
#include "convert_datatype.h"
#include "conversion_utils.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "methods.h"
#include "descriptor.h"
#include "iterators.h"
#include "mapping.h"
#include "getset.h"
#include "sequence.h"
#include "npy_buffer.h"
#include "array_assign.h"
#include "alloc.h"
#include "mem_overlap.h"
#include "numpyos.h"
#include "strfuncs.h"

#include "binop_override.h"
#include "array_coercion.h"
#include "getset.h"

/*NUMPY_API
  Compute the size of an array (in number of items)
*/
NPY_NO_EXPORT npy_intp
PyArray_Size(PyObject *op)
{
    if (PyArray_Check(op)) {
        return PyArray_SIZE((PyArrayObject *)op);
    }
    else {
        return 0;
    }
}

/*HPY_NUMPY_API
  Compute the size of an array (in number of items)
*/
NPY_NO_EXPORT npy_intp
HPyArray_Size(HPyContext *ctx, HPy op)
{
    if (HPyArray_Check(ctx, op)) {
        PyArrayObject *op_data = PyArrayObject_AsStruct(ctx, op);
        return HPyArray_SIZE(op_data);
    }
    else {
        return 0;
    }
}

/*NUMPY_API */
NPY_NO_EXPORT int
PyArray_SetUpdateIfCopyBase(PyArrayObject *arr, PyArrayObject *base)
{
    /* 2021-Dec-15 1.23*/
    PyErr_SetString(PyExc_RuntimeError,
        "PyArray_SetUpdateIfCopyBase is disabled, use "
        "PyArray_SetWritebackIfCopyBase instead, and be sure to call "
        "PyArray_ResolveWritebackIfCopy before the array is deallocated, "
        "i.e. before the last call to Py_DECREF. If cleaning up from an "
        "error, PyArray_DiscardWritebackIfCopy may be called instead to "
        "throw away the scratch buffer.");
    return -1;
}

/*NUMPY_API
 *
 * Precondition: 'arr' is a copy of 'base' (though possibly with different
 * strides, ordering, etc.). This function sets the WRITEBACKIFCOPY flag and the
 * ->base pointer on 'arr', call PyArray_ResolveWritebackIfCopy to copy any
 * changes back to 'base' before deallocating the array.
 *
 * Steals a reference to 'base'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_SetWritebackIfCopyBase(PyArrayObject *arr, PyArrayObject *base)
{
    if (base == NULL) {
        PyErr_SetString(PyExc_ValueError,
                  "Cannot WRITEBACKIFCOPY to NULL array");
        return -1;
    }
    if (PyArray_BASE(arr) != NULL) {
        PyErr_SetString(PyExc_ValueError,
                  "Cannot set array with existing base to WRITEBACKIFCOPY");
        goto fail;
    }
    if (PyArray_FailUnlessWriteable(base, "WRITEBACKIFCOPY base") < 0) {
        goto fail;
    }

    /*
     * Any writes to 'arr' will magically turn into writes to 'base', so we
     * should warn if necessary.
     */
    if (PyArray_FLAGS(base) & NPY_ARRAY_WARN_ON_WRITE) {
        PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WARN_ON_WRITE);
    }

    /*
     * Unlike PyArray_SetBaseObject, we do not compress the chain of base
     * references.
     */
    HPyContext *ctx = npy_get_context();
    HPy h_arr = HPy_FromPyObject(ctx, (PyObject*)arr);
    HPy h_base = HPy_FromPyObject(ctx, (PyObject*)base);
    HPyArray_SetBase(ctx, h_arr, h_base);
    HPy_Close(ctx, h_base);
    HPy_Close(ctx, h_arr);
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WRITEBACKIFCOPY);
    PyArray_CLEARFLAGS(base, NPY_ARRAY_WRITEABLE);

    Py_DECREF(base);
    return 0;

  fail:
    Py_DECREF(base);
    return -1;
}

/*HPY_NUMPY_API
 *
 * Precondition: 'arr' is a copy of 'base' (though possibly with different
 * strides, ordering, etc.). This function sets the WRITEBACKIFCOPY flag and the
 * ->base pointer on 'arr', call PyArray_ResolveWritebackIfCopy to copy any
 * changes back to 'base' before deallocating the array.
 *
 * ATTENTION: does not steal a reference to 'base'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
HPyArray_SetWritebackIfCopyBase(HPyContext *ctx, 
                                    HPy /* PyArrayObject * */ arr, 
                                    PyArrayObject *arr_data, 
                                    HPy /* PyArrayObject * */ base,
                                    PyArrayObject *base_data)
{
    if (HPy_IsNull(base)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                  "Cannot WRITEBACKIFCOPY to NULL array");
        return -1;
    }
    if (!HPy_IsNull(HPyArray_BASE(ctx, arr, arr_data))) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                  "Cannot set array with existing base to WRITEBACKIFCOPY");
        goto fail;
    }
    if (HPyArray_FailUnlessWriteableWithStruct(ctx, base, base_data, "WRITEBACKIFCOPY base") < 0) {
        goto fail;
    }

    /*
     * Any writes to 'arr' will magically turn into writes to 'base', so we
     * should warn if necessary.
     */
    if (PyArray_FLAGS(base_data) & NPY_ARRAY_WARN_ON_WRITE) {
        PyArray_ENABLEFLAGS(arr_data, NPY_ARRAY_WARN_ON_WRITE);
    }

    /*
     * Unlike PyArray_SetBaseObject, we do not compress the chain of base
     * references.
     */
    HPyArray_SetBase(ctx, arr, base);
    PyArray_ENABLEFLAGS(arr_data, NPY_ARRAY_WRITEBACKIFCOPY);
    PyArray_CLEARFLAGS(base_data, NPY_ARRAY_WRITEABLE);

    // Py_DECREF(base);
    return 0;

  fail:
    // Py_DECREF(base);
    return -1;
}

/*NUMPY_API
 * Sets the 'base' attribute of the array. This steals a reference
 * to 'obj'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_SetBaseObject(PyArrayObject *arr, PyObject *obj)
{
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency to NULL after initialization");
        return -1;
    }
    /*
     * Allow the base to be set only once. Once the object which
     * owns the data is set, it doesn't make sense to change it.
     */
    if (PyArray_BASE(arr) != NULL) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency more than once");
        return -1;
    }

    /*
     * Don't allow infinite chains of views, always set the base
     * to the first owner of the data.
     * That is, either the first object which isn't an array,
     * or the first object which owns its own data.
     */

    while (PyArray_Check(obj) && (PyObject *)arr != obj) {
        PyArrayObject *obj_arr = (PyArrayObject *)obj;
        PyObject *tmp;

        /* Propagate WARN_ON_WRITE through views. */
        if (PyArray_FLAGS(obj_arr) & NPY_ARRAY_WARN_ON_WRITE) {
            PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WARN_ON_WRITE);
        }

        /* If this array owns its own data, stop collapsing */
        if (PyArray_CHKFLAGS(obj_arr, NPY_ARRAY_OWNDATA)) {
            break;
        }

        tmp = PyArray_BASE(obj_arr);
        /* If there's no base, stop collapsing */
        if (tmp == NULL) {
            break;
        }
        /* Stop the collapse new base when the would not be of the same
         * type (i.e. different subclass).
         */
        if (Py_TYPE(tmp) != Py_TYPE(arr)) {
            break;
        }


        Py_INCREF(tmp);
        Py_DECREF(obj);
        obj = tmp;
    }

    /* Disallow circular references */
    if ((PyObject *)arr == obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError,
                "Cannot create a circular NumPy array 'base' dependency");
        return -1;
    }

    HPyContext *ctx = npy_get_context();
    HPy h_arr = HPy_FromPyObject(ctx, (PyObject*)arr);
    HPy h_base = HPy_FromPyObject(ctx, obj);
    HPyArray_SetBase(ctx, h_arr, h_base);
    HPy_Close(ctx, h_base);
    HPy_Close(ctx, h_arr);
    Py_DECREF(obj);

    return 0;
}

/*HPY_NUMPY_API
 *    ATTENTION: does not steal obj anymore
 */
NPY_NO_EXPORT int
HPyArray_SetBaseObject(HPyContext *ctx, HPy h_arr, PyArrayObject *arr, HPy obj_in)
{
    if (HPy_IsNull(obj_in)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency to NULL after initialization");
        return -1;
    }
    /*
     * Allow the base to be set only once. Once the object which
     * owns the data is set, it doesn't make sense to change it.
     */
    if (!HPy_IsNull(HPyArray_BASE(ctx, h_arr, arr))) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency more than once");
        return -1;
    }

    /*
     * Don't allow infinite chains of views, always set the base
     * to the first owner of the data.
     * That is, either the first object which isn't an array,
     * or the first object which owns its own data.
     */
    HPy obj = HPy_Dup(ctx, obj_in);
    while (HPyArray_Check(ctx, obj) && !HPy_Is(ctx, h_arr, obj)) {
        PyArrayObject *obj_arr = PyArrayObject_AsStruct(ctx, obj);

        /* Propagate WARN_ON_WRITE through views. */
        if (PyArray_FLAGS(obj_arr) & NPY_ARRAY_WARN_ON_WRITE) {
            PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WARN_ON_WRITE);
        }

        /* If this array owns its own data, stop collapsing */
        if (PyArray_CHKFLAGS(obj_arr, NPY_ARRAY_OWNDATA)) {
            break;
        }

        HPy tmp = HPyArray_BASE(ctx, obj, obj_arr);
        /* If there's no base, stop collapsing */
        if (HPy_IsNull(tmp)) {
            break;
        }
        /* Stop the collapse new base when the would not be of the same
         * type (i.e. different subclass).
         */
        HPy tmp_type = HPy_Type(ctx, tmp);
        HPy h_arr_type = HPy_Type(ctx, h_arr);
        int is_types_equal = HPy_Is(ctx, tmp_type, h_arr_type);
        HPy_Close(ctx, tmp_type);
        HPy_Close(ctx, h_arr_type);
        if (!is_types_equal) {
            HPy_Close(ctx, tmp);
            break;
        }

        HPy_Close(ctx, obj);
        obj = tmp;
    }

    /* Disallow circular references */
    if (HPy_Is(ctx, h_arr, obj)) {
        HPy_Close(ctx, obj);
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Cannot create a circular NumPy array 'base' dependency");
        return -1;
    }

    HPyArray_SetBase(ctx, h_arr, obj);
    // HPy_Close(ctx, obj); // There is an extra DECREF somewhere in the C API

    return 0;
}


/**
 * Assign an arbitrary object a NumPy array. This is largely basically
 * identical to PyArray_FromAny, but assigns directly to the output array.
 *
 * @param dest Array to be written to
 * @param src_object Object to be assigned, array-coercion rules apply.
 * @return 0 on success -1 on failures.
 */
/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_CopyObject(PyArrayObject *dest, PyObject *src_object)
{
    int ret = 0;
    PyArrayObject *view;
    PyArray_Descr *dtype = NULL;
    int ndim;
    npy_intp dims[NPY_MAXDIMS];
    coercion_cache_obj *cache = NULL;

    /*
     * We have to set the maximum number of dimensions here to support
     * sequences within object arrays.
     */
    ndim = PyArray_DiscoverDTypeAndShape(src_object,
            PyArray_NDIM(dest), dims, &cache,
            NPY_DTYPE(PyArray_DESCR(dest)), PyArray_DESCR(dest), &dtype, 0);
    if (ndim < 0) {
        return -1;
    }

    if (cache != NULL && !(cache->sequence)) {
        /* The input is an array or array object, so assign directly */
        CAPI_WARN("PyArray_AssignArray");
        HPyContext *ctx = npy_get_context();
        PyObject *tmp = HPy_AsPyObject(ctx, cache->converted_obj);
        assert(tmp == src_object);
        Py_DECREF(tmp);
        view = (PyArrayObject *)HPy_AsPyObject(ctx, cache->arr_or_sequence);
        Py_DECREF(dtype);
        ret = PyArray_AssignArray(dest, view, NULL, NPY_UNSAFE_CASTING);
        Py_DECREF(view);
        npy_free_coercion_cache(cache);
        return ret;
    }

    /*
     * We may need to broadcast, due to shape mismatches, in this case
     * create a temporary array first, and assign that after filling
     * it from the sequences/scalar.
     */
    if (ndim != PyArray_NDIM(dest) ||
            !PyArray_CompareLists(PyArray_DIMS(dest), dims, ndim)) {
        /*
         * Broadcasting may be necessary, so assign to a view first.
         * This branch could lead to a shape mismatch error later.
         */
        assert (ndim <= PyArray_NDIM(dest));  /* would error during discovery */
        view = (PyArrayObject *) PyArray_NewFromDescr(
                &PyArray_Type, dtype, ndim, dims, NULL, NULL,
                PyArray_FLAGS(dest) & NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (view == NULL) {
            npy_free_coercion_cache(cache);
            return -1;
        }
    }
    else {
        Py_DECREF(dtype);
        view = dest;
    }

    /* Assign the values to `view` (whichever array that is) */
    if (cache == NULL) {
        /* single (non-array) item, assign immediately */
        if (PyArray_Pack(
                PyArray_DESCR(view), PyArray_DATA(view), src_object) < 0) {
            goto fail;
        }
    }
    else {
        if (PyArray_AssignFromCache(view, cache) < 0) {
            goto fail;
        }
    }
    if (view == dest) {
        return 0;
    }
    ret = PyArray_AssignArray(dest, view, NULL, NPY_UNSAFE_CASTING);
    Py_DECREF(view);
    return ret;

  fail:
    if (view != dest) {
        Py_DECREF(view);
    }
    return -1;
}

/*HPY_NUMPY_API
    ATTENTION: does not steal obj anymore
 */
NPY_NO_EXPORT int
HPyArray_CopyObject(HPyContext *ctx, HPy h_dest, PyArrayObject *dest, HPy h_src_object)
{
    int ret = 0;
    HPy h_view;
    PyArrayObject *view;
    int ndim;
    npy_intp dims[NPY_MAXDIMS];
    coercion_cache_obj *cache = NULL;

    /*
     * We have to set the maximum number of dimensions here to support
     * sequences within object arrays.
     */
    HPy h_dest_descr = HPyArray_DESCR(ctx, h_dest, dest);
    HPy h_dest_descr_dtype = HNPY_DTYPE(ctx, h_dest_descr);
    HPy h_dtype = HPy_NULL;
    ndim = HPyArray_DiscoverDTypeAndShape(ctx, h_src_object,
            PyArray_NDIM(dest), dims, &cache,
            h_dest_descr_dtype, h_dest_descr, &h_dtype, 0);
    HPy_Close(ctx, h_dest_descr_dtype);
    HPy_Close(ctx, h_dest_descr);
    if (ndim < 0) {
        return -1;
    }

    if (cache != NULL && !(cache->sequence)) {
        /* The input is an array or array object, so assign directly */
        ret = HPyArray_AssignArray(ctx, h_dest, cache->arr_or_sequence, HPy_NULL, NPY_UNSAFE_CASTING);
        HPy_Close(ctx, cache->converted_obj);
        hnpy_free_coercion_cache(ctx, cache);
        return ret;
    }

    /*
     * We may need to broadcast, due to shape mismatches, in this case
     * create a temporary array first, and assign that after filling
     * it from the sequences/scalar.
     */
    if (ndim != PyArray_NDIM(dest) ||
            !PyArray_CompareLists(PyArray_DIMS(dest), dims, ndim)) {
        /*
         * Broadcasting may be necessary, so assign to a view first.
         * This branch could lead to a shape mismatch error later.
         */
        assert (ndim <= PyArray_NDIM(dest));  /* would error during discovery */
        HPy hpy_arr_type = HPyGlobal_Load(ctx, HPyArray_Type);
        h_view = HPyArray_NewFromDescr(
                ctx, hpy_arr_type, h_dtype, ndim, dims, NULL, NULL,
                PyArray_FLAGS(dest) & NPY_ARRAY_F_CONTIGUOUS, HPy_NULL);
        HPy_Close(ctx, hpy_arr_type);
        if (HPy_IsNull(h_view)) {
            HPy_Close(ctx, h_dtype);
            npy_free_coercion_cache(cache);
            return -1;
        }
        view = PyArrayObject_AsStruct(ctx, h_view);
    }
    else {
        h_view = HPy_Dup(ctx, h_dest);
        view = dest;
    }
    HPy_Close(ctx, h_dtype);

    /* Assign the values to `view` (whichever array that is) */
    if (cache == NULL) {
        /* single (non-array) item, assign immediately */
        HPy h_view_descr = HPyArray_DESCR(ctx, h_view, view);
        if (HPyArray_Pack(
                ctx, h_view_descr, PyArray_DATA(view), h_src_object) < 0) {
            HPy_Close(ctx, h_view_descr);
            goto fail;
        }
        HPy_Close(ctx, h_view_descr);
    }
    else {
        if (HPyArray_AssignFromCache(ctx, h_view, cache) < 0) {
            goto fail;
        }
    }
    if (HPy_Is(ctx, h_view, h_dest)) {
        return 0;
    }
    ret = HPyArray_AssignArray(ctx, h_dest, h_view, HPy_NULL, NPY_UNSAFE_CASTING);
    HPy_Close(ctx, h_view);
    return ret;

  fail:
    if (view != dest) {
        HPy_Close(ctx, h_view);
    }
    return -1;
}


/* returns an Array-Scalar Object of the type of arr
   from the given pointer to memory -- main Scalar creation function
   default new method calls this.
*/

/* Ideally, here the descriptor would contain all the information needed.
   So, that we simply need the data and the descriptor, and perhaps
   a flag
*/


/*
  Given a string return the type-number for
  the data-type with that string as the type-object name.
  Returns NPY_NOTYPE without setting an error if no type can be
  found.  Only works for user-defined data-types.
*/

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_TypeNumFromName(char const *str)
{
    int i;
    PyArray_Descr *descr;

    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr = userdescrs[i];
        if (strcmp(PyArray_Descr_typeobj(descr)->tp_name, str) == 0) {
            return descr->type_num;
        }
    }
    return NPY_NOTYPE;
}

/*NUMPY_API
 *
 * If WRITEBACKIFCOPY and self has data, reset the base WRITEABLE flag,
 * copy the local data to base, release the local data, and set flags
 * appropriately. Return 0 if not relevant, 1 if success, < 0 on failure
 */
NPY_NO_EXPORT int
PyArray_ResolveWritebackIfCopy(PyArrayObject * self)
{
    if (self == NULL) {
        return 0;
    }
    HPyContext *ctx = npy_get_context();
    HPy h_self = HPy_FromPyObject(ctx, (PyObject *)self);
    int res = HPyArray_ResolveWritebackIfCopy(ctx, h_self);
    HPy_Close(ctx, h_self);
    return res;
}

/*HPY_NUMPY_API
 */
NPY_NO_EXPORT int
HPyArray_ResolveWritebackIfCopy(HPyContext *ctx, HPy self)
{
    if (HPy_IsNull(self)) {
        return 0;
    }
    HPy h_base = HPyArray_GetBase(ctx, self);

    if (!HPy_IsNull(h_base)) {

        int flags = HPyArray_FLAGS(ctx, self);
        if (flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            /*
             * WRITEBACKIFCOPY means that base's data
             * should be updated with the contents
             * of self.
             * base->flags is not WRITEABLE to protect the relationship
             * unlock it.
             */
            int retval = 0;
            HPyArray_ENABLEFLAGS(ctx, h_base, NPY_ARRAY_WRITEABLE);
            HPyArray_CLEARFLAGS(ctx, self, NPY_ARRAY_WRITEBACKIFCOPY);
            CAPI_WARN("HPyArray_ResolveWritebackIfCopy");
            PyArrayObject *base = (PyArrayObject *)HPy_AsPyObject(ctx, h_base);
            PyArrayObject *py_self = (PyArrayObject *)HPy_AsPyObject(ctx, self);
            HPy_Close(ctx, h_base);
            retval = PyArray_CopyAnyInto(base, py_self);
            Py_DECREF(py_self);
            Py_DECREF(base);
            HPyArray_SetBase(ctx, self, HPy_NULL);

            if (retval < 0) {
                /* this should never happen, how did the two copies of data
                 * get out of sync?
                 */
                return retval;
            }
            return 1;
        }
    }
    return 0;
}

/*********************** end C-API functions **********************/


/* dealloc must not raise an error, best effort try to write
   to stderr and clear the error
*/

static NPY_INLINE void
WARN_IN_DEALLOC(PyObject* warning, const char * msg) {
    if (PyErr_WarnEx(warning, msg, 1) < 0) {
        PyObject * s;

        s = PyUnicode_FromString("array_finalize");
        if (s) {
            PyErr_WriteUnraisable(s);
            Py_DECREF(s);
        }
        else {
            PyErr_WriteUnraisable(Py_None);
        }
    }
}

/* array object functions */
HPyDef_SLOT(array_finalize, array_finalize_impl, HPy_tp_finalize)
static void
array_finalize_impl(HPyContext *ctx, HPy h_self)
{
    // TODO HPY LABS PORT: err fetch and restore?
    // PyObject *error_type, *error_value, *error_traceback;
    // PyErr_Fetch(&error_type, &error_value, &error_traceback);

    PyArrayObject *self = PyArrayObject_AsStruct(ctx, h_self);
    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;

    if (fa->weakreflist != NULL) {
        // HACK: subtype_dealloc doesn't clear weakrefs, so we do it here,
        // but PyObject_ClearWeakRefs() checks that ob_refcnt == 0 ...
        CAPI_WARN("array finalize: fa->weakreflist != NULL");
        PyObject *obj = (PyObject*)self;
        // 2 refs: h_self and self
        obj->ob_refcnt -= 2;
        PyObject_ClearWeakRefs(obj);
        obj->ob_refcnt += 2;
    }

    // if (_buffer_info_free(fa->_buffer_info, (PyObject *)self) < 0) {
    //     PyErr_WriteUnraisable(NULL);
    // }

    if (HPy_IsNull(HPyArray_BASE(ctx, h_self, self))) {
        int retval;
        if (PyArray_FLAGS(self) & NPY_ARRAY_WRITEBACKIFCOPY)
        {
            CAPI_WARN("NPY_ARRAY_WRITEBACKIFCOPY");
            char const * msg = "WRITEBACKIFCOPY detected in array_finalize. "
                " Required call to PyArray_ResolveWritebackIfCopy or "
                "PyArray_DiscardWritebackIfCopy is missing.";
            /*
             * prevent reaching 0 twice and thus recursing into dealloc.
             * Increasing sys.gettotalrefcount, but path should not be taken.
             */
            Py_INCREF(self);
            WARN_IN_DEALLOC(PyExc_RuntimeWarning, msg);
            retval = PyArray_ResolveWritebackIfCopy(self);
            if (retval < 0)
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        /*
         * If fa->base is non-NULL, it is something
         * to DECREF -- either a view or a buffer object
         */
    }

    if ((fa->flags & NPY_ARRAY_OWNDATA) && fa->data) {
        /* Free internal references if an Object array */
        HPy h_descr = HPyArray_DESCR(ctx, h_self, self);
        PyArray_Descr *descr = PyArray_Descr_AsStruct(ctx, h_descr);
        if (PyDataType_FLAGCHK(descr, NPY_ITEM_REFCOUNT)) {
            CAPI_WARN("NPY_ITEM_REFCOUNT flag");
            PyArray_XDECREF(self);
        }
        HPy h_mem_handler = HPyArray_GetHandler(ctx, h_self);
        if (HPy_IsNull(h_mem_handler)) {
            char *env = getenv("NUMPY_WARN_IF_NO_MEM_POLICY");
            if ((env != NULL) && (strncmp(env, "1", 1) == 0)) {
                char const * msg = "Trying to dealloc data, but a memory policy "
                    "is not set. If you take ownership of the data, you must "
                    "set a base owning the data (e.g. a PyCapsule).";
                WARN_IN_DEALLOC(PyExc_RuntimeWarning, msg);
            }
            // Guess at malloc/free ???
            free(fa->data);
        }
        else {
            /*
             * In theory `PyArray_NBYTES_ALLOCATED`, but differs somewhere?
             * So instead just use the knowledge that 0 is impossible.
             */
            size_t nbytes = HPyArray_NBYTES(ctx, h_self, self);
            if (nbytes == 0) {
                nbytes = 1;
            }
            HPyDataMem_UserFREE(ctx, fa->data, nbytes, h_mem_handler);
        }
    }

    // TODO HPY LABS PORT: moved to array_destroy
    /* must match allocation in PyArray_NewFromDescr */
    // npy_free_cache_dim(fa->dimensions, 2 * fa->nd);

    // PyErr_Restore(error_type, error_value, error_traceback);
}

/* array object functions */
HPyDef_SLOT(array_destroy, array_destroy_impl, HPy_tp_destroy)
static void
array_destroy_impl(void *data)
{
    PyArrayObject_fields *fa = (PyArrayObject_fields *)data;

    /* must match allocation in PyArray_NewFromDescr */
    npy_free_cache_dim(fa->dimensions, 2 * fa->nd);
}

HPyDef_SLOT(array_traverse, array_traverse_impl, HPy_tp_traverse)
static int
array_traverse_impl(void *self, HPyFunc_visitproc visit, void *arg)
{
    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;
    HPy_VISIT(&fa->f_descr);
    HPy_VISIT(&fa->f_base);
    HPy_VISIT(&fa->f_mem_handler);
    return 0;
}

/*NUMPY_API
 * Prints the raw data of the ndarray in a form useful for debugging
 * low-level C issues.
 */
NPY_NO_EXPORT void
PyArray_DebugPrint(PyArrayObject *obj)
{
    int i;
    PyArrayObject_fields *fobj = (PyArrayObject_fields *)obj;

    printf("-------------------------------------------------------\n");
    printf(" Dump of NumPy ndarray at address %p\n", obj);
    if (obj == NULL) {
        printf(" It's NULL!\n");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
        return;
    }
    printf(" ndim   : %d\n", fobj->nd);
    printf(" shape  :");
    for (i = 0; i < fobj->nd; ++i) {
        printf(" %" NPY_INTP_FMT, fobj->dimensions[i]);
    }
    printf("\n");

    printf(" dtype  : ");
    PyObject_Print((PyObject *)PyArray_DESCR(obj), stdout, 0);
    printf("\n");
    printf(" data   : %p\n", fobj->data);
    printf(" strides:");
    for (i = 0; i < fobj->nd; ++i) {
        printf(" %" NPY_INTP_FMT, fobj->strides[i]);
    }
    printf("\n");

    PyObject *base = PyArray_BASE(obj);

    printf(" base   : %p\n", base);

    printf(" flags :");
    if (fobj->flags & NPY_ARRAY_C_CONTIGUOUS)
        printf(" NPY_C_CONTIGUOUS");
    if (fobj->flags & NPY_ARRAY_F_CONTIGUOUS)
        printf(" NPY_F_CONTIGUOUS");
    if (fobj->flags & NPY_ARRAY_OWNDATA)
        printf(" NPY_OWNDATA");
    if (fobj->flags & NPY_ARRAY_ALIGNED)
        printf(" NPY_ALIGNED");
    if (fobj->flags & NPY_ARRAY_WRITEABLE)
        printf(" NPY_WRITEABLE");
    if (fobj->flags & NPY_ARRAY_WRITEBACKIFCOPY)
        printf(" NPY_WRITEBACKIFCOPY");
    printf("\n");

    if (base != NULL && PyArray_Check(base)) {
        printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
        printf("Dump of array's BASE:\n");
        PyArray_DebugPrint((PyArrayObject *)base);
        printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    }
    printf("-------------------------------------------------------\n");
    fflush(stdout);
}


/*NUMPY_API
 * This function is scheduled to be removed
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT void
PyArray_SetDatetimeParseFunction(PyObject *NPY_UNUSED(op))
{
}

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_CompareUCS4(npy_ucs4 const *s1, npy_ucs4 const *s2, size_t len)
{
    npy_ucs4 c1, c2;
    while(len-- > 0) {
        c1 = *s1++;
        c2 = *s2++;
        if (c1 != c2) {
            return (c1 < c2) ? -1 : 1;
        }
    }
    return 0;
}

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_CompareString(const char *s1, const char *s2, size_t len)
{
    const unsigned char *c1 = (unsigned char *)s1;
    const unsigned char *c2 = (unsigned char *)s2;
    size_t i;

    for(i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            return (c1[i] > c2[i]) ? 1 : -1;
        }
    }
    return 0;
}

static const char *might_be_writter_msg =
        "Numpy has detected that you (may be) writing to an array with\n"
        "overlapping memory from np.broadcast_arrays. If this is intentional\n"
        "set the WRITEABLE flag True or make a copy immediately before writing.";

/* Call this from contexts where an array might be written to, but we have no
 * way to tell. (E.g., when converting to a read-write buffer.)
 */
NPY_NO_EXPORT int
array_might_be_written(PyArrayObject *obj)
{
    if (PyArray_FLAGS(obj) & NPY_ARRAY_WARN_ON_WRITE) {
        capi_warn("array_might_be_written: warning...");
        if (DEPRECATE(might_be_writter_msg) < 0) {
            return -1;
        }
        /* Only warn once per array */
        while (1) {
            PyArray_CLEARFLAGS(obj, NPY_ARRAY_WARN_ON_WRITE);
            if (!PyArray_BASE(obj) || !PyArray_Check(PyArray_BASE(obj))) {
                break;
            }
            obj = (PyArrayObject *)PyArray_BASE(obj);
        }
    }
    return 0;
}

/*NUMPY_API
 *
 *  This function does nothing and returns 0 if *obj* is writeable.
 *  It raises an exception and returns -1 if *obj* is not writeable.
 *  It may also do other house-keeping, such as issuing warnings on
 *  arrays which are transitioning to become views. Always call this
 *  function at some point before writing to an array.
 *
 *  *name* is a name for the array, used to give better error messages.
 *  It can be something like "assignment destination", "output array",
 *  or even just "array".
 */
NPY_NO_EXPORT int
PyArray_FailUnlessWriteable(PyArrayObject *obj, const char *name)
{
    if (!PyArray_ISWRITEABLE(obj)) {
        PyErr_Format(PyExc_ValueError, "%s is read-only", name);
        return -1;
    }
    if (array_might_be_written(obj) < 0) {
        return -1;
    }
    return 0;
}

NPY_NO_EXPORT int
hpy_array_might_be_written(HPyContext *ctx, HPy obj, PyArrayObject *obj_data)
{
    const char *msg =
        "Numpy has detected that you (may be) writing to an array with\n"
        "overlapping memory from np.broadcast_arrays. If this is intentional\n"
        "set the WRITEABLE flag True or make a copy immediately before writing.";
    if (PyArray_FLAGS(obj_data) & NPY_ARRAY_WARN_ON_WRITE) {
        capi_warn("array_might_be_written: warning...");
        if (HPY_DEPRECATE(ctx, might_be_writter_msg) < 0) {
            return -1;
        }
        /* Only warn once per array */
        HPy x = HPy_Dup(ctx, obj);
        PyArrayObject *x_data = obj_data;
        while (1) {
            PyArray_CLEARFLAGS(x_data, NPY_ARRAY_WARN_ON_WRITE);
            HPy base = HPyArray_BASE(ctx, x, x_data);
            if (HPy_IsNull(base) || !HPyArray_Check(ctx, base)) {
                HPy_Close(ctx, base);
                break;
            }
            HPy_Close(ctx, x);
            x = base;
            x_data = PyArrayObject_AsStruct(ctx, x);
        }
    }
    return 0;
}

/*HPY_NUMPY_API
 */
NPY_NO_EXPORT int
HPyArray_FailUnlessWriteable(HPyContext *ctx, HPy obj, const char *name)
{
    PyArrayObject *obj_data = PyArrayObject_AsStruct(ctx, obj);
    return HPyArray_FailUnlessWriteableWithStruct(ctx, obj, obj_data, name);
}

NPY_NO_EXPORT int
HPyArray_FailUnlessWriteableWithStruct(HPyContext *ctx, HPy obj, PyArrayObject *obj_data, const char *name)
{
    if (!PyArray_ISWRITEABLE(obj_data)) {
        HPyErr_Format_p(ctx, ctx->h_ValueError, "%s is read-only", name);
        return -1;
    }
    if (hpy_array_might_be_written(ctx, obj, obj_data) < 0) {
        return -1;
    }
    return 0;
}

/* This also handles possibly mis-aligned data */
/* Compare s1 and s2 which are not necessarily NULL-terminated.
   s1 is of length len1
   s2 is of length len2
   If they are NULL terminated, then stop comparison.
*/
static int
_myunincmp(npy_ucs4 const *s1, npy_ucs4 const *s2, int len1, int len2)
{
    npy_ucs4 const *sptr;
    npy_ucs4 *s1t = NULL;
    npy_ucs4 *s2t = NULL;
    int val;
    npy_intp size;
    int diff;

    /* Replace `s1` and `s2` with aligned copies if needed */
    if ((npy_intp)s1 % sizeof(npy_ucs4) != 0) {
        size = len1*sizeof(npy_ucs4);
        s1t = malloc(size);
        memcpy(s1t, s1, size);
        s1 = s1t;
    }
    if ((npy_intp)s2 % sizeof(npy_ucs4) != 0) {
        size = len2*sizeof(npy_ucs4);
        s2t = malloc(size);
        memcpy(s2t, s2, size);
        s2 = s1t;
    }

    val = PyArray_CompareUCS4(s1, s2, PyArray_MIN(len1,len2));
    if ((val != 0) || (len1 == len2)) {
        goto finish;
    }
    if (len2 > len1) {
        sptr = s2+len1;
        val = -1;
        diff = len2-len1;
    }
    else {
        sptr = s1+len2;
        val = 1;
        diff=len1-len2;
    }
    while (diff--) {
        if (*sptr != 0) {
            goto finish;
        }
        sptr++;
    }
    val = 0;

 finish:
    /* Cleanup the aligned copies */
    if (s1t) {
        free(s1t);
    }
    if (s2t) {
        free(s2t);
    }
    return val;
}




/*
 * Compare s1 and s2 which are not necessarily NULL-terminated.
 * s1 is of length len1
 * s2 is of length len2
 * If they are NULL terminated, then stop comparison.
 */
static int
_mystrncmp(char const *s1, char const *s2, int len1, int len2)
{
    char const *sptr;
    int val;
    int diff;

    val = memcmp(s1, s2, PyArray_MIN(len1, len2));
    if ((val != 0) || (len1 == len2)) {
        return val;
    }
    if (len2 > len1) {
        sptr = s2 + len1;
        val = -1;
        diff = len2 - len1;
    }
    else {
        sptr = s1 + len2;
        val = 1;
        diff = len1 - len2;
    }
    while (diff--) {
        if (*sptr != 0) {
            return val;
        }
        sptr++;
    }
    return 0; /* Only happens if NULLs are everywhere */
}

/* Borrowed from Numarray */

#define SMALL_STRING 2048

static void _rstripw(char *s, int n)
{
    int i;
    for (i = n - 1; i >= 1; i--) { /* Never strip to length 0. */
        int c = s[i];

        if (!c || NumPyOS_ascii_isspace((int)c)) {
            s[i] = 0;
        }
        else {
            break;
        }
    }
}

static void _unistripw(npy_ucs4 *s, int n)
{
    int i;
    for (i = n - 1; i >= 1; i--) { /* Never strip to length 0. */
        npy_ucs4 c = s[i];
        if (!c || NumPyOS_ascii_isspace((int)c)) {
            s[i] = 0;
        }
        else {
            break;
        }
    }
}


static char *
_char_copy_n_strip(char const *original, char *temp, int nc)
{
    if (nc > SMALL_STRING) {
        temp = malloc(nc);
        if (!temp) {
            PyErr_NoMemory();
            return NULL;
        }
    }
    memcpy(temp, original, nc);
    _rstripw(temp, nc);
    return temp;
}

static void
_char_release(char *ptr, int nc)
{
    if (nc > SMALL_STRING) {
        free(ptr);
    }
}

static char *
_uni_copy_n_strip(char const *original, char *temp, int nc)
{
    if (nc*sizeof(npy_ucs4) > SMALL_STRING) {
        temp = malloc(nc*sizeof(npy_ucs4));
        if (!temp) {
            PyErr_NoMemory();
            return NULL;
        }
    }
    memcpy(temp, original, nc*sizeof(npy_ucs4));
    _unistripw((npy_ucs4 *)temp, nc);
    return temp;
}

static void
_uni_release(char *ptr, int nc)
{
    if (nc*sizeof(npy_ucs4) > SMALL_STRING) {
        free(ptr);
    }
}


/* End borrowed from numarray */

#define _rstrip_loop(CMP) {                                     \
        void *aptr, *bptr;                                      \
        char atemp[SMALL_STRING], btemp[SMALL_STRING];          \
        while(size--) {                                         \
            aptr = stripfunc(iself->dataptr, atemp, N1);        \
            if (!aptr) return -1;                               \
            bptr = stripfunc(iother->dataptr, btemp, N2);       \
            if (!bptr) {                                        \
                relfunc(aptr, N1);                              \
                return -1;                                      \
            }                                                   \
            val = compfunc(aptr, bptr, N1, N2);                 \
            *dptr = (val CMP 0);                                \
            PyArray_ITER_NEXT(iself);                           \
            PyArray_ITER_NEXT(iother);                          \
            dptr += 1;                                          \
            relfunc(aptr, N1);                                  \
            relfunc(bptr, N2);                                  \
        }                                                       \
    }

#define _reg_loop(CMP) {                                \
        while(size--) {                                 \
            val = compfunc((void *)iself->dataptr,      \
                          (void *)iother->dataptr,      \
                          N1, N2);                      \
            *dptr = (val CMP 0);                        \
            PyArray_ITER_NEXT(iself);                   \
            PyArray_ITER_NEXT(iother);                  \
            dptr += 1;                                  \
        }                                               \
    }

static int
_compare_strings(HPyContext *ctx, PyArrayObject *result, PyArrayMultiIterObject *multi,
                 int cmp_op, void *func, int rstrip)
{
    PyArrayIterObject *iself, *iother;
    npy_bool *dptr;
    npy_intp size;
    int val;
    int N1, N2;
    int (*compfunc)(void *, void *, int, int);
    void (*relfunc)(char *, int);
    char* (*stripfunc)(char const *, char *, int);

    compfunc = func;
    dptr = (npy_bool *)PyArray_DATA(result);
    iself = multi->iters[0];
    iother = multi->iters[1];
    size = multi->size;
    N1 = PyArray_DESCR(iself->ao)->elsize;
    N2 = PyArray_DESCR(iother->ao)->elsize;
    if ((void *)compfunc == (void *)_myunincmp) {
        N1 >>= 2;
        N2 >>= 2;
        stripfunc = _uni_copy_n_strip;
        relfunc = _uni_release;
    }
    else {
        stripfunc = _char_copy_n_strip;
        relfunc = _char_release;
    }
    switch (cmp_op) {
    case Py_EQ:
        if (rstrip) {
            _rstrip_loop(==);
        } else {
            _reg_loop(==);
        }
        break;
    case Py_NE:
        if (rstrip) {
            _rstrip_loop(!=);
        } else {
            _reg_loop(!=);
        }
        break;
    case Py_LT:
        if (rstrip) {
            _rstrip_loop(<);
        } else {
            _reg_loop(<);
        }
        break;
    case Py_LE:
        if (rstrip) {
            _rstrip_loop(<=);
        } else {
            _reg_loop(<=);
        }
        break;
    case Py_GT:
        if (rstrip) {
            _rstrip_loop(>);
        } else {
            _reg_loop(>);
        }
        break;
    case Py_GE:
        if (rstrip) {
            _rstrip_loop(>=);
        } else {
            _reg_loop(>=);
        }
        break;
    default:
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "bad comparison operator");
        return -1;
    }
    return 0;
}

#undef _reg_loop
#undef _rstrip_loop
#undef SMALL_STRING

NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip)
{
    PyArrayObject *result;
    PyArrayMultiIterObject *mit;
    int val;

    if (PyArray_TYPE(self) != PyArray_TYPE(other)) {
        /*
         * Comparison between Bytes and Unicode is not defined in Py3K;
         * we follow.
         */
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    if (PyArray_ISNOTSWAPPED(self) != PyArray_ISNOTSWAPPED(other)) {
        /* Cast `other` to the same byte order as `self` (both unicode here) */
        PyArray_Descr* unicode = PyArray_DescrNew(PyArray_DESCR(self));
        if (unicode == NULL) {
            return NULL;
        }
        unicode->elsize = PyArray_DESCR(other)->elsize;
        PyObject *new = PyArray_FromAny((PyObject *)other,
                unicode, 0, 0, 0, NULL);
        if (new == NULL) {
            return NULL;
        }
        other = (PyArrayObject *)new;
    }
    else {
        Py_INCREF(other);
    }

    /* Broad-cast the arrays to a common shape */
    mit = (PyArrayMultiIterObject *)PyArray_MultiIterNew(2, self, other);
    Py_DECREF(other);
    if (mit == NULL) {
        return NULL;
    }

    result = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                  PyArray_DescrFromType(NPY_BOOL),
                                  mit->nd,
                                  mit->dimensions,
                                  NULL, NULL, 0,
                                  NULL);
    if (result == NULL) {
        goto finish;
    }

    if (PyArray_TYPE(self) == NPY_UNICODE) {
        val = _compare_strings(npy_get_context(), result, mit, cmp_op, _myunincmp, rstrip);
    }
    else {
        val = _compare_strings(npy_get_context(), result, mit, cmp_op, _mystrncmp, rstrip);
    }

    if (val < 0) {
        Py_DECREF(result);
        result = NULL;
    }

 finish:
    Py_DECREF(mit);
    return (PyObject *)result;
}

NPY_NO_EXPORT HPy
_hpy_strings_richcompare(HPyContext *ctx, 
                            HPy /* PyArrayObject * */ h_self, 
                            HPy /* PyArrayObject * */ h_other, int cmp_op,
                     int rstrip)
{
    HPy result; // PyArrayObject *
    HPy h_local_other = h_other;
    PyArrayMultiIterObject *mit;
    int val;
    PyArrayObject *self = PyArrayObject_AsStruct(ctx, h_self);
    HPy h_self_descr = HPyArray_DESCR(ctx, h_self, self);
    PyArray_Descr *self_descr = PyArray_Descr_AsStruct(ctx, h_self_descr);
    int h_self_type_num = self_descr->type_num;

    PyArrayObject *other = PyArrayObject_AsStruct(ctx, h_local_other);
    HPy h_local_other_descr = HPyArray_DESCR(ctx, h_local_other, other);
    PyArray_Descr *local_other_descr = PyArray_Descr_AsStruct(ctx, h_local_other_descr);
    int h_local_other_type_num = local_other_descr->type_num;

    if (h_self_type_num != h_local_other_type_num) {
        /*
         * Comparison between Bytes and Unicode is not defined in Py3K;
         * we follow.
         */
        // Py_INCREF(Py_NotImplemented);
        HPy_Close(ctx, h_self_descr);
        HPy_Close(ctx, h_local_other_descr);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    if (PyArray_ISNOTSWAPPED(self) != PyArray_ISNOTSWAPPED(other)) {
        /* Cast `other` to the same byte order as `self` (both unicode here) */
        HPy h_unicode = HPyArray_DescrNew(ctx, h_self_descr); // PyArray_Descr* 
        if (HPy_IsNull(h_unicode)) {
            HPy_Close(ctx, h_self_descr);
            HPy_Close(ctx, h_local_other_descr);
            return HPy_NULL;
        }
        PyArray_Descr *unicode = PyArray_Descr_AsStruct(ctx, h_unicode);
        unicode->elsize = local_other_descr->elsize;
        HPy new = HPyArray_FromAny(ctx, h_other,
                h_unicode, 0, 0, 0, HPy_NULL);
        HPy_Close(ctx, h_unicode);
        if (HPy_IsNull(new)) {
            HPy_Close(ctx, h_self_descr);
            HPy_Close(ctx, h_local_other_descr);
            return HPy_NULL;
        }
        h_local_other = new;
    }
    else {
        h_local_other = HPy_Dup(ctx, h_other);
    }

    /* Broad-cast the arrays to a common shape */
    PyObject *py_self = HPy_AsPyObject(ctx, h_self);
    PyObject *py_local_other = HPy_AsPyObject(ctx, h_local_other);
    HPy_Close(ctx, h_local_other);
    mit = (PyArrayMultiIterObject *)PyArray_MultiIterNew(2, py_self, py_local_other);
    Py_DECREF(py_local_other);
    Py_DECREF(py_self);
    if (mit == NULL) {
        return HPy_NULL;
    }
    HPy h_PyArray_Type = HPyGlobal_Load(ctx, HPyArray_Type);
    HPy bool_descr = HPyArray_DescrFromType(ctx, NPY_BOOL);
    result = HPyArray_NewFromDescr(ctx, h_PyArray_Type,
                                  bool_descr,
                                  mit->nd,
                                  mit->dimensions,
                                  NULL, NULL, 0,
                                  HPy_NULL);
    HPy_Close(ctx, h_PyArray_Type);
    HPy_Close(ctx, bool_descr);
    if (HPy_IsNull(result)) {
        goto finish;
    }

    PyArrayObject *result_data = PyArrayObject_AsStruct(ctx, result);
    if (PyArray_TYPE(self) == NPY_UNICODE) {
        val = _compare_strings(ctx, result_data, mit, cmp_op, _myunincmp, rstrip);
    }
    else {
        val = _compare_strings(ctx, result_data, mit, cmp_op, _mystrncmp, rstrip);
    }

    if (val < 0) {
        HPy_Close(ctx, result);
        result = HPy_NULL;
    }

 finish:
    Py_DECREF(mit);
    return result;
}

static HPy
hpy_array_richcompare(HPyContext *ctx, /*PyArrayObject*/ HPy self, HPy other, HPy_RichCmpOp cmp_op);

/*
 * VOID-type arrays can only be compared equal and not-equal
 * in which case the fields are all compared by extracting the fields
 * and testing one at a time...
 * equality testing is performed using logical_ands on all the fields.
 * in-equality testing is performed using logical_ors on all the fields.
 *
 * VOID-type arrays without fields are compared for equality by comparing their
 * memory at each location directly (using string-code).
 */
static HPy
_void_compare(HPyContext *ctx, 
                HPy /* PyArrayObject * */ h_self, 
                HPy /* PyArrayObject * */ h_other, int cmp_op)
{
    if (!(cmp_op == HPy_EQ || cmp_op == HPy_NE)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Void-arrays can only be compared for equality.");
        return HPy_NULL;
    }
    PyArrayObject *self = PyArrayObject_AsStruct(ctx, h_self);
    HPy h_self_descr = HPyArray_DESCR(ctx, h_self, self);
    PyArray_Descr *self_descr = PyArray_Descr_AsStruct(ctx, h_self_descr);
    int h_self_type_num = self_descr->type_num;

    PyArrayObject *other = PyArrayObject_AsStruct(ctx, h_other);
    HPy h_other_descr = HPyArray_DESCR(ctx, h_other, other);
    PyArray_Descr *other_descr = PyArray_Descr_AsStruct(ctx, h_other_descr);
    int h_local_other_type_num = other_descr->type_num;
    if (PyArray_HASFIELDS(self)) {
        HPy res = HPy_NULL;
        HPy temp, temp2;
        npy_intp result_ndim = PyArray_NDIM(self) > PyArray_NDIM(other) ?
                            PyArray_NDIM(self) : PyArray_NDIM(other);
        HPy op = HPyGlobal_Load(ctx, cmp_op == HPy_EQ ? hpy_n_ops.logical_and : hpy_n_ops.logical_or);
        HPy fields = HPy_FromPyObject(ctx, self_descr->fields);
        HPy keys = HPyDict_Keys(ctx, fields);
        HPy_ssize_t keys_len = HPy_Length(ctx, keys);
        for (HPy_ssize_t i = 0; i < keys_len; i++) {
            HPy key = HPy_GetItem_i(ctx, keys, i);
            HPy value = HPy_GetItem(ctx, fields, key);
            if (HNPY_TITLE_KEY(ctx, key, value)) {
                HPy_Close(ctx, key);
                HPy_Close(ctx, value);
                continue;
            }
            HPy_Close(ctx, value);
            HPy a = array_subscript_asarray(ctx, h_self, key);
            if (HPy_IsNull(a)) {
                HPy_Close(ctx, key);
                HPy_Close(ctx, res);
                return HPy_NULL;
            }
            HPy b = array_subscript_asarray(ctx, h_other, key);
            HPy_Close(ctx, key);
            if (HPy_IsNull(b)) {
                HPy_Close(ctx, res);
                HPy_Close(ctx, a);
                return HPy_NULL;
            }
            temp = hpy_array_richcompare(ctx, a, b, cmp_op);
            HPy_Close(ctx, a);
            HPy_Close(ctx, b);
            if (HPy_IsNull(temp)) {
                HPy_Close(ctx, res);
                return HPy_NULL;
            }

            /*
             * If the field type has a non-trivial shape, additional
             * dimensions will have been appended to `a` and `b`.
             * In that case, reduce them using `op`.
             */
            if (HPyArray_Check(ctx, temp)) {
                PyArrayObject *temp_data = PyArrayObject_AsStruct(ctx, temp);
                if (PyArray_NDIM(temp_data) > result_ndim) {
                    /* If the type was multidimensional, collapse that part to 1-D
                     */
                    if (PyArray_NDIM(temp_data) != result_ndim+1) {
                        npy_intp dimensions[NPY_MAXDIMS];
                        PyArray_Dims newdims;

                        newdims.ptr = dimensions;
                        newdims.len = result_ndim+1;
                        if (result_ndim) {
                            memcpy(dimensions, PyArray_DIMS(temp_data),
                                sizeof(npy_intp)*result_ndim);
                        }
                        dimensions[result_ndim] = -1;
                        temp2 = HPyArray_Newshape(ctx, temp, temp_data,
                                                &newdims, NPY_ANYORDER);
                        if (HPy_IsNull(temp2)) {
                            HPy_Close(ctx, temp);
                            HPy_Close(ctx, res);
                            return HPy_NULL;
                        }
                        HPy_Close(ctx, temp);
                        temp = temp2;
                    }
                    /* Reduce the extra dimension of `temp` using `op` */
                    temp2 = HPyArray_GenericReduceFunction(ctx, temp,
                                                        op, result_ndim,
                                                        NPY_BOOL, HPy_NULL);
                    if (HPy_IsNull(temp2)) {
                        HPy_Close(ctx, temp);
                        HPy_Close(ctx, res);
                        return HPy_NULL;
                    }
                    HPy_Close(ctx, temp);
                    temp = temp2;
                    }
            }

            if (HPy_IsNull(res)) {
                res = temp;
            }
            else {
                HPy args = HPyTuple_Pack(ctx, 2, res, temp);
                temp2 = HPy_CallTupleDict(ctx, op, args, HPy_NULL);
                HPy_Close(ctx, args);
                HPy_Close(ctx, temp);
                HPy_Close(ctx, res);
                if (HPy_IsNull(temp2)) {
                    return HPy_NULL;
                }
                res = temp2;
            }
        }
        if (HPy_IsNull(res) && !HPyErr_Occurred(ctx)) {
            /* these dtypes had no fields. Use a MultiIter to broadcast them
             * to an output array, and fill with True (for EQ)*/
            PyArrayMultiIterObject *mit = (PyArrayMultiIterObject *)
                                          PyArray_MultiIterNew(2, self, other);
            if (mit == NULL) {
                return HPy_NULL;
            }
            HPy h_PyArray_Type = HPyGlobal_Load(ctx, HPyArray_Type);
            HPy h_bool = HPyArray_DescrFromType(ctx, NPY_BOOL);
            res = HPyArray_NewFromDescr(ctx, h_PyArray_Type,
                                       h_bool,
                                       mit->nd, mit->dimensions,
                                       NULL, NULL, 0, HPy_NULL);
            HPy_Close(ctx, h_PyArray_Type);
            HPy_Close(ctx, h_bool);
            Py_DECREF(mit);
            if (!HPy_IsNull(res)) {
                 PyArray_FILLWBYTE(PyArrayObject_AsStruct(ctx, res),
                                   cmp_op == HPy_EQ ? 1 : 0);
            }
        }
        return res;
    }
    else {
        /* compare as a string. Assumes self and other have same descr->type */
        return _hpy_strings_richcompare(ctx, h_self, h_other, cmp_op, 0);
    }
}

/*
 * Silence the current error and emit a deprecation warning instead.
 *
 * If warnings are raised as errors, this sets the warning __cause__ to the
 * silenced error.
 */
NPY_NO_EXPORT int
DEPRECATE_silence_error(const char *msg) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    if (DEPRECATE(msg) < 0) {
        npy_PyErr_ChainExceptionsCause(exc, val, tb);
        return -1;
    }
    Py_XDECREF(exc);
    Py_XDECREF(val);
    Py_XDECREF(tb);
    return 0;
}

/*
 * Comparisons can fail, but we do not always want to pass on the exception
 * (see comment in array_richcompare below), but rather return NotImplemented.
 * Here, an exception should be set on entrance.
 * Returns either NotImplemented with the exception cleared, or NULL
 * with the exception set.
 * Raises deprecation warnings for cases where behaviour is meant to change
 * (2015-05-14, 1.10)
 */

NPY_NO_EXPORT HPy
_failed_comparison_workaround(HPyContext *ctx, HPy /* (PyArrayObject *) */ self, HPy other, int cmp_op)
{
    // PyObject *exc, *val, *tb;
    HPy /* (PyArrayObject *) */ array_other;
    int other_is_flexible, ndim_other;
    HPy descr = HPyArray_GetDescr(ctx, self);
    int self_is_flexible = PyTypeNum_ISFLEXIBLE(PyArray_Descr_AsStruct(ctx, descr)->type_num);
    HPy_Close(ctx, descr);

    // TOOD HPY LABS PORT: PyErr_Fetch
    // PyErr_Fetch(&exc, &val, &tb);
    /*
     * Determine whether other has a flexible dtype; here, inconvertible
     * is counted as inflexible.  (This repeats work done in the ufunc,
     * but OK to waste some time in an unlikely path.)
     */
    array_other = HPyArray_FROM_O(ctx, other);
    if (!HPy_IsNull(array_other)) {
        descr = HPyArray_GetDescr(ctx, array_other);
        other_is_flexible = PyTypeNum_ISFLEXIBLE(
            PyArray_Descr_AsStruct(ctx, descr)->type_num);
        HPy_Close(ctx, descr);
        ndim_other = HPyArray_GetNDim(ctx, array_other);
        HPy_Close(ctx, array_other);
    }
    else {
        // HPyErr_Clear(ctx); /* we restore the original error if needed */
        other_is_flexible = 0;
        ndim_other = 0;
    }
    if (cmp_op == Py_EQ || cmp_op == Py_NE) {
        /*
         * note: for == and !=, a structured dtype self cannot get here,
         * but a string can. Other can be string or structured.
         */
        if (other_is_flexible || self_is_flexible) {
            /*
             * For scalars, returning NotImplemented is correct.
             * For arrays, we emit a future deprecation warning.
             * When this warning is removed, a correctly shaped
             * array of bool should be returned.
             */
            if (ndim_other != 0 || HPyArray_GetNDim(ctx, self) != 0) {
                /* 2015-05-14, 1.10 */
                if (HPY_DEPRECATE_FUTUREWARNING(ctx,
                        "elementwise comparison failed; returning scalar "
                        "instead, but in the future will perform "
                        "elementwise comparison") < 0) {
                    goto fail;
                }
            }
        }
        else {
            /*
             * If neither self nor other had a flexible dtype, the error cannot
             * have been caused by a lack of implementation in the ufunc.
             *
             * 2015-05-14, 1.10
             */
            if (HPY_DEPRECATE_FUTUREWARNING(ctx,
                    "elementwise comparison failed; "
                    "this will raise an error in the future.") < 0) {
                goto fail;
            }
        }
        // Py_XDECREF(exc);
        // Py_XDECREF(val);
        // Py_XDECREF(tb);
        HPyErr_Clear(ctx);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    else if (other_is_flexible || self_is_flexible) {
        /*
         * For LE, LT, GT, GE and a flexible self or other, we return
         * NotImplemented, which is the correct answer since the ufuncs do
         * not in fact implement loops for those.  This will get us the
         * desired TypeError.
         */
        // Py_XDECREF(exc);
        // Py_XDECREF(val);
        // Py_XDECREF(tb);
        HPyErr_Clear(ctx);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    else {
        /* LE, LT, GT, or GE with non-flexible other; just pass on error */
        goto fail;
    }

fail:
    /*
     * Reraise the original exception, possibly chaining with a new one.
     */
    // npy_PyErr_ChainExceptionsCause(exc, val, tb);
    return HPy_NULL;
}

HPyDef_SLOT(array_richcompare_def, hpy_array_richcompare, HPy_tp_richcompare);

PyObject *array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op)
{
    CAPI_WARN("array_richcompare");
    HPyContext *ctx = npy_get_context();
    HPy h_self = HPy_FromPyObject(ctx, (PyObject*)self);
    HPy h_other = HPy_FromPyObject(ctx, (PyObject*)other);
    HPy result = hpy_array_richcompare(ctx, h_self, h_other, (HPy_RichCmpOp) cmp_op);
    HPy_Close(ctx, h_self);
    HPy_Close(ctx, h_other);
    PyObject *res = HPy_AsPyObject(ctx, result);
    HPy_Close(ctx, result);
    return res;
}

static HPy
hpy_array_richcompare(HPyContext *ctx, /*PyArrayObject*/HPy self, HPy other, HPy_RichCmpOp cmp_op)
{
    HPy result = HPy_NULL;

    /* Special case for string arrays (which don't and currently can't have
     * ufunc loops defined, so there's no point in trying).
     */
    if (HPyArray_ISSTRING(ctx, self)) {
        HPy array_other = HPyArray_FromObject(ctx, other, NPY_NOTYPE, 0, 0);
        if (HPy_IsNull(array_other)) {
            HPyErr_Clear(ctx);
            /* Never mind, carry on, see what happens */
        }
        else if (!HPyArray_ISSTRING(ctx, array_other)) {
            HPy_Close(ctx, array_other);
            /* Never mind, carry on, see what happens */
        }
        else {
            result = _hpy_strings_richcompare(ctx, self, array_other, cmp_op, 0);
            HPy_Close(ctx, array_other);
            return result;
        }
        /* If we reach this point, it means that we are not comparing
         * string-to-string. It's possible that this will still work out,
         * e.g. if the other array is an object array, then both will be cast
         * to object or something? I don't know how that works actually, but
         * it does, b/c this works:
         *   l = ["a", "b"]
         *   assert np.array(l, dtype="S1") == np.array(l, dtype="O")
         * So we fall through and see what happens.
         */
    }

    switch (cmp_op) {
    case Py_LT:
        HPY_RICHCMP_GIVE_UP_IF_NEEDED(ctx, self, other);
        result = HPyArray_GenericBinaryFunction(
                ctx, self, other, hpy_n_ops.less);
        break;
    case Py_LE:
        HPY_RICHCMP_GIVE_UP_IF_NEEDED(ctx, self, other);
        result = HPyArray_GenericBinaryFunction(
                ctx, self, other, hpy_n_ops.less_equal);
        break;
    case Py_EQ:
        HPY_RICHCMP_GIVE_UP_IF_NEEDED(ctx, self, other);
        /*
         * The ufunc does not support void/structured types, so these
         * need to be handled specifically. Only a few cases are supported.
         */

        if (HPyArray_GetType(ctx, self) == NPY_VOID) {
            int _res;

            HPy array_other = HPyArray_FROM_O(ctx, other); // (PyArrayObject *)
            /*
             * If not successful, indicate that the items cannot be compared
             * this way.
             */
            if (HPy_IsNull(array_other)) {
                /* 2015-05-07, 1.10 */
                CAPI_WARN("missing PyErr_Fetch");
                if (DEPRECATE_silence_error(
                        "elementwise == comparison failed and returning scalar "
                        "instead; this will raise an error in the future.") < 0) {
                    return HPy_NULL;
                }
                return HPy_Dup(ctx, ctx->h_NotImplemented);
            }

            HPy self_descr = HPyArray_GetDescr(ctx, self);
            HPy array_other_descr = HPyArray_GetDescr(ctx, array_other);
            _res = HPyArray_CheckCastSafety(ctx,
                    NPY_EQUIV_CASTING,
                    self_descr, array_other_descr, HPy_NULL);
            HPy_Close(ctx, self_descr);
            HPy_Close(ctx, array_other_descr);
            if (_res < 0) {
                HPyErr_Clear(ctx);
                _res = 0;
            }
            if (_res == 0) {
                /* 2015-05-07, 1.10 */
                HPy_Close(ctx, array_other);
                if (HPY_DEPRECATE_FUTUREWARNING(ctx,
                        "elementwise == comparison failed and returning scalar "
                        "instead; this will raise an error or perform "
                        "elementwise comparison in the future.") < 0) {
                    return HPy_NULL;
                }
                return HPy_Dup(ctx, ctx->h_False);
            }
            else {
                result = _void_compare(ctx, self, array_other, cmp_op);
            }
            HPy_Close(ctx, array_other);
            return result;
        }

        result = HPyArray_GenericBinaryFunction(
                ctx, self, other, hpy_n_ops.equal);
        break;
    case Py_NE:
        HPY_RICHCMP_GIVE_UP_IF_NEEDED(ctx, self, other);
        /*
         * The ufunc does not support void/structured types, so these
         * need to be handled specifically. Only a few cases are supported.
         */

        if (HPyArray_GetType(ctx, self) == NPY_VOID) {
            int _res;

            HPy array_other = HPyArray_FROM_O(ctx, other); // (PyArrayObject *)
            /*
             * If not successful, indicate that the items cannot be compared
             * this way.
            */
            if (HPy_IsNull(array_other)) {
                /* 2015-05-07, 1.10 */
                CAPI_WARN("missing PyErr_Fetch");
                if (DEPRECATE_silence_error(
                        "elementwise != comparison failed and returning scalar "
                        "instead; this will raise an error in the future.") < 0) {
                    return HPy_NULL;
                }
                return HPy_Dup(ctx, ctx->h_NotImplemented);
            }
            HPy self_descr = HPyArray_GetDescr(ctx, self);
            HPy array_other_descr = HPyArray_GetDescr(ctx, array_other);
            _res = HPyArray_CheckCastSafety(ctx,
                    NPY_EQUIV_CASTING,
                    self_descr, array_other_descr, HPy_NULL);
            HPy_Close(ctx, self_descr);
            HPy_Close(ctx, array_other_descr);
            if (_res < 0) {
                HPyErr_Clear(ctx);
                _res = 0;
            }
            if (_res == 0) {
                /* 2015-05-07, 1.10 */
                HPy_Close(ctx, array_other);
                if (HPY_DEPRECATE_FUTUREWARNING(ctx,
                        "elementwise != comparison failed and returning scalar "
                        "instead; this will raise an error or perform "
                        "elementwise comparison in the future.") < 0) {
                    return HPy_NULL;
                }
                return HPy_Dup(ctx, ctx->h_True);
            }
            else {
                result = _void_compare(ctx, self, array_other, cmp_op);
                HPy_Close(ctx, array_other);
            }
            return result;
        }

        result = HPyArray_GenericBinaryFunction(
                ctx, self, other, hpy_n_ops.not_equal);
        break;
    case Py_GT:
        HPY_RICHCMP_GIVE_UP_IF_NEEDED(ctx, self, other);
        result = HPyArray_GenericBinaryFunction(
                ctx, self, other, hpy_n_ops.greater);
        break;
    case Py_GE:
        HPY_RICHCMP_GIVE_UP_IF_NEEDED(ctx, self, other);
        result = HPyArray_GenericBinaryFunction(
                ctx, self, other, hpy_n_ops.greater_equal);
        break;
    default:
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    if (HPy_IsNull(result)) {
        /*
         * 2015-05-14, 1.10; updated 2018-06-18, 1.16.
         *
         * Comparisons can raise errors when element-wise comparison is not
         * possible. Some of these, though, should not be passed on.
         * In particular, the ufuncs do not have loops for flexible dtype,
         * so those should be treated separately.  Furthermore, for EQ and NE,
         * we should never fail.
         *
         * Our ideal behaviour would be:
         *
         * 1. For EQ and NE:
         *   - If self and other are scalars, return NotImplemented,
         *     so that python can assign True of False as appropriate.
         *   - If either is an array, return an array of False or True.
         *
         * 2. For LT, LE, GE, GT:
         *   - If self or other was flexible, return NotImplemented
         *     (as is in fact the case), so python can raise a TypeError.
         *   - If other is not convertible to an array, pass on the error
         *     (MHvK, 2018-06-18: not sure about this, but it's what we have).
         *
         * However, for backwards compatibility, we cannot yet return arrays,
         * so we raise warnings instead.
         */
        result = _failed_comparison_workaround(ctx, self, other, cmp_op);
    }
    return result;
}

static int _PyArray_ElementStrides(PyArrayObject *arr) {
    int itemsize = PyArray_ITEMSIZE(arr);
    int ndim = PyArray_NDIM(arr);
    npy_intp *strides = PyArray_STRIDES(arr);

    for (int i = 0; i < ndim; i++) {
        if ((strides[i] % itemsize) != 0) {
            return 0;
        }
    }
    return 1;
}

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_ElementStrides(PyObject *obj)
{
    if (!PyArray_Check(obj)) {
        return 0;
    }

    return _PyArray_ElementStrides((PyArrayObject *)obj);
}

/*HPY_NUMPY_API
 */
NPY_NO_EXPORT int
HPyArray_ElementStrides(HPyContext *ctx, HPy obj)
{
    if (!HPyArray_Check(ctx, obj)) {
        return 0;
    }

    return _PyArray_ElementStrides(PyArrayObject_AsStruct(ctx, obj));
}

/*
 * This routine checks to see if newstrides (of length nd) will not
 * ever be able to walk outside of the memory implied numbytes and offset.
 *
 * The available memory is assumed to start at -offset and proceed
 * to numbytes-offset.  The strides are checked to ensure
 * that accessing memory using striding will not try to reach beyond
 * this memory for any of the axes.
 *
 * If numbytes is 0 it will be calculated using the dimensions and
 * element-size.
 *
 * This function checks for walking beyond the beginning and right-end
 * of the buffer and therefore works for any integer stride (positive
 * or negative).
 */

/*NUMPY_API*/
NPY_NO_EXPORT npy_bool
PyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset,
                     npy_intp const *dims, npy_intp const *newstrides)
{
    npy_intp begin, end;
    npy_intp lower_offset;
    npy_intp upper_offset;

    if (numbytes == 0) {
        numbytes = PyArray_MultiplyList(dims, nd) * elsize;
    }

    begin = -offset;
    end = numbytes - offset;

    offset_bounds_from_strides(elsize, nd, dims, newstrides,
                                        &lower_offset, &upper_offset);

    if ((upper_offset > end) || (lower_offset < begin)) {
        return NPY_FALSE;
    }
    return NPY_TRUE;
}


HPyDef_SLOT(array_new, array_new_impl, HPy_tp_new)
static HPy array_new_impl(HPyContext *ctx, HPy h_subtype, HPy *args_h,
                          HPy_ssize_t nargs, HPy h_kwds)
{
    static const char *kwlist[] = {"shape", "dtype", "buffer", "offset", "strides",
                             "order", NULL};
    HPy h_descr_in;
    int itemsize;
    PyArray_Dims dims = {NULL, 0};
    PyArray_Dims strides = {NULL, -1};
    HPyArray_Chunk buffer;
    npy_longlong offset = 0;
    NPY_ORDER order = NPY_CORDER;
    int is_f_order = 0;

    HPyTracker ht;
    HPy h_dims = HPy_NULL;
    HPy h_strides = HPy_NULL;
    HPy h_buffer = HPy_NULL;
    HPy h_order = HPy_NULL;

    /*
     * Usually called with shape and type but can also be called with buffer,
     * strides, and swapped info For now, let's just use this to create an
     * empty, contiguous array of a specific type and shape.
     */
    buffer.ptr = NULL;
    if (!HPyArg_ParseKeywords(ctx, &ht, args_h, nargs, h_kwds, "O|OOLOO", kwlist,
            &h_dims, &h_descr_in, &h_buffer, &offset, &h_strides, &h_order)) {
        return HPy_NULL;
    }

    HPy h_descr;
    if (HPyArray_IntpConverter(ctx, h_dims, &dims) != NPY_SUCCEED ||
        HPyArray_DescrConverter(ctx, h_descr_in, &h_descr) != NPY_SUCCEED) {
        goto fail;
    }
    if (!HPy_IsNull(h_descr)) {
        HPyTracker_Add(ctx, ht, h_descr);
    }
    if (HPyArray_BufferConverter(ctx, h_buffer, &buffer) != NPY_SUCCEED){
        goto fail;
    }
    if (!HPy_IsNull(buffer.base)) {
        HPyTracker_Add(ctx, ht, buffer.base);
    }
    if (HPyArray_OptionalIntpConverter(ctx, h_strides, &strides) != NPY_SUCCEED ||
        HPyArray_OrderConverter(ctx, h_order, &order) != NPY_SUCCEED) {
        goto fail;
    }

    if (order == NPY_FORTRANORDER) {
        is_f_order = 1;
    }
    if (HPy_IsNull(h_descr)) {
        h_descr = HPyArray_DescrFromType(ctx, NPY_DEFAULT_TYPE);
        HPyTracker_Add(ctx, ht, h_descr);
    }

    PyArray_Descr *descr = PyArray_Descr_AsStruct(ctx, h_descr);
    itemsize = descr->elsize;

    if (strides.len != -1) {
        npy_intp nb, off;
        if (strides.len != dims.len) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "strides, if given, must be "   \
                            "the same length as shape");
            goto fail;
        }

        if (buffer.ptr == NULL) {
            nb = 0;
            off = 0;
        }
        else {
            nb = buffer.len;
            off = (npy_intp) offset;
        }


        // HPy Note: PyArray_CheckStrides seems to not use any C API...
        if (!PyArray_CheckStrides(itemsize, dims.len,
                                  nb, off,
                                  dims.ptr, strides.ptr)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "strides is incompatible "      \
                            "with shape of requested "      \
                            "array and size of buffer");
            goto fail;
        }
    }

    HPy h_result;
    if (buffer.ptr == NULL) {
        h_result = HPyArray_NewFromDescr_int(
                ctx, h_subtype, h_descr,
                (int)dims.len, dims.ptr, strides.ptr, NULL,
                is_f_order, HPy_NULL, HPy_NULL,
                0, 1);
        if (HPy_IsNull(h_result)) {
            goto fail;
        }
        if (PyDataType_FLAGCHK(descr, NPY_ITEM_HASOBJECT)) {
            /* place Py_None in object positions */
            CAPI_WARN("PyArray_FillObjectArray");
            // TODO HPY LABS PORT
            PyObject *ret = HPy_AsPyObject(ctx, h_result);
            PyArray_FillObjectArray((PyArrayObject*)ret, Py_None);
            Py_DECREF(ret);
            if (HPyErr_Occurred(ctx)) {
                descr = NULL;
                goto fail;
            }
        }
    }
    else {
        /* buffer given -- use it */
        if (dims.len == 1 && dims.ptr[0] == -1) {
            dims.ptr[0] = (buffer.len-(npy_intp)offset) / itemsize;
        }
        else if ((strides.ptr == NULL) &&
                 (buffer.len < (offset + (((npy_intp)itemsize)*
                                          PyArray_MultiplyList(dims.ptr,
                                                               dims.len))))) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                            "buffer is too small for "      \
                            "requested array");
            goto fail;
        }
        /* get writeable and aligned */
        if (is_f_order) {
            buffer.flags |= NPY_ARRAY_F_CONTIGUOUS;
        }
        h_result = HPyArray_NewFromDescr_int(
                ctx, h_subtype, h_descr,
                dims.len, dims.ptr, strides.ptr, offset + (char *)buffer.ptr,
                buffer.flags, HPy_NULL, buffer.base,
                0, 1);
        if (HPy_IsNull(h_result)) {
            descr = NULL;
            goto fail;
        }
    }

    HPyTracker_Close(ctx, ht);
    npy_free_cache_dim_obj(dims);
    npy_free_cache_dim_obj(strides);
    return h_result;

 fail:
    HPyTracker_Close(ctx, ht);
    npy_free_cache_dim_obj(dims);
    npy_free_cache_dim_obj(strides);
    return HPy_NULL;
}


HPyDef_SLOT(array_iter, array_iter_impl, HPy_tp_iter)
static HPy
array_iter_impl(HPyContext *ctx, HPy h_arr)
{
    PyArrayObject *arr = PyArrayObject_AsStruct(ctx, h_arr);
    if (PyArray_NDIM(arr) == 0) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                        "iteration over a 0-d array");
        return HPy_NULL;
    }
    return HPySeqIter_New(ctx, h_arr);
}



static PyType_Slot PyArray_Type_slots[] = {
    {Py_tp_str, (reprfunc)array_str},

    {Py_tp_methods, array_methods},
    {Py_tp_getset, array_getsetlist}, // HPY: they are not needed for the example
    {0, NULL},
};

static HPyDef *array_defines[] = {
    &array_assign_subscript,
    &array_subscript,
    &mp_array_length,

    &array_getbuffer,
    &array_length,
    &array_item,
    &array_new,
    &array_traverse,
    &array_finalize,
    &array_destroy,
    &array_inplace_add,
    &array_power,
    &array_subtract,
    &array_true_divide,
    &array_add,
    &array_richcompare_def,
    &array_bitwise_and,
    &array_bitwise_or,
    &array_bitwise_xor,
    &array_iter,
    &array_float,
    &array_int,
    &array_multiply_slot,
    &array_remainder_slot,
    &array_divmod_slot,
    &array_negative_slot,
    &array_positive_slot,
    &array_absolute_slot,
    &_array_nonzero_slot,
    &array_invert_slot,
    &array_left_shift_slot,
    &array_right_shift_slot,
    &array_index_slot,
    &array_inplace_subtract_slot,
    &array_inplace_multiply_slot,
    &array_inplace_remainder_slot,
    &array_inplace_power_slot,
    &array_inplace_left_shift_slot,
    &array_inplace_right_shift_slot,
    &array_inplace_bitwise_and_slot,
    &array_inplace_bitwise_xor_slot,
    &array_inplace_bitwise_or_slot,
    &array_floor_divide_slot,
    &array_inplace_floor_divide_slot,
    &array_inplace_true_divide_slot,
    &array_matrix_multiply_slot,
    &array_inplace_matrix_multiply_slot,
    &array_concat_slot,
    &array_assign_item_slot,
    &array_contains_slot,
    &array_repr_slot,

    // getset
    &array_shape,
    &array_descr_dtype,
    &array_ndim_get,
    &array_strides,
    &array_priority_get,
    &array_flags_get,
    &array_ctypes_get,
    &array_transpose_get,
    &array_base_get,
    &array_itemsize_get,
    &array_size_get,
    &array_nbytes_get,

    // methods:
    &array_ravel,
    &array_transpose,
    &array_compress,
    &array_nonzero,
    &array_min_scalar_type,
    &array_promote_types,
    NULL,
};

NPY_NO_EXPORT HPyType_Spec PyArray_Type_spec = {
    .name = "numpy.ndarray",
    .basicsize = sizeof(PyArrayObject_fields),
    .flags = (HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_BASETYPE | HPy_TPFLAGS_HAVE_GC),
    .defines = array_defines,
    .legacy_slots = PyArray_Type_slots,
    .legacy = true,
};

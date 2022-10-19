/*
 * This module corresponds to the `Special functions for NPY_OBJECT`
 * section in the numpy reference for C-API.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "iterators.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "hpy_utils.h"

static void
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype);

static void
_hpy_fillobject(HPyContext *ctx, char *optr, HPy obj, HPy /* PyArray_Descr * */ dtype);

/*NUMPY_API
 * XINCREF all objects in a single array item. This is complicated for
 * structured datatypes where the position of objects needs to be extracted.
 * The function is execute recursively for each nested field or subarrays dtype
 * such as as `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }
    if (descr->type_num == NPY_OBJECT) {
        memcpy(&temp, data, sizeof(temp));
        Py_XINCREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(descr->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                return;
            }
            PyArray_Item_INCREF(data + offset, new);
        }
    }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        for (i = 0; i < size; i++){
            /* Recursively increment the reference count of subarray elements */
            PyArray_Item_INCREF(data + i * inner_elsize,
                                descr->subarray->base);
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}

/*HPY_NUMPY_API
 * XINCREF all objects in a single array item. This is complicated for
 * structured datatypes where the position of objects needs to be extracted.
 * The function is execute recursively for each nested field or subarrays dtype
 * such as as `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
HPyArray_Item_INCREF(HPyContext *ctx, char *data, HPy h_descr)
{
    PyArray_Descr *descr = PyArray_Descr_AsStruct(ctx, h_descr);
    if (!PyDataType_REFCHK(descr)) {
        return;
    }
    if (descr->type_num == NPY_OBJECT) {
        PyObject *temp;
        CAPI_WARN("leaked py refs in a PyArray");
        memcpy(&temp, data, sizeof(temp));
        Py_XINCREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
        HPy new; // PyArray_Descr *
        int offset;
        Py_ssize_t pos = 0;

        HPy fields = HPy_FromPyObject(ctx, descr->fields);
        HPy keys = HPyDict_Keys(ctx, fields);
        HPy_ssize_t keys_len = HPy_Length(ctx, keys);
        for (HPy_ssize_t i = 0; i < keys_len; i++) {
            HPy key = HPy_GetItem_i(ctx, keys, i);
            HPy value = HPy_GetItem(ctx, fields, key);
            if (HNPY_TITLE_KEY(ctx, key, value)) {
                HPy_Close(ctx, key);
                HPy_Close(ctx, value);
                return;
            }
            HPy_Close(ctx, key);
            if (!HPy_ExtractDictItems_OiO(ctx, value, &new, &offset, NULL)) {
                // error
                HPy_Close(ctx, value);
                HPy_Close(ctx, keys);
                HPy_Close(ctx, fields);
                return;
            }
            HPy_Close(ctx, value);
            HPyArray_Item_INCREF(ctx, data + offset, new);
            HPy_Close(ctx, new);
        }
        HPy_Close(ctx, keys);
        HPy_Close(ctx, fields);
    }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        HPy h_base = HPy_FromPyObject(ctx, descr->subarray->base);
        for (i = 0; i < size; i++){
            /* Recursively increment the reference count of subarray elements */
            HPyArray_Item_INCREF(ctx, data + i * inner_elsize, h_base);
        }
        HPy_Close(ctx, h_base);
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}

/*NUMPY_API
 *
 * XDECREF all objects in a single array item. This is complicated for
 * structured datatypes where the position of objects needs to be extracted.
 * The function is execute recursively for each nested field or subarrays dtype
 * such as as `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }

    if (descr->type_num == NPY_OBJECT) {
        memcpy(&temp, data, sizeof(temp));
        Py_XDECREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
            PyObject *key, *value, *title = NULL;
            PyArray_Descr *new;
            int offset;
            Py_ssize_t pos = 0;

            while (PyDict_Next(descr->fields, &pos, &key, &value)) {
                if (NPY_TITLE_KEY(key, value)) {
                    continue;
                }
                if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                      &title)) {
                    return;
                }
                PyArray_Item_XDECREF(data + offset, new);
            }
        }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        for (i = 0; i < size; i++){
            /* Recursively decrement the reference count of subarray elements */
            PyArray_Item_XDECREF(data + i * inner_elsize,
                                 descr->subarray->base);
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}

/*HPY_NUMPY_API
 *
 * XDECREF all objects in a single array item. This is complicated for
 * structured datatypes where the position of objects needs to be extracted.
 * The function is execute recursively for each nested field or subarrays dtype
 * such as as `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
HPyArray_Item_XDECREF(HPyContext *ctx, char *data, HPy h_descr)
{
    PyArray_Descr *descr = PyArray_Descr_AsStruct(ctx, h_descr);
    if (!PyDataType_REFCHK(descr)) {
        return;
    }

    if (descr->type_num == NPY_OBJECT) {
        PyObject *temp;
        CAPI_WARN("leaked py refs in a PyArray");
        memcpy(&temp, data, sizeof(temp));
        Py_XDECREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
        HPy new; // PyArray_Descr *
        int offset;
        Py_ssize_t pos = 0;

        HPy fields = HPy_FromPyObject(ctx, descr->fields);
        HPy keys = HPyDict_Keys(ctx, fields);
        HPy_ssize_t keys_len = HPy_Length(ctx, keys);
        for (HPy_ssize_t i = 0; i < keys_len; i++) {
            HPy key = HPy_GetItem_i(ctx, keys, i);
            HPy value = HPy_GetItem(ctx, fields, key);
            if (HNPY_TITLE_KEY(ctx, key, value)) {
                HPy_Close(ctx, key);
                HPy_Close(ctx, value);
                return;
            }
            HPy_Close(ctx, key);
            if (!HPy_ExtractDictItems_OiO(ctx, value, &new, &offset, NULL)) {
                // error
                HPy_Close(ctx, value);
                HPy_Close(ctx, keys);
                HPy_Close(ctx, fields);
                return;
            }
            HPy_Close(ctx, value);
            HPyArray_Item_XDECREF(ctx, data + offset, new);
            HPy_Close(ctx, new);
        }
        HPy_Close(ctx, keys);
        HPy_Close(ctx, fields);
    }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        HPy h_base = HPy_FromPyObject(ctx, descr->subarray->base);
        for (i = 0; i < size; i++){
            /* Recursively decrement the reference count of subarray elements */
            HPyArray_Item_XDECREF(ctx, data + i * inner_elsize, h_base);
        }
        HPy_Close(ctx, h_base);
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}

/* Used for arrays of python objects to increment the reference count of */
/* every python object in the array. */
/*NUMPY_API
  For object arrays, increment all internal references.
*/
NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp)
{
    npy_intp i, n;
    PyObject **data;
    PyObject *temp;
    PyArrayIterObject *it;

    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    if (PyArray_DESCR(mp)->type_num != NPY_OBJECT) {
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            PyArray_Item_INCREF(it->dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)PyArray_DATA(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) {
                Py_XINCREF(*data);
            }
        }
        else {
            for( i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                Py_XINCREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            memcpy(&temp, it->dataptr, sizeof(temp));
            Py_XINCREF(temp);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

/*NUMPY_API
  Decrement all internal references for object arrays.
  (or arrays with object fields)
*/
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp)
{
    npy_intp i, n;
    PyObject **data;
    PyObject *temp;
    /*
     * statically allocating it allows this function to not modify the
     * reference count of the array for use during dealloc.
     * (statically is not necessary as such)
     */
    PyArrayIterObject it;

    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    if (PyArray_DESCR(mp)->type_num != NPY_OBJECT) {
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            PyArray_Item_XDECREF(it.dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(&it);
        }
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)PyArray_DATA(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) Py_XDECREF(*data);
        }
        else {
            for (i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                Py_XDECREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            memcpy(&temp, it.dataptr, sizeof(temp));
            Py_XDECREF(temp);
            PyArray_ITER_NEXT(&it);
        }
    }
    return 0;
}

/*NUMPY_API
 * Assumes contiguous
 */
NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj)
{
    npy_intp i,n;
    n = PyArray_SIZE(arr);
    if (PyArray_DESCR(arr)->type_num == NPY_OBJECT) {
        PyObject **optr;
        optr = (PyObject **)(PyArray_DATA(arr));
        n = PyArray_SIZE(arr);
        if (obj == NULL) {
            for (i = 0; i < n; i++) {
                *optr++ = NULL;
            }
        }
        else {
            for (i = 0; i < n; i++) {
                Py_INCREF(obj);
                *optr++ = obj;
            }
        }
    }
    else {
        char *optr;
        optr = PyArray_DATA(arr);
        for (i = 0; i < n; i++) {
            _fillobject(optr, obj, PyArray_DESCR(arr));
            optr += PyArray_DESCR(arr)->elsize;
        }
    }
}

/*HPY_NUMPY_API
 * Assumes contiguous
 */
NPY_NO_EXPORT void
HPyArray_FillObjectArray(HPyContext *ctx, HPy /* PyArrayObject * */ arr, HPy obj)
{
    npy_intp i,n;
    PyArrayObject *arr_struct = PyArrayObject_AsStruct(ctx, arr);
    n = PyArray_SIZE(arr_struct);
    HPy arr_descr = HPyArray_DESCR(ctx, arr, arr_struct);
    PyArray_Descr *arr_descr_struct = PyArray_Descr_AsStruct(ctx, arr_descr);
    if (arr_descr_struct->type_num == NPY_OBJECT) {
        PyObject **optr;
        optr = (PyObject **)(PyArray_DATA(arr_struct));
        n = PyArray_SIZE(arr_struct);
        if (HPy_IsNull(obj)) {
            for (i = 0; i < n; i++) {
                *optr++ = NULL;
            }
        }
        else {
            CAPI_WARN("leaking py refs in HPyArray_FillObjectArray");
            PyObject *py_obj = HPy_AsPyObject(ctx, obj);
            for (i = 0; i < n; i++) {
                Py_INCREF(py_obj);
                *optr++ = py_obj;
            }
            Py_DECREF(py_obj);
        }
    }
    else {
        char *optr;
        optr = PyArray_DATA(arr_struct);
        for (i = 0; i < n; i++) {
            _hpy_fillobject(ctx, optr, obj, arr_descr);
            optr += arr_descr_struct->elsize;
        }
    }
}

static NPY_INLINE int
setitem_trampoline(PyArray_SetItemFunc *func, PyObject *obj, char *data, PyArrayObject *arr)
{
    HPyContext *ctx = npy_get_context();
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPy h_arr = HPy_FromPyObject(ctx, (PyObject *)arr);
    int res = func(ctx, h_obj, data, h_arr);
    HPy_Close(ctx, h_arr);
    HPy_Close(ctx, h_obj);
    return res;
}

static void
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        PyObject *arr;

        if ((obj == Py_None) ||
                (PyLong_Check(obj) && PyLong_AsLong(obj) == 0)) {
            return;
        }
        /* Clear possible long conversion error */
        PyErr_Clear();
        Py_INCREF(dtype);
        arr = PyArray_NewFromDescr(&PyArray_Type, dtype,
                                   0, NULL, NULL, NULL,
                                   0, NULL);
        if (arr!=NULL) {
            setitem_trampoline(dtype->f->setitem, obj, optr, arr);
        }
        Py_XDECREF(arr);
    }
    if (dtype->type_num == NPY_OBJECT) {
        Py_XINCREF(obj);
        memcpy(optr, &obj, sizeof(obj));
    }
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            _fillobject(optr + offset, obj, new);
        }
    }
    else if (PyDataType_HASSUBARRAY(dtype)) {
        int size, i, inner_elsize;

        inner_elsize = dtype->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = dtype->elsize / inner_elsize;

        /* Call _fillobject on each item recursively. */
        for (i = 0; i < size; i++){
            _fillobject(optr, obj, dtype->subarray->base);
            optr += inner_elsize;
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}

static void
_hpy_fillobject(HPyContext *ctx, char *optr, HPy obj, HPy /* PyArray_Descr * */ dtype)
{
    PyArray_Descr *dtype_struct = PyArray_Descr_AsStruct(ctx, dtype);
    if (!PyDataType_FLAGCHK(dtype_struct, NPY_ITEM_REFCOUNT)) {
        HPy arr;

        if (HPy_Is(ctx, obj, ctx->h_None) ||
                (HPyLong_Check(ctx, obj) && HPyLong_AsLong(ctx, obj) == 0)) {
            return;
        }
        /* Clear possible long conversion error */
        HPyErr_Clear(ctx);
        // Py_INCREF(dtype);
        HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
        arr = HPyArray_NewFromDescr(ctx, array_type, dtype,
                                   0, NULL, NULL, NULL,
                                   0, HPy_NULL);
        HPy_Close(ctx, array_type);
        if (!HPy_IsNull(arr)) {
            dtype_struct->f->setitem(ctx, obj, optr, arr);
        }
        HPy_Close(ctx, arr);
    }
    if (dtype_struct->type_num == NPY_OBJECT) {
        // Py_XINCREF(obj);
        CAPI_WARN("leaking py refs in _hpy_fillobject");
        PyObject *py_obj = HPy_AsPyObject(ctx, obj);
        memcpy(optr, &py_obj, sizeof(py_obj));
    }
    else if (PyDataType_HASFIELDS(dtype_struct)) {
        HPy new; // PyArray_Descr *
        int offset;
        HPy fields = HPy_FromPyObject(ctx, dtype_struct->fields);
        HPy keys = HPyDict_Keys(ctx, fields);
        HPy_ssize_t keys_len = HPy_Length(ctx, keys);
        for (HPy_ssize_t i = 0; i < keys_len; i++) {
            HPy key = HPy_GetItem_i(ctx, keys, i);
            HPy value = HPy_GetItem(ctx, fields, key);
            if (HNPY_TITLE_KEY(ctx, key, value)) {
                HPy_Close(ctx, key);
                HPy_Close(ctx, value);
                return;
            }
            HPy_Close(ctx, key);
            if (!HPy_ExtractDictItems_OiO(ctx, value, &new, &offset, NULL)) {
                // error
                HPy_Close(ctx, value);
                HPy_Close(ctx, keys);
                HPy_Close(ctx, fields);
                return;
            }
            HPy_Close(ctx, value);
            _hpy_fillobject(ctx, optr + offset, obj, new);
            HPy_Close(ctx, new);
        }
        HPy_Close(ctx, keys);
        HPy_Close(ctx, fields);
    }
    else if (PyDataType_HASSUBARRAY(dtype_struct)) {
        int size, i, inner_elsize;

        inner_elsize = dtype_struct->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = dtype_struct->elsize / inner_elsize;

        /* Call _hpy_fillobject on each item recursively. */
        for (i = 0; i < size; i++){
            HPy h_base = HPy_FromPyObject(ctx, dtype_struct->subarray->base);
            _hpy_fillobject(ctx, optr, obj, h_base);
            optr += inner_elsize;
            HPy_Close(ctx, h_base);
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}

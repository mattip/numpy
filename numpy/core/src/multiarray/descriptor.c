/* Array Descr Object */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"
#include "npy_ctypes.h"
#include "npy_pycompat.h"

#include "_datetime.h"
#include "common.h"
#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "descriptor.h"
#include "alloc.h"
#include "assert.h"
#include "npy_buffer.h"
#include "dtypemeta.h"
#include "hashdescr.h"

/*
 * offset:    A starting offset.
 * alignment: A power-of-two alignment.
 *
 * This macro returns the smallest value >= 'offset'
 * that is divisible by 'alignment'. Because 'alignment'
 * is a power of two and integers are twos-complement,
 * it is possible to use some simple bit-fiddling to do this.
 */
#define NPY_NEXT_ALIGNED_OFFSET(offset, alignment) \
                (((offset) + (alignment) - 1) & (-(alignment)))

#ifndef PyDictProxy_Check
#define PyDictProxy_Check(obj) (Py_TYPE(obj) == &PyDictProxy_Type)
#endif

HPyGlobal descr_typeDict;   /* Must be explicitly loaded */

static HPy // PyArray_Descr *
_hpy_try_convert_from_inherit_tuple(HPyContext *ctx, 
                                        HPy /* PyArray_Descr * */ type, HPy newobj);

static HPy
_hpy_convert_from_any(HPyContext *ctx, HPy obj, int align);

/*
 * This function creates a dtype object when the object is a ctypes subclass.
 *
 * Returns `Py_NotImplemented` if the type is not a ctypes subclass.
 */
static HPy // PyArray_Descr *
_hpy_try_convert_from_ctypes_type(HPyContext *ctx, HPy /* PyTypeObject * */ type)
{
    HPy _numpy_dtype_ctypes;
    HPy res; // PyArray_Descr *

    if (!hpy_npy_ctypes_check(ctx, type)) {
        // Py_INCREF(Py_NotImplemented);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }

    /* Call the python function of the same name. */
    _numpy_dtype_ctypes = HPyImport_ImportModule(ctx, "numpy.core._dtype_ctypes");
    if (HPy_IsNull(_numpy_dtype_ctypes)) {
        return HPy_NULL;
    }
    HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
    HPy args = HPyTuple_Pack(ctx, 1, type);
    HPy meth = HPy_GetAttr_s(ctx, _numpy_dtype_ctypes, "dtype_from_ctypes_type");
    res = HPy_CallTupleDict(ctx, meth, args, HPy_NULL);
    HPy_Close(ctx, args);
    HPy_Close(ctx, meth);
    HPy_Close(ctx, _numpy_dtype_ctypes);
    if (HPy_IsNull(res)) {
        return HPy_NULL;
    }

    /*
     * sanity check that dtype_from_ctypes_type returned the right type,
     * since getting it wrong would give segfaults.
     */
    HPy arraydescr_type = HPyGlobal_Load(ctx, HPyArrayDescr_Type);
    if (!HPy_TypeCheck(ctx, res, arraydescr_type)) {
        HPy_Close(ctx, arraydescr_type);
        HPy_Close(ctx, res);
        CAPI_WARN("missing PyErr_BadInternalCall");
        PyErr_BadInternalCall();
        return HPy_NULL;
    }
    HPy_Close(ctx, arraydescr_type);

    return res;
}


static HPy // PyArray_Descr *
_hpy_convert_from_any(HPyContext *ctx, HPy obj, int align);

/*
 * This function creates a dtype object when the object has a "dtype" attribute,
 * and it can be converted to a dtype object.
 *
 * Returns `Py_NotImplemented` if this is not possible.
 * Currently the only failure mode for a NULL return is a RecursionError.
 */
static HPy // PyArray_Descr *
_hpy_try_convert_from_dtype_attr(HPyContext *ctx, HPy obj)
{
    /* For arbitrary objects that have a "dtype" attribute */
    HPy dtypedescr = HPy_GetAttr_s(ctx, obj, "dtype");
    // HPy dtypedescr = HPyArray_GetDescr(ctx, obj);
    if (HPy_IsNull(dtypedescr)) {
        /*
         * This can be reached due to recursion limit being hit while fetching
         * the attribute (tested for py3.7). This removes the custom message.
         */
        goto fail;
    }

    if (HPyArray_DescrCheck(ctx, dtypedescr)) {
        /* The dtype attribute is already a valid descriptor */
        return dtypedescr;
    }

    CAPI_WARN("missing Py_EnterRecursiveCall");
    // if (Py_EnterRecursiveCall(
    //         " while trying to convert the given data type from its "
    //         "`.dtype` attribute.") != 0) {
    //     HPy_Close(ctx, dtypedescr);
    //     return HPy_NULL;
    // }

    HPy newdescr = _hpy_convert_from_any(ctx, dtypedescr, 0); // PyArray_Descr *
    HPy_Close(ctx, dtypedescr);
    // Py_LeaveRecursiveCall();
    if (HPy_IsNull(newdescr)) {
        goto fail;
    }

    /* Deprecated 2021-01-05, NumPy 1.21 */
    if (HPY_DEPRECATE(ctx, "in the future the `.dtype` attribute of a given data"
                  "type object must be a valid dtype instance. "
                  "`data_type.dtype` may need to be coerced using "
                  "`np.dtype(data_type.dtype)`. (Deprecated NumPy 1.20)") < 0) {
        HPy_Close(ctx, newdescr);
        return HPy_NULL;
    }

    return newdescr;

  fail:
    /* Ignore all but recursion errors, to give ctypes a full try. */
    if (!HPyErr_ExceptionMatches(ctx, ctx->h_RecursionError)) {
        HPyErr_Clear(ctx);
        // Py_INCREF(Py_NotImplemented);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    return HPy_NULL;
}

/* Expose to another file with a prefixed name */
NPY_NO_EXPORT PyArray_Descr *
_arraydescr_try_convert_from_dtype_attr(PyObject *obj)
{
    HPyContext *ctx = npy_get_context();
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPy ret = _hpy_try_convert_from_dtype_attr(ctx, h_obj);
    PyArray_Descr *py_ret = (PyArray_Descr *)HPy_AsPyObject(ctx, ret);
    HPy_Close(ctx, h_obj);
    HPy_Close(ctx, ret);
    return py_ret;
}

NPY_NO_EXPORT HPy // PyArray_Descr *
_hpy_arraydescr_try_convert_from_dtype_attr(HPyContext *ctx, HPy obj)
{
    return _hpy_try_convert_from_dtype_attr(ctx, obj);
}

/*
 * Sets the global typeDict object, which is a dictionary mapping
 * dtype names to numpy scalar types.
 */
HPyDef_METH(array_set_typeDict, "set_typeDict", HPyFunc_VARARGS)
NPY_NO_EXPORT HPy
array_set_typeDict_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), const HPy *args, size_t nargs)
{
    HPy dict;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O:set_typeDict", &dict)) {
        return HPy_NULL;
    }
    /* Decrement old reference (if any)*/
    HPyGlobal_Store(ctx, &descr_typeDict, dict);
    // typeDict = dict;
    /* Create an internal reference to it */
    // Py_INCREF(dict);
    return HPy_Dup(ctx, ctx->h_None);
}

#define _chk_byteorder(arg) (arg == '>' || arg == '<' ||        \
                             arg == '|' || arg == '=')

static int
_check_for_commastring(const char *type, Py_ssize_t len)
{
    Py_ssize_t i;
    int sqbracket;

    /* Check for ints at start of string */
    if ((type[0] >= '0'
                && type[0] <= '9')
            || ((len > 1)
                && _chk_byteorder(type[0])
                && (type[1] >= '0'
                && type[1] <= '9'))) {
        return 1;
    }
    /* Check for empty tuple */
    if (((len > 1)
                && (type[0] == '('
                && type[1] == ')'))
            || ((len > 3)
                && _chk_byteorder(type[0])
                && (type[1] == '('
                && type[2] == ')'))) {
        return 1;
    }
    /*
     * Check for presence of commas outside square [] brackets. This
     * allows commas inside of [], for parameterized dtypes to use.
     */
    sqbracket = 0;
    for (i = 0; i < len; i++) {
        switch (type[i]) {
            case ',':
                if (sqbracket == 0) {
                    return 1;
                }
                break;
            case '[':
                ++sqbracket;
                break;
            case ']':
                --sqbracket;
                break;
        }
    }
    return 0;
}

#undef _chk_byteorder

static int
is_datetime_typestr(char const *type, Py_ssize_t len)
{
    if (len < 2) {
        return 0;
    }
    if (type[1] == '8' && (type[0] == 'M' || type[0] == 'm')) {
        return 1;
    }
    if (len < 10) {
        return 0;
    }
    if (strncmp(type, "datetime64", 10) == 0) {
        return 1;
    }
    if (len < 11) {
        return 0;
    }
    if (strncmp(type, "timedelta64", 11) == 0) {
        return 1;
    }
    return 0;
}

static HPy // PyArray_Descr *
_hpy_convert_from_tuple(HPyContext *ctx, HPy obj, int align)
{
    if (HPy_Length(ctx, obj) != 2) {
        HPyErr_Format_p(ctx, ctx->h_TypeError,
	        "Tuple must have size 2, but has size %zd",
	        HPy_Length(ctx, obj));
        return HPy_NULL;
    }
    HPy item = HPy_GetItem_i(ctx, obj, 0);
    HPy type = _hpy_convert_from_any(ctx, item, align); // PyArray_Descr *
    HPy_Close(ctx, item);
    if (HPy_IsNull(type)) {
        return HPy_NULL;
    }
    HPy val = HPy_GetItem_i(ctx, obj, 1);
    /* try to interpret next item as a type */
    HPy res = _hpy_try_convert_from_inherit_tuple(ctx, type, val);
    if (!HPy_Is(ctx, res, ctx->h_NotImplemented)) {
        HPy_Close(ctx, val);
        HPy_Close(ctx, type);
        return res;
    }
    HPy_Close(ctx, res);
    /*
     * We get here if _try_convert_from_inherit_tuple failed without crashing
     */
    PyObject *py_val = HPy_AsPyObject(ctx, val);
    PyArray_Descr *type_struct = PyArray_Descr_AsStruct(ctx, type);
    if (PyDataType_ISUNSIZED(type_struct)) {
        Py_DECREF(py_val);
        /* interpret next item as a typesize */
        int itemsize = HPyArray_PyIntAsInt(ctx, val);
        HPy_Close(ctx, val);

        if (hpy_error_converting(ctx, itemsize)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "invalid itemsize in generic type tuple");
            HPy_Close(ctx, type);
            return HPy_NULL;
        }
        HPyArray_DESCR_REPLACE(ctx, type, type_struct);
        if (HPy_IsNull(type)) {
            return HPy_NULL;
        }
        if (type_struct->type_num == NPY_UNICODE) {
            type_struct->elsize = itemsize << 2;
        }
        else {
            type_struct->elsize = itemsize;
        }
        return type;
    }
    else if (type_struct->metadata && (HPyDict_Check(ctx, val) || PyDictProxy_Check(py_val))) {
        /* Assume it's a metadata dictionary */
        CAPI_WARN("missing PyDictProxy_Check & PyDict_Merge");
        if (PyDict_Merge(type_struct->metadata, py_val, 0) == -1) {
            HPy_Close(ctx, val);
            HPy_Close(ctx, type);
            return HPy_NULL;
        }
        
        return type;
    }
    else {
        /*
         * interpret next item as shape (if it's a tuple)
         * and reset the type to NPY_VOID with
         * a new fields attribute.
         */
        PyArray_Dims shape = {NULL, -1};
        if (!(HPyArray_IntpConverter(ctx, val, &shape)) || (shape.len > NPY_MAXDIMS)) {
            HPy_Close(ctx, val);
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "invalid shape in fixed-type tuple.");
            goto fail;
        }
        /* if (type, ()) was given it is equivalent to type... */
        if (shape.len == 0 && HPyTuple_Check(ctx, val)) {
            HPy_Close(ctx, val);
            npy_free_cache_dim_obj(shape);
            return type;
        }
        /* (type, 1) use to be equivalent to type, but is deprecated */
        if (shape.len == 1
                && shape.ptr[0] == 1
                && HPyNumber_Check(ctx, val)) {
            HPy_Close(ctx, val);
            /* 2019-05-20, 1.17 */
            if (HPY_DEPRECATE_FUTUREWARNING(ctx,
                        "Passing (type, 1) or '1type' as a synonym of type is "
                        "deprecated; in a future version of numpy, it will be "
                        "understood as (type, (1,)) / '(1,)type'.") < 0) {
                goto fail;
            }
            npy_free_cache_dim_obj(shape);
            return type;
        }
        HPy_Close(ctx, val);

        /* validate and set shape */
        for (int i=0; i < shape.len; i++) {
            if (shape.ptr[i] < 0) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                                "invalid shape in fixed-type tuple: "
                                "dimension smaller then zero.");
                goto fail;
            }
            if (shape.ptr[i] > NPY_MAX_INT) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                                "invalid shape in fixed-type tuple: "
                                "dimension does not fit into a C int.");
                goto fail;
            }
        }
        npy_intp items = PyArray_OverflowMultiplyList(shape.ptr, shape.len);
        int overflowed;
        int nbytes;
        if (items < 0 || items > NPY_MAX_INT) {
            overflowed = 1;
        }
        else {
            overflowed = npy_mul_with_overflow_int(
                &nbytes, type_struct->elsize, (int) items);
        }
        if (overflowed) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "invalid shape in fixed-type tuple: dtype size in "
                            "bytes must fit into a C int.");
            goto fail;
        }
        HPy newdescr = HPyArray_DescrNewFromType(ctx, NPY_VOID); // PyArray_Descr *
        if (HPy_IsNull(newdescr)) {
            goto fail;
        }
        PyArray_Descr *newdescr_struct = PyArray_Descr_AsStruct(ctx, newdescr);
        newdescr_struct->elsize = nbytes;
        newdescr_struct->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (newdescr_struct->subarray == NULL) {
            HPy_Close(ctx, newdescr);
            HPyErr_NoMemory(ctx);
            goto fail;
        }
        newdescr_struct->flags = type_struct->flags;
        newdescr_struct->alignment = type_struct->alignment;
        newdescr_struct->subarray->base = HPy_AsPyObject(ctx, type);
        HPy_Close(ctx, type);
        Py_XDECREF(newdescr_struct->fields);
        HPyField_Store(ctx, newdescr, &newdescr_struct->names, HPy_NULL);
        newdescr_struct->fields = NULL;
        newdescr_struct->names = HPyField_NULL;

        /*
         * Create a new subarray->shape tuple (it can be an arbitrary
         * sequence of integer like objects, neither of which is safe.
         */
        HPyTupleBuilder shape_tb = HPyTupleBuilder_New(ctx, shape.len);
        if (HPyTupleBuilder_IsNull(shape_tb)) {
            HPy_Close(ctx, newdescr);
            goto fail;
        }
        for (int i=0; i < shape.len; i++) {
            HPy item = HPyLong_FromLong(ctx, (long)shape.ptr[i]);
            HPyTupleBuilder_Set(ctx, shape_tb, i, item);
            HPy_Close(ctx, item);
        }
        HPy h_shape = HPyTupleBuilder_Build(ctx, shape_tb);
        newdescr_struct->subarray->shape = HPy_AsPyObject(ctx, h_shape);
        HPy_Close(ctx, h_shape);

        npy_free_cache_dim_obj(shape);
        return newdescr;

    fail:
        HPy_Close(ctx, type);
        npy_free_cache_dim_obj(shape);
        return HPy_NULL;
    }
}

/*
 * obj is a list.  Each item is a tuple with
 *
 * (field-name, data-type (either a list or a string), and an optional
 * shape parameter).
 *
 * field-name can be a string or a 2-tuple
 * data-type can now be a list, string, or 2-tuple
 *          (string, metadata dictionary)
 */
static HPy // PyArray_Descr *
_hpy_convert_from_array_descr(HPyContext *ctx, HPy obj, int align)
{
    int n = HPy_Length(ctx, obj);
    HPyTupleBuilder nameslist = HPyTupleBuilder_New(ctx, n);
    if (HPyTupleBuilder_IsNull(nameslist)) {
        return HPy_NULL;
    }

    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;
    int maxalign = 0;
    int totalsize = 0;
    HPy fields = HPyDict_New(ctx);
    if (HPy_IsNull(fields)) {
        return HPy_NULL;
    }
    for (int i = 0; i < n; i++) {
        HPy item = HPy_GetItem_i(ctx, obj, i);
        if (!HPyTuple_Check(ctx, item) || HPy_Length(ctx, item) < 2) {
            // PyErr_Format(PyExc_TypeError,
			//  "Field elements must be 2- or 3-tuples, got '%R'",
			//  item);
            HPyErr_SetString(ctx, ctx->h_TypeError,
			 "Field elements must be 2- or 3-tuples, got '%R'");
            goto fail;
        }
        HPy name = HPy_GetItem_i(ctx, item, 0);
        HPy title;
        if (HPyUnicode_Check(ctx, name)) {
            title = HPy_NULL;
        }
        else if (HPyTuple_Check(ctx, name)) {
            HPy_ssize_t name_len = HPy_Length(ctx,name);
            if (name_len != 2) {
                HPyErr_Format_p(ctx, ctx->h_TypeError,
				"If a tuple, the first element of a field tuple must have "
				"two elements, not %zd",
			       	name_len);
                goto fail;
            }
            title = HPy_GetItem_i(ctx, name, 0);
            HPy t = name;
            name = HPy_GetItem_i(ctx, name, 1);
            HPy_Close(ctx, t);
            if (!HPyUnicode_Check(ctx, name)) {
                HPyErr_SetString(ctx, ctx->h_TypeError, "Field name must be a str");
                goto fail;
            }
        }
        else {
            HPyErr_SetString(ctx, ctx->h_TypeError,
			            "First element of field tuple is "
			            "neither a tuple nor str");
            goto fail;
        }

        /* Insert name into nameslist */
        // Py_INCREF(name);

        if (HPy_Length(ctx, name) == 0) {
            HPy_Close(ctx, name);
            if (HPy_IsNull(title)) {
                name = HPyUnicode_FromFormat_p(ctx, "f%d", i);
                if (HPy_IsNull(name)) {
                    goto fail;
                }
            }
            /* On Py3, allow only non-empty Unicode strings as field names */
            else if (HPyUnicode_Check(ctx, title) && HPy_Length(ctx, title) > 0) {
                name = title;
                // Py_INCREF(name); // no need
            }
            else {
                HPyErr_SetString(ctx, ctx->h_TypeError, "Field titles must be non-empty strings");
                goto fail;
            }
        }
        HPyTupleBuilder_Set(ctx, nameslist, i, name);

        /* Process rest */
        HPy conv; // PyArray_Descr *
        HPy_ssize_t item_len = HPy_Length(ctx, item);
        if (item_len == 2) {
            HPy h_item = HPy_GetItem_i(ctx, item, 1);
            conv = _hpy_convert_from_any(ctx, h_item, align);
            HPy_Close(ctx, h_item);
            if (HPy_IsNull(conv)) {
                goto fail;
            }
        }
        else if (item_len == 3) {
            CAPI_WARN("missing PyTuple_GetSlice");
            PyObject *py_item = HPy_AsPyObject(ctx, item);
            PyObject *py_newobj = PyTuple_GetSlice(py_item, 1, 3);
            Py_DECREF(py_item);
            HPy newobj = HPy_FromPyObject(ctx, py_newobj);
            Py_DECREF(py_newobj);
            conv = _hpy_convert_from_any(ctx, newobj, align);
            HPy_Close(ctx, newobj);
            if (HPy_IsNull(conv)) {
                goto fail;
            }
        }
        else {
            // PyErr_Format(PyExc_TypeError,
            //         "Field elements must be tuples with at most 3 elements, got '%R'", item);
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "Field elements must be tuples with at most 3 elements, got '%R'");
            goto fail;
        }
        if (!(HPyDict_GetItemWithError_IsNull(ctx, fields, name))
             || (!HPy_IsNull(title)
                 && HPyUnicode_Check(ctx, title)
                 && !(HPyDict_GetItemWithError_IsNull(ctx, fields, title)))) {
            // PyErr_Format(PyExc_ValueError,
            //         "field %R occurs more than once", name);
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "field %R occurs more than once");
            HPy_Close(ctx, conv);
            goto fail;
        }
        else if (HPyErr_Occurred(ctx)) {
            /* Dict lookup crashed */
            HPy_Close(ctx, conv);
            goto fail;
        }
        PyArray_Descr *conv_struct = PyArray_Descr_AsStruct(ctx, conv);
        dtypeflags |= (conv_struct->flags & NPY_FROM_FIELDS);
        if (align) {
            int _align = conv_struct->alignment;
            if (_align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            maxalign = PyArray_MAX(maxalign, _align);
        }

        HPy h_totalsize = HPyLong_FromLong(ctx, (long) totalsize);

        /*
         * Title can be "meta-data".  Only insert it
         * into the fields dictionary if it is a string
         * and if it is not the same as the name.
         */
        if (!HPy_IsNull(title)) {
            HPy tup = HPyTuple_Pack(ctx, 3, conv, h_totalsize, title);
            if (HPy_SetItem(ctx, fields, name, tup) < 0) {
                HPy_Close(ctx, conv);
                HPy_Close(ctx, h_totalsize);
                HPy_Close(ctx, title);
                HPy_Close(ctx, tup);
                goto fail;
            }
            if (HPyUnicode_Check(ctx, title)) {
                HPy existing = HPyDict_GetItemWithError(ctx, fields, title);
                if (HPy_IsNull(existing) && HPyErr_Occurred(ctx)) {
                    HPy_Close(ctx, conv);
                    HPy_Close(ctx, h_totalsize);
                    HPy_Close(ctx, title);
                    HPy_Close(ctx, tup);
                    goto fail;
                }
                if (!HPy_IsNull(existing)) {
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                            "title already used as a name or title.");
                    HPy_Close(ctx, existing);
                    HPy_Close(ctx, conv);
                    HPy_Close(ctx, h_totalsize);
                    HPy_Close(ctx, title);
                    HPy_Close(ctx, tup);
                    goto fail;
                }
                if (HPy_SetItem(ctx, fields, title, tup) < 0) {
                    HPy_Close(ctx, conv);
                    HPy_Close(ctx, h_totalsize);
                    HPy_Close(ctx, title);
                    HPy_Close(ctx, tup);
                    goto fail;
                }
            }
            HPy_Close(ctx, conv);
            HPy_Close(ctx, h_totalsize);
            HPy_Close(ctx, title);
            HPy_Close(ctx, tup);
        }
        else {
            HPy tup = HPyTuple_Pack(ctx, 2, conv, h_totalsize);
            int r = HPy_SetItem(ctx, fields, name, tup);
            HPy_Close(ctx, conv);
            HPy_Close(ctx, h_totalsize);
            HPy_Close(ctx, tup);
            if (r < 0) {
                goto fail;
            }
        }

        totalsize += conv_struct->elsize;
        // Py_DECREF(tup);
    }

    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }

    HPy new = HPyArray_DescrNewFromType(ctx, NPY_VOID);
    if (HPy_IsNull(new)) {
        goto fail;
    }
    PyArray_Descr *new_struct = PyArray_Descr_AsStruct(ctx, new);
    new_struct->fields = HPy_AsPyObject(ctx, fields);
    HPyField_Store(ctx, new, &new_struct->names, HPyTupleBuilder_Build(ctx, nameslist));
    // HPy_Close(ctx, nameslist);
    new_struct->elsize = totalsize;
    new_struct->flags = dtypeflags;

    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new_struct->flags |= NPY_ALIGNED_STRUCT;
        new_struct->alignment = maxalign;
    }
    return new;

 fail:
    HPy_Close(ctx, fields);
    HPyTupleBuilder_Cancel(ctx, nameslist);
    return HPy_NULL;

}

/*
 * a list specifying a data-type can just be
 * a list of formats.  The names for the fields
 * will default to f0, f1, f2, and so forth.
 */
static HPy // PyArray_Descr *
_hpy_convert_from_list(HPyContext *ctx, HPy obj, int align)
{
    int n = HPy_Length(ctx, obj);
    /*
     * Ignore any empty string at end which _internal._commastring
     * can produce
     */
    HPy last_item = HPy_GetItem_i(ctx, obj, n-1);
    if (HPyUnicode_Check(ctx, last_item)) {
        HPy_ssize_t s = HPy_Length(ctx, last_item);
        if (s < 0) {
            return HPy_NULL;
        }
        if (s == 0) {
            n = n - 1;
        }
    }
    if (n == 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Expected at least one field name");
        return HPy_NULL;
    }
    HPyTupleBuilder nameslist = HPyTupleBuilder_New(ctx, n);
    if (HPyTupleBuilder_IsNull(nameslist)) {
        return HPy_NULL;
    }
    HPy fields = HPyDict_New(ctx);
    if (HPy_IsNull(fields)) {
        HPyTupleBuilder_Cancel(ctx, nameslist);
        return HPy_NULL;
    }

    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;
    int maxalign = 0;
    int totalsize = 0;
    for (int i = 0; i < n; i++) {
        HPy item = HPy_GetItem_i(ctx, obj, i);
        HPy conv = _hpy_convert_from_any(ctx, item, align); // PyArray_Descr *
        HPy_Close(ctx, item);
        if (HPy_IsNull(conv)) {
            goto fail;
        }
        PyArray_Descr *conv_struct = PyArray_Descr_AsStruct(ctx, conv);
        dtypeflags |= (conv_struct->flags & NPY_FROM_FIELDS);
        if (align) {
            int _align = conv_struct->alignment;
            if (_align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            maxalign = PyArray_MAX(maxalign, _align);
        }
        HPy size_obj = HPyLong_FromLong(ctx, (long) totalsize);
        if (HPy_IsNull(size_obj)) {
            HPy_Close(ctx, conv);
            goto fail;
        }
        HPy tup = HPyTuple_Pack(ctx, 2, conv, size_obj);
        HPy_Close(ctx, size_obj);
        HPy_Close(ctx, conv);
        if (HPy_IsNull(tup)) {
            goto fail;
        }
        HPy key = HPyUnicode_FromFormat_p(ctx, "f%d", i);
        if (HPy_IsNull(key)) {
            HPy_Close(ctx, tup);
            goto fail;
        }
        /* do not steal a reference to key */
        HPyTupleBuilder_Set(ctx, nameslist, i, key);
        int ret = HPy_SetItem(ctx, fields, key, tup);
        HPy_Close(ctx, key);
        HPy_Close(ctx, tup);
        if (ret < 0) {
            goto fail;
        }
        totalsize += conv_struct->elsize;
    }
    HPy new = HPyArray_DescrNewFromType(ctx, NPY_VOID);
    if (HPy_IsNull(new)) {
        goto fail;
    }
    PyArray_Descr *new_struct = PyArray_Descr_AsStruct(ctx, new);
    new_struct->fields = HPy_AsPyObject(ctx, fields);
    HPy_Close(ctx, fields);
    HPyField_Store(ctx, new, &new_struct->names, HPyTupleBuilder_Build(ctx, nameslist));
    
    new_struct->flags = dtypeflags;
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }
    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new_struct->flags |= NPY_ALIGNED_STRUCT;
        new_struct->alignment = maxalign;
    }
    new_struct->elsize = totalsize;
    return new;

 fail:
    HPyTupleBuilder_Cancel(ctx, nameslist);
    HPy_Close(ctx, fields);
    return HPy_NULL;
}

/*
 * comma-separated string
 * this is the format developed by the numarray records module and implemented
 * by the format parser in that module this is an alternative implementation
 * found in the _internal.py file patterned after that one -- the approach is
 * to try to convert to a list (with tuples if any repeat information is
 * present) and then call the _convert_from_list)
 *
 * TODO: Calling Python from C like this in critical-path code is not
 *       a good idea. This should all be converted to C code.
 */
static HPy // PyArray_Descr *
_hpy_convert_from_commastring(HPyContext *ctx, HPy obj, int align)
{
    assert(HPyUnicode_Check(ctx, obj));
    HPy _numpy_internal = HPyImport_ImportModule(ctx, "numpy.core._internal");
    if (HPy_IsNull(_numpy_internal)) {
        return HPy_NULL;
    }
    HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
    HPy args = HPyTuple_Pack(ctx, 1, obj);
    HPy meth = HPy_GetAttr_s(ctx, _numpy_internal, "_commastring");
    HPy listobj = HPy_CallTupleDict(ctx, meth, args, HPy_NULL);
    HPy_Close(ctx, args);
    HPy_Close(ctx, meth);
    HPy_Close(ctx, _numpy_internal);
    if (HPy_IsNull(listobj)) {
        return HPy_NULL;
    }
    if (!HPyList_Check(ctx, listobj) || HPy_Length(ctx, listobj) < 1) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "_commastring is not returning a list with len >= 1");
        HPy_Close(ctx, listobj);
        return HPy_NULL;
    }
    HPy res; // PyArray_Descr *
    if (HPy_Length(ctx, listobj) == 1) {
        HPy item = HPy_GetItem_i(ctx, listobj, 0);
        res = _hpy_convert_from_any(ctx, item, align);
        HPy_Close(ctx, item);
    }
    else {
        res = _hpy_convert_from_list(ctx, listobj, align);
    }
    HPy_Close(ctx, listobj);
    return res;
}

static int
_hpy_is_tuple_of_integers(HPyContext *ctx, HPy obj)
{
    int i;

    if (!HPyTuple_Check(ctx, obj)) {
        return 0;
    }
    for (i = 0; i < HPy_Length(ctx, obj); i++) {
        HPy item = HPy_GetItem_i(ctx, obj, i);
        if (!HPyArray_IsIntegerScalar(ctx, item)) {
            HPy_Close(ctx, item);
            return 0;
        }
        HPy_Close(ctx, item);
    }
    return 1;
}

/*
 * helper function for _try_convert_from_inherit_tuple to disallow dtypes of the form
 * (old_dtype, new_dtype) where either of the dtypes contains python
 * objects - these dtypes are not useful and can be a source of segfaults,
 * when an attempt is made to interpret a python object as a different dtype
 * or vice versa
 * an exception is made for dtypes of the form ('O', [('name', 'O')]), which
 * people have been using to add a field to an object array without fields
 */
static int
_hpy_validate_union_object_dtype(HPyContext *ctx, PyArray_Descr *new_struct, 
                                    HPy conv, PyArray_Descr *conv_struct)
{
    HPy name, tup;
    HPy dtype; // PyArray_Descr *

    if (!PyDataType_REFCHK(new_struct) && !PyDataType_REFCHK(conv_struct)) {
        return 0;
    }
    if (PyDataType_HASFIELDS(new_struct) || new_struct->kind != 'O') {
        goto fail;
    }
    HPy names = HPyField_Load(ctx, conv, conv_struct->names);
    if (!PyDataType_HASFIELDS(conv_struct) || HPy_Length(ctx, names) != 1) {
        HPy_Close(ctx, names);
        goto fail;
    }
    name = HPy_GetItem_i(ctx, names, 0);
    HPy_Close(ctx, names);
    if (HPy_IsNull(name)) {
        return -1;
    }
    HPy fields = HPy_FromPyObject(ctx, conv_struct->fields);
    tup = HPyDict_GetItemWithError(ctx, fields, name);
    HPy_Close(ctx, name);
    HPy_Close(ctx, fields);
    if (HPy_IsNull(tup)) {
        if (!HPyErr_Occurred(ctx)) {
            /* fields was missing the name it claimed to contain */
            PyErr_BadInternalCall();
        }
        return -1;
    }
    dtype = HPy_GetItem_i(ctx, tup, 0);
    if (HPy_IsNull(dtype)) {
        return -1;
    }
    if (PyArray_Descr_AsStruct(ctx, dtype)->kind != 'O') {
        goto fail;
    }
    return 0;

fail:
    HPyErr_SetString(ctx, ctx->h_ValueError,
            "dtypes of the form (old_dtype, new_dtype) containing the object "
            "dtype are not supported");
    return -1;
}

/*
 * A tuple type would be either (generic typeobject, typesize)
 * or (fixed-length data-type, shape)
 *
 * or (inheriting data-type, new-data-type)
 * The new data-type must have the same itemsize as the inheriting data-type
 * unless the latter is 0
 *
 * Thus (int32, {'real':(int16,0),'imag',(int16,2)})
 *
 * is one way to specify a descriptor that will give
 * a['real'] and a['imag'] to an int32 array.
 *
 * leave type reference alone
 *
 * Returns `Py_NotImplemented` if the second tuple item is not
 * appropriate.
 */
static HPy // PyArray_Descr *
_hpy_try_convert_from_inherit_tuple(HPyContext *ctx, 
                                        HPy /* PyArray_Descr * */ type, HPy newobj)
{
    if (HPyArray_IsScalar(ctx, newobj, Integer) || _hpy_is_tuple_of_integers(ctx, newobj)) {
        /* It's a subarray or flexible type instead */
        // Py_INCREF(Py_NotImplemented);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    HPy conv = _hpy_convert_from_any(ctx, newobj, 0); // PyArray_Descr *
    if (HPy_IsNull(conv)) {
        /* Let someone else try to convert this */
        HPyErr_Clear(ctx);
        // Py_INCREF(Py_NotImplemented);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
    HPy new = HPyArray_DescrNew(ctx, type); // PyArray_Descr *
    if (HPy_IsNull(new)) {
        goto fail;
    }
    PyArray_Descr *new_struct = PyArray_Descr_AsStruct(ctx, new);
    PyArray_Descr *conv_struct = PyArray_Descr_AsStruct(ctx, conv);
    if (PyDataType_ISUNSIZED(new_struct)) {
        new_struct->elsize = conv_struct->elsize;
    }
    else if (new_struct->elsize != conv_struct->elsize) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "mismatch in size of old and new data-descriptor");
        HPy_Close(ctx, new);
        goto fail;
    }
    else if (_hpy_validate_union_object_dtype(ctx, new_struct, conv, conv_struct) < 0) {
        HPy_Close(ctx, new);
        goto fail;
    }

    if (PyDataType_HASFIELDS(conv_struct)) {
        Py_XDECREF(new_struct->fields);
        new_struct->fields = conv_struct->fields;
        Py_XINCREF(new_struct->fields);

        HPy conv_names = HPyField_Load(ctx, conv, conv_struct->names);
        HPyField_Store(ctx, new, &new_struct->names, conv_names);
        HPy_Close(ctx, conv_names);
    }
    if (conv_struct->metadata != NULL) {
        Py_XDECREF(new_struct->metadata);
        new_struct->metadata = conv_struct->metadata;
        Py_XINCREF(new_struct->metadata);
    }
    /*
     * Certain flags must be inherited from the fields.  This is needed
     * only for void dtypes (or subclasses of it such as a record dtype).
     * For other dtypes, the field part will only be used for direct field
     * access and thus flag inheritance should not be necessary.
     * (We only allow object fields if the dtype is object as well.)
     * This ensures copying over of the NPY_FROM_FIELDS "inherited" flags.
     */
    if (new_struct->type_num == NPY_VOID) {
        new_struct->flags = conv_struct->flags;
    }
    HPy_Close(ctx, conv);
    return new;

 fail:
    HPy_Close(ctx, conv);
    return HPy_NULL;
}

/*
 * Validates that any field of the structured array 'dtype' which has
 * the NPY_ITEM_HASOBJECT flag set does not overlap with another field.
 *
 * This algorithm is worst case O(n^2). It could be done with a sort
 * and sweep algorithm, but the structured dtype representation is
 * rather ugly right now, so writing something better can wait until
 * that representation is made sane.
 *
 * Returns 0 on success, -1 if an exception is raised.
 */
static int
_hpy_validate_object_field_overlap(HPyContext *ctx, HPy dtype, PyArray_Descr *dtype_struct)
{
    HPy names, fields, key, tup;
    HPy_ssize_t i, j, names_size;
    HPy fld_dtype, fld2_dtype; // PyArray_Descr *
    int fld_offset, fld2_offset;
    int res;

    /* Get some properties from the dtype */
    names = HPyField_Load(ctx, dtype, dtype_struct->names);
    names_size = HPy_Length(ctx, names);
    fields = HPy_FromPyObject(ctx, dtype_struct->fields);

    for (i = 0; i < names_size; ++i) {
        key = HPy_GetItem_i(ctx, names, i);
        if (HPy_IsNull(key)) {
            res = -1;
            goto finish;
        }
        tup = HPyDict_GetItemWithError(ctx, fields, key);
        HPy_Close(ctx, key);
        if (HPy_IsNull(tup)) {
            if (!HPyErr_Occurred(ctx)) {
                /* fields was missing the name it claimed to contain */
                PyErr_BadInternalCall();
            }
            res = -1;
            goto finish;
        }
        if (!HPy_ExtractDictItems_OiO(ctx, tup, &fld_dtype, &fld_offset, NULL)) {
            res = -1;
            HPy_Close(ctx, tup);
            goto finish;
        }

        /* If this field has objects, check for overlaps */
        PyArray_Descr *fld_dtype_struct = PyArray_Descr_AsStruct(ctx, fld_dtype);
        if (PyDataType_REFCHK(fld_dtype_struct)) {
            for (j = 0; j < names_size; ++j) {
                if (i != j) {
                    key = HPy_GetItem_i(ctx, names, j);
                    if (HPy_IsNull(key)) {
                        res = -1;
                        goto finish;
                    }
                    tup = HPyDict_GetItemWithError(ctx, fields, key);
                    HPy_Close(ctx, key);
                    if (HPy_IsNull(tup)) {
                        if (!HPyErr_Occurred(ctx)) {
                            /* fields was missing the name it claimed to contain */
                            PyErr_BadInternalCall();
                        }
                        res = -1;
                        goto finish;
                    }
                    if (!HPy_ExtractDictItems_OiO(ctx, tup, &fld2_dtype,
                                                &fld2_offset, NULL)) {
                        res = -1;
                        HPy_Close(ctx, tup);
                        goto finish;
                    }
                    /* Raise an exception if it overlaps */
                    PyArray_Descr *fld2_dtype_struct = PyArray_Descr_AsStruct(ctx, fld2_dtype);
                    if (fld_offset < fld2_offset + fld2_dtype_struct->elsize &&
                                fld2_offset < fld_offset + fld_dtype_struct->elsize) {
                        HPyErr_SetString(ctx, ctx->h_TypeError,
                                "Cannot create a NumPy dtype with overlapping "
                                "object fields");
                        res = -1;
                        goto finish;
                    }
                }
            }
        }
    }

    /* It passed all the overlap tests */
    res = 0;
finish:
    HPy_Close(ctx, fields);
    HPy_Close(ctx, names);
    return res;
}

/*
 * a dictionary specifying a data-type
 * must have at least two and up to four
 * keys These must all be sequences of the same length.
 *
 * can also have an additional key called "metadata" which can be any dictionary
 *
 * "names" --- field names
 * "formats" --- the data-type descriptors for the field.
 *
 * Optional:
 *
 * "offsets" --- integers indicating the offset into the
 * record of the start of the field.
 * if not given, then "consecutive offsets"
 * will be assumed and placed in the dictionary.
 *
 * "titles" --- Allows the use of an additional key
 * for the fields dictionary.(if these are strings
 * or unicode objects) or
 * this can also be meta-data to
 * be passed around with the field description.
 *
 * Attribute-lookup-based field names merely has to query the fields
 * dictionary of the data-descriptor.  Any result present can be used
 * to return the correct field.
 *
 * So, the notion of what is a name and what is a title is really quite
 * arbitrary.
 *
 * What does distinguish a title, however, is that if it is not None,
 * it will be placed at the end of the tuple inserted into the
 * fields dictionary.and can therefore be used to carry meta-data around.
 *
 * If the dictionary does not have "names" and "formats" entries,
 * then it will be checked for conformity and used directly.
 */
static HPy // PyArray_Descr *
_hpy_convert_from_field_dict(HPyContext *ctx, HPy obj, int align)
{
    HPy _numpy_internal;
    HPy res; // PyArray_Descr *

    _numpy_internal = HPyImport_ImportModule(ctx, "numpy.core._internal");
    if (HPy_IsNull(_numpy_internal)) {
        return HPy_NULL;
    }
    HPy h_align = HPyLong_FromLong(ctx, align);
    HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
    HPy args = HPyTuple_Pack(ctx, 2, obj, h_align);
    HPy_Close(ctx, h_align);
    HPy meth = HPy_GetAttr_s(ctx, _numpy_internal, "_usefields");
    res = HPy_CallTupleDict(ctx, meth, args, HPy_NULL);
    HPy_Close(ctx, args);
    HPy_Close(ctx, meth);
    HPy_Close(ctx, _numpy_internal);
    return res;
}

/*
 * Creates a struct dtype object from a Python dictionary.
 */
static HPy // PyArray_Descr *
_hpy_convert_from_dict(HPyContext *ctx, HPy obj, int align)
{
    HPy fields = HPyDict_New(ctx);
    HPy tup_0 = HPy_NULL, tup_1 = HPy_NULL, tup_2 = HPy_NULL;
    if (HPy_IsNull(fields)) {
        return HPyErr_NoMemory(ctx);
    }
    /*
     * Use PyMapping_GetItemString to support dictproxy objects as well.
     */
    HPy names = HPy_GetItem_s(ctx, obj, "names");
    if (HPy_IsNull(names)) {
        HPy_Close(ctx, fields);
        /* XXX should check this is a KeyError */
        HPyErr_Clear(ctx);
        return _hpy_convert_from_field_dict(ctx, obj, align);
    }
    HPy descrs = HPy_GetItem_s(ctx, obj, "formats");
    if (HPy_IsNull(descrs)) {
        HPy_Close(ctx, fields);
        /* XXX should check this is a KeyError */
        HPyErr_Clear(ctx);
        HPy_Close(ctx, names);
        return _hpy_convert_from_field_dict(ctx, obj, align);
    }
    int n = HPy_Length(ctx, names);
    HPy offsets = HPy_GetItem_s(ctx, obj, "offsets");
    if (HPy_IsNull(offsets)) {
        HPyErr_Clear(ctx);
    }
    HPy titles = HPy_GetItem_s(ctx, obj, "titles");
    if (HPy_IsNull(titles)) {
        HPyErr_Clear(ctx);
    }

    if ((n > HPy_Length(ctx, descrs))
        || (!HPy_IsNull(offsets) && (n > HPy_Length(ctx, offsets)))
        || (!HPy_IsNull(titles) && (n > HPy_Length(ctx, titles)))) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "'names', 'formats', 'offsets', and 'titles' dict "
                "entries must have the same length");
        goto fail;
    }

    /*
     * If a property 'aligned' is in the dict, it overrides the align flag
     * to be True if it not already true.
     */
    HPy tmp = HPy_GetItem_s(ctx, obj, "aligned");
    if (HPy_IsNull(tmp)) {
        HPyErr_Clear(ctx);
    } else {
        if (HPy_Is(ctx, tmp, ctx->h_True)) {
            align = 1;
        }
        else if (!HPy_Is(ctx, tmp, ctx->h_False)) {
            HPy_Close(ctx, tmp);
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "NumPy dtype descriptor includes 'aligned' entry, "
                    "but its value is neither True nor False");
            goto fail;
        }
        HPy_Close(ctx, tmp);
    }

    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;
    int totalsize = 0;
    int maxalign = 0;
    int has_out_of_order_fields = 0;
    for (int i = 0; i < n; i++) {
        tup_0 = HPy_NULL;
        tup_1 = HPy_NULL; 
        tup_2 = HPy_NULL;
        /* Build item to insert (descr, offset, [title])*/
        int len = 2;
        HPy title = HPy_NULL;
        if (!HPy_IsNull(titles)) {
            title = HPy_GetItem_i(ctx, titles, i);
            if (!HPy_IsNull(title) && !HPy_Is(ctx, title, ctx->h_None)) {
                len = 3;
            }
            else {
                HPy_Close(ctx, title);
            }
            HPyErr_Clear(ctx);
        }
        HPy descr = HPy_GetItem_i(ctx, descrs, i);
        if (HPy_IsNull(descr)) {
            goto fail;
        }
        HPy newdescr = _hpy_convert_from_any(ctx, descr, align); // PyArray_Descr *
        HPy_Close(ctx, descr);
        if (HPy_IsNull(newdescr)) {
            goto fail;
        }
        PyArray_Descr *newdescr_struct = PyArray_Descr_AsStruct(ctx, newdescr);
        tup_0 = newdescr;
        int _align = 1;
        if (align) {
            _align = newdescr_struct->alignment;
            maxalign = PyArray_MAX(maxalign,_align);
        }
        if (!HPy_IsNull(offsets)) {
            HPy off = HPy_GetItem_i(ctx, offsets, i);
            if (HPy_IsNull(off)) {
                goto fail;
            }
            long offset = HPyArray_PyIntAsInt(ctx, off);
            if (hpy_error_converting(ctx, offset)) {
                HPy_Close(ctx, off);
                goto fail;
            }
            HPy_Close(ctx, off);
            if (offset < 0) {
                HPyErr_Format_p(ctx, ctx->h_ValueError, "offset %ld cannot be negative",
                             offset);
                goto fail;
            }

            tup_1 = HPyLong_FromLong(ctx, offset);
            /* Flag whether the fields are specified out of order */
            if (offset < totalsize) {
                has_out_of_order_fields = 1;
            }
            /* If align=True, enforce field alignment */
            if (align && offset % newdescr_struct->alignment != 0) {
                HPyErr_Format_p(ctx, ctx->h_ValueError,
                        "offset %ld for NumPy dtype with fields is "
                        "not divisible by the field alignment %d "
                        "with align=True",
                        offset, newdescr_struct->alignment);
                goto fail;
            }
            else if (offset + newdescr_struct->elsize > totalsize) {
                totalsize = offset + newdescr_struct->elsize;
            }
        }
        else {
            if (align && _align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            tup_1 = HPyLong_FromLong(ctx, totalsize);
            totalsize += newdescr_struct->elsize;
        }
        if (len == 3) {
            tup_2 = title;
        }
        HPy name = HPy_GetItem_i(ctx, names, i);
        if (HPy_IsNull(name)) {
            goto fail;
        }
        if (!HPyUnicode_Check(ctx, name)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "field names must be strings");
            goto fail;
        }

        /* Insert into dictionary */
        HPy item = HPyDict_GetItemWithError(ctx, fields, name);
        if (!HPy_IsNull(item)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "name already used as a name or title");
            goto fail;
        }
        else if (HPyErr_Occurred(ctx)) {
            /* MemoryError during dict lookup */
            goto fail;
        }
        HPy tup;
        if (len == 3) {
            tup = HPyTuple_Pack(ctx, 3, tup_0, tup_1, tup_2);
            HPy_Close(ctx, tup_0);
            HPy_Close(ctx, tup_1);
            HPy_Close(ctx, tup_2);
        } else {
            tup = HPyTuple_Pack(ctx, 2, tup_0, tup_1, tup_2);
            HPy_Close(ctx, tup_0);
            HPy_Close(ctx, tup_1);
        }
        int ret = HPy_SetItem(ctx, fields, name, tup);
        HPy_Close(ctx, name);
        if (ret < 0) {
            goto fail;
        }
        if (len == 3) {
            if (HPyUnicode_Check(ctx, title)) {
                HPy item = HPyDict_GetItemWithError(ctx, fields, title);
                if (!HPy_IsNull(item)) {
                    HPy_Close(ctx, item);
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                            "title already used as a name or title.");
                    HPy_Close(ctx, tup);
                    goto fail;
                }
                else if (HPyErr_Occurred(ctx)) {
                    /* MemoryError during dict lookup */
                    goto fail;
                }
                if (HPy_SetItem(ctx, fields, title, tup) < 0) {
                    HPy_Close(ctx, tup);
                    goto fail;
                }
            }
        }
        HPy_Close(ctx, tup);
        dtypeflags |= (newdescr_struct->flags & NPY_FROM_FIELDS);
    }

    HPy new = HPyArray_DescrNewFromType(ctx, NPY_VOID);
    if (HPy_IsNull(new)) {
        goto fail;
    }
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }
    PyArray_Descr *new_struct = PyArray_Descr_AsStruct(ctx, new);
    if (align) {
        new_struct->alignment = maxalign;
    }
    new_struct->elsize = totalsize;
    if (!HPyTuple_Check(ctx, names)) {
        HPy_SETREF(ctx, names, HPySequence_Tuple(ctx, names));
        if (HPy_IsNull(names)) {
            HPy_Close(ctx, new);
            goto fail;
        }
    }
    HPyField_Store(ctx, new, &new_struct->names, names);
    new_struct->fields = HPy_AsPyObject(ctx, fields);
    new_struct->flags = dtypeflags;
    /* new takes responsibility for DECREFing names, fields */
    HPy_Close(ctx, names);
    names = HPy_NULL;
    fields = HPy_NULL;

    /*
     * If the fields weren't in order, and there was an OBJECT type,
     * need to verify that no OBJECT types overlap with something else.
     */
    if (has_out_of_order_fields && PyDataType_REFCHK(new_struct)) {
        if (_hpy_validate_object_field_overlap(ctx, new, new_struct) < 0) {
            HPy_Close(ctx, new);
            goto fail;
        }
    }

    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new_struct->flags |= NPY_ALIGNED_STRUCT;
    }

    /* Override the itemsize if provided */
    tmp = HPy_GetItem_s(ctx, obj, "itemsize");
    if (HPy_IsNull(tmp)) {
        HPyErr_Clear(ctx);
    } else {
        int itemsize = (int)HPyArray_PyIntAsInt(ctx, tmp);
        HPy_Close(ctx, tmp);
        if (error_converting(itemsize)) {
            HPy_Close(ctx, new);
            goto fail;
        }
        /* Make sure the itemsize isn't made too small */
        if (itemsize < new_struct->elsize) {
            HPyErr_Format_p(ctx, ctx->h_ValueError,
                    "NumPy dtype descriptor requires %d bytes, "
                    "cannot override to smaller itemsize of %d",
                    new_struct->elsize, itemsize);
            HPy_Close(ctx, new);
            goto fail;
        }
        /* If align is set, make sure the alignment divides into the size */
        if (align && new_struct->alignment > 0 && itemsize % new_struct->alignment != 0) {
            HPyErr_Format_p(ctx, ctx->h_ValueError,
                    "NumPy dtype descriptor requires alignment of %d bytes, "
                    "which is not divisible into the specified itemsize %d",
                    new_struct->alignment, itemsize);
            HPy_Close(ctx, new);
            goto fail;
        }
        /* Set the itemsize */
        new_struct->elsize = itemsize;
    }

    /* Add the metadata if provided */
    HPy metadata = HPy_GetItem_s(ctx, obj, "metadata");

    if (HPy_IsNull(metadata)) {
        HPyErr_Clear(ctx);
    }
    else if (new_struct->metadata == NULL) {
        new_struct->metadata = HPy_AsPyObject(ctx, metadata);
        HPy_Close(ctx, metadata);
    }
    else {
        PyObject *py_metadata = HPy_AsPyObject(ctx, metadata);
        int ret = PyDict_Merge(new_struct->metadata, py_metadata, 0);
        Py_DECREF(py_metadata);
        HPy_Close(ctx, metadata);
        if (ret < 0) {
            HPy_Close(ctx, new);
            goto fail;
        }
    }

    HPy_Close(ctx, fields);
    HPy_Close(ctx, names);
    HPy_Close(ctx, descrs);
    HPy_Close(ctx, offsets);
    HPy_Close(ctx, titles);
    return new;

 fail:
    HPy_Close(ctx, tup_0);
    HPy_Close(ctx, tup_1);
    HPy_Close(ctx, tup_2);
    HPy_Close(ctx, fields);
    HPy_Close(ctx, names);
    HPy_Close(ctx, descrs);
    HPy_Close(ctx, offsets);
    HPy_Close(ctx, titles);
    return HPy_NULL;
}


/*NUMPY_API*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewFromType(int type_num)
{
    PyArray_Descr *old;
    PyArray_Descr *new;

    old = PyArray_DescrFromType(type_num);
    if (old == NULL) {
        return NULL;
    }
    new = PyArray_DescrNew(old);
    Py_DECREF(old);
    return new;
}

/*HPY_NUMPY_API*/
NPY_NO_EXPORT HPy /* (PyArray_Descr *) */
HPyArray_DescrNewFromType(HPyContext *ctx, int type_num)
{
    HPy old; /* (PyArray_Descr *) */
    HPy new; /* (PyArray_Descr *) */

    old = HPyArray_DescrFromType(ctx, type_num);
    if (HPy_IsNull(old)) {
        return HPy_NULL;
    }
    new = HPyArray_DescrNew(ctx, old);
    HPy_Close(ctx, old);
    return new;
}

/*HPY_NUMPY_API
 * Get typenum from an object -- None goes to HPy_NULL
 */
NPY_NO_EXPORT int
HPyArray_DescrConverter2(HPyContext *ctx, HPy obj, HPy *at)
{
    // HPY TODO: the default for kwargs parsing is NULL with HPY
    if (HPy_Is(ctx, obj, ctx->h_None) || HPy_IsNull(obj)) {
        *at = HPy_NULL;
        return NPY_SUCCEED;
    }
    else {
        return HPyArray_DescrConverter(ctx, obj, at);
    }
}

/*NUMPY_API
 * Get typenum from an object -- None goes to NULL
 */
NPY_NO_EXPORT int
PyArray_DescrConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (obj == Py_None) {
        *at = NULL;
        return NPY_SUCCEED;
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
}

// HPY TODO: once the necessary helper functions are in API, no need to include:
#include "arraytypes.h"
#include "scalarapi.h"

/**
 * Get a dtype instance from a python type
 */
static HPy
_hpy_convert_from_type(HPyContext *ctx, HPy typ) {
    // TODO: create instead global handles for those:
    HPy h_PyGenericArrType_Type = HPyGlobal_Load(ctx, HPyGenericArrType_Type);
    int isSubtype = HPyType_IsSubtype(ctx, typ, h_PyGenericArrType_Type);
    HPy_Close(ctx, h_PyGenericArrType_Type);
    if (isSubtype) {
        return HPyArray_DescrFromTypeObject(ctx, typ);
    }
    else if (HPy_Is(ctx, typ, ctx->h_LongType)) {
        return HPyArray_DescrFromType(ctx, NPY_LONG);
    }
    else if (HPy_Is(ctx, typ, ctx->h_FloatType)) {
        return HPyArray_DescrFromType(ctx, NPY_DOUBLE);
    }
    else if (HPy_Is(ctx, typ, ctx->h_ComplexType)) {
        return HPyArray_DescrFromType(ctx, NPY_CDOUBLE);
    }
    else if (HPy_Is(ctx, typ, ctx->h_BoolType)) {
        return HPyArray_DescrFromType(ctx, NPY_BOOL);
    }
    else if (HPy_Is(ctx, typ, ctx->h_BytesType)) {
        /*
         * TODO: This should be deprecated, and have special handling for
         *       dtype=bytes/"S" in coercion: It should not rely on "S0".
         */
        return HPyArray_DescrFromType(ctx, NPY_STRING);
    }
    else if (HPy_Is(ctx, typ, ctx->h_UnicodeType)) {
        /*
         * TODO: This should be deprecated, and have special handling for
         *       dtype=str/"U" in coercion: It should not rely on "U0".
         */
        return HPyArray_DescrFromType(ctx, NPY_UNICODE);
    }
    else if (HPy_Is(ctx, typ, ctx->h_MemoryViewType)) {
        return HPyArray_DescrFromType(ctx, NPY_VOID);
    }
    else if (HPy_Is(ctx, typ, ctx->h_BaseObjectType)) {
        return HPyArray_DescrFromType(ctx, NPY_OBJECT);
    }
    else {
        HPy ret = _hpy_try_convert_from_dtype_attr(ctx, typ);
        if (!HPy_Is(ctx, ret, ctx->h_NotImplemented)) {
            return ret;
        }
        HPy_Close(ctx, ret);

        /*
         * Note: this comes after _try_convert_from_dtype_attr because the ctypes
         * type might override the dtype if numpy does not otherwise
         * support it.
         */
        ret = _hpy_try_convert_from_ctypes_type(ctx, typ);
        if (!HPy_Is(ctx, ret, ctx->h_NotImplemented)) {
            return ret;
        }
        HPy_Close(ctx, ret);

        /*
         * All other classes are treated as object. This can be convenient
         * to convey an intention of using it for a specific python type
         * and possibly allow converting to a new type-specific dtype in the future. It may make sense to
         * only allow this only within `dtype=...` keyword argument context
         * in the future.
         */
        return HPyArray_DescrFromType(ctx, NPY_OBJECT);
    }
}


static HPy // PyArray_Descr *
_hpy_convert_from_str(HPyContext *ctx, HPy obj, int align);

static HPy // PyArray_Descr *
_hpy_convert_from_any(HPyContext *ctx, HPy obj, int align)
{
    /* default */
    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        return HPyArray_DescrFromType(ctx, NPY_DEFAULT_TYPE);
    }
    else if (HPyArray_DescrCheck(ctx, obj)) {
        // PyArray_Descr *ret = (PyArray_Descr *)obj;
        // Py_INCREF(ret);
        return HPy_Dup(ctx, obj);
    }
    // else if (PyType_Check(obj)) {
    else if (HPy_TypeCheck(ctx, obj, ctx->h_TypeType)) { // this might be incorrect
        return _hpy_convert_from_type(ctx, obj);
    }
    /* or a typecode string */
    else if (HPyBytes_Check(ctx, obj)) {
        /* Allow bytes format strings: convert to unicode */
        HPy obj2 = HPyUnicode_FromEncodedObject(ctx, obj, NULL, NULL);
        if (HPy_IsNull(obj2)) {
            /* Convert the exception into a TypeError */
            if (HPyErr_ExceptionMatches(ctx, ctx->h_UnicodeDecodeError)) {
                HPyErr_SetString(ctx, ctx->h_TypeError,
                        "data type not understood");
            }
            return HPy_NULL;
        }
        HPy ret = _hpy_convert_from_str(ctx, obj2, align); // PyArray_Descr *
        HPy_Close(ctx, obj2);
        return ret;
    }
    else if (HPyUnicode_Check(ctx, obj)) {
        return _hpy_convert_from_str(ctx, obj, align);
    }
    else if (HPyTuple_Check(ctx, obj)) {
        /* or a tuple */
        CAPI_WARN("missing Py_EnterRecursiveCall & Py_LeaveRecursiveCall");
        if (Py_EnterRecursiveCall(
                " while trying to convert the given data type from"
                " a tuple object" ) != 0) {
            return HPy_NULL;
        }
        HPy ret = _hpy_convert_from_tuple(ctx, obj, align); // PyArray_Descr *
        Py_LeaveRecursiveCall();
        return ret;
    }
    else if (HPyList_Check(ctx, obj)) {
        /* or a list */
        CAPI_WARN("missing Py_EnterRecursiveCall & Py_LeaveRecursiveCall");
        if (Py_EnterRecursiveCall(
                " while trying to convert the given data type from"
                " a list object" ) != 0) {
            return HPy_NULL;
        }
        HPy ret = _hpy_convert_from_array_descr(ctx, obj, align); // PyArray_Descr *
        Py_LeaveRecursiveCall();
        return ret;
    }
    else if (HPyDict_Check(ctx, obj)) { // || PyDictProxy_Check(HPy_AsPyObject(ctx, obj))) { // TODO: decref
        /* or a dictionary */
        if (Py_EnterRecursiveCall(
                " while trying to convert the given data type from"
                " a dict object" ) != 0) {
            return HPy_NULL;
        }
        HPy ret = _hpy_convert_from_dict(ctx, obj, align); // PyArray_Descr *
        Py_LeaveRecursiveCall();
        return ret;
    }
    else if (HPyArray_Check(ctx, obj)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "Cannot construct a dtype from an array");
        return HPy_NULL;
    }
    else {
        HPy ret = _hpy_try_convert_from_dtype_attr(ctx, obj); // PyArray_Descr *
        if (!HPy_Is(ctx, ret, ctx->h_NotImplemented)) {
            return ret;
        }
        HPy_Close(ctx, ret);
        /*
         * Note: this comes after _try_convert_from_dtype_attr because the ctypes
         * type might override the dtype if numpy does not otherwise
         * support it.
         */
        HPy obj_type = HPy_Type(ctx, obj);
        ret = _hpy_try_convert_from_ctypes_type(ctx, obj_type);
        if (!HPy_Is(ctx, ret, ctx->h_NotImplemented)) {
            return ret;
        }
        HPy_Close(ctx, ret);
        // PyErr_Format(PyExc_TypeError, "Cannot interpret '%R' as a data type", obj);
        HPyErr_SetString(ctx, ctx->h_TypeError, "Cannot interpret '%R' as a data type");
        return HPy_NULL;
    }
}

/*HPY_NUMPY_API
 * Get typenum from an object -- None goes to NPY_DEFAULT_TYPE
 * This function takes a Python object representing a type and converts it
 * to a the correct PyArray_Descr * structure to describe the type.
 *
 * Many objects can be used to represent a data-type which in NumPy is
 * quite a flexible concept.
 *
 * This is the central code that converts Python objects to
 * Type-descriptor objects that are used throughout numpy.
 *
 * Returns a new reference in *at, but the returned should not be
 * modified as it may be one of the canonical immutable objects or
 * a reference to the input obj.
 */
NPY_NO_EXPORT int
HPyArray_DescrConverter(HPyContext *ctx, HPy obj, HPy *at)
{
    *at = _hpy_convert_from_any(ctx, obj, 0);
    return HPy_IsNull(*at) ? NPY_FAIL : NPY_SUCCEED;
}


/*NUMPY_API
 * Get typenum from an object -- None goes to NPY_DEFAULT_TYPE
 * This function takes a Python object representing a type and converts it
 * to a the correct PyArray_Descr * structure to describe the type.
 *
 * Many objects can be used to represent a data-type which in NumPy is
 * quite a flexible concept.
 *
 * This is the central code that converts Python objects to
 * Type-descriptor objects that are used throughout numpy.
 *
 * Returns a new reference in *at, but the returned should not be
 * modified as it may be one of the canonical immutable objects or
 * a reference to the input obj.
 */
NPY_NO_EXPORT int
PyArray_DescrConverter(PyObject *obj, PyArray_Descr **at)
{
    HPyContext *ctx = npy_get_context();
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPy h_ret = _hpy_convert_from_any(ctx, h_obj, 0);
    PyObject *py_ret = HPy_AsPyObject(ctx, h_ret);
    *at = (PyArray_Descr *)py_ret;
    HPy_Close(ctx, h_obj);
    HPy_Close(ctx, h_ret);
    return (*at) ? NPY_SUCCEED : NPY_FAIL;
}

/** Convert a bytestring specification into a dtype */
static HPy // PyArray_Descr *
_hpy_convert_from_str(HPyContext *ctx, HPy obj, int align)
{
    /* Check for a string typecode. */
    Py_ssize_t len = 0;
    char const *type = HPyUnicode_AsUTF8AndSize(ctx, obj, &len);
    if (type == NULL) {
        return HPy_NULL;
    }

    /* Empty string is invalid */
    if (len == 0) {
        goto fail;
    }

    /* check for commas present or first (or second) element a digit */
    if (_check_for_commastring(type, len)) {
        return _hpy_convert_from_commastring(ctx, obj, align);
    }

    /* Process the endian character. '|' is replaced by '='*/
    char endian = '=';
    switch (type[0]) {
        case '>':
        case '<':
        case '=':
            endian = type[0];
            ++type;
            --len;
            break;

        case '|':
            endian = '=';
            ++type;
            --len;
            break;
    }

    /* Just an endian character is invalid */
    if (len == 0) {
        goto fail;
    }

    /* Check for datetime format */
    if (is_datetime_typestr(type, len)) {
        PyArray_Descr *ret = parse_dtype_from_datetime_typestr(type, len);
        if (ret == NULL) {
            return HPy_NULL;
        }
        /* ret has byte order '=' at this point */
        if (!PyArray_ISNBO(endian)) {
            ret->byteorder = endian;
        }
        HPy h_ret = HPy_FromPyObject(ctx, (PyObject *)ret);
        Py_DECREF(ret);
        return h_ret;
    }

    int check_num = NPY_NOTYPE + 10;
    int elsize = 0;
    /* A typecode like 'd' */
    if (len == 1) {
        /* Python byte string characters are unsigned */
        check_num = (unsigned char) type[0];
    }
    /* A kind + size like 'f8' */
    else {
        char *typeend = NULL;
        int kind;

        /* Parse the integer, make sure it's the rest of the string */
        elsize = (int)strtol(type + 1, &typeend, 10);
        if (typeend - type == len) {

            kind = type[0];
            switch (kind) {
                case NPY_STRINGLTR:
                case NPY_STRINGLTR2:
                    check_num = NPY_STRING;
                    break;

                /*
                 * When specifying length of UNICODE
                 * the number of characters is given to match
                 * the STRING interface.  Each character can be
                 * more than one byte and itemsize must be
                 * the number of bytes.
                 */
                case NPY_UNICODELTR:
                    check_num = NPY_UNICODE;
                    elsize <<= 2;
                    break;

                case NPY_VOIDLTR:
                    check_num = NPY_VOID;
                    break;

                default:
                    if (elsize == 0) {
                        check_num = NPY_NOTYPE+10;
                    }
                    /* Support for generic processing c8, i4, f8, etc...*/
                    else {
                        check_num = PyArray_TypestrConvert(elsize, kind);
                        if (check_num == NPY_NOTYPE) {
                            check_num += 10;
                        }
                        elsize = 0;
                    }
            }
        }
    }

    if (HPyErr_Occurred(ctx)) {
        goto fail;
    }

    HPy ret; // PyArray_Descr *
    if ((check_num == NPY_NOTYPE + 10) ||
            HPy_IsNull(ret = HPyArray_DescrFromType(ctx, check_num))) {
        HPyErr_Clear(ctx);
        /* Now check to see if the object is registered in typeDict */
        HPy typeDict = HPyGlobal_Load(ctx, descr_typeDict);
        if (HPy_IsNull(typeDict)) {
            goto fail;
        }
        HPy item = HPyDict_GetItemWithError(ctx, typeDict, obj);
        HPy_Close(ctx, typeDict);
        if (HPy_IsNull(item)) {
            if (HPyErr_Occurred(ctx)) {
                return HPy_NULL;
            }
            goto fail;
        }

        /*
         * Probably only ever dispatches to `_convert_from_type`, but who
         * knows what users are injecting into `np.typeDict`.
         */
        return _hpy_convert_from_any(ctx, item, align);
    }

    PyArray_Descr *ret_struct = PyArray_Descr_AsStruct(ctx, ret);

    if (PyDataType_ISUNSIZED(ret_struct) && ret_struct->elsize != elsize) {
        HPyArray_DESCR_REPLACE(ctx, ret, ret_struct);
        if (HPy_IsNull(ret)) {
            return HPy_NULL;
        }
        ret_struct->elsize = elsize;
    }
    if (endian != '=' && PyArray_ISNBO(endian)) {
        endian = '=';
    }
    if (endian != '=' && ret_struct->byteorder != '|' && ret_struct->byteorder != endian) {
        HPyArray_DESCR_REPLACE(ctx, ret, ret_struct);
        if (HPy_IsNull(ret)) {
            return HPy_NULL;
        }
        ret_struct = PyArray_Descr_AsStruct(ctx, ret);
        ret_struct->byteorder = endian;
    }
    return ret;

fail:
    // PyErr_Format(PyExc_TypeError, "data type %R not understood", obj);
    HPyErr_SetString(ctx, ctx->h_TypeError, "data type %R not understood");
    return HPy_NULL;
}

/** Array Descr Objects for dynamic types **/

/*
 * There are some statically-defined PyArray_Descr objects corresponding
 * to the basic built-in types.
 * These can and should be DECREF'd and INCREF'd as appropriate, anyway.
 * If a mistake is made in reference counting, deallocation on these
 * builtins will be attempted leading to problems.
 *
 * This lets us deal with all PyArray_Descr objects using reference
 * counting (regardless of whether they are statically or dynamically
 * allocated).
 */

/*NUMPY_API
 * base cannot be NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNew(PyArray_Descr *base)
{
    HPyContext *ctx = npy_get_context();
    HPy h_base = HPy_FromPyObject(ctx, (PyObject *)base);
    HPy h_res = HPyArray_DescrNew(ctx, h_base);
    PyArray_Descr *ret = (PyArray_Descr *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_base);
    HPy_Close(ctx, h_res);
    return ret;
}

/*HPY_NUMPY_API
 * base cannot be NULL
 */
NPY_NO_EXPORT HPy
HPyArray_DescrNew(HPyContext *ctx, HPy h_base)
{
    PyArray_Descr *newdescr;
    // ATTENTION! When changing to non-legacy: update also the memcopy below
    PyArray_Descr *base = PyArray_Descr_AsStruct(ctx, h_base);
    HPy h_newdescr = HPy_New(ctx, HPy_Type(ctx, h_base), &newdescr);

    if (HPy_IsNull(h_newdescr)) {
        return HPy_NULL;
    }
    /* Don't copy PyObject_HEAD part */
    size_t offset = 0;
    if (SHAPE(PyArray_Descr) == HPyType_BuiltinShape_Legacy) {
        offset = sizeof(PyObject);
    }
    memcpy((char *)newdescr + offset,
           (char *)base + offset,
           sizeof(PyArray_Descr) - offset);
    newdescr->names = HPyField_NULL;
    if (!HPyField_IsNull(base->names)) {
        HPy h = HPyField_Load(ctx, h_base, base->names);
        HPyField_Store(ctx, h_newdescr, &newdescr->names, h);
        HPy_Close(ctx, h);
    }

    newdescr->typeobj = HPyField_NULL;
    if (!HPyField_IsNull(base->typeobj)) {
        HPy h = HPyField_Load(ctx, h_base, base->typeobj);
        HPyField_Store(ctx, h_newdescr, &newdescr->typeobj, h);
        HPy_Close(ctx, h);
    }

    /*
     * The c_metadata has a by-value ownership model, need to clone it
     * (basically a deep copy, but the auxdata clone function has some
     * flexibility still) so the new PyArray_Descr object owns
     * a copy of the data. Having both 'base' and 'newdescr' point to
     * the same auxdata pointer would cause a double-free of memory.
     */
    if (base->c_metadata != NULL) {
        newdescr->c_metadata = NPY_AUXDATA_CLONE(base->c_metadata);
        if (newdescr->c_metadata == NULL) {
            HPyErr_NoMemory(ctx);
            return HPy_NULL;
        }
    }

    if (newdescr->fields == Py_None) {
        newdescr->fields = NULL;
    }
    Py_XINCREF(newdescr->fields);
    if (!HPyField_IsNull(newdescr->names)) {
        HPy h_names = HPyField_Load(ctx, h_newdescr, newdescr->names);
        HPyField_Store(ctx, h_newdescr, &newdescr->names, h_names);
        HPy_Close(ctx, h_names);
    }
    if (newdescr->subarray) {
        newdescr->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (newdescr->subarray == NULL) {
            HPyErr_NoMemory(ctx);
            return HPy_NULL;
        }
        memcpy(newdescr->subarray, base->subarray, sizeof(PyArray_ArrayDescr));
        Py_INCREF(newdescr->subarray->shape);
        Py_INCREF(newdescr->subarray->base);
    }
    if (!HPyField_IsNull(newdescr->typeobj)) {
        HPy h_typeobj = HPyField_Load(ctx, h_newdescr, newdescr->typeobj);
        HPyField_Store(ctx, h_newdescr, &newdescr->typeobj, h_typeobj);
        HPy_Close(ctx, h_typeobj);
    }
    Py_XINCREF(newdescr->metadata);
    newdescr->hash = -1;

    return h_newdescr;
}

/*
 * should never be called for builtin-types unless
 * there is a reference-count problem
 */
HPyDef_SLOT(arraydescr_dealloc, HPy_tp_destroy)
static void
arraydescr_dealloc_impl(void *self_p)
{
    PyArray_Descr *self = (PyArray_Descr *)self_p;
    if (self->fields == Py_None) {
        fprintf(stderr, "*** Reference count error detected: "
                "an attempt was made to deallocate the dtype %d (%c) ***\n",
                self->type_num, self->type);
        // Py_INCREF(self);
        // Py_INCREF(self);
        return;
    }
    // hpy_abort_not_implemented("non-builtin descriptors...");
    // Py_XDECREF(self->typeobj);
    // Py_XDECREF(self->names);
    // Py_XDECREF(self->fields);
    // if (self->subarray) {
    //     Py_XDECREF(self->subarray->shape);
    //     Py_DECREF(self->subarray->base);
    //     PyArray_free(self->subarray);
    // }
    // Py_XDECREF(self->metadata);
    // NPY_AUXDATA_FREE(self->c_metadata);
    // self->c_metadata = NULL;
    // Py_TYPE(self)->tp_free((PyObject *)self);
}

/*
 * we need to be careful about setting attributes because these
 * objects are pointed to by arrays that depend on them for interpreting
 * data.  Currently no attributes of data-type objects can be set
 * directly except names.
 */
HPyDef_MEMBER(arraydescr_type, "type", HPyMember_OBJECT, offsetof(PyArray_Descr, typeobj), .readonly=1)
HPyDef_MEMBER(arraydescr_kind, "kind", HPyMember_CHAR, offsetof(PyArray_Descr, kind), .readonly=1)
HPyDef_MEMBER(arraydescr_char, "char", HPyMember_CHAR, offsetof(PyArray_Descr, type), .readonly=1)
HPyDef_MEMBER(arraydescr_num, "num", HPyMember_INT, offsetof(PyArray_Descr, type_num), .readonly=1)
HPyDef_MEMBER(arraydescr_byteorder, "byteorder", HPyMember_CHAR, offsetof(PyArray_Descr, byteorder), .readonly=1)
HPyDef_MEMBER(arraydescr_itemsize, "itemsize", HPyMember_INT, offsetof(PyArray_Descr, elsize), .readonly=1)
HPyDef_MEMBER(arraydescr_alignment, "alignment", HPyMember_INT, offsetof(PyArray_Descr, alignment), .readonly=1)
HPyDef_MEMBER(arraydescr_flags, "flags", HPyMember_BYTE, offsetof(PyArray_Descr, flags), .readonly=1)

HPyDef_GET(arraydescr_subdescr, "subdtype")
static HPy
arraydescr_subdescr_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASSUBARRAY(self_struct)) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    HPy base = HPy_FromPyObject(ctx, (PyObject *)self_struct->subarray->base);
    HPy shape = HPy_FromPyObject(ctx, self_struct->subarray->shape);
    HPy ret = HPy_BuildValue(ctx, "OO", base, shape);
    HPy_Close(ctx, base);
    HPy_Close(ctx, shape);
    return ret;
}

HPyDef_GET(_arraydescr_protocol_typestr, "str")
NPY_NO_EXPORT HPy
_arraydescr_protocol_typestr_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    char basic_ = self_struct->kind;
    char endian = self_struct->byteorder;
    int size = self_struct->elsize;
    HPy ret;

    if (endian == '=') {
        endian = '<';
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    if (self_struct->type_num == NPY_UNICODE) {
        size >>= 2;
    }
    if (self_struct->type_num == NPY_OBJECT) {
        ret = HPyUnicode_FromFormat_p(ctx, "%c%c", endian, basic_);
    }
    else {
        ret = HPyUnicode_FromFormat_p(ctx, "%c%c%d", endian, basic_, size);
    }
    if (HPy_IsNull(ret)) {
        return HPy_NULL;
    }

    if (PyDataType_ISDATETIME(self_struct)) {
        PyArray_DatetimeMetaData *meta;
        meta = h_get_datetime_metadata_from_dtype(ctx, self_struct);
        if (meta == NULL) {
            HPy_Close(ctx, ret);
            return HPy_NULL;
        }
        HPy umeta = hpy_metastr_to_unicode(ctx, meta, 0);
        if (HPy_IsNull(umeta)) {
            HPy_Close(ctx, ret);
            return HPy_NULL;
        }

        HPy_SETREF(ctx, ret, HPyUnicode_Concat_t(ctx, ret, umeta));
        HPy_Close(ctx, umeta);
    }
    return ret;
}

/* non-static variant of '_arraydescr_protocol_typestr_get' since it is used in
   other files as well */
NPY_NO_EXPORT HPy
arraydescr_protocol_typestr_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored)){
    return _arraydescr_protocol_typestr_get(ctx, self, NULL);
}

HPyDef_GET(arraydescr_name, "name")
static HPy
arraydescr_name_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    /* let python handle this */
    HPy _numpy_dtype, args, meth;
    HPy res;
    _numpy_dtype = HPyImport_ImportModule(ctx, "numpy.core._dtype");
    if (HPy_IsNull(_numpy_dtype)) {
        return HPy_NULL;
    }

    HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
    args = HPyTuple_Pack(ctx, 1, self);
    meth = HPy_GetAttr_s(ctx, _numpy_dtype, "_name_get");
    res = HPy_CallTupleDict(ctx, meth, args, HPy_NULL);
    HPy_Close(ctx, args);
    HPy_Close(ctx, meth);
    HPy_Close(ctx, _numpy_dtype);
    return res;
}

HPyDef_GET(arraydescr_base, "base")
static HPy
arraydescr_base_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASSUBARRAY(self_struct)) {
        // Py_INCREF(self);
        return HPy_Dup(ctx, self);
    }
    // Py_INCREF(self->subarray->base);
    return HPy_FromPyObject(ctx, self_struct->subarray->base);
}

HPyDef_GET(arraydescr_shape, "shape")
static HPy
arraydescr_shape_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASSUBARRAY(self_struct)) {
        return HPyTuple_Pack(ctx, 0);
    }
    assert(PyTuple_Check(self_struct->subarray->shape));
    // Py_INCREF(self_struct->subarray->shape);
    return HPy_FromPyObject(ctx, self_struct->subarray->shape);
}

HPyDef_GET(arraydescr_ndim, "ndim")
static HPy
arraydescr_ndim_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    Py_ssize_t ndim;

    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASSUBARRAY(self_struct)) {
        return HPyLong_FromLong(ctx, 0);
    }

    /*
     * PyTuple_Size has built in check
     * for tuple argument
     */
    ndim = PyTuple_Size(self_struct->subarray->shape);
    return HPyLong_FromLong(ctx, ndim);
}


HPyDef_GET(arraydescr_descr, "descr")
NPY_NO_EXPORT HPy
arraydescr_descr_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    HPy dobj, res;
    HPy _numpy_internal, args, meth;

    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASFIELDS(self_struct)) {
        /* get default */
        HPyTupleBuilder dobj_tb = HPyTupleBuilder_New(ctx, 2);
        if (HPyTupleBuilder_IsNull(dobj_tb)) {
            return HPy_NULL;
        }
        HPy item0 = HPyUnicode_FromString(ctx, "");
        HPy item1 = _arraydescr_protocol_typestr_get(ctx, self, NULL);
        HPyTupleBuilder_Set(ctx, dobj_tb, 0, item0);
        HPyTupleBuilder_Set(ctx, dobj_tb, 1, item1);
        HPy_Close(ctx, item0);
        HPy_Close(ctx, item1);
        HPyListBuilder res_lb = HPyListBuilder_New(ctx, 1);
        if (HPyListBuilder_IsNull(res_lb)) {
            HPyTupleBuilder_Cancel(ctx, dobj_tb);
            return HPy_NULL;
        }
        dobj = HPyTupleBuilder_Build(ctx, dobj_tb);
        HPyListBuilder_Set(ctx, res_lb, 0, dobj);
        HPy_Close(ctx, dobj);
        return HPyListBuilder_Build(ctx, res_lb);
    }

    _numpy_internal = HPyImport_ImportModule(ctx, "numpy.core._internal");
    if (HPy_IsNull(_numpy_internal)) {
        return HPy_NULL;
    }
    HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
    args = HPyTuple_Pack(ctx, 1, self);
    meth = HPy_GetAttr_s(ctx, _numpy_internal, "_array_descr");
    res = HPy_CallTupleDict(ctx, meth, args, HPy_NULL);
    HPy_Close(ctx, args);
    HPy_Close(ctx, meth);
    HPy_Close(ctx, _numpy_internal);
    return res;
}

NPY_NO_EXPORT HPy
arraydescr_protocol_descr_get(HPyContext *ctx, HPy self, void *NPY_UNUSED(ignored))
{
    return arraydescr_descr_get(ctx, self, NULL);
}

/*
 * returns 1 for a builtin type
 * and 2 for a user-defined data-type descriptor
 * return 0 if neither (i.e. it's a copy of one)
 */
HPyDef_GET(arraydescr_isbuiltin, "isbuiltin")
static HPy
arraydescr_isbuiltin_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    long val;
    val = 0;
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    CAPI_WARN("PyArray_Descr.fields is a PyObject");
    if (self_struct->fields == Py_None) {
        val = 1;
    }
    if (PyTypeNum_ISUSERDEF(self_struct->type_num)) {
        val = 2;
    }
    return HPyLong_FromLong(ctx, val);
}

static int
_arraydescr_isnative(HPyContext *ctx, HPy /* PyArray_Descr * */ self)
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASFIELDS(self_struct)) {
        return PyArray_ISNBO(self_struct->byteorder);
    }
    else {
        PyObject *key, *value, *title = NULL;
        HPy new; // PyArray_Descr *
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self_struct->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return -1;
            }
            if (!_arraydescr_isnative(ctx, new)) {
                return 0;
            }
        }
    }
    return 1;
}

/*
 * return Py_True if this data-type descriptor
 * has native byteorder if no fields are defined
 *
 * or if all sub-fields have native-byteorder if
 * fields are defined
 */
HPyDef_GET(arraydescr_isnative, "isnative")
static HPy
arraydescr_isnative_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    HPy ret;
    int retval;
    retval = _arraydescr_isnative(ctx, self);
    if (retval == -1) {
        return HPy_NULL;
    }
    ret = HPy_Dup(ctx, retval ? ctx->h_True : ctx->h_False);
    // Py_INCREF(ret);
    return ret;
}

HPyDef_GET(arraydescr_isalignedstruct, "isalignedstruct")
static HPy
arraydescr_isalignedstruct_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    HPy ret;
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    ret = HPy_Dup(ctx, (self_struct->flags&NPY_ALIGNED_STRUCT) ? ctx->h_True : ctx->h_False);
    // Py_INCREF(ret);
    return ret;
}

HPyDef_GET(arraydescr_fields, "fields")
static HPy
arraydescr_fields_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASFIELDS(self_struct)) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    CAPI_WARN("missing PyDictProxy_New");
    PyObject *py_ret = PyDictProxy_New(self_struct->fields);
    HPy ret = HPy_FromPyObject(ctx, py_ret);
    Py_XDECREF(py_ret);
    return ret;
}

HPyDef_GET(arraydescr_metadata, "metadata")
static HPy
arraydescr_metadata_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (self_struct->metadata == NULL) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    CAPI_WARN("missing PyDictProxy_New");
    PyObject *py_ret = PyDictProxy_New(self_struct->metadata);
    HPy ret = HPy_FromPyObject(ctx, py_ret);
    Py_XDECREF(py_ret);
    return ret;
}

HPyDef_GET(arraydescr_hasobject, "hasobject")
static HPy
arraydescr_hasobject_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (PyDataType_FLAGCHK(self_struct, NPY_ITEM_HASOBJECT)) {
        return HPy_Dup(ctx, ctx->h_True);
    }
    else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}

HPyDef_GETSET(arraydescr_names, "names")
static HPy
arraydescr_names_get(HPyContext *ctx, HPy /* PyArray_Descr * */ self, void *NPY_UNUSED(ignored))
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASFIELDS(self_struct)) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    return HPyField_Load(ctx, self, self_struct->names);
}

static int
arraydescr_names_set(
        HPyContext *ctx, HPy /* PyArray_Descr * */ self, HPy val, void *NPY_UNUSED(ignored))
{
    int N = 0;
    int i;
    HPy new_names;
    HPy new_fields;

    if (HPy_IsNull(val)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete dtype names attribute");
        return -1;
    }
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (!PyDataType_HASFIELDS(self_struct)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "there are no fields defined");
        return -1;
    }

    /*
     * FIXME
     *
     * This deprecation has been temporarily removed for the NumPy 1.7
     * release. It should be re-added after the 1.7 branch is done,
     * and a convenience API to replace the typical use-cases for
     * mutable names should be implemented.
     *
     * if (DEPRECATE("Setting NumPy dtype names is deprecated, the dtype "
     *                "will become immutable in a future version") < 0) {
     *     return -1;
     * }
     */

    HPy names = HPyField_Load(ctx, self, self_struct->names);
    N = HPy_Length(ctx, names);
    if (!HPySequence_Check(ctx, val) || HPy_Length(ctx, val) != N) {
        HPyErr_Format_p(ctx, ctx->h_ValueError,
                "must replace all names at once with a sequence of length %d",
                N);
        return -1;
    }
    /* Make sure all entries are strings */
    for (i = 0; i < N; i++) {
        HPy item;
        int valid = 1;
        item = HPy_GetItem_i(ctx, val, i);
        valid = HPyUnicode_Check(ctx, item);
        HPy_Close(ctx, item);
        if (!valid) {
            
            HPyErr_Format_p(ctx, ctx->h_ValueError,
                    "item #%d of names is not string", i);
                    // TODO:
                    // "item #%d of names is of type %s and not string",
                    // i, Py_TYPE(item)->tp_name);
            return -1;
        }
    }
    /* Invalidate cached hash value */
    self_struct->hash = -1;
    /* Update dictionary keys in fields */
    new_names = val;
    // new_names = PySequence_Tuple(val);
    // if (HPy_IsNull(new_names)) {
    //     return -1;
    // }
    new_fields = HPyDict_New(ctx);
    if (HPy_IsNull(new_fields)) {
        // HPy_Close(ctx, new_names);
        return -1;
    }
    HPy fields = HPy_FromPyObject(ctx, self_struct->fields);
    for (i = 0; i < N; i++) {
        HPy key;
        HPy item;
        HPy new_key;
        int ret;
        key = HPy_GetItem_i(ctx, names, i);
        /* Borrowed references to item and new_key */
        item = HPyDict_GetItemWithError(ctx, fields, key);
        if (HPy_IsNull(item)) {
            if (!HPyErr_Occurred(ctx)) {
                /* fields was missing the name it claimed to contain */
                CAPI_WARN("missing PyErr_BadInternalCall");
                PyErr_BadInternalCall();
            }
            HPy_Close(ctx, fields);
            HPy_Close(ctx, new_fields);
            return -1;
        }
        new_key = HPy_GetItem_i(ctx, new_names, i);
        /* Check for duplicates */
        ret = HPy_Contains(ctx, new_fields, new_key);
        if (ret < 0) {
            HPy_Close(ctx, fields);
            HPy_Close(ctx, new_fields);
            return -1;
        }
        else if (ret != 0) {
            HPyErr_SetString(ctx, ctx->h_ValueError, "Duplicate field names given.");
            HPy_Close(ctx, fields);
            HPy_Close(ctx, new_fields);
            return -1;
        }
        if (HPy_SetItem(ctx, new_fields, new_key, item) < 0) {
            HPy_Close(ctx, fields);
            HPy_Close(ctx, new_fields);
            return -1;
        }
    }

    /* Replace names */
    HPyField_Store(ctx, self, &self_struct->names, new_names);
    HPy_Close(ctx, names);

    /* Replace fields */
    Py_DECREF(self_struct->fields);
    self_struct->fields = HPy_AsPyObject(ctx, new_fields);
    HPy_Close(ctx, new_fields);
    return 0;
}

HPyDef_SLOT(arraydescr_new, HPy_tp_new)
static HPy
arraydescr_new_impl(HPyContext *ctx, HPy subtype, const HPy *args,
                           HPy_ssize_t nargs, HPy kwds)
{
    if (!HPyGlobal_Is(ctx, subtype, HPyArrayDescr_Type)) {
        HPy subtype_type = HPy_Type(ctx, subtype);
        PyArray_DTypeMeta *DType = PyArray_DTypeMeta_AsStruct(ctx, subtype);
        PyTypeObject *subtype_typeobj = HPy_AsPyObject(ctx, subtype);
        if (HPyGlobal_Is(ctx, subtype_type, HPyArrayDTypeMeta_Type) &&
                (HNPY_DT_SLOTS(ctx, subtype)) != NULL &&
                !NPY_DT_is_legacy(DType) &&
                subtype_typeobj->tp_new != PyArrayDescr_Type.tp_new) {
            /*
             * Appears to be a properly initialized user DType. Allocate
             * it and initialize the main part as best we can.
             * TODO: This should probably be a user function, and enforce
             *       things like the `elsize` being correctly set.
             * TODO: This is EXPERIMENTAL API!
             */
            // PyArray_DTypeMeta *DType = (PyArray_DTypeMeta *)subtype;
            // PyArray_Descr *descr = (PyArray_Descr *)subtype->tp_alloc(subtype, 0);
            PyArray_Descr *descr;
            HPy h_descr = HPy_New(ctx, subtype, &descr);
            if (HPy_IsNull(h_descr) == 0) {
                HPyErr_NoMemory(ctx);
                return HPy_NULL;
            }
            // CAPI_WARN("missing PyObject_Init");
            // PyObject *py_descr = HPy_AsPyObject(ctx, h_descr);
            // PyObject_Init((PyObject *)descr, subtype_typeobj);
            descr->f = &NPY_DT_SLOTS(DType)->f;
            //Py_XINCREF(DType->scalar_type);
            HPy h_DType = HPy_FromPyObject(ctx,(PyObject *)DType);
            HPy h_scalar_type = HPyField_Load(ctx, h_DType, DType->scalar_type);
            HPy_Close(ctx, h_DType);
            HPyField_Store(ctx, h_descr, &descr->typeobj, h_scalar_type);
            HPy_Close(ctx, h_scalar_type);
            descr->type_num = DType->type_num;
            descr->flags = NPY_USE_GETITEM|NPY_USE_SETITEM;
            descr->byteorder = '|';  /* If DType uses it, let it override */
            descr->elsize = -1;  /* Initialize to invalid value */
            descr->hash = -1;
            return h_descr;
        }
        /* The DTypeMeta class should prevent this from happening. */
        // PyErr_Format(PyExc_SystemError,
        //         "'%S' must not inherit np.dtype.__new__(). User DTypes should "
        //         "currently call `PyArrayDescr_Type.tp_new` from their new.",
        //         subtype);
        HPyErr_SetString(ctx, ctx->h_SystemError,
                "'%S' must not inherit np.dtype.__new__(). User DTypes should "
                "currently call `PyArrayDescr_Type.tp_new` from their new.");
        return HPy_NULL;
    }

    HPy odescr, metadata = HPy_NULL;
    HPy conv; // PyArray_Descr *
    npy_bool align = NPY_FALSE;
    npy_bool copy = NPY_FALSE;
    npy_bool copied = NPY_FALSE;

    static const char *kwlist[] = {"dtype", "align", "copy", "metadata", NULL};
    HPy h_align = HPy_NULL, h_copy = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywordsDict(ctx, &ht, args, nargs, kwds, "O|OOO:dtype", kwlist,
                &odescr,
                &h_align,
                &h_copy,
                &metadata)) {
        return HPy_NULL;
    }

    if (HPyArray_BoolConverter(ctx, h_align, &align) != NPY_SUCCEED ||
            HPyArray_BoolConverter(ctx, h_copy, &copy) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "dtype: TODO");
        return HPy_NULL;
    }
    if (!HPy_IsNull(metadata) && !HPyDict_Check(ctx, metadata)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "dtype: TODO");
        return HPy_NULL;
    }

    conv = _hpy_convert_from_any(ctx, odescr, align);
    if (HPy_IsNull(conv)) {
        return HPy_NULL;
    }
    PyArray_Descr *conv_struct = PyArray_Descr_AsStruct(ctx, conv);

    /* Get a new copy of it unless it's already a copy */
    if (copy && conv_struct->fields == Py_None) {
        HPyArray_DESCR_REPLACE(ctx, conv, conv_struct);
        if (HPy_IsNull(conv)) {
            return HPy_NULL;
        }
        copied = NPY_TRUE;
    }

    if ((!HPy_IsNull(metadata))) {
        /*
         * We need to be sure to make a new copy of the data-type and any
         * underlying dictionary
         */
        if (!copied) {
            HPyArray_DESCR_REPLACE(ctx, conv, conv_struct);
            if (HPy_IsNull(conv)) {
                return HPy_NULL;
            }
            copied = NPY_TRUE;
        }
        if ((conv_struct->metadata != NULL)) {
            /*
             * Make a copy of the metadata before merging with the
             * input metadata so that this data-type descriptor has
             * it's own copy
             */
            /* Save a reference */
            CAPI_WARN("missing PyDict_Copy & PyDict_Merge");
            PyObject *py_odescr = conv_struct->metadata;
            conv_struct->metadata = PyDict_Copy(py_odescr);
            /* Decrement the old reference */
            Py_DECREF(py_odescr);

            /*
             * Update conv->metadata with anything new in metadata
             * keyword, but do not over-write anything already there
             */
            PyObject *py_metadata = HPy_AsPyObject(ctx, metadata);
            if (PyDict_Merge(conv_struct->metadata, py_metadata, 0) != 0) {
                Py_DECREF(py_metadata);
                HPy_Close(ctx, conv);
                return HPy_NULL;
            }
            Py_DECREF(py_metadata);
        }
        else {
            /* Make a copy of the input dictionary */
            PyObject *py_metadata = HPy_AsPyObject(ctx, metadata);
            conv_struct->metadata = PyDict_Copy(py_metadata);
            Py_DECREF(py_metadata);
        }
    }

    return conv;
}


/*
 * Return a tuple of
 * (cleaned metadata dictionary, tuple with (str, num))
 */
static HPy
_hpy_get_pickleabletype_from_datetime_metadata(HPyContext *ctx,
                                            HPy /* PyArray_Descr * */ dtype)
{
    PyArray_DatetimeMetaData *meta;

    /* Create the 2-item tuple to return */

    PyArray_Descr *dtype_struct = PyArray_Descr_AsStruct(ctx, dtype);
    /* Store the metadata dictionary */
    HPy ret_0;
    if (dtype_struct->metadata != NULL) {
        // Py_INCREF(dtype->metadata);
        ret_0 = HPy_FromPyObject(ctx, dtype_struct->metadata);
    } else {
        ret_0 = HPyDict_New(ctx);
    }

    /* Convert the datetime metadata into a tuple */
    meta = h_get_datetime_metadata_from_dtype(ctx, dtype_struct);
    if (meta == NULL) {
        HPy_Close(ctx, ret_0);
        return HPy_NULL;
    }
    /* Use a 4-tuple that numpy 1.6 knows how to unpickle */
    HPy dt_tuple_0 = HPyBytes_FromString(ctx, _datetime_strings[meta->base]);
    HPy dt_tuple_1 = HPyLong_FromLong(ctx, meta->num);
    HPy dt_tuple_2 = HPyLong_FromLong(ctx, 1);
    HPy dt_tuple_3 = HPyLong_FromLong(ctx, 1);
    HPy ret_1 = HPyTuple_Pack(ctx, 4, dt_tuple_0, dt_tuple_1, dt_tuple_2, dt_tuple_3);
    HPy_Close(ctx, dt_tuple_0);
    HPy_Close(ctx, dt_tuple_1);
    HPy_Close(ctx, dt_tuple_2);
    HPy_Close(ctx, dt_tuple_3);
    HPy ret = HPyTuple_Pack(ctx, 2, ret_0, ret_1);
    HPy_Close(ctx, ret_0);
    HPy_Close(ctx, ret_1);
    return ret;
}

/*
 * return a tuple of (callable object, args, state).
 *
 * TODO: This method needs to change so that unpickling doesn't
 *       use __setstate__. This is required for the dtype
 *       to be an immutable object.
 */
HPyDef_METH(arraydescr_reduce, "__reduce__", HPyFunc_NOARGS)
static HPy
arraydescr_reduce_impl(HPyContext *ctx, HPy self)
{
    /*
     * version number of this pickle type. Increment if we need to
     * change the format. Be sure to handle the old versions in
     * arraydescr_setstate.
    */
    const int version = 4;
    HPy ret, mod, obj;
    HPy state;
    char endian;
    int elsize, alignment;

    mod = HPyImport_ImportModule(ctx, "numpy.core._multiarray_umath");
    if (HPy_IsNull(mod)) {
        return HPy_NULL;
    }
    obj = HPy_GetAttr_s(ctx, mod, "dtype");
    HPy_Close(ctx, mod);
    if (HPy_IsNull(obj)) {
        return HPy_NULL;
    }
    HPy ret_0 = obj;
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    obj = HPyField_Load(ctx, self, self_struct->typeobj);
    HPy voidarrtype_type = HPyGlobal_Load(ctx, HPyVoidArrType_Type);
    if (PyTypeNum_ISUSERDEF(self_struct->type_num)
            || ((self_struct->type_num == NPY_VOID
                    && !HPy_Is(ctx, obj, voidarrtype_type)))) {
        // obj = (PyObject *)self_struct->typeobj;
        // Py_INCREF(obj);
        // nothing to do here.. pass through
    }
    else {
        elsize = self_struct->elsize;
        if (self_struct->type_num == NPY_UNICODE) {
            elsize >>= 2;
        }
        obj = HPyUnicode_FromFormat_p(ctx, "%c%d",self_struct->kind, elsize);
    }
    HPy ret_1 = HPyTuple_Pack(ctx, 3, obj, ctx->h_False, ctx->h_True);
    HPy_Close(ctx, obj);

    /*
     * Now return the state which is at least byteorder,
     * subarray, and fields
     */
    endian = self_struct->byteorder;
    if (endian == '=') {
        endian = '<';
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    HPy state_0;
    HPy state_8 = HPy_NULL;
    if (PyDataType_ISDATETIME(self_struct)) {
        HPy newobj;
        state_0 = HPyLong_FromLong(ctx, version);
        /*
         * newobj is a tuple of the Python metadata dictionary
         * and tuple of date_time info (str, num)
         */
        newobj = _hpy_get_pickleabletype_from_datetime_metadata(ctx, self);
        if (HPy_IsNull(newobj)) {
            return HPy_NULL;
        }
        state_8 = newobj;
    }
    else if (self_struct->metadata) {
        state_0 = HPyLong_FromLong(ctx, version);
        Py_INCREF(self_struct->metadata);
        state_8 = HPy_FromPyObject(ctx, self_struct->metadata);
    }
    else { /* Use version 3 pickle format */
        state_0 = HPyLong_FromLong(ctx, 3);
    }

    HPy state_1 = HPyUnicode_FromFormat_p(ctx, "%c", endian);
    HPy state_2 = arraydescr_subdescr_get(ctx, self, NULL);
    HPy names;
    HPy fields;
    if (PyDataType_HASFIELDS(self_struct)) {
        names = HPyField_Load(ctx, self, self_struct->names);
        fields = HPy_FromPyObject(ctx, self_struct->fields);
    } else {
        names = HPy_Dup(ctx, ctx->h_None);
        fields = HPy_Dup(ctx, ctx->h_None);
    }
    HPy state_3 = names;
    HPy state_4 = fields;

    /* for extended types it also includes elsize and alignment */
    if (PyTypeNum_ISEXTENDED(self_struct->type_num)) {
        elsize = self_struct->elsize;
        alignment = self_struct->alignment;
    }
    else {
        elsize = -1;
        alignment = -1;
    }
    HPy state_5 = HPyLong_FromLong(ctx, elsize);
    HPy state_6 = HPyLong_FromLong(ctx, alignment);
    HPy state_7 = HPyLong_FromLong(ctx, self_struct->flags);
    if (HPy_IsNull(state_8)) {
        state = HPyTuple_Pack(ctx, 8, state_0, state_1,
                                      state_2, state_3,
                                      state_4, state_5,
                                      state_6, state_7);
    } else {
        state = HPyTuple_Pack(ctx, 9, state_0, state_1,
                                      state_2, state_3,
                                      state_4, state_5,
                                      state_6, state_7, state_8);
        HPy_Close(ctx, state_8);
    }
    HPy_Close(ctx, state_0);
    HPy_Close(ctx, state_1);
    HPy_Close(ctx, state_2);
    HPy_Close(ctx, state_3);
    HPy_Close(ctx, state_4);
    HPy_Close(ctx, state_5);
    HPy_Close(ctx, state_6);
    HPy_Close(ctx, state_7);
    HPy ret_2 = state;
    ret = HPyTuple_Pack(ctx, 3, ret_0, ret_1, ret_2);
    HPy_Close(ctx, ret_0);
    HPy_Close(ctx, ret_1);
    HPy_Close(ctx, ret_2);

    return ret;
}

/*
 * returns NPY_OBJECT_DTYPE_FLAGS if this data-type has an object portion used
 * when setting the state because hasobject is not stored.
 */
static char
_hpy_descr_find_object(HPyContext *ctx, HPy self, PyArray_Descr *self_struct)
{
    if (self_struct->flags
            || self_struct->type_num == NPY_OBJECT
            || self_struct->kind == 'O') {
        return NPY_OBJECT_DTYPE_FLAGS;
    }
    if (PyDataType_HASFIELDS(self_struct)) {
        HPy key, value;
        HPy new;
        int offset;
        Py_ssize_t pos = 0;
        HPy fields = HPy_FromPyObject(ctx, self_struct->fields);
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
            HPy_Close(ctx, key);
            if (!HPy_ExtractDictItems_OiO(ctx, value, &new, &offset, NULL)) {
                HPyErr_Clear(ctx);
                HPy_Close(ctx, value);
               return 0;
            }
            HPy_Close(ctx, value);
            PyArray_Descr *new_struct = PyArray_Descr_AsStruct(ctx, new);
            if (_hpy_descr_find_object(ctx, new, new_struct)) {
                new_struct->flags = NPY_OBJECT_DTYPE_FLAGS;
                HPy_Close(ctx, new);
                return NPY_OBJECT_DTYPE_FLAGS;
            }
            HPy_Close(ctx, new);
        }
    }
    return 0;
}

/*
 * state is at least byteorder, subarray, and fields but could include elsize
 * and alignment for EXTENDED arrays
 */
 HPyDef_METH(arraydescr_setstate, "__setstate__", HPyFunc_VARARGS)
static HPy
arraydescr_setstate_impl(HPyContext *ctx, HPy /* PyArray_Descr * */ self, const HPy *args, size_t nargs)
{
    int elsize = -1, alignment = -1;
    int version = 4;
    char endian;
    HPy endian_obj;
    HPy subarray, fields, names = HPy_NULL, metadata=HPy_NULL;
    int incref_names = 1;
    int int_dtypeflags = 0;
    char dtypeflags;

    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    CAPI_WARN("PyArray_Descr.fields is a PyObject");
    if (self_struct->fields == Py_None) {
        return HPy_Dup(ctx, ctx->h_None);
    }
    if (nargs != 1 || !(HPyTuple_Check(ctx, args[0]))) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "bad argument to internal function");
        return HPy_NULL;
    }
    HPy args_tuple = args[0];
    HPy_ssize_t args_tuple_len = -1;
    HPy *args_t = HPy_TupleToArray(ctx, args[0], &args_tuple_len);
    switch (args_tuple_len) {
    case 9:
        if (!HPyArg_Parse(ctx, NULL, args_t, args_tuple_len, "iOOOOiiiO:__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &int_dtypeflags, &metadata)) {
            HPyErr_Clear(ctx);
            return HPy_NULL;
        }
        break;
    case 8:
        if (!HPyArg_Parse(ctx, NULL, args_t, args_tuple_len, "iOOOOiii:__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &int_dtypeflags)) {
            return HPy_NULL;
        }
        break;
    case 7:
        if (!HPyArg_Parse(ctx, NULL, args_t, args_tuple_len, "iOOOOii:__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment)) {
            return HPy_NULL;
        }
        break;
    case 6:
        if (!HPyArg_Parse(ctx, NULL, args_t, args_tuple_len, "iOOOii:__setstate__",
                    &version,
                    &endian_obj, &subarray, &fields,
                    &elsize, &alignment)) {
            return HPy_NULL;
        }
        break;
    case 5:
        version = 0;
        if (!HPyArg_Parse(ctx, NULL, args_t, args_tuple_len, "OOOii:__setstate__",
                    &endian_obj, &subarray, &fields, &elsize,
                    &alignment)) {
            return HPy_NULL;
        }
        break;
    default:
        /* raise an error */
        if (args_tuple_len > 5) {
            version = HPyLong_AsLong(ctx, args[0]);
        }
        else {
            version = -1;
        }
    }

    /*
     * If we ever need another pickle format, increment the version
     * number. But we should still be able to handle the old versions.
     */
    if (version < 0 || version > 4) {
        HPyErr_Format_p(ctx, ctx->h_ValueError,
                     "can't handle version %d of numpy.dtype pickle",
                     version);
        return HPy_NULL;
    }
    /* Invalidate cached hash value */
    self_struct->hash = -1;

    if (version == 1 || version == 0) {
        if (!HPy_Is(ctx, fields, ctx->h_None)) {
            HPy key, list;
            key = HPyLong_FromLong(ctx, -1);
            list = HPyDict_GetItemWithError(ctx, fields, key);
            if (HPy_IsNull(list)) {
                if (!HPyErr_Occurred(ctx)) {
                    /* fields was missing the name it claimed to contain */
                    PyErr_BadInternalCall();
                }
                return HPy_NULL;
            }
            // Py_INCREF(list);
            names = list;
            CAPI_WARN("missing PyDict_DelItem");
            PyObject *py_fields = HPy_AsPyObject(ctx, fields);
            PyObject *py_key = HPy_AsPyObject(ctx, key);
            PyDict_DelItem(py_fields, py_key);
            Py_DECREF(py_fields);
            Py_DECREF(py_key);
            incref_names = 0;
        }
        else {
            names = ctx->h_None;
        }
    }

    /* Parse endian */
    if (HPyUnicode_Check(ctx, endian_obj) || HPyBytes_Check(ctx, endian_obj)) {
        HPy tmp = HPy_NULL;
        const char *str;
        HPy_ssize_t len;

        if (HPyUnicode_Check(ctx, endian_obj)) {
            tmp = HPyUnicode_AsASCIIString(ctx, endian_obj);
            if (HPy_IsNull(tmp)) {
                return HPy_NULL;
            }
            endian_obj = tmp;
        }

        if ((str = HPyBytes_AsString(ctx, endian_obj)) == NULL) {
            HPy_Close(ctx, tmp);
            return HPy_NULL;
        }
        len = HPyBytes_GET_SIZE(ctx, endian_obj);
        if (len != 1) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "endian is not 1-char string in Numpy dtype unpickling");
            HPy_Close(ctx, tmp);
            return HPy_NULL;
        }
        endian = str[0];
        HPy_Close(ctx, tmp);
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "endian is not a string in Numpy dtype unpickling");
        return HPy_NULL;
    }

    if ((HPy_Is(ctx, fields, ctx->h_None) && !HPy_Is(ctx, names, ctx->h_None)) ||
        (HPy_Is(ctx, names, ctx->h_None) && !HPy_Is(ctx, fields, ctx->h_None))) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "inconsistent fields and names in Numpy dtype unpickling");
        return HPy_NULL;
    }

    if (!HPy_Is(ctx, names, ctx->h_None) && !HPyTuple_Check(ctx, names)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "non-tuple names in Numpy dtype unpickling");
        return HPy_NULL;
    }

    if (!HPy_Is(ctx, fields, ctx->h_None) && !HPyDict_Check(ctx, fields)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "non-dict fields in Numpy dtype unpickling");
        return HPy_NULL;
    }

    if (endian != '|' && PyArray_IsNativeByteOrder(endian)) {
        endian = '=';
    }
    self_struct->byteorder = endian;
    if (self_struct->subarray) {
        Py_XDECREF(self_struct->subarray->base);
        Py_XDECREF(self_struct->subarray->shape);
        PyArray_free(self_struct->subarray);
    }
    self_struct->subarray = NULL;

    if (!HPy_Is(ctx, subarray, ctx->h_None)) {
        HPy subarray_shape;

        /*
         * Ensure that subarray[0] is an ArrayDescr and
         * that subarray_shape obtained from subarray[1] is a tuple of integers.
         */
        HPy item = HPy_NULL;
        if (!(HPyTuple_Check(ctx, subarray) && HPy_Length(ctx, subarray) == 2 && 
                (!HPy_IsNull(item = HPy_GetItem_i(ctx, subarray, 0)) && HPyArray_DescrCheck(ctx, item)))) {
            HPy_Close(ctx, item);
            HPyErr_SetString(ctx, ctx->h_ValueError,
                        "incorrect subarray in __setstate__");
            return HPy_NULL;
        }
        HPy_Close(ctx, item);
        subarray_shape = HPy_GetItem_i(ctx, subarray, 1);
        if (HPyNumber_Check(ctx, subarray_shape)) {
            CAPI_WARN("missing PyNumber_Long");
            PyObject *py_subarray_shape = HPy_AsPyObject(ctx, subarray_shape);
            PyObject *tmp = PyNumber_Long(py_subarray_shape);
            Py_DECREF(py_subarray_shape);
            if (tmp == NULL) {
                return HPy_NULL;
            }
            HPy h_tmp = HPy_FromPyObject(ctx, tmp);
            Py_DECREF(tmp);
            subarray_shape = HPyTuple_Pack(ctx, 1, h_tmp);
            HPy_Close(ctx, h_tmp);
            if (HPy_IsNull(subarray_shape)) {
                return HPy_NULL;
            }
        }
        else if (_hpy_is_tuple_of_integers(ctx, subarray_shape)) {
            // Py_INCREF(subarray_shape);
        }
        else {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                         "incorrect subarray shape in __setstate__");
            return HPy_NULL;
        }

        self_struct->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (!PyDataType_HASSUBARRAY(self_struct)) {
            return HPyErr_NoMemory(ctx);
        }
        HPy base = HPy_GetItem_i(ctx, subarray, 0);
        self_struct->subarray->base = (PyArray_Descr *)HPy_AsPyObject(ctx, base);
        // Py_INCREF(self_struct->subarray->base);
        self_struct->subarray->shape = HPy_AsPyObject(ctx, subarray_shape);
    }

    if (!HPy_Is(ctx, fields, ctx->h_None)) {
        /*
         * Ensure names are of appropriate string type
         */
        HPy_ssize_t i;
        int names_ok = 1;
        HPy name;

        for (i = 0; i < HPy_Length(ctx, names); ++i) {
            name = HPy_GetItem_i(ctx, names, i);
            if (!HPyUnicode_Check(ctx, name)) {
                names_ok = 0;
                break;
            }
        }

        if (names_ok) {
            Py_XDECREF(self_struct->fields);
            self_struct->fields = HPy_AsPyObject(ctx, fields);
            // Py_INCREF(fields);
            HPyField_Store(ctx, self, &self_struct->names, names);
            if (!incref_names) {
                HPy_Close(ctx, names);
            }
        }
        else {
            /*
             * To support pickle.load(f, encoding='bytes') for loading Py2
             * generated pickles on Py3, we need to be more lenient and convert
             * field names from byte strings to unicode.
             */
            HPy tmp, new_name, field;

            tmp = HPyDict_New(ctx);
            if (HPy_IsNull(tmp)) {
                return HPy_NULL;
            }
            Py_XDECREF(self_struct->fields);
            self_struct->fields = HPy_AsPyObject(ctx, tmp);

            HPy_ssize_t names_len = HPy_Length(ctx, names);
            HPyTupleBuilder tmp_tb = HPyTupleBuilder_New(ctx, names_len);
            if (HPyTupleBuilder_IsNull(tmp_tb)) {
                return HPy_NULL;
            }

            for (i = 0; i < names_len; ++i) {
                name = HPy_GetItem_i(ctx, names, i);
                field = HPyDict_GetItemWithError(ctx, fields, name);
                if (HPy_IsNull(field)) {
                    HPy_Close(ctx, name);
                    HPyTupleBuilder_Cancel(ctx, tmp_tb);
                    if (!HPyErr_Occurred(ctx)) {
                        /* fields was missing the name it claimed to contain */
                        PyErr_BadInternalCall();
                    }
                    return HPy_NULL;
                }

                if (HPyUnicode_Check(ctx, name)) {
                    new_name = name;
                    // Py_INCREF(new_name);
                }
                else {
                    new_name = HPyUnicode_FromEncodedObject(ctx, name, "ASCII", "strict");
                    HPy_Close(ctx, name);
                    if (HPy_IsNull(new_name)) {
                        HPyTupleBuilder_Cancel(ctx, tmp_tb);
                        return HPy_NULL;
                    }
                }

                HPyTupleBuilder_Set(ctx, tmp_tb, i, new_name);
                HPy fields = HPy_FromPyObject(ctx, self_struct->fields);
                if (HPy_SetItem(ctx, fields, new_name, field) != 0) {
                    HPyTupleBuilder_Cancel(ctx, tmp_tb);
                    return HPy_NULL;
                }
            }
            HPyField_Store(ctx, self, &self_struct->names, HPyTupleBuilder_Build(ctx, tmp_tb));
            // Py_DECREF(tmp);
        }
    }

    if (PyTypeNum_ISEXTENDED(self_struct->type_num)) {
        self_struct->elsize = elsize;
        self_struct->alignment = alignment;
    }

    /*
     * We use an integer converted to char for backward compatibility with
     * pickled arrays. Pickled arrays created with previous versions encoded
     * flags as an int even though it actually was a char in the PyArray_Descr
     * structure
     */
    dtypeflags = int_dtypeflags;
    if (dtypeflags != int_dtypeflags) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                     "incorrect value for flags variable (overflow)");
        return HPy_NULL;
    }
    else {
        self_struct->flags = dtypeflags;
    }

    if (version < 3) {
        self_struct->flags = _hpy_descr_find_object(ctx, self, self_struct);
    }

    /*
     * We have a borrowed reference to metadata so no need
     * to alter reference count when throwing away Py_None.
     */
    if (HPy_Is(ctx, metadata, ctx->h_None)) {
        metadata = HPy_NULL;
    }

    if (PyDataType_ISDATETIME(self_struct) && !HPy_IsNull(metadata)) {
        PyArray_DatetimeMetaData temp_dt_data;

        if ((!HPyTuple_Check(ctx, metadata)) || (HPy_Length(ctx, metadata) != 2)) {
            // HPyErr_Format(ctx, ctx->h_ValueError,
            //         "Invalid datetime dtype (metadata, c_metadata): %R",
            //         metadata);
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Invalid datetime dtype (metadata, c_metadata): %R");
            return HPy_NULL;
        }
        HPy item = HPy_GetItem_i(ctx, metadata, 1);
        PyObject *py_itme = HPy_AsPyObject(ctx, item);
        HPy_Close(ctx, item);
        if (convert_datetime_metadata_tuple_to_datetime_metadata(
                                    py_itme,
                                    &temp_dt_data,
                                    NPY_TRUE) < 0) {
            Py_DECREF(py_itme);
            return HPy_NULL;
        }
        Py_DECREF(py_itme);

        PyObject *old_metadata = self_struct->metadata;
        item = HPy_GetItem_i(ctx, metadata, 0);
        self_struct->metadata = HPy_AsPyObject(ctx, item);
        HPy_Close(ctx, item);
        memcpy((char *) &((PyArray_DatetimeDTypeMetaData *)self_struct->c_metadata)->meta,
               (char *) &temp_dt_data,
               sizeof(PyArray_DatetimeMetaData));
        // Py_XINCREF(self_struct->metadata);
        Py_XDECREF(old_metadata);
    }
    else {
        PyObject *old_metadata = self_struct->metadata;
        self_struct->metadata = HPy_AsPyObject(ctx, metadata);
        // Py_XINCREF(self_struct->metadata);
        Py_XDECREF(old_metadata);
    }

    return HPy_Dup(ctx, ctx->h_None);
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to DEFAULT type.
 *
 * any object with the .fields attribute and/or .itemsize attribute (if the
 *.fields attribute does not give the total size -- i.e. a partial record
 * naming).  If itemsize is given it must be >= size computed from fields
 *
 * The .fields attribute must return a convertible dictionary if present.
 * Result inherits from NPY_VOID.
*/
NPY_NO_EXPORT int
PyArray_DescrAlignConverter(PyObject *obj, PyArray_Descr **at)
{
    HPyContext *ctx = npy_get_context();
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPy h_ret = _hpy_convert_from_any(ctx, h_obj, 1);
    PyObject *py_ret = HPy_AsPyObject(ctx, h_ret);
    *at = (PyArray_Descr *)py_ret;
    HPy_Close(ctx, h_obj);
    HPy_Close(ctx, h_ret);
    return (*at) ? NPY_SUCCEED : NPY_FAIL;
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to NULL.
 */
NPY_NO_EXPORT int
PyArray_DescrAlignConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (obj == Py_None) {
        *at = NULL;
        return NPY_SUCCEED;
    }
    else {
        return PyArray_DescrAlignConverter(obj, at);
    }
}



/*NUMPY_API
 *
 * returns a copy of the PyArray_Descr structure with the byteorder
 * altered:
 * no arguments:  The byteorder is swapped (in all subfields as well)
 * single argument:  The byteorder is forced to the given state
 * (in all subfields as well)
 *
 * Valid states:  ('big', '>') or ('little' or '<')
 * ('native', or '=')
 *
 * If a descr structure with | is encountered it's own
 * byte-order is not changed but any fields are:
 *
 *
 * Deep bytorder change of a data-type descriptor
 * *** Leaves reference count of self unchanged --- does not DECREF self ***
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewByteorder(PyArray_Descr *self, char newendian)
{
    PyArray_Descr *new;
    char endian;

    new = PyArray_DescrNew(self);
    if (new == NULL) {
        return NULL;
    }
    endian = new->byteorder;
    if (endian != NPY_IGNORE) {
        if (newendian == NPY_SWAP) {
            /* swap byteorder */
            if (PyArray_ISNBO(endian)) {
                endian = NPY_OPPBYTE;
            }
            else {
                endian = NPY_NATBYTE;
            }
            new->byteorder = endian;
        }
        else if (newendian != NPY_IGNORE) {
            new->byteorder = newendian;
        }
    }
    if (PyDataType_HASFIELDS(new)) {
        PyObject *newfields;
        PyObject *key, *value;
        PyObject *newvalue;
        PyObject *old;
        PyArray_Descr *newdescr;
        Py_ssize_t pos = 0;
        int len, i;

        newfields = PyDict_New();
        if (newfields == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        /* make new dictionary with replaced PyArray_Descr Objects */
        while (PyDict_Next(self->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyUnicode_Check(key) || !PyTuple_Check(value) ||
                ((len=PyTuple_GET_SIZE(value)) < 2)) {
                continue;
            }
            old = PyTuple_GET_ITEM(value, 0);
            if (!PyArray_DescrCheck(old)) {
                continue;
            }
            newdescr = PyArray_DescrNewByteorder(
                    (PyArray_Descr *)old, newendian);
            if (newdescr == NULL) {
                Py_DECREF(newfields); Py_DECREF(new);
                return NULL;
            }
            newvalue = PyTuple_New(len);
            PyTuple_SET_ITEM(newvalue, 0, (PyObject *)newdescr);
            for (i = 1; i < len; i++) {
                old = PyTuple_GET_ITEM(value, i);
                Py_INCREF(old);
                PyTuple_SET_ITEM(newvalue, i, old);
            }
            int ret = PyDict_SetItem(newfields, key, newvalue);
            Py_DECREF(newvalue);
            if (ret < 0) {
                Py_DECREF(newfields);
                Py_DECREF(new);
                return NULL;
            }
        }
        Py_DECREF(new->fields);
        new->fields = newfields;
    }
    if (PyDataType_HASSUBARRAY(new)) {
        Py_DECREF(new->subarray->base);
        new->subarray->base = PyArray_DescrNewByteorder(
                self->subarray->base, newendian);
        if (new->subarray->base == NULL) {
            Py_DECREF(new);
            return NULL;
        }
    }
    return new;
}


/*HPY_NUMPY_API
 *
 * returns a copy of the PyArray_Descr structure with the byteorder
 * altered:
 * no arguments:  The byteorder is swapped (in all subfields as well)
 * single argument:  The byteorder is forced to the given state
 * (in all subfields as well)
 *
 * Valid states:  ('big', '>') or ('little' or '<')
 * ('native', or '=')
 *
 * If a descr structure with | is encountered it's own
 * byte-order is not changed but any fields are:
 *
 *
 * Deep bytorder change of a data-type descriptor
 * *** Leaves reference count of self unchanged --- does not DECREF self ***
 */
NPY_NO_EXPORT HPy /* PyArray_Descr * */
HPyArray_DescrNewByteorder(HPyContext *ctx, HPy /* PyArray_Descr * */ self, char newendian)
{
    HPy new; // PyArray_Descr *
    char endian;

    new = HPyArray_DescrNew(ctx, self);
    if (HPy_IsNull(new)) {
        return HPy_NULL;
    }
    PyArray_Descr *new_data = PyArray_Descr_AsStruct(ctx, new);
    endian = new_data->byteorder;
    if (endian != NPY_IGNORE) {
        if (newendian == NPY_SWAP) {
            /* swap byteorder */
            if (PyArray_ISNBO(endian)) {
                endian = NPY_OPPBYTE;
            }
            else {
                endian = NPY_NATBYTE;
            }
            new_data->byteorder = endian;
        }
        else if (newendian != NPY_IGNORE) {
            new_data->byteorder = newendian;
        }
    }
    if (PyDataType_HASFIELDS(new_data)) {
        HPy newfields;
        PyObject *key, *value;
        HPy old;
        HPy newdescr; // PyArray_Descr *
        Py_ssize_t pos = 0;
        int len, i;

        newfields = HPyDict_New(ctx);
        if (HPy_IsNull(newfields)) {
            HPy_Close(ctx, new);
            return HPy_NULL;
        }
        /* make new dictionary with replaced PyArray_Descr Objects */
        HPy fields = HPy_FromPyObject(ctx, new_data->fields);
        HPy keys = HPyDict_Keys(ctx, fields);
        HPy_ssize_t keys_len = HPy_Length(ctx, keys);
        for (HPy_ssize_t i_key = 0; i_key < keys_len; i_key++) {
            HPy h_key = HPy_GetItem_i(ctx, keys, i_key);
            HPy h_value = HPy_GetItem(ctx, fields, h_key);
            if (HNPY_TITLE_KEY(ctx, h_key, h_value)) {
                HPy_Close(ctx, h_key);
                HPy_Close(ctx, h_value);
                continue;
            }
            if (!HPyUnicode_Check(ctx, h_key) || !HPyTuple_Check(ctx, h_value) ||
                ((len=HPy_Length(ctx, h_value)) < 2)) {
                HPy_Close(ctx, h_value);
                HPy_Close(ctx, h_key);
                continue;
            }
            old = HPy_GetItem_i(ctx, h_value, 0);
            HPy_Close(ctx, h_value);
            if (!HPyArray_DescrCheck(ctx, old)) {
                HPy_Close(ctx, h_key);
                continue;
            }
            newdescr = HPyArray_DescrNewByteorder(ctx, old, newendian);
            if (HPy_IsNull(newdescr)) {
                HPy_Close(ctx, h_key);
                HPy_Close(ctx, newfields); 
                HPy_Close(ctx, new);
                return HPy_NULL;
            }
            HPyTupleBuilder tb_newvalue = HPyTupleBuilder_New(ctx, len);
            HPyTupleBuilder_Set(ctx, tb_newvalue, 0, newdescr);
            for (i = 1; i < len; i++) {
                old = HPy_GetItem_i(ctx, h_value, i);
                HPyTupleBuilder_Set(ctx, tb_newvalue, i, old);
                HPy_Close(ctx, old);
            }
            HPy newvalue = HPyTupleBuilder_Build(ctx, tb_newvalue);
            if (HPy_IsNull(newvalue)) {
                HPy_Close(ctx, h_key);
                HPy_Close(ctx, newvalue);
                HPy_Close(ctx, newfields);
                HPy_Close(ctx, new);
                return HPy_NULL;
            }
            int ret = HPy_SetItem(ctx, newfields, h_key, newvalue);
            HPy_Close(ctx, h_key);
            HPy_Close(ctx, newvalue);
            if (ret < 0) {
                HPy_Close(ctx, newfields);
                HPy_Close(ctx, new);
                return HPy_NULL;
            }
        }
        Py_DECREF(new_data->fields);
        new_data->fields = HPy_AsPyObject(ctx, newfields);
    }
    if (PyDataType_HASSUBARRAY(new_data)) {
        Py_DECREF(new_data->subarray->base);
        PyArray_Descr *self_data = PyArray_Descr_AsStruct(ctx, self);
        CAPI_WARN("using subarray->base");
        HPy base = HPy_FromPyObject(ctx, self_data->subarray->base);
        HPy byteorder = HPyArray_DescrNewByteorder(ctx, base, newendian);
        HPy_Close(ctx, base);
        new_data->subarray->base = HPy_AsPyObject(ctx, byteorder);
        if (new_data->subarray->base == NULL) {
            HPy_Close(ctx, new);
            return HPy_NULL;
        }
        HPy_Close(ctx, byteorder);
    }
    return new;
}

HPyDef_METH(arraydescr_newbyteorder, "newbyteorder", HPyFunc_VARARGS)
static HPy
arraydescr_newbyteorder_impl(HPyContext *ctx, HPy /* PyArray_Descr * */ self, const HPy *args, size_t nargs)
{
    char endian=NPY_SWAP;
    HPy h_endian = HPy_NULL;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "|O:newbyteorder", &h_endian)) {
        return HPy_NULL;
    }
    if (!HPy_IsNull(h_endian) && HPyArray_ByteorderConverter(ctx, h_endian, &endian) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "newbyteorder: TODO");
        return HPy_NULL;
    }
    return HPyArray_DescrNewByteorder(ctx, self, endian);
}

static PyObject *
arraydescr_class_getitem(PyObject *cls, PyObject *args)
{
    PyObject *generic_alias;

#ifdef Py_GENERICALIASOBJECT_H
    Py_ssize_t args_len;

    args_len = PyTuple_Check(args) ? PyTuple_Size(args) : 1;
    if (args_len != 1) {
        return PyErr_Format(PyExc_TypeError,
                            "Too %s arguments for %s",
                            args_len > 1 ? "many" : "few",
                            ((PyTypeObject *)cls)->tp_name);
    }
    generic_alias = Py_GenericAlias(cls, args);
#else
    PyErr_SetString(PyExc_TypeError,
                    "Type subscription requires python >= 3.9");
    generic_alias = NULL;
#endif
    return generic_alias;
}

static PyMethodDef arraydescr_methods[] = {
    /* for typing; requires python >= 3.9 */
    {"__class_getitem__",
        (PyCFunction)arraydescr_class_getitem,
        METH_CLASS | METH_O, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

/*
 * Checks whether the structured data type in 'dtype'
 * has a simple layout, where all the fields are in order,
 * and follow each other with no alignment padding.
 *
 * When this returns true, the dtype can be reconstructed
 * from a list of the field names and dtypes with no additional
 * dtype parameters.
 *
 * Returns 1 if it has a simple layout, 0 otherwise.
 */
NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype)
{
    PyObject *names, *fields, *key, *tup, *title;
    Py_ssize_t i, names_size;
    PyArray_Descr *fld_dtype;
    int fld_offset;
    npy_intp total_offset;
    int res;

    /* Get some properties from the dtype */
    names = HPyField_LoadPyObj((PyObject *)dtype, dtype->names);
    names_size = PyTuple_GET_SIZE(names);
    fields = dtype->fields;

    /* Start at offset zero */
    total_offset = 0;

    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        if (key == NULL) {
            res = 0;
            goto finish;
        }
        tup = PyDict_GetItem(fields, key);
        if (tup == NULL) {
            res = 0;
            goto finish;
        }
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &fld_offset, &title)) {
            PyErr_Clear();
            res = 0;
            goto finish;
        }
        /* If this field doesn't follow the pattern, not a simple layout */
        if (total_offset != fld_offset) {
            res = 0;
            goto finish;
        }
        /* Get the next offset */
        total_offset += fld_dtype->elsize;
    }

    /*
     * If the itemsize doesn't match the final offset, it's
     * not a simple layout.
     */
    if (total_offset != dtype->elsize) {
        res = 0;
        goto finish;
    }

    /* It's a simple layout, since all the above tests passed */
    res = 1;
finish:
    Py_DECREF(names);
    return res;
}

/*
 * The general dtype repr function.
 */
HPyDef_SLOT(arraydescr_repr, HPy_tp_repr)
static HPy
arraydescr_repr_impl(HPyContext *ctx, HPy dtype)
{
    HPy _numpy_dtype;
    HPy res;
    _numpy_dtype = HPyImport_ImportModule(ctx, "numpy.core._dtype");
    if (HPy_IsNull(_numpy_dtype)) {
        return HPy_NULL;
    }
    HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
    HPy args = HPyTuple_Pack(ctx, 1, dtype);
    HPy meth = HPy_GetAttr_s(ctx, _numpy_dtype, "__repr__");
    res = HPy_CallTupleDict(ctx, meth, args, HPy_NULL);
    HPy_Close(ctx, args);
    HPy_Close(ctx, meth);
    HPy_Close(ctx, _numpy_dtype);
    return res;
}
/*
 * The general dtype str function.
 */
static PyObject *
arraydescr_str(PyArray_Descr *dtype)
{
    PyObject *_numpy_dtype;
    PyObject *res;
    _numpy_dtype = PyImport_ImportModule("numpy.core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype, "__str__", "O", dtype);
    Py_DECREF(_numpy_dtype);
    return res;
}

 HPyDef_SLOT(arraydescr_richcompare, HPy_tp_richcompare)
static HPy
arraydescr_richcompare_impl(HPyContext *ctx, HPy self, HPy other, HPy_RichCmpOp cmp_op)
{
    HPy new = _hpy_convert_from_any(ctx, other, 0); // PyArray_Descr *
    if (HPy_IsNull(new)) {
        /* Cannot convert `other` to dtype */
        HPyErr_Clear(ctx);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }

    npy_bool ret;
    switch (cmp_op) {
    case Py_LT:
        ret = !HPyArray_EquivTypes(ctx, self, new) && HPyArray_CanCastTo(ctx, self, new);
        HPy_Close(ctx, new);
        return HPyBool_FromLong(ctx, ret);
    case Py_LE:
        ret = HPyArray_CanCastTo(ctx, self, new);
        HPy_Close(ctx, new);
        return HPyBool_FromLong(ctx, ret);
    case Py_EQ:
        ret = HPyArray_EquivTypes(ctx, self, new);
        HPy_Close(ctx, new);
        return HPyBool_FromLong(ctx, ret);
    case Py_NE:
        ret = !HPyArray_EquivTypes(ctx, self, new);
        HPy_Close(ctx, new);
        return HPyBool_FromLong(ctx, ret);
    case Py_GT:
        ret = !HPyArray_EquivTypes(ctx, self, new) && HPyArray_CanCastTo(ctx, new, self);
        HPy_Close(ctx, new);
        return HPyBool_FromLong(ctx, ret);
    case Py_GE:
        ret = HPyArray_CanCastTo(ctx, new, self);
        HPy_Close(ctx, new);
        return HPyBool_FromLong(ctx, ret);
    default:
        HPy_Close(ctx, new);
        return HPy_Dup(ctx, ctx->h_NotImplemented);
    }
}

HPyDef_SLOT(descr_nonzero, HPy_nb_bool)
static int
descr_nonzero_impl(HPyContext *ctx, HPy NPY_UNUSED(self))
{
    /* `bool(np.dtype(...)) == True` for all dtypes. Needed to override default
     * nonzero implementation, which checks if `len(object) > 0`. */
    return 1;
}

/*************************************************************************
 ****************   Implement Mapping Protocol ***************************
 *************************************************************************/

HPyDef_SLOT(descr_length, HPy_mp_length)
static HPy_ssize_t
descr_length_impl(HPyContext *ctx, HPy self0)
{
    PyArray_Descr *self = PyArray_Descr_AsStruct(ctx, self0);
    if (PyDataType_HASFIELDS(self)) {
        HPy names = HPyField_Load(ctx, self0, self->names);
        HPy_ssize_t n = HPy_Length(ctx, names);
        HPy_Close(ctx, names);
        return n;
    }
    else {
        return 0;
    }
}

HPyDef_SLOT(descr_length_sq, HPy_sq_length)
static HPy_ssize_t
descr_length_sq_impl(HPyContext *ctx, HPy self0)
{
    return descr_length_impl(ctx, self0);
}

HPyDef_SLOT(descr_repeat, HPy_sq_repeat)
static HPy
descr_repeat_impl(HPyContext *ctx, HPy self, HPy_ssize_t length)
{
    HPy tup; 
    HPy new; // PyArray_Descr *
    if (length < 0) {
        HPyErr_Format_p(ctx, ctx->h_ValueError,
                "Array length must be >= 0, not %"NPY_INTP_FMT, (npy_intp)length);
        return HPy_NULL;
    }
    tup = HPy_BuildValue(ctx, "O" NPY_SSIZE_T_PYFMT, self, length);
    if (HPy_IsNull(tup)) {
        return HPy_NULL;
    }
    new = _hpy_convert_from_any(ctx, tup, 0);
    HPy_Close(ctx, tup);
    return new;
}

static int
_hpy_check_has_fields(HPyContext *ctx, 
                        HPy /* PyArray_Descr * */ self, PyArray_Descr *self_struct)
{
    if (!PyDataType_HASFIELDS(self_struct)) {
        // PyErr_Format(PyExc_KeyError, "There are no fields in dtype %S.", self);
        HPyErr_SetString(ctx, ctx->h_KeyError, "There are no fields in dtype"); // %S.", self);
        return -1;
    }
    else {
        return 0;
    }
}

static HPy
_hpy_subscript_by_name(HPyContext *ctx, 
                        HPy /* PyArray_Descr * */ self, PyArray_Descr *self_struct, 
                        HPy op)
{
    HPy fields = HPy_FromPyObject(ctx, self_struct->fields);
    HPy obj = HPyDict_GetItemWithError(ctx, fields, op);
    if (HPy_IsNull(obj)) {
        if (!HPyErr_Occurred(ctx)) {
            // PyErr_Format(PyExc_KeyError,
            //         "Field named %R not found.", op);
            HPyErr_SetString(ctx, ctx->h_KeyError,
                    "Field named %R not found.");
        }
        return HPy_NULL;
    }
    HPy descr = HPy_GetItem_i(ctx, obj, 0);
    // Py_INCREF(descr);
    return descr;
}

static HPy
_hpy_subscript_by_index(HPyContext *ctx, 
                        HPy /* PyArray_Descr * */ self, PyArray_Descr *self_struct, 
                        HPy_ssize_t i)
{
    HPy names = HPyField_Load(ctx, self, self_struct->names);
    HPy name = HPy_GetItem_i(ctx, names, i);
    HPy_Close(ctx, names);
    if (HPy_IsNull(name)) {
        HPyErr_Format_p(ctx, ctx->h_IndexError,
                     "Field index %zd out of range.", i);
        return HPy_NULL;
    }
    HPy ret = _hpy_subscript_by_name(ctx, self, self_struct, name);
    HPy_Close(ctx, name);
    return ret;
}

static npy_bool
_hpy_is_list_of_strings(HPyContext *ctx, HPy obj)
{
    int seqlen, i;
    HPy obj_type = HPy_Type(ctx, obj);
    if (!HPy_Is(ctx, obj_type, ctx->h_ListType)) {
        HPy_Close(ctx, obj_type);
        return NPY_FALSE;
    }
    HPy_Close(ctx, obj_type);
    seqlen = HPy_Length(ctx, obj);
    for (i = 0; i < seqlen; i++) {
        HPy item = HPy_GetItem_i(ctx, obj, i);
        if (!HPyUnicode_Check(ctx, item)) {
            HPy_Close(ctx, item);
            return NPY_FALSE;
        }
        HPy_Close(ctx, item);
    }

    return NPY_TRUE;
}

NPY_NO_EXPORT HPy
harraydescr_field_subset_view(HPyContext *ctx,
                    PyArray_Descr *self_data,
                    HPy ind)
{
    int seqlen, i;
    HPy fields = HPy_NULL;
    HPy names = HPy_NULL;
    HPyTupleBuilder names_tup;
    HPy view_dtype; // PyArray_Descr *

    seqlen = HPy_Length(ctx, ind);
    if (seqlen == -1) {
        return HPy_NULL;
    }

    fields = HPyDict_New(ctx);
    if (HPy_IsNull(fields)) {
        goto fail;
    }
    names_tup = HPyTupleBuilder_New(ctx, seqlen);
    if (HPyTupleBuilder_IsNull(names_tup)) {
        goto fail;
    }

    for (i = 0; i < seqlen; i++) {
        HPy name;
        HPy tup;

        name = HPy_GetItem_i(ctx, ind, i);
        if (HPy_IsNull(name)) {
            HPyTupleBuilder_Cancel(ctx, names_tup);
            goto fail;
        }

        /* Let the names tuple steal a reference now, so we don't need to
         * decref name if an error occurs further on.
         */
        HPyTupleBuilder_Set(ctx, names_tup, i, name);

        HPy h_fields = HPy_FromPyObject(ctx, self_data->fields);
        tup = HPy_GetItem(ctx, h_fields, name); // PyDict_GetItemWithError
        if (HPy_IsNull(tup)) {
            if (!HPyErr_Occurred(ctx)) {
                HPyErr_SetObject(ctx, ctx->h_KeyError, name);
            }
            HPyTupleBuilder_Cancel(ctx, names_tup);
            goto fail;
        }

        /* disallow use of titles as index */
        if (HPy_Length(ctx, tup) == 3) {
            HPy title = HPy_GetItem_i(ctx, tup, 2);
            int titlecmp = HPy_RichCompareBool(ctx, title, name, HPy_EQ);
            if (titlecmp < 0) {
                HPyTupleBuilder_Cancel(ctx, names_tup);
                goto fail;
            }
            if (titlecmp == 1) {
                /* if title == name, we were given a title, not a field name */
                HPyErr_SetString(ctx, ctx->h_KeyError,
                            "cannot use field titles in multi-field index");
                HPyTupleBuilder_Cancel(ctx, names_tup);
                goto fail;
            }
            if (HPy_SetItem(ctx, fields, title, tup) < 0) {
                HPyTupleBuilder_Cancel(ctx, names_tup);
                goto fail;
            }
        }
        /* disallow duplicate field indices */
        if (HPy_Contains(ctx, fields, name)) {
            HPy msg = HPy_NULL;
            HPy fmt = HPyUnicode_FromString(ctx,
                                   "duplicate field of name {!r}");
            if (!HPy_IsNull(fmt)) {
                HPY_PERFORMANCE_WARNING("packing args for HPy_CallTupleDict");
                HPy args = HPyTuple_Pack(ctx, 1, name);
                HPy h_format = HPy_GetAttr_s(ctx, fmt, "format");
                msg = HPy_CallTupleDict(ctx, h_format, args, HPy_NULL);
                HPy_Close(ctx, args);
                HPy_Close(ctx, h_format);
                HPy_Close(ctx, fmt);
            }
            HPyErr_SetObject(ctx, ctx->h_ValueError, msg);
            HPy_Close(ctx, msg);
            HPyTupleBuilder_Cancel(ctx, names_tup);
            goto fail;
        }
        if (HPy_SetItem(ctx, fields, name, tup) < 0) {
            HPyTupleBuilder_Cancel(ctx, names_tup);
            goto fail;
        }
    }

    view_dtype = HPyArray_DescrNewFromType(ctx, NPY_VOID);
    if (HPy_IsNull(view_dtype)) {
        HPyTupleBuilder_Cancel(ctx, names_tup);
        goto fail;
    }
    names = HPyTupleBuilder_Build(ctx, names_tup);
    PyArray_Descr *view_dtype_data = PyArray_Descr_AsStruct(ctx, view_dtype);
    view_dtype_data->elsize = self_data->elsize;
    HPyField_Store(ctx, view_dtype, &view_dtype_data->names, names);
    HPy_Close(ctx, names);
    view_dtype_data->fields = HPy_AsPyObject(ctx, fields);
    HPy_Close(ctx, fields);
    view_dtype_data->flags = self_data->flags;
    return view_dtype;

fail:
    HPy_Close(ctx, fields);
    HPy_Close(ctx, names);
    return HPy_NULL;
}

NPY_NO_EXPORT PyArray_Descr *
arraydescr_field_subset_view(PyArray_Descr *self, PyObject *ind)
{
    HPyContext *ctx = npy_get_context();
    HPy h_self = HPy_FromPyObject(ctx, (PyObject *)self);
    HPy h_ind = HPy_FromPyObject(ctx, ind);
    HPy h_ret = harraydescr_field_subset_view(ctx, PyArray_Descr_AsStruct(ctx, h_self), h_ind);
    PyArray_Descr *ret = (PyArray_Descr *)HPy_AsPyObject(ctx, h_ret);
    HPy_Close(ctx, h_self);
    HPy_Close(ctx, h_ind);
    HPy_Close(ctx, h_ret);
    return ret;
}

HPyDef_SLOT(descr_subscript, HPy_mp_subscript)
static HPy
descr_subscript_impl(HPyContext *ctx, HPy self, HPy op)
{
    PyArray_Descr *self_struct = PyArray_Descr_AsStruct(ctx, self);
    if (_hpy_check_has_fields(ctx, self, self_struct) < 0) {
        return HPy_NULL;
    }

    if (HPyUnicode_Check(ctx, op)) {
        return _hpy_subscript_by_name(ctx, self, self_struct, op);
    }
    else if (_hpy_is_list_of_strings(ctx, op)) {
        return harraydescr_field_subset_view(ctx, self_struct, op);
    }
    else {
        HPy_ssize_t i = HPyArray_PyIntAsIntp(ctx, op);
        if (hpy_error_converting(ctx, i)) {
            /* if converting to an int gives a type error, adjust the message */
            if (HPyErr_Occurred(ctx) && HPyErr_ExceptionMatches(ctx, ctx->h_TypeError)) {
                HPyErr_SetString(ctx, ctx->h_TypeError,
                        "Field key must be an integer field offset, "
                        "single field name, or list of field names.");
            }
            return HPy_NULL;
        }
        return _hpy_subscript_by_index(ctx, self, self_struct, i);
    }
}

/****************** End of Mapping Protocol ******************************/


// HPY TODO: global variable supplementing the PyArray_Descr static type
// and global C variable at the same time. We cannot use non-pointer
// PyArray_Descr once "dtype" class (PyArray_Descr) was converted to heap type.
// The access macro PyArray_Descr was changed to return this pointer refererenced
// to avoid having to change all the uses
NPY_NO_EXPORT PyTypeObject *_PyArrayDescr_Type_p;

static PyType_Slot PyArrayDescr_TypeFull_legacy_slots[] = {
    {Py_tp_str, (reprfunc)arraydescr_str},
    {Py_tp_methods, arraydescr_methods},
    {Py_tp_hash, PyArray_DescrHash},
    {0, 0},
};

static HPyDef *PyArrayDescr_TypeFull_defines[] = {
    // slots
    &arraydescr_dealloc,
    &arraydescr_repr,
    &arraydescr_richcompare,
    &arraydescr_new,
    &descr_nonzero,
    &descr_length,
    &descr_length_sq,
    &descr_repeat,
    &descr_subscript,

    // members
    &arraydescr_type,
    &arraydescr_kind,
    &arraydescr_char,
    &arraydescr_num,
    &arraydescr_byteorder,
    &arraydescr_itemsize,
    &arraydescr_alignment,
    &arraydescr_flags,

    // getsets
    &arraydescr_subdescr,
    &arraydescr_descr,
    &_arraydescr_protocol_typestr,
    &arraydescr_name,
    &arraydescr_base,
    &arraydescr_shape,
    &arraydescr_ndim,
    &arraydescr_isbuiltin,
    &arraydescr_isnative,
    &arraydescr_isalignedstruct,
    &arraydescr_fields,
    &arraydescr_metadata,
    &arraydescr_names,
    &arraydescr_hasobject,

    // methods
    /* for pickling */
    &arraydescr_reduce,
    &arraydescr_setstate,
    &arraydescr_newbyteorder,

    NULL
};

NPY_NO_EXPORT HPyType_Spec PyArrayDescr_TypeFull_spec = {
    .name = "numpy.dtype",
    .basicsize = sizeof(PyArray_Descr),
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_BASETYPE,
    .defines = PyArrayDescr_TypeFull_defines,
    .legacy_slots = PyArrayDescr_TypeFull_legacy_slots,
    .builtin_shape = SHAPE(PyArray_Descr),
};

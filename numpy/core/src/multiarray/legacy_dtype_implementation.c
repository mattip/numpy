/*
 * The only function exported here is `PyArray_LegacyCanCastTypeTo`, which
 * is currently still in use when first registering a userdtype.
 *
 * The extremely limited use means that it can probably remain unmaintained
 * until such a time where legay user dtypes are deprecated and removed
 * entirely.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "numpy/arrayobject.h"
#include "scalartypes.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "convert_datatype.h"

#include "legacy_dtype_implementation.h"


/*
 * Compare the field dictionaries for two types.
 *
 * Return 1 if the field types and field names of the two descrs are equal and
 * in the same order, 0 if not.
 */
static int
_equivalent_fields(PyArray_Descr *type1, PyArray_Descr *type2) {

    int val, res;
    PyObject *names1 = NULL, *names2 = NULL;

    if (type1->fields == type2->fields) {
        names1 = HPyField_LoadPyObj((PyObject *)type1, type1->names);
        names2 = HPyField_LoadPyObj((PyObject *)type2, type2->names);
        if (names1 == names2) {
            res = 1;
            goto finish;
        }
    }
    if (type1->fields == NULL || type2->fields == NULL) {
        res = 0;
        goto finish;
    }

    val = PyObject_RichCompareBool(type1->fields, type2->fields, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        res = 0;
        goto finish;
    }

    if (names1 == NULL) {
        names1 = HPyField_LoadPyObj((PyObject *)type1, type1->names);
    }
    if (names2 == NULL) {
        names2 = HPyField_LoadPyObj((PyObject *)type2, type2->names);
    }
    val = PyObject_RichCompareBool(names1, names2, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        res = 0;
        goto finish;
    }

    res = 1;
    // fall through

finish:
    Py_XDECREF(names1);
    Py_XDECREF(names2);
    return res;
}

static int
_hpy_equivalent_fields(HPyContext *ctx, 
                            HPy /* PyArray_Descr * */ h_type1,
                            PyArray_Descr *type1,
                            HPy /* PyArray_Descr * */ h_type2,
                            PyArray_Descr *type2) {

    int val, res;
    HPy names1 = HPy_NULL, names2 = HPy_NULL;

    if (type1->fields == type2->fields) {
        names1 = HPyField_Load(ctx, h_type1, type1->names);
        names2 = HPyField_Load(ctx, h_type2, type2->names);
        if (HPy_Is(ctx, names1, names2)) {
            res = 1;
            goto finish;
        }
    }
    if (type1->fields == NULL || type2->fields == NULL) {
        res = 0;
        goto finish;
    }

    CAPI_WARN("PyArray_Descr.fields is PyObject");
    val = PyObject_RichCompareBool(type1->fields, type2->fields, Py_EQ);
    if (val != 1 || HPyErr_Occurred(ctx)) {
        HPyErr_Clear(ctx);
        res = 0;
        goto finish;
    }

    if (HPy_IsNull(names1)) {
        names1 = HPyField_Load(ctx, h_type1, type1->names);
    }
    if (HPy_IsNull(names2)) {
        names2 = HPyField_Load(ctx, h_type2, type2->names);
    }
    val = HPy_RichCompareBool(ctx, names1, names2, HPy_EQ);
    if (val != 1 || HPyErr_Occurred(ctx)) {
        HPyErr_Clear(ctx);
        res = 0;
        goto finish;
    }

    res = 1;
    // fall through

finish:
    HPy_Close(ctx, names1);
    HPy_Close(ctx, names2);
    return res;
}

/*
 * Compare the subarray data for two types.
 * Return 1 if they are the same, 0 if not.
 */
static int
_equivalent_subarrays(PyArray_ArrayDescr *sub1, PyArray_ArrayDescr *sub2)
{
    int val;

    if (sub1 == sub2) {
        return 1;

    }
    if (sub1 == NULL || sub2 == NULL) {
        return 0;
    }

    val = PyObject_RichCompareBool(sub1->shape, sub2->shape, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    return PyArray_EquivTypes(sub1->base, sub2->base);
}


static unsigned char
PyArray_LegacyEquivTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    int type_num1, type_num2, size1, size2;

    if (type1 == type2) {
        return NPY_TRUE;
    }

    type_num1 = type1->type_num;
    type_num2 = type2->type_num;
    size1 = type1->elsize;
    size2 = type2->elsize;

    if (size1 != size2) {
        return NPY_FALSE;
    }
    if (PyArray_ISNBO(type1->byteorder) != PyArray_ISNBO(type2->byteorder)) {
        return NPY_FALSE;
    }
    if (type1->subarray || type2->subarray) {
        return ((type_num1 == type_num2)
                && _equivalent_subarrays(type1->subarray, type2->subarray));
    }
    if (type_num1 == NPY_VOID || type_num2 == NPY_VOID) {
        return ((type_num1 == type_num2) && _equivalent_fields(type1, type2));
    }
    if (type_num1 == NPY_DATETIME
        || type_num1 == NPY_TIMEDELTA
        || type_num2 == NPY_DATETIME
        || type_num2 == NPY_TIMEDELTA) {
        return ((type_num1 == type_num2)
                && has_equivalent_datetime_metadata(type1, type2));
    }
    return type1->kind == type2->kind;
}

static unsigned char
HPyArray_LegacyEquivTypes(HPyContext *ctx, 
                            HPy /* PyArray_Descr * */ h_type1,
                            PyArray_Descr *type1,
                            HPy /* PyArray_Descr * */ h_type2,
                            PyArray_Descr *type2)
{
    int type_num1, type_num2, size1, size2;

    if (HPy_Is(ctx, h_type1, h_type2)) {
        return NPY_TRUE;
    }

    type_num1 = type1->type_num;
    type_num2 = type2->type_num;
    size1 = type1->elsize;
    size2 = type2->elsize;

    if (size1 != size2) {
        return NPY_FALSE;
    }
    if (PyArray_ISNBO(type1->byteorder) != PyArray_ISNBO(type2->byteorder)) {
        return NPY_FALSE;
    }
    if (type1->subarray || type2->subarray) {
        CAPI_WARN("PyArray_Descr.subarray is _arr_descr to HPyField");
        return ((type_num1 == type_num2)
                && _equivalent_subarrays(type1->subarray, type2->subarray));
    }
    if (type_num1 == NPY_VOID || type_num2 == NPY_VOID) {
        return ((type_num1 == type_num2) && _hpy_equivalent_fields(ctx, h_type1, type1, h_type2, type2));
    }
    if (type_num1 == NPY_DATETIME
        || type_num1 == NPY_TIMEDELTA
        || type_num2 == NPY_DATETIME
        || type_num2 == NPY_TIMEDELTA) {
        return ((type_num1 == type_num2)
                && hpy_has_equivalent_datetime_metadata(ctx, h_type1, type1, h_type2, type2));
    }
    return type1->kind == type2->kind;
}


static unsigned char
PyArray_LegacyEquivTypenums(int typenum1, int typenum2)
{
    PyArray_Descr *d1, *d2;
    npy_bool ret;

    if (typenum1 == typenum2) {
        return NPY_SUCCEED;
    }

    d1 = PyArray_DescrFromType(typenum1);
    d2 = PyArray_DescrFromType(typenum2);
    ret = PyArray_LegacyEquivTypes(d1, d2);
    Py_DECREF(d1);
    Py_DECREF(d2);
    return ret;
}


static int
PyArray_LegacyCanCastSafely(int fromtype, int totype)
{
    PyArray_Descr *from;

    /* Fast table lookup for small type numbers */
    if ((unsigned int)fromtype < NPY_NTYPES &&
        (unsigned int)totype < NPY_NTYPES) {
        return _npy_can_cast_safely_table[fromtype][totype];
    }

    /* Identity */
    if (fromtype == totype) {
        return 1;
    }

    from = PyArray_DescrFromType(fromtype);
    /*
     * cancastto is a NPY_NOTYPE terminated C-int-array of types that
     * the data-type can be cast to safely.
     */
    if (from->f->cancastto) {
        int *curtype = from->f->cancastto;

        while (*curtype != NPY_NOTYPE) {
            if (*curtype++ == totype) {
                Py_DECREF(from);
                return 1;
            }
        }
    }
    Py_DECREF(from);
    return 0;
}

static int
HPyArray_LegacyCanCastSafely(HPyContext *ctx, int fromtype, int totype)
{
    HPy from; // PyArray_Descr *

    /* Fast table lookup for small type numbers */
    if ((unsigned int)fromtype < NPY_NTYPES &&
        (unsigned int)totype < NPY_NTYPES) {
        return _npy_can_cast_safely_table[fromtype][totype];
    }

    /* Identity */
    if (fromtype == totype) {
        return 1;
    }

    from = HPyArray_DescrFromType(ctx, fromtype);
    /*
     * cancastto is a NPY_NOTYPE terminated C-int-array of types that
     * the data-type can be cast to safely.
     */
    PyArray_Descr *from_data = PyArray_Descr_AsStruct(ctx, from);
    if (from_data->f->cancastto) {
        int *curtype = from_data->f->cancastto;

        while (*curtype != NPY_NOTYPE) {
            if (*curtype++ == totype) {
                HPy_Close(ctx, from);
                return 1;
            }
        }
    }
    HPy_Close(ctx, from);
    return 0;
}


static npy_bool
PyArray_LegacyCanCastTo(PyArray_Descr *from, PyArray_Descr *to)
{
    int from_type_num = from->type_num;
    int to_type_num = to->type_num;
    npy_bool ret;

    ret = (npy_bool) PyArray_LegacyCanCastSafely(from_type_num, to_type_num);
    if (ret) {
        /* Check String and Unicode more closely */
        if (from_type_num == NPY_STRING) {
            if (to_type_num == NPY_STRING) {
                ret = (from->elsize <= to->elsize);
            }
            else if (to_type_num == NPY_UNICODE) {
                ret = (from->elsize << 2 <= to->elsize);
            }
        }
        else if (from_type_num == NPY_UNICODE) {
            if (to_type_num == NPY_UNICODE) {
                ret = (from->elsize <= to->elsize);
            }
        }
            /*
             * For datetime/timedelta, only treat casts moving towards
             * more precision as safe.
             */
        else if (from_type_num == NPY_DATETIME && to_type_num == NPY_DATETIME) {
            PyArray_DatetimeMetaData *meta1, *meta2;
            meta1 = get_datetime_metadata_from_dtype(from);
            if (meta1 == NULL) {
                PyErr_Clear();
                return 0;
            }
            meta2 = get_datetime_metadata_from_dtype(to);
            if (meta2 == NULL) {
                PyErr_Clear();
                return 0;
            }

            return can_cast_datetime64_metadata(meta1, meta2,
                    NPY_SAFE_CASTING);
        }
        else if (from_type_num == NPY_TIMEDELTA &&
                 to_type_num == NPY_TIMEDELTA) {
            PyArray_DatetimeMetaData *meta1, *meta2;
            meta1 = get_datetime_metadata_from_dtype(from);
            if (meta1 == NULL) {
                PyErr_Clear();
                return 0;
            }
            meta2 = get_datetime_metadata_from_dtype(to);
            if (meta2 == NULL) {
                PyErr_Clear();
                return 0;
            }

            return can_cast_timedelta64_metadata(meta1, meta2,
                    NPY_SAFE_CASTING);
        }
            /*
             * If to_type_num is STRING or unicode
             * see if the length is long enough to hold the
             * stringified value of the object.
             */
        else if (to_type_num == NPY_STRING || to_type_num == NPY_UNICODE) {
            /*
             * Boolean value cast to string type is 5 characters max
             * for string 'False'.
             */
            int char_size = 1;
            if (to_type_num == NPY_UNICODE) {
                char_size = 4;
            }

            ret = 0;
            if (PyDataType_ISUNSIZED(to)) {
                ret = 1;
            }
                /*
                 * Need at least 5 characters to convert from boolean
                 * to 'True' or 'False'.
                 */
            else if (from->kind == 'b' && to->elsize >= 5 * char_size) {
                ret = 1;
            }
            else if (from->kind == 'u') {
                /* Guard against unexpected integer size */
                if (from->elsize > 8 || from->elsize < 0) {
                    ret = 0;
                }
                else if (to->elsize >=
                         REQUIRED_STR_LEN[from->elsize] * char_size) {
                    ret = 1;
                }
            }
            else if (from->kind == 'i') {
                /* Guard against unexpected integer size */
                if (from->elsize > 8 || from->elsize < 0) {
                    ret = 0;
                }
                    /* Extra character needed for sign */
                else if (to->elsize >=
                         (REQUIRED_STR_LEN[from->elsize] + 1) * char_size) {
                    ret = 1;
                }
            }
        }
    }
    return ret;
}

static npy_bool
HPyArray_LegacyCanCastTo(HPyContext *ctx, 
                            HPy /* PyArray_Descr * */ h_from,
                            PyArray_Descr *from,
                            HPy /* PyArray_Descr * */ h_to,
                            PyArray_Descr *to)
{
    int from_type_num = from->type_num;
    int to_type_num = to->type_num;
    npy_bool ret;

    ret = (npy_bool) HPyArray_LegacyCanCastSafely(ctx, from_type_num, to_type_num);
    if (ret) {
        /* Check String and Unicode more closely */
        if (from_type_num == NPY_STRING) {
            if (to_type_num == NPY_STRING) {
                ret = (from->elsize <= to->elsize);
            }
            else if (to_type_num == NPY_UNICODE) {
                ret = (from->elsize << 2 <= to->elsize);
            }
        }
        else if (from_type_num == NPY_UNICODE) {
            if (to_type_num == NPY_UNICODE) {
                ret = (from->elsize <= to->elsize);
            }
        }
            /*
             * For datetime/timedelta, only treat casts moving towards
             * more precision as safe.
             */
        else if (from_type_num == NPY_DATETIME && to_type_num == NPY_DATETIME) {
            PyArray_DatetimeMetaData *meta1, *meta2;
            meta1 = h_get_datetime_metadata_from_dtype(ctx, from);
            if (meta1 == NULL) {
                PyErr_Clear();
                return 0;
            }
            meta2 = h_get_datetime_metadata_from_dtype(ctx, to);
            if (meta2 == NULL) {
                PyErr_Clear();
                return 0;
            }

            return can_cast_datetime64_metadata(meta1, meta2,
                    NPY_SAFE_CASTING);
        }
        else if (from_type_num == NPY_TIMEDELTA &&
                 to_type_num == NPY_TIMEDELTA) {
            PyArray_DatetimeMetaData *meta1, *meta2;
            meta1 = h_get_datetime_metadata_from_dtype(ctx, from);
            if (meta1 == NULL) {
                PyErr_Clear();
                return 0;
            }
            meta2 = h_get_datetime_metadata_from_dtype(ctx, to);
            if (meta2 == NULL) {
                PyErr_Clear();
                return 0;
            }

            return can_cast_timedelta64_metadata(meta1, meta2,
                    NPY_SAFE_CASTING);
        }
            /*
             * If to_type_num is STRING or unicode
             * see if the length is long enough to hold the
             * stringified value of the object.
             */
        else if (to_type_num == NPY_STRING || to_type_num == NPY_UNICODE) {
            /*
             * Boolean value cast to string type is 5 characters max
             * for string 'False'.
             */
            int char_size = 1;
            if (to_type_num == NPY_UNICODE) {
                char_size = 4;
            }

            ret = 0;
            if (PyDataType_ISUNSIZED(to)) {
                ret = 1;
            }
                /*
                 * Need at least 5 characters to convert from boolean
                 * to 'True' or 'False'.
                 */
            else if (from->kind == 'b' && to->elsize >= 5 * char_size) {
                ret = 1;
            }
            else if (from->kind == 'u') {
                /* Guard against unexpected integer size */
                if (from->elsize > 8 || from->elsize < 0) {
                    ret = 0;
                }
                else if (to->elsize >=
                         REQUIRED_STR_LEN[from->elsize] * char_size) {
                    ret = 1;
                }
            }
            else if (from->kind == 'i') {
                /* Guard against unexpected integer size */
                if (from->elsize > 8 || from->elsize < 0) {
                    ret = 0;
                }
                    /* Extra character needed for sign */
                else if (to->elsize >=
                         (REQUIRED_STR_LEN[from->elsize] + 1) * char_size) {
                    ret = 1;
                }
            }
        }
    }
    return ret;
}


/*
 * Compare two field dictionaries for castability.
 *
 * Return 1 if 'field1' can be cast to 'field2' according to the rule
 * 'casting', 0 if not.
 *
 * Castabiliy of field dictionaries is defined recursively: 'field1' and
 * 'field2' must have the same field names (possibly in different
 * orders), and the corresponding field types must be castable according
 * to the given casting rule.
 */
static int
can_cast_fields(PyObject *field1, PyObject *field2, NPY_CASTING casting)
{
    Py_ssize_t ppos;
    PyObject *key;
    PyObject *tuple1, *tuple2;

    if (field1 == field2) {
        return 1;
    }
    if (field1 == NULL || field2 == NULL) {
        return 0;
    }
    if (PyDict_Size(field1) != PyDict_Size(field2)) {
        return 0;
    }

    /* Iterate over all the fields and compare for castability */
    ppos = 0;
    while (PyDict_Next(field1, &ppos, &key, &tuple1)) {
        if ((tuple2 = PyDict_GetItem(field2, key)) == NULL) {
            return 0;
        }
        /* Compare the dtype of the field for castability */
        if (!PyArray_CanCastTypeTo(
                        (PyArray_Descr *)PyTuple_GET_ITEM(tuple1, 0),
                        (PyArray_Descr *)PyTuple_GET_ITEM(tuple2, 0),
                        casting)) {
            return 0;
        }
    }

    return 1;
}


NPY_NO_EXPORT npy_bool
PyArray_LegacyCanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting)
{
    /*
     * Fast paths for equality and for basic types.
     */
    if (from == to ||
        ((NPY_LIKELY(PyDataType_ISNUMBER(from)) ||
          PyDataType_ISOBJECT(from)) &&
         NPY_LIKELY(from->type_num == to->type_num) &&
         NPY_LIKELY(from->byteorder == to->byteorder))) {
        return 1;
    }
    /*
     * Cases with subarrays and fields need special treatment.
     */
    if (PyDataType_HASFIELDS(from)) {
        /*
         * If from is a structured data type, then it can be cast to a simple
         * non-object one only for unsafe casting *and* if it has a single
         * field; recurse just in case the single field is itself structured.
         */
        if (!PyDataType_HASFIELDS(to) && !PyDataType_ISOBJECT(to)) {
            if (casting == NPY_UNSAFE_CASTING &&
                    PyDict_Size(from->fields) == 1) {
                Py_ssize_t ppos = 0;
                PyObject *tuple;
                PyArray_Descr *field;
                PyDict_Next(from->fields, &ppos, NULL, &tuple);
                field = (PyArray_Descr *)PyTuple_GET_ITEM(tuple, 0);
                /*
                 * For a subarray, we need to get the underlying type;
                 * since we already are casting unsafely, we can ignore
                 * the shape.
                 */
                if (PyDataType_HASSUBARRAY(field)) {
                    field = field->subarray->base;
                }
                return PyArray_LegacyCanCastTypeTo(field, to, casting);
            }
            else {
                return 0;
            }
        }
        /*
         * Casting from one structured data type to another depends on the fields;
         * we pass that case on to the EquivTypenums case below.
         *
         * TODO: move that part up here? Need to check whether equivalent type
         * numbers is an addition constraint that is needed.
         *
         * TODO/FIXME: For now, always allow structured to structured for unsafe
         * casting; this is not correct, but needed since the treatment in can_cast
         * below got out of sync with astype; see gh-13667.
         */
        if (casting == NPY_UNSAFE_CASTING) {
            return 1;
        }
    }
    else if (PyDataType_HASFIELDS(to)) {
        /*
         * If "from" is a simple data type and "to" has fields, then only
         * unsafe casting works (and that works always, even to multiple fields).
         */
        return casting == NPY_UNSAFE_CASTING;
    }
    /*
     * Everything else we consider castable for unsafe for now.
     * FIXME: ensure what we do here is consistent with "astype",
     * i.e., deal more correctly with subarrays and user-defined dtype.
     */
    else if (casting == NPY_UNSAFE_CASTING) {
        return 1;
    }
    /*
     * Equivalent simple types can be cast with any value of 'casting', but
     * we need to be careful about structured to structured.
     */
    if (PyArray_LegacyEquivTypenums(from->type_num, to->type_num)) {
        /* For complicated case, use EquivTypes (for now) */
        if (PyTypeNum_ISUSERDEF(from->type_num) ||
                        from->subarray != NULL) {
            int ret;

            /* Only NPY_NO_CASTING prevents byte order conversion */
            if ((casting != NPY_NO_CASTING) &&
                                (!PyArray_ISNBO(from->byteorder) ||
                                 !PyArray_ISNBO(to->byteorder))) {
                PyArray_Descr *nbo_from, *nbo_to;

                nbo_from = PyArray_DescrNewByteorder(from, NPY_NATIVE);
                nbo_to = PyArray_DescrNewByteorder(to, NPY_NATIVE);
                if (nbo_from == NULL || nbo_to == NULL) {
                    Py_XDECREF(nbo_from);
                    Py_XDECREF(nbo_to);
                    PyErr_Clear();
                    return 0;
                }
                ret = PyArray_LegacyEquivTypes(nbo_from, nbo_to);
                Py_DECREF(nbo_from);
                Py_DECREF(nbo_to);
            }
            else {
                ret = PyArray_LegacyEquivTypes(from, to);
            }
            return ret;
        }

        if (PyDataType_HASFIELDS(from)) {
            switch (casting) {
                case NPY_EQUIV_CASTING:
                case NPY_SAFE_CASTING:
                case NPY_SAME_KIND_CASTING:
                    /*
                     * `from' and `to' must have the same fields, and
                     * corresponding fields must be (recursively) castable.
                     */
                    return can_cast_fields(from->fields, to->fields, casting);

                case NPY_NO_CASTING:
                default:
                    return PyArray_LegacyEquivTypes(from, to);
            }
        }

        switch (from->type_num) {
            case NPY_DATETIME: {
                PyArray_DatetimeMetaData *meta1, *meta2;
                meta1 = get_datetime_metadata_from_dtype(from);
                if (meta1 == NULL) {
                    PyErr_Clear();
                    return 0;
                }
                meta2 = get_datetime_metadata_from_dtype(to);
                if (meta2 == NULL) {
                    PyErr_Clear();
                    return 0;
                }

                if (casting == NPY_NO_CASTING) {
                    return PyArray_ISNBO(from->byteorder) ==
                                        PyArray_ISNBO(to->byteorder) &&
                            can_cast_datetime64_metadata(meta1, meta2, casting);
                }
                else {
                    return can_cast_datetime64_metadata(meta1, meta2, casting);
                }
            }
            case NPY_TIMEDELTA: {
                PyArray_DatetimeMetaData *meta1, *meta2;
                meta1 = get_datetime_metadata_from_dtype(from);
                if (meta1 == NULL) {
                    PyErr_Clear();
                    return 0;
                }
                meta2 = get_datetime_metadata_from_dtype(to);
                if (meta2 == NULL) {
                    PyErr_Clear();
                    return 0;
                }

                if (casting == NPY_NO_CASTING) {
                    return PyArray_ISNBO(from->byteorder) ==
                                        PyArray_ISNBO(to->byteorder) &&
                        can_cast_timedelta64_metadata(meta1, meta2, casting);
                }
                else {
                    return can_cast_timedelta64_metadata(meta1, meta2, casting);
                }
            }
            default:
                switch (casting) {
                    case NPY_NO_CASTING:
                        return PyArray_LegacyEquivTypes(from, to);
                    case NPY_EQUIV_CASTING:
                        return (from->elsize == to->elsize);
                    case NPY_SAFE_CASTING:
                        return (from->elsize <= to->elsize);
                    default:
                        return 1;
                }
                break;
        }
    }
    /* If safe or same-kind casts are allowed */
    else if (casting == NPY_SAFE_CASTING || casting == NPY_SAME_KIND_CASTING) {
        if (PyArray_LegacyCanCastTo(from, to)) {
            return 1;
        }
        else if(casting == NPY_SAME_KIND_CASTING) {
            /*
             * Also allow casting from lower to higher kinds, according
             * to the ordering provided by dtype_kind_to_ordering.
             * Some kinds, like datetime, don't fit in the hierarchy,
             * and are special cased as -1.
             */
            int from_order, to_order;

            from_order = dtype_kind_to_ordering(from->kind);
            to_order = dtype_kind_to_ordering(to->kind);

            if (to->kind == 'm') {
                /* both types being timedelta is already handled before. */
                int integer_order = dtype_kind_to_ordering('i');
                return (from_order != -1) && (from_order <= integer_order);
            }

            return (from_order != -1) && (from_order <= to_order);
        }
        else {
            return 0;
        }
    }
    /* NPY_NO_CASTING or NPY_EQUIV_CASTING was specified */
    else {
        return 0;
    }
}


NPY_NO_EXPORT npy_bool
HPyArray_LegacyCanCastTypeTo(HPyContext *ctx, 
        HPy /* PyArray_Descr * */ from, 
        HPy /* PyArray_Descr * */ to,
        NPY_CASTING casting)
{
    /*
     * Fast paths for equality and for basic types.
     */
    PyArray_Descr *from_data = PyArray_Descr_AsStruct(ctx, from);
    PyArray_Descr *to_data = PyArray_Descr_AsStruct(ctx, to);
    if (HPy_Is(ctx, from, to) ||
        ((NPY_LIKELY(PyDataType_ISNUMBER(from_data)) ||
          PyDataType_ISOBJECT(from_data)) &&
         NPY_LIKELY(from_data->type_num == to_data->type_num) &&
         NPY_LIKELY(from_data->byteorder == to_data->byteorder))) {
        return 1;
    }
    /*
     * Cases with subarrays and fields need special treatment.
     */
    if (PyDataType_HASFIELDS(from_data)) {
        /*
         * If from is a structured data type, then it can be cast to a simple
         * non-object one only for unsafe casting *and* if it has a single
         * field; recurse just in case the single field is itself structured.
         */
        if (!PyDataType_HASFIELDS(to_data) && !PyDataType_ISOBJECT(to_data)) {
            if (casting == NPY_UNSAFE_CASTING &&
                    PyDict_Size(from_data->fields) == 1) {
                Py_ssize_t ppos = 0;
                PyObject *tuple;
                HPy field;
                PyDict_Next(from_data->fields, &ppos, NULL, &tuple);
                HPy h_tuple = HPy_FromPyObject(ctx, tuple);
                field = HPy_GetItem_i(ctx, h_tuple, 0);
                HPy_Close(ctx, h_tuple);
                /*
                 * For a subarray, we need to get the underlying type;
                 * since we already are casting unsafely, we can ignore
                 * the shape.
                 */
                PyArray_Descr *field_data = PyArray_Descr_AsStruct(ctx, field);
                if (PyDataType_HASSUBARRAY(field_data)) {
                    PyArray_Descr *py_field = field_data->subarray->base;
                    HPy_Close(ctx, field);
                    field = HPy_FromPyObject(ctx, py_field);
                    npy_bool ret = HPyArray_LegacyCanCastTypeTo(ctx, field, to, casting);
                    HPy_Close(ctx, field);
                    return ret;
                }
                npy_bool ret = HPyArray_LegacyCanCastTypeTo(ctx, field, to, casting);
                HPy_Close(ctx, field);
                return ret;
            }
            else {
                return 0;
            }
        }
        /*
         * Casting from one structured data type to another depends on the fields;
         * we pass that case on to the EquivTypenums case below.
         *
         * TODO: move that part up here? Need to check whether equivalent type
         * numbers is an addition constraint that is needed.
         *
         * TODO/FIXME: For now, always allow structured to structured for unsafe
         * casting; this is not correct, but needed since the treatment in can_cast
         * below got out of sync with astype; see gh-13667.
         */
        if (casting == NPY_UNSAFE_CASTING) {
            return 1;
        }
    }
    else if (PyDataType_HASFIELDS(to_data)) {
        /*
         * If "from" is a simple data type and "to" has fields, then only
         * unsafe casting works (and that works always, even to multiple fields).
         */
        return casting == NPY_UNSAFE_CASTING;
    }
    /*
     * Everything else we consider castable for unsafe for now.
     * FIXME: ensure what we do here is consistent with "astype",
     * i.e., deal more correctly with subarrays and user-defined dtype.
     */
    else if (casting == NPY_UNSAFE_CASTING) {
        return 1;
    }
    /*
     * Equivalent simple types can be cast with any value of 'casting', but
     * we need to be careful about structured to structured.
     */
    if (PyArray_LegacyEquivTypenums(from_data->type_num, to_data->type_num)) {
        /* For complicated case, use EquivTypes (for now) */
        if (PyTypeNum_ISUSERDEF(from_data->type_num) ||
                        from_data->subarray != NULL) {
            int ret;

            /* Only NPY_NO_CASTING prevents byte order conversion */
            if ((casting != NPY_NO_CASTING) &&
                                (!PyArray_ISNBO(from_data->byteorder) ||
                                 !PyArray_ISNBO(to_data->byteorder))) {
                HPy nbo_from, nbo_to; // PyArray_Descr *

                nbo_from = HPyArray_DescrNewByteorder(ctx, from, NPY_NATIVE);
                nbo_to = HPyArray_DescrNewByteorder(ctx, to, NPY_NATIVE);
                if (HPy_IsNull(nbo_from) || HPy_IsNull(nbo_to)) {
                    HPy_Close(ctx, nbo_from);
                    HPy_Close(ctx, nbo_to);
                    HPyErr_Clear(ctx);
                    return 0;
                }
                PyArray_Descr *nbo_from_data = PyArray_Descr_AsStruct(ctx, nbo_from);
                PyArray_Descr *nbo_to_data = PyArray_Descr_AsStruct(ctx, nbo_to);
                ret = HPyArray_LegacyEquivTypes(ctx, nbo_from, nbo_from_data, nbo_to, nbo_to_data);
                HPy_Close(ctx, nbo_from);
                HPy_Close(ctx, nbo_to);
            }
            else {
                ret = HPyArray_LegacyEquivTypes(ctx, from, from_data, to, to_data);
            }
            return ret;
        }

        if (PyDataType_HASFIELDS(from_data)) {
            switch (casting) {
                case NPY_EQUIV_CASTING:
                case NPY_SAFE_CASTING:
                case NPY_SAME_KIND_CASTING:
                    /*
                     * `from' and `to' must have the same fields, and
                     * corresponding fields must be (recursively) castable.
                     */
                    return can_cast_fields(from_data->fields, to_data->fields, casting);

                case NPY_NO_CASTING:
                default:
                    return HPyArray_LegacyEquivTypes(ctx, from, from_data, to, to_data);
            }
        }

        switch (from_data->type_num) {
            case NPY_DATETIME: {
                PyArray_DatetimeMetaData *meta1, *meta2;
                meta1 = h_get_datetime_metadata_from_dtype(ctx, from_data);
                if (meta1 == NULL) {
                    PyErr_Clear();
                    return 0;
                }
                meta2 = h_get_datetime_metadata_from_dtype(ctx, to_data);
                if (meta2 == NULL) {
                    HPyErr_Clear(ctx);
                    return 0;
                }

                if (casting == NPY_NO_CASTING) {
                    return PyArray_ISNBO(from_data->byteorder) ==
                                        PyArray_ISNBO(to_data->byteorder) &&
                            can_cast_datetime64_metadata(meta1, meta2, casting);
                }
                else {
                    return can_cast_datetime64_metadata(meta1, meta2, casting);
                }
            }
            case NPY_TIMEDELTA: {
                PyArray_DatetimeMetaData *meta1, *meta2;
                meta1 = h_get_datetime_metadata_from_dtype(ctx, from_data);
                if (meta1 == NULL) {
                    PyErr_Clear();
                    return 0;
                }
                meta2 = h_get_datetime_metadata_from_dtype(ctx, to_data);
                if (meta2 == NULL) {
                    PyErr_Clear();
                    return 0;
                }

                if (casting == NPY_NO_CASTING) {
                    return PyArray_ISNBO(from_data->byteorder) ==
                                        PyArray_ISNBO(to_data->byteorder) &&
                        can_cast_timedelta64_metadata(meta1, meta2, casting);
                }
                else {
                    return can_cast_timedelta64_metadata(meta1, meta2, casting);
                }
            }
            default:
                switch (casting) {
                    case NPY_NO_CASTING:
                        return HPyArray_LegacyEquivTypes(ctx, from, from_data, to, to_data);
                    case NPY_EQUIV_CASTING:
                        return (from_data->elsize == to_data->elsize);
                    case NPY_SAFE_CASTING:
                        return (from_data->elsize <= to_data->elsize);
                    default:
                        return 1;
                }
                break;
        }
    }
    /* If safe or same-kind casts are allowed */
    else if (casting == NPY_SAFE_CASTING || casting == NPY_SAME_KIND_CASTING) {
        if (HPyArray_LegacyCanCastTo(ctx, from, from_data, to, to_data)) {
            return 1;
        }
        else if(casting == NPY_SAME_KIND_CASTING) {
            /*
             * Also allow casting from lower to higher kinds, according
             * to the ordering provided by dtype_kind_to_ordering.
             * Some kinds, like datetime, don't fit in the hierarchy,
             * and are special cased as -1.
             */
            int from_order, to_order;

            from_order = dtype_kind_to_ordering(from_data->kind);
            to_order = dtype_kind_to_ordering(to_data->kind);

            if (to_data->kind == 'm') {
                /* both types being timedelta is already handled before. */
                int integer_order = dtype_kind_to_ordering('i');
                return (from_order != -1) && (from_order <= integer_order);
            }

            return (from_order != -1) && (from_order <= to_order);
        }
        else {
            return 0;
        }
    }
    /* NPY_NO_CASTING or NPY_EQUIV_CASTING was specified */
    else {
        return 0;
    }
}


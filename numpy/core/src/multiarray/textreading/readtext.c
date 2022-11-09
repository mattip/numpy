#include <stdio.h>
#include <stdbool.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "npy_argparse.h"
#include "common.h"
#include "conversion_utils.h"

#include "textreading/parser_config.h"
#include "textreading/stream_pyobject.h"
#include "textreading/field_types.h"
#include "textreading/rows.h"
#include "textreading/str_to_int.h"


//
// `usecols` must point to a Python object that is Py_None or a 1-d contiguous
// numpy array with data type int32.
//
// `dtype` must point to a Python object that is Py_None or a numpy dtype
// instance.  If the latter, code and sizes must be arrays of length
// num_dtype_fields, holding the flattened data field type codes and byte
// sizes. (num_dtype_fields, codes, and sizes can be inferred from dtype,
// but we do that in Python code.)
//
// If both `usecols` and `dtype` are not None, and the data type is compound,
// then len(usecols) must equal num_dtype_fields.
//
// If `dtype` is given and it is compound, and `usecols` is None, then the
// number of columns in the file must match the number of fields in `dtype`.
//
static PyObject *
_readtext_from_stream(stream *s,
        parser_config *pc, Py_ssize_t num_usecols, Py_ssize_t usecols[],
        Py_ssize_t skiplines, Py_ssize_t max_rows,
        PyObject *converters, PyObject *dtype)
{
    PyArrayObject *arr = NULL;
    PyArray_Descr *out_dtype = NULL;
    field_type *ft = NULL;

    /*
     * If dtypes[0] is dtype the input was not structured and the result
     * is considered "homogeneous" and we have to discover the number of
     * columns/
     */
    out_dtype = (PyArray_Descr *)dtype;
    Py_INCREF(out_dtype);

    Py_ssize_t num_fields = field_types_create(out_dtype, &ft);
    if (num_fields < 0) {
        goto finish;
    }
    bool homogeneous = num_fields == 1 && ft[0].descr == out_dtype;

    if (!homogeneous && usecols != NULL && num_usecols != num_fields) {
        PyErr_Format(PyExc_TypeError,
                "If a structured dtype is used, the number of columns in "
                "`usecols` must match the effective number of fields. "
                "But %zd usecols were given and the number of fields is %zd.",
                num_usecols, num_fields);
        goto finish;
    }

    arr = read_rows(
            s, max_rows, num_fields, ft, pc,
            num_usecols, usecols, skiplines, converters,
            NULL, out_dtype, homogeneous);
    if (arr == NULL) {
        goto finish;
    }

  finish:
    Py_XDECREF(out_dtype);
    field_types_xclear(num_fields, ft);
    return (PyObject *)arr;
}


static int
parse_control_character(PyObject *obj, Py_UCS4 *character)
{
    if (obj == Py_None) {
        *character = (Py_UCS4)-1;  /* character beyond unicode range */
        return 1;
    }
    if (!PyUnicode_Check(obj) || PyUnicode_GetLength(obj) != 1) {
        PyErr_Format(PyExc_TypeError,
                "Text reading control character must be a single unicode "
                "character or None; but got: %.100R", obj);
        return 0;
    }
    *character = PyUnicode_READ_CHAR(obj, 0);
    return 1;
}

static int
hpy_parse_control_character(HPyContext *ctx, HPy obj, HPy_UCS4 *character)
{
    if (HPy_IsNull(obj) || HPy_Is(ctx, obj, ctx->h_None)) {
        *character = (Py_UCS4)-1;  /* character beyond unicode range */
        return 1;
    }
    if (!HPyUnicode_Check(ctx, obj) || HPy_Length(ctx, obj) != 1) {
        // PyErr_Format(PyExc_TypeError,
        //         "Text reading control character must be a single unicode "
        //         "character or None; but got: %.100R", obj);
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Text reading control character must be a single unicode "
                "character or None; but got: %.100R");
        return 0;
    }
    
    *character = HPyUnicode_ReadChar(ctx, obj, 0);
    return 1;
}


/*
 * A (somewhat verbose) check that none of the control characters match or are
 * newline.  Most of these combinations are completely fine, just weird or
 * surprising.
 * (I.e. there is an implicit priority for control characters, so if a comment
 * matches a delimiter, it would just be a comment.)
 * In theory some `delimiter=None` paths could have a "meaning", but let us
 * assume that users are better of setting one of the control chars to `None`
 * for clarity.
 *
 * This also checks that the control characters cannot be newlines.
 */
static int
hpy_error_if_matching_control_characters(HPyContext *ctx,
        Py_UCS4 delimiter, Py_UCS4 quote, Py_UCS4 comment)
{
    char *control_char1;
    char *control_char2 = NULL;
    if (comment != (Py_UCS4)-1) {
        control_char1 = "comment";
        if (comment == '\r' || comment == '\n') {
            goto error;
        }
        else if (comment == quote) {
            control_char2 = "quotechar";
            goto error;
        }
        else if (comment == delimiter) {
            control_char2 = "delimiter";
            goto error;
        }
    }
    if (quote != (Py_UCS4)-1) {
        control_char1 = "quotechar";
        if (quote == '\r' || quote == '\n') {
            goto error;
        }
        else if (quote == delimiter) {
            control_char2 = "delimiter";
            goto error;
        }
    }
    if (delimiter != (Py_UCS4)-1) {
        control_char1 = "delimiter";
        if (delimiter == '\r' || delimiter == '\n') {
            goto error;
        }
    }
    CAPI_WARN("calling Py_UNICODE_ISSPACE");
    /* The above doesn't work with delimiter=None, which means "whitespace" */
    if (delimiter == (Py_UCS4)-1) {
        control_char1 = "delimiter";
        if (Py_UNICODE_ISSPACE(comment)) {
            control_char2 = "comment";
            goto error;
        }
        else if (Py_UNICODE_ISSPACE(quote)) {
            control_char2 = "quotechar";
            goto error;
        }
    }
    return 0;

  error:
    if (control_char2 != NULL) {
        HPyErr_Format_p(ctx, ctx->h_TypeError,
                "The values for control characters '%s' and '%s' are "
                "incompatible",
                control_char1, control_char2);
    }
    else {
        HPyErr_Format_p(ctx, ctx->h_TypeError,
                "control character '%s' cannot be a newline (`\\r` or `\\n`).",
                control_char1);
    }
    return -1;
}


HPyDef_METH(_load_from_filelike, "_load_from_filelike", _load_from_filelike_impl, HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
_load_from_filelike_impl(HPyContext *ctx, HPy NPY_UNUSED(mod),
                            HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy file;
    HPy_ssize_t skiplines = 0;
    HPy_ssize_t max_rows = -1;
    HPy usecols_obj = ctx->h_None;
    HPy converters = ctx->h_None;

    HPy dtype = ctx->h_None;
    HPy encoding_obj = ctx->h_None;
    const char *encoding = NULL;

    parser_config pc = {
        .delimiter = ',',
        .comment = '#',
        .quote = '"',
        .imaginary_unit = 'j',
        .delimiter_is_whitespace = false,
        .ignore_leading_whitespace = false,
        .python_byte_converters = false,
        .c_byte_converters = false,
    };
    bool filelike = true;

    HPy arr = HPy_NULL;

    static char *kwlist[] = {"file", "delimiter", "comment", "quote", "imaginary_unit",
        "usecols", "skiplines", "max_rows", "converters", "dtype",
        "encoding", "filelike", "byte_converters", "c_byte_converters", NULL};
    HPy h_delimiter = HPy_NULL;
    HPy h_comment = HPy_NULL;
    HPy h_quote = HPy_NULL;
    HPy h_imaginary_unit = HPy_NULL;
    HPy h_skiplines = HPy_NULL;
    HPy h_max_rows = HPy_NULL;
    HPy h_filelike = HPy_NULL;
    HPy h_python_byte_converters = HPy_NULL;
    HPy h_c_byte_converters = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "O|OOOOOOOOOOOOO:_load_from_filelike",
                kwlist,
                &file,
                &h_delimiter,
                &h_comment,
                &h_quote,
                &h_imaginary_unit,
                &usecols_obj,
                &h_skiplines,
                &h_max_rows,
                &converters,
                &dtype,
                &encoding_obj,
                &h_filelike,
                &h_python_byte_converters,
                &h_c_byte_converters)) {
        return HPy_NULL;
    }

    if (hpy_parse_control_character(ctx, h_delimiter, &pc.delimiter) != NPY_SUCCEED ||
            hpy_parse_control_character(ctx, h_comment, &pc.comment) != NPY_SUCCEED ||
            hpy_parse_control_character(ctx, h_quote, &pc.quote) != NPY_SUCCEED ||
            hpy_parse_control_character(ctx, h_imaginary_unit, &pc.imaginary_unit) != NPY_SUCCEED ||
            HPyArray_IntpFromPyIntConverter(ctx, h_skiplines, &skiplines) != NPY_SUCCEED ||
            HPyArray_IntpFromPyIntConverter(ctx, h_max_rows, &max_rows) != NPY_SUCCEED ||
            HPyArray_BoolConverter(ctx, h_filelike, &filelike) != NPY_SUCCEED ||
            HPyArray_BoolConverter(ctx, h_python_byte_converters, &pc.python_byte_converters) != NPY_SUCCEED ||
            HPyArray_BoolConverter(ctx, h_c_byte_converters, &pc.c_byte_converters) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, ": TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    // NPY_PREPARE_ARGPARSER;
    // if (npy_parse_arguments("_load_from_filelike", args, len_args, kwnames,
    //         "file", NULL, &file,
    //         "|delimiter", &parse_control_character, &pc.delimiter,
    //         "|comment", &parse_control_character, &pc.comment,
    //         "|quote", &parse_control_character, &pc.quote,
    //         "|imaginary_unit", &parse_control_character, &pc.imaginary_unit,
    //         "|usecols", NULL, &usecols_obj,
    //         "|skiplines", &PyArray_IntpFromPyIntConverter, &skiplines,
    //         "|max_rows", &PyArray_IntpFromPyIntConverter, &max_rows,
    //         "|converters", NULL, &converters,
    //         "|dtype", NULL, &dtype,
    //         "|encoding", NULL, &encoding_obj,
    //         "|filelike", &PyArray_BoolConverter, &filelike,
    //         "|byte_converters", &PyArray_BoolConverter, &pc.python_byte_converters,
    //         "|c_byte_converters", PyArray_BoolConverter, &pc.c_byte_converters,
    //         NULL, NULL, NULL) < 0) {
    //     return HPy_NULL;
    // }

    /* Reject matching control characters, they just rarely make sense anyway */
    if (hpy_error_if_matching_control_characters(ctx,
            pc.delimiter, pc.quote, pc.comment) < 0) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (pc.delimiter == (Py_UCS4)-1) {
        pc.delimiter_is_whitespace = true;
        /* Ignore leading whitespace to match `string.split(None)` */
        pc.ignore_leading_whitespace = true;
    }

    if (!HPyArray_DescrCheck(ctx, dtype) ) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "internal error: dtype must be provided and be a NumPy dtype");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!HPy_Is(ctx, encoding_obj, ctx->h_None)) {
        if (!HPyUnicode_Check(ctx, encoding_obj)) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "encoding must be a unicode string.");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        HPy_ssize_t encoding_len;
        encoding = HPyUnicode_AsUTF8AndSize(ctx, encoding_obj, &encoding_len);
        if (encoding == NULL) {
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
    }

    /*
     * Parse usecols, the rest of NumPy has no clear helper for this, so do
     * it here manually.
     */
    HPy_ssize_t num_usecols = -1;
    HPy_ssize_t *usecols = NULL;
    if (!HPy_Is(ctx, usecols_obj, ctx->h_None)) {
        num_usecols = HPy_Length(ctx, usecols_obj);
        if (num_usecols < 0) {
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        /* Calloc just to not worry about overflow */
        usecols = calloc(num_usecols, sizeof(Py_ssize_t));
        for (HPy_ssize_t i = 0; i < num_usecols; i++) {
            HPy tmp = HPy_GetItem_i(ctx, usecols_obj, i);
            if (HPy_IsNull(tmp)) {
                HPyTracker_Close(ctx, ht);
                free(usecols);
                return HPy_NULL;
            }
            usecols[i] = HPyNumber_AsSsize_t(ctx, tmp, ctx->h_OverflowError);
            if (error_converting(usecols[i])) {
                if (HPyErr_ExceptionMatches(ctx, ctx->h_TypeError)) {
                    // PyErr_Format(PyExc_TypeError,
                    //         "usecols must be an int or a sequence of ints but "
                    //         "it contains at least one element of type '%s'",
                    //         Py_TYPE(tmp)->tp_name);
                    HPyErr_SetString(ctx, ctx->h_TypeError,
                            "usecols must be an int or a sequence of ints but "
                            "it contains at least one element of type '%s'");
                }
                HPy_Close(ctx, tmp);
                HPyTracker_Close(ctx, ht);
                PyMem_FREE(usecols);
                return HPy_NULL;
            }
            HPy_Close(ctx, tmp);
        }
    }

    stream *s;
    CAPI_WARN("calling stream_python_*");
    PyObject *py_file = HPy_AsPyObject(ctx, file);
    if (filelike) {
        s = stream_python_file(py_file, encoding);
    }
    else {
        s = stream_python_iterable(py_file, encoding);
    }
    if (s == NULL) {
        Py_DECREF(py_file);
        HPyTracker_Close(ctx, ht);
        free(usecols);
        return HPy_NULL;
    }
    
    CAPI_WARN("calling _readtext_from_stream");
    PyObject *py_converters = HPy_AsPyObject(ctx, converters);
    PyObject *py_dtype = HPy_AsPyObject(ctx, dtype);
    PyObject *py_arr = _readtext_from_stream(
            s, &pc, num_usecols, usecols, skiplines, max_rows, py_converters, py_dtype);
    arr = HPy_FromPyObject(ctx, py_arr);
    Py_DECREF(py_file);
    Py_DECREF(py_converters);
    Py_DECREF(py_dtype);
    Py_DECREF(py_arr);
    stream_close(s);
    free(usecols);
    HPyTracker_Close(ctx, ht);
    return arr;
}


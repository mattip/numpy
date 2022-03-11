/*
 * Provides namedtuples for numpy.core.multiarray.typeinfo
 * Unfortunately, we need two different types to cover the cases where min/max
 * do and do not appear in the tuple.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "npy_pycompat.h"
#include "typeinfo.h"

#if (defined(PYPY_VERSION_NUM) && (PYPY_VERSION_NUM <= 0x07030000))
/* PyPy issue 3160 */
#include <structseq.h>
#endif



static HPy PyArray_typeinfoType;
static HPy PyArray_typeinforangedType;

static HPyStructSequence_Field typeinfo_fields[] = {
    {"char",      "The character used to represent the type"},
    {"num",       "The numeric id assigned to the type"},
    {"bits",      "The number of bits in the type"},
    {"alignment", "The alignment of the type in bytes"},
    {"type",      "The python type object this info is about"},
    {NULL, NULL,}
};

static HPyStructSequence_Field typeinforanged_fields[] = {
    {"char",      "The character used to represent the type"},
    {"num",       "The numeric id assigned to the type"},
    {"bits",      "The number of bits in the type"},
    {"alignment", "The alignment of the type in bytes"},
    {"max",       "The maximum value of this type"},
    {"min",       "The minimum value of this type"},
    {"type",      "The python type object this info is about"},
    {NULL, NULL,}
};

static HPyStructSequence_Desc typeinfo_desc = {
    "numpy.core.multiarray.typeinfo",         /* name          */
    "Information about a scalar numpy type",  /* doc           */
    typeinfo_fields,                          /* fields        */
    5,                                        /* n_in_sequence */
};

static HPyStructSequence_Desc typeinforanged_desc = {
    "numpy.core.multiarray.typeinforanged",                /* name          */
    "Information about a scalar numpy type with a range",  /* doc           */
    typeinforanged_fields,                                 /* fields        */
    7,                                                     /* n_in_sequence */
};

NPY_NO_EXPORT PyObject *
PyArray_typeinfo(
    char typechar, int typenum, int nbits, int align,
    PyTypeObject *type_obj)
{
    PyObject *entry = PyStructSequence_New((PyTypeObject*)HPy_AsPyObject(NULL, PyArray_typeinfoType));
    if (entry == NULL)
        return NULL;
    PyStructSequence_SET_ITEM(entry, 0, Py_BuildValue("C", typechar));
    PyStructSequence_SET_ITEM(entry, 1, Py_BuildValue("i", typenum));
    PyStructSequence_SET_ITEM(entry, 2, Py_BuildValue("i", nbits));
    PyStructSequence_SET_ITEM(entry, 3, Py_BuildValue("i", align));
    PyStructSequence_SET_ITEM(entry, 4, Py_BuildValue("O", (PyObject *) type_obj));

    if (PyErr_Occurred()) {
        Py_DECREF(entry);
        return NULL;
    }

    return entry;
}

NPY_NO_EXPORT PyObject *
PyArray_typeinforanged(
    char typechar, int typenum, int nbits, int align,
    PyObject *max, PyObject *min, PyTypeObject *type_obj)
{
    PyObject *entry = PyStructSequence_New((PyTypeObject*)HPy_AsPyObject(NULL, PyArray_typeinforangedType));
    if (entry == NULL)
        return NULL;
    PyStructSequence_SET_ITEM(entry, 0, Py_BuildValue("C", typechar));
    PyStructSequence_SET_ITEM(entry, 1, Py_BuildValue("i", typenum));
    PyStructSequence_SET_ITEM(entry, 2, Py_BuildValue("i", nbits));
    PyStructSequence_SET_ITEM(entry, 3, Py_BuildValue("i", align));
    PyStructSequence_SET_ITEM(entry, 4, max);
    PyStructSequence_SET_ITEM(entry, 5, min);
    PyStructSequence_SET_ITEM(entry, 6, Py_BuildValue("O", (PyObject *) type_obj));

    if (PyErr_Occurred()) {
        Py_DECREF(entry);
        return NULL;
    }

    return entry;
}

/* Python version needed for older PyPy */
#if (defined(PYPY_VERSION_NUM) && (PYPY_VERSION_NUM < 0x07020000))
    static int
    PyStructSequence_InitType2(PyTypeObject *type, PyStructSequence_Desc *desc) {
        PyStructSequence_InitType(type, desc);
        if (PyErr_Occurred()) {
            return -1;
        }
        return 0;
    }
#endif

NPY_NO_EXPORT int
typeinfo_init_structsequences(HPyContext *ctx, HPy multiarray_dict)
{
    PyArray_typeinfoType = HPyStructSequence_NewType(ctx, &typeinfo_desc);
    if (HPy_IsNull(PyArray_typeinfoType)) {
        return -1;
    }
    PyArray_typeinforangedType = HPyStructSequence_NewType(ctx, &typeinforanged_desc);
    if (HPy_IsNull(PyArray_typeinforangedType)) {
        return -1;
    }
    if (HPy_SetItem_s(ctx, multiarray_dict,
            "typeinfo", PyArray_typeinfoType) < 0) {
        return -1;
    }
    if (HPy_SetItem_s(ctx, multiarray_dict,
            "typeinforanged", PyArray_typeinforangedType) < 0) {
        return -1;
    }
    return 0;
}

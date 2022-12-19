#ifndef NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_
#define NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_

static NPY_INLINE npy_bool
_is_basic_python_type(PyTypeObject *tp)
{
    return (
        /* Basic number types */
        tp == &PyBool_Type ||
        tp == &PyLong_Type ||
        tp == &PyFloat_Type ||
        tp == &PyComplex_Type ||

        /* Basic sequence types */
        tp == &PyList_Type ||
        tp == &PyTuple_Type ||
        tp == &PyDict_Type ||
        tp == &PySet_Type ||
        tp == &PyFrozenSet_Type ||
        tp == &PyUnicode_Type ||
        tp == &PyBytes_Type ||

        /* other builtins */
        tp == &PySlice_Type ||
        tp == Py_TYPE(Py_None) ||
        tp == Py_TYPE(Py_Ellipsis) ||
        tp == Py_TYPE(Py_NotImplemented) ||

        /* TODO: ndarray, but we can't see PyArray_Type here */

        /* sentinel to swallow trailing || */
        NPY_FALSE
    );
}

static NPY_INLINE npy_bool
_hpy_is_basic_python_type(HPyContext *ctx, HPy obj)
{
    if (HPy_Is(ctx, obj, ctx->h_None) ||
            HPy_Is(ctx, obj, ctx->h_Ellipsis) ||
            HPy_Is(ctx, obj, ctx->h_NotImplemented)) {
        return NPY_TRUE;
    }

     HPy tp = HPy_Type(ctx, obj);

    npy_bool ret = (
        /* Basic number types */
        HPy_Is(ctx, tp, ctx->h_BoolType) ||
        HPy_Is(ctx, tp, ctx->h_LongType) ||
        HPy_Is(ctx, tp, ctx->h_FloatType) ||
        HPy_Is(ctx, tp, ctx->h_ComplexType) ||

        /* Basic sequence types */
        HPy_Is(ctx, tp, ctx->h_ListType) ||
        HPy_Is(ctx, tp, ctx->h_TupleType) ||
        // HPy_Is(ctx, tp, ctx->h_DictType) ||
        // HPy_Is(ctx, tp, ctx->h_SetType) ||
        // HPy_Is(ctx, tp, ctx->h_FrozenSetType) ||
        HPy_Is(ctx, tp, ctx->h_UnicodeType) ||
        HPy_Is(ctx, tp, ctx->h_BytesType) ||

        /* other builtins */
        HPy_Is(ctx, tp, ctx->h_SliceType) ||
        // HPy_Is(ctx, tp, ctx->h_None) ||
        // HPy_Is(ctx, tp, ctx->h_TYPE(Py_Ellipsis) ||
        // HPy_Is(ctx, tp, ctx->h_TYPE(Py_NotImplemented) ||

        /* TODO: ndarray, but we can't see PyArray_Type here */

        /* sentinel to swallow trailing || */
        NPY_FALSE
    );
    HPy_Close(ctx, tp);
    return ret;
}


/*
 * Stripped down version of PyObject_GetAttrString(obj, name) that does not
 * raise PyExc_AttributeError.
 *
 * This allows it to avoid creating then discarding exception objects when
 * performing lookups on objects without any attributes.
 *
 * Returns attribute value on success, NULL without an exception set if
 * there is no such attribute, and NULL with an exception on failure.
 */
static NPY_INLINE PyObject *
maybe_get_attr(PyObject *obj, char const *name)
{
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *res = (PyObject *)NULL;

    /* Attribute referenced by (char *)name */
    if (tp->tp_getattr != NULL) {
        res = (*tp->tp_getattr)(obj, (char *)name);
        if (res == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
        }
    }
    /* Attribute referenced by (PyObject *)name */
    else if (tp->tp_getattro != NULL) {
        PyObject *w = PyUnicode_InternFromString(name);
        if (w == NULL) {
            return (PyObject *)NULL;
        }
        res = (*tp->tp_getattro)(obj, w);
        Py_DECREF(w);
        if (res == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
        }
    }
    return res;
}

static NPY_INLINE HPy
hpy_maybe_get_attr(HPyContext *ctx, HPy obj, char const *name)
{
    return HPy_MaybeGetAttr_s(ctx, obj, name);
}

/*
 * Lookup a special method, following the python approach of looking up
 * on the type object, rather than on the instance itself.
 *
 * Assumes that the special method is a numpy-specific one, so does not look
 * at builtin types, nor does it look at a base ndarray.
 *
 * In future, could be made more like _Py_LookupSpecial
 */
static NPY_INLINE PyObject *
PyArray_LookupSpecial(PyObject *obj, char const *name)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }
    return maybe_get_attr((PyObject *)tp, name);
}

static NPY_INLINE HPy
HPyArray_LookupSpecial_OnType(HPyContext *ctx, HPy type, char const *name)
{
    // HPy variant for callsites that already have the type
    // /* We do not need to check for special attributes on trivial types */
    // // In HPy this would mean multiple HPy_Is calls, so probably not faster than hpy_maybe_get_attr
    // if (_is_basic_python_type(tp)) {
    //     return NULL;
    // }
    return hpy_maybe_get_attr(ctx, type, name);
}

/*
 * PyArray_LookupSpecial_OnInstance:
 *
 * Implements incorrect special method lookup rules, that break the python
 * convention, and looks on the instance, not the type.
 *
 * Kept for backwards compatibility. In future, we should deprecate this.
 */
static NPY_INLINE PyObject *
PyArray_LookupSpecial_OnInstance(PyObject *obj, char const *name)
{
    PyTypeObject *tp = Py_TYPE(obj);

    /* We do not need to check for special attributes on trivial types */
    if (_is_basic_python_type(tp)) {
        return NULL;
    }

    return maybe_get_attr(obj, name);
}

static NPY_INLINE HPy
HPyArray_LookupSpecial_OnInstance(HPyContext *ctx, HPy obj, char const *name)
{
    // /* We do not need to check for special attributes on trivial types */
    // // In HPy this would mean multiple HPy_Is calls, so probably not faster than hpy_maybe_get_attr
    // XXX: However, it breaks otherwise.
    if (_hpy_is_basic_python_type(ctx, obj)) {
        return HPy_NULL;
    }
    return hpy_maybe_get_attr(ctx, obj, name);
}

#endif  /* NUMPY_CORE_SRC_COMMON_GET_ATTR_STRING_H_ */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_


/* DType flags, currently private, since we may just expose functions */
#define NPY_DT_LEGACY 1 << 0
#define NPY_DT_ABSTRACT 1 << 1
#define NPY_DT_PARAMETRIC 1 << 2


typedef PyArray_Descr *(discover_descr_from_pyobject_function)(
        PyArray_DTypeMeta *cls, PyObject *obj);
typedef HPy (hdiscover_descr_from_pyobject_function)(
        HPyContext *ctx, HPy cls, HPy obj);

/*
 * Before making this public, we should decide whether it should pass
 * the type, or allow looking at the object. A possible use-case:
 * `np.array(np.array([0]), dtype=np.ndarray)`
 * Could consider arrays that are not `dtype=ndarray` "scalars".
 */
//typedef int (is_known_scalar_type_function)(
//        PyArray_DTypeMeta *cls, PyTypeObject *obj);
typedef int (is_known_scalar_type_function)(HPyContext *ctx,
        HPy cls, HPy obj);

// typedef PyArray_Descr *(default_descr_function)(PyArray_DTypeMeta *cls);
typedef HPy (default_descr_function)(HPyContext *ctx, HPy cls);
typedef PyArray_DTypeMeta *(common_dtype_function)(
        PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2);
typedef PyArray_Descr *(common_instance_function)(
        PyArray_Descr *dtype1, PyArray_Descr *dtype2);

/*
 * TODO: These two functions are currently only used for experimental DType
 *       API support.  Their relation should be "reversed": NumPy should
 *       always use them internally.
 *       There are open points about "casting safety" though, e.g. setting
 *       elements is currently always unsafe.
 */
typedef int(setitemfunction)(PyArray_Descr *, PyObject *, char *);
typedef PyObject *(getitemfunction)(PyArray_Descr *, char *);


typedef struct {
    /* DType methods, these could be moved into its own struct */
    discover_descr_from_pyobject_function *discover_descr_from_pyobject;
    hdiscover_descr_from_pyobject_function *hdiscover_descr_from_pyobject;
    is_known_scalar_type_function *is_known_scalar_type;
    default_descr_function *default_descr;
    common_dtype_function *common_dtype;
    common_instance_function *common_instance;
    /*
     * Currently only used for experimental user DTypes.
     * Typing as `void *` until NumPy itself uses these (directly).
     */
    setitemfunction *setitem;
    getitemfunction *getitem;
    /*
     * The casting implementation (ArrayMethod) to convert between two
     * instances of this DType, stored explicitly for fast access:
     */
    HPyField within_dtype_castingimpl;
    /*
     * Dictionary of ArrayMethods representing most possible casts
     * (structured and object are exceptions).
     * This should potentially become a weak mapping in the future.
     */
    HPyField castingimpls;

    /*
     * Storage for `descr->f`, since we may need to allow some customizatoin
     * here at least in a transition period and we need to set it on every
     * dtype instance for backward compatibility.  (Keep this at end)
     */
    PyArray_ArrFuncs f;
} NPY_DType_Slots;

#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))
#define HNPY_DTYPE(ctx, descr) (HPy_Type(ctx, descr))
#define NPY_DT_SLOTS(dtype) ((NPY_DType_Slots *)(dtype)->dt_slots)
#define HNPY_DT_SLOTS(ctx, dtype) ((NPY_DType_Slots *)(PyArray_DTypeMeta_AsStruct(ctx, dtype)->dt_slots))

#define NPY_DT_is_legacy(dtype) (((dtype)->flags & NPY_DT_LEGACY) != 0)
#define NPY_DT_is_abstract(dtype) (((dtype)->flags & NPY_DT_ABSTRACT) != 0)
#define NPY_DT_is_parametric(dtype) (((dtype)->flags & NPY_DT_PARAMETRIC) != 0)

static inline PyObject* DTYPE_SLOTS_CASTINGIMPL(PyArray_DTypeMeta *meta) {
    if (HPyField_IsNull(NPY_DT_SLOTS(meta)->castingimpls)) {
        return NULL;
    }
    HPyContext *ctx = npy_get_context();
    HPy owner = HPy_FromPyObject(ctx, (PyObject*) meta);
    HPy res = HPyField_Load(ctx, owner, NPY_DT_SLOTS(meta)->castingimpls);
    PyObject *py_res = HPy_AsPyObject(ctx, res);
    HPy_Close(ctx, owner);
    HPy_Close(ctx, res);
    Py_DECREF(py_res);  // to simulate the borrowed reference...
    return py_res;
}

static inline PyObject* DTYPE_SLOTS_WITHIN_DTYPE_CASTINGIMPL(PyArray_DTypeMeta *meta) {
    if (HPyField_IsNull(NPY_DT_SLOTS(meta)->within_dtype_castingimpl)) {
        return NULL;
    }
    HPyContext *ctx = npy_get_context();
    HPy owner = HPy_FromPyObject(ctx, (PyObject*) meta);
    HPy res = HPyField_Load(ctx, owner, NPY_DT_SLOTS(meta)->within_dtype_castingimpl);
    PyObject *py_res = HPy_AsPyObject(ctx, res);
    HPy_Close(ctx, owner);
    HPy_Close(ctx, res);
    Py_DECREF(py_res);  // to simulate the borrowed reference...
    return py_res;
}

static inline HPy HPY_DTYPE_SLOTS_CASTINGIMPL(HPyContext *ctx, HPy h_meta, PyArray_DTypeMeta *meta) {
    if (HPyField_IsNull(NPY_DT_SLOTS(meta)->castingimpls)) {
        return HPy_NULL;
    }
    return HPyField_Load(ctx, h_meta, NPY_DT_SLOTS(meta)->castingimpls);
}

static inline HPy HPY_DTYPE_SLOTS_CASTINGIMPL0(HPyContext *ctx, HPy h_meta) {
    NPY_DType_Slots *slots = HNPY_DT_SLOTS(ctx, h_meta);
    if (HPyField_IsNull(slots->castingimpls)) {
        return HPy_NULL;
    }
    return HPyField_Load(ctx, h_meta, slots->castingimpls);
}

static inline HPy HPY_DTYPE_SLOTS_WITHIN_DTYPE_CASTINGIMPL(HPyContext *ctx, HPy h_meta, PyArray_DTypeMeta *meta) {
    if (HPyField_IsNull(NPY_DT_SLOTS(meta)->within_dtype_castingimpl)) {
        return HPy_NULL;
    }
    return HPyField_Load(ctx, h_meta, NPY_DT_SLOTS(meta)->within_dtype_castingimpl);
}


/*
 * Macros for convenient classmethod calls, since these require
 * the DType both for the slot lookup and as first arguments.
 *
 * (Macros may include NULL checks where appropriate)
 */
#define NPY_DT_CALL_discover_descr_from_pyobject(dtype, obj)  \
    NPY_DT_SLOTS(dtype)->discover_descr_from_pyobject(dtype, obj)
#define HNPY_DT_CALL_discover_descr_from_pyobject(ctx, dtype, obj)  \
    HNPY_DT_SLOTS(ctx, dtype)->hdiscover_descr_from_pyobject(ctx, dtype, obj)
#define NPY_DT_CALL_is_known_scalar_type(dtype, obj)  \
    (NPY_DT_SLOTS(dtype)->is_known_scalar_type != NULL  \
        && NPY_DT_SLOTS(dtype)->is_known_scalar_type(dtype, obj))
#define NPY_DT_CALL_default_descr(dtype)  \
    default_descr_function_trampoline(dtype)
#define HNPY_DT_CALL_default_descr(ctx, dtype, dtype_data)  \
    (((NPY_DType_Slots *)((dtype_data)->dt_slots))->default_descr(ctx, dtype))
#define NPY_DT_CALL_common_dtype(dtype, other)  \
    NPY_DT_SLOTS(dtype)->common_dtype(dtype, other)
#define NPY_DT_CALL_getitem(descr, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->getitem(descr, data_ptr)
#define NPY_DT_CALL_setitem(descr, value, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->setitem(descr, value, data_ptr)

static inline int
HNPY_DT_CALL_is_known_scalar_type(HPyContext *ctx, HPy h_meta, PyArray_DTypeMeta *meta, HPy obj) {
    NPY_DType_Slots *slots = NPY_DT_SLOTS(meta);
    int res = 0;
    if (slots->is_known_scalar_type != NULL) {
        res = slots->is_known_scalar_type(ctx, h_meta, obj);
    }
    return res;
}

static NPY_INLINE HPy
hdtypemeta_call_default_descr(HPyContext *ctx, HPy dtype_meta)
{
    return HNPY_DT_SLOTS(ctx, dtype_meta)->default_descr(ctx, dtype_meta);
}
/*
 * This function will hopefully be phased out or replaced, but was convenient
 * for incremental implementation of new DTypes based on DTypeMeta.
 * (Error checking is not required for DescrFromType, assuming that the
 * type is valid.)
 */
static NPY_INLINE PyArray_DTypeMeta *
PyArray_DTypeFromTypeNum(int typenum)
{
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    PyArray_DTypeMeta *dtype = NPY_DTYPE(descr);
    Py_INCREF(dtype);
    Py_DECREF(descr);
    return dtype;
}

static NPY_INLINE HPy /* (PyArray_DTypeMeta *) */
HPyArray_DTypeFromTypeNum(HPyContext *ctx, int typenum)
{
    HPy /* (PyArray_Descr *) */ descr = HPyArray_DescrFromType(ctx, typenum);
    HPy /* (PyArray_DTypeMeta *) */ dtype = HNPY_DTYPE(ctx, descr);
    HPy_Close(ctx, descr);
    return dtype;
}

static NPY_INLINE PyObject *
dtypemeta_get_castingimpls(PyArray_DTypeMeta *cls)
{
    HPyContext *ctx = npy_get_context();
    HPy h_cls = HPy_FromPyObject(ctx, (PyObject *)cls);
    HPy h_castingimpls = HPyField_Load(ctx, h_cls, HNPY_DT_SLOTS(ctx, h_cls)->castingimpls);
    PyObject *res = HPy_AsPyObject(ctx, h_castingimpls);
    HPy_Close(ctx, h_castingimpls);
    HPy_Close(ctx, h_cls);
    return res;
}

static NPY_INLINE HPy
hdtypemeta_get_singleton(HPyContext *ctx, HPy h_meta)
{
    PyArray_DTypeMeta *data = PyArray_DTypeMeta_AsStruct(ctx, h_meta);
    return HPyField_Load(ctx, h_meta, data->singleton);
}

static NPY_INLINE PyArray_Descr *
dtypemeta_get_singleton(PyArray_DTypeMeta *meta)
{
    HPyContext *ctx = npy_get_context();
    HPy h_meta = HPy_FromPyObject(ctx, (PyObject *)meta);
    HPy h_singleton = HPyField_Load(ctx, h_meta, meta->singleton);
    PyArray_Descr *res = (PyArray_Descr *)HPy_AsPyObject(ctx, h_singleton);
    HPy_Close(ctx, h_meta);
    HPy_Close(ctx, h_singleton);
    Py_DECREF(res);  // to simulate the borrowed reference...
    return res;
}

NPY_NO_EXPORT int
python_builtins_are_known_scalar_types(HPyContext *ctx,
        HPy cls, HPy pytype);

NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(HPyContext *ctx, HPy h_descr, PyArray_Descr *dtypem);

NPY_NO_EXPORT PyArray_Descr *
default_descr_function_trampoline(PyArray_DTypeMeta *cls);

NPY_NO_EXPORT PyArray_Descr *
discover_descr_from_pyobject_function_trampoline(PyArray_DTypeMeta *cls, PyObject *obj);

NPY_NO_EXPORT HPy
hdiscover_descr_from_pyobject_function_trampoline(HPyContext *ctx, HPy cls, HPy obj);

extern NPY_NO_EXPORT HPyType_Spec PyArrayDTypeMeta_Type_spec;
extern NPY_NO_EXPORT HPyGlobal HPyArrayDTypeMeta_Type;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_ */

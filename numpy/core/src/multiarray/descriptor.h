#ifndef NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_

NPY_NO_EXPORT HPy
arraydescr_protocol_typestr_get(HPyContext *, HPy, void *);
NPY_NO_EXPORT HPy
arraydescr_protocol_descr_get(HPyContext *, HPy, void *);

extern NPY_NO_EXPORT HPyDef array_set_typeDict;
extern HPyGlobal descr_typeDict;

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_try_convert_from_dtype_attr(PyObject *obj);

NPY_NO_EXPORT HPy // PyArray_Descr *
_hpy_arraydescr_try_convert_from_dtype_attr(HPyContext *ctx, HPy obj);

NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype);

/*
 * Filter the fields of a dtype to only those in the list of strings, ind.
 *
 * No type checking is performed on the input.
 *
 * Raises:
 *   ValueError - if a field is repeated
 *   KeyError - if an invalid field name (or any field title) is used
 */
NPY_NO_EXPORT PyArray_Descr *
arraydescr_field_subset_view(PyArray_Descr *self, PyObject *ind);

NPY_NO_EXPORT HPy
harraydescr_field_subset_view(HPyContext *ctx,
                    PyArray_Descr *self_data,
                    HPy ind);

extern NPY_NO_EXPORT char const *_datetime_strings[];

extern NPY_NO_EXPORT HPyType_Spec PyArrayDescr_TypeFull_spec;

NPY_NO_EXPORT int
HPyArray_DescrConverter(HPyContext *ctx, HPy obj, HPy *at);

NPY_NO_EXPORT int
HPyArray_DescrConverter2(HPyContext *ctx, HPy obj, HPy *at);

NPY_NO_EXPORT HPy
HPyArray_DescrNewFromType(HPyContext *ctx, int type_num);

NPY_NO_EXPORT HPy
HPyArray_DescrNew(HPyContext *ctx, HPy h_base);

NPY_NO_EXPORT extern PyArray_DTypeMeta PyArrayDescr_TypeFull;
NPY_NO_EXPORT extern PyTypeObject *_PyArrayDescr_Type_p;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DESCRIPTOR_H_ */

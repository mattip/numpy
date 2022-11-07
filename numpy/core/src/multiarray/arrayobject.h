#ifndef _MULTIARRAYMODULE
#error You should not include this
#endif

#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_
#include "hpy.h"

NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip);

NPY_NO_EXPORT HPy
_hpy_strings_richcompare(HPyContext *ctx, 
                            HPy /* PyArrayObject * */ h_self, 
                            HPy /* PyArrayObject * */ h_other, int cmp_op,
                     int rstrip);

NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op);

NPY_NO_EXPORT int
array_might_be_written(PyArrayObject *obj);

/*
 * This flag is used to mark arrays which we would like to, in the future,
 * turn into views. It causes a warning to be issued on the first attempt to
 * write to the array (but the write is allowed to succeed).
 *
 * This flag is for internal use only, and may be removed in a future release,
 * which is why the #define is not exposed to user code. Currently it is set
 * on arrays returned by ndarray.diagonal.
 */
static const int NPY_ARRAY_WARN_ON_WRITE = (1 << 31);

extern NPY_NO_EXPORT HPyType_Spec PyArray_Type_spec;
extern NPY_NO_EXPORT HPyGlobal HPyArray_Type;
extern NPY_NO_EXPORT HPyType_Spec PyArrayFlags_Type_Spec;
extern NPY_NO_EXPORT HPyGlobal HPyArrayDescr_Type;

NPY_NO_EXPORT extern HPyGlobal g_checkfunc;

NPY_NO_EXPORT int
HPyArray_ElementStrides(HPyContext *ctx, HPy obj);

NPY_NO_EXPORT int
HPyArray_SetBaseObject(HPyContext *ctx, HPy h_arr, PyArrayObject *arr, HPy obj_in);

NPY_NO_EXPORT int
HPyArray_FailUnlessWriteable(HPyContext *ctx, HPy obj, const char *name);

NPY_NO_EXPORT int
HPyArray_FailUnlessWriteableWithStruct(HPyContext *ctx, HPy obj, PyArrayObject *obj_data, const char *name);

NPY_NO_EXPORT int
HPyArray_CopyObject(HPyContext *ctx, HPy h_dest, PyArrayObject *dest, HPy h_src_object);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_ */

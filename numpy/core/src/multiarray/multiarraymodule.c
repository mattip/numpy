/*
  Python Multiarray Module -- A useful collection of functions for creating and
  using ndarrays

  Original file
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  Modified for numpy in 2005

  Travis E. Oliphant
  oliphant@ee.byu.edu
  Brigham Young University
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"
#include "npy_argparse.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "convert_datatype.h"
#include "legacy_dtype_implementation.h"

NPY_NO_EXPORT int NPY_NUMUSERTYPES = 0;

/* Internal APIs */
#include "alloc.h"
#include "abstractdtypes.h"
#include "array_coercion.h"
#include "arrayfunction_override.h"
#include "arraytypes.h"
#include "npy_buffer.h"
#include "arrayobject.h"
#include "iterators.h"
#include "mapping.h"
#include "hashdescr.h"
#include "descriptor.h"
#include "dragon4.h"
#include "calculation.h"
#include "number.h"
#include "scalartypes.h"
#include "convert_datatype.h"
#include "conversion_utils.h"
#include "nditer_pywrap.h"
#include "methods.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "datetime_busday.h"
#include "datetime_busdaycal.h"
#include "item_selection.h"
#include "shape.h"
#include "ctors.h"
#include "array_assign.h"
#include "common.h"
#include "multiarraymodule.h"
#include "cblasfuncs.h"
#include "vdot.h"
#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "compiled_base.h"
#include "mem_overlap.h"
#include "typeinfo.h"
#include "scalarapi.h"

#include "get_attr_string.h"
#include "experimental_public_dtype_api.h"  /* _get_experimental_dtype_api */
#include "textreading/readtext.h"  /* _readtext_from_file_object */

#include "npy_dlpack.h"

// Added for HPy port:
#include "hpy.h"
#include "hpy_utils.h"
#include "scalarapi.h"
#include "nditer_hpy.h"

/*
 *****************************************************************************
 **                    INCLUDE GENERATED CODE                               **
 *****************************************************************************
 */
#include "funcs.inc"
#include "umathmodule.h"

NPY_NO_EXPORT int initscalarmath(PyObject *);
NPY_NO_EXPORT int set_matmul_flags(HPyContext *ctx, HPy d); /* in ufunc_object.c */

/*
 * global variable to determine if legacy printing is enabled, accessible from
 * C. For simplicity the mode is encoded as an integer where INT_MAX means no
 * legacy mode, and '113'/'121' means 1.13/1.21 legacy mode; and 0 maps to
 * INT_MAX. We can upgrade this if we have more complex requirements in the
 * future.
 */
int npy_legacy_print_mode = INT_MAX;

HPyDef_METH(set_legacy_print_mode, "set_legacy_print_mode", HPyFunc_VARARGS)
static HPy
set_legacy_print_mode_impl(HPyContext *ctx, HPy NPY_UNUSED(self), HPy *args, HPy_ssize_t nargs)
{
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "i", &npy_legacy_print_mode)) {
        return HPy_NULL;
    }
    if (!npy_legacy_print_mode) {
        npy_legacy_print_mode = INT_MAX;
    }
    return HPy_Dup(ctx, ctx->h_None);
}

NPY_NO_EXPORT PyTypeObject* _PyArray_Type_p = NULL;
NPY_NO_EXPORT HPyContext *numpy_global_ctx = NULL;
NPY_NO_EXPORT HPyGlobal HPyArray_Type;
NPY_NO_EXPORT HPyGlobal HPyArrayDescr_Type;
NPY_NO_EXPORT HPyGlobal HPyArrayFlags_Type;

/* Only here for API compatibility */
NPY_NO_EXPORT PyTypeObject PyBigArray_Type;

extern NPY_NO_EXPORT HPyType_Spec NpyIter_Type_Spec;


/*NUMPY_API
 * Get Priority from object
 */
NPY_NO_EXPORT double
PyArray_GetPriority(PyObject *obj, double default_)
{
    PyObject *ret;
    double priority = NPY_PRIORITY;

    if (PyArray_CheckExact(obj)) {
        return priority;
    }
    else if (PyArray_CheckAnyScalarExact(obj)) {
        return NPY_SCALAR_PRIORITY;
    }

    ret = PyArray_LookupSpecial_OnInstance(obj, "__array_priority__");
    if (ret == NULL) {
        if (PyErr_Occurred()) {
            PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
        }
        return default_;
    }

    priority = PyFloat_AsDouble(ret);
    Py_DECREF(ret);
    return priority;
}

/*HPY_NUMPY_API
 * Get Priority from object
 */
NPY_NO_EXPORT double
HPyArray_GetPriority(HPyContext *ctx, HPy obj, double default_)
{
    double priority = NPY_PRIORITY;

    if (HPyArray_CheckExact(ctx, obj)) {
        return priority;
    }
    else if (HPyArray_CheckAnyScalarExact(ctx, obj)) {
        return NPY_SCALAR_PRIORITY;
    }

    HPy ret = HPyArray_LookupSpecial_OnInstance(ctx, obj, "__array_priority__");
    if (HPy_IsNull(ret)) {
        if (HPyErr_Occurred(ctx)) {
            HPyErr_Clear(ctx); /* TODO[gh-14801]: propagate crashes during attribute access? */
        }
        return default_;
    }

    priority = HPyFloat_AsDouble(ctx, ret);
    HPy_Close(ctx, ret);
    return priority;
}

/*NUMPY_API
 * Multiply a List of ints
 */
NPY_NO_EXPORT int
PyArray_MultiplyIntList(int const *l1, int n)
{
    int s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Multiply a List
 */
NPY_NO_EXPORT npy_intp
PyArray_MultiplyList(npy_intp const *l1, int n)
{
    npy_intp s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Multiply a List of Non-negative numbers with over-flow detection.
 */
NPY_NO_EXPORT npy_intp
PyArray_OverflowMultiplyList(npy_intp const *l1, int n)
{
    npy_intp prod = 1;
    int i;

    for (i = 0; i < n; i++) {
        npy_intp dim = l1[i];

        if (dim == 0) {
            return 0;
        }
        if (npy_mul_with_overflow_intp(&prod, prod, dim)) {
            return -1;
        }
    }
    return prod;
}

/*NUMPY_API
 * Produce a pointer into array
 */
NPY_NO_EXPORT void *
PyArray_GetPtr(PyArrayObject *obj, npy_intp const* ind)
{
    int n = PyArray_NDIM(obj);
    npy_intp *strides = PyArray_STRIDES(obj);
    char *dptr = PyArray_DATA(obj);

    while (n--) {
        dptr += (*strides++) * (*ind++);
    }
    return (void *)dptr;
}

/*NUMPY_API
 * Compare Lists
 */
NPY_NO_EXPORT int
PyArray_CompareLists(npy_intp const *l1, npy_intp const *l2, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        if (l1[i] != l2[i]) {
            return 0;
        }
    }
    return 1;
}

/*
 * simulates a C-style 1-3 dimensional array which can be accessed using
 * ptr[i]  or ptr[i][j] or ptr[i][j][k] -- requires pointer allocation
 * for 2-d and 3-d.
 *
 * For 2-d and up, ptr is NOT equivalent to a statically defined
 * 2-d or 3-d array.  In particular, it cannot be passed into a
 * function that requires a true pointer to a fixed-size array.
 */

/*NUMPY_API
 * Simulate a C-array
 * steals a reference to typedescr -- can be NULL
 */
NPY_NO_EXPORT int
PyArray_AsCArray(PyObject **op, void *ptr, npy_intp *dims, int nd,
                 PyArray_Descr* typedescr)
{
    PyArrayObject *ap;
    npy_intp n, m, i, j;
    char **ptr2;
    char ***ptr3;

    if ((nd < 1) || (nd > 3)) {
        PyErr_SetString(PyExc_ValueError,
                        "C arrays of only 1-3 dimensions available");
        Py_XDECREF(typedescr);
        return -1;
    }
    if ((ap = (PyArrayObject*)PyArray_FromAny(*op, typedescr, nd, nd,
                                      NPY_ARRAY_CARRAY, NULL)) == NULL) {
        return -1;
    }
    switch(nd) {
    case 1:
        *((char **)ptr) = PyArray_DATA(ap);
        break;
    case 2:
        n = PyArray_DIMS(ap)[0];
        ptr2 = (char **)PyArray_malloc(n * sizeof(char *));
        if (!ptr2) {
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; i++) {
            ptr2[i] = PyArray_BYTES(ap) + i*PyArray_STRIDES(ap)[0];
        }
        *((char ***)ptr) = ptr2;
        break;
    case 3:
        n = PyArray_DIMS(ap)[0];
        m = PyArray_DIMS(ap)[1];
        ptr3 = (char ***)PyArray_malloc(n*(m+1) * sizeof(char *));
        if (!ptr3) {
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; i++) {
            ptr3[i] = (char **) &ptr3[n + m * i];
            for (j = 0; j < m; j++) {
                ptr3[i][j] = PyArray_BYTES(ap) + i*PyArray_STRIDES(ap)[0] + j*PyArray_STRIDES(ap)[1];
            }
        }
        *((char ****)ptr) = ptr3;
    }
    if (nd) {
        memcpy(dims, PyArray_DIMS(ap), nd*sizeof(npy_intp));
    }
    *op = (PyObject *)ap;
    return 0;
}

/* Deprecated --- Use PyArray_AsCArray instead */

/*NUMPY_API
 * Convert to a 1D C-array
 */
NPY_NO_EXPORT int
PyArray_As1D(PyObject **NPY_UNUSED(op), char **NPY_UNUSED(ptr),
             int *NPY_UNUSED(d1), int NPY_UNUSED(typecode))
{
    /* 2008-07-14, 1.5 */
    PyErr_SetString(PyExc_NotImplementedError,
                "PyArray_As1D: use PyArray_AsCArray.");
    return -1;
}

/*NUMPY_API
 * Convert to a 2D C-array
 */
NPY_NO_EXPORT int
PyArray_As2D(PyObject **NPY_UNUSED(op), char ***NPY_UNUSED(ptr),
             int *NPY_UNUSED(d1), int *NPY_UNUSED(d2), int NPY_UNUSED(typecode))
{
    /* 2008-07-14, 1.5 */
    PyErr_SetString(PyExc_NotImplementedError,
                "PyArray_As2D: use PyArray_AsCArray.");
    return -1;
}

/* End Deprecated */

/*NUMPY_API
 * Free pointers created if As2D is called
 */
NPY_NO_EXPORT int
PyArray_Free(PyObject *op, void *ptr)
{
    PyArrayObject *ap = (PyArrayObject *)op;

    if ((PyArray_NDIM(ap) < 1) || (PyArray_NDIM(ap) > 3)) {
        return -1;
    }
    if (PyArray_NDIM(ap) >= 2) {
        PyArray_free(ptr);
    }
    Py_DECREF(ap);
    return 0;
}

/*
 * Get the ndarray subclass with the highest priority
 */
NPY_NO_EXPORT PyTypeObject *
PyArray_GetSubType(int narrays, PyArrayObject **arrays) {
    PyTypeObject *subtype = &PyArray_Type;
    double priority = NPY_PRIORITY;
    int i;

    /* Get the priority subtype for the array */
    for (i = 0; i < narrays; ++i) {
        if (Py_TYPE(arrays[i]) != subtype) {
            double pr = PyArray_GetPriority((PyObject *)(arrays[i]), 0.0);
            if (pr > priority) {
                priority = pr;
                subtype = Py_TYPE(arrays[i]);
            }
        }
    }

    return subtype;
}

NPY_NO_EXPORT HPy
HPyArray_GetSubType(HPyContext *ctx, int narrays, HPy /* PyArrayObject ** */ *arrays) {
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    HPy subtype = array_type; // PyTypeObject *
    double priority = NPY_PRIORITY;
    int i;

    /* Get the priority subtype for the array */
    for (i = 0; i < narrays; ++i) {
        if (!HPy_TypeCheck(ctx, arrays[i], subtype)) {
            double pr = HPyArray_GetPriority(ctx, (arrays[i]), 0.0);
            if (pr > priority) {
                priority = pr;
                HPy_Close(ctx, subtype);
                subtype = HPy_Dup(ctx, arrays[i]);
            }
        }
    }

    return subtype;
}

/*
 * Concatenates a list of ndarrays.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ConcatenateArrays(int narrays, PyArrayObject **arrays, int axis,
                          PyArrayObject* ret, PyArray_Descr *dtype,
                          NPY_CASTING casting)
{
    int iarrays, idim, ndim;
    npy_intp shape[NPY_MAXDIMS];
    PyArrayObject_fields *sliding_view = NULL;

    if (narrays <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need at least one array to concatenate");
        return NULL;
    }

    /* All the arrays must have the same 'ndim' */
    ndim = PyArray_NDIM(arrays[0]);

    if (ndim == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "zero-dimensional arrays cannot be concatenated");
        return NULL;
    }

    /* Handle standard Python negative indexing */
    if (check_and_adjust_axis(&axis, ndim) < 0) {
        return NULL;
    }

    /*
     * Figure out the final concatenated shape starting from the first
     * array's shape.
     */
    memcpy(shape, PyArray_SHAPE(arrays[0]), ndim * sizeof(shape[0]));
    for (iarrays = 1; iarrays < narrays; ++iarrays) {
        npy_intp *arr_shape;

        if (PyArray_NDIM(arrays[iarrays]) != ndim) {
            PyErr_Format(PyExc_ValueError,
                         "all the input arrays must have same number of "
                         "dimensions, but the array at index %d has %d "
                         "dimension(s) and the array at index %d has %d "
                         "dimension(s)",
                         0, ndim, iarrays, PyArray_NDIM(arrays[iarrays]));
            return NULL;
        }
        arr_shape = PyArray_SHAPE(arrays[iarrays]);

        for (idim = 0; idim < ndim; ++idim) {
            /* Build up the size of the concatenation axis */
            if (idim == axis) {
                shape[idim] += arr_shape[idim];
            }
            /* Validate that the rest of the dimensions match */
            else if (shape[idim] != arr_shape[idim]) {
                PyErr_Format(PyExc_ValueError,
                             "all the input array dimensions for the "
                             "concatenation axis must match exactly, but "
                             "along dimension %d, the array at index %d has "
                             "size %d and the array at index %d has size %d",
                             idim, 0, shape[idim], iarrays, arr_shape[idim]);
                return NULL;
            }
        }
    }

    if (ret != NULL) {
        assert(dtype == NULL);
        if (PyArray_NDIM(ret) != ndim) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array has wrong dimensionality");
            return NULL;
        }
        if (!PyArray_CompareLists(shape, PyArray_SHAPE(ret), ndim)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array is the wrong shape");
            return NULL;
        }
        Py_INCREF(ret);
    }
    else {
        npy_intp s, strides[NPY_MAXDIMS];
        int strideperm[NPY_MAXDIMS];

        /* Get the priority subtype for the array */
        PyTypeObject *subtype = PyArray_GetSubType(narrays, arrays);
        PyArray_Descr *descr = PyArray_FindConcatenationDescriptor(
                narrays, arrays,  (PyObject *)dtype);
        if (descr == NULL) {
            return NULL;
        }

        /*
         * Figure out the permutation to apply to the strides to match
         * the memory layout of the input arrays, using ambiguity
         * resolution rules matching that of the NpyIter.
         */
        PyArray_CreateMultiSortedStridePerm(narrays, arrays, ndim, strideperm);
        s = descr->elsize;
        for (idim = ndim-1; idim >= 0; --idim) {
            int iperm = strideperm[idim];
            strides[iperm] = s;
            s *= shape[iperm];
        }

        /* Allocate the array for the result. This steals the 'dtype' reference. */
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                subtype, descr, ndim, shape, strides, NULL, 0, NULL,
                NULL, 0, 1);
        if (ret == NULL) {
            return NULL;
        }
        assert(PyArray_DESCR(ret) == descr);
    }

    /*
     * Create a view which slides through ret for assigning the
     * successive input arrays.
     */
    sliding_view = (PyArrayObject_fields *)PyArray_View(ret,
                                                        NULL, &PyArray_Type);
    if (sliding_view == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        /* Set the dimension to match the input array's */
        sliding_view->dimensions[axis] = PyArray_SHAPE(arrays[iarrays])[axis];

        /* Copy the data for this array */
        if (PyArray_AssignArray((PyArrayObject *)sliding_view, arrays[iarrays],
                            NULL, casting) < 0) {
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* Slide to the start of the next window */
        sliding_view->data += sliding_view->dimensions[axis] *
                                 sliding_view->strides[axis];
    }

    Py_DECREF(sliding_view);
    return ret;
}


/*
 * Concatenates a list of ndarrays.
 */
NPY_NO_EXPORT HPy // PyArrayObject *
HPyArray_ConcatenateArrays(HPyContext *ctx, int narrays, 
                            HPy /* PyArrayObject ** */ *arrays, int axis,
                            HPy /* PyArrayObject* */ ret,
                            HPy /* PyArray_Descr * */ dtype,
                            NPY_CASTING casting)
{
    int iarrays, idim, ndim;
    npy_intp shape[NPY_MAXDIMS];
    HPy sliding_view = HPy_NULL; // PyArrayObject_fields *

    if (narrays <= 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "need at least one array to concatenate");
        return HPy_NULL;
    }

    /* All the arrays must have the same 'ndim' */
    PyArrayObject *array_0 = PyArrayObject_AsStruct(ctx, arrays[0]);
    ndim = PyArray_NDIM(array_0);

    if (ndim == 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "zero-dimensional arrays cannot be concatenated");
        return HPy_NULL;
    }

    CAPI_WARN("calling check_and_adjust_axis");
    /* Handle standard Python negative indexing */
    if (check_and_adjust_axis(&axis, ndim) < 0) {
        return HPy_NULL;
    }

    /*
     * Figure out the final concatenated shape starting from the first
     * array's shape.
     */
    memcpy(shape, PyArray_SHAPE(array_0), ndim * sizeof(shape[0]));
    for (iarrays = 1; iarrays < narrays; ++iarrays) {
        npy_intp *arr_shape;

        PyArrayObject *iarrays_struct = PyArrayObject_AsStruct(ctx, arrays[iarrays]);
        if (PyArray_NDIM(iarrays_struct) != ndim) {
            HPyErr_Format_p(ctx, ctx->h_ValueError,
                         "all the input arrays must have same number of "
                         "dimensions, but the array at index %d has %d "
                         "dimension(s) and the array at index %d has %d "
                         "dimension(s)",
                         0, ndim, iarrays, PyArray_NDIM(iarrays_struct));
            return HPy_NULL;
        }
        arr_shape = PyArray_SHAPE(iarrays_struct);

        for (idim = 0; idim < ndim; ++idim) {
            /* Build up the size of the concatenation axis */
            if (idim == axis) {
                shape[idim] += arr_shape[idim];
            }
            /* Validate that the rest of the dimensions match */
            else if (shape[idim] != arr_shape[idim]) {
                HPyErr_Format_p(ctx, ctx->h_ValueError,
                             "all the input array dimensions for the "
                             "concatenation axis must match exactly, but "
                             "along dimension %d, the array at index %d has "
                             "size %d and the array at index %d has size %d",
                             idim, 0, shape[idim], iarrays, arr_shape[idim]);
                return HPy_NULL;
            }
        }
    }

    PyArrayObject* ret_struct = PyArrayObject_AsStruct(ctx, ret);
    if (!HPy_IsNull(ret)) {
        assert(HPy_IsNull(dtype));
        if (PyArray_NDIM(ret_struct) != ndim) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Output array has wrong dimensionality");
            return HPy_NULL;
        }
        if (!PyArray_CompareLists(shape, PyArray_SHAPE(ret_struct), ndim)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Output array is the wrong shape");
            return HPy_NULL;
        }
        HPy_Close(ctx, ret);
    }
    else {
        npy_intp s, strides[NPY_MAXDIMS];
        int strideperm[NPY_MAXDIMS];

        /* Get the priority subtype for the array */
        HPy subtype = HPyArray_GetSubType(ctx, narrays, arrays); // PyTypeObject *
        HPy descr = HPyArray_FindConcatenationDescriptor(ctx,
                narrays, arrays, dtype); // PyArray_Descr *
        if (HPy_IsNull(descr)) {
            return HPy_NULL;
        }

        /*
         * Figure out the permutation to apply to the strides to match
         * the memory layout of the input arrays, using ambiguity
         * resolution rules matching that of the NpyIter.
         */
        HPyArray_CreateMultiSortedStridePerm(ctx, narrays, arrays, ndim, strideperm);
        PyArray_Descr *descr_struct = PyArray_Descr_AsStruct(ctx, descr);
        s = descr_struct->elsize;
        for (idim = ndim-1; idim >= 0; --idim) {
            int iperm = strideperm[idim];
            strides[iperm] = s;
            s *= shape[iperm];
        }

        /* Allocate the array for the result. This steals the 'dtype' reference. */
        ret = HPyArray_NewFromDescr_int(ctx,
                subtype, descr, ndim, shape, strides, NULL, 0, HPy_NULL,
                HPy_NULL, 0, 1);
        if (HPy_IsNull(ret)) {
            return HPy_NULL;
        }
        // assert(PyArray_DESCR(ret) == descr); TODO
    }

    /*
     * Create a view which slides through ret for assigning the
     * successive input arrays.
     */
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    HPy ret_descr = HPyArray_DESCR(ctx, ret, ret_struct);
    sliding_view = HPyArray_View(ctx, ret, ret_struct, ret_descr, HPy_NULL, array_type);
    HPy_Close(ctx, ret_descr);
    if (HPy_IsNull(sliding_view)) {
        HPy_Close(ctx, ret);
        return HPy_NULL;
    }
    PyArrayObject_fields *sliding_view_struct = (PyArrayObject_fields *)PyArrayObject_AsStruct(ctx, sliding_view);
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        PyArrayObject *iarrays_struct = PyArrayObject_AsStruct(ctx, arrays[iarrays]);
        /* Set the dimension to match the input array's */
        sliding_view_struct->dimensions[axis] = PyArray_SHAPE(iarrays_struct)[axis];

        /* Copy the data for this array */
        if (HPyArray_AssignArray(ctx, sliding_view, arrays[iarrays],
                            HPy_NULL, casting) < 0) {
            HPy_Close(ctx, sliding_view);
            HPy_Close(ctx, ret);
            return HPy_NULL;
        }

        /* Slide to the start of the next window */
        sliding_view_struct->data += sliding_view_struct->dimensions[axis] *
                                 sliding_view_struct->strides[axis];
    }

    HPy_Close(ctx, sliding_view);
    return ret;
}

/*
 * Concatenates a list of ndarrays, flattening each in the specified order.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ConcatenateFlattenedArrays(int narrays, PyArrayObject **arrays,
                                   NPY_ORDER order, PyArrayObject *ret,
                                   PyArray_Descr *dtype, NPY_CASTING casting,
                                   npy_bool casting_not_passed)
{
    int iarrays;
    npy_intp shape = 0;
    PyArrayObject_fields *sliding_view = NULL;

    if (narrays <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need at least one array to concatenate");
        return NULL;
    }

    /*
     * Figure out the final concatenated shape starting from the first
     * array's shape.
     */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        shape += PyArray_SIZE(arrays[iarrays]);
        /* Check for overflow */
        if (shape < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "total number of elements "
                            "too large to concatenate");
            return NULL;
        }
    }

    int out_passed = 0;
    if (ret != NULL) {
        assert(dtype == NULL);
        out_passed = 1;
        if (PyArray_NDIM(ret) != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array must be 1D");
            return NULL;
        }
        if (shape != PyArray_SIZE(ret)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array is the wrong size");
            return NULL;
        }
        Py_INCREF(ret);
    }
    else {
        npy_intp stride;

        /* Get the priority subtype for the array */
        PyTypeObject *subtype = PyArray_GetSubType(narrays, arrays);

        PyArray_Descr *descr = PyArray_FindConcatenationDescriptor(
                narrays, arrays, (PyObject *)dtype);
        if (descr == NULL) {
            return NULL;
        }

        stride = descr->elsize;

        /* Allocate the array for the result. This steals the 'dtype' reference. */
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                subtype, descr,  1, &shape, &stride, NULL, 0, NULL,
                NULL, 0, 1);
        if (ret == NULL) {
            return NULL;
        }
        assert(PyArray_DESCR(ret) == descr);
    }

    /*
     * Create a view which slides through ret for assigning the
     * successive input arrays.
     */
    sliding_view = (PyArrayObject_fields *)PyArray_View(ret,
                                                        NULL, &PyArray_Type);
    if (sliding_view == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    int give_deprecation_warning = 1;  /* To give warning for just one input array. */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        /* Adjust the window dimensions for this array */
        sliding_view->dimensions[0] = PyArray_SIZE(arrays[iarrays]);

        if (!PyArray_CanCastArrayTo(
                arrays[iarrays], PyArray_DESCR(ret), casting)) {
            /* This should be an error, but was previously allowed here. */
            if (casting_not_passed && out_passed) {
                /* NumPy 1.20, 2020-09-03 */
                if (give_deprecation_warning && DEPRECATE(
                        "concatenate() with `axis=None` will use same-kind "
                        "casting by default in the future. Please use "
                        "`casting='unsafe'` to retain the old behaviour. "
                        "In the future this will be a TypeError.") < 0) {
                    Py_DECREF(sliding_view);
                    Py_DECREF(ret);
                    return NULL;
                }
                give_deprecation_warning = 0;
            }
            else {
                npy_set_invalid_cast_error(
                        PyArray_DESCR(arrays[iarrays]), PyArray_DESCR(ret),
                        casting, PyArray_NDIM(arrays[iarrays]) == 0);
                Py_DECREF(sliding_view);
                Py_DECREF(ret);
                return NULL;
            }
        }

        /* Copy the data for this array */
        if (PyArray_CopyAsFlat((PyArrayObject *)sliding_view, arrays[iarrays],
                            order) < 0) {
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* Slide to the start of the next window */
        sliding_view->data +=
            sliding_view->strides[0] * PyArray_SIZE(arrays[iarrays]);
    }

    Py_DECREF(sliding_view);
    return ret;
}

NPY_NO_EXPORT HPy // PyArrayObject *
HPyArray_ConcatenateFlattenedArrays(HPyContext *ctx, int narrays, 
                                   HPy /* PyArrayObject ** */ *arrays,
                                   NPY_ORDER order, HPy /* PyArrayObject * */ ret,
                                   HPy /* PyArray_Descr * */ dtype, NPY_CASTING casting,
                                   npy_bool casting_not_passed)
{
    int iarrays;
    npy_intp shape = 0;
    HPy sliding_view = HPy_NULL; // PyArrayObject_fields *

    if (narrays <= 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "need at least one array to concatenate");
        return HPy_NULL;
    }

    /*
     * Figure out the final concatenated shape starting from the first
     * array's shape.
     */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        PyArrayObject *iarrays_struct = PyArrayObject_AsStruct(ctx, arrays[iarrays]);
        shape += HPyArray_SIZE(iarrays_struct);
        /* Check for overflow */
        if (shape < 0) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "total number of elements "
                            "too large to concatenate");
            return HPy_NULL;
        }
    }

    int out_passed = 0;
    PyArrayObject *ret_struct = NULL;
    HPy ret_descr = HPy_NULL;
    if (!HPy_IsNull(ret)) {
        assert(HPy_IsNull(dtype));
        out_passed = 1;
        ret_struct = PyArrayObject_AsStruct(ctx, ret);
        if (PyArray_NDIM(ret_struct) != 1) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Output array must be 1D");
            return HPy_NULL;
        }
        if (shape != HPyArray_SIZE(ret_struct)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Output array is the wrong size");
            return HPy_NULL;
        }
        ret = HPy_Dup(ctx, ret);
        ret_descr = HPyArray_DESCR(ctx, ret, ret_struct);
    }
    else {
        npy_intp stride;

        /* Get the priority subtype for the array */
        HPy subtype = HPyArray_GetSubType(ctx, narrays, arrays); // PyTypeObject *

        HPy descr = HPyArray_FindConcatenationDescriptor(ctx,
                narrays, arrays, dtype); // PyArray_Descr *
        if (HPy_IsNull(descr)) {
            return HPy_NULL;
        }
        PyArray_Descr *descr_struct = PyArray_Descr_AsStruct(ctx, descr);
        stride = descr_struct->elsize;

        /* Allocate the array for the result. This steals the 'dtype' reference. */
        ret = HPyArray_NewFromDescr_int(ctx,
                subtype, descr,  1, &shape, &stride, NULL, 0, HPy_NULL,
                HPy_NULL, 0, 1);
        if (HPy_IsNull(ret)) {
            return HPy_NULL;
        }
        ret_struct = PyArrayObject_AsStruct(ctx, ret);
        ret_descr = HPyArray_DESCR(ctx, ret, ret_struct);
        assert(HPy_Is(ctx, ret_descr, descr));
    }

    /*
     * Create a view which slides through ret for assigning the
     * successive input arrays.
     */
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    sliding_view = HPyArray_View(ctx, ret, ret_struct, ret_descr, HPy_NULL, array_type);
    if (HPy_IsNull(sliding_view)) {
        HPy_Close(ctx, ret);
        return HPy_NULL;
    }

    int give_deprecation_warning = 1;  /* To give warning for just one input array. */
    PyArrayObject_fields *sliding_view_struct = (PyArrayObject_fields *)PyArrayObject_AsStruct(ctx, sliding_view);
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        PyArrayObject *iarrays_struct = PyArrayObject_AsStruct(ctx, arrays[iarrays]);
        /* Adjust the window dimensions for this array */
        sliding_view_struct->dimensions[0] = PyArray_SIZE(iarrays_struct);

        if (!HPyArray_CanCastArrayTo(ctx,
                arrays[iarrays], ret_descr, casting)) {
            /* This should be an error, but was previously allowed here. */
            if (casting_not_passed && out_passed) {
                /* NumPy 1.20, 2020-09-03 */
                if (give_deprecation_warning && HPY_DEPRECATE(ctx,
                        "concatenate() with `axis=None` will use same-kind "
                        "casting by default in the future. Please use "
                        "`casting='unsafe'` to retain the old behaviour. "
                        "In the future this will be a TypeError.") < 0) {
                    HPy_Close(ctx, sliding_view);
                    HPy_Close(ctx, ret);
                    return HPy_NULL;
                }
                give_deprecation_warning = 0;
            }
            else {
                HPy iarrays_descr = HPyArray_DESCR(ctx, arrays[iarrays], iarrays_struct);
                hpy_npy_set_invalid_cast_error(ctx,
                        iarrays_descr, ret_descr,
                        casting, PyArray_NDIM(iarrays_struct) == 0);
                HPy_Close(ctx, iarrays_descr);
                HPy_Close(ctx, sliding_view);
                HPy_Close(ctx, ret);
                return HPy_NULL;
            }
        }

        /* Copy the data for this array */
        if (HPyArray_CopyAsFlat(ctx, sliding_view, arrays[iarrays],
                            order) < 0) {
            HPy_Close(ctx, sliding_view);
            HPy_Close(ctx, ret);
            return HPy_NULL;
        }

        /* Slide to the start of the next window */
        sliding_view_struct->data +=
            sliding_view_struct->strides[0] * HPyArray_SIZE(iarrays_struct);
    }

    HPy_Close(ctx, sliding_view);
    return ret;
}

/**
 * Implementation for np.concatenate
 *
 * @param op Sequence of arrays to concatenate
 * @param axis Axis to concatenate along
 * @param ret output array to fill
 * @param dtype Forced output array dtype (cannot be combined with ret)
 * @param casting Casting mode used
 * @param casting_not_passed Deprecation helper
 */
NPY_NO_EXPORT PyObject *
PyArray_ConcatenateInto(PyObject *op,
        int axis, PyArrayObject *ret, PyArray_Descr *dtype,
        NPY_CASTING casting, npy_bool casting_not_passed)
{
    int iarrays, narrays;
    PyArrayObject **arrays;

    if (!PySequence_Check(op)) {
        PyErr_SetString(PyExc_TypeError,
                        "The first input argument needs to be a sequence");
        return NULL;
    }
    if (ret != NULL && dtype != NULL) {
        PyErr_SetString(PyExc_TypeError,
                "concatenate() only takes `out` or `dtype` as an "
                "argument, but both were provided.");
        return NULL;
    }

    /* Convert the input list into arrays */
    narrays = PySequence_Size(op);
    if (narrays < 0) {
        return NULL;
    }
    arrays = PyArray_malloc(narrays * sizeof(arrays[0]));
    if (arrays == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        PyObject *item = PySequence_GetItem(op, iarrays);
        if (item == NULL) {
            narrays = iarrays;
            goto fail;
        }
        arrays[iarrays] = (PyArrayObject *)PyArray_FROM_O(item);
        Py_DECREF(item);
        if (arrays[iarrays] == NULL) {
            narrays = iarrays;
            goto fail;
        }
    }

    if (axis >= NPY_MAXDIMS) {
        ret = PyArray_ConcatenateFlattenedArrays(
                narrays, arrays, NPY_CORDER, ret, dtype,
                casting, casting_not_passed);
    }
    else {
        ret = PyArray_ConcatenateArrays(
                narrays, arrays, axis, ret, dtype, casting);
    }

    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        Py_DECREF(arrays[iarrays]);
    }
    PyArray_free(arrays);

    return (PyObject *)ret;

fail:
    /* 'narrays' was set to how far we got in the conversion */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        Py_DECREF(arrays[iarrays]);
    }
    PyArray_free(arrays);

    return NULL;
}

/*NUMPY_API
 * Concatenate
 *
 * Concatenate an arbitrary Python sequence into an array.
 * op is a python object supporting the sequence interface.
 * Its elements will be concatenated together to form a single
 * multidimensional array. If axis is NPY_MAXDIMS or bigger, then
 * each sequence object will be flattened before concatenation
*/
NPY_NO_EXPORT PyObject *
PyArray_Concatenate(PyObject *op, int axis)
{
    /* retain legacy behaviour for casting */
    NPY_CASTING casting;
    if (axis >= NPY_MAXDIMS) {
        casting = NPY_UNSAFE_CASTING;
    }
    else {
        casting = NPY_SAME_KIND_CASTING;
    }
    return PyArray_ConcatenateInto(
            op, axis, NULL, NULL, casting, 0);
}


/**
 * Implementation for np.concatenate
 *
 * @param op Sequence of arrays to concatenate
 * @param axis Axis to concatenate along
 * @param ret output array to fill
 * @param dtype Forced output array dtype (cannot be combined with ret)
 * @param casting Casting mode used
 * @param casting_not_passed Deprecation helper
 */
NPY_NO_EXPORT HPy
HPyArray_ConcatenateInto(HPyContext *ctx, HPy op, int axis, 
        HPy /* PyArrayObject * */ ret, 
        HPy /* PyArray_Descr * */ dtype,
        NPY_CASTING casting, npy_bool casting_not_passed)
{
    int iarrays, narrays;
    HPy *arrays; // PyArrayObject *

    if (!HPySequence_Check(ctx, op)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                        "The first input argument needs to be a sequence");
        return HPy_NULL;
    }
    if (!HPy_IsNull(ret) && !HPy_IsNull(dtype)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "concatenate() only takes `out` or `dtype` as an "
                "argument, but both were provided.");
        return HPy_NULL;
    }

    /* Convert the input list into arrays */
    narrays = HPy_Length(ctx, op);
    if (narrays < 0) {
        return HPy_NULL;
    }
    arrays = PyArray_malloc(narrays * sizeof(arrays[0]));
    if (arrays == NULL) {
        HPyErr_NoMemory(ctx);
        return HPy_NULL;
    }
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        HPy item = HPy_GetItem_i(ctx, op, iarrays);
        if (HPy_IsNull(item)) {
            narrays = iarrays;
            goto fail;
        }
        arrays[iarrays] = HPyArray_FROM_O(ctx, item);
        HPy_Close(ctx, item);
        if (HPy_IsNull(arrays[iarrays])) {
            narrays = iarrays;
            goto fail;
        }
    }

    if (axis >= NPY_MAXDIMS) {
        ret = HPyArray_ConcatenateFlattenedArrays(ctx,
                narrays, arrays, NPY_CORDER, ret, dtype,
                casting, casting_not_passed);
    }
    else {
        ret = HPyArray_ConcatenateArrays(ctx,
                narrays, arrays, axis, ret, dtype, casting);
    }

    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        HPy_Close(ctx, arrays[iarrays]);
    }
    PyArray_free(arrays);

    return ret;

fail:
    /* 'narrays' was set to how far we got in the conversion */
    HPy_CloseAndFreeArray(ctx, arrays, narrays);
    // for (iarrays = 0; iarrays < narrays; ++iarrays) {
    //     Py_DECREF(arrays[iarrays]);
    // }
    // PyArray_free(arrays);

    return HPy_NULL;
}

/*HPY_NUMPY_API
 * Concatenate
 *
 * Concatenate an arbitrary Python sequence into an array.
 * op is a python object supporting the sequence interface.
 * Its elements will be concatenated together to form a single
 * multidimensional array. If axis is NPY_MAXDIMS or bigger, then
 * each sequence object will be flattened before concatenation
*/
NPY_NO_EXPORT HPy
HPyArray_Concatenate(HPyContext *ctx, HPy op, int axis)
{
    /* retain legacy behaviour for casting */
    NPY_CASTING casting;
    if (axis >= NPY_MAXDIMS) {
        casting = NPY_UNSAFE_CASTING;
    }
    else {
        casting = NPY_SAME_KIND_CASTING;
    }
    return HPyArray_ConcatenateInto(ctx,
            op, axis, HPy_NULL, HPy_NULL, casting, 0);
}

static int
_signbit_set(PyArrayObject *arr)
{
    static char bitmask = (char) 0x80;
    char *ptr;  /* points to the npy_byte to test */
    char byteorder;
    int elsize;

    elsize = PyArray_DESCR(arr)->elsize;
    byteorder = PyArray_DESCR(arr)->byteorder;
    ptr = PyArray_DATA(arr);
    if (elsize > 1 &&
        (byteorder == NPY_LITTLE ||
         (byteorder == NPY_NATIVE &&
          PyArray_ISNBO(NPY_LITTLE)))) {
        ptr += elsize - 1;
    }
    return ((*ptr & bitmask) != 0);
}


/*NUMPY_API
 * ScalarKind
 *
 * Returns the scalar kind of a type number, with an
 * optional tweak based on the scalar value itself.
 * If no scalar is provided, it returns INTPOS_SCALAR
 * for both signed and unsigned integers, otherwise
 * it checks the sign of any signed integer to choose
 * INTNEG_SCALAR when appropriate.
 */
NPY_NO_EXPORT NPY_SCALARKIND
PyArray_ScalarKind(int typenum, PyArrayObject **arr)
{
    NPY_SCALARKIND ret = NPY_NOSCALAR;

    if ((unsigned int)typenum < NPY_NTYPES) {
        ret = _npy_scalar_kinds_table[typenum];
        /* Signed integer types are INTNEG in the table */
        if (ret == NPY_INTNEG_SCALAR) {
            if (!arr || !_signbit_set(*arr)) {
                ret = NPY_INTPOS_SCALAR;
            }
        }
    } else if (PyTypeNum_ISUSERDEF(typenum)) {
        PyArray_Descr* descr = PyArray_DescrFromType(typenum);

        if (descr->f->scalarkind) {
            ret = descr->f->scalarkind((arr ? *arr : NULL));
        }
        Py_DECREF(descr);
    }

    return ret;
}

/*NUMPY_API
 *
 * Determines whether the data type 'thistype', with
 * scalar kind 'scalar', can be coerced into 'neededtype'.
 */
NPY_NO_EXPORT int
PyArray_CanCoerceScalar(int thistype, int neededtype,
                        NPY_SCALARKIND scalar)
{
    PyArray_Descr* from;
    int *castlist;

    /* If 'thistype' is not a scalar, it must be safely castable */
    if (scalar == NPY_NOSCALAR) {
        return PyArray_CanCastSafely(thistype, neededtype);
    }
    if ((unsigned int)neededtype < NPY_NTYPES) {
        NPY_SCALARKIND neededscalar;

        if (scalar == NPY_OBJECT_SCALAR) {
            return PyArray_CanCastSafely(thistype, neededtype);
        }

        /*
         * The lookup table gives us exactly what we need for
         * this comparison, which PyArray_ScalarKind would not.
         *
         * The rule is that positive scalars can be coerced
         * to a signed ints, but negative scalars cannot be coerced
         * to unsigned ints.
         *   _npy_scalar_kinds_table[int]==NEGINT > POSINT,
         *      so 1 is returned, but
         *   _npy_scalar_kinds_table[uint]==POSINT < NEGINT,
         *      so 0 is returned, as required.
         *
         */
        neededscalar = _npy_scalar_kinds_table[neededtype];
        if (neededscalar >= scalar) {
            return 1;
        }
        if (!PyTypeNum_ISUSERDEF(thistype)) {
            return 0;
        }
    }

    from = PyArray_DescrFromType(thistype);
    if (from->f->cancastscalarkindto
        && (castlist = from->f->cancastscalarkindto[scalar])) {
        while (*castlist != NPY_NOTYPE) {
            if (*castlist++ == neededtype) {
                Py_DECREF(from);
                return 1;
            }
        }
    }
    Py_DECREF(from);

    return 0;
}

/* Could perhaps be redone to not make contiguous arrays */

/*NUMPY_API
 * Numeric.innerproduct(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_InnerProduct(PyObject *op1, PyObject *op2)
{
    PyArrayObject *ap1 = NULL;
    PyArrayObject *ap2 = NULL;
    int typenum;
    PyArray_Descr *typec = NULL;
    PyObject* ap2t = NULL;
    npy_intp dims[NPY_MAXDIMS];
    PyArray_Dims newaxes = {dims, 0};
    int i;
    PyObject* ret = NULL;

    typenum = PyArray_ObjectType(op1, 0);
    if (typenum == NPY_NOTYPE && PyErr_Occurred()) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    if (typec == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");
        }
        goto fail;
    }

    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        goto fail;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        goto fail;
    }

    newaxes.len = PyArray_NDIM(ap2);
    if ((PyArray_NDIM(ap1) >= 1) && (newaxes.len >= 2)) {
        for (i = 0; i < newaxes.len - 2; i++) {
            dims[i] = (npy_intp)i;
        }
        dims[newaxes.len - 2] = newaxes.len - 1;
        dims[newaxes.len - 1] = newaxes.len - 2;

        ap2t = PyArray_Transpose(ap2, &newaxes);
        if (ap2t == NULL) {
            goto fail;
        }
    }
    else {
        ap2t = (PyObject *)ap2;
        Py_INCREF(ap2);
    }

    ret = PyArray_MatrixProduct2((PyObject *)ap1, ap2t, NULL);
    if (ret == NULL) {
        goto fail;
    }


    Py_DECREF(ap1);
    Py_DECREF(ap2);
    Py_DECREF(ap2t);
    return ret;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ap2t);
    Py_XDECREF(ret);
    return NULL;
}

/*HPY_NUMPY_API
 * Numeric.innerproduct(a,v)
 */
NPY_NO_EXPORT HPy
HPyArray_InnerProduct(HPyContext *ctx, HPy op1, HPy op2)
{
    HPy ap1 = HPy_NULL; // PyArrayObject *
    HPy ap2 = HPy_NULL; // PyArrayObject *
    int typenum;
    HPy typec = HPy_NULL; // PyArray_Descr *
    HPy ap2t = HPy_NULL;
    npy_intp dims[NPY_MAXDIMS];
    PyArray_Dims newaxes = {dims, 0};
    int i;
    HPy ret = HPy_NULL;

    typenum = HPyArray_ObjectType(ctx, op1, 0);
    if (typenum == NPY_NOTYPE && HPyErr_Occurred(ctx)) {
        return HPy_NULL;
    }
    typenum = HPyArray_ObjectType(ctx, op2, typenum);
    typec = HPyArray_DescrFromType(ctx, typenum);
    if (HPy_IsNull(typec)) {
        if (!HPyErr_Occurred(ctx)) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                            "Cannot find a common data type.");
        }
        goto fail;
    }

    // Py_INCREF(typec);
    ap1 = HPyArray_FromAny(ctx, op1, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, HPy_NULL);
    if (HPy_IsNull(ap1)) {
        // Py_DECREF(typec);
        goto fail;
    }
    ap2 = HPyArray_FromAny(ctx, op2, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, HPy_NULL);
    if (HPy_IsNull(ap2)) {
        goto fail;
    }


    PyArrayObject *ap1_struct = PyArrayObject_AsStruct(ctx, ap1);
    PyArrayObject *ap2_struct = PyArrayObject_AsStruct(ctx, ap2);
    newaxes.len = PyArray_NDIM(ap2_struct);
    if ((PyArray_NDIM(ap1_struct) >= 1) && (newaxes.len >= 2)) {
        for (i = 0; i < newaxes.len - 2; i++) {
            dims[i] = (npy_intp)i;
        }
        dims[newaxes.len - 2] = newaxes.len - 1;
        dims[newaxes.len - 1] = newaxes.len - 2;

        ap2t = HPyArray_Transpose(ctx, ap2, ap2_struct, &newaxes);
        if (HPy_IsNull(ap2t)) {
            goto fail;
        }
    }
    else {
        ap2t = HPy_Dup(ctx, ap2);
        // Py_INCREF(ap2);
    }

    ret = HPyArray_MatrixProduct2(ctx, ap1, ap2t, HPy_NULL, NULL);
    if (HPy_IsNull(ret)) {
        goto fail;
    }


    HPy_Close(ctx, ap1);
    HPy_Close(ctx, ap2);
    HPy_Close(ctx, ap2t);
    return ret;

fail:
    HPy_Close(ctx, ap1);
    HPy_Close(ctx, ap2);
    HPy_Close(ctx, ap2t);
    HPy_Close(ctx, ret);
    return HPy_NULL;
}

/*NUMPY_API
 * Numeric.matrixproduct(a,v)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct(PyObject *op1, PyObject *op2)
{
    return PyArray_MatrixProduct2(op1, op2, NULL);
}

/*NUMPY_API
 * Numeric.matrixproduct2(a,v,out)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct2(PyObject *op1, PyObject *op2, PyArrayObject* out)
{
    PyArrayObject *ap1, *ap2, *out_buf = NULL, *result = NULL;
    PyArrayIterObject *it1, *it2;
    npy_intp i, j, l;
    int typenum, nd, axis, matchDim;
    npy_intp is1, is2, os;
    char *op;
    npy_intp dimensions[NPY_MAXDIMS];
    PyArray_DotFunc *dot;
    PyArray_Descr *typec = NULL;
    NPY_BEGIN_THREADS_DEF;

    typenum = PyArray_ObjectType(op1, 0);
    if (typenum == NPY_NOTYPE && PyErr_Occurred()) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    if (typec == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");
        }
        return NULL;
    }

    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }

#if defined(HAVE_CBLAS)
    if (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2 &&
            (NPY_DOUBLE == typenum || NPY_CDOUBLE == typenum ||
             NPY_FLOAT == typenum || NPY_CFLOAT == typenum)) {
        return cblas_matrixproduct(typenum, ap1, ap2, out);
    }
#endif

    if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
        result = (PyArray_NDIM(ap1) == 0 ? ap1 : ap2);
        result = (PyArrayObject *)Py_TYPE(result)->tp_as_number->nb_multiply(
                                        (PyObject *)ap1, (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return (PyObject *)result;
    }
    l = PyArray_DIMS(ap1)[PyArray_NDIM(ap1) - 1];
    if (PyArray_NDIM(ap2) > 1) {
        matchDim = PyArray_NDIM(ap2) - 2;
    }
    else {
        matchDim = 0;
    }
    if (PyArray_DIMS(ap2)[matchDim] != l) {
        dot_alignment_error(ap1, PyArray_NDIM(ap1) - 1, ap2, matchDim);
        goto fail;
    }
    nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;
    if (nd > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "dot: too many dimensions in result");
        goto fail;
    }
    j = 0;
    for (i = 0; i < PyArray_NDIM(ap1) - 1; i++) {
        dimensions[j++] = PyArray_DIMS(ap1)[i];
    }
    for (i = 0; i < PyArray_NDIM(ap2) - 2; i++) {
        dimensions[j++] = PyArray_DIMS(ap2)[i];
    }
    if (PyArray_NDIM(ap2) > 1) {
        dimensions[j++] = PyArray_DIMS(ap2)[PyArray_NDIM(ap2)-1];
    }

    is1 = PyArray_STRIDES(ap1)[PyArray_NDIM(ap1)-1];
    is2 = PyArray_STRIDES(ap2)[matchDim];
    /* Choose which subtype to return */
    out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
    if (out_buf == NULL) {
        goto fail;
    }
    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (PyArray_SIZE(ap1) == 0 && PyArray_SIZE(ap2) == 0) {
        memset(PyArray_DATA(out_buf), 0, PyArray_NBYTES(out_buf));
    }

    dot = PyArray_DESCR(out_buf)->f->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "dot not available for this type");
        goto fail;
    }

    op = PyArray_DATA(out_buf);
    os = PyArray_DESCR(out_buf)->elsize;
    axis = PyArray_NDIM(ap1)-1;
    it1 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap1, &axis);
    if (it1 == NULL) {
        goto fail;
    }
    it2 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap2, &matchDim);
    if (it2 == NULL) {
        Py_DECREF(it1);
        goto fail;
    }
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
    while (it1->index < it1->size) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
            op += os;
            PyArray_ITER_NEXT(it2);
        }
        PyArray_ITER_NEXT(it1);
        PyArray_ITER_RESET(it2);
    }
    NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
    Py_DECREF(it1);
    Py_DECREF(it2);
    if (PyErr_Occurred()) {
        /* only for OBJECT arrays */
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* Trigger possible copy-back into `result` */
    PyArray_ResolveWritebackIfCopy(out_buf);
    Py_DECREF(out_buf);

    return (PyObject *)result;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(out_buf);
    Py_XDECREF(result);
    return NULL;
}

/*HPY_NUMPY_API
 * Numeric.matrixproduct(a,v)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT HPy
HPyArray_MatrixProduct(HPyContext *ctx, HPy op1, HPy op2)
{
    return HPyArray_MatrixProduct2(ctx, op1, op2, HPy_NULL, NULL);
}

/*HPY_NUMPY_API
 * Numeric.matrixproduct2(a,v,out)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT HPy
HPyArray_MatrixProduct2(HPyContext *ctx, HPy op1, HPy op2, HPy out, PyArrayObject* out_struct)
{
    HPy ap1, ap2, out_buf = HPy_NULL, result = HPy_NULL; // PyArrayObject *
    PyArrayIterObject *it1, *it2;
    npy_intp i, j, l;
    int typenum, nd, axis, matchDim;
    npy_intp is1, is2, os;
    char *op;
    npy_intp dimensions[NPY_MAXDIMS];
    PyArray_DotFunc *dot;
    HPy typec = HPy_NULL; // PyArray_Descr *
    HPY_NPY_BEGIN_THREADS_DEF;

    typenum = HPyArray_ObjectType(ctx, op1, 0);
    if (typenum == NPY_NOTYPE && HPyErr_Occurred(ctx)) {
        return HPy_NULL;
    }
    typenum = HPyArray_ObjectType(ctx, op2, typenum);
    typec = HPyArray_DescrFromType(ctx, typenum);
    if (HPy_IsNull(typec)) {
        if (!HPyErr_Occurred(ctx)) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                            "Cannot find a common data type.");
        }
        return HPy_NULL;
    }

    // Py_INCREF(typec);
    ap1 = HPyArray_FromAny(ctx, op1, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, HPy_NULL);
    if (HPy_IsNull(ap1)) {
        HPy_Close(ctx, typec);
        return HPy_NULL;
    }
    ap2 = HPyArray_FromAny(ctx, op2, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, HPy_NULL);
    if (HPy_IsNull(ap2)) {
        HPy_Close(ctx, ap1);
        return HPy_NULL;
    }
    PyArrayObject *ap1_struct = PyArrayObject_AsStruct(ctx, ap1);
    PyArrayObject *ap2_struct = PyArrayObject_AsStruct(ctx, ap2);

#if defined(HAVE_CBLAS)
    if (PyArray_NDIM(ap1_struct) <= 2 && PyArray_NDIM(ap2_struct) <= 2 &&
            (NPY_DOUBLE == typenum || NPY_CDOUBLE == typenum ||
             NPY_FLOAT == typenum || NPY_CFLOAT == typenum)) {
        result = hpy_cblas_matrixproduct(ctx, typenum, 
                                        ap1, ap1_struct,
                                        ap2, ap2_struct,
                                        out, out_struct);
        HPy_Close(ctx, ap1);
        HPy_Close(ctx, ap2);
        return result;
    }
#endif

    if (PyArray_NDIM(ap1_struct) == 0 || PyArray_NDIM(ap2_struct) == 0) {
        HPy r_type = HPy_Type(ctx, PyArray_NDIM(ap1_struct) == 0 ? ap1 : ap2);
        PyTypeObject *py_r_type = (PyTypeObject *)HPy_AsPyObject(ctx, r_type);
        PyObject *py_ap1 = HPy_AsPyObject(ctx, ap1);
        PyObject *py_ap2 = HPy_AsPyObject(ctx, ap2);
        CAPI_WARN("calling tp_as_number->nb_multiply");
        PyObject *py_result = py_r_type->tp_as_number->nb_multiply(
                                        py_ap1, py_ap2);
        result = HPy_FromPyObject(ctx, py_result);
        Py_DECREF(py_ap1);
        Py_DECREF(py_ap2);
        Py_DECREF(py_result);
        HPy_Close(ctx, ap1);
        HPy_Close(ctx, ap2);
        return result;
    }
    l = PyArray_DIMS(ap1_struct)[PyArray_NDIM(ap1_struct) - 1];
    if (PyArray_NDIM(ap2_struct) > 1) {
        matchDim = PyArray_NDIM(ap2_struct) - 2;
    }
    else {
        matchDim = 0;
    }
    if (PyArray_DIMS(ap2_struct)[matchDim] != l) {
        hpy_dot_alignment_error(ctx, ap1_struct, PyArray_NDIM(ap1_struct) - 1, ap2_struct, matchDim);
        goto fail;
    }
    nd = PyArray_NDIM(ap1_struct) + PyArray_NDIM(ap2_struct) - 2;
    if (nd > NPY_MAXDIMS) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "dot: too many dimensions in result");
        goto fail;
    }
    j = 0;
    for (i = 0; i < PyArray_NDIM(ap1_struct) - 1; i++) {
        dimensions[j++] = PyArray_DIMS(ap1_struct)[i];
    }
    for (i = 0; i < PyArray_NDIM(ap2_struct) - 2; i++) {
        dimensions[j++] = PyArray_DIMS(ap2_struct)[i];
    }
    if (PyArray_NDIM(ap2_struct) > 1) {
        dimensions[j++] = PyArray_DIMS(ap2_struct)[PyArray_NDIM(ap2_struct)-1];
    }

    is1 = PyArray_STRIDES(ap1_struct)[PyArray_NDIM(ap1_struct)-1];
    is2 = PyArray_STRIDES(ap2_struct)[matchDim];
    /* Choose which subtype to return */
    out_buf = hpy_new_array_for_sum(ctx, 
                                        ap1, ap1_struct, 
                                        ap2, ap2_struct, 
                                        out, out_struct, 
                                        nd, dimensions, typenum, &result);
    if (HPy_IsNull(out_buf)) {
        goto fail;
    }
    PyArrayObject *out_buf_struct = PyArrayObject_AsStruct(ctx, out_buf);
    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (PyArray_SIZE(ap1_struct) == 0 && PyArray_SIZE(ap2_struct) == 0) {
        memset(PyArray_DATA(out_buf_struct), 0, PyArray_NBYTES(out_buf_struct));
    }

    HPy out_buf_descr = HPyArray_DESCR(ctx, out_buf, out_buf_struct);
    PyArray_Descr *out_buf_descr_struct = PyArray_Descr_AsStruct(ctx, out_buf_descr);
    dot = out_buf_descr_struct->f->dotfunc;
    if (dot == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "dot not available for this type");
        HPy_Close(ctx, out_buf_descr);
        goto fail;
    }

    op = PyArray_DATA(out_buf_struct);
    os = out_buf_descr_struct->elsize;
    HPy_Close(ctx, out_buf_descr);
    axis = PyArray_NDIM(ap1_struct)-1;
    PyObject *py_ap1 = HPy_AsPyObject(ctx, ap1);
    CAPI_WARN("calling PyArray_IterAllButAxis");
    it1 = (PyArrayIterObject *)
        PyArray_IterAllButAxis(py_ap1, &axis);
    Py_DECREF(py_ap1);
    if (it1 == NULL) {
        goto fail;
    }
    PyObject *py_ap2 = HPy_AsPyObject(ctx, ap2);
    it2 = (PyArrayIterObject *)
        PyArray_IterAllButAxis(py_ap2, &matchDim);
    Py_DECREF(py_ap2);
    if (it2 == NULL) {
        Py_DECREF(it1);
        goto fail;
    }
    HPy ap2_descr = HPyArray_DESCR(ctx, ap2, ap2_struct);
    PyArray_Descr *ap2_descr_struct = PyArray_Descr_AsStruct(ctx, ap2_descr);

    HPY_NPY_BEGIN_THREADS_DESCR(ctx, ap2_descr_struct);
    while (it1->index < it1->size) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
            op += os;
            PyArray_ITER_NEXT(it2);
        }
        PyArray_ITER_NEXT(it1);
        PyArray_ITER_RESET(it2);
    }
    HPY_NPY_END_THREADS_DESCR(ctx, ap2_descr_struct);
    HPy_Close(ctx, ap2_descr);
    Py_DECREF(it1);
    Py_DECREF(it2);
    if (HPyErr_Occurred(ctx)) {
        /* only for OBJECT arrays */
        goto fail;
    }
    HPy_Close(ctx, ap1);
    HPy_Close(ctx, ap2);

    /* Trigger possible copy-back into `result` */
    HPyArray_ResolveWritebackIfCopy(ctx, out_buf);
    HPy_Close(ctx, out_buf);

    return result;

fail:
    HPy_Close(ctx, ap1);
    HPy_Close(ctx, ap2);
    HPy_Close(ctx, out_buf);
    HPy_Close(ctx, result);
    return HPy_NULL;
}

/*NUMPY_API
 * Copy and Transpose
 *
 * Could deprecate this function, as there isn't a speed benefit over
 * calling Transpose and then Copy.
 */
NPY_NO_EXPORT PyObject *
PyArray_CopyAndTranspose(PyObject *op)
{
    PyArrayObject *arr, *tmp, *ret;
    int i;
    npy_intp new_axes_values[NPY_MAXDIMS];
    PyArray_Dims new_axes;

    /* Make sure we have an array */
    arr = (PyArrayObject *)PyArray_FROM_O(op);
    if (arr == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(arr) > 1) {
        /* Set up the transpose operation */
        new_axes.len = PyArray_NDIM(arr);
        for (i = 0; i < new_axes.len; ++i) {
            new_axes_values[i] = new_axes.len - i - 1;
        }
        new_axes.ptr = new_axes_values;

        /* Do the transpose (always returns a view) */
        tmp = (PyArrayObject *)PyArray_Transpose(arr, &new_axes);
        if (tmp == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
    }
    else {
        tmp = arr;
        arr = NULL;
    }

    /* TODO: Change this to NPY_KEEPORDER for NumPy 2.0 */
    ret = (PyArrayObject *)PyArray_NewCopy(tmp, NPY_CORDER);

    Py_XDECREF(arr);
    Py_DECREF(tmp);
    return (PyObject *)ret;
}

/*HPY_NUMPY_API
 * Copy and Transpose
 *
 * Could deprecate this function, as there isn't a speed benefit over
 * calling Transpose and then Copy.
 */
NPY_NO_EXPORT HPy
HPyArray_CopyAndTranspose(HPyContext *ctx, HPy op)
{
    HPy arr, tmp, ret; // PyArrayObject *
    int i;
    npy_intp new_axes_values[NPY_MAXDIMS];
    PyArray_Dims new_axes;

    /* Make sure we have an array */
    arr = HPyArray_FROM_O(ctx, op);
    if (HPy_IsNull(arr)) {
        return HPy_NULL;
    }
    PyArrayObject *arr_struct = PyArrayObject_AsStruct(ctx, arr);
    if (PyArray_NDIM(arr_struct) > 1) {
        /* Set up the transpose operation */
        new_axes.len = PyArray_NDIM(arr_struct);
        for (i = 0; i < new_axes.len; ++i) {
            new_axes_values[i] = new_axes.len - i - 1;
        }
        new_axes.ptr = new_axes_values;

        /* Do the transpose (always returns a view) */
        tmp = HPyArray_Transpose(ctx, arr, arr_struct, &new_axes);
        if (HPy_IsNull(tmp)) {
            HPy_Close(ctx, arr);
            return HPy_NULL;
        }
    }
    else {
        tmp = arr;
        arr = HPy_NULL;
    }

    /* TODO: Change this to NPY_KEEPORDER for NumPy 2.0 */
    ret = HPyArray_NewCopy(ctx, tmp, NPY_CORDER);

    HPy_Close(ctx, arr);
    HPy_Close(ctx, tmp);
    return ret;
}

/*
 * Implementation which is common between PyArray_Correlate
 * and PyArray_Correlate2.
 *
 * inverted is set to 1 if computed correlate(ap2, ap1), 0 otherwise
 */
static PyArrayObject*
_pyarray_correlate(PyArrayObject *ap1, PyArrayObject *ap2, int typenum,
                   int mode, int *inverted)
{
    PyArrayObject *ret;
    npy_intp length;
    npy_intp i, n1, n2, n, n_left, n_right;
    npy_intp is1, is2, os;
    char *ip1, *ip2, *op;
    PyArray_DotFunc *dot;

    NPY_BEGIN_THREADS_DEF;

    n1 = PyArray_DIMS(ap1)[0];
    n2 = PyArray_DIMS(ap2)[0];
    if (n1 == 0) {
        PyErr_SetString(PyExc_ValueError, "first array argument cannot be empty");
        return NULL;
    }
    if (n2 == 0) {
        PyErr_SetString(PyExc_ValueError, "second array argument cannot be empty");
        return NULL;
    }
    if (n1 < n2) {
        ret = ap1;
        ap1 = ap2;
        ap2 = ret;
        ret = NULL;
        i = n1;
        n1 = n2;
        n2 = i;
        *inverted = 1;
    } else {
        *inverted = 0;
    }

    length = n1;
    n = n2;
    switch(mode) {
    case 0:
        length = length - n + 1;
        n_left = n_right = 0;
        break;
    case 1:
        n_left = (npy_intp)(n/2);
        n_right = n - n_left - 1;
        break;
    case 2:
        n_right = n - 1;
        n_left = n - 1;
        length = length + n - 1;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "mode must be 0, 1, or 2");
        return NULL;
    }

    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    ret = new_array_for_sum(ap1, ap2, NULL, 1, &length, typenum, NULL);
    if (ret == NULL) {
        return NULL;
    }
    dot = PyArray_DESCR(ret)->f->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
        goto clean_ret;
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ret));
    is1 = PyArray_STRIDES(ap1)[0];
    is2 = PyArray_STRIDES(ap2)[0];
    op = PyArray_DATA(ret);
    os = PyArray_DESCR(ret)->elsize;
    ip1 = PyArray_DATA(ap1);
    ip2 = PyArray_BYTES(ap2) + n_left*is2;
    n = n - n_left;
    for (i = 0; i < n_left; i++) {
        dot(ip1, is1, ip2, is2, op, n, ret);
        n++;
        ip2 -= is2;
        op += os;
    }
    if (small_correlate(ip1, is1, n1 - n2 + 1, PyArray_TYPE(ap1),
                        ip2, is2, n, PyArray_TYPE(ap2),
                        op, os)) {
        ip1 += is1 * (n1 - n2 + 1);
        op += os * (n1 - n2 + 1);
    }
    else {
        for (i = 0; i < (n1 - n2 + 1); i++) {
            dot(ip1, is1, ip2, is2, op, n, ret);
            ip1 += is1;
            op += os;
        }
    }
    for (i = 0; i < n_right; i++) {
        n--;
        dot(ip1, is1, ip2, is2, op, n, ret);
        ip1 += is1;
        op += os;
    }

    NPY_END_THREADS_DESCR(PyArray_DESCR(ret));
    if (PyErr_Occurred()) {
        goto clean_ret;
    }

    return ret;

clean_ret:
    Py_DECREF(ret);
    return NULL;
}

/*
 * Revert a one dimensional array in-place
 *
 * Return 0 on success, other value on failure
 */
static int
_pyarray_revert(PyArrayObject *ret)
{
    npy_intp length = PyArray_DIM(ret, 0);
    npy_intp os = PyArray_DESCR(ret)->elsize;
    char *op = PyArray_DATA(ret);
    char *sw1 = op;
    char *sw2;

    if (PyArray_ISNUMBER(ret) && !PyArray_ISCOMPLEX(ret)) {
        /* Optimization for unstructured dtypes */
        PyArray_CopySwapNFunc *copyswapn = PyArray_DESCR(ret)->f->copyswapn;
        sw2 = op + length * os - 1;
        /* First reverse the whole array byte by byte... */
        while(sw1 < sw2) {
            const char tmp = *sw1;
            *sw1++ = *sw2;
            *sw2-- = tmp;
        }
        /* ...then swap in place every item */
        copyswapn(op, os, NULL, 0, length, 1, NULL);
    }
    else {
        char *tmp = PyArray_malloc(PyArray_DESCR(ret)->elsize);
        if (tmp == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        sw2 = op + (length - 1) * os;
        while (sw1 < sw2) {
            memcpy(tmp, sw1, os);
            memcpy(sw1, sw2, os);
            memcpy(sw2, tmp, os);
            sw1 += os;
            sw2 -= os;
        }
        PyArray_free(tmp);
    }

    return 0;
}

/*NUMPY_API
 * correlate(a1,a2,mode)
 *
 * This function computes the usual correlation (correlate(a1, a2) !=
 * correlate(a2, a1), and conjugate the second argument for complex inputs
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate2(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    PyArray_Descr *typec;
    int inverted;
    int st;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
                                        NPY_ARRAY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
                                        NPY_ARRAY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto clean_ap1;
    }

    if (PyArray_ISCOMPLEX(ap2)) {
        PyArrayObject *cap2;
        cap2 = (PyArrayObject *)PyArray_Conjugate(ap2, NULL);
        if (cap2 == NULL) {
            goto clean_ap2;
        }
        Py_DECREF(ap2);
        ap2 = cap2;
    }

    ret = _pyarray_correlate(ap1, ap2, typenum, mode, &inverted);
    if (ret == NULL) {
        goto clean_ap2;
    }

    /*
     * If we inverted input orders, we need to reverse the output array (i.e.
     * ret = ret[::-1])
     */
    if (inverted) {
        st = _pyarray_revert(ret);
        if (st) {
            goto clean_ret;
        }
    }

    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

clean_ret:
    Py_DECREF(ret);
clean_ap2:
    Py_DECREF(ap2);
clean_ap1:
    Py_DECREF(ap1);
    return NULL;
}

/*NUMPY_API
 * Numeric.correlate(a1,a2,mode)
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    int unused;
    PyArray_Descr *typec;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
                                            NPY_ARRAY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
                                           NPY_ARRAY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail;
    }

    ret = _pyarray_correlate(ap1, ap2, typenum, mode, &unused);
    if (ret == NULL) {
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}


HPyDef_METH(array_putmask, "putmask", HPyFunc_KEYWORDS)
static HPy
array_putmask_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy mask, values;
    HPy array;

    static const char *kwlist[] = {"arr", "mask", "values", NULL};

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "OOO:putmask", kwlist,
                &array, &mask, &values)) {
        return HPy_NULL;
    }
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    if (!HPy_TypeCheck(ctx, array, array_type)) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "putmask");
        HPyTracker_Close(ctx, ht);
        HPy_Close(ctx, array_type);
        return HPy_NULL;
    }
    HPy_Close(ctx, array_type);
    HPy ret = HPyArray_PutMask(ctx, array, values, mask);
    HPyTracker_Close(ctx, ht);
    return ret;
}


/*NUMPY_API
 *
 * This function returns true if the two typecodes are
 * equivalent (same basic kind and same itemsize).
 */
NPY_NO_EXPORT unsigned char
PyArray_EquivTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    if (type1 == type2) {
        return 1;
    }

    if (Py_TYPE(Py_TYPE(type1)) == &PyType_Type) {
        /*
         * 2021-12-17: This case is nonsense and should be removed eventually!
         *
         * boost::python has/had a bug effectively using EquivTypes with
         * `type(arbitrary_obj)`.  That is clearly wrong as that cannot be a
         * `PyArray_Descr *`.  We assume that `type(type(type(arbitrary_obj))`
         * is always in practice `type` (this is the type of the metaclass),
         * but for our descriptors, `type(type(descr))` is DTypeMeta.
         *
         * In that case, we just return False.  There is a possibility that
         * this actually _worked_ effectively (returning 1 sometimes).
         * We ignore that possibility for simplicity; it really is not our bug.
         */
        return 0;
    }

    /*
     * Do not use PyArray_CanCastTypeTo because it supports legacy flexible
     * dtypes as input.
     */
    npy_intp view_offset;
    NPY_CASTING safety = PyArray_GetCastInfo(type1, type2, NULL, &view_offset);
    if (safety < 0) {
        PyErr_Clear();
        return 0;
    }
    /* If casting is "no casting" this dtypes are considered equivalent. */
    return PyArray_MinCastSafety(safety, NPY_NO_CASTING) == NPY_NO_CASTING;
}

/*HPY_NUMPY_API
 *
 * This function returns true if the two typecodes are
 * equivalent (same basic kind and same itemsize).
 */
NPY_NO_EXPORT unsigned char
HPyArray_EquivTypes(HPyContext *ctx, /*PyArray_Descr*/HPy type1, /*PyArray_Descr*/HPy type2)
{
    if (HPy_Is(ctx, type1, type2)) {
        return 1;
    }

    HPy type1_type = HPy_Type(ctx, type1);
    HPy type1_type_type = HPy_Type(ctx, type1_type);
    if (HPy_Is(ctx, type1_type_type, ctx->h_TypeType)) {
        /*
         * 2021-12-17: This case is nonsense and should be removed eventually!
         *
         * boost::python has/had a bug effectively using EquivTypes with
         * `type(arbitrary_obj)`.  That is clearly wrong as that cannot be a
         * `PyArray_Descr *`.  We assume that `type(type(type(arbitrary_obj))`
         * is always in practice `type` (this is the type of the metaclass),
         * but for our descriptors, `type(type(descr))` is DTypeMeta.
         *
         * In that case, we just return False.  There is a possibility that
         * this actually _worked_ effectively (returning 1 sometimes).
         * We ignore that possibility for simplicity; it really is not our bug.
         */
        return 0;
    }

    /*
     * Do not use PyArray_CanCastTypeTo because it supports legacy flexible
     * dtypes as input.
     */
    npy_intp view_offset;
    NPY_CASTING safety = HPyArray_GetCastInfo(ctx, type1, type2, HPy_NULL, &view_offset);
    if (safety < 0) {
        HPyErr_Clear(ctx);
        return 0;
    }
    /* If casting is "no casting" this dtypes are considered equivalent. */
    return PyArray_MinCastSafety(safety, NPY_NO_CASTING) == NPY_NO_CASTING;
}


/*NUMPY_API*/
NPY_NO_EXPORT unsigned char
PyArray_EquivTypenums(int typenum1, int typenum2)
{
    PyArray_Descr *d1, *d2;
    npy_bool ret;

    if (typenum1 == typenum2) {
        return NPY_SUCCEED;
    }

    d1 = PyArray_DescrFromType(typenum1);
    d2 = PyArray_DescrFromType(typenum2);
    ret = PyArray_EquivTypes(d1, d2);
    Py_DECREF(d1);
    Py_DECREF(d2);
    return ret;
}

/*** END C-API FUNCTIONS **/
/*
 * NPY_RELAXED_STRIDES_CHECKING: If the strides logic is changed, the
 * order specific stride setting is not necessary.
 */
static NPY_STEALS_REF_TO_ARG(1) PyObject *
_prepend_ones(PyArrayObject *arr, int nd, int ndmin, NPY_ORDER order)
{
    npy_intp newdims[NPY_MAXDIMS];
    npy_intp newstrides[NPY_MAXDIMS];
    npy_intp newstride;
    int i, k, num;
    PyObject *ret;
    PyArray_Descr *dtype;

    if (order == NPY_FORTRANORDER || PyArray_ISFORTRAN(arr) || PyArray_NDIM(arr) == 0) {
        newstride = PyArray_DESCR(arr)->elsize;
    }
    else {
        newstride = PyArray_STRIDES(arr)[0] * PyArray_DIMS(arr)[0];
    }

    num = ndmin - nd;
    for (i = 0; i < num; i++) {
        newdims[i] = 1;
        newstrides[i] = newstride;
    }
    for (i = num; i < ndmin; i++) {
        k = i - num;
        newdims[i] = PyArray_DIMS(arr)[k];
        newstrides[i] = PyArray_STRIDES(arr)[k];
    }
    dtype = PyArray_DESCR(arr);
    Py_INCREF(dtype);
    ret = PyArray_NewFromDescrAndBase(
            Py_TYPE(arr), dtype,
            ndmin, newdims, newstrides, PyArray_DATA(arr),
            PyArray_FLAGS(arr), (PyObject *)arr, (PyObject *)arr);
    Py_DECREF(arr);

    return ret;
}

static HPy
_hpy_prepend_ones(HPyContext *ctx, 
                        HPy arr,
                        PyArrayObject *arr_data, 
                        int nd, int ndmin, NPY_ORDER order)
{
    npy_intp newdims[NPY_MAXDIMS];
    npy_intp newstrides[NPY_MAXDIMS];
    npy_intp newstride;
    int i, k, num;
    HPy ret;
    HPy dtype;

    HPy arr_descr = HPyArray_DESCR(ctx, arr, arr_data);
    PyArray_Descr *arr_descr_data = PyArray_Descr_AsStruct(ctx, arr_descr);

    if (order == NPY_FORTRANORDER || PyArray_ISFORTRAN(arr_data) || PyArray_NDIM(arr_data) == 0) {
        newstride = arr_descr_data->elsize;
    }
    else {
        newstride = PyArray_STRIDES(arr_data)[0] * PyArray_DIMS(arr_data)[0];
    }

    num = ndmin - nd;
    for (i = 0; i < num; i++) {
        newdims[i] = 1;
        newstrides[i] = newstride;
    }
    for (i = num; i < ndmin; i++) {
        k = i - num;
        newdims[i] = PyArray_DIMS(arr_data)[k];
        newstrides[i] = PyArray_STRIDES(arr_data)[k];
    }
    dtype = arr_descr;
    HPy arr_type = HPy_Type(ctx, arr);
    // Py_INCREF(dtype);
    ret = HPyArray_NewFromDescrAndBase(ctx,
            arr_type, dtype,
            ndmin, newdims, newstrides, PyArray_DATA(arr_data),
            PyArray_FLAGS(arr_data), arr, arr);
    HPy_Close(ctx, arr_type);
    // Py_DECREF(arr);

    return ret;
}

#define STRIDING_OK(op, order) \
                ((order) == NPY_ANYORDER || \
                 (order) == NPY_KEEPORDER || \
                 ((order) == NPY_CORDER && PyArray_IS_C_CONTIGUOUS(op)) || \
                 ((order) == NPY_FORTRANORDER && PyArray_IS_F_CONTIGUOUS(op)))

static NPY_INLINE PyObject *
_array_fromobject_generic(
        PyObject *op, PyArray_Descr *type, _PyArray_CopyMode copy, NPY_ORDER order,
        npy_bool subok, int ndmin)
{
    PyArrayObject *oparr = NULL, *ret = NULL;
    PyArray_Descr *oldtype = NULL;
    int nd, flags = 0;

    if (ndmin > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                "ndmin bigger than allowable number of dimensions "
                "NPY_MAXDIMS (=%d)", NPY_MAXDIMS);
        return NULL;
    }
    /* fast exit if simple call */
    if (PyArray_CheckExact(op) || (subok && PyArray_Check(op))) {
        oparr = (PyArrayObject *)op;
        if (type == NULL) {
            if (copy != NPY_COPY_ALWAYS && STRIDING_OK(oparr, order)) {
                ret = oparr;
                Py_INCREF(ret);
                goto finish;
            }
            else {
                if (copy == NPY_COPY_NEVER) {
                    PyErr_SetString(PyExc_ValueError,
                            "Unable to avoid copy while creating a new array.");
                    return NULL;
                }
                ret = (PyArrayObject *)PyArray_NewCopy(oparr, order);
                goto finish;
            }
        }
        /* One more chance */
        oldtype = PyArray_DESCR(oparr);
        if (PyArray_EquivTypes(oldtype, type)) {
            if (copy != NPY_COPY_ALWAYS && STRIDING_OK(oparr, order)) {
                Py_INCREF(op);
                ret = oparr;
                goto finish;
            }
            else {
                if (copy == NPY_COPY_NEVER) {
                    PyErr_SetString(PyExc_ValueError,
                            "Unable to avoid copy while creating a new array.");
                    return NULL;
                }
                ret = (PyArrayObject *)PyArray_NewCopy(oparr, order);
                if (oldtype == type || ret == NULL) {
                    goto finish;
                }
                _set_descr(ret, oldtype);
                goto finish;
            }
        }
    }

    if (copy == NPY_COPY_ALWAYS) {
        flags = NPY_ARRAY_ENSURECOPY;
    }
    else if (copy == NPY_COPY_NEVER ) {
        flags = NPY_ARRAY_ENSURENOCOPY;
    }
    if (order == NPY_CORDER) {
        flags |= NPY_ARRAY_C_CONTIGUOUS;
    }
    else if ((order == NPY_FORTRANORDER)
                 /* order == NPY_ANYORDER && */
                 || (PyArray_Check(op) &&
                     PyArray_ISFORTRAN((PyArrayObject *)op))) {
        flags |= NPY_ARRAY_F_CONTIGUOUS;
    }
    if (!subok) {
        flags |= NPY_ARRAY_ENSUREARRAY;
    }

    flags |= NPY_ARRAY_FORCECAST;
    Py_XINCREF(type);
    ret = (PyArrayObject *)PyArray_CheckFromAny(op, type,
                                                0, 0, flags, NULL);

finish:
    if (ret == NULL) {
        return NULL;
    }

    nd = PyArray_NDIM(ret);
    if (nd >= ndmin) {
        return (PyObject *)ret;
    }
    /*
     * create a new array from the same data with ones in the shape
     * steals a reference to ret
     */
    return _prepend_ones(ret, nd, ndmin, order);
}

#include "convert.h"

static NPY_INLINE HPy
_hpy_array_fromobject_generic(
        HPyContext *ctx, HPy op, HPy type, _PyArray_CopyMode copy, NPY_ORDER order,
        npy_bool subok, int ndmin)
{
    HPy array_type = HPy_NULL;
    HPy ret = HPy_NULL;
    HPy oldtype = HPy_NULL;
    PyArray_Descr *oldtype_data;
    PyArray_Descr *type_data = PyArray_Descr_AsStruct(ctx, type);
    int nd, flags = 0;

    if (ndmin > NPY_MAXDIMS) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "ndmin bigger than allowable number of dimensions "
                "NPY_MAXDIMS (=%d)"/*, HPY TODO: NPY_MAXDIMS*/);
        goto fail;
    }
    /* fast exit if simple call */
    // TODO HPY LABS PORT: original code uses "check exact"
    // HPy version has to use two calls HPy_Type and HPy_Is
    // It would be faster to check subok first and then exact or subclass check
    array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    HPy op_type = HPy_Type(ctx, op);
    int is_op_HPyArray_Type = HPy_Is(ctx, op_type, array_type);
    HPy_Close(ctx, op_type);
    if (is_op_HPyArray_Type ||
        (subok && HPy_TypeCheck(ctx, op, array_type))) {
        PyArrayObject *oparr = PyArrayObject_AsStruct(ctx, op);
        if (HPy_IsNull(type)) {
            if (copy != NPY_COPY_ALWAYS && STRIDING_OK(oparr, order)) {
                ret = HPy_Dup(ctx, op);
                goto finish;
            }
            else {
                if (copy == NPY_COPY_NEVER) {
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Unable to avoid copy while creating a new array.");
                    goto fail;
                }                
                ret = HPyArray_NewCopy(ctx, op, order);
                goto finish;
            }
        }
        /* One more chance */
        oldtype = HPyArray_DESCR(ctx, op, oparr);
        oldtype_data = PyArray_Descr_AsStruct(ctx, oldtype);
        if (HPyArray_EquivTypes(ctx, oldtype, type)) {
            if (copy != NPY_COPY_ALWAYS && STRIDING_OK(oparr, order)) {
                ret = HPy_Dup(ctx, op);
                goto finish;
            }
            else {
                if (copy == NPY_COPY_NEVER) {
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Unable to avoid copy while creating a new array.");
                    goto fail;
                }
                ret = HPyArray_NewCopy(ctx, op, order);
                if (oldtype_data == type_data || HPy_IsNull(ret)) {
                    goto finish;
                }
                _hpy_set_descr(ctx, ret, oparr, oldtype);
                goto finish;
            }
        }
    }

    if (copy == NPY_COPY_ALWAYS) {
        flags = NPY_ARRAY_ENSURECOPY;
    }
    else if (copy == NPY_COPY_NEVER ) {
        flags = NPY_ARRAY_ENSURENOCOPY;
    }
    if (order == NPY_CORDER) {
        flags |= NPY_ARRAY_C_CONTIGUOUS;
    }
    else if ((order == NPY_FORTRANORDER)
                 /* order == NPY_ANYORDER && */
                 || (HPy_TypeCheck(ctx, op, array_type) &&
                     PyArray_ISFORTRAN(PyArrayObject_AsStruct(ctx, op)))) {
        flags |= NPY_ARRAY_F_CONTIGUOUS;
    }
    if (!subok) {
        flags |= NPY_ARRAY_ENSUREARRAY;
    }

    flags |= NPY_ARRAY_FORCECAST;
    /*
     * We don't need to dup 'type' here because in contrast to
     * PyArray_CheckFromAny, function HPyArray_CheckFromAny is not stealing the
     * reference.
     */
    ret = HPyArray_CheckFromAny(ctx, op, type,
        0, 0, flags, HPy_NULL);

finish:
    HPy_Close(ctx, array_type);
    HPy_Close(ctx, oldtype);

    if (HPy_IsNull(ret)) {
        return ret;
    }

    nd = HPyArray_GetNDim(ctx, ret);
    if (nd >= ndmin) {
        return ret;
    }

    /*
     * create a new array from the same data with ones in the shape
     * Does not steal a reference to ret
     */
    return _hpy_prepend_ones(ctx, ret, PyArrayObject_AsStruct(ctx, ret), nd, ndmin, order);

fail:
    HPy_Close(ctx, array_type);
    HPy_Close(ctx, oldtype);
    return HPy_NULL;
}

#undef STRIDING_OK


#include <stdio.h>
#include <stdlib.h>

HPyDef_METH(array_array, "array", HPyFunc_KEYWORDS)
static HPy
array_array_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kw)
{
    static const char *kwlist[] = {"object", "dtype", "copy", "order", "subok", "ndmin", "like", NULL};

    HPy op = HPy_NULL;
    HPy h_type = HPy_NULL;
    npy_bool subok = NPY_FALSE;
    _PyArray_CopyMode copy = NPY_COPY_ALWAYS;
    int ndmin = 0;
    NPY_ORDER order = NPY_KEEPORDER;
    HPyTracker tracker;

    if (nargs != 1 || !HPy_IsNull(kw)) {
        HPy h_type_in = HPy_NULL, h_copy = HPy_NULL, h_order = HPy_NULL;
        HPy h_subok = HPy_NULL, h_ndmin = HPy_NULL, h_like = HPy_NULL;
        if (!HPyArg_ParseKeywords(ctx, &tracker, args, nargs, kw, "O|OOOOOO", kwlist,
            &op, &h_type_in, &h_copy, &h_order, &h_subok, &h_ndmin, &h_like)) {
            HPyTracker_Close(ctx, tracker);
            return HPy_NULL;
        }

        if (HPyArray_DescrConverter2(ctx, h_type_in, &h_type) == NPY_FAIL) {
            HPyTracker_Close(ctx, tracker);
            return HPy_NULL;
        }
        if (!HPy_IsNull(h_copy) && HPyArray_CopyConverter(ctx, h_copy, &copy) == NPY_FAIL) {
            HPyTracker_Close(ctx, tracker);
            return HPy_NULL;
        }
        if (!HPy_IsNull(h_order) && HPyArray_OrderConverter(ctx, h_order, &order) == NPY_FAIL) {
            HPyTracker_Close(ctx, tracker);
            return HPy_NULL;
        }
        if (!HPy_IsNull(h_subok) && HPyArray_BoolConverter(ctx, h_subok, &subok) == NPY_FAIL) {
            HPyTracker_Close(ctx, tracker);
            return HPy_NULL;
        }
        if (!HPy_IsNull(h_ndmin) && HPyArray_PythonPyIntFromInt(ctx, h_ndmin, &ndmin) == NPY_FAIL) {
            HPyTracker_Close(ctx, tracker);
            return HPy_NULL;
        }
        if (!HPy_IsNull(h_like)) {
            HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                    "array", h_like, kw, args, nargs);
            if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
                HPy_Close(ctx, h_type);
                return deferred;
            }
        }
    }
    else {
        /* Fast path for symmetry (we copy by default which is slow) */
        op = args[0];
    }

    HPy res = _hpy_array_fromobject_generic(
           ctx, op, h_type, copy, order, subok, ndmin);

    HPy_Close(ctx, h_type);
    return res;
}

HPyDef_METH(array_asarray, "asarray", HPyFunc_KEYWORDS)
static HPy
array_asarray_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t len_args, HPy kw)
{
    static const char *kwlist[] = {"a", "dtype", "order", "like", NULL};

    HPy op;
    HPy type = HPy_NULL; // PyArray_Descr *
    NPY_ORDER order = NPY_KEEPORDER;
    HPy like = HPy_NULL;
    // NPY_PREPARE_ARGPARSER;


    if (len_args != 1 || !HPy_IsNull(kw)) {
        HPyTracker ht;
        HPy h_type = HPy_NULL, h_order = HPy_NULL;
        // HPY TODO: uses npy_parse_arguments METH_FASTCALL|METH_KEYWORDS
        // 'like' is expected to be passed as keyword only.. we are ignoring this for now
        if (!HPyArg_ParseKeywords(ctx, &ht, args, len_args, kw, "O|OOO", kwlist,
                &op, &h_type, &h_order, &like)) {
            return HPy_NULL;
        }
        // if (npy_parse_arguments("asarray", args, len_args, kwnames,
        //         "a", NULL, &op,
        //         "|dtype", &PyArray_DescrConverter2, &h_type,
        //         "|order", &PyArray_OrderConverter, &h_order,
        //         "$like", NULL, &like,
        //         NULL, NULL, NULL) < 0) {
        //     HPy_Close(ctx, type);
        //     return HPy_NULL;
        // }
        if (HPyArray_DescrConverter2(ctx, h_type, &type) != NPY_SUCCEED ||
                HPyArray_OrderConverter(ctx, h_order, &order) != NPY_SUCCEED) {
            HPyErr_SetString(ctx, ctx->h_SystemError, "asarray: TODO");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        if (!HPy_IsNull(like)) {
            HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                    "asarray", like, kw, args, len_args);
            if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
                HPy_Close(ctx, type);
                HPyTracker_Close(ctx, ht);
                return deferred;
            }
            // HPy_Close(ctx, deferred); ?
        }
        HPyTracker_Close(ctx, ht);
    }
    else {
        op = args[0];
    }

    HPy res = _hpy_array_fromobject_generic(ctx,
            op, type, NPY_FALSE, order, NPY_FALSE, 0);
    HPy_Close(ctx, type);
    return res;
}

HPyDef_METH(array_asanyarray, "asanyarray", HPyFunc_KEYWORDS)
static HPy
array_asanyarray_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t len_args, HPy kw)
{
    static const char *kwlist[] = {"a", "dtype", "order", "like", NULL};

    HPy op;
    HPy type = HPy_NULL; // PyArray_Descr *
    NPY_ORDER order = NPY_KEEPORDER;
    HPy like = HPy_NULL;
    // NPY_PREPARE_ARGPARSER;

    if (len_args != 1 || !HPy_IsNull(kw)) {
        HPyTracker ht;
        HPy h_type = HPy_NULL, h_order = HPy_NULL;
        // HPY TODO: uses npy_parse_arguments METH_FASTCALL|METH_KEYWORDS
        // 'like' is expected to be passed as keyword only.. we are ignoring this for now
        if (!HPyArg_ParseKeywords(ctx, &ht, args, len_args, kw, "O|OOO:asanyarray", kwlist,
                &op, &h_type, &h_order, &like)) {
            return HPy_NULL;
        }
        // if (npy_parse_arguments("asanyarray", args, len_args, kwnames,
        //         "a", NULL, &op,
        //         "|dtype", &PyArray_DescrConverter2, &type,
        //         "|order", &PyArray_OrderConverter, &order,
        //         "$like", NULL, &like,
        //         NULL, NULL, NULL) < 0) {
        //     HPy_Close(ctx, type);
        //     return HPy_NULL;
        // }
        if (HPyArray_DescrConverter2(ctx, h_type, &type) != NPY_SUCCEED ||
                HPyArray_OrderConverter(ctx, h_order, &order) != NPY_SUCCEED) {
            HPyErr_SetString(ctx, ctx->h_SystemError, "asanyarray: TODO");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        if (!HPy_IsNull(like)) {
            HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                    "asanyarray", like, kw, args, len_args);
            if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
                HPy_Close(ctx, type);
                HPyTracker_Close(ctx, ht);
                return deferred;
            }
            // HPy_Close(ctx, deferred); ?
        }
        HPyTracker_Close(ctx, ht);
    }
    else {
        op = args[0];
    }

    HPy res = _hpy_array_fromobject_generic(ctx,
            op, type, NPY_FALSE, order, NPY_TRUE, 0);
    HPy_Close(ctx, type);
    return res;
}


HPyDef_METH(array_ascontiguousarray, "ascontiguousarray", HPyFunc_KEYWORDS)
static HPy
array_ascontiguousarray_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t len_args, HPy kw)
{
    static const char *kwlist[] = {"a", "dtype", "like", NULL};

    HPy op;
    HPy type = HPy_NULL; // PyArray_Descr *
    HPy like = HPy_NULL;
    // NPY_PREPARE_ARGPARSER;

    if (len_args != 1 || !HPy_IsNull(kw)) {
        HPyTracker ht;
        HPy h_type = HPy_NULL;
        // HPY TODO: uses npy_parse_arguments METH_FASTCALL|METH_KEYWORDS
        // 'like' is expected to be passed as keyword only.. we are ignoring this for now
        if (!HPyArg_ParseKeywords(ctx, &ht, args, len_args, kw, "O|OOO:ascontiguousarray", kwlist,
                &op, &h_type, &like)) {
            return HPy_NULL;
        }
        // if (npy_parse_arguments("ascontiguousarray", args, len_args, kwnames,
        //         "a", NULL, &op,
        //         "|dtype", &PyArray_DescrConverter2, &type,
        //         "$like", NULL, &like,
        //         NULL, NULL, NULL) < 0) {
        //     HPy_Close(ctx, type);
        //     return HPy_NULL;
        // }
        if (HPyArray_DescrConverter2(ctx, h_type, &type) != NPY_SUCCEED) {
            HPyErr_SetString(ctx, ctx->h_SystemError, "ascontiguousarray: TODO");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        if (!HPy_IsNull(like)) {
            HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                    "ascontiguousarray", like, kw, args, len_args);
            if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
                HPy_Close(ctx, type);
                HPyTracker_Close(ctx, ht);
                return deferred;
            }
            // HPy_Close(ctx, deferred); ?
        }
        HPyTracker_Close(ctx, ht);
    }
    else {
        op = args[0];
    }

    HPy res = _hpy_array_fromobject_generic(ctx,
            op, type, NPY_FALSE, NPY_CORDER, NPY_FALSE, 1);
    HPy_Close(ctx, type);
    return res;
}

HPyDef_METH(array_asfortranarray, "asfortranarray", HPyFunc_KEYWORDS)
static HPy
array_asfortranarray_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t len_args, HPy kw)
{
    static const char *kwlist[] = {"a", "dtype", "like", NULL};

    HPy op;
    HPy type = HPy_NULL; // PyArray_Descr *
    HPy like = HPy_NULL;
    // NPY_PREPARE_ARGPARSER;

    if (len_args != 1 || !HPy_IsNull(kw)) {
        HPyTracker ht;
        HPy h_type = HPy_NULL;
        // HPY TODO: uses npy_parse_arguments METH_FASTCALL|METH_KEYWORDS
        // 'like' is expected to be passed as keyword only.. we are ignoring this for now
        if (!HPyArg_ParseKeywords(ctx, &ht, args, len_args, kw, "O|OOO:asfortranarray", kwlist,
                &op, &h_type, &like)) {
            return HPy_NULL;
        }
        // if (npy_parse_arguments("asfortranarray", args, len_args, kwnames,
        //         "a", NULL, &op,
        //         "|dtype", &PyArray_DescrConverter2, &type,
        //         "$like", NULL, &like,
        //         NULL, NULL, NULL) < 0) {
        //     HPy_Close(ctx, type);
        //     return HPy_NULL;
        // }
        if (HPyArray_DescrConverter2(ctx, h_type, &type) != NPY_SUCCEED) {
            HPyErr_SetString(ctx, ctx->h_SystemError, "asfortranarray: TODO");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        if (!HPy_IsNull(like)) {
            HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                    "asfortranarray", like, kw, args, len_args);
            if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
                HPy_Close(ctx, type);
                HPyTracker_Close(ctx, ht);
                return deferred;
            }
            // HPy_Close(ctx, deferred); ?
        }
        HPyTracker_Close(ctx, ht);
    }
    else {
        op = args[0];
    }

    HPy res = _hpy_array_fromobject_generic(ctx,
            op, type, NPY_FALSE, NPY_FORTRANORDER, NPY_FALSE, 1);
    HPy_Close(ctx, type);
    return res;
}


HPyDef_METH(array_copyto, "copyto", HPyFunc_KEYWORDS)
static HPy
array_copyto_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    static const char *kwlist[] = {"dst", "src", "casting", "where", NULL};

    HPy wheremask_in = HPy_NULL, h_src = HPy_NULL, h_casting = HPy_NULL;
    HPy dst = HPy_NULL, src = HPy_NULL, wheremask = HPy_NULL; // PyArrayObject *
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;

    HPyTracker ht;

    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "OO|OO:copyto", kwlist,
            &dst, &h_src, &h_casting, &wheremask_in)) {
        goto fail;
    }

    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    if (!HPy_TypeCheck(ctx, dst, array_type)) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "copyto: TODO");
        HPyTracker_Close(ctx, ht);
        HPy_Close(ctx, array_type);
        return HPy_NULL;
    }
    HPy_Close(ctx, array_type);
    if (HPyArray_Converter(ctx, h_src, &src) != NPY_SUCCEED ||
            (!HPy_IsNull(h_casting) && HPyArray_CastingConverter(ctx, h_casting, &casting) != NPY_SUCCEED)) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "copyto: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    if (!HPy_IsNull(wheremask_in)) {
        /* Get the boolean where mask */
        HPy dtype = HPyArray_DescrFromType(ctx, NPY_BOOL); // PyArray_Descr *
        if (HPy_IsNull(dtype)) {
            goto fail;
        }
        wheremask = HPyArray_FromAny(ctx, wheremask_in,
                                        dtype, 0, 0, 0, HPy_NULL);
        if (HPy_IsNull(wheremask)) {
            goto fail;
        }
    }

    if (HPyArray_AssignArray(ctx, dst, src, wheremask, casting) < 0) {
        goto fail;
    }

    HPy_Close(ctx, src);
    HPy_Close(ctx, wheremask);

    return HPy_Dup(ctx, ctx->h_None);;

fail:
    HPy_Close(ctx, src);
    HPy_Close(ctx, wheremask);
    return HPy_NULL;
}

HPyDef_METH(array_empty, "empty", HPyFunc_KEYWORDS)
static HPy
array_empty_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t len_args, HPy kw)
{
    static const char *kwlist[] = {"shape", "dtype", "order", "like", NULL};

    HPy typecode = HPy_NULL; // PyArray_Descr *
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order;
    HPy ret = HPy_NULL; // PyArrayObject *
    HPy like = HPy_NULL;
    // NPY_PREPARE_ARGPARSER;

    HPyTracker ht;
    HPy h_shape = HPy_NULL, h_type = HPy_NULL, h_order = HPy_NULL;
    // HPY TODO: uses npy_parse_arguments METH_FASTCALL|METH_KEYWORDS
    // 'like' is expected to be passed as keyword only.. we are ignoring this for now
    if (!HPyArg_ParseKeywords(ctx, &ht, args, len_args, kw, "O|OOO:empty", kwlist,
            &h_shape, &h_type, &h_order, &like)) {
        return HPy_NULL;
    }
    if (HPyArray_IntpConverter(ctx, h_shape, &shape) != NPY_SUCCEED ||
            HPyArray_DescrConverter(ctx, h_type, &typecode) != NPY_SUCCEED ||
            HPyArray_OrderConverter(ctx, h_order, &order) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "empty: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    // if (npy_parse_arguments("empty", args, len_args, kwnames,
    //         "shape", &PyArray_IntpConverter, &shape,
    //         "|dtype", &PyArray_DescrConverter, &typecode,
    //         "|order", &PyArray_OrderConverter, &order,
    //         "$like", NULL, &like,
    //         NULL, NULL, NULL) < 0) {
    //     goto fail;
    // }

    if (!HPy_IsNull(like)) {
        HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                "empty", like, kw, args, len_args);
        if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
            HPy_Close(ctx, typecode);
            HPyTracker_Close(ctx, ht);
            npy_free_cache_dim_obj(shape);
            return deferred;
        }
        // HPy_Close(ctx, deferred); ?
    }
    HPyTracker_Close(ctx, ht);

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    ret = HPyArray_Empty(ctx, shape.len, shape.ptr,
                                            typecode, is_f_order);

    npy_free_cache_dim_obj(shape);
    return ret;

fail:
    HPy_Close(ctx, typecode);
    npy_free_cache_dim_obj(shape);
    return HPy_NULL;
}

HPyDef_METH(array_empty_like, "empty_like", HPyFunc_KEYWORDS)
static HPy
array_empty_like_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{

    static const char *kwlist[] = {"prototype", "dtype", "order", "subok", "shape", NULL};

    HPy prototype = HPy_NULL; // PyArrayObject *
    HPy dtype = HPy_NULL; // PyArray_Descr *
    NPY_ORDER order = NPY_KEEPORDER;
    HPy ret = HPy_NULL; // PyArrayObject *
    int subok = 1;
    /* -1 is a special value meaning "not specified" */
    PyArray_Dims shape = {NULL, -1};
    HPy h_prototype = HPy_NULL, h_type = HPy_NULL, h_order = HPy_NULL, h_shape = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "O|OOiO:empty_like", kwlist,
                &h_prototype, &h_type, &h_order, &subok, &h_shape)) {
        goto fail;
    }
    if (HPyArray_Converter(ctx, h_prototype, &prototype) != NPY_SUCCEED ||
            HPyArray_DescrConverter2(ctx, h_type, &dtype) != NPY_SUCCEED ||
            HPyArray_OrderConverter(ctx, h_order, &order) != NPY_SUCCEED ||
            HPyArray_OptionalIntpConverter(ctx, h_shape, &shape) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "empty_like: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    /* steals the reference to dtype if it's not NULL */
    ret = HPyArray_NewLikeArrayWithShape(ctx, prototype, order, dtype,
                                                         shape.len, shape.ptr, subok);
    npy_free_cache_dim_obj(shape);
    if (HPy_IsNull(ret)) {
        goto fail;
    }
    HPy_Close(ctx, prototype);

    return ret;

fail:
    HPy_Close(ctx, prototype);
    HPy_Close(ctx, dtype);
    return HPy_NULL;
}

/*
 * This function is needed for supporting Pickles of
 * numpy scalar objects.
 */
HPyDef_METH(array_scalar, "scalar", HPyFunc_KEYWORDS)
static HPy
array_scalar_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{

    static const char *kwlist[] = {"dtype", "obj", NULL};

    HPy typecode; // PyArray_Descr *
    HPy obj = HPy_NULL, tmpobj = HPy_NULL;
    int alloc = 0;
    void *dptr;
    HPy ret;
    HPy base = HPy_NULL;

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "O|O:scalar", kwlist,
                &typecode, &obj)) {
        return HPy_NULL;
    }
    HPy arraydescr_type = HPyGlobal_Load(ctx, HPyArrayDescr_Type);
    if (!HPy_TypeCheck(ctx, typecode, arraydescr_type)) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "..");
        return HPy_NULL;
    }

    PyArray_Descr *typecode_struct = PyArray_Descr_AsStruct(ctx, typecode);
    if (PyDataType_FLAGCHK(typecode_struct, NPY_LIST_PICKLE)) {
        if (typecode_struct->type_num == NPY_OBJECT) {
            /* Deprecated 2020-11-24, NumPy 1.20 */
            if (HPY_DEPRECATE(ctx,
                    "Unpickling a scalar with object dtype is deprecated. "
                    "Object scalars should never be created. If this was a "
                    "properly created pickle, please open a NumPy issue. In "
                    "a best effort this returns the original object.") < 0) {
                return HPy_NULL;
            }
            if (!HPy_IsNull(obj)) {
                obj = HPy_Dup(ctx, obj);
            }
            HPyTracker_Close(ctx, ht);
            return obj;
        }
        /* We store the full array to unpack it here: */
        if (!HPyArray_CheckExact(ctx, obj)) {
            /* We pickle structured voids as arrays currently */
            HPyErr_SetString(ctx, ctx->h_RuntimeError,
                    "Unpickling NPY_LIST_PICKLE (structured void) scalar "
                    "requires an array.  The pickle file may be corrupted?");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        PyArrayObject *obj_struct = PyArrayObject_AsStruct(ctx, obj);
        if (!HPyArray_EquivTypes(ctx, obj, typecode)) {
            HPyErr_SetString(ctx, ctx->h_RuntimeError,
                    "Pickled array is not compatible with requested scalar "
                    "dtype.  The pickle file may be corrupted?");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
        base = obj;
        dptr = PyArray_BYTES(obj_struct);
    }
    else if (PyDataType_FLAGCHK(typecode_struct, NPY_ITEM_IS_POINTER)) {
        if (HPy_IsNull(obj)) {
            obj = ctx->h_None;
        }
        dptr = &obj;
    }
    else {
        if (HPy_IsNull(obj)) {
            if (typecode_struct->elsize == 0) {
                typecode_struct->elsize = 1;
            }
            dptr = PyArray_malloc(typecode_struct->elsize);
            if (dptr == NULL) {
                return HPyErr_NoMemory(ctx);
            }
            memset(dptr, '\0', typecode_struct->elsize);
            alloc = 1;
        }
        else {
            /* Backward compatibility with Python 2 NumPy pickles */
            if (HPyUnicode_Check(ctx, obj)) {
                tmpobj = HPyUnicode_AsLatin1String(ctx, obj);
                obj = tmpobj;
                if (HPy_IsNull(tmpobj)) {
                    /* More informative error message */
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Failed to encode Numpy scalar data string to "
                            "latin1,\npickle.load(a, encoding='latin1') is "
                            "assumed if unpickling.");
                    HPyTracker_Close(ctx, ht);
                    return HPy_NULL;
                }
            }
            if (!HPyBytes_Check(ctx, obj)) {
                HPyErr_SetString(ctx, ctx->h_TypeError,
                        "initializing object must be a bytes object");
                HPy_Close(ctx, tmpobj);
                HPyTracker_Close(ctx, ht);
                return HPy_NULL;
            }
            if (HPyBytes_GET_SIZE(ctx, obj) < typecode_struct->elsize) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "initialization string is too small");
                HPy_Close(ctx, tmpobj);
                HPyTracker_Close(ctx, ht);
                return HPy_NULL;
            }
            dptr = (void *)HPyBytes_AS_STRING(ctx, obj);
        }
    }
    ret = HPyArray_Scalar(ctx, dptr, typecode, base, PyArrayObject_AsStruct(ctx, base));

    /* free dptr which contains zeros */
    if (alloc) {
        PyArray_free(dptr);
    }
    HPy_Close(ctx, tmpobj);
    HPyTracker_Close(ctx, ht);
    return ret;
}

HPyDef_METH(array_zeros, "zeros", HPyFunc_KEYWORDS)
static HPy
array_zeros_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kw)
{
    static const char *kwlist[] = {"shape", "dtype", "order", "like", NULL};

    HPy h_shape = HPy_NULL, h_typecode = HPy_NULL;
    HPy h_order = HPy_NULL, h_like = HPy_NULL;
    HPyTracker ht;

    // HPY TODO: uses npy_parse_arguments METH_FASTCALL|METH_KEYWORDS
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kw, "O|OOO:zeros", kwlist,
            &h_shape, &h_typecode, &h_order, &h_like)) {
        goto cleanup;
    }

    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order = NPY_FALSE;
    HPy ret = HPy_NULL;
    HPy type_descr;

    if (HPyArray_IntpConverter(ctx, h_shape, &shape) != NPY_SUCCEED ||
            HPyArray_DescrConverter(ctx, h_typecode, &type_descr) != NPY_SUCCEED ||
            HPyArray_OrderConverter(ctx, h_order, &order) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "zeros: TODO");
        goto cleanup;
    }

    if (!HPy_IsNull(h_like)) {
        // HPY TODO: expects the kwnames from METH_FASTCALL|METH_KEYWORDS calling convention
        HPy_FatalError(ctx, "array_zeros with like != None is not supported in HPy port yet");
        // PyObject *deferred = array_implement_c_array_function_creation(
        //         "zeros", like, NULL, NULL, args, len_args, kwnames);
        // if (deferred != Py_NotImplemented) {
        //     HPy_Close(ctx, typecode);
        //     npy_free_cache_dim_obj(shape);
        //     return deferred;
        // }
    }

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            HPyErr_SetString(ctx, ctx->h_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto cleanup;
    }

    ret = HPyArray_Zeros(ctx, shape.len, shape.ptr, type_descr, (int) is_f_order);

cleanup:
    npy_free_cache_dim_obj(shape);
    HPy_Close(ctx, type_descr);
    HPyTracker_Close(ctx, ht);
    return ret;
}

static PyObject *
array_count_nonzero(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyArrayObject *array;
    npy_intp count;

    if (!PyArg_ParseTuple(args, "O&:count_nonzero", PyArray_Converter, &array)) {
        return NULL;
    }

    count =  PyArray_CountNonzero(array);

    Py_DECREF(array);

    if (count == -1) {
        return NULL;
    }
    return PyLong_FromSsize_t(count);
}

HPyDef_METH(array_fromstring, "fromstring", HPyFunc_KEYWORDS)
static HPy
array_fromstring_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy keywds)
{
    char *data;
    HPy_ssize_t nin = -1;
    char *sep = NULL;
    HPy_ssize_t s;
    static const char *kwlist[] = {"string", "dtype", "count", "sep", "like", NULL};
    HPy like = HPy_NULL;
    HPy h_descr = HPy_NULL, descr = HPy_NULL; // PyArray_Descr *
    HPyTracker ht;
    // 'like' is expected to be passed as keyword only.. we are ignoring this for now
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, keywds,
                "s|O" NPY_SSIZE_T_PYFMT "sO:fromstring", kwlist,
                &data, &s, &h_descr, &nin, &sep, &like)) {
        return HPy_NULL;
    }
    s = strlen(data);
    if (HPyArray_DescrConverter(ctx, h_descr, &descr) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "fromstring: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    if (!HPy_IsNull(like)) {
        HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                "fromstring", like, keywds, args, nargs);
        if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
            HPyTracker_Close(ctx, ht);
            return deferred;
        }
        // HPy_Close(ctx, deferred); ?
    }
    HPyTracker_Close(ctx, ht);

    /* binary mode, condition copied from PyArray_FromString */
    if (sep == NULL || strlen(sep) == 0) {
        /* Numpy 1.14, 2017-10-19 */
        if (HPY_DEPRECATE(ctx,
                "The binary mode of fromstring is deprecated, as it behaves "
                "surprisingly on unicode inputs. Use frombuffer instead") < 0) {
            HPy_Close(ctx, descr);
            return HPy_NULL;
        }
    }
    return HPyArray_FromString(ctx, data, (npy_intp)s, descr, (npy_intp)nin, sep);
}



HPyDef_METH(array_fromfile, "fromfile", HPyFunc_KEYWORDS)
static HPy
array_fromfile_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy keywds)
{
    HPy file = HPy_NULL, ret = HPy_NULL;
    PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
    char *sep = "";
    Py_ssize_t nin = -1;
    static const char *kwlist[] = {"file", "dtype", "count", "sep", "offset", "like", NULL};
    HPy like = HPy_NULL;
    HPy h_type = HPy_NULL, type = HPy_NULL; // PyArray_Descr *
    int own;
    npy_off_t orig_pos = 0, offset = 0;
    FILE *fp;

    HPyTracker ht;
    // 'like' is expected to be passed as keyword only.. we are ignoring this for now
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, keywds,
                "O|O" NPY_SSIZE_T_PYFMT "s" NPY_OFF_T_PYFMT "O:fromfile", kwlist,
                &file, &h_type, &nin, &sep, &offset, &like)) {
        return HPy_NULL;
    }
    if (HPyArray_DescrConverter(ctx, h_type, &type) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "fromstring: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!HPy_IsNull(like)) {
        HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                "fromfile", like, keywds, args, nargs);
        if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
            HPyTracker_Close(ctx, ht);
            return deferred;
        }
        // HPy_Close(ctx, deferred); ?
    }

    PyObject *py_file = HPy_AsPyObject(ctx, file);
    PyObject *py_file_ret = NpyPath_PathlikeToFspath(py_file);
    Py_DECREF(py_file);
    if (py_file_ret == NULL) {
        HPy_Close(ctx, type);
        return HPy_NULL;
    }
    file = HPy_FromPyObject(ctx, py_file_ret);
    Py_DECREF(py_file_ret);
    HPyTracker_Close(ctx, ht);

    if (offset != 0 && strcmp(sep, "") != 0) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "'offset' argument only permitted for binary files");
        HPy_Close(ctx, type);
        HPy_Close(ctx, file);
        return HPy_NULL;
    }
    if (HPyBytes_Check(ctx, file) || HPyUnicode_Check(ctx, file)) {
        py_file = HPy_AsPyObject(ctx, file);
        HPy_Close(ctx, file);
        py_file_ret = npy_PyFile_OpenFile(py_file, "rb");
        Py_DECREF(py_file);
        if (py_file_ret == NULL) {
            HPy_Close(ctx, type);
            return HPy_NULL;
        }
        file = HPy_FromPyObject(ctx, py_file_ret);
        Py_DECREF(py_file_ret);
        own = 1;
    }
    else {
        own = 0;
    }
    py_file = HPy_AsPyObject(ctx, file);
    fp = npy_PyFile_Dup2(py_file, "rb", &orig_pos);
    Py_DECREF(py_file);
    if (fp == NULL) {
        HPy_Close(ctx, file);
        HPy_Close(ctx, type);
        return HPy_NULL;
    }
    if (npy_fseek(fp, offset, SEEK_CUR) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto cleanup;
    }
    if (HPy_IsNull(type)) {
        type = HPyArray_DescrFromType(ctx, NPY_DEFAULT_TYPE);
    }
    ret = HPyArray_FromFile(ctx, fp, type, (npy_intp) nin, sep);

    /* If an exception is thrown in the call to PyArray_FromFile
     * we need to clear it, and restore it later to ensure that
     * we can cleanup the duplicated file descriptor properly.
     */
cleanup:
    PyErr_Fetch(&err_type, &err_value, &err_traceback);
    py_file = HPy_AsPyObject(ctx, file);
    if (npy_PyFile_DupClose2(py_file, fp, orig_pos) < 0) {
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        goto fail;
    }
    if (own && npy_PyFile_CloseFile(py_file) < 0) {
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        goto fail;
    }
    PyErr_Restore(err_type, err_value, err_traceback);
    Py_DECREF(py_file);
    HPy_Close(ctx, file);
    return ret;

fail:
    Py_DECREF(py_file);
    HPy_Close(ctx, file);
    HPy_Close(ctx, ret);
    return HPy_NULL;
}

HPyDef_METH(array_fromiter, "fromiter", HPyFunc_KEYWORDS)
static HPy
array_fromiter_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy keywds)
{
    HPy iter = HPy_NULL;
    HPy_ssize_t nin = -1;
    static const char *kwlist[] = {"iter", "dtype", "count", "like", NULL};
    HPy like = HPy_NULL;
    HPy h_descr = HPy_NULL, descr = HPy_NULL; // PyArray_Descr *

    HPyTracker ht;
    // 'like' is expected to be passed as keyword only.. we are ignoring this for now
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, keywds,
                "OO|" NPY_SSIZE_T_PYFMT "O:fromiter", kwlist,
                &iter, &h_descr, &nin, &like)) {
        HPy_Close(ctx, descr);
        return HPy_NULL;
    }

    if (HPyArray_DescrConverter(ctx, h_descr, &descr) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "fromiter: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!HPy_IsNull(like)) {
        HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                "fromiter", like, keywds, args, nargs);
        if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
            HPyTracker_Close(ctx, ht);
            return deferred;
        }
        // HPy_Close(ctx, deferred); ?
    }

    return HPyArray_FromIter(ctx, iter, descr, (npy_intp)nin);
}

HPyDef_METH(array_frombuffer, "frombuffer", HPyFunc_KEYWORDS)
static HPy
array_frombuffer_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy keywds)
{
    HPy obj = HPy_NULL;
    Py_ssize_t nin = -1, offset = 0;
    static const char *kwlist[] = {"buffer", "dtype", "count", "offset", "like", NULL};
    HPy like = HPy_NULL;
    HPy h_type = HPy_NULL, type = HPy_NULL; // PyArray_Descr *

    HPyTracker ht;
    // 'like' is expected to be passed as keyword only.. we are ignoring this for now
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, keywds,
                "O|O" NPY_SSIZE_T_PYFMT NPY_SSIZE_T_PYFMT "$O:frombuffer", kwlist,
                &obj, &h_type, &nin, &offset, &like)) {
        HPy_Close(ctx, type);
        return HPy_NULL;
    }

    if (HPyArray_DescrConverter(ctx, h_type, &type) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "frombuffer: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!HPy_IsNull(like)) {
        HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                "frombuffer", like, keywds, args, nargs);
        if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
            HPyTracker_Close(ctx, ht);
            return deferred;
        }
        // HPy_Close(ctx, deferred); ?
    }

    if (HPy_IsNull(type)) {
        type = HPyArray_DescrFromType(ctx, NPY_DEFAULT_TYPE);
    }
    return HPyArray_FromBuffer(ctx, obj, type, (npy_intp)nin, (npy_intp)offset);
}

HPyDef_METH(array_concatenate, "concatenate", HPyFunc_KEYWORDS)
static HPy
array_concatenate_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy a0;
    HPy out = HPy_NULL;
    HPy h_dtype = HPy_NULL, dtype = HPy_NULL; // PyArray_Descr *
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;
    HPy casting_obj = HPy_NULL;
    HPy res;
    int axis = 0;
    static const char *kwlist[] = {"seq", "axis", "out", "dtype", "casting", NULL};
    HPy h_axis = HPy_NULL;
    HPyTracker ht;
    // 'dtype' and 'casting' are expected to be passed as keywords only.. we are ignoring this for now
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "O|OOOO:concatenate", kwlist,
                &a0, &h_axis, &out, 
                &h_dtype, &casting_obj)) {
        return HPy_NULL;
    }

    if (HPyArray_AxisConverter(ctx, h_axis, &axis) != NPY_SUCCEED ||
            HPyArray_DescrConverter2(ctx, h_dtype, &dtype) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "concatenate: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    int casting_not_passed = 0;
    if (HPy_IsNull(casting_obj)) {
        /*
         * Casting was not passed in, needed for deprecation only.
         * This should be simplified once the deprecation is finished.
         */
        casting_not_passed = 1;
    }
    else if (!HPyArray_CastingConverter(ctx, casting_obj, &casting)) {
        HPy_Close(ctx, dtype);
        return HPy_NULL;
    }
    if (!HPy_IsNull(out)) {
        if (HPy_Is(ctx, out, ctx->h_None)) {
            out = HPy_NULL;
        }
        else if (!HPyArray_Check(ctx, out)) {
            HPyErr_SetString(ctx, ctx->h_TypeError, "'out' must be an array");
            HPy_Close(ctx, dtype);
            return HPy_NULL;
        }
    }
    res = HPyArray_ConcatenateInto(ctx, a0, axis, out, dtype,
            casting, casting_not_passed);
    HPy_Close(ctx, dtype);
    return res;
}

HPyDef_METH(array_innerproduct, "inner", HPyFunc_VARARGS)
static HPy
array_innerproduct_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs)
{
    HPy b0, a0;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OO:innerproduct", &a0, &b0)) {
        return HPy_NULL;
    }
    HPy inner = HPyArray_InnerProduct(ctx, a0, b0);
    HPy ret = HPyArray_Return(ctx, inner);
    HPy_Close(ctx, inner);
    return ret;
}

HPyDef_METH(array_matrixproduct, "dot", HPyFunc_KEYWORDS)
static HPy
array_matrixproduct_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy v, a, o = HPy_NULL;
    HPy ret; // PyArrayObject *
    static const char* kwlist[] = {"a", "b", "out", NULL};

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "OO|O:matrixproduct",
                                     kwlist, &a, &v, &o)) {
        return HPy_NULL;
    }
    if (!HPy_IsNull(o)) {
        if (HPy_Is(ctx, o, ctx->h_None)) {
            o = HPy_NULL;
        }
        else if (!HPyArray_Check(ctx, o)) {
            HPyErr_SetString(ctx, ctx->h_TypeError, "'out' must be an array");
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
    }
    PyArrayObject *o_struct = PyArrayObject_AsStruct(ctx, o);
    ret = HPyArray_MatrixProduct2(ctx, a, v, o, o_struct);
    HPyTracker_Close(ctx, ht);
    return HPyArray_Return(ctx, ret);
}


HPyDef_METH(array_vdot, "vdot", HPyFunc_VARARGS)
static HPy
array_vdot_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs)
{
    int typenum;
    char *ip1, *ip2, *op;
    npy_intp n, stride1, stride2;
    HPy op1, op2;
    npy_intp newdimptr[1] = {-1};
    PyArray_Dims newdims = {newdimptr, 1};
    HPy ap1 = HPy_NULL, ap2  = HPy_NULL, ret = HPy_NULL; // PyArrayObject *
    HPy type; // PyArray_Descr *
    PyArray_DotFunc * vdot;
    HPY_NPY_BEGIN_THREADS_DEF(ctx);

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OO:vdot", &op1, &op2)) {
        return HPy_NULL;
    }

    /*
     * Conjugating dot product using the BLAS for vectors.
     * Flattens both op1 and op2 before dotting.
     */
    typenum = HPyArray_ObjectType(ctx, op1, 0);
    typenum = HPyArray_ObjectType(ctx, op2, typenum);

    type = HPyArray_DescrFromType(ctx, typenum);
    // Py_INCREF(type); type is not a borrowed ref
    ap1 = HPyArray_FromAny(ctx, op1, type, 0, 0, 0, HPy_NULL);
    if (HPy_IsNull(ap1)) {
        HPy_Close(ctx, type);
        goto fail;
    }
    PyArrayObject *ap1_struct = PyArrayObject_AsStruct(ctx, ap1);
    op1 = HPyArray_Newshape(ctx, ap1, ap1_struct, &newdims, NPY_CORDER);
    if (HPy_IsNull(op1)) {
        HPy_Close(ctx, type);
        goto fail;
    }
    HPy_Close(ctx, ap1);
    ap1 = op1;

    ap2 = HPyArray_FromAny(ctx, op2, type, 0, 0, 0, HPy_NULL);
    if (HPy_IsNull(ap2)) {
        goto fail;
    }
    PyArrayObject *ap2_struct = PyArrayObject_AsStruct(ctx, ap2);
    op2 = HPyArray_Newshape(ctx, ap2, ap2_struct, &newdims, NPY_CORDER);
    if (HPy_IsNull(op2)) {
        goto fail;
    }
    HPy_Close(ctx, ap2);
    ap2 = op2;

    ap1_struct = PyArrayObject_AsStruct(ctx, ap1);
    ap2_struct = PyArrayObject_AsStruct(ctx, ap2);
    if (PyArray_DIM(ap2_struct, 0) != PyArray_DIM(ap1_struct, 0)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "vectors have different lengths");
        goto fail;
    }

    /* array scalar output */
    ret = hpy_new_array_for_sum(ctx, 
                                    ap1, ap1_struct, 
                                    ap2, ap2_struct, 
                                    HPy_NULL, NULL,
                                    0, (npy_intp *)NULL, typenum, NULL);
    if (HPy_IsNull(ret)) {
        goto fail;
    }

    n = PyArray_DIM(ap1_struct, 0);
    stride1 = PyArray_STRIDE(ap1_struct, 0);
    stride2 = PyArray_STRIDE(ap2_struct, 0);
    ip1 = PyArray_DATA(ap1_struct);
    ip2 = PyArray_DATA(ap2_struct);
    op = PyArray_DATA(PyArrayObject_AsStruct(ctx, ret));

    PyArray_Descr *type_struct = PyArray_Descr_AsStruct(ctx, type);
    switch (typenum) {
        case NPY_CFLOAT:
            vdot = (PyArray_DotFunc *)CFLOAT_vdot;
            break;
        case NPY_CDOUBLE:
            vdot = (PyArray_DotFunc *)CDOUBLE_vdot;
            break;
        case NPY_CLONGDOUBLE:
            vdot = (PyArray_DotFunc *)CLONGDOUBLE_vdot;
            break;
        case NPY_OBJECT:
            vdot = (PyArray_DotFunc *)OBJECT_vdot;
            break;
        default:
            vdot = type_struct->f->dotfunc;
            if (vdot == NULL) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "function not available for this data type");
                goto fail;
            }
    }

    if (n < 500) {
        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
    }
    else {
        HPY_NPY_BEGIN_THREADS_DESCR(ctx, type_struct);
        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
        HPY_NPY_END_THREADS_DESCR(ctx, type_struct);
    }

    HPy_Close(ctx, ap1);
    HPy_Close(ctx, ap2);
    return HPyArray_Return(ctx, ret);
fail:
    HPy_Close(ctx, ap1);
    HPy_Close(ctx, ap2);
    HPy_Close(ctx, ret);
    return HPy_NULL;
}

static int
hpy_einsum_sub_op_from_str(HPyContext *ctx, HPy *args, HPy_ssize_t nargs, HPy *str_obj, char **subscripts,
                       HPy /* PyArrayObject ** */ *op)
{
    int i, nop;
    HPy subscripts_str;

    nop = nargs - 1;
    if (nop <= 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "too many operands");
        return -1;
    }

    /* Get the subscripts string */
    subscripts_str = args[0];
    if (HPyUnicode_Check(ctx, subscripts_str)) {
        *str_obj = HPyUnicode_AsASCIIString(ctx, subscripts_str);
        if (HPy_IsNull(*str_obj)) {
            return -1;
        }
        subscripts_str = *str_obj;
    }

    /* XXX: This cast seems to be a bit dangerous but is necessary */
    *subscripts = (char *)HPyBytes_AsString(ctx, subscripts_str);
    if (*subscripts == NULL) {
        HPy_Close(ctx, *str_obj);
        *str_obj = HPy_NULL;
        return -1;
    }

    /* Set the operands to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = HPy_NULL;
    }

    /* Get the operands */
    for (i = 0; i < nop; ++i) {
        HPy obj = args[i+1];

        op[i] = HPyArray_FROM_OF(ctx, obj, NPY_ARRAY_ENSUREARRAY);
        if (HPy_IsNull(op[i])) {
            goto fail;
        }
    }

    return nop;

fail:
    for (i = 0; i < nop; ++i) {
        HPy_Close(ctx, op[i]);
        op[i] = HPy_NULL;
    }

    return -1;
}

/*
 * Converts a list of subscripts to a string.
 *
 * Returns -1 on error, the number of characters placed in subscripts
 * otherwise.
 */
static int
hpy_einsum_list_to_subscripts(HPyContext *ctx, HPy obj, char *subscripts, int subsize)
{
    int ellipsis = 0, subindex = 0;
    npy_intp i, size;
    HPy item;

    // obj = PySequence_Fast(obj, "the subscripts for each operand must "
    //                            "be a list or a tuple");
    if (!HPySequence_Check(ctx, obj)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, 
                                "the subscripts for each operand must "
                                "be a list or a tuple");
        return -1;
    }
    size = HPy_Length(ctx, obj);

    for (i = 0; i < size; ++i) {
        item = HPy_GetItem_i(ctx, obj, i);
        /* Ellipsis */
        if (HPy_Is(ctx, item, ctx->h_Ellipsis)) {
            if (ellipsis) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "each subscripts list may have only one ellipsis");
                HPy_Close(ctx, item);
                HPy_Close(ctx, obj);
                return -1;
            }
            if (subindex + 3 >= subsize) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "subscripts list is too long");
                HPy_Close(ctx, item);
                HPy_Close(ctx, obj);
                return -1;
            }
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            ellipsis = 1;
        }
        /* Subscript */
        else {
            npy_intp s = HPyArray_PyIntAsIntp(ctx, item);
            /* Invalid */
            if (hpy_error_converting(ctx, s)) {
                HPyErr_SetString(ctx, ctx->h_TypeError,
                        "each subscript must be either an integer "
                        "or an ellipsis");
                HPy_Close(ctx, item);
                HPy_Close(ctx, obj);
                return -1;
            }
            npy_bool bad_input = 0;

            if (subindex + 1 >= subsize) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "subscripts list is too long");
                HPy_Close(ctx, item);
                HPy_Close(ctx, obj);
                return -1;
            }

            if (s < 0) {
                bad_input = 1;
            }
            else if (s < 26) {
                subscripts[subindex++] = 'A' + (char)s;
            }
            else if (s < 2*26) {
                subscripts[subindex++] = 'a' + (char)s - 26;
            }
            else {
                bad_input = 1;
            }

            if (bad_input) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "subscript is not within the valid range [0, 52)");
                HPy_Close(ctx, item);
                HPy_Close(ctx, obj);
                return -1;
            }
        }
        HPy_Close(ctx, item);
    }

    HPy_Close(ctx, obj);

    return subindex;
}

/*
 * Fills in the subscripts, with maximum size subsize, and op,
 * with the values in the tuple 'args'.
 *
 * Returns -1 on error, number of operands placed in op otherwise.
 */
static int
hpy_einsum_sub_op_from_lists(HPyContext *ctx, HPy *args, HPy_ssize_t nargs,
                char *subscripts, int subsize, HPy *op)
{
    int subindex = 0;
    npy_intp i, nop;

    nop = nargs/2;

    if (nop == 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "must provide at least an "
                        "operand and a subscripts list to einsum");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "too many operands");
        return -1;
    }

    /* Set the operands to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = HPy_NULL;
    }

    /* Get the operands and build the subscript string */
    for (i = 0; i < nop; ++i) {
        HPy obj = args[2*i];
        int n;

        /* Comma between the subscripts for each operand */
        if (i != 0) {
            subscripts[subindex++] = ',';
            if (subindex >= subsize) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "subscripts list is too long");
                goto fail;
            }
        }

        op[i] = HPyArray_FROM_OF(ctx, obj, NPY_ARRAY_ENSUREARRAY);
        if (HPy_IsNull(op[i])) {
            goto fail;
        }

        obj = args[2*i+1];
        n = hpy_einsum_list_to_subscripts(ctx, obj, subscripts+subindex,
                                      subsize-subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    /* Add the '->' to the string if provided */
    if (nargs == 2*nop+1) {
        HPy obj;
        int n;

        if (subindex + 2 >= subsize) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "subscripts list is too long");
            goto fail;
        }
        subscripts[subindex++] = '-';
        subscripts[subindex++] = '>';

        obj = args[2*nop];
        n = hpy_einsum_list_to_subscripts(ctx, obj, subscripts+subindex,
                                      subsize-subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    /* NULL-terminate the subscripts string */
    subscripts[subindex] = '\0';

    return nop;

fail:
    for (i = 0; i < nop; ++i) {
        HPy_Close(ctx, op[i]);
        op[i] = HPy_NULL;
    }

    return -1;
}

HPyDef_METH(array_einsum, "c_einsum", HPyFunc_KEYWORDS)
static HPy
array_einsum_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    char *subscripts = NULL, subscripts_buffer[256];
    HPy str_obj = HPy_NULL, str_key_obj = HPy_NULL;
    HPy arg0;
    int i, nop;
    HPy op[NPY_MAXARGS]; // PyArrayObject *
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    HPy out = HPy_NULL; // PyArrayObject *
    HPy dtype = HPy_NULL; // PyArray_Descr *
    HPy ret = HPy_NULL;

    if (nargs < 1) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand, or at least one operand "
                        "and its corresponding subscripts list");
        return HPy_NULL;
    }
    arg0 = args[0];

    /* einsum('i,j', a, b), einsum('i,j->ij', a, b) */
    if (HPyBytes_Check(ctx, arg0) || HPyUnicode_Check(ctx, arg0)) {
        nop = hpy_einsum_sub_op_from_str(ctx, args, nargs, &str_obj, &subscripts, op);
    }
    /* einsum(a, [0], b, [1]), einsum(a, [0], b, [1], [0,1]) */
    else {
        nop = hpy_einsum_sub_op_from_lists(ctx, args, nargs, subscripts_buffer,
                                    sizeof(subscripts_buffer), op);
        subscripts = subscripts_buffer;
    }
    if (nop <= 0) {
        goto finish;
    }

    /* Get the keyword arguments */
    if (!HPy_IsNull(kwds)) {
        HPy keys = HPyDict_Keys(ctx, kwds);
        HPy_ssize_t keys_len = HPy_Length(ctx, keys);
        for (HPy_ssize_t i = 0; i < keys_len; i++) {
            HPy key = HPy_GetItem_i(ctx, keys, i);
            const char *str = NULL;

            HPy_Close(ctx, str_key_obj);
            str_key_obj = HPyUnicode_AsASCIIString(ctx, key);
            if (!HPy_IsNull(str_key_obj)) {
                key = str_key_obj;
            }

            str = HPyBytes_AsString(ctx, key);

            if (str == NULL) {
                HPy_Close(ctx, key);
                HPy_Close(ctx, keys);
                HPyErr_Clear(ctx);
                HPyErr_SetString(ctx, ctx->h_TypeError, "invalid keyword");
                goto finish;
            }

            HPy value = HPy_GetItem(ctx, kwds, key);
            HPy_Close(ctx, key);
            if (strcmp(str,"out") == 0) {
                if (HPyArray_Check(ctx, value)) {
                    HPy_Close(ctx, out);
                    out = value;
                }
                else {
                    HPy_Close(ctx, value);
                    HPy_Close(ctx, keys);
                    HPyErr_SetString(ctx, ctx->h_TypeError,
                                "keyword parameter out must be an "
                                "array for einsum");
                    goto finish;
                }
            }
            else if (strcmp(str,"order") == 0) {
                if (!HPyArray_OrderConverter(ctx, value, &order)) {
                    HPy_Close(ctx, value);
                    HPy_Close(ctx, keys);
                    goto finish;
                }
            }
            else if (strcmp(str,"casting") == 0) {
                if (!HPyArray_CastingConverter(ctx, value, &casting)) {
                    HPy_Close(ctx, value);
                    HPy_Close(ctx, keys);
                    goto finish;
                }
            }
            else if (strcmp(str,"dtype") == 0) {
                if (!HPyArray_DescrConverter2(ctx, value, &dtype)) {
                    HPy_Close(ctx, value);
                    HPy_Close(ctx, keys);
                    goto finish;
                }
            }
            else {
                HPy_Close(ctx, value);
                HPy_Close(ctx, keys);
                HPyErr_Format_p(ctx, ctx->h_TypeError,
                            "'%s' is an invalid keyword for einsum",
                            str);
                goto finish;
            }
        }
        HPy_Close(ctx, keys);
    }
    CAPI_WARN("calling PyArray_EinsteinSum");
    PyArrayObject **py_op_in = HPy_AsPyObjectArray(ctx, op, nop);
    PyArray_Descr *py_dtype = HPy_AsPyObject(ctx, dtype);
    PyArrayObject *py_out = HPy_AsPyObject(ctx, out);
    PyObject *py_ret = (PyObject *)PyArray_EinsteinSum(subscripts, nop, py_op_in, py_dtype,
                                        order, casting, py_out);
    ret = HPy_FromPyObject(ctx, py_ret);
    HPy_DecrefAndFreeArray(ctx, py_op_in, nop);
    Py_DECREF(py_dtype);
    Py_DECREF(py_out);
    Py_DECREF(py_ret);
    /* If no output was supplied, possibly convert to a scalar */
    if (!HPy_IsNull(ret) && HPy_IsNull(out)) {
        ret = HPyArray_Return(ctx, ret);
    }

finish:
    for (i = 0; i < nop; ++i) {
        HPy_Close(ctx, op[i]);
    }
    HPy_Close(ctx, dtype);
    HPy_Close(ctx, str_obj);
    HPy_Close(ctx, str_key_obj);
    HPy_Close(ctx, out);
    /* out is not a borrowed reference */

    return ret;
}

HPyDef_METH(_fastCopyAndTranspose, "_fastCopyAndTranspose", HPyFunc_VARARGS)
static HPy
_fastCopyAndTranspose_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs)
{
    HPy a0;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O:_fastCopyAndTranspose", &a0)) {
        return HPy_NULL;
    }
    return HPyArray_Return(ctx, HPyArray_CopyAndTranspose(ctx, a0));
}

static PyObject *
array_correlate(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *shape, *a0;
    int mode = 0;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("correlate", args, len_args, kwnames,
            "a", NULL, &a0,
            "v", NULL, &shape,
            "|mode", &PyArray_CorrelatemodeConverter, &mode,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    return PyArray_Correlate(a0, shape, mode);
}

static PyObject*
array_correlate2(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *shape, *a0;
    int mode = 0;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("correlate2", args, len_args, kwnames,
            "a", NULL, &a0,
            "v", NULL, &shape,
            "|mode", &PyArray_CorrelatemodeConverter, &mode,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    return PyArray_Correlate2(a0, shape, mode);
}

HPyDef_METH(array_arange, "arange", HPyFunc_KEYWORDS)
static HPy
array_arange_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kw)
{
    HPy o_start = HPy_NULL, o_stop = HPy_NULL, o_step = HPy_NULL, range = HPy_NULL;
    HPy typecode = HPy_NULL; // PyArray_Descr *
    HPy like = HPy_NULL;
    static const char* kwlist[] = {"start", "stop", "step", "dtype", "like", NULL};
    // NPY_PREPARE_ARGPARSER;

    // HPY TODO: uses npy_parse_arguments METH_FASTCALL|METH_KEYWORDS
    // 'like' is expected to be passed as keyword only.. we are ignoring this for now
    HPy h_typecode = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kw, "O|OOOO:arange", kwlist,
            &o_start, &o_stop, &o_step, &h_typecode, &like)) {
        return HPy_NULL;
    }
    if (HPyArray_DescrConverter2(ctx, h_typecode, &typecode) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "arange: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    // if (npy_parse_arguments("arange", args, len_args, kwnames,
    //         "|start", NULL, &o_start,
    //         "|stop", NULL, &o_stop,
    //         "|step", NULL, &o_step,
    //         "|dtype", &PyArray_DescrConverter2, &typecode,
    //         "$like", NULL, &like,
    //         NULL, NULL, NULL) < 0) {
    //     HPy_Close(ctx, typecode);
    //     return HPy_NULL;
    // }

    if (!HPy_IsNull(like)) {
        HPy deferred = hpy_array_implement_c_array_function_creation(ctx,
                "frombuffer", like, kw, args, nargs);
        if (!HPy_Is(ctx, deferred, ctx->h_NotImplemented)) {
            HPy_Close(ctx, typecode);
            HPyTracker_Close(ctx, ht);
            return deferred;
        }
        // HPy_Close(ctx, deferred); ?
    }

    if (HPy_IsNull(o_stop)) {
        if (nargs == 0){
            HPyErr_SetString(ctx, ctx->h_TypeError,
                "arange() requires stop to be specified.");
            HPy_Close(ctx, typecode);
            return HPy_NULL;
        }
    }
    else if (HPy_IsNull(o_start)) {
        o_start = o_stop;
        o_stop = HPy_NULL;
    }

    range = HPyArray_ArangeObj(ctx, o_start, o_stop, o_step, typecode);
    HPy_Close(ctx, typecode);

    return range;
}

/*NUMPY_API
 *
 * Included at the very first so not auto-grabbed and thus not labeled.
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCVersion(void)
{
    return (unsigned int)NPY_ABI_VERSION;
}

/*HPY_NUMPY_API
 *
 * Included at the very first so not auto-grabbed and thus not labeled.
 */
NPY_NO_EXPORT unsigned int
HPyArray_GetNDArrayCVersion(void)
{
    return PyArray_GetNDArrayCVersion();
}

/*NUMPY_API
 * Returns the built-in (at compilation time) C API version
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCFeatureVersion(void)
{
    return (unsigned int)NPY_API_VERSION;
}

HPyDef_METH(_get_ndarray_c_version, "_get_ndarray_c_version", HPyFunc_NOARGS)
static HPy
_get_ndarray_c_version_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy))
{
    // static const char *kwlist[] = {NULL};

    // if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist )) {
    //     return NULL;
    // }
    return HPyLong_FromLong(ctx, (long) PyArray_GetNDArrayCVersion() );
}

/*NUMPY_API
*/
NPY_NO_EXPORT int
PyArray_GetEndianness(void)
{
    const union {
        npy_uint32 i;
        char c[4];
    } bint = {0x01020304};

    if (bint.c[0] == 1) {
        return NPY_CPU_BIG;
    }
    else if (bint.c[0] == 4) {
        return NPY_CPU_LITTLE;
    }
    else {
        return NPY_CPU_UNKNOWN_ENDIAN;
    }
}

HPyDef_METH(_reconstruct, "_reconstruct", HPyFunc_VARARGS)
static HPy
_reconstruct_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs)
{

    HPy ret;
    HPy subtype; // PyTypeObject *
    PyArray_Dims shape = {NULL, 0};
    HPy dtype = HPy_NULL; // PyArray_Descr *

    evil_global_disable_warn_O4O8_flag = 1;
    HPy h_shape = HPy_NULL;
    HPy h_dtype = HPy_NULL;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OOO:_reconstruct",
                &subtype, &shape, &dtype)) {
        goto fail;
    }

    if (!HPy_TypeCheck(ctx, subtype, ctx->h_TypeType)) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "");
        return HPy_NULL;
    }

    if (!HPyArray_IntpConverter(ctx, h_shape, &shape)) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "");
        return HPy_NULL;
    }

    if (!HPyArray_DescrConverter(ctx, h_dtype, &dtype)) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "");
        return HPy_NULL;
    }
    
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    if (!HPyType_IsSubtype(ctx, subtype, array_type)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "_reconstruct: First argument must be a sub-type of ndarray");
        goto fail;
    }
    HPy_Close(ctx, array_type);
    ret = HPyArray_NewFromDescr(ctx, subtype, dtype,
            (int)shape.len, shape.ptr, NULL, NULL, 0, HPy_NULL);
    HPy_Close(ctx, dtype);
    npy_free_cache_dim_obj(shape);

    evil_global_disable_warn_O4O8_flag = 0;

    return ret;

fail:
    evil_global_disable_warn_O4O8_flag = 0;

    HPy_Close(ctx, dtype);
    HPy_Close(ctx, array_type);
    npy_free_cache_dim_obj(shape);
    return HPy_NULL;
}

static PyObject *
array_set_string_function(PyObject *NPY_UNUSED(self), PyObject *args,
        PyObject *kwds)
{
    PyObject *op = NULL;
    int repr = 1;
    static char *kwlist[] = {"f", "repr", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Oi:set_string_function", kwlist, &op, &repr)) {
        return NULL;
    }
    /* reset the array_repr function to built-in */
    if (op == Py_None) {
        op = NULL;
    }
    if (op != NULL && !PyCallable_Check(op)) {
        PyErr_SetString(PyExc_TypeError,
                "Argument must be callable.");
        return NULL;
    }
    PyArray_SetStringFunction(op, repr);
    Py_RETURN_NONE;
}

HPyDef_METH(array_set_ops_function, "set_numeric_ops", HPyFunc_KEYWORDS)
static HPy
array_set_ops_function_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy oldops = HPy_NULL;

    if (HPy_IsNull(oldops = _PyArray_GetNumericOps(ctx))) {
        return HPy_NULL;
    }
    /*
     * Should probably ensure that objects are at least callable
     *  Leave this to the caller for now --- error will be raised
     *  later when use is attempted
     */
    if (!HPy_IsNull(kwds) && HPyArray_SetNumericOps(ctx, kwds) == -1) {
        HPy_Close(ctx, oldops);
        if (!HPyErr_Occurred(ctx)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                "one or more objects not callable");
        }
        return HPy_NULL;
    }
    return oldops;
}

static PyObject *
array_set_datetimeparse_function(PyObject *NPY_UNUSED(self),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    PyErr_SetString(PyExc_RuntimeError, "This function has been removed");
    return NULL;
}

/*
 * inner loop with constant size memcpy arguments
 * this allows the compiler to replace function calls while still handling the
 * alignment requirements of the platform.
 */
#define INNER_WHERE_LOOP(size) \
    do { \
        npy_intp i; \
        for (i = 0; i < n; i++) { \
            if (*csrc) { \
                memcpy(dst, xsrc, size); \
            } \
            else { \
                memcpy(dst, ysrc, size); \
            } \
            dst += size; \
            xsrc += xstride; \
            ysrc += ystride; \
            csrc += cstride; \
        } \
    } while(0)


/*HPY_NUMPY_API
 * Where
 */
NPY_NO_EXPORT HPy
HPyArray_Where(HPyContext *ctx, HPy condition, HPy x, HPy y)
{
    HPy h_arr, h_ax, h_ay;
    HPy h_ret = HPy_NULL;
    PyObject *ret = NULL;

    h_arr = HPyArray_FROM_O(ctx, condition);
    if (HPy_IsNull(h_arr)) {
        return HPy_NULL;
    }
    if (HPy_IsNull(x) && HPy_IsNull(y)) {
        h_ret = HPyArray_Nonzero(ctx, h_arr);
        HPy_Close(ctx, h_arr);
        return h_ret;
    }
    if (HPy_IsNull(x) || HPy_IsNull(y)) {
        HPy_Close(ctx, h_arr);
        PyErr_SetString(PyExc_ValueError,
                "either both or neither of x and y should be given");
        return HPy_NULL;
    }

    h_ax = HPyArray_FROM_O(ctx, x);
    h_ay = HPyArray_FROM_O(ctx, y);
    if (HPy_IsNull(h_ax) || HPy_IsNull(h_ay)) {
        goto fail;
    }
    else {
        npy_uint32 flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED |
                           NPY_ITER_REFS_OK | NPY_ITER_ZEROSIZE_OK;
        HPy h_op_in[4] = {
            HPy_NULL, h_arr, h_ax, h_ay
        };
        npy_uint32 op_flags[4] = {
            NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NO_SUBTYPE,
            NPY_ITER_READONLY, NPY_ITER_READONLY, NPY_ITER_READONLY
        };
        HPy h_npy_bool = HPyArray_DescrFromType(ctx, NPY_BOOL);
        HPy h_common_dt = HPyArray_ResultType(ctx, 2, &h_op_in[0] + 2,
                                                       0, NULL);
        HPy h_op_dt[4] = {h_common_dt, h_npy_bool,
                          h_common_dt, h_common_dt};
        NpyIter * iter;
        int needs_api;
        NPY_BEGIN_THREADS_DEF;

        if (HPy_IsNull(h_common_dt) || HPy_IsNull(h_op_dt[1])) {
            HPy_Close(ctx, h_op_dt[1]);
            HPy_Close(ctx, h_common_dt);
            goto fail;
        }
        iter =  HNpyIter_MultiNew(ctx, 4, h_op_in, flags,
                                 NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                 op_flags, h_op_dt);
        HPy_Close(ctx, h_op_dt[1]);
        if (iter == NULL) {
            goto fail;
        }

        needs_api = NpyIter_IterationNeedsAPI(iter);

        /* Get the result from the iterator object array */
        ret = (PyObject*)NpyIter_GetOperandArray(iter)[0];

        CAPI_WARN("NPY_BEGIN_THREADS_NDITER and other NpyIter_*");
        NPY_BEGIN_THREADS_NDITER(iter);

        if (NpyIter_GetIterSize(iter) != 0) {
            NpyIter_IterNextFunc *iternext = HNpyIter_GetIterNext(ctx, iter, NULL);
            npy_intp * innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
            char **dataptrarray = NpyIter_GetDataPtrArray(iter);

            do {
                PyArray_Descr * dtx = NpyIter_GetDescrArray(iter)[2];
                PyArray_Descr * dty = NpyIter_GetDescrArray(iter)[3];
                int axswap = PyDataType_ISBYTESWAPPED(dtx);
                int ayswap = PyDataType_ISBYTESWAPPED(dty);
                PyArray_CopySwapFunc *copyswapx = dtx->f->copyswap;
                PyArray_CopySwapFunc *copyswapy = dty->f->copyswap;
                int native = (axswap == ayswap) && (axswap == 0) && !needs_api;
                npy_intp n = (*innersizeptr);
                npy_intp itemsize = NpyIter_GetDescrArray(iter)[0]->elsize;
                npy_intp cstride = NpyIter_GetInnerStrideArray(iter)[1];
                npy_intp xstride = NpyIter_GetInnerStrideArray(iter)[2];
                npy_intp ystride = NpyIter_GetInnerStrideArray(iter)[3];
                char * dst = dataptrarray[0];
                char * csrc = dataptrarray[1];
                char * xsrc = dataptrarray[2];
                char * ysrc = dataptrarray[3];

                /* constant sizes so compiler replaces memcpy */
                if (native && itemsize == 16) {
                    INNER_WHERE_LOOP(16);
                }
                else if (native && itemsize == 8) {
                    INNER_WHERE_LOOP(8);
                }
                else if (native && itemsize == 4) {
                    INNER_WHERE_LOOP(4);
                }
                else if (native && itemsize == 2) {
                    INNER_WHERE_LOOP(2);
                }
                else if (native && itemsize == 1) {
                    INNER_WHERE_LOOP(1);
                }
                else {
                    /* copyswap is faster than memcpy even if we are native */
                    npy_intp i;
                    for (i = 0; i < n; i++) {
                        CAPI_WARN("Not clear what dtx/y->f->copyswap may call...");
                        if (*csrc) {
                            copyswapx(dst, xsrc, axswap, ret);
                        }
                        else {
                            copyswapy(dst, ysrc, ayswap, ret);
                        }
                        dst += itemsize;
                        xsrc += xstride;
                        ysrc += ystride;
                        csrc += cstride;
                    }
                }
            } while (iternext(ctx, iter));
        }

        NPY_END_THREADS;

        Py_INCREF(ret);
        HPy_Close(ctx, h_arr);
        HPy_Close(ctx, h_ax);
        HPy_Close(ctx, h_ay);

        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            return HPy_NULL;
        }

        return HPy_FromPyObject(ctx, ret);
    }

fail:
    HPy_Close(ctx, h_arr);
    HPy_Close(ctx, h_ax);
    HPy_Close(ctx, h_ay);
    return HPy_NULL;
}

/*NUMPY_API
 * Where
 */
NPY_NO_EXPORT PyObject *
PyArray_Where(PyObject *condition, PyObject *x, PyObject *y)
{
    HPyContext *ctx = npy_get_context();
    HPy h_condition = HPy_FromPyObject(ctx, condition);
    HPy h_x = HPy_FromPyObject(ctx, x);
    HPy h_y = HPy_FromPyObject(ctx, y);
    HPy h_res = HPyArray_Where(ctx, h_condition, h_x, h_y);
    PyObject *res = HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_condition);
    HPy_Close(ctx, h_x);
    HPy_Close(ctx, h_y);
    HPy_Close(ctx, h_res);
    return res;
}

#undef INNER_WHERE_LOOP

HPyDef_METH(array_where, "where", HPyFunc_VARARGS)
static HPy
array_where_impl(HPyContext *ctx, HPy ignored, HPy *args, HPy_ssize_t nargs)
{
    HPy obj = HPy_NULL, x = HPy_NULL, y = HPy_NULL;

    HPyTracker ht;
    if (!HPyArg_Parse(ctx, &ht, args, nargs, "O|OO:where", &obj, &x, &y)) {
        return HPy_NULL;
    }
    HPy res = HPyArray_Where(ctx, obj, x, y);
    HPyTracker_Close(ctx, ht);
    return res;
}

HPyDef_METH(array_lexsort, "lexsort", HPyFunc_KEYWORDS)
static HPy
array_lexsort_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    int axis = -1;
    HPy obj;
    static const char *kwlist[] = {"keys", "axis", NULL};

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "O|i:lexsort", kwlist, &obj, &axis)) {
        return HPy_NULL;
    }
    PyObject *py_obj = HPy_AsPyObject(ctx, obj);
    CAPI_WARN("calling PyArray_LexSort");
    PyObject *py_ret = PyArray_LexSort(py_obj, axis);
    HPy ret = HPy_FromPyObject(ctx, py_ret);
    Py_DECREF(py_obj);
    Py_DECREF(py_ret);
    
    HPy r = HPyArray_Return(ctx, ret);
    HPy_Close(ctx, ret);
    return r;
}

HPyDef_METH(array_can_cast_safely, "can_cast", HPyFunc_KEYWORDS)
static HPy
array_can_cast_safely_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy from_obj = HPy_NULL;
    HPy d1 = HPy_NULL; // PyArray_Descr *
    HPy d2 = HPy_NULL; // PyArray_Descr *
    int ret;
    HPy retobj = HPy_NULL;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    static const char *kwlist[] = {"from_", "to", "casting", NULL};

    HPy h_d2 = HPy_NULL, h_casting = HPy_NULL;
    HPyTracker ht;
    if(!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "OO|O:can_cast", kwlist,
                &from_obj,
                &h_d2,
                &h_casting)) {
        goto finish;
    }
    if (HPyArray_DescrConverter2(ctx, h_d2, &d2) != NPY_SUCCEED ||
            HPyArray_CastingConverter(ctx, h_casting, &casting) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "can_cast: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    if (HPy_IsNull(d2)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "did not understand one of the types; 'None' not accepted");
        goto finish;
    }

    /* If the first parameter is an object or scalar, use CanCastArrayTo */
    if (HPyArray_Check(ctx, from_obj)) {
        ret = HPyArray_CanCastArrayTo(ctx, from_obj, d2, casting);
    }
    else if (HPyArray_IsScalar(ctx, from_obj, Generic) ||
                                HPyArray_IsPythonNumber(ctx, from_obj)) {
        HPy arr; // PyArrayObject *
        arr = HPyArray_FROM_O(ctx, from_obj);
        if (HPy_IsNull(arr)) {
            goto finish;
        }
        ret = HPyArray_CanCastArrayTo(ctx, arr, d2, casting);
        HPy_Close(ctx, arr);
    }
    /* Otherwise use CanCastTypeTo */
    else {
        if (!HPyArray_DescrConverter2(ctx, from_obj, &d1) || HPy_IsNull(d1)) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "did not understand one of the types; 'None' not accepted");
            goto finish;
        }
        ret = HPyArray_CanCastTypeTo(ctx, d1, d2, casting);
    }

    retobj = HPy_Dup(ctx, ret ? ctx->h_True : ctx->h_False);
    // Py_INCREF(retobj);

 finish:
    HPy_Close(ctx, d1);
    HPy_Close(ctx, d2);
    return retobj;
}

HPyDef_METH(array_promote_types, "promote_types", HPyFunc_VARARGS)
static HPy
array_promote_types_impl(HPyContext *ctx, HPy dummy, HPy *args, HPy_ssize_t nargs)
{
    HPy d1 = HPy_NULL, h_d1;
    HPy d2 = HPy_NULL, h_d2;
    HPy ret = HPy_NULL;
    if (!HPyArg_Parse(ctx, NULL, args, nargs, "OO:promote_types",
                &h_d1, &h_d2)) {
        goto finish;
    }

    if (HPyArray_DescrConverter2(ctx, h_d1, &d1) == NPY_FAIL) {
        goto finish;
    }

    if (HPyArray_DescrConverter2(ctx, h_d2, &d2) == NPY_FAIL) {
        goto finish;
    }

    if (HPy_IsNull(d1) || HPy_IsNull(d2)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "did not understand one of the types");
        goto finish;
    }

    ret = HPyArray_PromoteTypes(ctx, d1, d2);

 finish:
    HPy_Close(ctx, d1);
    HPy_Close(ctx, d2);
    return ret;
}

HPyDef_METH(array_min_scalar_type, "min_scalar_type", HPyFunc_VARARGS)
static HPy
array_min_scalar_type_impl(HPyContext *ctx, HPy dummy, HPy *args, HPy_ssize_t nargs)
{
    HPy array_in = HPy_NULL;
    HPy array;
    HPy ret = HPy_NULL;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O:min_scalar_type", &array_in)) {
        return HPy_NULL;
    }

    array = HPyArray_FROM_O(ctx, array_in);
    if (HPy_IsNull(array)) {
        return HPy_NULL;
    }

    ret = HPyArray_MinScalarType(ctx, array);
    HPy_Close(ctx, array);
    return ret;
}

HPyDef_METH(array_result_type, "result_type", HPyFunc_VARARGS)
static HPy
array_result_type_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs)
{
    npy_intp i, len, narr = 0, ndtypes = 0;
    HPy *arr = NULL; // PyArrayObject **
    HPy *dtypes = NULL; // PyArray_Descr **
    HPy ret = HPy_NULL;

    len = nargs;
    if (len == 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                        "at least one array or dtype is required");
        goto finish;
    }

    arr = PyArray_malloc(2 * len * sizeof(HPy)); // sizeof(void *));
    if (arr == NULL) {
        return HPyErr_NoMemory(ctx);
    }
    dtypes = &arr[len]; // (PyArray_Descr**)

    for (i = 0; i < len; ++i) {
        HPy obj = args[i];
        if (HPyArray_Check(ctx, obj)) {
            // Py_INCREF(obj);
            arr[narr] = HPy_Dup(ctx, obj);
            ++narr;
        }
        else if (HPyArray_IsScalar(ctx, obj, Generic) ||
                                    HPyArray_IsPythonNumber(ctx, obj)) {
            arr[narr] = HPyArray_FROM_O(ctx, obj);
            if (HPy_IsNull(arr[narr])) {
                goto finish;
            }
            HPy obj_type = HPy_Type(ctx, obj);
            if (HPy_Is(ctx, obj, ctx->h_LongType) || 
                    HPy_Is(ctx, obj, ctx->h_FloatType) ||
                    HPy_Is(ctx, obj, ctx->h_ComplexType)) {
                ((PyArrayObject_fields *)PyArrayObject_AsStruct(ctx, arr[narr]))->flags |= _NPY_ARRAY_WAS_PYSCALAR;
            }
            HPy_Close(ctx, obj_type);
            ++narr;
        }
        else {
            if (!HPyArray_DescrConverter(ctx, obj, &dtypes[ndtypes])) {
                goto finish;
            }
            ++ndtypes;
        }
    }

    ret = HPyArray_ResultType(ctx, narr, arr, ndtypes, dtypes);

finish:
    for (i = 0; i < narr; ++i) {
        HPy_Close(ctx, arr[i]);
    }
    for (i = 0; i < ndtypes; ++i) {
        HPy_Close(ctx, dtypes[i]);
    }
    PyArray_free(arr);
    return ret;
}

HPyDef_METH(array_datetime_data, "datetime_data", HPyFunc_VARARGS)
static HPy
array_datetime_data_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs)
{
    HPy h_dtype = HPy_NULL, dtype =  HPy_NULL; // PyArray_Descr *
    PyArray_DatetimeMetaData *meta;

    if (!HPyArg_Parse(ctx, NULL, args, nargs, "O:datetime_data", &h_dtype)) {
        return HPy_NULL;
    }

    if (HPyArray_DescrConverter(ctx, h_dtype, &dtype) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "datetime_data: TODO");
        return HPy_NULL;
    }

    meta = h_get_datetime_metadata_from_dtype(ctx, PyArray_Descr_AsStruct(ctx, dtype));
    if (meta == NULL) {
        HPy_Close(ctx, dtype);
        return HPy_NULL;
    }

    HPy res = hpy_convert_datetime_metadata_to_tuple(ctx, meta);
    HPy_Close(ctx, dtype);
    return res;
}


static int
trimmode_converter(PyObject *obj, TrimMode *trim)
{
    if (!PyUnicode_Check(obj) || PyUnicode_GetLength(obj) != 1) {
        goto error;
    }
    const char *trimstr = PyUnicode_AsUTF8AndSize(obj, NULL);

    if (trimstr != NULL) {
        if (trimstr[0] == 'k') {
            *trim = TrimMode_None;
        }
        else if (trimstr[0] == '.') {
            *trim = TrimMode_Zeros;
        }
        else if (trimstr[0] ==  '0') {
            *trim = TrimMode_LeaveOneZero;
        }
        else if (trimstr[0] ==  '-') {
            *trim = TrimMode_DptZeros;
        }
        else {
            goto error;
        }
    }
    return NPY_SUCCEED;

error:
    PyErr_Format(PyExc_TypeError,
            "if supplied, trim must be 'k', '.', '0' or '-' found `%100S`",
            obj);
    return NPY_FAIL;
}

static int
hpy_trimmode_converter(HPyContext *ctx, HPy obj, TrimMode *trim)
{
    if (!HPyUnicode_Check(ctx, obj) || HPy_Length(ctx, obj) != 1) {
        goto error;
    }
    const char *trimstr = HPyUnicode_AsUTF8AndSize(ctx, obj, NULL);

    if (trimstr != NULL) {
        if (trimstr[0] == 'k') {
            *trim = TrimMode_None;
        }
        else if (trimstr[0] == '.') {
            *trim = TrimMode_Zeros;
        }
        else if (trimstr[0] ==  '0') {
            *trim = TrimMode_LeaveOneZero;
        }
        else if (trimstr[0] ==  '-') {
            *trim = TrimMode_DptZeros;
        }
        else {
            goto error;
        }
    }
    return NPY_SUCCEED;

error:
    // PyErr_Format(PyExc_TypeError,
    //         "if supplied, trim must be 'k', '.', '0' or '-' found `%100S`",
    //         obj);
    HPyErr_SetString(ctx, ctx->h_TypeError,
            "if supplied, trim must be 'k', '.', '0' or '-' found `%100S`");
    return NPY_FAIL;
}


/*
 * Prints floating-point scalars using the Dragon4 algorithm, scientific mode.
 * See docstring of `np.format_float_scientific` for description of arguments.
 * The differences is that a value of -1 is valid for pad_left, exp_digits,
 * precision, which is equivalent to `None`.
 */
HPyDef_METH(dragon4_scientific, "dragon4_scientific", HPyFunc_KEYWORDS)
static HPy
dragon4_scientific_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy obj;
    int precision=-1, pad_left=-1, exp_digits=-1, min_digits=-1;
    DigitMode digit_mode;
    TrimMode trim = TrimMode_None;
    int sign=0, unique=1;
    static const char *kwlist[] = {"x", "precision", "unique", "sign",
                            "trim", "pad_left", "exp_digits", "min_digits", NULL};
    // NPY_PREPARE_ARGPARSER;
    HPy h_precision = HPy_NULL;
    HPy h_unique = HPy_NULL;
    HPy h_sign = HPy_NULL;
    HPy h_trim = HPy_NULL;
    HPy h_pad_left = HPy_NULL;
    HPy h_exp_digits = HPy_NULL;
    HPy h_min_digits = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "O|OOOOOOOO:dragon4_scientific",
                kwlist,
                &obj,
                &h_precision,
                &h_unique,
                &h_sign,
                &h_trim,
                &h_pad_left,
                &h_exp_digits,
                &h_min_digits)) {
        return HPy_NULL;
    }
    if (hpy_trimmode_converter(ctx, h_trim, &trim) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_precision, &precision) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_unique, &unique) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_sign, &sign) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_pad_left, &pad_left) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_exp_digits, &exp_digits) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_min_digits, &min_digits) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "dragon4_scientific: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    // if (npy_parse_arguments("dragon4_scientific", args, len_args, kwnames,
    //         "x", NULL , &obj,
    //         "|precision", &PyArray_PythonPyIntFromInt, &precision,
    //         "|unique", &PyArray_PythonPyIntFromInt, &unique,
    //         "|sign", &PyArray_PythonPyIntFromInt, &sign,
    //         "|trim", &trimmode_converter, &trim,
    //         "|pad_left", &PyArray_PythonPyIntFromInt, &pad_left,
    //         "|exp_digits", &PyArray_PythonPyIntFromInt, &exp_digits,
    //         "|min_digits", &PyArray_PythonPyIntFromInt, &min_digits,
    //         NULL, NULL, NULL) < 0) {
    //     return NULL;
    // }

    digit_mode = unique ? DigitMode_Unique : DigitMode_Exact;

    if (unique == 0 && precision < 0) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
            "in non-unique mode `precision` must be supplied");
        return HPy_NULL;
    }

    return Dragon4_Scientific(ctx, obj, digit_mode, precision, min_digits, sign, trim,
                              pad_left, exp_digits);
}

/*
 * Prints floating-point scalars using the Dragon4 algorithm, positional mode.
 * See docstring of `np.format_float_positional` for description of arguments.
 * The differences is that a value of -1 is valid for pad_left, pad_right,
 * precision, which is equivalent to `None`.
 */
HPyDef_METH(dragon4_positional, "dragon4_positional", HPyFunc_KEYWORDS)
static HPy
dragon4_positional_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy obj;
    int precision=-1, pad_left=-1, pad_right=-1, min_digits=-1;
    CutoffMode cutoff_mode;
    DigitMode digit_mode;
    TrimMode trim = TrimMode_None;
    int sign=0, unique=1, fractional=0;
    static const char *kwlist[] = {"x", "precision", "unique", "fractional", "sign",
                            "trim", "pad_left", "pad_right", "min_digits", NULL};
    // NPY_PREPARE_ARGPARSER;

    HPy h_precision = HPy_NULL;
    HPy h_unique = HPy_NULL;
    HPy h_fractional = HPy_NULL;
    HPy h_sign = HPy_NULL;
    HPy h_trim = HPy_NULL;
    HPy h_pad_left = HPy_NULL;
    HPy h_pad_right = HPy_NULL;
    HPy h_min_digits = HPy_NULL;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "O|OOOOOOOO:dragon4_positional",
                kwlist,
                &obj,
                &h_precision,
                &h_unique,
                &h_fractional,
                &h_sign,
                &h_trim,
                &h_pad_left,
                &h_pad_right,
                &h_min_digits)) {
        return HPy_NULL;
    }
    if (hpy_trimmode_converter(ctx, h_trim, &trim) != NPY_SUCCEED|
            HPyArray_PythonPyIntFromInt(ctx, h_precision, &precision) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_unique, &unique) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_fractional, &fractional) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_sign, &sign) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_pad_left, &pad_left) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_pad_right, &pad_right) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_min_digits, &min_digits) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "dragon4_positional: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    digit_mode = unique ? DigitMode_Unique : DigitMode_Exact;
    cutoff_mode = fractional ? CutoffMode_FractionLength :
                               CutoffMode_TotalLength;

    if (unique == 0 && precision < 0) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
            "in non-unique mode `precision` must be supplied");
        return HPy_NULL;
    }

    return Dragon4_Positional(ctx, obj, digit_mode, cutoff_mode, precision,
                              min_digits, sign, trim, pad_left, pad_right);
}

HPyDef_METH(format_longfloat, "format_longfloat", HPyFunc_KEYWORDS)
static HPy
format_longfloat_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy obj;
    unsigned int precision;
    static const char *kwlist[] = {"x", "precision", NULL};

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "OI:format_longfloat", kwlist,
                &obj, &precision)) {
        return HPy_NULL;
    }
    if (!HPyArray_IsScalar(ctx, obj, LongDouble)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "not a longfloat");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    HPy ret = Dragon4_Scientific(ctx, obj, DigitMode_Unique, precision, -1, 0,
                              TrimMode_LeaveOneZero, -1, -1);
    HPyTracker_Close(ctx, ht);
    return ret;
}

HPyDef_METH(compare_chararrays, "compare_chararrays", HPyFunc_KEYWORDS)
static HPy
compare_chararrays_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    HPy array;
    HPy other;
    HPy newarr, newoth; // PyArrayObject *
    int cmp_op;
    npy_bool rstrip;
    char *cmp_str;
    HPy_ssize_t strlength;
    HPy res = HPy_NULL;
    static char msg[] = "comparison must be '==', '!=', '<', '>', '<=', '>='";
    static const char *kwlist[] = {"a1", "a2", "cmp", "rstrip", NULL};

    HPy h_rstrip;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "OOsO:compare_chararrays",
                kwlist,
                &array, &other, &cmp_str,
                &h_rstrip)) {
        return HPy_NULL;
    }
    strlength = strlen(cmp_str);
    if (HPyArray_BoolConverter(ctx, h_rstrip, &rstrip) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, ": TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    if (strlength < 1 || strlength > 2) {
        goto err;
    }
    if (strlength > 1) {
        if (cmp_str[1] != '=') {
            goto err;
        }
        if (cmp_str[0] == '=') {
            cmp_op = Py_EQ;
        }
        else if (cmp_str[0] == '!') {
            cmp_op = Py_NE;
        }
        else if (cmp_str[0] == '<') {
            cmp_op = Py_LE;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GE;
        }
        else {
            goto err;
        }
    }
    else {
        if (cmp_str[0] == '<') {
            cmp_op = Py_LT;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GT;
        }
        else {
            goto err;
        }
    }

    newarr = HPyArray_FROM_O(ctx, array);
    if (HPy_IsNull(newarr)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    newoth = HPyArray_FROM_O(ctx, other);
    if (HPy_IsNull(newoth)) {
        HPy_Close(ctx, newarr);
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }
    if (PyArray_ISSTRING(PyArrayObject_AsStruct(ctx, newarr)) && 
            PyArray_ISSTRING(PyArrayObject_AsStruct(ctx, newoth))) {
        res = _hpy_strings_richcompare(ctx, newarr, newoth, cmp_op, rstrip != 0);
    }
    else {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "comparison of non-string arrays");
    }
    HPy_Close(ctx, newarr);
    HPy_Close(ctx, newoth);
    HPyTracker_Close(ctx, ht);
    return res;

 err:
    HPyErr_SetString(ctx, ctx->h_ValueError, msg);
    HPyTracker_Close(ctx, ht);
    return HPy_NULL;
}

static PyObject *
_vec_string_with_args(PyArrayObject* char_array, PyArray_Descr* type,
                      PyObject* method, PyObject* args)
{
    PyObject* broadcast_args[NPY_MAXARGS];
    PyArrayMultiIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;
    Py_ssize_t i, n, nargs;

    nargs = PySequence_Size(args) + 1;
    if (nargs == -1 || nargs > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "len(args) must be < %d", NPY_MAXARGS - 1);
        Py_DECREF(type);
        goto err;
    }

    broadcast_args[0] = (PyObject*)char_array;
    for (i = 1; i < nargs; i++) {
        PyObject* item = PySequence_GetItem(args, i-1);
        if (item == NULL) {
            Py_DECREF(type);
            goto err;
        }
        broadcast_args[i] = item;
        Py_DECREF(item);
    }
    in_iter = (PyArrayMultiIterObject*)PyArray_MultiIterFromObjects
        (broadcast_args, nargs, 0);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }
    n = in_iter->numiter;

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(in_iter->nd,
            in_iter->dimensions, type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    while (PyArray_MultiIter_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* args_tuple = PyTuple_New(n);
        if (args_tuple == NULL) {
            goto err;
        }

        for (i = 0; i < n; i++) {
            PyArrayIterObject* it = in_iter->iters[i];
            PyObject* arg = PyArray_ToScalar(PyArray_ITER_DATA(it), it->ao);
            if (arg == NULL) {
                Py_DECREF(args_tuple);
                goto err;
            }
            /* Steals ref to arg */
            PyTuple_SetItem(args_tuple, i, arg);
        }

        item_result = PyObject_CallObject(method, args_tuple);
        Py_DECREF(args_tuple);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                    "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_MultiIter_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string_no_args(PyArrayObject* char_array,
                                   PyArray_Descr* type, PyObject* method)
{
    /*
     * This is a faster version of _vec_string_args to use when there
     * are no additional arguments to the string method.  This doesn't
     * require a broadcast iterator (and broadcast iterators don't work
     * with 1 argument anyway).
     */
    PyArrayIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;

    in_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)char_array);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(
            PyArray_NDIM(char_array), PyArray_DIMS(char_array), type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    while (PyArray_ITER_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* item = PyArray_ToScalar(in_iter->dataptr, in_iter->ao);
        if (item == NULL) {
            goto err;
        }

        item_result = PyObject_CallFunctionObjArgs(method, item, NULL);
        Py_DECREF(item);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_ITER_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *NPY_UNUSED(kwds))
{
    PyArrayObject* char_array = NULL;
    PyArray_Descr *type;
    PyObject* method_name;
    PyObject* args_seq = NULL;

    PyObject* method = NULL;
    PyObject* result = NULL;

    if (!PyArg_ParseTuple(args, "O&O&O|O",
                PyArray_Converter, &char_array,
                PyArray_DescrConverter, &type,
                &method_name, &args_seq)) {
        goto err;
    }

    if (PyArray_TYPE(char_array) == NPY_STRING) {
        method = PyObject_GetAttr((PyObject *)&PyBytes_Type, method_name);
    }
    else if (PyArray_TYPE(char_array) == NPY_UNICODE) {
        method = PyObject_GetAttr((PyObject *)&PyUnicode_Type, method_name);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "string operation on non-string array");
        Py_DECREF(type);
        goto err;
    }
    if (method == NULL) {
        Py_DECREF(type);
        goto err;
    }

    if (args_seq == NULL
            || (PySequence_Check(args_seq) && PySequence_Size(args_seq) == 0)) {
        result = _vec_string_no_args(char_array, type, method);
    }
    else if (PySequence_Check(args_seq)) {
        result = _vec_string_with_args(char_array, type, method, args_seq);
    }
    else {
        Py_DECREF(type);
        PyErr_SetString(PyExc_TypeError,
                "'args' must be a sequence of arguments");
        goto err;
    }
    if (result == NULL) {
        goto err;
    }

    Py_DECREF(char_array);
    Py_DECREF(method);

    return (PyObject*)result;

 err:
    Py_XDECREF(char_array);
    Py_XDECREF(method);

    return 0;
}

HPyDef_METH(_hpy_vec_string, "_vec_string", HPyFunc_VARARGS)
static HPy
_hpy_vec_string_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs)
{
    HPyErr_SetString(ctx, ctx->h_SystemError, "not ported to HPy yet");
    return HPy_NULL;
}

#ifndef __NPY_PRIVATE_NO_SIGNAL

static NPY_TLS int sigint_buf_init = 0;
static NPY_TLS NPY_SIGJMP_BUF _NPY_SIGINT_BUF;

/*NUMPY_API
 */
NPY_NO_EXPORT void
_PyArray_SigintHandler(int signum)
{
    PyOS_setsig(signum, SIG_IGN);
    /*
     * jump buffer may be uninitialized as SIGINT allowing functions are usually
     * run in other threads than the master thread that receives the signal
     */
    if (sigint_buf_init > 0) {
        NPY_SIGLONGJMP(_NPY_SIGINT_BUF, signum);
    }
    /*
     * sending SIGINT to the worker threads to cancel them is job of the
     * application
     */
}

/*NUMPY_API
 */
NPY_NO_EXPORT void*
_PyArray_GetSigintBuf(void)
{
    sigint_buf_init = 1;
    return (void *)&_NPY_SIGINT_BUF;
}

#else

NPY_NO_EXPORT void
_PyArray_SigintHandler(int signum)
{
    return;
}

NPY_NO_EXPORT void*
_PyArray_GetSigintBuf(void)
{
    return NULL;
}

#endif


static HPy
_array_shares_memory_impl(HPyContext *ctx, HPy *args, HPy_ssize_t nargs,
                         HPy kwds, Py_ssize_t default_max_work,
                         int raise_exceptions)
{
    HPy self_obj = HPy_NULL;
    HPy other_obj = HPy_NULL;
    HPy self = HPy_NULL; // PyArrayObject *
    HPy other = HPy_NULL; // PyArrayObject *
    HPy max_work_obj = HPy_NULL;
    static const char *kwlist[] = {"self", "other", "max_work", NULL};

    mem_overlap_t result;
    static PyObject *too_hard_cls = NULL;
    Py_ssize_t max_work;
    HPY_NPY_BEGIN_THREADS_DEF;

    max_work = default_max_work;

    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kwds, "OO|O:shares_memory_impl", kwlist,
                                     &self_obj, &other_obj, &max_work_obj)) {
        return HPy_NULL;
    }

    if (HPyArray_Check(ctx, self_obj)) {
        self = HPy_Dup(ctx, self_obj);
        // Py_INCREF(self);
    }
    else {
        /* Use FromAny to enable checking overlap for objects exposing array
           interfaces etc. */
        self = HPyArray_FROM_O(ctx, self_obj);
        if (HPy_IsNull(self)) {
            goto fail;
        }
    }

    if (HPyArray_Check(ctx, other_obj)) {
        other = HPy_Dup(ctx, other_obj);
        // Py_INCREF(other);
    }
    else {
        other = HPyArray_FROM_O(ctx, other_obj);
        if (HPy_IsNull(other)) {
            goto fail;
        }
    }

    if (HPy_IsNull(max_work_obj) || HPy_Is(ctx, max_work_obj, ctx->h_None)) {
        /* noop */
    }
    else if (HPyLong_Check(ctx, max_work_obj)) {
        max_work = HPyLong_AsSsize_t(ctx, max_work_obj);
        if (HPyErr_Occurred(ctx)) {
            goto fail;
        }
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError, "max_work must be an integer");
        goto fail;
    }

    if (max_work < -2) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Invalid value for max_work");
        goto fail;
    }

    PyArrayObject *self_struct = PyArrayObject_AsStruct(ctx, self);
    PyArrayObject *other_struct = PyArrayObject_AsStruct(ctx, other);
    HPY_NPY_BEGIN_THREADS(ctx);
    result = hpy_solve_may_share_memory(ctx, self, self_struct, other, other_struct, max_work);
    HPY_NPY_END_THREADS(ctx);

    HPy_Close(ctx, self);
    HPy_Close(ctx, other);

    if (result == MEM_OVERLAP_NO) {
        return HPy_Dup(ctx, ctx->h_False);
    }
    else if (result == MEM_OVERLAP_YES) {
        return HPy_Dup(ctx, ctx->h_True);
    }
    else if (result == MEM_OVERLAP_OVERFLOW) {
        if (raise_exceptions) {
            HPyErr_SetString(ctx, ctx->h_OverflowError,
                            "Integer overflow in computing overlap");
            return HPy_NULL;
        }
        else {
            /* Don't know, so say yes */
            return HPy_Dup(ctx, ctx->h_True);
        }
    }
    else if (result == MEM_OVERLAP_TOO_HARD) {
        if (raise_exceptions) {
            npy_cache_import("numpy.core._exceptions", "TooHardError",
                             &too_hard_cls);
            if (too_hard_cls) {
                PyErr_SetString(too_hard_cls, "Exceeded max_work");
            }
            return HPy_NULL;
        }
        else {
            /* Don't know, so say yes */
            return HPy_Dup(ctx, ctx->h_True);
        }
    }
    else {
        /* Doesn't happen usually */
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                        "Error in computing overlap");
        return HPy_NULL;
    }

fail:
    HPy_Close(ctx, self);
    HPy_Close(ctx, other);
    return HPy_NULL;
}


HPyDef_METH(array_shares_memory, "shares_memory", HPyFunc_KEYWORDS)
static HPy
array_shares_memory_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    return _array_shares_memory_impl(ctx, args, nargs, kwds, NPY_MAY_SHARE_EXACT, 1);
}


HPyDef_METH(array_may_share_memory, "may_share_memory", HPyFunc_KEYWORDS)
static HPy
array_may_share_memory_impl(HPyContext *ctx, HPy NPY_UNUSED(ignored), HPy *args, HPy_ssize_t nargs, HPy kwds)
{
    return _array_shares_memory_impl(ctx, args, nargs, kwds, NPY_MAY_SHARE_BOUNDS, 0);
}

HPyDef_METH(normalize_axis_index, "normalize_axis_index", HPyFunc_KEYWORDS)
static HPy
normalize_axis_index_impl(HPyContext *ctx, HPy NPY_UNUSED(dummy), HPy *args, HPy_ssize_t nargs, HPy kw)
{
    static const char *kwlist[] = {"axis", "ndim", "msg_prefix", NULL};

    int axis;
    int ndim;
    HPy msg_prefix = ctx->h_None;
    // NPY_PREPARE_ARGPARSER;

    HPy h_axis, h_ndim;
    HPyTracker ht;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kw, "OO|O:normalize_axis_index", kwlist,
            &axis, &ndim, &msg_prefix)) {
        return HPy_NULL;
    }
    if (HPyArray_PythonPyIntFromInt(ctx, h_axis, &axis) != NPY_SUCCEED ||
            HPyArray_PythonPyIntFromInt(ctx, h_ndim, &ndim) != NPY_SUCCEED) {
        HPyErr_SetString(ctx, ctx->h_SystemError, "normalize_axis_index: TODO");
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    // if (npy_parse_arguments("normalize_axis_index", args, len_args, kwnames,
    //         "axis", &PyArray_PythonPyIntFromInt, &axis,
    //         "ndim", &PyArray_PythonPyIntFromInt, &ndim,
    //         "|msg_prefix", NULL, &msg_prefix,
    //         NULL, NULL, NULL) < 0) {
    //     return NULL;
    // }
    if (hpy_check_and_adjust_axis_msg(ctx, &axis, ndim, msg_prefix) < 0) {
        return HPy_NULL;
    }

    return HPyLong_FromLong(ctx, axis);
}


static PyObject *
_reload_guard(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args)) {
    static int initialized = 0;

#if !defined(PYPY_VERSION)
    if (PyThreadState_Get()->interp != PyInterpreterState_Main()) {
        if (PyErr_WarnEx(PyExc_UserWarning,
                "NumPy was imported from a Python sub-interpreter but "
                "NumPy does not properly support sub-interpreters. "
                "This will likely work for most users but might cause hard to "
                "track down issues or subtle bugs. "
                "A common user of the rare sub-interpreter feature is wsgi "
                "which also allows single-interpreter mode.\n"
                "Improvements in the case of bugs are welcome, but is not "
                "on the NumPy roadmap, and full support may require "
                "significant effort to achieve.", 2) < 0) {
            return NULL;
        }
        /* No need to give the other warning in a sub-interpreter as well... */
        initialized = 1;
        Py_RETURN_NONE;
    }
#endif
    if (initialized) {
        if (PyErr_WarnEx(PyExc_UserWarning,
                "The NumPy module was reloaded (imported a second time). "
                "This can in some cases result in small but subtle issues "
                "and is discouraged.", 2) < 0) {
            return NULL;
        }
    }
    initialized = 1;
    Py_RETURN_NONE;
}

static struct PyMethodDef array_module_methods[] = {
    {"set_string_function",
        (PyCFunction)array_set_string_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_datetimeparse_function",
        (PyCFunction)array_set_datetimeparse_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"count_nonzero",
        (PyCFunction)array_count_nonzero,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"correlate",
        (PyCFunction)array_correlate,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"correlate2",
        (PyCFunction)array_correlate2,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"_discover_array_parameters", (PyCFunction)_discover_array_parameters,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_get_experimental_dtype_api", (PyCFunction)_get_experimental_dtype_api,
        METH_O, NULL},
    /* from umath */
    {"get_handler_name",
        (PyCFunction) get_handler_name,
        METH_VARARGS, NULL},
    {"get_handler_version",
        (PyCFunction) get_handler_version,
        METH_VARARGS, NULL},
    {"_reload_guard", (PyCFunction)_reload_guard,
        METH_NOARGS,
        "Give a warning on reload and big warning in sub-interpreters."},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

#include "__multiarray_api.c"
#include "array_method.h"

static PyNumberMethods dummy_as_number = {
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (ternaryfunc)1,
    (unaryfunc)1,
    (unaryfunc)1,
    (unaryfunc)1,
    (inquiry)1,
    (unaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (unaryfunc)1,
    (void *)1,
    (unaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (ternaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
    (unaryfunc)1,
    (binaryfunc)1,
    (binaryfunc)1,
};

static PySequenceMethods dummy_as_sequence = {
    (lenfunc)1,
    (binaryfunc)1,
    (ssizeargfunc)1,
    (ssizeargfunc)1,
    (void*)1,
    (ssizeobjargproc)1,
    (void*)1,
    (objobjproc)1,
    (binaryfunc)1,
    (ssizeargfunc)1,
};

static PyMappingMethods dummy_as_mapping = {
    (lenfunc)1,
    (binaryfunc)1,
    (objobjargproc)1,
};

#ifndef GRAALVM_PYTHON
static PyAsyncMethods dummy_as_async = {
    (unaryfunc)1,
    (unaryfunc)1,
    (unaryfunc)1,
};
#endif

static PyBufferProcs dummy_as_buffer = {
     (getbufferproc)1,
     (releasebufferproc)1,
};

// HPY HACK: we don't have an alternative way to create a type that has 
//           multiple bases and only inherit slots from one of them only.
//           So, we replace slots of the bases that should not considered 
//           with dummy slots, assign them back, and clean up the new type
//           from those dummy slots.
static void set_as_func(PyTypeObject *type) {
#ifndef GRAALVM_PYTHON
    type->tp_as_async = &dummy_as_async;
    type->tp_as_async = NULL;
#endif

    type->tp_as_number = &dummy_as_number;
    type->tp_as_mapping = &dummy_as_mapping;
    type->tp_as_sequence = &dummy_as_sequence;
    type->tp_as_buffer = &dummy_as_buffer;
}

static void cleanup(PyTypeObject *type) {

    PyNumberMethods *n = type->tp_as_number; 
    n->nb_absolute = (intptr_t)n->nb_absolute == (intptr_t)1 ? NULL : n->nb_absolute;
    n->nb_add = (intptr_t)n->nb_add == (intptr_t)1 ? NULL : n->nb_add;
    n->nb_subtract = (intptr_t)n->nb_subtract == (intptr_t)1 ? NULL : n->nb_subtract;
    n->nb_multiply = (intptr_t)n->nb_multiply == (intptr_t)1 ? NULL : n->nb_multiply;
    n->nb_remainder = (intptr_t)n->nb_remainder == (intptr_t)1 ? NULL : n->nb_remainder;
    n->nb_divmod = (intptr_t)n->nb_divmod == (intptr_t)1 ? NULL : n->nb_divmod;
    n->nb_power = (intptr_t)n->nb_power == (intptr_t)1 ? NULL : n->nb_power;
    n->nb_negative = (intptr_t)n->nb_negative == (intptr_t)1 ? NULL : n->nb_negative;
    n->nb_positive = (intptr_t)n->nb_positive == (intptr_t)1 ? NULL : n->nb_positive;
    n->nb_absolute = (intptr_t)n->nb_absolute == (intptr_t)1 ? NULL : n->nb_absolute;
    n->nb_bool = (intptr_t)n->nb_bool == (intptr_t)1 ? NULL : n->nb_bool;
    n->nb_invert = (intptr_t)n->nb_invert == (intptr_t)1 ? NULL : n->nb_invert;
    n->nb_lshift = (intptr_t)n->nb_lshift == (intptr_t)1 ? NULL : n->nb_lshift;
    n->nb_rshift = (intptr_t)n->nb_rshift == (intptr_t)1 ? NULL : n->nb_rshift;
    n->nb_and = (intptr_t)n->nb_and == (intptr_t)1 ? NULL : n->nb_and;
    n->nb_xor = (intptr_t)n->nb_xor == (intptr_t)1 ? NULL : n->nb_xor;
    n->nb_or = (intptr_t)n->nb_or == (intptr_t)1 ? NULL : n->nb_or;
    n->nb_int = (intptr_t)n->nb_int == (intptr_t)1 ? NULL : n->nb_int;
    n->nb_reserved = (intptr_t)n->nb_reserved == (intptr_t)1 ? NULL : n->nb_reserved;
    n->nb_float = (intptr_t)n->nb_float == (intptr_t)1 ? NULL : n->nb_float;
    n->nb_inplace_add = (intptr_t)n->nb_inplace_add == (intptr_t)1 ? NULL : n->nb_inplace_add;
    n->nb_inplace_subtract = (intptr_t)n->nb_inplace_subtract == (intptr_t)1 ? NULL : n->nb_inplace_subtract;
    n->nb_inplace_multiply = (intptr_t)n->nb_inplace_multiply == (intptr_t)1 ? NULL : n->nb_inplace_multiply;
    n->nb_inplace_remainder = (intptr_t)n->nb_inplace_remainder == (intptr_t)1 ? NULL : n->nb_inplace_remainder;
    n->nb_inplace_power = (intptr_t)n->nb_inplace_power == (intptr_t)1 ? NULL : n->nb_inplace_power;
    n->nb_inplace_lshift = (intptr_t)n->nb_inplace_lshift == (intptr_t)1 ? NULL : n->nb_inplace_lshift;
    n->nb_inplace_rshift = (intptr_t)n->nb_inplace_rshift == (intptr_t)1 ? NULL : n->nb_inplace_rshift;
    n->nb_inplace_and = (intptr_t)n->nb_inplace_and == (intptr_t)1 ? NULL : n->nb_inplace_and;
    n->nb_inplace_xor = (intptr_t)n->nb_inplace_xor == (intptr_t)1 ? NULL : n->nb_inplace_xor;
    n->nb_inplace_or = (intptr_t)n->nb_inplace_or == (intptr_t)1 ? NULL : n->nb_inplace_or;
    n->nb_floor_divide = (intptr_t)n->nb_floor_divide == (intptr_t)1 ? NULL : n->nb_floor_divide;
    n->nb_true_divide = (intptr_t)n->nb_true_divide == (intptr_t)1 ? NULL : n->nb_true_divide;
    n->nb_inplace_floor_divide = (intptr_t)n->nb_inplace_floor_divide == (intptr_t)1 ? NULL : n->nb_inplace_floor_divide;
    n->nb_inplace_true_divide = (intptr_t)n->nb_inplace_true_divide == (intptr_t)1 ? NULL : n->nb_inplace_true_divide;
    n->nb_index = (intptr_t)n->nb_index == (intptr_t)1 ? NULL : n->nb_index;
    n->nb_matrix_multiply = (intptr_t)n->nb_matrix_multiply == (intptr_t)1 ? NULL : n->nb_matrix_multiply;
    n->nb_inplace_matrix_multiply = (intptr_t)n->nb_inplace_matrix_multiply == (intptr_t)1 ? NULL : n->nb_inplace_matrix_multiply;

    PySequenceMethods *sq = type->tp_as_sequence;
    sq->sq_length = (intptr_t)sq->sq_length == (intptr_t)1 ? NULL : sq->sq_length;
    sq->sq_concat = (intptr_t)sq->sq_concat == (intptr_t)1 ? NULL : sq->sq_concat;
    sq->sq_repeat = (intptr_t)sq->sq_repeat == (intptr_t)1 ? NULL : sq->sq_repeat;
    sq->sq_item = (intptr_t)sq->sq_item == (intptr_t)1 ? NULL : sq->sq_item;
    sq->was_sq_slice = (intptr_t)sq->was_sq_slice == (intptr_t)1 ? NULL : sq->was_sq_slice;
    sq->sq_ass_item = (intptr_t)sq->sq_ass_item == (intptr_t)1 ? NULL : sq->sq_ass_item;
    sq->was_sq_ass_slice = (intptr_t)sq->was_sq_ass_slice == (intptr_t)1 ? NULL : sq->was_sq_ass_slice;
    sq->sq_contains = (intptr_t)sq->sq_contains == (intptr_t)1 ? NULL : sq->sq_contains;
    sq->sq_inplace_concat = (intptr_t)sq->sq_inplace_concat == (intptr_t)1 ? NULL : sq->sq_inplace_concat;
    sq->sq_inplace_repeat = (intptr_t)sq->sq_inplace_repeat == (intptr_t)1 ? NULL : sq->sq_inplace_repeat;

    PyMappingMethods *map = type->tp_as_mapping;
    map->mp_length = (intptr_t)map->mp_length == (intptr_t)1 ? NULL : map->mp_length;
    map->mp_subscript = (intptr_t)map->mp_subscript == (intptr_t)1 ? NULL : map->mp_subscript;
    map->mp_ass_subscript = (intptr_t)map->mp_ass_subscript == (intptr_t)1 ? NULL : map->mp_ass_subscript;

#ifndef GRAALVM_PYTHON
    PyAsyncMethods *as = type->tp_as_async;
    as->am_await = (intptr_t)as->am_await == (intptr_t)1 ? NULL : as->am_await;
    as->am_aiter = (intptr_t)as->am_aiter == (intptr_t)1 ? NULL : as->am_aiter;
    as->am_anext = (intptr_t)as->am_anext == (intptr_t)1 ? NULL : as->am_anext;
    type->tp_as_async = NULL;
#endif

    PyBufferProcs *buf = type->tp_as_buffer;
    buf->bf_getbuffer = (intptr_t)buf->bf_getbuffer == (intptr_t)1 ? NULL : buf->bf_getbuffer;
    buf->bf_releasebuffer = (intptr_t)buf->bf_releasebuffer == (intptr_t)1 ? NULL : buf->bf_releasebuffer;

}

/* Establish scalar-type hierarchy
 *
 *  For dual inheritance we need to make sure that the objects being
 *  inherited from have the tp->mro object initialized.  This is
 *  not necessarily true for the basic type objects of Python (it is
 *  checked for single inheritance but not dual in PyType_Ready).
 *
 *  Thus, we call PyType_Ready on the standard Python Types, here.
 */
static int
setup_scalartypes(HPyContext *ctx)
{
    // HPY TODO: is it really necessary to do this for the builtins,
    // do the comments above this function apply even in HPy case?
    // if (PyType_Ready(&PyBool_Type) < 0) {
    //     return -1;
    // }
    // if (PyType_Ready(&PyFloat_Type) < 0) {
    //     return -1;
    // }
    // if (PyType_Ready(&PyComplex_Type) < 0) {
    //     return -1;
    // }
    // if (PyType_Ready(&PyBytes_Type) < 0) {
    //     return -1;
    // }
    // if (PyType_Ready(&PyUnicode_Type) < 0) {
    //     return -1;
    // }

    // HPY TODO: is HPyTracker good fit for this?, we ignore tracker error
    // handling for now
    HPyTracker tracker = HPyTracker_New(ctx, 40);
    int result = -1;

#define SINGLE_INHERIT(child, parent)                                   \
    HPyType_SpecParam child##_params[] = {                              \
        { HPyType_SpecParam_Base, h_Py##parent##ArrType_Type },         \
        { 0 },                                                          \
    };                                                                  \
    HPy h_Py##child##ArrType_Type =                                     \
        HPyType_FromSpec(ctx, &Py##child##ArrType_Type_spec, child##_params); \
    if (HPy_IsNull(h_Py##child##ArrType_Type)) {                        \
        /* HPY TODO: PyErr_Print();*/                                   \
        /* HPY TODO: PyErr_Format(PyExc_SystemError,*/                  \
                     /*"could not initialize Py%sArrType_Type",*/       \
                     /*#child);*/                                       \
        char msgbuf[255];                                               \
        snprintf(msgbuf, 255,                                           \
            "could not initialize Py%sArrType_Type",                    \
            #child);                                                    \
        /*HPyErr_SetString(ctx, ctx->h_SystemError, msgbuf);*/          \
        goto cleanup;                                                   \
    }                                                                   \
    HPyTracker_Add(ctx, tracker, h_Py##child##ArrType_Type);            \
    HPyGlobal_Store(ctx, &HPy##child##ArrType_Type,                     \
        h_Py##child##ArrType_Type);                                     \
    _Py##child##ArrType_Type_p =                                        \
        (PyTypeObject*) HPy_AsPyObject(ctx, h_Py##child##ArrType_Type);

    HPy h_PyGenericArrType_Type = HPyType_FromSpec(ctx, &PyGenericArrType_Type_spec, NULL);
    if (HPy_IsNull(h_PyGenericArrType_Type)) {
        goto cleanup;
    }
    // HPY TODO: global variable + local variable to mimick the original global in the SINGLE_INHERIT&co macros
    _PyGenericArrType_Type_p = (PyTypeObject*) HPy_AsPyObject(ctx, h_PyGenericArrType_Type);
    HPyTracker_Add(ctx, tracker, h_PyGenericArrType_Type);
    HPyGlobal_Store(ctx, &HPyGenericArrType_Type, h_PyGenericArrType_Type);

    SINGLE_INHERIT(Number, Generic);
    SINGLE_INHERIT(Integer, Number);
    SINGLE_INHERIT(Inexact, Number);
    SINGLE_INHERIT(SignedInteger, Integer);
    SINGLE_INHERIT(UnsignedInteger, Integer);
    SINGLE_INHERIT(Floating, Inexact);
    SINGLE_INHERIT(ComplexFloating, Inexact);
    SINGLE_INHERIT(Flexible, Generic);
    SINGLE_INHERIT(Character, Flexible);

#define DUAL_INHERIT(child, parent1, parent2)                           \
    HPyType_SpecParam child##_params[] = {                              \
        { HPyType_SpecParam_Base, h_Py##parent2##ArrType_Type  },          \
        { HPyType_SpecParam_Base, ctx->h_##parent1##Type       },          \
        { 0 },                                                          \
    };                                                                  \
    /* HPY TODO: Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;*/ \
    HPy h_Py##child##ArrType_Type =                                     \
        HPyType_FromSpec(ctx, &Py##child##ArrType_Type_spec, child##_params); \
    if (HPy_IsNull(h_Py##child##ArrType_Type)) {                        \
        /* HPY TODO: PyErr_Print();*/                                   \
        /* HPY TODO: PyErr_Format(PyExc_SystemError,*/                  \
                     /*"could not initialize Py%sArrType_Type",*/       \
                     /*#child);*/                                       \
        char msgbuf[255];                                               \
        snprintf(msgbuf, 255,                                          \
            "could not initialize Py%sArrType_Type",                    \
            #child);                                                    \
        HPyErr_SetString(ctx, ctx->h_SystemError, msgbuf);              \
        goto cleanup;                                                   \
    }                                                                   \
    HPyTracker_Add(ctx, tracker, h_Py##child##ArrType_Type);            \
    HPyGlobal_Store(ctx, &HPy##child##ArrType_Type,                     \
        h_Py##child##ArrType_Type);                                     \
    _Py##child##ArrType_Type_p =                                        \
        (PyTypeObject*) HPy_AsPyObject(ctx, h_Py##child##ArrType_Type);

#ifndef GRAALVM_PYTHON
    PyAsyncMethods *tmp_PyAsyncMethods = NULL;
    PyNumberMethods *tmp_PyNumberMethods = NULL;
    PyMappingMethods *tmp_PyMappingMethods = NULL;
    PySequenceMethods *tmp_PySequenceMethods = NULL;
    PyBufferProcs *tmp_PyBufferProcs = NULL;
#define DUAL_INHERIT2(child, parent1, parent2)                          \
    tmp_PyAsyncMethods = _Py##parent2##ArrType_Type_p->tp_as_async;        \
    tmp_PyNumberMethods = _Py##parent2##ArrType_Type_p->tp_as_number;      \
    tmp_PyMappingMethods = _Py##parent2##ArrType_Type_p->tp_as_mapping;        \
    tmp_PySequenceMethods = _Py##parent2##ArrType_Type_p->tp_as_sequence;      \
    tmp_PyBufferProcs = _Py##parent2##ArrType_Type_p->tp_as_buffer;      \
    set_as_func(_Py##parent2##ArrType_Type_p); \
    HPyType_SpecParam child##_params[] = {                              \
        { HPyType_SpecParam_Base, ctx->h_##parent1##Type },             \
        { HPyType_SpecParam_Base, h_Py##parent2##ArrType_Type },        \
        { 0 },                                                          \
    };                                                                  \
    /* HPY TODO: Py##child##ArrType_Type.tp_richcompare = */            \
        /*Py##parent1##_Type.tp_richcompare;*/                          \
    /*Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;*/   \
    HPy h_Py##child##ArrType_Type =                                     \
        HPyType_FromSpec(ctx, &Py##child##ArrType_Type_spec, child##_params); \
    _Py##parent2##ArrType_Type_p->tp_as_async = tmp_PyAsyncMethods;        \
    _Py##parent2##ArrType_Type_p->tp_as_number = tmp_PyNumberMethods;      \
    _Py##parent2##ArrType_Type_p->tp_as_mapping = tmp_PyMappingMethods;        \
    _Py##parent2##ArrType_Type_p->tp_as_sequence = tmp_PySequenceMethods;      \
    _Py##parent2##ArrType_Type_p->tp_as_buffer = tmp_PyBufferProcs;      \
    if (HPy_IsNull(h_Py##child##ArrType_Type)) {                        \
        /* HPY TODO: PyErr_Print();*/                                   \
        /* HPY TODO: PyErr_Format(PyExc_SystemError,*/                  \
                     /*"could not initialize Py%sArrType_Type",*/       \
                     /*#child);*/                                       \
        char msgbuf[255];                                               \
        snprintf(msgbuf, 255,                                           \
            "could not initialize Py%sArrType_Type",                    \
            #child);                                                    \
        HPyErr_SetString(ctx, ctx->h_SystemError, msgbuf);              \
        goto cleanup;                                                   \
    }                                                                   \
    HPyTracker_Add(ctx, tracker, h_Py##child##ArrType_Type);            \
    HPyGlobal_Store(ctx, &HPy##child##ArrType_Type,                     \
        h_Py##child##ArrType_Type);                                     \
    _Py##child##ArrType_Type_p =                                        \
        (PyTypeObject*) HPy_AsPyObject(ctx, h_Py##child##ArrType_Type); \
    cleanup(_Py##child##ArrType_Type_p);
#else
#define DUAL_INHERIT2(child, parent1, parent2)                          \
    HPyType_SpecParam child##_params[] = {                              \
        { HPyType_SpecParam_Base, ctx->h_##parent1##Type },             \
        { HPyType_SpecParam_Base, h_Py##parent2##ArrType_Type },        \
        { 0 },                                                          \
    };                                                                  \
    /* HPY TODO: Py##child##ArrType_Type.tp_richcompare = */            \
        /*Py##parent1##_Type.tp_richcompare;*/                          \
    /*Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;*/   \
    HPy h_Py##child##ArrType_Type =                                     \
        HPyType_FromSpec(ctx, &Py##child##ArrType_Type_spec, child##_params); \
    if (HPy_IsNull(h_Py##child##ArrType_Type)) {                        \
        /* HPY TODO: PyErr_Print();*/                                   \
        /* HPY TODO: PyErr_Format(PyExc_SystemError,*/                  \
                     /*"could not initialize Py%sArrType_Type",*/       \
                     /*#child);*/                                       \
        char msgbuf[255];                                               \
        snprintf(msgbuf, 255,                                           \
            "could not initialize Py%sArrType_Type",                    \
            #child);                                                    \
        HPyErr_SetString(ctx, ctx->h_SystemError, msgbuf);              \
        goto cleanup;                                                   \
    }                                                                   \
    HPyTracker_Add(ctx, tracker, h_Py##child##ArrType_Type);            \
    HPyGlobal_Store(ctx, &HPy##child##ArrType_Type,                     \
        h_Py##child##ArrType_Type);                                     \
    _Py##child##ArrType_Type_p =                                        \
        (PyTypeObject*) HPy_AsPyObject(ctx, h_Py##child##ArrType_Type); 
#endif

    SINGLE_INHERIT(Bool, Generic);
    SINGLE_INHERIT(Byte, SignedInteger);
    SINGLE_INHERIT(Short, SignedInteger);
    SINGLE_INHERIT(Int, SignedInteger);
    SINGLE_INHERIT(Long, SignedInteger);
    SINGLE_INHERIT(LongLong, SignedInteger);

    /* Datetime doesn't fit in any category */
    SINGLE_INHERIT(Datetime, Generic);
    /* Timedelta is an integer with an associated unit */
    SINGLE_INHERIT(Timedelta, SignedInteger);

    SINGLE_INHERIT(UByte, UnsignedInteger);
    SINGLE_INHERIT(UShort, UnsignedInteger);
    SINGLE_INHERIT(UInt, UnsignedInteger);
    SINGLE_INHERIT(ULong, UnsignedInteger);
    SINGLE_INHERIT(ULongLong, UnsignedInteger);

    SINGLE_INHERIT(Half, Floating);
    SINGLE_INHERIT(Float, Floating);
    DUAL_INHERIT(Double, Float, Floating);
    SINGLE_INHERIT(LongDouble, Floating);

    SINGLE_INHERIT(CFloat, ComplexFloating);
    DUAL_INHERIT(CDouble, Complex, ComplexFloating);
    SINGLE_INHERIT(CLongDouble, ComplexFloating);

    DUAL_INHERIT2(String, Bytes, Character);
    DUAL_INHERIT2(Unicode, Unicode, Character);

    SINGLE_INHERIT(Void, Flexible);

    SINGLE_INHERIT(Object, Generic);

    result = 0;
cleanup:
    HPyTracker_Close(ctx, tracker);
    return result;

#undef SINGLE_INHERIT
#undef DUAL_INHERIT
#undef DUAL_INHERIT2

    /*
     * Clean up string and unicode array types so they act more like
     * strings -- get their tables from the standard types.
     */
}

/* place a flag dictionary in d */

static void
set_flaginfo(HPyContext *ctx, HPy d)
{
    HPy s;
    HPy newd = HPyDict_New(ctx);

#define _addnew(key, val, one)                                \
    HPy_SetItem_s(ctx, newd, #key, s=HPyLong_FromLong(ctx, val));    \
    HPy_Close(ctx, s);                                        \
    HPy_SetItem_s(ctx, newd, #one, s=HPyLong_FromLong(ctx, val));    \
    HPy_Close(ctx, s)

#define _addone(key, val)                                  \
    HPy_SetItem_s(ctx, newd, #key, s=HPyLong_FromLong(ctx, val));    \
    HPy_Close(ctx, s)

    _addnew(OWNDATA, NPY_ARRAY_OWNDATA, O);
    _addnew(FORTRAN, NPY_ARRAY_F_CONTIGUOUS, F);
    _addnew(CONTIGUOUS, NPY_ARRAY_C_CONTIGUOUS, C);
    _addnew(ALIGNED, NPY_ARRAY_ALIGNED, A);
    _addnew(WRITEBACKIFCOPY, NPY_ARRAY_WRITEBACKIFCOPY, X);
    _addnew(WRITEABLE, NPY_ARRAY_WRITEABLE, W);
    _addone(C_CONTIGUOUS, NPY_ARRAY_C_CONTIGUOUS);
    _addone(F_CONTIGUOUS, NPY_ARRAY_F_CONTIGUOUS);

#undef _addone
#undef _addnew

    HPy_SetItem_s(ctx, d, "_flagdict", newd);
    HPy_Close(ctx, newd);
    return;
}

NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_array_wrap;
NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_array_finalize;
NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_implementation;
NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_axis1;
NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_axis2;
NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_like;
NPY_VISIBILITY_HIDDEN HPyGlobal npy_ma_str_numpy;

static int
intern_strings(HPyContext *ctx)
{
    HPy h_str;
    #define INTERN(global, str)                         \
        h_str = HPyUnicode_InternFromString(ctx, str);  \
        if (HPy_IsNull(h_str)) return 1;                \
        HPyGlobal_Store(ctx, &global, h_str);           \
        HPy_Close(ctx, h_str);

    INTERN(npy_ma_str_array_wrap, "__array_wrap__");
    INTERN(npy_ma_str_array_finalize, "__array_finalize__");
    INTERN(npy_ma_str_implementation, "_implementation");
    INTERN(npy_ma_str_axis1, "axis1");
    INTERN(npy_ma_str_axis2, "axis2");
    INTERN(npy_ma_str_like, "like");
    INTERN(npy_ma_str_numpy, "numpy");
    return 0;
}

static HPyDef *array_module_hpy_methods[] = {
    &array_zeros,
    &array_array,
    &array_promote_types,
    &array_min_scalar_type,
    &_get_castingimpl,
    &NpyIter_NestedIters,
    &get_sfloat_dtype,
    &hpy_add_docstring,
    &implement_array_function,
    &_get_implementing_args,
    &_get_ndarray_c_version,
    &_fastCopyAndTranspose,
    &_from_dlpack,
    &_insert,
    &_reconstruct,
    &_monotonicity,
    &_set_madvise_hugepage,
    &array_scalar,
    &array_putmask,
    &array_asarray,
    &array_asanyarray,
    &array_ascontiguousarray,
    &array_asfortranarray,
    &array_copyto,
    &array_empty,
    &array_empty_like,
    &array_scalar,
    &array_fromstring,
    &array_fromfile,
    &array_fromiter,
    &array_frombuffer,
    &array_concatenate,
    &array_innerproduct,
    &array_matrixproduct,
    &array_vdot,
    &array_arange,
    &array_set_ops_function,
    &array_where,
    &array_lexsort,
    &array_can_cast_safely,
    &array_result_type,
    &array_datetime_data,
    &array_shares_memory,
    &array_may_share_memory,
    &datetime_as_string,
    &ufunc_geterr,
    &ufunc_seterr,
    &array_is_busday,
    &array_busday_offset,
    &array_busday_count,
    &arr_bincount,
    &arr_ravel_multi_index,
    &arr_unravel_index,
    &io_pack,
    &io_unpack,
    &frompyfunc,
    &add_newdoc_ufunc,
    &array_set_typeDict,
    &compare_chararrays,
    &normalize_axis_index,
    &dragon4_positional,
    &dragon4_scientific,
    &format_longfloat,
    &set_legacy_print_mode,
    &array_einsum,
    &_load_from_filelike,
    &arr_interp,
    &arr_interp_complex,

    // HPy Port TODO: implement them.
    &_hpy_vec_string,
    NULL
};

static HPyGlobal *module_globals[] = {
    &HPyArrayDescr_Type,
    &HPyArray_Type,
    &HPyArrayDTypeMeta_Type,
    &HPyArrayMethod_Type,
    &HPyBoundArrayMethod_Type,
    &HPyGenericArrType_Type,

    // array_coercion:
    &_global_pytype_to_type_dict,

    // scalartypes:
    &HPyGenericArrType_Type,
    &HPyGenericArrType_Type,
    &HPyBoolArrType_Type,
    &HPyNumberArrType_Type,
    &HPyIntegerArrType_Type,
    &HPySignedIntegerArrType_Type,
    &HPyUnsignedIntegerArrType_Type,
    &HPyInexactArrType_Type,
    &HPyFloatingArrType_Type,
    &HPyComplexFloatingArrType_Type,
    &HPyFlexibleArrType_Type,
    &HPyCharacterArrType_Type,
    &HPyByteArrType_Type,
    &HPyShortArrType_Type,
    &HPyIntArrType_Type,
    &HPyLongArrType_Type,
    &HPyLongLongArrType_Type,
    &HPyUByteArrType_Type,
    &HPyUShortArrType_Type,
    &HPyUIntArrType_Type,
    &HPyULongArrType_Type,
    &HPyULongLongArrType_Type,
    &HPyFloatArrType_Type,
    &HPyDoubleArrType_Type,
    &HPyLongDoubleArrType_Type,
    &HPyCFloatArrType_Type,
    &HPyCDoubleArrType_Type,
    &HPyCLongDoubleArrType_Type,
    &HPyObjectArrType_Type,
    &HPyStringArrType_Type,
    &HPyUnicodeArrType_Type,
    &HPyVoidArrType_Type,
    &HPyDatetimeArrType_Type,
    &HPyTimedeltaArrType_Type,
    &HPyHalfArrType_Type,

    &_HPyArrayScalar_BoolValues[0],
    &_HPyArrayScalar_BoolValues[1],

    // arraytypes
    &_hpy_builtin_descrs[NPY_VOID],
    &_hpy_builtin_descrs[NPY_STRING],
    &_hpy_builtin_descrs[NPY_UNICODE],
    &_hpy_builtin_descrs[NPY_BOOL],
    &_hpy_builtin_descrs[NPY_BYTE],
    &_hpy_builtin_descrs[NPY_UBYTE],
    &_hpy_builtin_descrs[NPY_SHORT],
    &_hpy_builtin_descrs[NPY_USHORT],
    &_hpy_builtin_descrs[NPY_INT],
    &_hpy_builtin_descrs[NPY_UINT],
    &_hpy_builtin_descrs[NPY_LONG],
    &_hpy_builtin_descrs[NPY_ULONG],
    &_hpy_builtin_descrs[NPY_LONGLONG],
    &_hpy_builtin_descrs[NPY_ULONGLONG],
    &_hpy_builtin_descrs[NPY_HALF],
    &_hpy_builtin_descrs[NPY_FLOAT],
    &_hpy_builtin_descrs[NPY_DOUBLE],
    &_hpy_builtin_descrs[NPY_LONGDOUBLE],
    &_hpy_builtin_descrs[NPY_CFLOAT],
    &_hpy_builtin_descrs[NPY_CDOUBLE],
    &_hpy_builtin_descrs[NPY_CLONGDOUBLE],
    &_hpy_builtin_descrs[NPY_OBJECT],
    &_hpy_builtin_descrs[NPY_DATETIME],
    &_hpy_builtin_descrs[NPY_TIMEDELTA],

    // abstract dtypes
    &HPyArray_PyIntAbstractDType,
    &HPyArray_PyFloatAbstractDType,
    &HPyArray_PyComplexAbstractDType,

    // hpy_n_ops struct/cache:
    &hpy_n_ops.add,
    &hpy_n_ops.subtract,
    &hpy_n_ops.multiply,
    &hpy_n_ops.divide,
    &hpy_n_ops.remainder,
    &hpy_n_ops.divmod,
    &hpy_n_ops.power,
    &hpy_n_ops.square,
    &hpy_n_ops.reciprocal,
    &hpy_n_ops._ones_like,
    &hpy_n_ops.sqrt,
    &hpy_n_ops.cbrt,
    &hpy_n_ops.negative,
    &hpy_n_ops.positive,
    &hpy_n_ops.absolute,
    &hpy_n_ops.invert,
    &hpy_n_ops.left_shift,
    &hpy_n_ops.right_shift,
    &hpy_n_ops.bitwise_and,
    &hpy_n_ops.bitwise_or,
    &hpy_n_ops.bitwise_xor,
    &hpy_n_ops.less,
    &hpy_n_ops.less_equal,
    &hpy_n_ops.equal,
    &hpy_n_ops.not_equal,
    &hpy_n_ops.greater,
    &hpy_n_ops.greater_equal,
    &hpy_n_ops.floor_divide,
    &hpy_n_ops.true_divide,
    &hpy_n_ops.logical_or,
    &hpy_n_ops.logical_and,
    &hpy_n_ops.floor,
    &hpy_n_ops.ceil,
    &hpy_n_ops.maximum,
    &hpy_n_ops.minimum,
    &hpy_n_ops.rint,
    &hpy_n_ops.conjugate,
    &hpy_n_ops.matmul,
    &hpy_n_ops.clip,
    &current_handler,

    // interned strings
    &npy_ma_str_array_wrap,
    &npy_ma_str_array_finalize,
    &npy_ma_str_implementation,
    &npy_ma_str_axis1,
    &npy_ma_str_axis2,
    &npy_ma_str_like,
    &npy_ma_str_numpy,

    &g_dummy_arr,

    &HPyArray_SFloatDType,
    &SFloatSingleton,

    &descr_typeDict,
    &g_checkfunc,
    &g_AxisError_cls,
    NULL
};

static HPyModuleDef moduledef = {
    /* HPY TODO: Unclear if a dotted name is legit in .m_name, but universal
     * mode requires it.
     */
    .name = "numpy.core._multiarray_umath",
    .doc = NULL,
    .size = -1,
#ifndef GRAALVM_PYTHON
    .legacy_methods = array_module_methods,
#endif
    .defines = array_module_hpy_methods,
    .globals = module_globals,
};

/* Initialization function for the module */
HPy_MODINIT(_multiarray_umath)
static HPy init__multiarray_umath_impl(HPyContext *ctx) {
    HPy h_mod, h_d = HPy_NULL, h_s;
    HPy result = HPy_NULL;
    HPy h_array_type = HPy_NULL;
    HPy h_arrayIterType = HPy_NULL;
    HPy h_npyiter_type = HPy_NULL;
    HPy h_arrayMultiIter_type = HPy_NULL;
    HPy h_PyArrayDescr_Type = HPy_NULL;
    HPy h_arrayFlagsType = HPy_NULL;
    HPy local_PyDataMem_DefaultHandler = HPy_NULL;

    /* Create the module and add the functions */
    h_mod = HPyModule_Create(ctx, &moduledef);
    if (HPy_IsNull(h_mod)) {
        return HPy_NULL;
    }

    /* Initialize CPU features */
    if (npy_cpu_init() < 0) {
        goto err;
    }

#if defined(MS_WIN64) && defined(__GNUC__)
  PyErr_WarnEx(PyExc_Warning,
        "Numpy built with MINGW-W64 on Windows 64 bits is experimental, " \
        "and only available for \n" \
        "testing. You are advised not to use it for production. \n\n" \
        "CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO NUMPY DEVELOPERS",
        1);
#endif

    /* Initialize access to the PyDateTime API */
    numpy_pydatetime_import();

    if (HPyErr_Occurred(ctx)) {
        goto err;
    }

    /* Add some symbolic constants to the module */
    h_d = HPy_GetAttr_s(ctx, h_mod, "__dict__");
    if (HPy_IsNull(h_d)) {
        goto err;
    }

    // HPY HACK:
    /* Store the context so legacy functions and extensions can access it */
    assert(numpy_global_ctx == NULL);
    numpy_global_ctx = ctx;
    h_s = HPyCapsule_New(ctx, (void *)ctx, NULL, NULL);
    if (HPy_IsNull(h_s)) {
        goto err;
    }
    HPy_SetItem_s(ctx, h_d, "_HPY_CONTEXT", h_s);
    HPy_Close(ctx, h_s);


    HPy h_PyUFunc_Type = HPyType_FromSpec(ctx, &PyUFunc_Type_Spec, NULL);
    if (HPy_IsNull(h_PyUFunc_Type)) {
        goto err;
    }
    _PyUFunc_Type_p = (PyTypeObject*) HPy_AsPyObject(ctx, h_PyUFunc_Type);
    HPyGlobal_Store(ctx, &HPyUFunc_Type, h_PyUFunc_Type);
    HPy_Close(ctx, h_PyUFunc_Type);

    HPyType_SpecParam dtypemeta_params[] = {
        { HPyType_SpecParam_Base, ctx->h_TypeType },
        { 0 }
    };
    HPy h_PyArrayDTypeMeta_Type = HPyType_FromSpec(ctx, &PyArrayDTypeMeta_Type_spec, dtypemeta_params);
    if (HPy_IsNull(h_PyArrayDTypeMeta_Type)) {
        goto err;
    }

    HPyType_SpecParam dtype_params[] = {
        { HPyType_SpecParam_Metaclass, h_PyArrayDTypeMeta_Type },
        { 0 }
    };
    h_PyArrayDescr_Type = HPyType_FromSpec(ctx, &PyArrayDescr_TypeFull_spec, dtype_params);
    if (HPy_IsNull(h_PyArrayDescr_Type)) {
        goto err;
    }
    // HPY note: we initialize to the same as the previous static initializer
    PyArray_DTypeMeta *pyarry_descr_data = PyArray_DTypeMeta_AsStruct(ctx, h_PyArrayDescr_Type);
    pyarry_descr_data->type_num = -1;
    pyarry_descr_data->flags = NPY_DT_ABSTRACT;
    pyarry_descr_data->singleton = HPyField_NULL;
    pyarry_descr_data->scalar_type = HPyField_NULL;

    // TODO HPY LABS PORT: storing the types to globals to support legacy code, and HPy code w/o module state
    _PyArrayDescr_Type_p = (PyTypeObject*) HPy_AsPyObject(ctx, h_PyArrayDescr_Type);
    PyArrayDTypeMeta_Type = (PyTypeObject*) HPy_AsPyObject(ctx, h_PyArrayDTypeMeta_Type);

    HPyGlobal_Store(ctx, &HPyArrayDescr_Type, h_PyArrayDescr_Type);
    HPyGlobal_Store(ctx, &HPyArrayDTypeMeta_Type, h_PyArrayDTypeMeta_Type);

    HPy_Close(ctx, h_PyArrayDTypeMeta_Type);

    initialize_casting_tables();
    // initialize_numeric_types();

    if (initscalarmath(NULL) < 0) {
        goto err;
    }

    h_array_type = HPyType_FromSpec(ctx, &PyArray_Type_spec, NULL);
    if (HPy_IsNull(h_array_type)) {
        goto err;
    }
    _PyArray_Type_p = (PyTypeObject*)HPy_AsPyObject(ctx, h_array_type);
    HPyGlobal_Store(ctx, &HPyArray_Type, h_array_type);
    // PyArray_Type.tp_weaklistoffset = offsetof(PyArrayObject_fields, weakreflist); // HPY: needs new API

    if (setup_scalartypes(ctx) < 0) {
        goto err;
    }

    // init for array_coercion.c:
    if (init_global_pytype_to_type_dict(ctx) != 0) {
        goto err;
    }
    
    init_arraytypes_hpy_global_state(ctx);
    // HPY: TODO comment on this
    if (init_scalartypes_basetypes(ctx) != 0) {
        goto err;
    }

    h_arrayIterType = HPyType_FromSpec(ctx, &PyArrayIter_Type_Spec, NULL);
    if (HPy_IsNull(h_arrayIterType)) {
        goto err;
    }
    _PyArrayIter_Type_p = (PyTypeObject*)HPy_AsPyObject(ctx, h_arrayIterType);

    HPy h_arrayMapIterType = HPyType_FromSpec(ctx, &PyArrayMapIter_Type_Spec, NULL);
    if (HPy_IsNull(h_arrayMapIterType)) {
        goto err;
    }
    PyArrayMapIter_Type = (PyTypeObject*)HPy_AsPyObject(ctx, h_arrayMapIterType);

    h_arrayMultiIter_type = HPyType_FromSpec(ctx, &PyArrayMultiIter_Type_Spec, NULL);
    if (HPy_IsNull(h_arrayMultiIter_type)) {
        goto err;
    }
    _PyArrayMultiIter_Type_p = (PyTypeObject*)HPy_AsPyObject(ctx, h_arrayMultiIter_type);

    HPy h_neighborhoodIterType = HPyType_FromSpec(ctx, &PyArrayNeighborhoodIter_Type_Spec, NULL);
    if (HPy_IsNull(h_neighborhoodIterType)) {
        goto err;
    }
    PyArrayNeighborhoodIter_Type = (PyTypeObject*)HPy_AsPyObject(ctx, h_neighborhoodIterType);

    h_npyiter_type = HPyType_FromSpec(ctx, &NpyIter_Type_Spec, NULL);
    if (HPy_IsNull(h_npyiter_type)) {
        goto err;
    }
    _NpyIter_Type_p = (PyTypeObject*)HPy_AsPyObject(ctx, h_npyiter_type);

    h_arrayFlagsType = HPyType_FromSpec(ctx, &PyArrayFlags_Type_Spec, NULL);
    if (HPy_IsNull(h_arrayFlagsType)) {
        goto err;
    }
    HPyGlobal_Store(ctx, &HPyArrayFlags_Type, h_arrayFlagsType);
    _PyArrayFlags_Type_p = (PyTypeObject*)HPy_AsPyObject(ctx, h_arrayFlagsType);

    // Ignored for the HPy example port
    // NpyBusDayCalendar_Type.tp_new = PyType_GenericNew;
    // if (PyType_Ready(&NpyBusDayCalendar_Type) < 0) {
    //     goto err;
    // }

    HPy c_api = HPyCapsule_New(ctx, (void *)PyArray_API, NULL, NULL);
    if (HPy_IsNull(c_api)) {
        goto err;
    }
    HPy_SetItem_s(ctx, h_d, "_ARRAY_API", c_api);
    HPy_Close(ctx, c_api);
    init_array_api();

    c_api = HPyCapsule_New(ctx, (void *)PyUFunc_API, NULL, NULL);
    if (HPy_IsNull(c_api)) {
        goto err;
    }
    HPy_SetItem_s(ctx, h_d, "_UFUNC_API", c_api);
    HPy_Close(ctx, c_api);
    if (HPyErr_Occurred(ctx)) {
        goto err;
    }

    HPy hpy_api = HPyCapsule_New(ctx, (void *)HPyArray_API, NULL, NULL);
    if (HPy_IsNull(hpy_api)) {
        goto err;
    }
    HPy_SetItem_s(ctx, h_d, "_HPY_ARRAY_API", hpy_api);
    HPy_Close(ctx, hpy_api);
    init_hpy_array_api();

    /*
     * PyExc_Exception should catch all the standard errors that are
     * now raised instead of the string exception "multiarray.error"

     * This is for backward compatibility with existing code.
     */
    HPy_SetItem_s(ctx, h_d, "error", ctx->h_Exception);

    h_s = HPyLong_FromLong(ctx, NPY_TRACE_DOMAIN);
    HPy_SetItem_s(ctx, h_d, "tracemalloc_domain", h_s);
    HPy_Close(ctx, h_s);

    h_s = HPyUnicode_FromString(ctx, "3.1");
    HPy_SetItem_s(ctx, h_d, "__version__", h_s);
    HPy_Close(ctx, h_s);

    h_s = npy_cpu_features_dict(ctx);
    if (HPy_IsNull(h_s)) {
        goto err;
    }
    if (HPy_SetItem_s(ctx, h_d, "__cpu_features__", h_s) < 0) {
        HPy_Close(ctx, h_s);
        goto err;
    }
    HPy_Close(ctx, h_s);

    h_s = npy_cpu_baseline_list(ctx);
    if (HPy_IsNull(h_s)) {
        goto err;
    }
    if (HPy_SetItem_s(ctx, h_d, "__cpu_baseline__", h_s) < 0) {
        HPy_Close(ctx, h_s);
        goto err;
    }
    HPy_Close(ctx, h_s);

    h_s = npy_cpu_dispatch_list(ctx);
    if (HPy_IsNull(h_s)) {
        goto err;
    }
    if (HPy_SetItem_s(ctx, h_d, "__cpu_dispatch__", h_s) < 0) {
        HPy_Close(ctx, h_s);
        goto err;
    }
    HPy_Close(ctx, h_s);

    h_s = HPyCapsule_New(ctx, (void *)_datetime_strings, NULL, NULL);
    if (HPy_IsNull(h_s)) {
        goto err;
    }
    HPy_SetItem_s(ctx, h_d, "DATETIMEUNITS", h_s);
    HPy_Close(ctx, h_s);

#define ADDCONST(NAME)                          \
    h_s = HPyLong_FromLong(ctx, NPY_##NAME);    \
    HPy_SetItem_s(ctx, h_d, #NAME, h_s);        \
    HPy_Close(ctx, h_s)


    ADDCONST(ALLOW_THREADS);
    ADDCONST(BUFSIZE);
    ADDCONST(CLIP);

    ADDCONST(ITEM_HASOBJECT);
    ADDCONST(LIST_PICKLE);
    ADDCONST(ITEM_IS_POINTER);
    ADDCONST(NEEDS_INIT);
    ADDCONST(NEEDS_PYAPI);
    ADDCONST(USE_GETITEM);
    ADDCONST(USE_SETITEM);

    ADDCONST(RAISE);
    ADDCONST(WRAP);
    ADDCONST(MAXDIMS);

    ADDCONST(MAY_SHARE_BOUNDS);
    ADDCONST(MAY_SHARE_EXACT);
#undef ADDCONST

    HPy_SetItem_s(ctx, h_d, "ndarray", h_array_type);
    HPy_SetItem_s(ctx, h_d, "flatiter", h_arrayIterType);
    HPy_SetItem_s(ctx, h_d, "nditer", h_npyiter_type);
    HPy_SetItem_s(ctx, h_d, "broadcast", h_arrayMultiIter_type);
    HPy_SetItem_s(ctx, h_d, "dtype", h_PyArrayDescr_Type);
    HPy_SetItem_s(ctx, h_d, "flagsobj", h_arrayFlagsType);

    /* Business day calendar object */
    // Ignored for the HPy example port
    // PyDict_SetItemString(d, "busdaycalendar",
    //                         (PyObject *)&NpyBusDayCalendar_Type);
    set_flaginfo(ctx, h_d);

    /* Create the typeinfo types */
    if (typeinfo_init_structsequences(ctx, h_d) < 0) {
        goto err;
    }

    if (intern_strings(ctx) < 0) {
        goto err;
    }

    if (set_typeinfo(ctx, h_d) != 0) {
        goto err;
    }
    HPy h_array_method_type = HPyType_FromSpec(ctx, &PyArrayMethod_Type_Spec, NULL);
    if (HPy_IsNull(h_array_method_type)) {
        goto err;
    }
    PyArrayMethod_Type = (PyTypeObject*)HPy_AsPyObject(ctx, h_array_method_type);
    HPyGlobal_Store(ctx, &HPyArrayMethod_Type, h_array_method_type);
    HPy_Close(ctx, h_array_method_type);

    HPy h_bound_array_method_type = HPyType_FromSpec(ctx, &PyBoundArrayMethod_Type_Spec, NULL);
    if (HPy_IsNull(h_bound_array_method_type)) {
        goto err;
    }
    PyBoundArrayMethod_Type = (PyTypeObject*)HPy_AsPyObject(ctx, h_bound_array_method_type);
    HPyGlobal_Store(ctx, &HPyBoundArrayMethod_Type, h_bound_array_method_type);
    HPy_Close(ctx, h_bound_array_method_type);

    if (initialize_and_map_pytypes_to_dtypes(ctx) < 0) {
        goto err;
    }

    if (PyArray_InitializeCasts(ctx) < 0) {
        goto err;
    }

    /* Load the ufunc operators into the array module's namespace */
    if (InitOperators(ctx, h_d) < 0) {
        goto err;
    }

    if (set_matmul_flags(ctx, h_d) < 0) {
        goto err;
    }

    if (initumath(ctx, h_mod, h_d) != 0) {
        goto err;
    }
    /*
     * Initialize the default PyDataMem_Handler capsule singleton.
     */
    local_PyDataMem_DefaultHandler = HPyCapsule_New(ctx, &default_handler, "mem_handler", NULL);
    if (HPy_IsNull(local_PyDataMem_DefaultHandler)) {
        goto err;
    }
#if (!defined(PYPY_VERSION_NUM) || PYPY_VERSION_NUM >= 0x07030600)
    /*
     * Initialize the context-local current handler
     * with the default PyDataMem_Handler capsule.
    */
    HPy h_current_handler = HPyContextVar_New(ctx, "current_allocator", local_PyDataMem_DefaultHandler);
    if (HPy_IsNull(h_current_handler)) {
        goto err;
    }
    HPyGlobal_Store(ctx, &current_handler, h_current_handler);
    HPy_Close(ctx, h_current_handler);
#endif

    result = h_mod;

 cleanup:
    HPy_Close(ctx, h_d);
    HPy_Close(ctx, h_array_type);
    HPy_Close(ctx, h_arrayIterType);
    HPy_Close(ctx, h_npyiter_type);
    HPy_Close(ctx, h_arrayMultiIter_type);
    HPy_Close(ctx, h_PyArrayDescr_Type);
    HPy_Close(ctx, h_arrayFlagsType);
    HPy_Close(ctx, local_PyDataMem_DefaultHandler);
    return result;

 err:
    if (!HPyErr_Occurred(ctx)) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                        "cannot load multiarray module.");
    }
    HPy_Close(ctx, h_mod);
    result = HPy_NULL;
    goto cleanup;
}

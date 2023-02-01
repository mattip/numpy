/*
 * This file implements assignment from a scalar to an ndarray.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/ndarraytypes.h>

#include "npy_config.h"
#include "npy_pycompat.h"

#include "convert_datatype.h"
#include "methods.h"
#include "shape.h"
#include "lowlevel_strided_loops.h"

#include "array_assign.h"
#include "dtype_transfer.h"

// HPy includes
#include "multiarraymodule.h"
#include "arrayobject.h"
#include "convert_datatype.h"

/*
 * Assigns the scalar value to every element of the destination raw array.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
hpy_raw_array_assign_scalar(HPyContext *ctx, int ndim, npy_intp const *shape,
        HPy h_dst_dtype, char *dst_data, npy_intp const *dst_strides,
        HPy h_src_dtype, char *src_data)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];
    PyArray_Descr *dst_dtype = PyArray_Descr_AsStruct(ctx, h_dst_dtype);
    PyArray_Descr *src_dtype = PyArray_Descr_AsStruct(ctx, h_src_dtype);

    int aligned, needs_api = 0;

    HPY_NPY_BEGIN_THREADS_DEF;

    /* Check both uint and true alignment */
    aligned = raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   npy_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   dst_dtype->alignment) &&
              npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize) &&
              npy_is_aligned(src_data, src_dtype->alignment));

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareOneRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    if (HPyArray_GetDTypeTransferFunction(ctx, aligned,
                        0, dst_strides_it[0],
                        h_src_dtype, h_dst_dtype,
                        0,
                        &cast_info, &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        HPY_NPY_BEGIN_THREADS_THRESHOLDED(ctx, nitems);
    }

    npy_intp strides[2] = {0, dst_strides_it[0]};

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        char *args[2] = {src_data, dst_data};
        if (cast_info.func(ctx, &cast_info.context,
                args, &shape_it[0], strides, cast_info.auxdata) < 0) {
            goto fail;
        }
    } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord,
                            shape_it, dst_data, dst_strides_it);

    HPY_NPY_END_THREADS(ctx);
    HNPY_cast_info_xfree(ctx, &cast_info);
    return 0;
fail:
    HPY_NPY_END_THREADS(ctx);
    HNPY_cast_info_xfree(ctx, &cast_info);
    return -1;
}

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' value is True.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_scalar(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp const *wheremask_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned, needs_api = 0;

    NPY_BEGIN_THREADS_DEF;

    /* Check both uint and true alignment */
    aligned = raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   npy_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(ndim, shape, dst_data, dst_strides,
                                   dst_dtype->alignment) &&
              npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize) &&
              npy_is_aligned(src_data, src_dtype->alignment));

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        0, dst_strides_it[0], wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &cast_info, &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    npy_intp strides[2] = {0, dst_strides_it[0]};

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        HPyArray_MaskedStridedUnaryOp *stransfer;
        stransfer = (HPyArray_MaskedStridedUnaryOp *)cast_info.func;

        char *args[2] = {src_data, dst_data};
        if (stransfer(npy_get_context(), &cast_info.context,
                args, &shape_it[0], strides,
                (npy_bool *)wheremask_data, wheremask_strides_it[0],
                cast_info.auxdata) < 0) {
            break;
        }
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            wheremask_data, wheremask_strides_it);

    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);
    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * Assigns a scalar value specified by 'src_dtype' and 'src_data'
 * to elements of 'dst'.
 *
 * dst: The destination array.
 * src_dtype: The data type of the source scalar.
 * src_data: The memory element of the source scalar.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the assignment violates this
 *          casting rule.
 *
 * This function is implemented in array_assign_scalar.c.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignRawScalar(PyArrayObject *dst,
                        PyArray_Descr *src_dtype, char *src_data,
                        PyArrayObject *wheremask,
                        NPY_CASTING casting)
{
    HPyContext *ctx = npy_get_context();
    HPy h_dst = HPy_FromPyObject(ctx, (PyObject *)dst);
    HPy h_src_dtype = HPy_FromPyObject(ctx, (PyObject *)src_dtype);
    HPy h_wheremask = HPy_FromPyObject(ctx, (PyObject *)wheremask);
    int res = HPyArray_AssignRawScalar(ctx, h_dst, h_src_dtype, src_data, h_wheremask, casting);
    HPy_Close(ctx, h_wheremask);
    HPy_Close(ctx, h_src_dtype);
    HPy_Close(ctx, h_dst);
    return res;
}

NPY_NO_EXPORT int
HPyArray_AssignRawScalar(HPyContext *ctx, /*PyArrayObject*/HPy h_dst,
                        /*PyArray_Descr*/HPy h_src_dtype, char *src_data,
                        /*PyArrayObject*/HPy h_wheremask,
                        NPY_CASTING casting)
{
    HPy updated_src_dtype = HPy_NULL;
    int allocated_src_data = 0;
    npy_longlong scalarbuffer[4];

    PyArrayObject *dst = PyArrayObject_AsStruct(ctx, h_dst);
    if (HPyArray_FailUnlessWriteableWithStruct(ctx, h_dst, dst, "assignment destination") < 0) {
        return -1;
    }

    PyArray_Descr *src_dtype = PyArray_Descr_AsStruct(ctx, h_src_dtype);
    HPy h_dst_dtype = HPyArray_GetDescr(ctx, h_dst);
    PyArray_Descr *dst_dtype = PyArray_Descr_AsStruct(ctx, h_dst_dtype);

    /* Check the casting rule */
    if (!hpy_can_cast_scalar_to(ctx, h_src_dtype, src_data, h_dst_dtype, casting)) {
        CAPI_WARN("npy_set_invalid_cast_error");
        npy_set_invalid_cast_error(
                src_dtype, dst_dtype, casting, NPY_TRUE);
        return -1;
    }

    /*
     * Make a copy of the src data if it's a different dtype than 'dst'
     * or isn't aligned, and the destination we're copying to has
     * more than one element. To avoid having to manage object lifetimes,
     * we also skip this if 'dst' has an object dtype.
     */
    if ((!HPyArray_EquivTypes(ctx, h_dst_dtype, h_src_dtype) ||
            !(npy_is_aligned(src_data, npy_uint_alignment(src_dtype->elsize)) &&
              npy_is_aligned(src_data, src_dtype->alignment))) &&
                    HPyArray_SIZE(dst) > 1 &&
                    !HPyDataType_REFCHK(ctx, h_dst_dtype)) {
        char *tmp_src_data;

        /*
         * Use a static buffer to store the aligned/cast version,
         * or allocate some memory if more space is needed.
         */
        PyArray_Descr *dst_dtype = PyArray_Descr_AsStruct(ctx, h_dst_dtype);
        if ((int)sizeof(scalarbuffer) >= dst_dtype->elsize) {
            tmp_src_data = (char *)&scalarbuffer[0];
        }
        else {
            CAPI_WARN("PyArray_malloc");
            tmp_src_data = PyArray_malloc(dst_dtype->elsize);
            if (tmp_src_data == NULL) {
                HPyErr_NoMemory(ctx);
                goto fail;
            }
            allocated_src_data = 1;
        }

        if (PyDataType_FLAGCHK(dst_dtype, NPY_NEEDS_INIT)) {
            memset(tmp_src_data, 0, dst_dtype->elsize);
        }

        CAPI_WARN("PyArray_CastRawArrays");
        if (PyArray_CastRawArrays(1, src_data, tmp_src_data, 0, 0,
                            src_dtype, PyArray_DESCR(dst), 0) != NPY_SUCCEED) {
            src_data = tmp_src_data;
            goto fail;
        }

        /* Replace src_data/src_dtype */
        src_data = tmp_src_data;
        /*
         * HPy note: we *MUST NOT* assign to variable 'src_dtype' since that's
         * just the pointer to the native space of the dtype. We need to update
         * 'h_src_dtype'.
         */
        updated_src_dtype = HPyArray_DESCR(ctx, h_dst, dst);
        h_src_dtype = updated_src_dtype;
    }

    if (HPy_IsNull(h_wheremask)) {
        /* A straightforward value assignment */
        /* Do the assignment with raw array iteration */
        if (hpy_raw_array_assign_scalar(ctx, PyArray_NDIM(dst), PyArray_DIMS(dst),
                h_dst_dtype, PyArray_DATA(dst), PyArray_STRIDES(dst),
                h_src_dtype, src_data) < 0) {
            goto fail;
        }
    }
    else {
        CAPI_WARN("non-straightforward value assignment");
        PyArrayObject *wheremask = PyArrayObject_AsStruct(ctx, h_wheremask);
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* Broadcast the wheremask to 'dst' for raw iteration */
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* Do the masked assignment with raw array iteration */
        if (raw_array_wheremasked_assign_scalar(
                PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                src_dtype, src_data,
                PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                wheremask_strides) < 0) {
            goto fail;
        }
    }

    if (allocated_src_data) {
        PyArray_free(src_data);
    }
    HPy_Close(ctx, updated_src_dtype);

    return 0;

fail:
    if (allocated_src_data) {
        PyArray_free(src_data);
    }
    HPy_Close(ctx, updated_src_dtype);

    return -1;
}

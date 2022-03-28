/*
 * This file implements the construction, copying, and destruction
 * aspects of NumPy's nditer.
 *
 * Copyright (c) 2010-2011 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * Copyright (c) 2011 Enthought, Inc
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* Allow this .c file to include nditer_impl.h */
#define NPY_ITERATOR_IMPLEMENTATION_CODE

#include "nditer_impl.h"
#include "arrayobject.h"
#include "array_coercion.h"
#include "templ_common.h"
#include "array_assign.h"
#include "hpy_utils.h"
#include "ndarrayobject.h"
#include "nditer_hpy.h"
#include "ctors.h"

/* Internal helper functions private to this file */
static int
npyiter_check_global_flags(npy_uint32 flags, npy_uint32* itflags);
static int
hnpyiter_check_op_axes(HPyContext *ctx, int nop, int oa_ndim, int **op_axes,
                        const npy_intp *itershape);
static int
hnpyiter_calculate_ndim(HPyContext *ctx, int nop, HPy *op_in,
                       int oa_ndim);
static int
hnpyiter_check_per_op_flags(HPyContext *ctx, npy_uint32 flags, npyiter_opitflags *op_itflags);
static int
hnpyiter_prepare_one_operand(HPyContext *ctx, HPy *op,
                        char **op_dataptr,
                        HPy op_request_dtype,
                        HPy *op_dtype,
                        npy_uint32 flags,
                        npy_uint32 op_flags, npyiter_opitflags *op_itflags);
static int
hnpyiter_prepare_operands(HPyContext *ctx, int nop,
                    HPy *op_in,
                    HPy *op,
                    char **op_dataptr,
                    HPy *op_request_dtypes,
                    HPy *op_dtype,
                    npy_uint32 flags,
                    npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                    npy_int8 *out_maskop);
static int
hnpyiter_check_casting(HPyContext *ctx, int nop, HPy *op,
                    HPy *op_dtype,
                    NPY_CASTING casting,
                    npyiter_opitflags *op_itflags);
static int
hnpyiter_fill_axisdata(HPyContext *ctx, NpyIter *iter, npy_uint32 flags, npyiter_opitflags *op_itflags,
                    char **op_dataptr,
                    const npy_uint32 *op_flags, int **op_axes,
                    npy_intp const *itershape);
static NPY_INLINE int
npyiter_get_op_axis(int axis, npy_bool *reduction_axis);
static void
npyiter_replace_axisdata(
        NpyIter *iter, int iop, PyArrayObject *op,
        int orig_op_ndim, const int *op_axes);
static void
npyiter_compute_index_strides(NpyIter *iter, npy_uint32 flags);
static void
npyiter_apply_forced_iteration_order(HPyContext *ctx, NpyIter *iter, NPY_ORDER order);
static void
npyiter_flip_negative_strides(NpyIter *iter);
static void
npyiter_reverse_axis_ordering(NpyIter *iter);
static void
npyiter_find_best_axis_ordering(NpyIter *iter);
static PyArray_Descr *
npyiter_get_common_dtype(int nop, PyArrayObject **op,
                        const npyiter_opitflags *op_itflags, PyArray_Descr **op_dtype,
                        PyArray_Descr **op_request_dtypes,
                        int only_inputs);
static HPy
hnpyiter_new_temp_array(HPyContext *ctx, NpyIter *iter, HPy subtype,
                npy_uint32 flags, npyiter_opitflags *op_itflags,
                int op_ndim, npy_intp const *shape,
                HPy op_dtype, const int *op_axes);
static int
hnpyiter_allocate_arrays(HPyContext *ctx, NpyIter *iter,
                        npy_uint32 flags,
                        HPy *op_dtype, HPy subtype,
                        const npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                        int **op_axes);
static void
npyiter_get_priority_subtype(int nop, PyArrayObject **op,
                            const npyiter_opitflags *op_itflags,
                            double *subtype_priority, PyTypeObject **subtype);
static int
hnpyiter_allocate_transfer_functions(HPyContext *ctx, NpyIter *iter);


/*NUMPY_API
 * Allocate a new iterator for multiple array objects, and advanced
 * options for controlling the broadcasting, shape, and buffer size.
 */
NPY_NO_EXPORT NpyIter *
NpyIter_AdvancedNew(int nop, PyArrayObject **op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 PyArray_Descr **op_request_dtypes,
                 int oa_ndim, int **op_axes, npy_intp *itershape,
                 npy_intp buffersize)
{
    NpyIter *res;
    HPyContext *ctx = npy_get_context();
    HPy *h_op_in = HPy_FromPyObjectArray(ctx, (PyObject **)op_in, nop);
    HPy *h_op_request_dtypes = HPy_FromPyObjectArray(ctx, (PyObject **)op_request_dtypes, nop);
    
    res = HNpyIter_AdvancedNew(ctx, nop, h_op_in, flags, order, casting, op_flags, h_op_request_dtypes, oa_ndim, op_axes, itershape, buffersize);

    HPy_CloseAndFreeArray(ctx, h_op_in, nop);
    HPy_CloseAndFreeArray(ctx, h_op_request_dtypes, nop);
    
    return res;
}

NPY_NO_EXPORT NpyIter *
HNpyIter_AdvancedNew(HPyContext *ctx, int nop, HPy *op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 HPy *op_request_dtypes,
                 int oa_ndim, int **op_axes, npy_intp *itershape,
                 npy_intp buffersize)
{
    npy_uint32 itflags = NPY_ITFLAG_IDENTPERM;
    int idim, ndim;
    int iop;

    /* The iterator being constructed */
    NpyIter *iter;

    /* Per-operand values */
    HPy *op;
    HPy *op_dtype;
    npyiter_opitflags *op_itflags;
    char **op_dataptr;

    npy_int8 *perm;
    NpyIter_BufferData *bufferdata = NULL;
    int any_allocate = 0, any_missing_dtypes = 0, need_subtype = 0;

    /* The subtype for automatically allocated outputs */
    double subtype_priority = NPY_PRIORITY;

#if NPY_IT_CONSTRUCTION_TIMING
    npy_intp c_temp,
            c_start,
            c_check_op_axes,
            c_check_global_flags,
            c_calculate_ndim,
            c_malloc,
            c_prepare_operands,
            c_fill_axisdata,
            c_compute_index_strides,
            c_apply_forced_iteration_order,
            c_find_best_axis_ordering,
            c_get_priority_subtype,
            c_find_output_common_dtype,
            c_check_casting,
            c_allocate_arrays,
            c_coalesce_axes,
            c_prepare_buffers;
#endif

    NPY_IT_TIME_POINT(c_start);

    if (nop > NPY_MAXARGS) {
        HPyErr_Format_p(ctx, ctx->h_ValueError,
            "Cannot construct an iterator with more than %d operands "
            "(%d were requested)", NPY_MAXARGS, nop);
        return NULL;
    }

    /*
     * Before 1.8, if `oa_ndim == 0`, this meant `op_axes != NULL` was an error.
     * With 1.8, `oa_ndim == -1` takes this role, while op_axes in that case
     * enforces a 0-d iterator. Using `oa_ndim == 0` with `op_axes == NULL`
     * is thus an error in 1.13 after deprecation.
     */
    if ((oa_ndim == 0) && (op_axes == NULL)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
            "Using `oa_ndim == 0` when `op_axes` is NULL. "
            "Use `oa_ndim == -1` or the MultiNew "
            "iterator for NumPy <1.8 compatibility");
        return NULL;
    }

    /* Error check 'oa_ndim' and 'op_axes', which must be used together */
    if (!hnpyiter_check_op_axes(ctx, nop, oa_ndim, op_axes, itershape)) {
        return NULL;
    }

    NPY_IT_TIME_POINT(c_check_op_axes);

    /* Check the global iterator flags */
    if (!npyiter_check_global_flags(flags, &itflags)) {
        return NULL;
    }

    NPY_IT_TIME_POINT(c_check_global_flags);

    /* Calculate how many dimensions the iterator should have */
    ndim = hnpyiter_calculate_ndim(ctx, nop, op_in, oa_ndim);

    NPY_IT_TIME_POINT(c_calculate_ndim);

    /* Allocate memory for the iterator */
    iter = (NpyIter*)
                PyObject_Malloc(NIT_SIZEOF_ITERATOR(itflags, ndim, nop));

    NPY_IT_TIME_POINT(c_malloc);

    /* Fill in the basic data */
    NIT_ITFLAGS(iter) = itflags;
    NIT_NDIM(iter) = ndim;
    NIT_NOP(iter) = nop;
    NIT_MASKOP(iter) = -1;
    NIT_ITERINDEX(iter) = 0;
    memset(NIT_BASEOFFSETS(iter), 0, (nop+1)*NPY_SIZEOF_INTP);

    op = NIT_OPERANDS(iter);
    op_dtype = NIT_DTYPES(iter);
    op_itflags = NIT_OPITFLAGS(iter);
    op_dataptr = NIT_RESETDATAPTR(iter);

    /* Prepare all the operands */
    if (!hnpyiter_prepare_operands(ctx, nop, op_in, op, op_dataptr,
                        op_request_dtypes, op_dtype,
                        flags,
                        op_flags, op_itflags,
                        &NIT_MASKOP(iter))) {
        PyObject_Free(iter);
        return NULL;
    }
    /* Set resetindex to zero as well (it's just after the resetdataptr) */
    op_dataptr[nop] = 0;

    NPY_IT_TIME_POINT(c_prepare_operands);

    /*
     * Initialize buffer data (must set the buffers and transferdata
     * to NULL before we might deallocate the iterator).
     */
    if (itflags & NPY_ITFLAG_BUFFER) {
        bufferdata = NIT_BUFFERDATA(iter);
        NBF_SIZE(bufferdata) = 0;
        memset(NBF_BUFFERS(bufferdata), 0, nop*NPY_SIZEOF_INTP);
        memset(NBF_PTRS(bufferdata), 0, nop*NPY_SIZEOF_INTP);
        /* Ensure that the transferdata/auxdata is NULLed */
        memset(NBF_TRANSFERINFO(bufferdata), 0, nop * sizeof(NpyIter_TransferInfo));
    }

    /* Fill in the AXISDATA arrays and set the ITERSIZE field */
    if (!hnpyiter_fill_axisdata(ctx, iter, flags, op_itflags, op_dataptr,
                                        op_flags, op_axes, itershape)) {
        HNpyIter_Deallocate(ctx, iter);
        return NULL;
    }

    NPY_IT_TIME_POINT(c_fill_axisdata);

    if (itflags & NPY_ITFLAG_BUFFER) {
        /*
         * If buffering is enabled and no buffersize was given, use a default
         * chosen to be big enough to get some amortization benefits, but
         * small enough to be cache-friendly.
         */
        if (buffersize <= 0) {
            buffersize = NPY_BUFSIZE;
        }
        /* No point in a buffer bigger than the iteration size */
        if (buffersize > NIT_ITERSIZE(iter)) {
            buffersize = NIT_ITERSIZE(iter);
        }
        NBF_BUFFERSIZE(bufferdata) = buffersize;

        /*
         * Initialize for use in FirstVisit, which may be called before
         * the buffers are filled and the reduce pos is updated.
         */
        NBF_REDUCE_POS(bufferdata) = 0;
    }

    /*
     * If an index was requested, compute the strides for it.
     * Note that we must do this before changing the order of the
     * axes
     */
    npyiter_compute_index_strides(iter, flags);

    NPY_IT_TIME_POINT(c_compute_index_strides);

    /* Initialize the perm to the identity */
    perm = NIT_PERM(iter);
    for(idim = 0; idim < ndim; ++idim) {
        perm[idim] = (npy_int8)idim;
    }

    /*
     * If an iteration order is being forced, apply it.
     */
    npyiter_apply_forced_iteration_order(ctx, iter, order);
    itflags = NIT_ITFLAGS(iter);

    NPY_IT_TIME_POINT(c_apply_forced_iteration_order);

    /* Set some flags for allocated outputs */
    for (iop = 0; iop < nop; ++iop) {
        if (HPy_IsNull(op[iop])) {
            /* Flag this so later we can avoid flipping axes */
            any_allocate = 1;
            /* If a subtype may be used, indicate so */
            if (!(op_flags[iop] & NPY_ITER_NO_SUBTYPE)) {
                need_subtype = 1;
            }
            /*
             * If the data type wasn't provided, will need to
             * calculate it.
             */
            if (HPy_IsNull(op_dtype[iop])) {
                any_missing_dtypes = 1;
            }
        }
    }

    /*
     * If the ordering was not forced, reorder the axes
     * and flip negative strides to find the best one.
     */
    if (!(itflags & NPY_ITFLAG_FORCEDORDER)) {
        if (ndim > 1) {
            npyiter_find_best_axis_ordering(iter);
        }
        /*
         * If there's an output being allocated, we must not negate
         * any strides.
         */
        if (!any_allocate && !(flags & NPY_ITER_DONT_NEGATE_STRIDES)) {
            npyiter_flip_negative_strides(iter);
        }
        itflags = NIT_ITFLAGS(iter);
    }

    NPY_IT_TIME_POINT(c_find_best_axis_ordering);

    if (need_subtype) {
        /* TODO HPY LABS PORT: cut off */
        hpy_abort_not_implemented("HNpyIter_AdvancedNew");
        /*
        PyTypeObject *subtype = &PyArray_Type;
        npyiter_get_priority_subtype(nop, op, op_itflags,
                                     &subtype_priority, &subtype);
        */
    }

    NPY_IT_TIME_POINT(c_get_priority_subtype);

    /*
     * If an automatically allocated output didn't have a specified
     * dtype, we need to figure it out now, before allocating the outputs.
     */
    if (any_missing_dtypes || (flags & NPY_ITER_COMMON_DTYPE)) {
        /* TODO HPY LABS PORT: cut off */
        hpy_abort_not_implemented("HNpyIter_AdvancedNew");
//        PyArray_Descr *dtype;
//        int only_inputs = !(flags & NPY_ITER_COMMON_DTYPE);
//
//        op = NIT_OPERANDS(iter);
//        op_dtype = NIT_DTYPES(iter);
//
//        dtype = npyiter_get_common_dtype(nop, op,
//                                    op_itflags, op_dtype,
//                                    op_request_dtypes,
//                                    only_inputs);
//        if (dtype == NULL) {
//            NpyIter_Deallocate(iter);
//            return NULL;
//        }
//        if (flags & NPY_ITER_COMMON_DTYPE) {
//            NPY_IT_DBG_PRINT("Iterator: Replacing all data types\n");
//            /* Replace all the data types */
//            for (iop = 0; iop < nop; ++iop) {
//                if (!HPy_Is(ctx, op_dtype[iop], dtype)) {
//                    Py_XDECREF(op_dtype[iop]);
//                    Py_INCREF(dtype);
//                    op_dtype[iop] = dtype;
//                }
//            }
//        }
//        else {
//            NPY_IT_DBG_PRINT("Iterator: Setting unset output data types\n");
//            /* Replace the NULL data types */
//            for (iop = 0; iop < nop; ++iop) {
//                if (op_dtype[iop] == NULL) {
//                    Py_INCREF(dtype);
//                    op_dtype[iop] = dtype;
//                }
//            }
//        }
//        Py_DECREF(dtype);
    }

    NPY_IT_TIME_POINT(c_find_output_common_dtype);

    /*
     * All of the data types have been settled, so it's time
     * to check that data type conversions are following the
     * casting rules.
     */
    if (!hnpyiter_check_casting(ctx, nop, op, op_dtype, casting, op_itflags)) {
        HNpyIter_Deallocate(ctx, iter);
        return NULL;
    }

    NPY_IT_TIME_POINT(c_check_casting);

    /*
     * At this point, the iteration order has been finalized. so
     * any allocation of ops that were NULL, or any temporary
     * copying due to casting/byte order/alignment can be
     * done now using a memory layout matching the iterator.
     */
    HPy h_PyArray_Type = HPyGlobal_Load(ctx, HPyArray_Type);
    if (!hnpyiter_allocate_arrays(ctx, iter, flags, op_dtype, h_PyArray_Type, op_flags,
                            op_itflags, op_axes)) {
        HPy_Close(ctx, h_PyArray_Type);
        HNpyIter_Deallocate(ctx, iter);
        return NULL;
    }
    HPy_Close(ctx, h_PyArray_Type);

    NPY_IT_TIME_POINT(c_allocate_arrays);

    /*
     * Finally, if a multi-index wasn't requested,
     * it may be possible to coalesce some axes together.
     */
    if (ndim > 1 && !(itflags & NPY_ITFLAG_HASMULTIINDEX)) {
        npyiter_coalesce_axes(iter);
        /*
         * The operation may have changed the layout, so we have to
         * get the internal pointers again.
         */
        itflags = NIT_ITFLAGS(iter);
        ndim = NIT_NDIM(iter);
        op = NIT_OPERANDS(iter);
        op_dtype = NIT_DTYPES(iter);
        op_itflags = NIT_OPITFLAGS(iter);
        op_dataptr = NIT_RESETDATAPTR(iter);
    }

    NPY_IT_TIME_POINT(c_coalesce_axes);

    /*
     * Now that the axes are finished, check whether we can apply
     * the single iteration optimization to the iternext function.
     */
    if (!(itflags & NPY_ITFLAG_BUFFER)) {
        NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
        if (itflags & NPY_ITFLAG_EXLOOP) {
            if (NIT_ITERSIZE(iter) == NAD_SHAPE(axisdata)) {
                NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
            }
        }
        else if (NIT_ITERSIZE(iter) == 1) {
            NIT_ITFLAGS(iter) |= NPY_ITFLAG_ONEITERATION;
        }
    }

    /*
     * If REFS_OK was specified, check whether there are any
     * reference arrays and flag it if so.
     *
     * NOTE: This really should be unnecessary, but chances are someone relies
     *       on it.  The iterator itself does not require the API here
     *       as it only does so for casting/buffering.  But in almost all
     *       use-cases the API will be required for whatever operation is done.
     */
    if (flags & NPY_ITER_REFS_OK) {
        for (iop = 0; iop < nop; ++iop) {
            if ((PyArray_Descr_AsStruct(ctx, op_dtype[iop])->flags & (NPY_ITEM_REFCOUNT |
                                     NPY_ITEM_IS_POINTER |
                                     NPY_NEEDS_PYAPI)) != 0) {
                /* Iteration needs API access */
                NIT_ITFLAGS(iter) |= NPY_ITFLAG_NEEDSAPI;
            }
        }
    }

    /* If buffering is set without delayed allocation */
    if (itflags & NPY_ITFLAG_BUFFER) {
        if (!hnpyiter_allocate_transfer_functions(ctx, iter)) {
            HNpyIter_Deallocate(ctx, iter);
            return NULL;
        }
        if (!(itflags & NPY_ITFLAG_DELAYBUF)) {
            /* Allocate the buffers */
            if (!hnpyiter_allocate_buffers(ctx, iter, NULL)) {
                HNpyIter_Deallocate(ctx, iter);
                return NULL;
            }

            /* Prepare the next buffers and set iterend/size */
            if (hnpyiter_copy_to_buffers(ctx, iter, NULL) < 0) {
                HNpyIter_Deallocate(ctx, iter);
                return NULL;
            }
        }
    }

    NPY_IT_TIME_POINT(c_prepare_buffers);

#if NPY_IT_CONSTRUCTION_TIMING
    printf("\nIterator construction timing:\n");
    NPY_IT_PRINT_TIME_START(c_start);
    NPY_IT_PRINT_TIME_VAR(c_check_op_axes);
    NPY_IT_PRINT_TIME_VAR(c_check_global_flags);
    NPY_IT_PRINT_TIME_VAR(c_calculate_ndim);
    NPY_IT_PRINT_TIME_VAR(c_malloc);
    NPY_IT_PRINT_TIME_VAR(c_prepare_operands);
    NPY_IT_PRINT_TIME_VAR(c_fill_axisdata);
    NPY_IT_PRINT_TIME_VAR(c_compute_index_strides);
    NPY_IT_PRINT_TIME_VAR(c_apply_forced_iteration_order);
    NPY_IT_PRINT_TIME_VAR(c_find_best_axis_ordering);
    NPY_IT_PRINT_TIME_VAR(c_get_priority_subtype);
    NPY_IT_PRINT_TIME_VAR(c_find_output_common_dtype);
    NPY_IT_PRINT_TIME_VAR(c_check_casting);
    NPY_IT_PRINT_TIME_VAR(c_allocate_arrays);
    NPY_IT_PRINT_TIME_VAR(c_coalesce_axes);
    NPY_IT_PRINT_TIME_VAR(c_prepare_buffers);
    printf("\n");
#endif

    return iter;
}

/*NUMPY_API
 * Allocate a new iterator for more than one array object, using
 * standard NumPy broadcasting rules and the default buffer size.
 */
NPY_NO_EXPORT NpyIter *
NpyIter_MultiNew(int nop, PyArrayObject **op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 PyArray_Descr **op_request_dtypes)
{
    return NpyIter_AdvancedNew(nop, op_in, flags, order, casting,
                            op_flags, op_request_dtypes,
                            -1, NULL, NULL, 0);
}

/*NUMPY_API
 * Allocate a new iterator for one array object.
 */
NPY_NO_EXPORT NpyIter *
NpyIter_New(PyArrayObject *op, npy_uint32 flags,
                  NPY_ORDER order, NPY_CASTING casting,
                  PyArray_Descr* dtype)
{
    /* Split the flags into separate global and op flags */
    npy_uint32 op_flags = flags & NPY_ITER_PER_OP_FLAGS;
    flags &= NPY_ITER_GLOBAL_FLAGS;

    return NpyIter_AdvancedNew(1, &op, flags, order, casting,
                            &op_flags, &dtype,
                            -1, NULL, NULL, 0);
}

/*NUMPY_API
 * Makes a copy of the iterator
 */
NPY_NO_EXPORT NpyIter *
NpyIter_Copy(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);
    int out_of_memory = 0;

    npy_intp size;
    NpyIter *newiter;
    PyArrayObject **objects;
    PyArray_Descr **dtypes;

    /* Allocate memory for the new iterator */
    size = NIT_SIZEOF_ITERATOR(itflags, ndim, nop);
    newiter = (NpyIter*)PyObject_Malloc(size);

    /* Copy the raw values to the new iterator */
    memcpy(newiter, iter, size);

    /* Take ownership of references to the operands and dtypes */
    objects = NIT_PY_OPERANDS(newiter);
    dtypes = NIT_PY_DTYPES(newiter);
    for (iop = 0; iop < nop; ++iop) {
        Py_INCREF(objects[iop]);
        Py_INCREF(dtypes[iop]);
    }

    /* Allocate buffers and make copies of the transfer data if necessary */
    if (itflags & NPY_ITFLAG_BUFFER) {
        NpyIter_BufferData *bufferdata;
        npy_intp buffersize, itemsize;
        char **buffers;

        bufferdata = NIT_BUFFERDATA(newiter);
        buffers = NBF_BUFFERS(bufferdata);
        buffersize = NBF_BUFFERSIZE(bufferdata);
        NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);

        for (iop = 0; iop < nop; ++iop) {
            if (buffers[iop] != NULL) {
                if (out_of_memory) {
                    buffers[iop] = NULL;
                }
                else {
                    itemsize = dtypes[iop]->elsize;
                    buffers[iop] = PyArray_malloc(itemsize*buffersize);
                    if (buffers[iop] == NULL) {
                        out_of_memory = 1;
                    }
                    else {
                        if (PyDataType_FLAGCHK(dtypes[iop], NPY_NEEDS_INIT)) {
                            memset(buffers[iop], '\0', itemsize*buffersize);
                        }
                    }
                }
            }

            if (transferinfo[iop].read.func != NULL) {
                if (out_of_memory) {
                    transferinfo[iop].read.func = NULL;  /* No cleanup */
                }
                else {
                    if (NPY_cast_info_copy(&transferinfo[iop].read,
                                           &transferinfo[iop].read) < 0) {
                        out_of_memory = 1;
                    }
                }
            }

            if (transferinfo[iop].write.func != NULL) {
                if (out_of_memory) {
                    transferinfo[iop].write.func = NULL;  /* No cleanup */
                }
                else {
                    if (NPY_cast_info_copy(&transferinfo[iop].write,
                                           &transferinfo[iop].write) < 0) {
                        out_of_memory = 1;
                    }
                }
            }
        }

        /* Initialize the buffers to the current iterindex */
        if (!out_of_memory && NBF_SIZE(bufferdata) > 0) {
            npyiter_goto_iterindex(newiter, NIT_ITERINDEX(newiter));

            /* Prepare the next buffers and set iterend/size */
            npyiter_copy_to_buffers(newiter, NULL);
        }
    }

    if (out_of_memory) {
        NpyIter_Deallocate(newiter);
        PyErr_NoMemory();
        return NULL;
    }

    return newiter;
}

/*NUMPY_API
 * Deallocate an iterator.
 *
 * To correctly work when an error is in progress, we have to check
 * `PyErr_Occurred()`. This is necessary when buffers are not finalized
 * or WritebackIfCopy is used. We could avoid that check by exposing a new
 * function which is passed in whether or not a Python error is already set.
 */
NPY_NO_EXPORT int
NpyIter_Deallocate(NpyIter *iter)
{
    return HNpyIter_Deallocate(npy_get_context(), iter);
}

NPY_NO_EXPORT int
HNpyIter_Deallocate(HPyContext *ctx, NpyIter *iter)
{
    int success = !HPyErr_Occurred(ctx);

    npy_uint32 itflags;
    /*int ndim = NIT_NDIM(iter);*/
    int iop, nop;
    // PyArray_Descr **dtype;
    // PyArrayObject **object;
    HPy *dtype;
    HPy *object;
    npyiter_opitflags *op_itflags;

    if (iter == NULL) {
        return success;
    }

    itflags = NIT_ITFLAGS(iter);
    nop = NIT_NOP(iter);
    dtype = NIT_DTYPES(iter);
    object = NIT_OPERANDS(iter);
    op_itflags = NIT_OPITFLAGS(iter);

    /* Deallocate any buffers and buffering data */
    if (itflags & NPY_ITFLAG_BUFFER) {
        /* Ensure no data is held by the buffers before they are cleared */
        if (success) {
            if (hnpyiter_copy_from_buffers(ctx, iter) < 0) {
                success = NPY_FAIL;
            }
        }
        else {
            hnpyiter_clear_buffers(ctx, iter);
        }

        NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
        char **buffers;

        /* buffers */
        buffers = NBF_BUFFERS(bufferdata);
        for (iop = 0; iop < nop; ++iop, ++buffers) {
            PyArray_free(*buffers);
        }

        NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);
        /* read bufferdata */
        for (iop = 0; iop < nop; ++iop, ++transferinfo) {
            NPY_cast_info_xfree(&transferinfo->read);
            NPY_cast_info_xfree(&transferinfo->write);
        }
    }

    /*
     * Deallocate all the dtypes and objects that were iterated and resolve
     * any writeback buffers created by the iterator.
     */
    for (iop = 0; iop < nop; ++iop, ++dtype, ++object) {
        if (op_itflags[iop] & NPY_OP_ITFLAG_HAS_WRITEBACK) {
            if (success && HPyArray_ResolveWritebackIfCopy(ctx, *object) < 0) {
                success = 0;
            }
            else {
                HPyArray_DiscardWritebackIfCopy(ctx, *object);
            }
        }
        HPy_Close(ctx, *dtype);
        HPy_Close(ctx, *object);
    }

    /* Deallocate the iterator memory */
    PyObject_Free(iter);
    return success;
}


/* Checks 'flags' for (C|F)_ORDER_INDEX, MULTI_INDEX, and EXTERNAL_LOOP,
 * setting the appropriate internal flags in 'itflags'.
 *
 * Returns 1 on success, 0 on error.
 */
static int
npyiter_check_global_flags(npy_uint32 flags, npy_uint32* itflags)
{
    if ((flags & NPY_ITER_PER_OP_FLAGS) != 0) {
        PyErr_SetString(PyExc_ValueError,
                    "A per-operand flag was passed as a global flag "
                    "to the iterator constructor");
        return 0;
    }

    /* Check for an index */
    if (flags & (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) {
        if ((flags & (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) ==
                    (NPY_ITER_C_INDEX | NPY_ITER_F_INDEX)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flags C_INDEX and "
                    "F_INDEX cannot both be specified");
            return 0;
        }
        (*itflags) |= NPY_ITFLAG_HASINDEX;
    }
    /* Check if a multi-index was requested */
    if (flags & NPY_ITER_MULTI_INDEX) {
        /*
         * This flag primarily disables dimension manipulations that
         * would produce an incorrect multi-index.
         */
        (*itflags) |= NPY_ITFLAG_HASMULTIINDEX;
    }
    /* Check if the caller wants to handle inner iteration */
    if (flags & NPY_ITER_EXTERNAL_LOOP) {
        if ((*itflags) & (NPY_ITFLAG_HASINDEX | NPY_ITFLAG_HASMULTIINDEX)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flag EXTERNAL_LOOP cannot be used "
                    "if an index or multi-index is being tracked");
            return 0;
        }
        (*itflags) |= NPY_ITFLAG_EXLOOP;
    }
    /* Ranged */
    if (flags & NPY_ITER_RANGED) {
        (*itflags) |= NPY_ITFLAG_RANGE;
        if ((flags & NPY_ITER_EXTERNAL_LOOP) &&
                                    !(flags & NPY_ITER_BUFFERED)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator flag RANGED cannot be used with "
                    "the flag EXTERNAL_LOOP unless "
                    "BUFFERED is also enabled");
            return 0;
        }
    }
    /* Buffering */
    if (flags & NPY_ITER_BUFFERED) {
        (*itflags) |= NPY_ITFLAG_BUFFER;
        if (flags & NPY_ITER_GROWINNER) {
            (*itflags) |= NPY_ITFLAG_GROWINNER;
        }
        if (flags & NPY_ITER_DELAY_BUFALLOC) {
            (*itflags) |= NPY_ITFLAG_DELAYBUF;
        }
    }

    return 1;
}

static int
hnpyiter_check_op_axes(HPyContext *ctx, int nop, int oa_ndim, int **op_axes,
                        const npy_intp *itershape)
{
    char axes_dupcheck[NPY_MAXDIMS];
    int iop, idim;

    if (oa_ndim < 0) {
        /*
         * If `oa_ndim < 0`, `op_axes` and `itershape` are signalled to
         * be unused and should be NULL. (Before NumPy 1.8 this was
         * signalled by `oa_ndim == 0`.)
         */
        if (op_axes != NULL || itershape != NULL) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "If 'op_axes' or 'itershape' is not NULL in the iterator "
                    "constructor, 'oa_ndim' must be zero or greater");
            return 0;
        }
        return 1;
    }
    if (oa_ndim > NPY_MAXDIMS) {
        HPyErr_Format_p(ctx, ctx->h_ValueError,
                "Cannot construct an iterator with more than %d dimensions "
                "(%d were requested for op_axes)",
                NPY_MAXDIMS, oa_ndim);
        return 0;
    }
    if (op_axes == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "If 'oa_ndim' is zero or greater in the iterator "
                "constructor, then op_axes cannot be NULL");
        return 0;
    }

    /* Check that there are no duplicates in op_axes */
    for (iop = 0; iop < nop; ++iop) {
        int *axes = op_axes[iop];
        if (axes != NULL) {
            memset(axes_dupcheck, 0, NPY_MAXDIMS);
            for (idim = 0; idim < oa_ndim; ++idim) {
                int i = npyiter_get_op_axis(axes[idim], NULL);

                if (i >= 0) {
                    if (i >= NPY_MAXDIMS) {
                        HPyErr_Format_p(ctx, ctx->h_ValueError,
                                "The 'op_axes' provided to the iterator "
                                "constructor for operand %d "
                                "contained invalid "
                                "values %d", iop, i);
                        return 0;
                    }
                    else if (axes_dupcheck[i] == 1) {
                        HPyErr_Format_p(ctx, ctx->h_ValueError,
                                "The 'op_axes' provided to the iterator "
                                "constructor for operand %d "
                                "contained duplicate "
                                "value %d", iop, i);
                        return 0;
                    }
                    else {
                        axes_dupcheck[i] = 1;
                    }
                }
            }
        }
    }

    return 1;
}

static int
hnpyiter_calculate_ndim(HPyContext *ctx, int nop, HPy *op_in,
                       int oa_ndim)
{
    /* If 'op_axes' is being used, force 'ndim' */
    if (oa_ndim >= 0 ) {
        return oa_ndim;
    }
    /* Otherwise it's the maximum 'ndim' from the operands */
    else {
        int ndim = 0, iop;

        for (iop = 0; iop < nop; ++iop) {
            if (!HPy_IsNull(op_in[iop])) {
                int ondim = PyArray_NDIM(PyArrayObject_AsStruct(ctx, op_in[iop]));
                if (ondim > ndim) {
                    ndim = ondim;
                }
            }

        }

        return ndim;
    }
}

/*
 * Checks the per-operand input flags, and fills in op_itflags.
 *
 * Returns 1 on success, 0 on failure.
 */
static int
hnpyiter_check_per_op_flags(HPyContext *ctx, npy_uint32 op_flags, npyiter_opitflags *op_itflags)
{
    if ((op_flags & NPY_ITER_GLOBAL_FLAGS) != 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                    "A global iterator flag was passed as a per-operand flag "
                    "to the iterator constructor");
        return 0;
    }

    /* Check the read/write flags */
    if (op_flags & NPY_ITER_READONLY) {
        /* The read/write flags are mutually exclusive */
        if (op_flags & (NPY_ITER_READWRITE|NPY_ITER_WRITEONLY)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Only one of the iterator flags READWRITE, "
                    "READONLY, and WRITEONLY may be "
                    "specified for an operand");
            return 0;
        }

        *op_itflags = NPY_OP_ITFLAG_READ;
    }
    else if (op_flags & NPY_ITER_READWRITE) {
        /* The read/write flags are mutually exclusive */
        if (op_flags & NPY_ITER_WRITEONLY) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Only one of the iterator flags READWRITE, "
                    "READONLY, and WRITEONLY may be "
                    "specified for an operand");
            return 0;
        }

        *op_itflags = NPY_OP_ITFLAG_READ|NPY_OP_ITFLAG_WRITE;
    }
    else if(op_flags & NPY_ITER_WRITEONLY) {
        *op_itflags = NPY_OP_ITFLAG_WRITE;
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "None of the iterator flags READWRITE, "
                "READONLY, or WRITEONLY were "
                "specified for an operand");
        return 0;
    }

    /* Check the flags for temporary copies */
    if (((*op_itflags) & NPY_OP_ITFLAG_WRITE) &&
                (op_flags & (NPY_ITER_COPY |
                           NPY_ITER_UPDATEIFCOPY)) == NPY_ITER_COPY) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "If an iterator operand is writeable, must use "
                "the flag UPDATEIFCOPY instead of "
                "COPY");
        return 0;
    }

    /* Check the flag for a write masked operands */
    if (op_flags & NPY_ITER_WRITEMASKED) {
        if (!((*op_itflags) & NPY_OP_ITFLAG_WRITE)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                "The iterator flag WRITEMASKED may only "
                "be used with READWRITE or WRITEONLY");
            return 0;
        }
        if ((op_flags & NPY_ITER_ARRAYMASK) != 0) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                "The iterator flag WRITEMASKED may not "
                "be used together with ARRAYMASK");
            return 0;
        }
        *op_itflags |= NPY_OP_ITFLAG_WRITEMASKED;
    }

    if ((op_flags & NPY_ITER_VIRTUAL) != 0) {
        if ((op_flags & NPY_ITER_READWRITE) == 0) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                "The iterator flag VIRTUAL should be "
                "be used together with READWRITE");
            return 0;
        }
        *op_itflags |= NPY_OP_ITFLAG_VIRTUAL;
    }

    return 1;
}

/*
 * Prepares a constructor operand.  Assumes a reference to 'op'
 * is owned, and that 'op' may be replaced.  Fills in 'op_dataptr',
 * 'op_dtype', and may modify 'op_itflags'.
 *
 * Returns 1 on success, 0 on failure.
 */
static int
hnpyiter_prepare_one_operand(HPyContext *ctx, HPy *op,
                        char **op_dataptr,
                        HPy op_request_dtype,
                        HPy *op_dtype,
                        npy_uint32 flags,
                        npy_uint32 op_flags, npyiter_opitflags *op_itflags)
{
    /* required for cut-off to legacy API */
    PyObject *py_op = NULL;

    /* NULL operands must be automatically allocated outputs */
    if (HPy_IsNull(*op)) {
        /* ALLOCATE or VIRTUAL should be enabled */
        if ((op_flags & (NPY_ITER_ALLOCATE|NPY_ITER_VIRTUAL)) == 0) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Iterator operand was NULL, but neither the "
                    "ALLOCATE nor the VIRTUAL flag was specified");
            return 0;
        }

        if (op_flags & NPY_ITER_ALLOCATE) {
            /* Writing should be enabled */
            if (!((*op_itflags) & NPY_OP_ITFLAG_WRITE)) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "Automatic allocation was requested for an iterator "
                        "operand, but it wasn't flagged for writing");
                return 0;
            }
            /*
             * Reading should be disabled if buffering is enabled without
             * also enabling NPY_ITER_DELAY_BUFALLOC.  In all other cases,
             * the caller may initialize the allocated operand to a value
             * before beginning iteration.
             */
            if (((flags & (NPY_ITER_BUFFERED |
                            NPY_ITER_DELAY_BUFALLOC)) == NPY_ITER_BUFFERED) &&
                    ((*op_itflags) & NPY_OP_ITFLAG_READ)) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "Automatic allocation was requested for an iterator "
                        "operand, and it was flagged as readable, but "
                        "buffering  without delayed allocation was enabled");
                return 0;
            }

            /* If a requested dtype was provided, use it, otherwise NULL */
            *op_dtype = HPy_Dup(ctx, op_request_dtype);
        }
        else {
            *op_dtype = HPy_NULL;
        }

        /* Specify bool if no dtype was requested for the mask */
        if (op_flags & NPY_ITER_ARRAYMASK) {
            if (HPy_IsNull(*op_dtype)) {
                /* TODO HPY LABS PORT: cut off to Numpy API */
                CAPI_WARN("hnpyiter_prepare_one_operand");
                PyArray_Descr *py_op_dtype = PyArray_DescrFromType(NPY_BOOL);
                *op_dtype = HPy_FromPyObject(ctx, (PyObject*) py_op_dtype);
                Py_DECREF(py_op_dtype);
                if (HPy_IsNull(*op_dtype)) {
                    return 0;
                }
            }
        }

        *op_dataptr = NULL;

        return 1;
    }

    /* VIRTUAL operands must be NULL */
    if (op_flags & NPY_ITER_VIRTUAL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator operand flag VIRTUAL was specified, "
                "but the operand was not NULL");
        return 0;
    }


    if (HPyArray_Check(ctx, *op)) {

        if ((*op_itflags) & NPY_OP_ITFLAG_WRITE) {
            if (HPyArray_FailUnlessWriteable(ctx, *op,
                    "operand array with iterator write flag set") < 0) {
                goto error;
            }
        }
        PyArrayObject *op_data = PyArrayObject_AsStruct(ctx, *op);
        if (!(flags & NPY_ITER_ZEROSIZE_OK) && PyArray_SIZE(op_data) == 0) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Iteration of zero-sized operands is not enabled");
            goto error;
        }
        *op_dataptr = PyArray_BYTES(op_data);
        *op_dtype = HPyArray_DESCR(ctx, *op, op_data);
        if (HPy_IsNull(*op_dtype)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Iterator input operand has no dtype descr");
            goto error;
        }
        /*
         * If references weren't specifically allowed, make sure there
         * are no references in the inputs or requested dtypes.
         */
        if (!(flags & NPY_ITER_REFS_OK)) {
            HPy h_dt = HPyArray_GetDescr(ctx, *op);
            PyArray_Descr *dt = PyArray_Descr_AsStruct(ctx, h_dt);
            if (((dt->flags & (NPY_ITEM_REFCOUNT |
                           NPY_ITEM_IS_POINTER)) != 0) ||
                    (!HPy_Is(ctx, h_dt, *op_dtype) &&
                        ((PyArray_Descr_AsStruct(ctx, *op_dtype)->flags & (NPY_ITEM_REFCOUNT |
                                             NPY_ITEM_IS_POINTER))) != 0)) {
                HPyErr_SetString(ctx, ctx->h_TypeError,
                        "Iterator operand or requested dtype holds "
                        "references, but the REFS_OK flag was not enabled");
                HPy_Close(ctx, h_dt);
                goto error;
            }
            HPy_Close(ctx, h_dt);
        }
        /*
         * Checking whether casts are valid is done later, once the
         * final data types have been selected.  For now, just store the
         * requested type.
         */
        if (!HPy_IsNull(op_request_dtype)) {
            /* We just have a borrowed reference to op_request_dtype */
            HPy h = HPyArray_AdaptDescriptorToArray(ctx, *op, op_request_dtype);
            if (HPy_IsNull(h)) {
                goto error;
            }
            HPy_Close(ctx, *op_dtype);
            *op_dtype = h;
        }

        /* Check if the operand is in the byte order requested */
        if (op_flags & NPY_ITER_NBO) {
            /* Check byte order */
            if (!PyArray_ISNBO(PyArray_Descr_AsStruct(ctx, *op_dtype)->byteorder)) {
                /* Replace with a new descr which is in native byte order */
                /* TODO HPY LABS PORT: cut off to Numpy API 'PyArray_DescrNewByteorder'*/
                CAPI_WARN("hnpyiter_prepare_one_operand");
                PyObject *py_op_dtype = HPy_AsPyObject(ctx, *op_dtype);
                PyArray_Descr *py_new_op_dtype = PyArray_DescrNewByteorder((PyArray_Descr*) py_op_dtype, NPY_NATIVE);
                Py_DECREF(py_op_dtype);
                HPy_Close(ctx, *op_dtype);
                *op_dtype = HPy_FromPyObject(ctx, (PyObject *)py_new_op_dtype);
                Py_DECREF(py_new_op_dtype);
                if (HPy_IsNull(*op_dtype)) {
                    goto error;
                }                
                NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                                    "because of NPY_ITER_NBO\n");
                /* Indicate that byte order or alignment needs fixing */
                *op_itflags |= NPY_OP_ITFLAG_CAST;
            }
        }
        /* Check if the operand is aligned */
        if (op_flags & NPY_ITER_ALIGNED) {
            /* Check alignment */
            if (!IsAligned(op_data)) {
                NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                                    "because of NPY_ITER_ALIGNED\n");
                *op_itflags |= NPY_OP_ITFLAG_CAST;
            }
        }
        /*
         * The check for NPY_ITER_CONTIG can only be done later,
         * once the final iteration order is settled.
         */
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator inputs must be ndarrays");
        goto error;
    }

    return 1;

error:
    Py_XDECREF(py_op);
    return 0;
}

/*
 * Process all the operands, copying new references so further processing
 * can replace the arrays if copying is necessary.
 */
static int
hnpyiter_prepare_operands(HPyContext *ctx, int nop, HPy *op_in,
                    HPy *op,
                    char **op_dataptr,
                    HPy *op_request_dtypes,
                    HPy *op_dtype,
                    npy_uint32 flags,
                    npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                    npy_int8 *out_maskop)
{
    int iop, i;
    npy_int8 maskop = -1;
    int any_writemasked_ops = 0;

    /*
     * Here we just prepare the provided operands.
     */
    for (iop = 0; iop < nop; ++iop) {
        op[iop] = HPy_Dup(ctx, op_in[iop]);
        op_dtype[iop] = HPy_NULL;

        /* Check the readonly/writeonly flags, and fill in op_itflags */
        if (!hnpyiter_check_per_op_flags(ctx, op_flags[iop], &op_itflags[iop])) {
            goto fail_iop;
        }

        /* Extract the operand which is for masked iteration */
        if ((op_flags[iop] & NPY_ITER_ARRAYMASK) != 0) {
            if (maskop != -1) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "Only one iterator operand may receive an "
                        "ARRAYMASK flag");
                goto fail_iop;
            }

            maskop = iop;
            *out_maskop = iop;
        }

        if (op_flags[iop] & NPY_ITER_WRITEMASKED) {
            any_writemasked_ops = 1;
        }

        /*
         * Prepare the operand.  This produces an op_dtype[iop] reference
         * on success.
         */
        if (!hnpyiter_prepare_one_operand(ctx, &op[iop],
                        &op_dataptr[iop],
                        op_request_dtypes ? op_request_dtypes[iop] : HPy_NULL,
                        &op_dtype[iop],
                        flags,
                        op_flags[iop], &op_itflags[iop])) {
            goto fail_iop;
        }
    }

    /* If all the operands were NULL, it's an error */
    if (HPy_IsNull(op[0])) {
        int all_null = 1;
        for (iop = 1; iop < nop; ++iop) {
            if (!HPy_IsNull(op[iop])) {
                all_null = 0;
                break;
            }
        }
        if (all_null) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "At least one iterator operand must be non-NULL");
            goto fail_nop;
        }
    }

    if (any_writemasked_ops && maskop < 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "An iterator operand was flagged as WRITEMASKED, "
                "but no ARRAYMASK operand was given to supply "
                "the mask");
        goto fail_nop;
    }
    else if (!any_writemasked_ops && maskop >= 0) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "An iterator operand was flagged as the ARRAYMASK, "
                "but no WRITEMASKED operands were given to use "
                "the mask");
        goto fail_nop;
    }

    return 1;

  fail_nop:
    iop = nop - 1;
  fail_iop:
    for (i = 0; i < iop+1; ++i) {
        HPy_Close(ctx, op[i]);
        HPy_Close(ctx, op_dtype[i]);
    }
    return 0;
}

static const char *
npyiter_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";
        case NPY_EQUIV_CASTING:
            return "'equiv'";
        case NPY_SAFE_CASTING:
            return "'safe'";
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";
        default:
            return "<unknown>";
    }
}


static int
hnpyiter_check_casting(HPyContext *ctx, int nop, HPy *op,
                    HPy *h_op_dtype,
                    NPY_CASTING casting,
                    npyiter_opitflags *op_itflags)
{
    // PyArrayObject **op,
    // PyArray_Descr **op_dtype,
    int iop;

    for(iop = 0; iop < nop; ++iop) {
        NPY_IT_DBG_PRINT1("Iterator: Checking casting for operand %d\n",
                            (int)iop);
#if NPY_IT_DBG_TRACING
#ifdef HPY
#error "not supported in HPy"
#else
        
        printf("op: ");
        if (op[iop] != NULL) {
            PyObject_Print((PyObject *)PyArray_DESCR(op[iop]), stdout, 0);
        }
        else {
            printf("<null>");
        }
        printf(", iter: ");
        PyObject_Print((PyObject *)op_dtype[iop], stdout, 0);
        printf("\n");
#endif
#endif
        /* If the types aren't equivalent, a cast is necessary */
        if (!HPy_IsNull(op[iop])) {
            HPy h_op_descr = HPyArray_GetDescr(ctx, op[iop]); /* PyArray_DESCR(op[iop]) */
            PyArray_Descr *op_descr = (PyArray_Descr *) HPy_AsPyObject(ctx, h_op_descr);
            HPy_Close(ctx, h_op_descr);

            PyArray_Descr *op_dtype_iop = (PyArray_Descr *) HPy_AsPyObject(ctx, h_op_dtype[iop]);
            if (!PyArray_EquivTypes(op_descr, op_dtype_iop)) {
                /* Check read (op -> temp) casting */
                /* TODO HPY LABS PORT cut-off 'PyArray_CanCastArrayTo' */
                CAPI_WARN("hnpyiter_prepare_one_operand");
                PyArrayObject *py_op_iop = (PyArrayObject *) HPy_AsPyObject(ctx, op[iop]);
                if ((op_itflags[iop] & NPY_OP_ITFLAG_READ) &&
                        !PyArray_CanCastArrayTo(py_op_iop,
                                op_dtype_iop,
                                casting)) {
                    HPyErr_SetString(ctx, ctx->h_TypeError,
                            "Iterator operand %d dtype could not be cast from "
                            "%R to %R according to the rule %s");
                    /* TODO HPY LABS PORT PyErr_Format
                PyErr_Format(PyExc_TypeError,
                        "Iterator operand %d dtype could not be cast from "
                        "%R to %R according to the rule %s",
                        iop, PyArray_DESCR(op[iop]), op_dtype[iop],
                        npyiter_casting_to_string(casting));
                     */
                    Py_XDECREF(op_dtype_iop);
                    Py_XDECREF(op_descr);
                    Py_XDECREF(py_op_iop);
                    return 0;
                }
                Py_XDECREF(py_op_iop);
                /* Check write (temp -> op) casting */
                if ((op_itflags[iop] & NPY_OP_ITFLAG_WRITE) &&
                        !PyArray_CanCastTypeTo(op_dtype_iop,
                                op_descr,
                                casting)) {
                    HPyErr_SetString(ctx, ctx->h_TypeError,
                            "Iterator requested dtype could not be cast from "
                            "%R to %R, the operand %d dtype, "
                            "according to the rule %s");
                    /* TODO HPY LABS PORT PyErr_Format
                PyErr_Format(PyExc_TypeError,
                        "Iterator requested dtype could not be cast from "
                        "%R to %R, the operand %d dtype, "
                        "according to the rule %s",
                        op_dtype[iop], PyArray_DESCR(op[iop]), iop,
                        npyiter_casting_to_string(casting));
                     */
                    Py_XDECREF(op_dtype_iop);
                    Py_XDECREF(op_descr);
                    return 0;
                }
                Py_XDECREF(op_dtype_iop);
                Py_XDECREF(op_descr);

                NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                        "because the types aren't equivalent\n");
                /* Indicate that this operand needs casting */
                op_itflags[iop] |= NPY_OP_ITFLAG_CAST;
            }
        }
    }

    return 1;
}

/*
 * Checks that the mask broadcasts to the WRITEMASK REDUCE
 * operand 'iop', but 'iop' never broadcasts to the mask.
 * If 'iop' broadcasts to the mask, the result would be more
 * than one mask value per reduction element, something which
 * is invalid.
 *
 * This check should only be called after all the operands
 * have been filled in.
 *
 * Returns 1 on success, 0 on error.
 */
static int
check_mask_for_writemasked_reduction(NpyIter *iter, int iop)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);
    int maskop = NIT_MASKOP(iter);

    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    for(idim = 0; idim < ndim; ++idim) {
        npy_intp maskstride, istride;

        istride = NAD_STRIDES(axisdata)[iop];
        maskstride = NAD_STRIDES(axisdata)[maskop];

        /*
         * If 'iop' is being broadcast to 'maskop', we have
         * the invalid situation described above.
         */
        if (maskstride != 0 && istride == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator reduction operand is WRITEMASKED, "
                    "but also broadcasts to multiple mask values. "
                    "There can be only one mask value per WRITEMASKED "
                    "element.");
            return 0;
        }

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    return 1;
}

/*
 * Check whether a reduction is OK based on the flags and the operand being
 * readwrite. This path is deprecated, since usually only specific axes
 * should be reduced. If axes are specified explicitly, the flag is
 * unnecessary.
 */
static int
npyiter_check_reduce_ok_and_set_flags(
        NpyIter *iter, npy_uint32 flags, npyiter_opitflags *op_itflags,
        int dim) {
    /* If it's writeable, this means a reduction */
    if (*op_itflags & NPY_OP_ITFLAG_WRITE) {
        if (!(flags & NPY_ITER_REDUCE_OK)) {
            PyErr_Format(PyExc_ValueError,
                    "output operand requires a reduction along dimension %d, "
                    "but the reduction is not enabled. The dimension size of 1 "
                    "does not match the expected output shape.", dim);
            return 0;
        }
        if (!(*op_itflags & NPY_OP_ITFLAG_READ)) {
            PyErr_SetString(PyExc_ValueError,
                    "output operand requires a reduction, but is flagged as "
                    "write-only, not read-write");
            return 0;
        }
        NPY_IT_DBG_PRINT("Iterator: Indicating that a reduction is"
                         "occurring\n");

        NIT_ITFLAGS(iter) |= NPY_ITFLAG_REDUCE;
        *op_itflags |= NPY_OP_ITFLAG_REDUCE;
    }
    return 1;
}

/**
 * Removes the (additive) NPY_ITER_REDUCTION_AXIS indication and sets
 * is_forced_broadcast to 1 if it is set. Otherwise to 0.
 *
 * @param axis The op_axes[i] to normalize.
 * @param reduction_axis Output 1 if a reduction axis, otherwise 0.
 * @returns The normalized axis (without reduce axis flag).
 */
static NPY_INLINE int
npyiter_get_op_axis(int axis, npy_bool *reduction_axis) {
    npy_bool forced_broadcast = axis >= NPY_ITER_REDUCTION_AXIS(-1);

    if (reduction_axis != NULL) {
        *reduction_axis = forced_broadcast;
    }
    if (forced_broadcast) {
        return axis - NPY_ITER_REDUCTION_AXIS(0);
    }
    return axis;
}

/*
 * Fills in the AXISDATA for the 'nop' operands, broadcasting
 * the dimensionas as necessary.  Also fills
 * in the ITERSIZE data member.
 *
 * If op_axes is not NULL, it should point to an array of ndim-sized
 * arrays, one for each op.
 *
 * Returns 1 on success, 0 on failure.
 */
static int
hnpyiter_fill_axisdata(HPyContext *ctx, NpyIter *iter, npy_uint32 flags, npyiter_opitflags *op_itflags,
                    char **op_dataptr,
                    const npy_uint32 *op_flags, int **op_axes,
                    npy_intp const *itershape)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);
    int maskop = NIT_MASKOP(iter);
    PyArrayObject *op_iop_data = NULL;

    int ondim;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    HPy *op = NIT_OPERANDS(iter);
    HPy op_cur;
    npy_intp broadcast_shape[NPY_MAXDIMS];

    /* First broadcast the shapes together */
    if (itershape == NULL) {
        for (idim = 0; idim < ndim; ++idim) {
            broadcast_shape[idim] = 1;
        }
    }
    else {
        for (idim = 0; idim < ndim; ++idim) {
            broadcast_shape[idim] = itershape[idim];
            /* Negative shape entries are deduced from the operands */
            if (broadcast_shape[idim] < 0) {
                broadcast_shape[idim] = 1;
            }
        }
    }
    for (iop = 0; iop < nop; ++iop) {
        op_cur = op[iop];
        if (!HPy_IsNull(op_cur)) {
            PyArrayObject *op_cur_data = PyArrayObject_AsStruct(ctx, op_cur);
            npy_intp *shape = PyArray_DIMS(op_cur_data);
            ondim = PyArray_NDIM(op_cur_data);

            if (op_axes == NULL || op_axes[iop] == NULL) {
                /*
                 * Possible if op_axes are being used, but
                 * op_axes[iop] is NULL
                 */
                if (ondim > ndim) {
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                            "input operand has more dimensions than allowed "
                            "by the axis remapping");
                    return 0;
                }
                for (idim = 0; idim < ondim; ++idim) {
                    npy_intp bshape = broadcast_shape[idim+ndim-ondim];
                    npy_intp op_shape = shape[idim];

                    if (bshape == 1) {
                        broadcast_shape[idim+ndim-ondim] = op_shape;
                    }
                    else if (bshape != op_shape && op_shape != 1) {
                        goto broadcast_error;
                    }
                }
            }
            else {
                int *axes = op_axes[iop];
                for (idim = 0; idim < ndim; ++idim) {
                    int i = npyiter_get_op_axis(axes[idim], NULL);

                    if (i >= 0) {
                        if (i < ondim) {
                            npy_intp bshape = broadcast_shape[idim];
                            npy_intp op_shape = shape[i];

                            if (bshape == 1) {
                                broadcast_shape[idim] = op_shape;
                            }
                            else if (bshape != op_shape && op_shape != 1) {
                                goto broadcast_error;
                            }
                        }
                        else {
                            HPyErr_Format_p(ctx, ctx->h_ValueError,
                                    "Iterator input op_axes[%d][%d] (==%d) "
                                    "is not a valid axis of op[%d], which "
                                    "has %d dimensions ",
                                    iop, (ndim-idim-1), i,
                                    iop, ondim);
                            return 0;
                        }
                    }
                }
            }
        }
    }
    /*
     * If a shape was provided with a 1 entry, make sure that entry didn't
     * get expanded by broadcasting.
     */
    if (itershape != NULL) {
        for (idim = 0; idim < ndim; ++idim) {
            if (itershape[idim] == 1 && broadcast_shape[idim] != 1) {
                goto broadcast_error;
            }
        }
    }

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    if (ndim == 0) {
        /* Need to fill the first axisdata, even if the iterator is 0-d */
        NAD_SHAPE(axisdata) = 1;
        NAD_INDEX(axisdata) = 0;
        memcpy(NAD_PTRS(axisdata), op_dataptr, NPY_SIZEOF_INTP*nop);
        memset(NAD_STRIDES(axisdata), 0, NPY_SIZEOF_INTP*nop);
    }

    /* Now process the operands, filling in the axisdata */
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp bshape = broadcast_shape[ndim-idim-1];
        npy_intp *strides = NAD_STRIDES(axisdata);

        NAD_SHAPE(axisdata) = bshape;
        NAD_INDEX(axisdata) = 0;
        memcpy(NAD_PTRS(axisdata), op_dataptr, NPY_SIZEOF_INTP*nop);

        for (iop = 0; iop < nop; ++iop) {
            op_cur = op[iop];
            PyArrayObject *op_cur_data = PyArrayObject_AsStruct(ctx, op_cur);

            if (op_axes == NULL || op_axes[iop] == NULL) {
                if (HPy_IsNull(op_cur)) {
                    strides[iop] = 0;
                }
                else {
                    ondim = PyArray_NDIM(op_cur_data);
                    if (bshape == 1) {
                        strides[iop] = 0;
                        if (idim >= ondim &&
                                    (op_flags[iop] & NPY_ITER_NO_BROADCAST)) {
                            goto operand_different_than_broadcast;
                        }
                    }
                    else if (idim >= ondim ||
                                    PyArray_DIM(op_cur_data, ondim-idim-1) == 1) {
                        strides[iop] = 0;
                        if (op_flags[iop] & NPY_ITER_NO_BROADCAST) {
                            goto operand_different_than_broadcast;
                        }
                        /* If it's writeable, this means a reduction */
                        if (op_itflags[iop] & NPY_OP_ITFLAG_WRITE) {
                            if (!(flags & NPY_ITER_REDUCE_OK)) {
                                HPyErr_SetString(ctx, ctx->h_ValueError,
                                        "output operand requires a "
                                        "reduction, but reduction is "
                                        "not enabled");
                                return 0;
                            }
                            if (!(op_itflags[iop] & NPY_OP_ITFLAG_READ)) {
                                HPyErr_SetString(ctx, ctx->h_ValueError,
                                        "output operand requires a "
                                        "reduction, but is flagged as "
                                        "write-only, not read-write");
                                return 0;
                            }
                            /*
                             * The ARRAYMASK can't be a reduction, because
                             * it would be possible to write back to the
                             * array once when the ARRAYMASK says 'True',
                             * then have the reduction on the ARRAYMASK
                             * later flip to 'False', indicating that the
                             * write back should never have been done,
                             * and violating the strict masking semantics
                             */
                            if (iop == maskop) {
                                HPyErr_SetString(ctx, ctx->h_ValueError,
                                        "output operand requires a "
                                        "reduction, but is flagged as "
                                        "the ARRAYMASK operand which "
                                        "is not permitted to be the "
                                        "result of a reduction");
                                return 0;
                            }

                            NIT_ITFLAGS(iter) |= NPY_ITFLAG_REDUCE;
                            op_itflags[iop] |= NPY_OP_ITFLAG_REDUCE;
                        }
                    }
                    else {
                        strides[iop] = PyArray_STRIDE(op_cur_data, ondim-idim-1);
                    }
                }
            }
            else {
                int *axes = op_axes[iop];
                npy_bool reduction_axis;
                int i;
                i = npyiter_get_op_axis(axes[ndim - idim - 1], &reduction_axis);

                if (reduction_axis) {
                    /* This is explicitly a reduction axis */
                    strides[iop] = 0;
                    NIT_ITFLAGS(iter) |= NPY_ITFLAG_REDUCE;
                    op_itflags[iop] |= NPY_OP_ITFLAG_REDUCE;

                    if (NPY_UNLIKELY((i >= 0) && !HPy_IsNull(op_cur) &&
                            (PyArray_DIM(op_cur_data, i) != 1))) {
                        HPyErr_Format_p(ctx, ctx->h_ValueError,
                                "operand was set up as a reduction along axis "
                                "%d, but the length of the axis is %zd "
                                "(it has to be 1)",
                                i, (Py_ssize_t)PyArray_DIM(op_cur_data, i));
                        return 0;
                    }
                }
                else if (bshape == 1) {
                    /*
                     * If the full iterator shape is 1, zero always works.
                     * NOTE: We thus always allow broadcast dimensions (i = -1)
                     *       if the shape is 1.
                     */
                    strides[iop] = 0;
                }
                else if (i >= 0) {
                    if (HPy_IsNull(op_cur)) {
                        /* stride is filled later, shape will match `bshape` */
                        strides[iop] = 0;
                    }
                    else if (PyArray_DIM(op_cur_data, i) == 1) {
                        strides[iop] = 0;
                        if (op_flags[iop] & NPY_ITER_NO_BROADCAST) {
                            goto operand_different_than_broadcast;
                        }
                        if (!npyiter_check_reduce_ok_and_set_flags(
                                iter, flags, &op_itflags[iop], i)) {
                            return 0;
                        }
                    }
                    else {
                        strides[iop] = PyArray_STRIDE(op_cur_data, i);
                    }
                }
                else {
                    strides[iop] = 0;
                    if (!npyiter_check_reduce_ok_and_set_flags(
                            iter, flags, &op_itflags[iop], i)) {
                        return 0;
                    }
                }
            }
        }

        NIT_ADVANCE_AXISDATA(axisdata, 1);
    }

    /* Now fill in the ITERSIZE member */
    NIT_ITERSIZE(iter) = 1;
    for (idim = 0; idim < ndim; ++idim) {
        if (npy_mul_with_overflow_intp(&NIT_ITERSIZE(iter),
                    NIT_ITERSIZE(iter), broadcast_shape[idim])) {
            if ((itflags & NPY_ITFLAG_HASMULTIINDEX) &&
                    !(itflags & NPY_ITFLAG_HASINDEX) &&
                    !(itflags & NPY_ITFLAG_BUFFER)) {
                /*
                 * If RemoveAxis may be called, the size check is delayed
                 * until either the multi index is removed, or GetIterNext
                 * is called.
                 */
                NIT_ITERSIZE(iter) = -1;
                break;
            }
            else {
                HPyErr_SetString(ctx, ctx->h_ValueError, "iterator is too large");
                return 0;
            }
        }
    }
    /* The range defaults to everything */
    NIT_ITERSTART(iter) = 0;
    NIT_ITEREND(iter) = NIT_ITERSIZE(iter);

    return 1;

broadcast_error: {
        npy_intp remdims[NPY_MAXDIMS];

        if (op_axes == NULL) {
            HPy shape1 = HPyUnicode_FromString(ctx, "");
            if (HPy_IsNull(shape1)) {
                return 0;
            }
            for (iop = 0; iop < nop; ++iop) {
                if (!HPy_IsNull(op[iop])) {
                    op_iop_data = PyArrayObject_AsStruct(ctx, op[iop]);
                    int ndims = PyArray_NDIM(op_iop_data);
                    npy_intp *dims = PyArray_DIMS(op_iop_data);
                    CAPI_WARN("hnpyiter_fill_axisdata");
                    PyObject *py_tmp = convert_shape_to_string(ndims, dims, " ");
                    if (py_tmp == NULL) {
                        HPy_Close(ctx, shape1);
                        return 0;
                    }
                    HPy tmp = HPy_FromPyObject(ctx, py_tmp);
                    Py_DECREF(py_tmp);
                    HPy_SETREF(ctx, shape1, HPy_Add(ctx, shape1, tmp));
                    HPy_Close(ctx, tmp);
                    if (HPy_IsNull(shape1)) {
                        return 0;
                    }
                }
            }
            if (itershape == NULL) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "operands could not be broadcast together with "
                        "shapes %S");
                /* TODO HPY LABS PORT: PyErr_Format
                PyErr_Format(PyExc_ValueError,
                        "operands could not be broadcast together with "
                        "shapes %S", shape1);
                */
                HPy_Close(ctx, shape1);
                return 0;
            }
            else {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "operands could not be broadcast together with "
                        "shapes %S and requested shape %S");
                /* TODO HPY LABS PORT: PyErr_Format
                CAPI_WARN("hnpyiter_fill_axisdata");
                PyObject *shape2 = convert_shape_to_string(ndim, itershape, "");
                if (shape2 == NULL) {
                    Py_DECREF(shape1);
                    return 0;
                }
                PyErr_Format(PyExc_ValueError,
                        "operands could not be broadcast together with "
                        "shapes %S and requested shape %S", shape1, shape2);
                Py_DECREF(shape2);
                */
                HPy_Close(ctx, shape1);
                return 0;
            }
        }
        else {
            HPy shape1 = HPyUnicode_FromString(ctx, "");
            if (HPy_IsNull(shape1)) {
                return 0;
            }
            for (iop = 0; iop < nop; ++iop) {
                if (!HPy_IsNull(op[iop])) {
                    op_iop_data = PyArrayObject_AsStruct(ctx, op[iop]);
                    int *axes = op_axes[iop];
                    int ndims = PyArray_NDIM(op_iop_data);
                    npy_intp *dims = PyArray_DIMS(op_iop_data);
                    char *tmpstr = (axes == NULL) ? " " : "->";

                    CAPI_WARN("hpyiter_fill_axisdata");
                    PyObject *py_tmp = convert_shape_to_string(ndims, dims, tmpstr);
                    if (py_tmp == NULL) {
                        HPy_Close(ctx, shape1);
                        return 0;
                    }
                    HPy tmp = HPy_FromPyObject(ctx, py_tmp);
                    Py_DECREF(py_tmp);
                    HPy_SETREF(ctx, shape1, HPy_Add(ctx, shape1, tmp));
                    HPy_Close(ctx, tmp);
                    if (HPy_IsNull(shape1)) {
                        return 0;
                    }

                    if (axes != NULL) {
                        for (idim = 0; idim < ndim; ++idim) {
                            int i = npyiter_get_op_axis(axes[idim], NULL);

                            if (i >= 0 && i < PyArray_NDIM(op_iop_data)) {
                                remdims[idim] = PyArray_DIM(op_iop_data, i);
                            }
                            else {
                                remdims[idim] = -1;
                            }
                        }
                        CAPI_WARN("hpyiter_fill_axisdata");
                        PyObject *py_tmp = convert_shape_to_string(ndim, remdims, " ");
                        if (py_tmp == NULL) {
                            HPy_Close(ctx, shape1);
                            return 0;
                        }
                        HPy tmp = HPy_FromPyObject(ctx, py_tmp);
                        Py_DECREF(py_tmp);
                        HPy_SETREF(ctx, shape1, HPy_Add(ctx, shape1, tmp));
                        HPy_Close(ctx, tmp);
                        if (HPy_IsNull(shape1)) {
                            return 0;
                        }
                    }
                }
            }
            if (itershape == NULL) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "operands could not be broadcast together with "
                        "remapped shapes [original->remapped]: %S");
                /* TODO HPY LABS PORT: PyErr_Format
                PyErr_Format(PyExc_ValueError,
                        "operands could not be broadcast together with "
                        "remapped shapes [original->remapped]: %S", shape1);
                */
                HPy_Close(ctx, shape1);
                return 0;
            }
            else {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "operands could not be broadcast together with "
                        "remapped shapes [original->remapped]: %S and "
                        "requested shape %S");
                /* TODO HPY LABS PORT: PyErr_Format
                CAPI_WARN("hpyiter_fill_axisdata");
                PyObject *py_shape2 = convert_shape_to_string(ndim, itershape, "");
                if (py_shape2 == NULL) {
                    HPy_Close(ctx, shape1);
                    return 0;
                }
                Py_DECREF(py_shape2);
                PyErr_Format(PyExc_ValueError,
                        "operands could not be broadcast together with "
                        "remapped shapes [original->remapped]: %S and "
                        "requested shape %S", shape1, shape2);
                HPy_Close(ctx, shape2);
                */
                HPy_Close(ctx, shape1);
                return 0;
            }
        }
    }

operand_different_than_broadcast: {
        /* operand shape */
        int ndims = PyArray_NDIM(op_iop_data);
        npy_intp *dims = PyArray_DIMS(op_iop_data);
        CAPI_WARN("hpyiter_fill_axisdata");
        PyObject *py_shape1 = convert_shape_to_string(ndims, dims, "");
        if (py_shape1 == NULL) {
            return 0;
        }
        HPy shape1 = HPy_FromPyObject(ctx, py_shape1);
        Py_DECREF(py_shape1);

        /* Broadcast shape */
        /* TODO HPY LABS PORT: only needed for below PyErr_Format
        CAPI_WARN("hpyiter_fill_axisdata");
        PyObject *py_shape2 = convert_shape_to_string(ndim, broadcast_shape, "");
        if (py_shape2 == NULL) {
            HPy_Close(ctx, shape1);
            return 0;
        }
        HPy shape2 = HPy_FromPyObject(ctx, py_shape2);
        Py_DECREF(py_shape2);
        */

        if (op_axes == NULL || op_axes[iop] == NULL) {
            /* operand shape not remapped */

            if (op_flags[iop] & NPY_ITER_READONLY) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                    "non-broadcastable operand with shape %S doesn't "
                    "match the broadcast shape %S");
                /* TODO HPY LABS PORT: PyErr_Format
                PyErr_Format(PyExc_ValueError,
                    "non-broadcastable operand with shape %S doesn't "
                    "match the broadcast shape %S", shape1, shape2);
                */
            }
            else {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                    "non-broadcastable output operand with shape %S doesn't "
                    "match the broadcast shape %S");
                /* TODO HPY LABS PORT: PyErr_Format
                PyErr_Format(PyExc_ValueError,
                    "non-broadcastable output operand with shape %S doesn't "
                    "match the broadcast shape %S", shape1, shape2);
                */
            }
            HPy_Close(ctx, shape1);
            // HPy_Close(ctx, shape2);
            return 0;
        }
        else {
            /* operand shape remapped */

            npy_intp remdims[NPY_MAXDIMS];
            int *axes = op_axes[iop];
            for (idim = 0; idim < ndim; ++idim) {
                npy_intp i = axes[ndim - idim - 1];
                if (i >= 0 && i < PyArray_NDIM(op_iop_data)) {
                    remdims[idim] = PyArray_DIM(op_iop_data, i);
                }
                else {
                    remdims[idim] = -1;
                }
            }

            /* TODO HPY LABS PORT: only needed for below PyErr_Format
            PyObject *shape3 = convert_shape_to_string(ndim, remdims, "");
            if (shape3 == NULL) {
                HPy_Close(ctx, shape1);
                Py_DECREF(shape2);
                return 0;
            }
            */

            if (op_flags[iop] & NPY_ITER_READONLY) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                    "non-broadcastable operand with shape %S "
                    "[remapped to %S] doesn't match the broadcast shape %S");
                /* TODO HPY LABS PORT: PyErr_Format
                PyErr_Format(PyExc_ValueError,
                    "non-broadcastable operand with shape %S "
                    "[remapped to %S] doesn't match the broadcast shape %S",
                    shape1, shape3, shape2);
                */
            }
            else {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                    "non-broadcastable output operand with shape %S "
                    "[remapped to %S] doesn't match the broadcast shape %S");
                /* TODO HPY LABS PORT: PyErr_Format
                PyErr_Format(PyExc_ValueError,
                    "non-broadcastable output operand with shape %S "
                    "[remapped to %S] doesn't match the broadcast shape %S",
                    shape1, shape3, shape2);
                */
            }
            HPy_Close(ctx, shape1);
            // HPy_Close(ctx, shape2);
            // HPy_Close(ctx, shape3);
            return 0;
        }
    }
}

/*
 * Replaces the AXISDATA for the iop'th operand, broadcasting
 * the dimensions as necessary.  Assumes the replacement array is
 * exactly the same shape as the original array used when
 * npy_fill_axisdata was called.
 *
 * If op_axes is not NULL, it should point to an ndim-sized
 * array.
 */
static void
npyiter_replace_axisdata(
        NpyIter *iter, int iop, PyArrayObject *op,
        int orig_op_ndim, const int *op_axes)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);
    char *op_dataptr = PyArray_DATA(op);

    NpyIter_AxisData *axisdata0, *axisdata;
    npy_intp sizeof_axisdata;
    npy_int8 *perm;
    npy_intp baseoffset = 0;

    perm = NIT_PERM(iter);
    axisdata0 = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /*
     * Replace just the strides which were non-zero, and compute
     * the base data address.
     */
    axisdata = axisdata0;

    if (op_axes != NULL) {
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            int i;
            npy_bool axis_flipped;
            npy_intp shape;

            /* Apply perm to get the original axis, and check if its flipped */
            i = npyiter_undo_iter_axis_perm(idim, ndim, perm, &axis_flipped);

            i = npyiter_get_op_axis(op_axes[i], NULL);
            assert(i < orig_op_ndim);
            if (i >= 0) {
                shape = PyArray_DIM(op, i);
                if (shape != 1) {
                    npy_intp stride = PyArray_STRIDE(op, i);
                    if (axis_flipped) {
                        NAD_STRIDES(axisdata)[iop] = -stride;
                        baseoffset += stride*(shape-1);
                    }
                    else {
                        NAD_STRIDES(axisdata)[iop] = stride;
                    }
                }
            }
        }
    }
    else {
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            int i;
            npy_bool axis_flipped;
            npy_intp shape;

            i = npyiter_undo_iter_axis_perm(
                    idim, orig_op_ndim, perm, &axis_flipped);

            if (i >= 0) {
                shape = PyArray_DIM(op, i);
                if (shape != 1) {
                    npy_intp stride = PyArray_STRIDE(op, i);
                    if (axis_flipped) {
                        NAD_STRIDES(axisdata)[iop] = -stride;
                        baseoffset += stride*(shape-1);
                    }
                    else {
                        NAD_STRIDES(axisdata)[iop] = stride;
                    }
                }
            }
        }
    }

    op_dataptr += baseoffset;

    /* Now the base data pointer is calculated, set it everywhere it's needed */
    NIT_RESETDATAPTR(iter)[iop] = op_dataptr;
    NIT_BASEOFFSETS(iter)[iop] = baseoffset;
    axisdata = axisdata0;
    /* Fill at least one axisdata, for the 0-d case */
    NAD_PTRS(axisdata)[iop] = op_dataptr;
    NIT_ADVANCE_AXISDATA(axisdata, 1);
    for (idim = 1; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        NAD_PTRS(axisdata)[iop] = op_dataptr;
    }
}

/*
 * Computes the iterator's index strides and initializes the index values
 * to zero.
 *
 * This must be called before the axes (i.e. the AXISDATA array) may
 * be reordered.
 */
static void
npyiter_compute_index_strides(NpyIter *iter, npy_uint32 flags)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp indexstride;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;

    /*
     * If there is only one element being iterated, we just have
     * to touch the first AXISDATA because nothing will ever be
     * incremented. This also initializes the data for the 0-d case.
     */
    if (NIT_ITERSIZE(iter) == 1) {
        if (itflags & NPY_ITFLAG_HASINDEX) {
            axisdata = NIT_AXISDATA(iter);
            NAD_PTRS(axisdata)[nop] = 0;
        }
        return;
    }

    if (flags & NPY_ITER_C_INDEX) {
        sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
        axisdata = NIT_AXISDATA(iter);
        indexstride = 1;
        for(idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_intp shape = NAD_SHAPE(axisdata);

            if (shape == 1) {
                NAD_STRIDES(axisdata)[nop] = 0;
            }
            else {
                NAD_STRIDES(axisdata)[nop] = indexstride;
            }
            NAD_PTRS(axisdata)[nop] = 0;
            indexstride *= shape;
        }
    }
    else if (flags & NPY_ITER_F_INDEX) {
        sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
        axisdata = NIT_INDEX_AXISDATA(NIT_AXISDATA(iter), ndim-1);
        indexstride = 1;
        for(idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, -1)) {
            npy_intp shape = NAD_SHAPE(axisdata);

            if (shape == 1) {
                NAD_STRIDES(axisdata)[nop] = 0;
            }
            else {
                NAD_STRIDES(axisdata)[nop] = indexstride;
            }
            NAD_PTRS(axisdata)[nop] = 0;
            indexstride *= shape;
        }
    }
}

/*
 * If the order is NPY_KEEPORDER, lets the iterator find the best
 * iteration order, otherwise forces it.  Indicates in the itflags that
 * whether the iteration order was forced.
 */
static void
npyiter_apply_forced_iteration_order(HPyContext *ctx, NpyIter *iter, NPY_ORDER order)
{
    /*npy_uint32 itflags = NIT_ITFLAGS(iter);*/
    int ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    switch (order) {
    case NPY_CORDER:
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;
        break;
    case NPY_FORTRANORDER:
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;
        /* Only need to actually do something if there is more than 1 dim */
        if (ndim > 1) {
            npyiter_reverse_axis_ordering(iter);
        }
        break;
    case NPY_ANYORDER:
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_FORCEDORDER;
        /* Only need to actually do something if there is more than 1 dim */
        if (ndim > 1) {
            HPy *op = NIT_OPERANDS(iter);
            int forder = 1;

            /* Check that all the array inputs are fortran order */
            for (iop = 0; iop < nop; ++iop, ++op) {
                if (!HPy_IsNull(*op) && !PyArray_CHKFLAGS(PyArrayObject_AsStruct(ctx, *op), NPY_ARRAY_F_CONTIGUOUS)) {
                    forder = 0;
                    break;
                }
            }

            if (forder) {
                npyiter_reverse_axis_ordering(iter);
            }
        }
        break;
    case NPY_KEEPORDER:
        /* Don't set the forced order flag here... */
        break;
    }
}

/*
 * This function negates any strides in the iterator
 * which are negative.  When iterating more than one
 * object, it only flips strides when they are all
 * negative or zero.
 */
static void
npyiter_flip_negative_strides(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    npy_intp istrides, nstrides = NAD_NSTRIDES();
    NpyIter_AxisData *axisdata, *axisdata0;
    npy_intp *baseoffsets;
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    int any_flipped = 0;

    axisdata0 = axisdata = NIT_AXISDATA(iter);
    baseoffsets = NIT_BASEOFFSETS(iter);
    for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
        npy_intp *strides = NAD_STRIDES(axisdata);
        int any_negative = 0;

        /*
         * Check the signs of all the operand strides.
         */
        for (iop = 0; iop < nop; ++iop) {
            if (strides[iop] < 0) {
                any_negative = 1;
            }
            else if (strides[iop] != 0) {
                break;
            }
        }
        /*
         * If at least one stride is negative and none are positive,
         * flip all the strides for this dimension.
         */
        if (any_negative && iop == nop) {
            npy_intp shapem1 = NAD_SHAPE(axisdata) - 1;

            for (istrides = 0; istrides < nstrides; ++istrides) {
                npy_intp stride = strides[istrides];

                /* Adjust the base pointers to start at the end */
                baseoffsets[istrides] += shapem1 * stride;
                /* Flip the stride */
                strides[istrides] = -stride;
            }
            /*
             * Make the perm entry negative so get_multi_index
             * knows it's flipped
             */
            NIT_PERM(iter)[idim] = -1-NIT_PERM(iter)[idim];

            any_flipped = 1;
        }
    }

    /*
     * If any strides were flipped, the base pointers were adjusted
     * in the first AXISDATA, and need to be copied to all the rest
     */
    if (any_flipped) {
        char **resetdataptr = NIT_RESETDATAPTR(iter);

        for (istrides = 0; istrides < nstrides; ++istrides) {
            resetdataptr[istrides] += baseoffsets[istrides];
        }
        axisdata = axisdata0;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            char **ptrs = NAD_PTRS(axisdata);
            for (istrides = 0; istrides < nstrides; ++istrides) {
                ptrs[istrides] = resetdataptr[istrides];
            }
        }
        /*
         * Indicate that some of the perm entries are negative,
         * and that it's not (strictly speaking) the identity perm.
         */
        NIT_ITFLAGS(iter) = (NIT_ITFLAGS(iter)|NPY_ITFLAG_NEGPERM) &
                            ~NPY_ITFLAG_IDENTPERM;
    }
}

static void
npyiter_reverse_axis_ordering(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int ndim = NIT_NDIM(iter);
    int nop = NIT_NOP(iter);

    npy_intp i, temp, size;
    npy_intp *first, *last;
    npy_int8 *perm;

    size = NIT_AXISDATA_SIZEOF(itflags, ndim, nop)/NPY_SIZEOF_INTP;
    first = (npy_intp*)NIT_AXISDATA(iter);
    last = first + (ndim-1)*size;

    /* This loop reverses the order of the AXISDATA array */
    while (first < last) {
        for (i = 0; i < size; ++i) {
            temp = first[i];
            first[i] = last[i];
            last[i] = temp;
        }
        first += size;
        last -= size;
    }

    /* Store the perm we applied */
    perm = NIT_PERM(iter);
    for(i = ndim-1; i >= 0; --i, ++perm) {
        *perm = (npy_int8)i;
    }

    NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_IDENTPERM;
}

static NPY_INLINE npy_intp
intp_abs(npy_intp x)
{
    return (x < 0) ? -x : x;
}

static void
npyiter_find_best_axis_ordering(NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    npy_intp ax_i0, ax_i1, ax_ipos;
    npy_int8 ax_j0, ax_j1;
    npy_int8 *perm;
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    npy_intp sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
    int permuted = 0;

    perm = NIT_PERM(iter);

    /*
     * Do a custom stable insertion sort.  Note that because
     * the AXISDATA has been reversed from C order, this
     * is sorting from smallest stride to biggest stride.
     */
    for (ax_i0 = 1; ax_i0 < ndim; ++ax_i0) {
        npy_intp *strides0;

        /* 'ax_ipos' is where perm[ax_i0] will get inserted */
        ax_ipos = ax_i0;
        ax_j0 = perm[ax_i0];

        strides0 = NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, ax_j0));
        for (ax_i1 = ax_i0-1; ax_i1 >= 0; --ax_i1) {
            int ambig = 1, shouldswap = 0;
            npy_intp *strides1;

            ax_j1 = perm[ax_i1];

            strides1 = NAD_STRIDES(NIT_INDEX_AXISDATA(axisdata, ax_j1));

            for (iop = 0; iop < nop; ++iop) {
                if (strides0[iop] != 0 && strides1[iop] != 0) {
                    if (intp_abs(strides1[iop]) <=
                                            intp_abs(strides0[iop])) {
                        /*
                         * Set swap even if it's not ambiguous already,
                         * because in the case of conflicts between
                         * different operands, C-order wins.
                         */
                        shouldswap = 0;
                    }
                    else {
                        /* Only set swap if it's still ambiguous */
                        if (ambig) {
                            shouldswap = 1;
                        }
                    }

                    /*
                     * A comparison has been done, so it's
                     * no longer ambiguous
                     */
                    ambig = 0;
                }
            }
            /*
             * If the comparison was unambiguous, either shift
             * 'ax_ipos' to 'ax_i1' or stop looking for an insertion
             * point
             */
            if (!ambig) {
                if (shouldswap) {
                    ax_ipos = ax_i1;
                }
                else {
                    break;
                }
            }
        }

        /* Insert perm[ax_i0] into the right place */
        if (ax_ipos != ax_i0) {
            for (ax_i1 = ax_i0; ax_i1 > ax_ipos; --ax_i1) {
                perm[ax_i1] = perm[ax_i1-1];
            }
            perm[ax_ipos] = ax_j0;
            permuted = 1;
        }
    }

    /* Apply the computed permutation to the AXISDATA array */
    if (permuted == 1) {
        npy_intp i, size = sizeof_axisdata/NPY_SIZEOF_INTP;
        NpyIter_AxisData *ad_i;

        /* Use the index as a flag, set each to 1 */
        ad_i = axisdata;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(ad_i, 1)) {
            NAD_INDEX(ad_i) = 1;
        }
        /* Apply the permutation by following the cycles */
        for (idim = 0; idim < ndim; ++idim) {
            ad_i = NIT_INDEX_AXISDATA(axisdata, idim);

            /* If this axis hasn't been touched yet, process it */
            if (NAD_INDEX(ad_i) == 1) {
                npy_int8 pidim = perm[idim];
                npy_intp tmp;
                NpyIter_AxisData *ad_p, *ad_q;

                if (pidim != idim) {
                    /* Follow the cycle, copying the data */
                    for (i = 0; i < size; ++i) {
                        pidim = perm[idim];
                        ad_q = ad_i;
                        tmp = *((npy_intp*)ad_q + i);
                        while (pidim != idim) {
                            ad_p = NIT_INDEX_AXISDATA(axisdata, pidim);
                            *((npy_intp*)ad_q + i) = *((npy_intp*)ad_p + i);

                            ad_q = ad_p;
                            pidim = perm[(int)pidim];
                        }
                        *((npy_intp*)ad_q + i) = tmp;
                    }
                    /* Follow the cycle again, marking it as done */
                    pidim = perm[idim];

                    while (pidim != idim) {
                        NAD_INDEX(NIT_INDEX_AXISDATA(axisdata, pidim)) = 0;
                        pidim = perm[(int)pidim];
                    }
                }
                NAD_INDEX(ad_i) = 0;
            }
        }
        /* Clear the identity perm flag */
        NIT_ITFLAGS(iter) &= ~NPY_ITFLAG_IDENTPERM;
    }
}

/*
 * Calculates a dtype that all the types can be promoted to, using the
 * ufunc rules.  If only_inputs is 1, it leaves any operands that
 * are not read from out of the calculation.
 */
static PyArray_Descr *
npyiter_get_common_dtype(int nop, PyArrayObject **op,
                        const npyiter_opitflags *op_itflags, PyArray_Descr **op_dtype,
                        PyArray_Descr **op_request_dtypes,
                        int only_inputs)
{
    int iop;
    npy_intp narrs = 0, ndtypes = 0;
    PyArrayObject *arrs[NPY_MAXARGS];
    PyArray_Descr *dtypes[NPY_MAXARGS];
    PyArray_Descr *ret;

    NPY_IT_DBG_PRINT("Iterator: Getting a common data type from operands\n");

    for (iop = 0; iop < nop; ++iop) {
        if (op_dtype[iop] != NULL &&
                    (!only_inputs || (op_itflags[iop] & NPY_OP_ITFLAG_READ))) {
            /* If no dtype was requested and the op is a scalar, pass the op */
            if ((op_request_dtypes == NULL ||
                            op_request_dtypes[iop] == NULL) &&
                                            PyArray_NDIM(op[iop]) == 0) {
                arrs[narrs++] = op[iop];
            }
            /* Otherwise just pass in the dtype */
            else {
                dtypes[ndtypes++] = op_dtype[iop];
            }
        }
    }

    if (narrs == 0) {
        npy_intp i;
        ret = dtypes[0];
        for (i = 1; i < ndtypes; ++i) {
            if (ret != dtypes[i])
                break;
        }
        if (i == ndtypes) {
            if (ndtypes == 1 || PyArray_ISNBO(ret->byteorder)) {
                Py_INCREF(ret);
            }
            else {
                ret = PyArray_DescrNewByteorder(ret, NPY_NATIVE);
            }
        }
        else {
            ret = PyArray_ResultType(narrs, arrs, ndtypes, dtypes);
        }
    }
    else {
        ret = PyArray_ResultType(narrs, arrs, ndtypes, dtypes);
    }

    return ret;
}

/*
 * Allocates a temporary array which can be used to replace op
 * in the iteration.  Its dtype will be op_dtype.
 *
 * The result array has a memory ordering which matches the iterator,
 * which may or may not match that of op.  The parameter 'shape' may be
 * NULL, in which case it is filled in from the iterator's shape.
 *
 * This function must be called before any axes are coalesced.
 */
static HPy
hnpyiter_new_temp_array(HPyContext *ctx, NpyIter *iter, HPy subtype,
                npy_uint32 flags, npyiter_opitflags *op_itflags,
                int op_ndim, npy_intp const *shape,
                HPy op_dtype, const int *op_axes)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int used_op_ndim;
    int nop = NIT_NOP(iter);

    const npy_intp op_dtype_elsize = PyArray_Descr_AsStruct(ctx, op_dtype)->elsize;

    npy_int8 *perm = NIT_PERM(iter);
    npy_intp new_shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    npy_intp stride = op_dtype_elsize;
    NpyIter_AxisData *axisdata;
    npy_intp sizeof_axisdata;
    int i;
    
    HPy ret;

    /*
     * There is an interaction with array-dtypes here, which
     * generally works. Let's say you make an nditer with an
     * output dtype of a 2-double array. All-scalar inputs
     * will result in a 1-dimensional output with shape (2).
     * Everything still works out in the nditer, because the
     * new dimension is always added on the end, and it cares
     * about what happens at the beginning.
     */

    /* If it's a scalar, don't need to check the axes */
    if (op_ndim == 0) {
        ret = HPyArray_NewFromDescr(ctx, subtype, op_dtype, 0,
                               NULL, NULL, NULL, 0, HPy_NULL);
        return ret;
    }

    axisdata = NIT_AXISDATA(iter);
    sizeof_axisdata = NIT_AXISDATA_SIZEOF(itflags, ndim, nop);

    /* Initialize the strides to invalid values */
    for (i = 0; i < op_ndim; ++i) {
        strides[i] = NPY_MAX_INTP;
    }

    if (op_axes != NULL) {
        used_op_ndim = 0;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            npy_bool reduction_axis;

            /* Apply the perm to get the original axis */
            i = npyiter_undo_iter_axis_perm(idim, ndim, perm, NULL);
            i = npyiter_get_op_axis(op_axes[i], &reduction_axis);

            if (i >= 0) {
                NPY_IT_DBG_PRINT3("Iterator: Setting allocated stride %d "
                                    "for iterator dimension %d to %d\n", (int)i,
                                    (int)idim, (int)stride);
                used_op_ndim += 1;
                strides[i] = stride;
                if (shape == NULL) {
                    if (reduction_axis) {
                        /* reduction axes always have a length of 1 */
                        new_shape[i] = 1;
                    }
                    else {
                        new_shape[i] = NAD_SHAPE(axisdata);
                    }
                    stride *= new_shape[i];
                    if (i >= ndim) {
                        HPyErr_Format_p(ctx, ctx->h_ValueError,
                                "automatically allocated output array "
                                "specified with an inconsistent axis mapping; "
                                "the axis mapping cannot include dimension %d "
                                "which is too large for the iterator dimension "
                                "of %d.", i, ndim);
                        return HPy_NULL;
                    }
                }
                else {
                    assert(!reduction_axis || shape[i] == 1);
                    stride *= shape[i];
                }
            }
            else {
                if (shape == NULL) {
                    /*
                     * If deleting this axis produces a reduction, but
                     * reduction wasn't enabled, throw an error.
                     * NOTE: We currently always allow new-axis if the iteration
                     *       size is 1 (thus allowing broadcasting sometimes).
                     */
                    if (!reduction_axis && NAD_SHAPE(axisdata) != 1) {
                        if (!npyiter_check_reduce_ok_and_set_flags(
                                iter, flags, op_itflags, i)) {
                            return HPy_NULL;
                        }
                    }
                }
            }
        }
    }
    else {
        used_op_ndim = ndim;
        for (idim = 0; idim < ndim; ++idim, NIT_ADVANCE_AXISDATA(axisdata, 1)) {
            /* Apply the perm to get the original axis */
            i = npyiter_undo_iter_axis_perm(idim, op_ndim, perm, NULL);

            if (i >= 0) {
                NPY_IT_DBG_PRINT3("Iterator: Setting allocated stride %d "
                                    "for iterator dimension %d to %d\n", (int)i,
                                    (int)idim, (int)stride);
                strides[i] = stride;
                if (shape == NULL) {
                    new_shape[i] = NAD_SHAPE(axisdata);
                    stride *= new_shape[i];
                }
                else {
                    stride *= shape[i];
                }
            }
        }
    }

    if (shape == NULL) {
        /* If shape was NULL, use the shape we calculated */
        op_ndim = used_op_ndim;
        shape = new_shape;
        /*
         * If there's a gap in the array's dimensions, it's an error.
         * For instance, if op_axes [0, 2] is specified, there will a place
         * in the strides array where the value is not set.
         */
        for (i = 0; i < op_ndim; i++) {
            if (strides[i] == NPY_MAX_INTP) {
                HPyErr_Format_p(ctx, ctx->h_ValueError,
                        "automatically allocated output array "
                        "specified with an inconsistent axis mapping; "
                        "the axis mapping is missing an entry for "
                        "dimension %d.", i);
                return HPy_NULL;
            }
        }
    }
    else if (used_op_ndim < op_ndim) {
        /*
         * If custom axes were specified, some dimensions may not have
         * been used. These are additional axes which are ignored in the
         * iterator but need to be handled here.
         */
        npy_intp factor, itemsize, new_strides[NPY_MAXDIMS];

        /* Fill in the missing strides in C order */
        factor = 1;
        itemsize = op_dtype_elsize;
        for (i = op_ndim-1; i >= 0; --i) {
            if (strides[i] == NPY_MAX_INTP) {
                new_strides[i] = factor * itemsize;
                factor *= shape[i];
            }
        }

        /*
         * Copy the missing strides, and multiply the existing strides
         * by the calculated factor.  This way, the missing strides
         * are tighter together in memory, which is good for nested
         * loops.
         */
        for (i = 0; i < op_ndim; ++i) {
            if (strides[i] == NPY_MAX_INTP) {
                strides[i] = new_strides[i];
            }
            else {
                strides[i] *= factor;
            }
        }
    }

    /* Allocate the temporary array */
    ret = HPyArray_NewFromDescr_int(ctx, subtype, op_dtype, op_ndim,
                               shape, strides, NULL, 0, HPy_NULL, HPy_NULL, 0, 0);
    if (HPy_IsNull(ret)) {
        return HPy_NULL;
    }

    /* Double-check that the subtype didn't mess with the dimensions */
    if (!HPyGlobal_Is(ctx, subtype, HPyArray_Type)) {
        PyArrayObject *ret_data = PyArrayObject_AsStruct(ctx, ret);
        /*
         * TODO: the dtype could have a subarray, which adds new dimensions
         *       to `ret`, that should typically be fine, but will break
         *       in this branch.
         */
        if (PyArray_NDIM(ret_data) != op_ndim ||
                    !PyArray_CompareLists(shape, PyArray_DIMS(ret_data), op_ndim)) {
            HPyErr_SetString(ctx, ctx->h_RuntimeError,
                    "Iterator automatic output has an array subtype "
                    "which changed the dimensions of the output");
            HPy_Close(ctx, ret);
            return HPy_NULL;
        }
    }

    return ret;
}

static int
hnpyiter_allocate_arrays(HPyContext *ctx, NpyIter *iter,
                        npy_uint32 flags,
                        HPy *op_dtype, HPy subtype,
                        const npy_uint32 *op_flags, npyiter_opitflags *op_itflags,
                        int **op_axes)
{
    // PyArray_Descr **op_dtype, PyTypeObject *subtype,
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    int idim, ndim = NIT_NDIM(iter);
    int iop, nop = NIT_NOP(iter);

    int check_writemasked_reductions = 0;

    NpyIter_BufferData *bufferdata = NULL;
    HPy *op = NIT_OPERANDS(iter);

    if (itflags & NPY_ITFLAG_BUFFER) {
        bufferdata = NIT_BUFFERDATA(iter);
    }

    if (flags & NPY_ITER_COPY_IF_OVERLAP) {
        /*
         * Perform operand memory overlap checks, if requested.
         *
         * If any write operand has memory overlap with any read operand,
         * eliminate all overlap by making temporary copies, by enabling
         * NPY_OP_ITFLAG_FORCECOPY for the write operand to force WRITEBACKIFCOPY.
         *
         * Operands with NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE enabled are not
         * considered overlapping if the arrays are exactly the same. In this
         * case, the iterator loops through them in the same order element by
         * element.  (As usual, the user-provided inner loop is assumed to be
         * able to deal with this level of simple aliasing.)
         */
        for (iop = 0; iop < nop; ++iop) {
            int may_share_memory = 0;
            int iother;

            if (HPy_IsNull(op[iop])) {
                /* Iterator will always allocate */
                continue;
            }

            if (!(op_itflags[iop] & NPY_OP_ITFLAG_WRITE)) {
                /*
                 * Copy output operands only, not inputs.
                 * A more sophisticated heuristic could be
                 * substituted here later.
                 */
                continue;
            }

            for (iother = 0; iother < nop; ++iother) {
                if (iother == iop || HPy_IsNull(op[iother])) {
                    continue;
                }

                if (!(op_itflags[iother] & NPY_OP_ITFLAG_READ)) {
                    /* No data dependence for arrays not read from */
                    continue;
                }

                if (op_itflags[iother] & NPY_OP_ITFLAG_FORCECOPY) {
                    /* Already copied */
                    continue;
                }

                /*
                 * If the arrays are views to exactly the same data, no need
                 * to make copies, if the caller (eg ufunc) says it accesses
                 * data only in the iterator order.
                 *
                 * However, if there is internal overlap (e.g. a zero stride on
                 * a non-unit dimension), a copy cannot be avoided.
                 */
                PyArrayObject *op_data = PyArrayObject_AsStruct(ctx, op[iop]);
                if ((op_flags[iop] & NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE) &&
                    (op_flags[iother] & NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE) &&
                    PyArray_BYTES(op_data) == PyArray_BYTES(PyArrayObject_AsStruct(ctx, op[iother])) &&
                    PyArray_NDIM(op_data) == PyArray_NDIM(PyArrayObject_AsStruct(ctx, op[iother])) &&
                    PyArray_CompareLists(PyArray_DIMS(op_data),
                                         PyArray_DIMS(PyArrayObject_AsStruct(ctx, op[iother])),
                                         PyArray_NDIM(op_data)) &&
                    PyArray_CompareLists(PyArray_STRIDES(op_data),
                                         PyArray_STRIDES(PyArrayObject_AsStruct(ctx, op[iother])),
                                         PyArray_NDIM(op_data))) {
                    HPy op_iop_descr = HPyArray_GetDescr(ctx, op[iother]);
                    HPy op_iother_descr = HPyArray_GetDescr(ctx, op[iother]);
                    int same_descr = HPy_Is(ctx, op_iop_descr, op_iother_descr);
                    HPy_Close(ctx, op_iop_descr);
                    HPy_Close(ctx, op_iother_descr);
                    
                    if (same_descr && solve_may_have_internal_overlap(op_data, 1) == 0) {
                        continue;
                    }
                }

                /*
                 * Use max work = 1. If the arrays are large, it might
                 * make sense to go further.
                 */
                may_share_memory = solve_may_share_memory(PyArrayObject_AsStruct(ctx, op[iop]),
                                                          PyArrayObject_AsStruct(ctx, op[iother]),
                                                          1);

                if (may_share_memory) {
                    op_itflags[iop] |= NPY_OP_ITFLAG_FORCECOPY;
                    break;
                }
            }
        }
    }

    for (iop = 0; iop < nop; ++iop) {
        /*
         * Check whether there are any WRITEMASKED REDUCE operands
         * which should be validated after all the strides are filled
         * in.
         */
        if ((op_itflags[iop] &
                (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) ==
                        (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) {
            check_writemasked_reductions = 1;
        }

        /* NULL means an output the iterator should allocate */
        if (HPy_IsNull(op[iop])) {
            HPy out;
            HPy op_subtype;
            int no_subtype = op_flags[iop] & NPY_ITER_NO_SUBTYPE;

            /* Check whether the subtype was disabled */
            op_subtype = no_subtype ? HPyGlobal_Load(ctx, HPyArray_Type) : subtype;

            /*
             * Allocate the output array.
             *
             * Note that here, ndim is always correct if no op_axes was given
             * (but the actual dimension of op can be larger). If op_axes
             * is given, ndim is not actually used.
             */
            out = hnpyiter_new_temp_array(ctx, iter, op_subtype,
                                        flags, &op_itflags[iop],
                                        ndim,
                                        NULL,
                                        op_dtype[iop],
                                        op_axes ? op_axes[iop] : NULL);
            /* close loaded array type if necessary */
            if (no_subtype) {
                HPy_Close(ctx, op_subtype);
            }

            if (HPy_IsNull(out)) {
                return 0;
            }

            // op[iop] = HPy_FromPyObject(ctx, (PyObject *) out);

            /*
             * Now we need to replace the pointers and strides with values
             * from the new array.
             */
            PyArrayObject *out_data = PyArrayObject_AsStruct(ctx, out);
            npyiter_replace_axisdata(iter, iop, out_data, ndim,
                    op_axes ? op_axes[iop] : NULL);

            /*
             * New arrays are guaranteed true-aligned, but copy/cast code
             * needs uint-alignment in addition.
             */
            if (HIsUintAligned(ctx, out, out_data)) {
                op_itflags[iop] |= NPY_OP_ITFLAG_ALIGNED;
            }
            op[iop] = out;
            /* New arrays need no cast */
            op_itflags[iop] &= ~NPY_OP_ITFLAG_CAST;
        }
        /*
         * If casting is required, the operand is read-only, and
         * it's an array scalar, make a copy whether or not the
         * copy flag is enabled.
         */
        else if ((op_itflags[iop] & (NPY_OP_ITFLAG_CAST |
                         NPY_OP_ITFLAG_READ |
                         NPY_OP_ITFLAG_WRITE)) == (NPY_OP_ITFLAG_CAST |
                                                   NPY_OP_ITFLAG_READ) &&
                          PyArray_NDIM(PyArrayObject_AsStruct(ctx, op[iop])) == 0) {
            PyArrayObject *temp;            
            PyArray_Descr *py_op_dtype_iop = (PyArray_Descr *) HPy_AsPyObject(ctx, op_dtype[iop]); // implicit increfcnt
            temp = (PyArrayObject *)PyArray_NewFromDescr(
                                        &PyArray_Type, py_op_dtype_iop,
                                        0, NULL, NULL, NULL, 0, NULL);
            if (temp == NULL) {
                return 0;
            }
            PyArrayObject *py_op_iop = (PyArrayObject *) HPy_AsPyObject(ctx, op[iop]);
            CAPI_WARN("hnpyiter_allocate_arrays");
            if (PyArray_CopyInto(temp, py_op_iop) != 0) {
                Py_DECREF(py_op_iop);
                Py_DECREF(temp);
                return 0;
            }
            Py_DECREF(py_op_iop);
            HPy_Close(ctx, op[iop]);
            op[iop] = HPy_FromPyObject(ctx, (PyObject *) temp);
            Py_DECREF(temp);

            /*
             * Now we need to replace the pointers and strides with values
             * from the temporary array.
             */
            CAPI_WARN("hnpyiter_allocate_arrays");
            py_op_iop = (PyArrayObject *) HPy_AsPyObject(ctx, op[iop]);
            npyiter_replace_axisdata(iter, iop, py_op_iop, 0, NULL);
            Py_DECREF(py_op_iop);

            /*
             * New arrays are guaranteed true-aligned, but copy/cast code
             * needs uint-alignment in addition.
             */
            if (HIsUintAligned(ctx, op[iop], PyArrayObject_AsStruct(ctx, op[iop]))) {
                op_itflags[iop] |= NPY_OP_ITFLAG_ALIGNED;
            }
            /*
             * New arrays need no cast, and in the case
             * of scalars, always have stride 0 so never need buffering
             */
            op_itflags[iop] |= NPY_OP_ITFLAG_BUFNEVER;
            op_itflags[iop] &= ~NPY_OP_ITFLAG_CAST;
            if (itflags & NPY_ITFLAG_BUFFER) {
                NBF_STRIDES(bufferdata)[iop] = 0;
            }
        }
        /*
         * Make a temporary copy if,
         * 1. If casting is required and permitted, or,
         * 2. If force-copy is requested
         */
        else if (((op_itflags[iop] & NPY_OP_ITFLAG_CAST) &&
                        (op_flags[iop] &
                        (NPY_ITER_COPY|NPY_ITER_UPDATEIFCOPY))) ||
                 (op_itflags[iop] & NPY_OP_ITFLAG_FORCECOPY)) {
            HPy temp;
            PyArrayObject *op_iop_data = PyArrayObject_AsStruct(ctx, op[iop]);
            int ondim = PyArray_NDIM(op_iop_data);

            /* Allocate the temporary array, if possible */
            HPy h_PyArray_Type = HPyGlobal_Load(ctx, HPyArray_Type);
            temp = hnpyiter_new_temp_array(ctx, iter, h_PyArray_Type,
                                        flags, &op_itflags[iop],
                                        ondim,
                                        PyArray_DIMS(op_iop_data),
                                        op_dtype[iop],
                                        op_axes ? op_axes[iop] : NULL);
            HPy_Close(ctx, h_PyArray_Type);
            if (HPy_IsNull(temp)) {
                return 0;
            }

            PyArrayObject *py_temp = (PyArrayObject *) HPy_AsPyObject(ctx, temp);

            /*
             * If the data will be read, copy it into temp.
             * TODO: It might be possible to do a view into
             *       op[iop]'s mask instead here.
             */
            if (op_itflags[iop] & NPY_OP_ITFLAG_READ) {
                PyArrayObject *py_op_iop = (PyArrayObject *) HPy_AsPyObject(ctx, op[iop]);
                CAPI_WARN("hnpyiter_allocate_arrays");
                if (PyArray_CopyInto(py_temp, py_op_iop) != 0) {
                    Py_DECREF(py_op_iop);
                    Py_DECREF(py_temp);
                    return 0;
                }
                Py_DECREF(py_op_iop);
            }
            /* If the data will be written to, set WRITEBACKIFCOPY
               and require a context manager */
            PyArrayObject *op_iop = NULL;
            if (op_itflags[iop] & NPY_OP_ITFLAG_WRITE) {
                op_iop = (PyArrayObject *) HPy_AsPyObject(ctx, op[iop]);
                CAPI_WARN("hnpyiter_allocate_arrays");
                if (PyArray_SetWritebackIfCopyBase(py_temp, op_iop) < 0) {
                    Py_DECREF(py_temp);
                    return 0;
                }
                op_itflags[iop] |= NPY_OP_ITFLAG_HAS_WRITEBACK;
            }

            Py_XDECREF(op_iop);
            op[iop] = HPy_FromPyObject(ctx, (PyObject *) py_temp);
            Py_DECREF(py_temp);

            /*
             * Now we need to replace the pointers and strides with values
             * from the temporary array.
             */
            op_iop = (PyArrayObject *) HPy_AsPyObject(ctx, op[iop]);
            CAPI_WARN("hnpyiter_allocate_arrays");
            npyiter_replace_axisdata(iter, iop, op_iop, ondim,
                    op_axes ? op_axes[iop] : NULL);
            Py_DECREF(op_iop);

            /*
             * New arrays are guaranteed true-aligned, but copy/cast code
             * additionally needs uint-alignment in addition.
             */
            if (HIsUintAligned(ctx, op[iop], PyArrayObject_AsStruct(ctx, op[iop]))) {
                op_itflags[iop] |= NPY_OP_ITFLAG_ALIGNED;
            }
            /* The temporary copy needs no cast */
            op_itflags[iop] &= ~NPY_OP_ITFLAG_CAST;
        }
        else {
            /*
             * Buffering must be enabled for casting/conversion if copy
             * wasn't specified.
             */
            if ((op_itflags[iop] & NPY_OP_ITFLAG_CAST) &&
                                  !(itflags & NPY_ITFLAG_BUFFER)) {
                HPyErr_SetString(ctx, ctx->h_TypeError,
                        "Iterator operand required copying or buffering, "
                        "but neither copying nor buffering was enabled");
                return 0;
            }

            /*
             * If the operand is aligned, any buffering can use aligned
             * optimizations.
             */
            if (HIsUintAligned(ctx, op[iop], PyArrayObject_AsStruct(ctx, op[iop]))) {
                op_itflags[iop] |= NPY_OP_ITFLAG_ALIGNED;
            }
        }

        /* Here we can finally check for contiguous iteration */
        if (op_flags[iop] & NPY_ITER_CONTIG) {
            NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
            npy_intp stride = NAD_STRIDES(axisdata)[iop];

            if (stride != PyArray_Descr_AsStruct(ctx, op_dtype[iop])->elsize) {
                NPY_IT_DBG_PRINT("Iterator: Setting NPY_OP_ITFLAG_CAST "
                                    "because of NPY_ITER_CONTIG\n");
                op_itflags[iop] |= NPY_OP_ITFLAG_CAST;
                if (!(itflags & NPY_ITFLAG_BUFFER)) {
                    HPyErr_SetString(ctx, ctx->h_TypeError,
                            "Iterator operand required buffering, "
                            "to be contiguous as requested, but "
                            "buffering is not enabled");
                    return 0;
                }
            }
        }

        /*
         * If no alignment, byte swap, or casting is needed,
         * the inner stride of this operand works for the whole
         * array, we can set NPY_OP_ITFLAG_BUFNEVER.
         */
        if ((itflags & NPY_ITFLAG_BUFFER) &&
                                !(op_itflags[iop] & NPY_OP_ITFLAG_CAST)) {
            NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
            if (ndim <= 1) {
                op_itflags[iop] |= NPY_OP_ITFLAG_BUFNEVER;
                NBF_STRIDES(bufferdata)[iop] = NAD_STRIDES(axisdata)[iop];
            }
            else if (PyArray_NDIM(PyArrayObject_AsStruct(ctx, op[iop])) > 0) {
                npy_intp stride, shape, innerstride = 0, innershape;
                npy_intp sizeof_axisdata =
                                    NIT_AXISDATA_SIZEOF(itflags, ndim, nop);
                /* Find stride of the first non-empty shape */
                for (idim = 0; idim < ndim; ++idim) {
                    innershape = NAD_SHAPE(axisdata);
                    if (innershape != 1) {
                        innerstride = NAD_STRIDES(axisdata)[iop];
                        break;
                    }
                    NIT_ADVANCE_AXISDATA(axisdata, 1);
                }
                ++idim;
                NIT_ADVANCE_AXISDATA(axisdata, 1);
                /* Check that everything could have coalesced together */
                for (; idim < ndim; ++idim) {
                    stride = NAD_STRIDES(axisdata)[iop];
                    shape = NAD_SHAPE(axisdata);
                    if (shape != 1) {
                        /*
                         * If N times the inner stride doesn't equal this
                         * stride, the multi-dimensionality is needed.
                         */
                        if (innerstride*innershape != stride) {
                            break;
                        }
                        else {
                            innershape *= shape;
                        }
                    }
                    NIT_ADVANCE_AXISDATA(axisdata, 1);
                }
                /*
                 * If we looped all the way to the end, one stride works.
                 * Set that stride, because it may not belong to the first
                 * dimension.
                 */
                if (idim == ndim) {
                    op_itflags[iop] |= NPY_OP_ITFLAG_BUFNEVER;
                    NBF_STRIDES(bufferdata)[iop] = innerstride;
                }
            }
        }
    }

    if (check_writemasked_reductions) {
        for (iop = 0; iop < nop; ++iop) {
            /*
             * Check whether there are any WRITEMASKED REDUCE operands
             * which should be validated now that all the strides are filled
             * in.
             */
            if ((op_itflags[iop] &
                    (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) ==
                        (NPY_OP_ITFLAG_WRITEMASKED | NPY_OP_ITFLAG_REDUCE)) {
                /*
                 * If the ARRAYMASK has 'bigger' dimensions
                 * than this REDUCE WRITEMASKED operand,
                 * the result would be more than one mask
                 * value per reduction element, something which
                 * is invalid. This function provides validation
                 * for that.
                 */
                if (!check_mask_for_writemasked_reduction(iter, iop)) {
                    return 0;
                }
            }
        }
    }

    return 1;
}

/*
 * The __array_priority__ attribute of the inputs determines
 * the subtype of any output arrays.  This function finds the
 * subtype of the input array with highest priority.
 */
static void
npyiter_get_priority_subtype(int nop, PyArrayObject **op,
                            const npyiter_opitflags *op_itflags,
                            double *subtype_priority,
                            PyTypeObject **subtype)
{
    int iop;

    for (iop = 0; iop < nop; ++iop) {
        if (op[iop] != NULL && op_itflags[iop] & NPY_OP_ITFLAG_READ) {
            double priority = PyArray_GetPriority((PyObject *)op[iop], 0.0);
            if (priority > *subtype_priority) {
                *subtype_priority = priority;
                *subtype = Py_TYPE(op[iop]);
            }
        }
    }
}

static int
hnpyiter_allocate_transfer_functions(HPyContext *ctx, NpyIter *iter)
{
    npy_uint32 itflags = NIT_ITFLAGS(iter);
    /*int ndim = NIT_NDIM(iter);*/
    int iop = 0, nop = NIT_NOP(iter);

    npy_intp i;
    npyiter_opitflags *op_itflags = NIT_OPITFLAGS(iter);
    NpyIter_BufferData *bufferdata = NIT_BUFFERDATA(iter);
    NpyIter_AxisData *axisdata = NIT_AXISDATA(iter);
    // PyArrayObject **op = NIT_OPERANDS(iter);
    // PyArray_Descr **op_dtype = NIT_DTYPES(iter);
    HPy *op = NIT_OPERANDS(iter);
    HPy *op_dtype = NIT_DTYPES(iter);
    npy_intp *strides = NAD_STRIDES(axisdata), op_stride;
    NpyIter_TransferInfo *transferinfo = NBF_TRANSFERINFO(bufferdata);

    int needs_api = 0;

    for (iop = 0; iop < nop; ++iop) {
        npyiter_opitflags flags = op_itflags[iop];
        PyArrayObject *op_iop_data = PyArrayObject_AsStruct(ctx, op[iop]);
        PyArray_Descr *op_dtype_iop_data = PyArray_Descr_AsStruct(ctx, op_dtype[iop]);
        /*
         * Reduction operands may be buffered with a different stride,
         * so we must pass NPY_MAX_INTP to the transfer function factory.
         */
        op_stride = (flags & NPY_OP_ITFLAG_REDUCE) ? NPY_MAX_INTP :
                                                   strides[iop];

        /*
         * If we have determined that a buffer may be needed,
         * allocate the appropriate transfer functions
         */
        if (!(flags & NPY_OP_ITFLAG_BUFNEVER)) {
            if (flags & NPY_OP_ITFLAG_READ) {
                int move_references = 0;
                HPy op_iop_dtype = HPyArray_DESCR(ctx, op[iop], op_iop_data);
                if (HPyArray_GetDTypeTransferFunction(ctx,
                                        (flags & NPY_OP_ITFLAG_ALIGNED) != 0,
                                        op_stride,
                                        op_dtype_iop_data->elsize,
                                        op_iop_dtype,
                                        op_dtype[iop],
                                        move_references,
                                        &transferinfo[iop].read,
                                        &needs_api) != NPY_SUCCEED) {
                    iop -= 1;  /* This one cannot be cleaned up yet. */
                    HPy_Close(ctx, op_iop_dtype);
                    goto fail;
                }
                HPy_Close(ctx, op_iop_dtype);
            }
            else {
                transferinfo[iop].read.func = NULL;
            }
            if (flags & NPY_OP_ITFLAG_WRITE) {
                int move_references = 1;
                HPy op_iop_dtype = HPyArray_DESCR(ctx, op[iop], op_iop_data);

                /* If the operand is WRITEMASKED, use a masked transfer fn */
                if (flags & NPY_OP_ITFLAG_WRITEMASKED) {
                    int maskop = NIT_MASKOP(iter);
                    HPy mask_dtype = HPyArray_GetDescr(ctx, op[maskop]);
                    int mask_dtype_elsize = PyArray_Descr_AsStruct(ctx, mask_dtype)->elsize;


                    /*
                     * If the mask's stride is contiguous, use it, otherwise
                     * the mask may or may not be buffered, so the stride
                     * could be inconsistent.
                     */
                    if (HPyArray_GetMaskedDTypeTransferFunction(ctx,
                            (flags & NPY_OP_ITFLAG_ALIGNED) != 0,
                            op_dtype_iop_data->elsize,
                            op_stride,
                            (strides[maskop] == mask_dtype_elsize) ?
                                mask_dtype_elsize : NPY_MAX_INTP,
                            op_dtype[iop],
                            op_iop_dtype,
                            mask_dtype,
                            move_references,
                            &transferinfo[iop].write,
                            &needs_api) != NPY_SUCCEED) {
                        HPy_Close(ctx, op_iop_dtype);
                        HPy_Close(ctx, mask_dtype);
                        goto fail;
                    }
                    HPy_Close(ctx, mask_dtype);
                }
                else {
                    if (HPyArray_GetDTypeTransferFunction(ctx,
                            (flags & NPY_OP_ITFLAG_ALIGNED) != 0,
                            op_dtype_iop_data->elsize,
                            op_stride,
                            op_dtype[iop],
                            op_iop_dtype,
                            move_references,
                            &transferinfo[iop].write,
                            &needs_api) != NPY_SUCCEED) {
                        HPy_Close(ctx, op_iop_dtype);
                        goto fail;
                    }
                }
                HPy_Close(ctx, op_iop_dtype);
            }
            /* If no write back but there are references make a decref fn */
            else if (PyDataType_REFCHK(op_dtype_iop_data)) {
                /*
                 * By passing NULL to dst_type and setting move_references
                 * to 1, we get back a function that just decrements the
                 * src references.
                 */
                if (HPyArray_GetDTypeTransferFunction(ctx,
                        (flags & NPY_OP_ITFLAG_ALIGNED) != 0,
                        op_dtype_iop_data->elsize, 0,
                        op_dtype[iop], HPy_NULL,
                        1,
                        &transferinfo[iop].write,
                        &needs_api) != NPY_SUCCEED) {
                    goto fail;
                }
            }
            else {
                transferinfo[iop].write.func = NULL;
            }
        }
        else {
            transferinfo[iop].read.func = NULL;
            transferinfo[iop].write.func = NULL;
        }
    }

    /* If any of the dtype transfer functions needed the API, flag it */
    if (needs_api) {
        NIT_ITFLAGS(iter) |= NPY_ITFLAG_NEEDSAPI;
    }

    return 1;

fail:
    for (i = 0; i < iop+1; ++i) {
        NPY_cast_info_xfree(&transferinfo[iop].read);
        NPY_cast_info_xfree(&transferinfo[iop].write);
    }
    return 0;
}

#undef NPY_ITERATOR_IMPLEMENTATION_CODE

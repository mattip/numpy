/*
 * This file implements the CPython wrapper of NpyIter
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "alloc.h"
#include "common.h"
#include "conversion_utils.h"
#include "ctors.h"

#include "nditer_hpy.h"

/* Functions not part of the public NumPy C API */
npy_bool npyiter_has_writeback(NpyIter *iter);


typedef struct NewNpyArrayIterObject_tag NewNpyArrayIterObject;

struct NewNpyArrayIterObject_tag {
    PyObject_HEAD
    /* The iterator */
    NpyIter *iter;
    /* Flag indicating iteration started/stopped */
    char started, finished;
    /* Child to update for nested iteration */
    HPyField nested_child; // NewNpyArrayIterObject *
    /* Cached values from the iterator */
    NpyIter_IterNextFunc *iternext;
    NpyIter_GetMultiIndexFunc *get_multi_index;
    char **dataptrs;
    HPyField *dtypes; // PyArray_Descr **
    HPyField *operands; // PyArrayObject **
    npy_intp *innerstrides, *innerloopsizeptr;
    char readflags[NPY_MAXARGS];
    char writeflags[NPY_MAXARGS];
};

HPyType_LEGACY_HELPERS(NewNpyArrayIterObject)

static int npyiter_cache_values(HPyContext *ctx, HPy h_self, NewNpyArrayIterObject *self)
{
    NpyIter *iter = self->iter;

    /* iternext and get_multi_index functions */
    self->iternext = HNpyIter_GetIterNext(ctx, iter, NULL);
    if (self->iternext == NULL) {
        return -1;
    }

    if (NpyIter_HasMultiIndex(iter) && !NpyIter_HasDelayedBufAlloc(iter)) {
        self->get_multi_index = HNpyIter_GetGetMultiIndex(ctx, iter, NULL);
    }
    else {
        self->get_multi_index = NULL;
    }

    /* Internal data pointers */
    self->dataptrs = NpyIter_GetDataPtrArray(iter);
    int nop = NpyIter_GetNOp(iter);
    HPy_CloseAndFreeFieldArray(ctx, h_self, self->dtypes, nop);
    HPy_CloseAndFreeFieldArray(ctx, h_self, self->operands, nop);
    HPy *h_dtypes = HNpyIter_GetDescrArray(iter);
    self->dtypes = (HPyField *)calloc(nop, sizeof(HPyField));
    HPy *h_operands = HNpyIter_GetOperandArray(iter);
    self->operands = (HPyField *)calloc(nop, sizeof(HPyField));
    for (npy_int i = 0; i < nop; i++) {
        HPyField_Store(ctx, h_self, &self->dtypes[i], h_dtypes[i]);
        HPyField_Store(ctx, h_self, &self->operands[i], h_operands[i]);
    }
    
    if (NpyIter_HasExternalLoop(iter)) {
        self->innerstrides = NpyIter_GetInnerStrideArray(iter);
        self->innerloopsizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    }
    else {
        self->innerstrides = NULL;
        self->innerloopsizeptr = NULL;
    }

    /* The read/write settings */
    NpyIter_GetReadFlags(iter, self->readflags);
    NpyIter_GetWriteFlags(iter, self->writeflags);
    return 0;
}

HPyDef_SLOT(npyiter_new, HPy_tp_new)
static HPy
npyiter_new_impl(HPyContext *ctx, HPy h_subtype, const HPy *NPY_UNUSED(args_h),
                          HPy_ssize_t NPY_UNUSED(nargs), HPy NPY_UNUSED(kwds))
{
    NewNpyArrayIterObject *self;

    HPy h_self = HPy_New(ctx, h_subtype, &self);
    if (!HPy_IsNull(h_self)) {
        self->iter = NULL;
        self->nested_child = HPyField_NULL;
    }

    return h_self;
}

static int
HNpyIter_GlobalFlagsConverter(HPyContext *ctx, HPy flags_in, npy_uint32 *flags)
{
    npy_uint32 tmpflags = 0;
    int iflags, nflags;

    HPy f;
    const char *str = NULL;
    HPy_ssize_t length = 0;
    npy_uint32 flag;

    if (HPy_IsNull(flags_in) || HPy_Is(ctx, flags_in, ctx->h_None)) {
        return 1;
    }

    if (!HPyTuple_Check(ctx, flags_in) && !HPyList_Check(ctx, flags_in)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator global flags must be a list or tuple of strings");
        return 0;
    }

    nflags = HPy_Length(ctx, flags_in);

    for (iflags = 0; iflags < nflags; ++iflags) {
        f = HPy_GetItem_i(ctx, flags_in, iflags);
        if (HPy_IsNull(f)) {
            return 0;
        }

        if (HPyUnicode_Check(ctx, f)) {
            /* accept unicode input */
            HPy f_str;
            f_str = HPyUnicode_AsASCIIString(ctx, f);
            if (HPy_IsNull(f_str)) {
                HPy_Close(ctx, f);
                return 0;
            }
            HPy_Close(ctx, f);
            f = f_str;
        }

        if (!HPyBytes_Check(ctx, f)) {
            HPy_Close(ctx, f);
            return 0;
        }
        str = HPyBytes_AS_STRING(ctx, f);
        length = HPyBytes_GET_SIZE(ctx, f);
        /* Use switch statements to quickly isolate the right flag */
        flag = 0;
        switch (str[0]) {
            case 'b':
                if (strcmp(str, "buffered") == 0) {
                    flag = NPY_ITER_BUFFERED;
                }
                break;
            case 'c':
                if (length >= 6) switch (str[5]) {
                    case 'e':
                        if (strcmp(str, "c_index") == 0) {
                            flag = NPY_ITER_C_INDEX;
                        }
                        break;
                    case 'i':
                        if (strcmp(str, "copy_if_overlap") == 0) {
                            flag = NPY_ITER_COPY_IF_OVERLAP;
                        }
                        break;
                    case 'n':
                        if (strcmp(str, "common_dtype") == 0) {
                            flag = NPY_ITER_COMMON_DTYPE;
                        }
                        break;
                }
                break;
            case 'd':
                if (strcmp(str, "delay_bufalloc") == 0) {
                    flag = NPY_ITER_DELAY_BUFALLOC;
                }
                break;
            case 'e':
                if (strcmp(str, "external_loop") == 0) {
                    flag = NPY_ITER_EXTERNAL_LOOP;
                }
                break;
            case 'f':
                if (strcmp(str, "f_index") == 0) {
                    flag = NPY_ITER_F_INDEX;
                }
                break;
            case 'g':
                /*
                 * Documentation is grow_inner, but initial implementation
                 * was growinner, so allowing for either.
                 */
                if (strcmp(str, "grow_inner") == 0 ||
                            strcmp(str, "growinner") == 0) {
                    flag = NPY_ITER_GROWINNER;
                }
                break;
            case 'm':
                if (strcmp(str, "multi_index") == 0) {
                    flag = NPY_ITER_MULTI_INDEX;
                }
                break;
            case 'r':
                if (strcmp(str, "ranged") == 0) {
                    flag = NPY_ITER_RANGED;
                }
                else if (strcmp(str, "refs_ok") == 0) {
                    flag = NPY_ITER_REFS_OK;
                }
                else if (strcmp(str, "reduce_ok") == 0) {
                    flag = NPY_ITER_REDUCE_OK;
                }
                break;
            case 'z':
                if (strcmp(str, "zerosize_ok") == 0) {
                    flag = NPY_ITER_ZEROSIZE_OK;
                }
                break;
        }
        if (flag == 0) {
            PyErr_Format(PyExc_ValueError,
                    "Unexpected iterator global flag \"%s\"", str);
            HPy_Close(ctx, f);
            return 0;
        }
        else {
            tmpflags |= flag;
        }
        HPy_Close(ctx, f);
    }

    *flags |= tmpflags;
    return 1;
}

static int
HNpyIter_OpFlagsConverter(HPyContext *ctx, HPy op_flags_in,
                         npy_uint32 *op_flags)
{
    int iflags, nflags;
    npy_uint32 flag;

    if (!HPyTuple_Check(ctx, op_flags_in) && !HPyList_Check(ctx, op_flags_in)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "op_flags must be a tuple or array of per-op flag-tuples");
        return 0;
    }

    nflags = HPy_Length(ctx, op_flags_in);

    *op_flags = 0;
    for (iflags = 0; iflags < nflags; ++iflags) {
        HPy f;
        const char *str = NULL;
        Py_ssize_t length = 0;

        f = HPy_GetItem_i(ctx, op_flags_in, iflags);
        if (HPy_IsNull(f)) {
            return 0;
        }

        if (HPyUnicode_Check(ctx, f)) {
            /* accept unicode input */
            HPy f_str;
            f_str = HPyUnicode_AsASCIIString(ctx, f);
            HPy_Close(ctx, f);
            if (HPy_IsNull(f_str)) {
                return 0;
            }
            f = f_str;
        }
        if (!HPyBytes_Check(ctx, f)) {
            HPyErr_Clear(ctx);
            HPy_Close(ctx, f);
            HPyErr_SetString(ctx, ctx->h_ValueError,
                   "op_flags must be a tuple or array of per-op flag-tuples");
            return 0;
        }
        str = HPyBytes_AS_STRING(ctx, f);
        length = HPyBytes_GET_SIZE(ctx, f);

        /* Use switch statements to quickly isolate the right flag */
        flag = 0;
        switch (str[0]) {
            case 'a':
                if (length > 2) switch(str[2]) {
                    case 'i':
                        if (strcmp(str, "aligned") == 0) {
                            flag = NPY_ITER_ALIGNED;
                        }
                        break;
                    case 'l':
                        if (strcmp(str, "allocate") == 0) {
                            flag = NPY_ITER_ALLOCATE;
                        }
                        break;
                    case 'r':
                        if (strcmp(str, "arraymask") == 0) {
                            flag = NPY_ITER_ARRAYMASK;
                        }
                        break;
                }
                break;
            case 'c':
                if (strcmp(str, "copy") == 0) {
                    flag = NPY_ITER_COPY;
                }
                if (strcmp(str, "contig") == 0) {
                    flag = NPY_ITER_CONTIG;
                }
                break;
            case 'n':
                switch (str[1]) {
                    case 'b':
                        if (strcmp(str, "nbo") == 0) {
                            flag = NPY_ITER_NBO;
                        }
                        break;
                    case 'o':
                        if (strcmp(str, "no_subtype") == 0) {
                            flag = NPY_ITER_NO_SUBTYPE;
                        }
                        else if (strcmp(str, "no_broadcast") == 0) {
                            flag = NPY_ITER_NO_BROADCAST;
                        }
                        break;
                }
                break;
            case 'o':
                if (strcmp(str, "overlap_assume_elementwise") == 0) {
                    flag = NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
                }
                break;
            case 'r':
                if (length > 4) switch (str[4]) {
                    case 'o':
                        if (strcmp(str, "readonly") == 0) {
                            flag = NPY_ITER_READONLY;
                        }
                        break;
                    case 'w':
                        if (strcmp(str, "readwrite") == 0) {
                            flag = NPY_ITER_READWRITE;
                        }
                        break;
                }
                break;
            case 'u':
                switch (str[1]) {
                    case 'p':
                        if (strcmp(str, "updateifcopy") == 0) {
                            flag = NPY_ITER_UPDATEIFCOPY;
                        }
                        break;
                }
                break;
            case 'v':
                if (strcmp(str, "virtual") == 0) {
                    flag = NPY_ITER_VIRTUAL;
                }
                break;
            case 'w':
                if (length > 5) switch (str[5]) {
                    case 'o':
                        if (strcmp(str, "writeonly") == 0) {
                            flag = NPY_ITER_WRITEONLY;
                        }
                        break;
                    case 'm':
                        if (strcmp(str, "writemasked") == 0) {
                            flag = NPY_ITER_WRITEMASKED;
                        }
                        break;
                }
                break;
        }
        if (flag == 0) {
            HPyErr_Format_p(ctx, ctx->h_ValueError,
                    "Unexpected per-op iterator flag \"%s\"", str);
            HPy_Close(ctx, f);
            return 0;
        }
        else {
            *op_flags |= flag;
        }
        HPy_Close(ctx, f);
    }

    return 1;
}

static int
npyiter_convert_op_flags_array(HPyContext *ctx, HPy op_flags_in,
                         npy_uint32 *op_flags_array, npy_intp nop)
{
    npy_intp iop;

    if (!HPyTuple_Check(ctx, op_flags_in) && !HPyList_Check(ctx, op_flags_in)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "op_flags must be a tuple or array of per-op flag-tuples");
        return 0;
    }

    if (HPy_Length(ctx, op_flags_in) != nop) {
        goto try_single_flags;
    }

    for (iop = 0; iop < nop; ++iop) {
        HPy f = HPy_GetItem_i(ctx, op_flags_in, iop);
        if (HPy_IsNull(f)) {
            return 0;
        }
        /* If the first item is a string, try as one set of flags */
        if (iop == 0 && (HPyBytes_Check(ctx, f) || HPyUnicode_Check(ctx, f))) {
            HPy_Close(ctx, f);
            goto try_single_flags;
        }
        if (HNpyIter_OpFlagsConverter(ctx, f,
                        &op_flags_array[iop]) != 1) {
            HPy_Close(ctx, f);
            return 0;
        }

        HPy_Close(ctx, f);
    }

    return 1;

try_single_flags:
    if (HNpyIter_OpFlagsConverter(ctx, op_flags_in,
                        &op_flags_array[0]) != 1) {
        return 0;
    }

    for (iop = 1; iop < nop; ++iop) {
        op_flags_array[iop] = op_flags_array[0];
    }

    return 1;
}

static int
npyiter_convert_dtypes(HPyContext *ctx, HPy op_dtypes_in,
                        HPy /* PyArray_Descr ** */ *op_dtypes,
                        npy_intp nop)
{
    npy_intp iop;

    /*
     * If the input isn't a tuple of dtypes, try converting it as-is
     * to a dtype, and replicating to all operands.
     */
    if ((!HPyTuple_Check(ctx, op_dtypes_in) && !HPyList_Check(ctx, op_dtypes_in)) ||
                                    HPy_Length(ctx, op_dtypes_in) != nop) {
        goto try_single_dtype;
    }

    for (iop = 0; iop < nop; ++iop) {
        HPy dtype = HPy_GetItem_i(ctx, op_dtypes_in, iop);
        if (HPy_IsNull(dtype)) {
            npy_intp i;
            for (i = 0; i < iop; ++i ) {
                HPy_Close(ctx, op_dtypes[i]);
            }
            return 0;
        }

        /* Try converting the object to a descr */
        if (HPyArray_DescrConverter2(ctx, dtype, &op_dtypes[iop]) != 1) {
            npy_intp i;
            for (i = 0; i < iop; ++i ) {
                HPy_Close(ctx, op_dtypes[i]);
            }
            HPy_Close(ctx, dtype);
            HPyErr_Clear(ctx);
            goto try_single_dtype;
        }

        HPy_Close(ctx, dtype);
    }

    return 1;

try_single_dtype:
    if (HPyArray_DescrConverter2(ctx, op_dtypes_in, &op_dtypes[0]) == 1) {
        for (iop = 1; iop < nop; ++iop) {
            op_dtypes[iop] = HPy_Dup(ctx, op_dtypes[0]);
            // Py_XINCREF(op_dtypes[iop]);
        }
        return 1;
    }

    return 0;
}

static int
npyiter_convert_op_axes(HPyContext *ctx, HPy op_axes_in, int nop,
                        int **op_axes, int *oa_ndim)
{
    HPy a;
    int iop;

    if ((!HPyTuple_Check(ctx, op_axes_in) && !HPyList_Check(ctx, op_axes_in)) ||
                                HPy_Length(ctx, op_axes_in) != nop) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "op_axes must be a tuple/list matching the number of ops");
        return 0;
    }

    *oa_ndim = -1;

    /* Copy the tuples into op_axes */
    for (iop = 0; iop < nop; ++iop) {
        int idim;
        a = HPy_GetItem_i(ctx, op_axes_in, iop);
        if (HPy_IsNull(a)) {
            return 0;
        }
        if (HPy_Is(ctx, a, ctx->h_None)) {
            op_axes[iop] = NULL;
        } else {
            if (!HPyTuple_Check(ctx, a) && !HPyList_Check(ctx, a)) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "Each entry of op_axes must be None "
                        "or a tuple/list");
                HPy_Close(ctx, a);
                return 0;
            }
            if (*oa_ndim == -1) {
                *oa_ndim = HPy_Length(ctx, a);
                if (*oa_ndim > NPY_MAXDIMS) {
                    HPyErr_SetString(ctx, ctx->h_ValueError,
                            "Too many dimensions in op_axes");
                    HPy_Close(ctx, a);
                    return 0;
                }
            }
            if (HPy_Length(ctx, a) != *oa_ndim) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "Each entry of op_axes must have the same size");
                HPy_Close(ctx, a);
                return 0;
            }
            for (idim = 0; idim < *oa_ndim; ++idim) {
                HPy v = HPy_GetItem_i(ctx, a, idim);
                if (HPy_IsNull(v)) {
                    HPy_Close(ctx, a);
                    return 0;
                }
                /* numpy.newaxis is None */
                if (HPy_Is(ctx, v, ctx->h_None)) {
                    op_axes[iop][idim] = -1;
                }
                else {
                    op_axes[iop][idim] = HPyArray_PyIntAsInt(ctx, v);
                    if (op_axes[iop][idim]==-1 &&
                                                PyErr_Occurred()) {
                        HPy_Close(ctx, a);
                        HPy_Close(ctx, v);
                        return 0;
                    }
                }
                HPy_Close(ctx, v);
            }
        }
        HPy_Close(ctx, a);
    }

    if (*oa_ndim == -1) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "If op_axes is provided, at least one list of axes "
                "must be contained within it");
        return 0;
    }

    return 1;
}

/*
 * Converts the operand array and op_flags array into the form
 * NpyIter_AdvancedNew needs.  Sets nop, and on success, each
 * op[i] owns a reference to an array object.
 */
static int
npyiter_convert_ops(HPyContext *ctx, HPy op_in, HPy op_flags_in,
                    HPy /* PyArrayObject ** */ *op, npy_uint32 *op_flags,
                    int *nop_out)
{
    int iop, nop;

    /* nop and op */
    if (HPyTuple_Check(ctx, op_in) || HPyList_Check(ctx, op_in)) {
        nop = HPy_Length(ctx, op_in);
        if (nop == 0) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Must provide at least one operand");
            return 0;
        }
        if (nop > NPY_MAXARGS) {
            HPyErr_SetString(ctx, ctx->h_ValueError, "Too many operands");
            return 0;
        }

        for (iop = 0; iop < nop; ++iop) {
            HPy item = HPy_GetItem_i(ctx, op_in, iop);
            if (HPy_IsNull(item)) {
                npy_intp i;
                for (i = 0; i < iop; ++i) {
                    HPy_Close(ctx, op[i]);
                }
                return 0;
            }
            else if (HPy_Is(ctx,item, ctx->h_None)) {
                HPy_Close(ctx, item);
                item = HPy_NULL;
            }
            /* This is converted to an array after op flags are retrieved */
            op[iop] = item;
        }
    }
    else {
        nop = 1;
        /* Is converted to an array after op flags are retrieved */
        // Py_INCREF(op_in);
        op[0] = HPy_Dup(ctx, op_in);
    }

    *nop_out = nop;

    /* op_flags */
    if (HPy_IsNull(op_flags_in) || HPy_Is(ctx, op_flags_in, ctx->h_None)) {
        for (iop = 0; iop < nop; ++iop) {
            /*
             * By default, make NULL operands writeonly and flagged for
             * allocation, and everything else readonly.  To write
             * to a provided operand, you must specify the write flag manually.
             */
            if (HPy_IsNull(op[iop])) {
                op_flags[iop] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
            }
            else {
                op_flags[iop] = NPY_ITER_READONLY;
            }
        }
    }
    else if (npyiter_convert_op_flags_array(ctx, op_flags_in,
                                      op_flags, nop) != 1) {
        for (iop = 0; iop < nop; ++iop) {
            HPy_Close(ctx, op[iop]);
        }
        *nop_out = 0;
        return 0;
    }

    /* Now that we have the flags - convert all the ops to arrays */
    for (iop = 0; iop < nop; ++iop) {
        if (!HPy_IsNull(op[iop])) {
            HPy ao; // PyArrayObject *
            int fromanyflags = 0;

            if (op_flags[iop]&(NPY_ITER_READWRITE|NPY_ITER_WRITEONLY)) {
                fromanyflags |= NPY_ARRAY_WRITEBACKIFCOPY;
            }
            ao = HPyArray_FROM_OF(ctx, op[iop],
                                                  fromanyflags);
            if (HPy_IsNull(ao)) {
                if (HPyErr_Occurred(ctx) &&
                            HPyErr_ExceptionMatches(ctx, ctx->h_TypeError)) {
                    HPyErr_SetString(ctx, ctx->h_TypeError,
                            "Iterator operand is flagged as writeable, "
                            "but is an object which cannot be written "
                            "back to via WRITEBACKIFCOPY");
                }
                for (iop = 0; iop < nop; ++iop) {
                    HPy_Close(ctx, op[iop]);
                }
                *nop_out = 0;
                return 0;
            }
            HPy_Close(ctx, op[iop]);
            op[iop] = ao;
        }
    }

    return 1;
}

HPyDef_SLOT(npyiter_init, HPy_tp_init)
static int
npyiter_init_impl(HPyContext *ctx, HPy h_self,
        const HPy *args, HPy_ssize_t len_args, HPy kwds)
{
    static const char *kwlist[] = {"op", "flags", "op_flags", "op_dtypes",
                             "order", "casting", "op_axes", "itershape",
                             "buffersize",
                             NULL};

    HPy op_in = HPy_NULL, op_flags_in = HPy_NULL,
                op_dtypes_in = HPy_NULL, op_axes_in = HPy_NULL;

    int iop, nop = 0;
    HPy op[NPY_MAXARGS]; // PyArrayObject *
    npy_uint32 flags = 0;
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    npy_uint32 op_flags[NPY_MAXARGS];
    HPy op_request_dtypes[NPY_MAXARGS]; // PyArray_Descr *
    int oa_ndim = -1;
    int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
    int *op_axes[NPY_MAXARGS];
    PyArray_Dims itershape = {NULL, -1};
    int buffersize = 0;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter != NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator was already initialized");
        return -1;
    }

    HPyTracker ht;
    HPy h_flags = HPy_NULL, h_order = HPy_NULL, h_casting = HPy_NULL, h_itershape = HPy_NULL;
    if (!HPyArg_ParseKeywordsDict(ctx, &ht, args, len_args, kwds, "O|OOOOOOOi:nditer", kwlist,
                    &op_in,
                    &h_flags,
                    &op_flags_in,
                    &op_dtypes_in,
                    &h_order,
                    &h_casting,
                    &op_axes_in,
                    &h_itershape,
                    &buffersize)) {
        npy_free_cache_dim_obj(itershape);
        return -1;
    }

    if (!HNpyIter_GlobalFlagsConverter(ctx, h_flags, &flags)) {
        npy_free_cache_dim_obj(itershape);
        HPyTracker_Close(ctx, ht);
        return -1;
    }

    if (!HPyArray_OrderConverter(ctx, h_order, &order)) {
        npy_free_cache_dim_obj(itershape);
        HPyTracker_Close(ctx, ht);
        return -1;
    }

    if (!HPy_IsNull(h_casting) && !HPy_Is(ctx, h_casting, ctx->h_None)) {
        if (!HPyArray_CastingConverter(ctx, h_casting, &casting)) {
            npy_free_cache_dim_obj(itershape);
            HPyTracker_Close(ctx, ht);
            return -1;
        }
    }

    if (!HPyArray_OptionalIntpConverter(ctx, h_itershape, &itershape)) {
        npy_free_cache_dim_obj(itershape);
        HPyTracker_Close(ctx, ht);
        return -1;
    }

    /* Set the dtypes and ops to all NULL to start */
    memset(op_request_dtypes, 0, sizeof(op_request_dtypes));

    /* op and op_flags */
    if (npyiter_convert_ops(ctx, op_in, op_flags_in, op, op_flags, &nop)
                                                        != 1) {
        goto fail;
    }

    /* op_request_dtypes */
    if (!HPy_IsNull(op_dtypes_in) && !HPy_Is(ctx, op_dtypes_in, ctx->h_None) &&
            npyiter_convert_dtypes(ctx, op_dtypes_in,
                                   op_request_dtypes, nop) != 1) {
        goto fail;
    }

    /* op_axes */
    if (!HPy_IsNull(op_axes_in) && !HPy_Is(ctx, op_axes_in, ctx->h_None)) {
        /* Initialize to point to the op_axes arrays */
        for (iop = 0; iop < nop; ++iop) {
            op_axes[iop] = op_axes_arrays[iop];
        }

        if (npyiter_convert_op_axes(ctx, op_axes_in, nop,
                                    op_axes, &oa_ndim) != 1) {
            goto fail;
        }
    }

    if (itershape.len != -1) {
        if (oa_ndim == -1) {
            oa_ndim = itershape.len;
            memset(op_axes, 0, sizeof(op_axes[0]) * nop);
        }
        else if (oa_ndim != itershape.len) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                        "'op_axes' and 'itershape' must have the same number "
                        "of entries equal to the iterator ndim");
            goto fail;
        }
    }

    self->iter = HNpyIter_AdvancedNew(ctx, nop, op, flags, order, casting, op_flags,
                                  op_request_dtypes,
                                  oa_ndim, oa_ndim >= 0 ? op_axes : NULL,
                                  itershape.ptr,
                                  buffersize);

    if (self->iter == NULL) {
        goto fail;
    }

    /* Cache some values for the member functions to use */
    if (npyiter_cache_values(ctx, h_self, self) < 0) {
        goto fail;
    }

    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    npy_free_cache_dim_obj(itershape);

    /* Release the references we got to the ops and dtypes */
    for (iop = 0; iop < nop; ++iop) {
        HPy_Close(ctx, op[iop]);
        HPy_Close(ctx, op_request_dtypes[iop]);
    }

    return 0;

fail:
    npy_free_cache_dim_obj(itershape);
    for (iop = 0; iop < nop; ++iop) {
        HPy_Close(ctx, op[iop]);
        HPy_Close(ctx, op_request_dtypes[iop]);
    }
    return -1;
}

HPyDef_METH(NpyIter_NestedIters, "nested_iters", HPyFunc_KEYWORDS)
NPY_NO_EXPORT HPy
NpyIter_NestedIters_impl(HPyContext *ctx, HPy NPY_UNUSED(self),
        const HPy *args, size_t len_args, HPy kwnames)
{
    static const char *kwlist[] = {"op", "axes", "flags", "op_flags",
                             "op_dtypes", "order",
                             "casting", "buffersize",
                             NULL};

    HPy op_in = HPy_NULL, axes_in = HPy_NULL,
            op_flags_in = HPy_NULL, op_dtypes_in = HPy_NULL;

    int iop, nop = 0, inest, nnest = 0;
    HPy op[NPY_MAXARGS]; // PyArrayObject *
    npy_uint32 flags = 0, flags_inner;
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    npy_uint32 op_flags[NPY_MAXARGS], op_flags_inner[NPY_MAXARGS];
    HPy op_request_dtypes[NPY_MAXARGS],
        op_request_dtypes_inner[NPY_MAXARGS]; // PyArray_Descr *
    int op_axes_data[NPY_MAXDIMS];
    int *nested_op_axes[NPY_MAXDIMS];
    int nested_naxes[NPY_MAXDIMS], iaxes, naxes;
    int negones[NPY_MAXDIMS];
    char used_axes[NPY_MAXDIMS];
    int buffersize = 0;

    HPy ret = HPy_NULL;

    HPyTracker ht;
    HPy h_flags = HPy_NULL, h_order = HPy_NULL, h_casting = HPy_NULL;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, len_args, kwnames, "OO|OOOOOi", kwlist,
                    &op_in,
                    &axes_in,
                    &h_flags,
                    &op_flags_in,
                    &op_dtypes_in,
                    &h_order,
                    &h_casting,
                    &buffersize)) {
        return HPy_NULL;
    }

    if (!HNpyIter_GlobalFlagsConverter(ctx, h_flags, &flags)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!HPyArray_OrderConverter(ctx, h_order, &order)) {
        HPyTracker_Close(ctx, ht);
        return HPy_NULL;
    }

    if (!HPy_IsNull(h_casting) && !HPy_Is(ctx, h_casting, ctx->h_None)) {
        if (!HPyArray_CastingConverter(ctx, h_casting, &casting)) {
            HPyTracker_Close(ctx, ht);
            return HPy_NULL;
        }
    }

    /* axes */
    if (!HPyTuple_Check(ctx, axes_in) && !HPyList_Check(ctx, axes_in)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "axes must be a tuple of axis arrays");
        return HPy_NULL;
    }
    nnest = HPy_Length(ctx, axes_in);
    if (nnest < 2) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "axes must have at least 2 entries for nested iteration");
        return HPy_NULL;
    }
    naxes = 0;
    memset(used_axes, 0, NPY_MAXDIMS);
    for (inest = 0; inest < nnest; ++inest) {
        HPy item = HPy_GetItem_i(ctx, axes_in, inest);
        npy_intp i;
        if (HPy_IsNull(item)) {
            return HPy_NULL;
        }
        if (!HPyTuple_Check(ctx, item) && !HPyList_Check(ctx, item)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Each item in axes must be a an integer tuple");
            HPy_Close(ctx, item);
            return HPy_NULL;
        }
        nested_naxes[inest] = HPy_Length(ctx, item);
        if (naxes + nested_naxes[inest] > NPY_MAXDIMS) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Too many axes given");
            HPy_Close(ctx, item);
            return HPy_NULL;
        }
        for (i = 0; i < nested_naxes[inest]; ++i) {
            HPy v = HPy_GetItem_i(ctx, item, i);
            npy_intp axis;
            if (HPy_IsNull(v)) {
                HPy_Close(ctx, item);
                return HPy_NULL;
            }
            axis = HPyLong_AsLong(ctx, v);
            HPy_Close(ctx, v);
            if (axis < 0 || axis >= NPY_MAXDIMS) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "An axis is out of bounds");
                HPy_Close(ctx, item);
                return HPy_NULL;
            }
            /*
             * This check is very important, without it out of bounds
             * data accesses are possible.
             */
            if (used_axes[axis] != 0) {
                HPyErr_SetString(ctx, ctx->h_ValueError,
                        "An axis is used more than once");
                HPy_Close(ctx, item);
                return HPy_NULL;
            }
            used_axes[axis] = 1;
            op_axes_data[naxes+i] = axis;
        }
        nested_op_axes[inest] = &op_axes_data[naxes];
        naxes += nested_naxes[inest];
        HPy_Close(ctx, item);
    }

    /* op and op_flags */
    if (npyiter_convert_ops(ctx, op_in, op_flags_in, op, op_flags, &nop)
                                                        != 1) {
        return HPy_NULL;
    }

    /* Set the dtypes to all NULL to start as well */
    memset(op_request_dtypes, 0, sizeof(op_request_dtypes[0])*nop);
    memset(op_request_dtypes_inner, 0,
                        sizeof(op_request_dtypes_inner[0])*nop);

    /* op_request_dtypes */
    if (!HPy_IsNull(op_dtypes_in) && !HPy_Is(ctx, op_dtypes_in, ctx->h_None) &&
            npyiter_convert_dtypes(ctx, op_dtypes_in,
                                   op_request_dtypes, nop) != 1) {
        goto fail;
    }

    /* For broadcasting allocated arrays */
    for (iaxes = 0; iaxes < naxes; ++iaxes) {
        negones[iaxes] = -1;
    }

    /*
     * Clear any unnecessary ALLOCATE flags, so we can use them
     * to indicate exactly the allocated outputs.  Also, separate
     * the inner loop flags.
     */
    for (iop = 0; iop < nop; ++iop) {
        if ((op_flags[iop]&NPY_ITER_ALLOCATE) && !HPy_IsNull(op[iop])) {
            op_flags[iop] &= ~NPY_ITER_ALLOCATE;
        }

        /*
         * Clear any flags allowing copies or output allocation for
         * the inner loop.
         */
        op_flags_inner[iop] = op_flags[iop] & ~(NPY_ITER_COPY|
                             NPY_ITER_UPDATEIFCOPY|
                             NPY_ITER_ALLOCATE);
        /*
         * If buffering is enabled and copying is not,
         * clear the nbo_aligned flag and strip the data type
         * for the outer loops.
         */
        if ((flags&(NPY_ITER_BUFFERED)) &&
                !(op_flags[iop]&(NPY_ITER_COPY|
                                   NPY_ITER_UPDATEIFCOPY|
                                   NPY_ITER_ALLOCATE))) {
            op_flags[iop] &= ~(NPY_ITER_NBO|NPY_ITER_ALIGNED|NPY_ITER_CONTIG);
            op_request_dtypes_inner[iop] = op_request_dtypes[iop];
            op_request_dtypes[iop] = HPy_NULL;
        }
    }

    /* Only the inner loop gets the buffering and no inner flags */
    flags_inner = flags&~NPY_ITER_COMMON_DTYPE;
    flags &= ~(NPY_ITER_EXTERNAL_LOOP|
                    NPY_ITER_BUFFERED);

    HPyTupleBuilder tb_ret = HPyTupleBuilder_New(ctx, nnest);
    if (HPyTupleBuilder_IsNull(tb_ret)) {
        goto fail;
    }

    for (inest = 0; inest < nnest; ++inest) {
        NewNpyArrayIterObject *iter;
        int *op_axes_nop[NPY_MAXARGS];

        /*
         * All the operands' op_axes are the same, except for
         * allocated outputs.
         */
        for (iop = 0; iop < nop; ++iop) {
            if (op_flags[iop]&NPY_ITER_ALLOCATE) {
                if (inest == 0) {
                    op_axes_nop[iop] = NULL;
                }
                else {
                    op_axes_nop[iop] = negones;
                }
            }
            else {
                op_axes_nop[iop] = nested_op_axes[inest];
            }
        }

        /*
        printf("\n");
        for (iop = 0; iop < nop; ++iop) {
            npy_intp i;

            for (i = 0; i < nested_naxes[inest]; ++i) {
                printf("%d ", (int)op_axes_nop[iop][i]);
            }
            printf("\n");
        }
        */

        /* Allocate the iterator */
        HPy h_NpyIter_Type = HPy_FromPyObject(ctx, (PyObject *)&NpyIter_Type);
        HPy h_iter = npyiter_new_impl(ctx, h_NpyIter_Type, NULL, 0, HPy_NULL);
        iter = NewNpyArrayIterObject_AsStruct(ctx, h_iter);

        if (HPy_IsNull(h_iter)) {
            HPyTupleBuilder_Cancel(ctx, tb_ret);
            goto fail;
        }

        if (inest < nnest-1) {
            iter->iter = HNpyIter_AdvancedNew(ctx, nop, op, flags, order,
                                casting, op_flags, op_request_dtypes,
                                nested_naxes[inest], op_axes_nop,
                                NULL,
                                0);
        }
        else {
            iter->iter = HNpyIter_AdvancedNew(ctx, nop, op, flags_inner, order,
                                casting, op_flags_inner,
                                op_request_dtypes_inner,
                                nested_naxes[inest], op_axes_nop,
                                NULL,
                                buffersize);
        }

        if (iter->iter == NULL) {
            HPyTupleBuilder_Cancel(ctx, tb_ret);
            goto fail;
        }

        /* Cache some values for the member functions to use */
        if (npyiter_cache_values(ctx, h_iter, iter) < 0) {
            HPyTupleBuilder_Cancel(ctx, tb_ret);
            goto fail;
        }

        if (NpyIter_GetIterSize(iter->iter) == 0) {
            iter->started = 1;
            iter->finished = 1;
        }
        else {
            iter->started = 0;
            iter->finished = 0;
        }

        /*
         * If there are any allocated outputs or any copies were made,
         * adjust op so that the other iterators use the same ones.
         */
        if (inest == 0) {
            HPy *operands = HNpyIter_GetOperandArray(iter->iter); // PyArrayObject **
            for (iop = 0; iop < nop; ++iop) {
                if (!HPy_Is(ctx, op[iop], operands[iop])) {
                    HPy_Close(ctx, op[iop]);
                    op[iop] = HPy_Dup(ctx, operands[iop]);
                    // Py_INCREF(op[iop]);
                }

                /*
                 * Clear any flags allowing copies for
                 * the rest of the iterators
                 */
                op_flags[iop] &= ~(NPY_ITER_COPY|
                                 NPY_ITER_UPDATEIFCOPY);
            }
            /* Clear the common dtype flag for the rest of the iterators */
            flags &= ~NPY_ITER_COMMON_DTYPE;
        }

        HPyTupleBuilder_Set(ctx, tb_ret, inest, h_iter);
    }

    /* Release our references to the ops and dtypes */
    for (iop = 0; iop < nop; ++iop) {
        HPy_Close(ctx, op[iop]);
        HPy_Close(ctx, op_request_dtypes[iop]);
        HPy_Close(ctx, op_request_dtypes_inner[iop]);
    }

    ret = HPyTupleBuilder_Build(ctx, tb_ret);

    /* Set up the nested child references */
    for (inest = 0; inest < nnest-1; ++inest) {
        HPy h_iter; // NewNpyArrayIterObject *
        h_iter = HPy_GetItem_i(ctx, ret, inest);
        /*
         * Indicates which iterator to reset with new base pointers
         * each iteration step.
         */
        NewNpyArrayIterObject *iter = NewNpyArrayIterObject_AsStruct(ctx, h_iter);
        HPy h_next_iter = HPy_GetItem_i(ctx, ret, inest+1);
        HPyField_Store(ctx, h_iter, &iter->nested_child, h_next_iter);
        /*
         * Need to do a nested reset so all the iterators point
         * at the right data
         */
        NewNpyArrayIterObject *next_iter = NewNpyArrayIterObject_AsStruct(ctx, h_next_iter);
        if (HNpyIter_ResetBasePointers(ctx, next_iter->iter,
                                iter->dataptrs, NULL) != NPY_SUCCEED) {
            HPy_Close(ctx, h_next_iter);
            HPy_Close(ctx, ret);
            return HPy_NULL;
        }
        HPy_Close(ctx, h_next_iter);
    }

    return ret;

fail:
    for (iop = 0; iop < nop; ++iop) {
        HPy_Close(ctx, op[iop]);
        HPy_Close(ctx, op_request_dtypes[iop]);
        HPy_Close(ctx, op_request_dtypes_inner[iop]);
    }
    return HPy_NULL;
}


static void
npyiter_dealloc(NewNpyArrayIterObject *self)
{
    if (self->iter) {
        HPyContext *ctx = npy_get_context();
        /* Store error, so that WriteUnraisable cannot clear an existing one */
        CAPI_WARN("missing PyErr_Fetch & PyErr_Restore");
        PyObject *exc, *val, *tb;
        PyErr_Fetch(&exc, &val, &tb);
        if (npyiter_has_writeback(self->iter)) {
            if (HPyErr_WarnEx(ctx, ctx->h_RuntimeWarning,
                    "Temporary data has not been written back to one of the "
                    "operands. Typically nditer is used as a context manager "
                    "otherwise 'close' must be called before reading iteration "
                    "results.", 1) < 0) {
                HPy s;

                s = HPyUnicode_FromString(ctx, "npyiter_dealloc");
                if (!HPy_IsNull(s)) {
                    HPyErr_WriteUnraisable(ctx, s);
                    HPy_Close(ctx, s);
                }
                else {
                    HPyErr_WriteUnraisable(ctx, ctx->h_None);
                }
            }
        }
        if (!HNpyIter_Deallocate(ctx, self->iter)) {
            HPyErr_WriteUnraisable(ctx, ctx->h_None);
        }
        self->iter = NULL;
        HPy h_self = HPy_FromPyObject(ctx, (PyObject *)self);
        HPyField_Store(ctx, h_self, &self->nested_child, HPy_NULL);
        HPy_Close(ctx, h_self);
        self->nested_child = HPyField_NULL;
        PyErr_Restore(exc, val, tb);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
npyiter_resetbasepointers(HPyContext *ctx, HPy h_self, NewNpyArrayIterObject *self)
{
    HPy h_cur_self = h_self;
    NewNpyArrayIterObject *cur_self = self;
    int skip_close = 1;
    while (!HPyField_IsNull(cur_self->nested_child)) {
        HPy h_nested_child = HPyField_Load(ctx, h_cur_self, cur_self->nested_child);
        NewNpyArrayIterObject *nested_child = NewNpyArrayIterObject_AsStruct(ctx, h_nested_child);
        if (HNpyIter_ResetBasePointers(ctx, nested_child->iter,
                                        cur_self->dataptrs, NULL) != NPY_SUCCEED) {
            return NPY_FAIL;
        }
        if (!skip_close) {
            HPy_Close(ctx, h_cur_self);
        }
        h_cur_self = h_nested_child;
        cur_self = nested_child;
        if (NpyIter_GetIterSize(cur_self->iter) == 0) {
            cur_self->started = 1;
            cur_self->finished = 1;
        }
        else {
            cur_self->started = 0;
            cur_self->finished = 0;
        }
        skip_close = 0;
    }

    return NPY_SUCCEED;
}

HPyDef_METH(npyiter_reset, "reset", HPyFunc_NOARGS)
static HPy
npyiter_reset_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    if (NpyIter_Reset(self->iter, NULL) != NPY_SUCCEED) {
        return HPy_NULL;
    }
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    if (self->get_multi_index == NULL && NpyIter_HasMultiIndex(self->iter)) {
        self->get_multi_index = HNpyIter_GetGetMultiIndex(ctx, self->iter, NULL);
    }

    /* If there is nesting, the nested iterators should be reset */
    if (npyiter_resetbasepointers(ctx, h_self, self) != NPY_SUCCEED) {
        return HPy_NULL;
    }

    return HPy_Dup(ctx, ctx->h_None);
}

/*
 * Makes a copy of the iterator.  Note that the nesting is not
 * copied.
 */
HPyDef_METH(npyiter_copy, "copy", HPyFunc_NOARGS)
static HPy
npyiter_copy_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    HPy h_iter; // NewNpyArrayIterObject *

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    /* Allocate the iterator */
    HPy h_NpyIter_Type = HPy_FromPyObject(ctx, (PyObject *)&NpyIter_Type);
    h_iter = npyiter_new_impl(ctx, h_NpyIter_Type, NULL, 0, HPy_NULL);
    HPy_Close(ctx, h_NpyIter_Type);
    if (HPy_IsNull(h_iter)) {
        return HPy_NULL;
    }

    /* Copy the C iterator */
    NewNpyArrayIterObject *iter = NewNpyArrayIterObject_AsStruct(ctx, h_iter);
    iter->iter = NpyIter_Copy(self->iter);
    if (iter->iter == NULL) {
        HPy_Close(ctx, h_iter);
        return HPy_NULL;
    }

    /* Cache some values for the member functions to use */
    if (npyiter_cache_values(ctx, h_iter, iter) < 0) {
        HPy_Close(ctx, h_iter);
        return HPy_NULL;
    }

    iter->started = self->started;
    iter->finished = self->finished;

    return h_iter;
}

HPyDef_METH(npyiter___copy__, "__copy__", HPyFunc_NOARGS)
static HPy
npyiter___copy___impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    return npyiter_copy_impl(ctx, h_self);
}

HPyDef_METH(npyiter_iternext, "iternext", HPyFunc_NOARGS)
static HPy
npyiter_iternext_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter != NULL && self->iternext != NULL &&
                        !self->finished && self->iternext(ctx, self->iter)) {
        /* If there is nesting, the nested iterators should be reset */
        if (npyiter_resetbasepointers(ctx, h_self, self) != NPY_SUCCEED) {
            HPy_Close(ctx, h_self);
            return HPy_NULL;
        }
        HPy_Close(ctx, h_self);

        return HPy_Dup(ctx, ctx->h_True);
    }
    else {
        if (HPyErr_Occurred(ctx)) {
            /* casting error, buffer cleanup will occur at reset or dealloc */
            return HPy_NULL;
        }
        self->finished = 1;
        return HPy_Dup(ctx, ctx->h_False);
    }
}

HPyDef_METH(npyiter_remove_axis, "remove_axis", HPyFunc_VARARGS)
static HPy
npyiter_remove_axis_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, 
                    HPy *args, HPy_ssize_t len_args)
{
    int axis = 0;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    if (!HPyArg_Parse(ctx, NULL, args, len_args, "i:remove_axis", &axis)) {
        return HPy_NULL;
    }

    if (HNpyIter_RemoveAxis(ctx, self->iter, axis) != NPY_SUCCEED) {
        return HPy_NULL;
    }
    /* RemoveAxis invalidates cached values */
    if (npyiter_cache_values(ctx, h_self, self) < 0) {
        return HPy_NULL;
    }
    /* RemoveAxis also resets the iterator */
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    return HPy_Dup(ctx, ctx->h_None);
}

HPyDef_METH(npyiter_remove_multi_index, "remove_multi_index", HPyFunc_NOARGS)
static HPy
npyiter_remove_multi_index_impl(
    HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    HNpyIter_RemoveMultiIndex(ctx, self->iter);
    /* RemoveMultiIndex invalidates cached values */
    npyiter_cache_values(ctx, h_self, self);
    /* RemoveMultiIndex also resets the iterator */
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    return HPy_Dup(ctx, ctx->h_None);
}

HPyDef_METH(npyiter_enable_external_loop, "enable_external_loop", HPyFunc_NOARGS)
static HPy
npyiter_enable_external_loop_impl(
    HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    HNpyIter_EnableExternalLoop(ctx, self->iter);
    /* EnableExternalLoop invalidates cached values */
    npyiter_cache_values(ctx, h_self, self);
    /* EnableExternalLoop also resets the iterator */
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    return HPy_Dup(ctx, ctx->h_None);
}

HPyDef_METH(npyiter_debug_print, "debug_print", HPyFunc_NOARGS)
static HPy
npyiter_debug_print_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter != NULL) {
        HNpyIter_DebugPrint(ctx, self->iter);
    }
    else {
        printf("Iterator: (nil)\n");
    }

    return HPy_Dup(ctx, ctx->h_None);
}

static HPy
npyiter_seq_item_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, HPy_ssize_t i);

HPyDef_GET(npyiter_value, "value")
static HPy
npyiter_value_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    HPy ret;

    npy_intp iop, nop;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return HPy_NULL;
    }

    nop = NpyIter_GetNOp(self->iter);

    /* Return an array  or tuple of arrays with the values */
    if (nop == 1) {
        ret = npyiter_seq_item_impl(ctx, h_self, 0);
    }
    else {
        HPyTupleBuilder tb_ret = HPyTupleBuilder_New(ctx, nop);
        if (HPyTupleBuilder_IsNull(tb_ret)) {
            return HPy_NULL;
        }
        for (iop = 0; iop < nop; ++iop) {
            HPy a = npyiter_seq_item_impl(ctx, h_self, iop);
            if (HPy_IsNull(a)) {
                HPyTupleBuilder_Cancel(ctx, tb_ret);
                return HPy_NULL;
            }
            HPyTupleBuilder_Set(ctx, tb_ret, iop, a);
        }
        ret = HPyTupleBuilder_Build(ctx, tb_ret);
    }

    return ret;
}

HPyDef_GET(npyiter_operands, "operands")
static HPy
npyiter_operands_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    HPyTupleBuilder ret;

    npy_intp iop, nop;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    ret = HPyTupleBuilder_New(ctx, nop);
    if (HPyTupleBuilder_IsNull(ret)) {
        return HPy_NULL;
    }
    for (iop = 0; iop < nop; ++iop) {
        HPy operand = HPyField_Load(ctx, h_self, self->operands[iop]);
        HPyTupleBuilder_Set(ctx, ret, iop, operand);
        HPy_Close(ctx, operand);
    }

    return HPyTupleBuilder_Build(ctx, ret);
}

HPyDef_GET(npyiter_itviews, "itviews")
static HPy
npyiter_itviews_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    HPyTupleBuilder ret;

    npy_intp iop, nop;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    ret = HPyTupleBuilder_New(ctx, nop);
    if (HPyTupleBuilder_IsNull(ret)) {
        return HPy_NULL;
    }
    for (iop = 0; iop < nop; ++iop) {
        HPy view = HNpyIter_GetIterView(ctx, self->iter, iop);

        if (HPy_IsNull(view)) {
            HPyTupleBuilder_Cancel(ctx, ret);
            return HPy_NULL;
        }
        HPyTupleBuilder_Set(ctx, ret, iop, view);
    }

    return HPyTupleBuilder_Build(ctx, ret);
}

// Not supported by HPy
// HPyDef_SLOT(npyiter_next, HPy_tp_iternext)
static HPy
hpy_npyiter_next(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->iternext == NULL ||
                self->finished) {
        return HPy_NULL;
    }

    /*
     * Use the started flag for the Python iteration protocol to work
     * when buffering is enabled.
     */
    if (self->started) {
        if (!self->iternext(ctx, self->iter)) {
            /*
             * A casting error may be set here (or no error causing a
             * StopIteration). Buffers may only be cleaned up later.
             */
            self->finished = 1;
            return HPy_NULL;
        }

        /* If there is nesting, the nested iterators should be reset */
        if (npyiter_resetbasepointers(ctx, h_self, self) != NPY_SUCCEED) {
            return HPy_NULL;
        }
    }
    self->started = 1;

    return npyiter_value_get(ctx, h_self, NULL);
};

static PyObject *
npyiter_next(NewNpyArrayIterObject *self)
{
    HPyContext *ctx = npy_get_context();
    HPy h_self = HPy_FromPyObject(ctx, self);
    HPy h_ret = hpy_npyiter_next(ctx, h_self);
    PyObject *ret = HPy_AsPyObject(ctx, h_ret);
    HPy_Close(ctx, h_ret);
    return ret;
}

HPyDef_GET(npyiter_shape, "shape")
static HPy
npyiter_shape_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    npy_intp ndim, shape[NPY_MAXDIMS];

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return HPy_NULL;
    }

    if (NpyIter_GetShape(self->iter, shape) == NPY_SUCCEED) {
        ndim = NpyIter_GetNDim(self->iter);
        return HPyArray_IntTupleFromIntp(ctx, ndim, shape);
    }

    return HPy_NULL;
}

HPyDef_GETSET(npyiter_multi_index, "multi_index")

static HPy
npyiter_multi_index_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    npy_intp ndim, multi_index[NPY_MAXDIMS];

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return HPy_NULL;
    }

    if (self->get_multi_index != NULL) {
        ndim = NpyIter_GetNDim(self->iter);
        self->get_multi_index(self->iter, multi_index);
        return HPyArray_IntTupleFromIntp(ctx, ndim, multi_index);
    }
    else {
        if (!NpyIter_HasMultiIndex(self->iter)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Iterator is not tracking a multi-index");
            return HPy_NULL;
        }
        else if (NpyIter_HasDelayedBufAlloc(self->iter)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Iterator construction used delayed buffer allocation, "
                    "and no reset has been done yet");
            return HPy_NULL;
        }
        else {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Iterator is in an invalid state");
            return HPy_NULL;
        }
    }
}

static int
npyiter_multi_index_set(
        HPyContext *ctx, HPy h_self, HPy value, void *NPY_UNUSED(ignored))
{
    npy_intp idim, ndim, multi_index[NPY_MAXDIMS];

    if (HPy_IsNull(value)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete nditer multi_index");
        return -1;
    }
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return -1;
    }

    if (NpyIter_HasMultiIndex(self->iter)) {
        ndim = NpyIter_GetNDim(self->iter);
        if (!HPySequence_Check(ctx, value)) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "multi_index must be set with a sequence");
            return -1;
        }
        if (HPy_Length(ctx, value) != ndim) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Wrong number of indices");
            return -1;
        }
        for (idim = 0; idim < ndim; ++idim) {
            HPy v = HPy_GetItem_i(ctx, value, idim);
            multi_index[idim] = HPyLong_AsLong(ctx, v);
            HPy_Close(ctx, v);
            if (error_converting(multi_index[idim])) {
                return -1;
            }
        }
        if (NpyIter_GotoMultiIndex(self->iter, multi_index) != NPY_SUCCEED) {
            return -1;
        }
        self->started = 0;
        self->finished = 0;

        /* If there is nesting, the nested iterators should be reset */
        if (npyiter_resetbasepointers(ctx, h_self, self) != NPY_SUCCEED) {
            return -1;
        }

        return 0;
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is not tracking a multi-index");
        return -1;
    }
}

static HPy
npyiter_index_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return HPy_NULL;
    }

    if (NpyIter_HasIndex(self->iter)) {
        npy_intp ind = *NpyIter_GetIndexPtr(self->iter);
        return HPyLong_FromLong(ctx, ind);
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator does not have an index");
        return HPy_NULL;
    }
}

HPyDef_GETSET(npyiter_index, "index")
static int
npyiter_index_set(
        HPyContext *ctx, HPy h_self, HPy value, void *NPY_UNUSED(ignored))
{
    if (HPy_IsNull(value)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete nditer index");
        return -1;
    }
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return -1;
    }

    if (NpyIter_HasIndex(self->iter)) {
        npy_intp ind;
        ind = HPyLong_AsLong(ctx, value);
        if (error_converting(ind)) {
            return -1;
        }
        if (NpyIter_GotoIndex(self->iter, ind) != NPY_SUCCEED) {
            return -1;
        }
        self->started = 0;
        self->finished = 0;

        /* If there is nesting, the nested iterators should be reset */
        if (npyiter_resetbasepointers(ctx, h_self, self) != NPY_SUCCEED) {
            return -1;
        }

        return 0;
    }
    else {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator does not have an index");
        return -1;
    }
}

static HPy
npyiter_iterindex_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return HPy_NULL;
    }

    return HPyLong_FromLong(ctx, NpyIter_GetIterIndex(self->iter));
}

HPyDef_GETSET(npyiter_iterindex, "iterindex")
static int
npyiter_iterindex_set(
        HPyContext *ctx, HPy h_self, HPy value, void *NPY_UNUSED(ignored))
{
    npy_intp iterindex;

    if (HPy_IsNull(value)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete nditer iterindex");
        return -1;
    }

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return -1;
    }

    iterindex = HPyLong_AsLong(ctx, value);
    if (error_converting(iterindex)) {
        return -1;
    }
    if (HNpyIter_GotoIterIndex(ctx, self->iter, iterindex) != NPY_SUCCEED) {
        return -1;
    }
    self->started = 0;
    self->finished = 0;

    /* If there is nesting, the nested iterators should be reset */
    if (npyiter_resetbasepointers(ctx, h_self, self) != NPY_SUCCEED) {
        return -1;
    }

    return 0;
}

static HPy
npyiter_iterrange_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    npy_intp istart = 0, iend = 0;
    HPy ret;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    NpyIter_GetIterIndexRange(self->iter, &istart, &iend);

    HPy s = HPyLong_FromLong(ctx, istart);
    HPy e = HPyLong_FromLong(ctx, iend);
    ret = HPyTuple_Pack(ctx, 2, s, e);
    HPy_Close(ctx, s);
    HPy_Close(ctx, e);
    return ret;
}

HPyDef_GETSET(npyiter_iterrange, "iterrange")
static int
npyiter_iterrange_set(
        HPyContext *ctx, HPy h_self, HPy value, void *NPY_UNUSED(ignored))
{
    npy_intp istart = 0, iend = 0;

    if (HPy_IsNull(value)) {
        HPyErr_SetString(ctx, ctx->h_AttributeError,
                "Cannot delete nditer iterrange");
        return -1;
    }
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return -1;
    }
    HPy_ssize_t args_len = HPy_Length(ctx, value);
    HPy *args = malloc(args_len * sizeof(HPy));
    for (HPy_ssize_t i = 0; i < args_len; i++) {
        args[i] = HPy_GetItem_i(ctx, value, i);
    }
    if (!HPyArg_Parse(ctx, NULL, args, args_len, "nn", &istart, &iend)) {
        HPy_CloseAndFreeArray(ctx, args, args_len);
        return -1;
    }
    HPy_CloseAndFreeArray(ctx, args, args_len);

    if (NpyIter_ResetToIterIndexRange(self->iter, istart, iend, NULL)
                                                    != NPY_SUCCEED) {
        return -1;
    }
    if (istart < iend) {
        self->started = self->finished = 0;
    }
    else {
        self->started = self->finished = 1;
    }

    if (self->get_multi_index == NULL && NpyIter_HasMultiIndex(self->iter)) {
        self->get_multi_index = NpyIter_GetGetMultiIndex(self->iter, NULL);
    }

    /* If there is nesting, the nested iterators should be reset */
    if (npyiter_resetbasepointers(ctx, h_self, self) != NPY_SUCCEED) {
        return -1;
    }

    return 0;
}

HPyDef_GET(npyiter_has_delayed_bufalloc, "has_delayed_bufalloc")
static HPy
npyiter_has_delayed_bufalloc_get(
        HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        return HPy_Dup(ctx, ctx->h_True);
    }
    else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}

HPyDef_GET(npyiter_iterationneedsapi, "iterationneedsapi")
static HPy
npyiter_iterationneedsapi_get(
        HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    if (NpyIter_IterationNeedsAPI(self->iter)) {
        return HPy_Dup(ctx, ctx->h_True);
    }
    else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}

HPyDef_GET(npyiter_has_multi_index, "has_multi_index")
static HPy
npyiter_has_multi_index_get(
        HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    if (NpyIter_HasMultiIndex(self->iter)) {
        return HPy_Dup(ctx, ctx->h_True);
    }
    else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}

HPyDef_GET(npyiter_has_index, "has_index")
static HPy
npyiter_has_index_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    if (NpyIter_HasIndex(self->iter)) {
        return HPy_Dup(ctx, ctx->h_True);
    }
    else {
        return HPy_Dup(ctx, ctx->h_False);
    }
}

HPyDef_GET(npyiter_dtypes, "dtypes")
static HPy
npyiter_dtypes_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    HPyTupleBuilder ret;

    npy_intp iop, nop;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    ret = HPyTupleBuilder_New(ctx, nop);
    if (HPyTupleBuilder_IsNull(ret)) {
        return HPy_NULL;
    }

    for (iop = 0; iop < nop; ++iop) {
        HPy dtype = HPyField_Load(ctx, h_self, self->dtypes[iop]);
        HPyTupleBuilder_Set(ctx, ret, iop, dtype);
        HPy_Close(ctx, dtype);
    }

    return HPyTupleBuilder_Build(ctx, ret);
}

HPyDef_GET(npyiter_ndim, "ndim")
static HPy
npyiter_ndim_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    return HPyLong_FromLong(ctx, NpyIter_GetNDim(self->iter));
}

HPyDef_GET(npyiter_nop, "nop")
static HPy
npyiter_nop_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    return HPyLong_FromLong(ctx, NpyIter_GetNOp(self->iter));
}

HPyDef_GET(npyiter_itersize, "itersize")
static HPy
npyiter_itersize_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is invalid");
        return HPy_NULL;
    }

    return HPyLong_FromLong(ctx, NpyIter_GetIterSize(self->iter));
}

HPyDef_GET(npyiter_finished, "finished")
static HPy
npyiter_finished_get(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, void *NPY_UNUSED(ignored))
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || !self->finished) {
        return HPy_Dup(ctx, ctx->h_False);
    }
    else {
        return HPy_Dup(ctx, ctx->h_True);
    }
}

HPyDef_SLOT(npyiter_seq_length_sq, HPy_sq_length)
static HPy_ssize_t
npyiter_seq_length_sq_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        return 0;
    }
    else {
        return NpyIter_GetNOp(self->iter);
    }
}

HPyDef_SLOT(npyiter_seq_length_mp, HPy_mp_length)
static HPy_ssize_t
npyiter_seq_length_mp_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    return npyiter_seq_length_sq_impl(ctx, h_self);
}

HPyDef_SLOT(npyiter_seq_item, HPy_sq_item)
static HPy
npyiter_seq_item_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, HPy_ssize_t i)
{
    npy_intp ret_ndim;
    npy_intp nop, innerloopsize, innerstride;
    char *dataptr;
    HPy dtype; // PyArray_Descr *
    int has_external_loop;
    HPy_ssize_t i_orig = i;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return HPy_NULL;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return HPy_NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    /* Negative indexing */
    if (i < 0) {
        i += nop;
    }

    if (i < 0 || i >= nop) {
        HPyErr_Format_p(ctx, ctx->h_IndexError,
                "Iterator operand index %zd is out of bounds", i_orig);
        return HPy_NULL;
    }

#if 0
    /*
     * This check is disabled because it prevents things like
     * np.add(it[0], it[1], it[2]), where it[2] is a write-only
     * parameter.  When write-only, the value of it[i] is
     * likely random junk, as if it were allocated with an
     * np.empty(...) call.
     */
    if (!self->readflags[i]) {
        PyErr_Format(PyExc_RuntimeError,
                "Iterator operand %zd is write-only", i);
        return HPy_NULL;
    }
#endif

    dataptr = self->dataptrs[i];
    dtype = HPyField_Load(ctx, h_self, self->dtypes[i]);
    has_external_loop = NpyIter_HasExternalLoop(self->iter);

    if (has_external_loop) {
        innerloopsize = *self->innerloopsizeptr;
        innerstride = self->innerstrides[i];
        ret_ndim = 1;
    }
    else {
        innerloopsize = 1;
        innerstride = 0;
        /* If the iterator is going over every element, return array scalars */
        ret_ndim = 0;
    }

    // Py_INCREF(dtype);
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    HPy ret = HPyArray_NewFromDescrAndBase(ctx,
            array_type, dtype,
            ret_ndim, &innerloopsize, &innerstride, dataptr,
            self->writeflags[i] ? NPY_ARRAY_WRITEABLE : 0,
            HPy_NULL, h_self);
    HPy_Close(ctx, dtype);
    HPy_Close(ctx, array_type);
    return ret;
}

NPY_NO_EXPORT HPy
npyiter_seq_slice(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self,
                    HPy_ssize_t ilow, HPy_ssize_t ihigh)
{
    npy_intp nop;
    HPy_ssize_t i;
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return HPy_NULL;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return HPy_NULL;
    }
    nop = NpyIter_GetNOp(self->iter);
    if (ilow < 0) {
        ilow = 0;
    }
    else if (ilow >= nop) {
        ilow = nop-1;
    }
    if (ihigh < ilow) {
        ihigh = ilow;
    }
    else if (ihigh > nop) {
        ihigh = nop;
    }

    HPyTupleBuilder tb_ret = HPyTupleBuilder_New(ctx, ihigh-ilow);
    if (HPyTupleBuilder_IsNull(tb_ret)) {
        return HPy_NULL;
    }
    for (i = ilow; i < ihigh ; ++i) {
        HPy item = npyiter_seq_item_impl(ctx, h_self, i);
        if (HPy_IsNull(item)) {
            HPyTupleBuilder_Cancel(ctx, tb_ret);
            return HPy_NULL;
        }
        HPyTupleBuilder_Set(ctx, tb_ret, i-ilow, item);
    }

    return HPyTupleBuilder_Build(ctx, tb_ret);
}

HPyDef_SLOT(npyiter_seq_ass_item, HPy_sq_ass_item)
NPY_NO_EXPORT int
npyiter_seq_ass_item_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, HPy_ssize_t i, HPy v)
{

    npy_intp nop, innerloopsize, innerstride;
    char *dataptr;
    HPy dtype; // PyArray_Descr *
    HPy tmp; // PyArrayObject *
    int ret, has_external_loop;
    Py_ssize_t i_orig = i;

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (HPy_IsNull(v)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Cannot delete iterator elements");
        return -1;
    }

    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return -1;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return -1;
    }
    nop = NpyIter_GetNOp(self->iter);

    /* Negative indexing */
    if (i < 0) {
        i += nop;
    }

    if (i < 0 || i >= nop) {
        HPyErr_Format_p(ctx, ctx->h_IndexError,
                "Iterator operand index %zd is out of bounds", i_orig);
        return -1;
    }
    if (!self->writeflags[i]) {
        HPyErr_Format_p(ctx, ctx->h_RuntimeError,
                "Iterator operand %zd is not writeable", i_orig);
        return -1;
    }

    dataptr = self->dataptrs[i];
    dtype = HPyField_Load(ctx, h_self, self->dtypes[i]);
    has_external_loop = NpyIter_HasExternalLoop(self->iter);

    if (has_external_loop) {
        innerloopsize = *self->innerloopsizeptr;
        innerstride = self->innerstrides[i];
    }
    else {
        innerloopsize = 1;
        innerstride = 0;
    }

    /* TODO - there should be a better way than this... */
    // Py_INCREF(dtype);
    HPy array_type = HPyGlobal_Load(ctx, HPyArray_Type);
    tmp = HPyArray_NewFromDescr(ctx, array_type, dtype,
                                1, &innerloopsize,
                                &innerstride, dataptr,
                                NPY_ARRAY_WRITEABLE, HPy_NULL);
    if (HPy_IsNull(tmp)) {
        return -1;
    }

    PyArrayObject *tmp_data = PyArrayObject_AsStruct(ctx, tmp);
    ret = HPyArray_CopyObject(ctx, tmp, tmp_data, v);
    HPy_Close(ctx, tmp);
    return ret;
}

static int
npyiter_seq_ass_slice(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, HPy_ssize_t ilow,
                HPy_ssize_t ihigh, HPy v)
{
    npy_intp nop;
    Py_ssize_t i;

    if (HPy_IsNull(v)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Cannot delete iterator elements");
        return -1;
    }

    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return -1;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return -1;
    }
    nop = NpyIter_GetNOp(self->iter);
    if (ilow < 0) {
        ilow = 0;
    }
    else if (ilow >= nop) {
        ilow = nop-1;
    }
    if (ihigh < ilow) {
        ihigh = ilow;
    }
    else if (ihigh > nop) {
        ihigh = nop;
    }

    if (!HPySequence_Check(ctx, v) || HPy_Length(ctx, v) != ihigh-ilow) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Wrong size to assign to iterator slice");
        return -1;
    }

    for (i = ilow; i < ihigh ; ++i) {
        HPy item = HPy_GetItem_i(ctx, v, i-ilow);
        if (HPy_IsNull(item)) {
            return -1;
        }
        if (npyiter_seq_ass_item_impl(ctx, h_self, i, item) < 0) {
            HPy_Close(ctx, item);
            return -1;
        }
        HPy_Close(ctx, item);
    }

    return 0;
}

HPyDef_SLOT(npyiter_subscript, HPy_mp_subscript)
static HPy
npyiter_subscript_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, HPy op)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return HPy_NULL;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return HPy_NULL;
    }
    CAPI_WARN("missing PyIndex_Check, PySlice_Check, PySlice_GetIndicesEx");
    PyObject *py_op = HPy_AsPyObject(ctx, op);
    if (HPyLong_Check(ctx, op) ||
                    (PyIndex_Check(py_op) && !HPySequence_Check(ctx, op))) {
        Py_DECREF(py_op);
        npy_intp i = HPyArray_PyIntAsIntp(ctx, op);
        if (hpy_error_converting(ctx, i)) {
            return HPy_NULL;
        }
        return npyiter_seq_item_impl(ctx, h_self, i);
    }
    else if (PySlice_Check(py_op)) {
        Py_ssize_t istart = 0, iend = 0, istep = 0, islicelength;
        if (PySlice_GetIndicesEx(py_op, NpyIter_GetNOp(self->iter),
                                 &istart, &iend, &istep, &islicelength) < 0) {
            Py_DECREF(py_op);
            return HPy_NULL;
        }
        Py_DECREF(py_op);
        if (istep != 1) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Iterator slicing only supports a step of 1");
            return HPy_NULL;
        }
        return npyiter_seq_slice(ctx, h_self, istart, iend);
    }
    Py_DECREF(py_op);

    HPyErr_SetString(ctx, ctx->h_TypeError,
            "invalid index type for iterator indexing");
    return HPy_NULL;
}

HPyDef_SLOT(npyiter_ass_subscript, HPy_mp_ass_subscript)
static int
npyiter_ass_subscript_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, HPy op,
                        HPy value)
{
    if (HPy_IsNull(value)) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Cannot delete iterator elements");
        return -1;
    }
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL || self->finished) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator is past the end");
        return -1;
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        HPyErr_SetString(ctx, ctx->h_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return -1;
    }

    CAPI_WARN("missing PyIndex_Check, PySlice_Check, PySlice_GetIndicesEx");
    PyObject *py_op = HPy_AsPyObject(ctx, op);
    if (HPyLong_Check(ctx, op) ||
                    (PyIndex_Check(py_op) && !HPySequence_Check(ctx, op))) {
        Py_DECREF(py_op);
        npy_intp i = HPyArray_PyIntAsIntp(ctx, op);
        if (hpy_error_converting(ctx, i)) {
            return -1;
        }
        return npyiter_seq_ass_item_impl(ctx, h_self, i, value);
    }
    else if (PySlice_Check(py_op)) {
        Py_ssize_t istart = 0, iend = 0, istep = 0, islicelength = 0;
        if (PySlice_GetIndicesEx(py_op, NpyIter_GetNOp(self->iter),
                                 &istart, &iend, &istep, &islicelength) < 0) {
            Py_DECREF(py_op);
            return -1;
        }
        Py_DECREF(py_op);
        if (istep != 1) {
            HPyErr_SetString(ctx, ctx->h_ValueError,
                    "Iterator slice assignment only supports a step of 1");
            return -1;
        }
        return npyiter_seq_ass_slice(ctx, h_self, istart, iend, value);
    }
    Py_DECREF(py_op);

    HPyErr_SetString(ctx, ctx->h_TypeError,
            "invalid index type for iterator indexing");
    return -1;
}

HPyDef_METH(npyiter_enter, "__enter__", HPyFunc_NOARGS)
static HPy
npyiter_enter_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    if (self->iter == NULL) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "operation on non-initialized iterator");
        return HPy_NULL;
    }

    return HPy_Dup(ctx, h_self);
}

HPyDef_METH(npyiter_close, "close", HPyFunc_NOARGS)
static HPy
npyiter_close_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self)
{
    NewNpyArrayIterObject *self = NewNpyArrayIterObject_AsStruct(ctx, h_self);
    NpyIter *iter = self->iter;
    int ret;
    if (self->iter == NULL) {
        HPy_Dup(ctx, ctx->h_None);
    }
    ret = NpyIter_Deallocate(iter);
    self->iter = NULL;
    // Py_XDECREF(self->nested_child);
    HPyField_Store(ctx, h_self, &self->nested_child, HPy_NULL);
    self->nested_child = HPyField_NULL;
    if (ret != NPY_SUCCEED) {
        HPy_Dup(ctx, ctx->h_None);
    }
    return HPy_Dup(ctx, ctx->h_None);
}

HPyDef_METH(npyiter_exit, "__exit__", HPyFunc_VARARGS)
static HPy
npyiter_exit_impl(HPyContext *ctx, HPy /* NewNpyArrayIterObject * */ h_self, 
                    HPy *args, HPy_ssize_t len_args)
{
    /* even if called via exception handling, writeback any data */
    return npyiter_close_impl(ctx, h_self);
}

NPY_NO_EXPORT PyType_Slot npyiter_slots[] = {
        {Py_tp_dealloc, npyiter_dealloc},
        {Py_tp_iter, PyObject_SelfIter},
        {Py_tp_iternext, npyiter_next},
        {0, NULL}
};

static HPyDef *npyiter_defines[] = {
    // slots
    &npyiter_new,
    &npyiter_init,
    &npyiter_seq_length_sq,
    &npyiter_seq_length_mp,
    &npyiter_seq_item,
    &npyiter_seq_ass_item,
    &npyiter_ass_subscript,
    &npyiter_subscript,

    // getset
    &npyiter_value,
    &npyiter_shape,
    &npyiter_multi_index,
    &npyiter_index,
    &npyiter_iterindex,
    &npyiter_iterrange,
    &npyiter_operands,
    &npyiter_itviews,
    &npyiter_has_delayed_bufalloc,
    &npyiter_iterationneedsapi,
    &npyiter_has_multi_index,
    &npyiter_has_index,
    &npyiter_dtypes,
    &npyiter_ndim,
    &npyiter_nop,
    &npyiter_itersize,
    &npyiter_finished,

    // methods
    &npyiter_reset,
    &npyiter_copy,
    &npyiter___copy__,
    &npyiter_iternext,
    &npyiter_remove_axis,
    &npyiter_remove_multi_index,
    &npyiter_debug_print,
    &npyiter_enable_external_loop,
    &npyiter_enter,
    &npyiter_exit,
    &npyiter_close,
    NULL,
};

NPY_NO_EXPORT HPyType_Spec NpyIter_Type_Spec = {
    .name = "numpy.nditer",
    .basicsize = sizeof(NewNpyArrayIterObject),
    .flags = HPy_TPFLAGS_DEFAULT,
    .builtin_shape = SHAPE(NewNpyArrayIterObject),
    .legacy_slots = npyiter_slots,
    .defines = npyiter_defines,
};

NPY_NO_EXPORT PyTypeObject *_NpyIter_Type_p;

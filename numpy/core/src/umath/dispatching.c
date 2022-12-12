/*
 * This file implements universal function dispatching and promotion (which
 * is necessary to happen before dispatching).
 * This is part of the UFunc object.  Promotion and dispatching uses the
 * following things:
 *
 * - operand_DTypes:  The datatypes as passed in by the user.
 * - signature: The DTypes fixed by the user with `dtype=` or `signature=`.
 * - ufunc._loops: A list of all ArrayMethods and promoters, it contains
 *   tuples `(dtypes, ArrayMethod)` or `(dtypes, promoter)`.
 * - ufunc._dispatch_cache: A cache to store previous promotion and/or
 *   dispatching results.
 * - The actual arrays are used to support the old code paths where necessary.
 *   (this includes any value-based casting/promotion logic)
 *
 * In general, `operand_Dtypes` is always overridden by `signature`.  If a
 * DType is included in the `signature` it must match precisely.
 *
 * The process of dispatching and promotion can be summarized in the following
 * steps:
 *
 * 1. Override any `operand_DTypes` from `signature`.
 * 2. Check if the new `operand_Dtypes` is cached (if it is, got to 4.)
 * 3. Find the best matching "loop".  This is done using multiple dispatching
 *    on all `operand_DTypes` and loop `dtypes`.  A matching loop must be
 *    one whose DTypes are superclasses of the `operand_DTypes` (that are
 *    defined).  The best matching loop must be better than any other matching
 *    loop.  This result is cached.
 * 4. If the found loop is a promoter: We call the promoter. It can modify
 *    the `operand_DTypes` currently.  Then go back to step 2.
 *    (The promoter can call arbitrary code, so it could even add the matching
 *    loop first.)
 * 5. The final `ArrayMethod` is found, its registered `dtypes` is copied
 *    into the `signature` so that it is available to the ufunc loop.
 *
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "common.h"

#include "dispatching.h"
#include "dtypemeta.h"
#include "common_dtype.h"
#include "npy_hashtable.h"
#include "legacy_array_method.h"
#include "ufunc_object.h"
#include "ufunc_type_resolution.h"


#define PROMOTION_DEBUG_TRACING 0


/* forward declaration */
static NPY_INLINE PyObject *
promote_and_get_info_and_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool allow_legacy_promotion);

static NPY_INLINE HPy
hpy_promote_and_get_info_and_ufuncimpl(HPyContext *ctx,
        HPy /* (PyUFuncObject *) */ ufunc,
        HPy /* (PyArrayObject *) */ const *ops,
        HPy /* (PyArray_DTypeMeta *) */ signature[],
        HPy /* (PyArray_DTypeMeta *) */ op_dtypes[],
        npy_bool allow_legacy_promotion);

/**
 * Function to add a new loop to the ufunc.  This mainly appends it to the
 * list (as it currently is just a list).
 *
 * @param ufunc The universal function to add the loop to.
 * @param info The tuple (dtype_tuple, ArrayMethod/promoter).
 * @param ignore_duplicate If 1 and a loop with the same `dtype_tuple` is
 *        found, the function does nothing.
 */
NPY_NO_EXPORT int
PyUFunc_AddLoop(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate)
{
    HPyContext *ctx = npy_get_context();
    HPy h_ufunc = HPy_FromPyObject(ctx, (PyObject *)ufunc);
    HPy h_info = HPy_FromPyObject(ctx, info);
    int res = HPyUFunc_AddLoop(ctx, h_ufunc, h_info, ignore_duplicate);
    HPy_Close(ctx, h_info);
    HPy_Close(ctx, h_ufunc);
    return res;
}

NPY_NO_EXPORT int
HPyUFunc_AddLoop(HPyContext *ctx, HPy /* (PyUFuncObject *) */ ufunc, HPy info, int ignore_duplicate)
{
    int res;
    HPy DType_tuple = HPy_NULL;
    HPy loops = HPy_NULL;
    HPy meth_or_promoter = HPy_NULL;
    PyUFuncObject *ufunc_data = PyUFuncObject_AsStruct(ctx, ufunc);
    /*
     * Validate the info object, this should likely move to a different
     * entry-point in the future (and is mostly unnecessary currently).
     */
    if (!HPyTuple_CheckExact(ctx, info) || HPy_Length(ctx, info) != 2) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Info must be a tuple: "
                "(tuple of DTypes or None, ArrayMethod or promoter)");
        return -1;
    }
    DType_tuple = HPy_GetItem_i(ctx, info, 0);
    HPy_ssize_t n_DType_tuple = HPy_Length(ctx, DType_tuple);
    if (n_DType_tuple != ufunc_data->nargs) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "DType tuple length does not match ufunc number of operands");
        res = -1;
        goto finish;
    }
    for (HPy_ssize_t i = 0; i < n_DType_tuple; i++) {
        HPy item = HPy_GetItem_i(ctx, DType_tuple, i);
        if (HPy_IsNull(item)) {
            res = -1;
            goto finish;
        }
        if (!HPy_Is(ctx, item, ctx->h_None)
                && !HPyGlobal_TypeCheck(ctx, item, HPyArrayDTypeMeta_Type)) {
            HPyErr_SetString(ctx, ctx->h_TypeError,
                    "DType tuple may only contain None and DType classes");
            HPy_Close(ctx, item);
            res = -1;
            goto finish;
        }
        HPy_Close(ctx, item);
    }
    meth_or_promoter = HPy_GetItem_i(ctx, info, 1);
    if (!HPyGlobal_TypeCheck(ctx, meth_or_promoter, HPyArrayMethod_Type)
            && !HPyCapsule_IsValid(ctx, meth_or_promoter, "numpy._ufunc_promoter")) {
        HPyErr_SetString(ctx, ctx->h_TypeError,
                "Second argument to info must be an ArrayMethod or promoter");
        res = -1;
        goto finish;
    }

    if (!HPyField_IsNull(ufunc_data->_loops)) {
        loops = HPyField_Load(ctx, ufunc, ufunc_data->_loops);
    }

    if (HPy_IsNull(loops)) {
        loops = HPyList_New(ctx, 0);
        if (HPy_IsNull(loops)) {
            res = -1;
            goto finish;
        }
        HPyField_Store(ctx, ufunc, &ufunc_data->_loops, loops);
        HPy_Close(ctx, loops);
    }

    HPy_ssize_t length = HPy_Length(ctx, loops);
    for (HPy_ssize_t i = 0; i < length; i++) {
        HPy item = HPy_GetItem_i(ctx, loops, i);
        HPy cur_DType_tuple = HPy_GetItem_i(ctx, item, 0);
        HPy_Close(ctx, item);
        int cmp = HPy_RichCompareBool(ctx, cur_DType_tuple, DType_tuple, HPy_EQ);
        HPy_Close(ctx, cur_DType_tuple);
        if (cmp < 0) {
            res = -1;
            goto finish;
        }
        if (cmp == 0) {
            continue;
        }
        if (ignore_duplicate) {
            res = 0;
            goto finish;
        }
        HPyErr_Format_p(ctx, ctx->h_TypeError,
                "A loop/promoter has already been registered with '%s' for ??",
                ufunc_get_name_cstr(ufunc_data));
        // TODO HPY LABS PORT: PyErr_Format
        // PyErr_Format(PyExc_TypeError,
        //        "A loop/promoter has already been registered with '%s' for %R",
        //        ufunc_get_name_cstr(ufunc), DType_tuple);
        res = -1;
        goto finish;
    }

    if (HPyList_Append(ctx, loops, info) < 0) {
        res = -1;
        goto finish;
    }
    res = 0;

finish:
    HPy_Close(ctx, DType_tuple);
    HPy_Close(ctx, loops);
    HPy_Close(ctx, meth_or_promoter);
    return res;
}


/**
 * Resolves the implementation to use, this uses typical multiple dispatching
 * methods of finding the best matching implementation or resolver.
 * (Based on `isinstance()`, the knowledge that non-abstract DTypes cannot
 * be subclassed is used, however.)
 *
 * NOTE: This currently does not take into account output dtypes which do not
 *       have to match.  The possible extension here is that if an output
 *       is given (and thus an output dtype), but not part of the signature
 *       we could ignore it for matching, but *prefer* a loop that matches
 *       better.
 *       Why is this not done currently?  First, it seems a niche feature that
 *       loops can only be distinguished based on the output dtype.  Second,
 *       there are some nasty theoretical things because:
 *
 *            np.add(f4, f4, out=f8)
 *            np.add(f4, f4, out=f8, dtype=f8)
 *
 *       are different, the first uses the f4 loop, the second the f8 loop.
 *       The problem is, that the current cache only uses the op_dtypes and
 *       both are `(f4, f4, f8)`.  The cache would need to store also which
 *       output was provided by `dtype=`/`signature=`.
 *
 * @param ufunc
 * @param op_dtypes The DTypes that are either passed in (defined by an
 *        operand) or defined by the `signature` as also passed in as
 *        `fixed_DTypes`.
 * @param out_info Returns the tuple describing the best implementation
 *        (consisting of dtypes and ArrayMethod or promoter).
 *        WARNING: Returns a borrowed reference!
 * @returns -1 on error 0 on success.  Note that the output can be NULL on
 *          success if nothing is found.
 */
static int
resolve_implementation_info(HPyContext *ctx,
        HPy ufunc, PyUFuncObject *ufunc_data,
        HPy /* (PyArray_DTypeMeta *) */ op_dtypes[], npy_bool only_promoters,
        HPy *out_info)
{
    int res;
    int nin = ufunc_data->nin, nargs = ufunc_data->nargs;
    HPy _loops = HPyField_Load(ctx, ufunc, ufunc_data->_loops);
    HPy_ssize_t size = HPy_Length(ctx, _loops);
    HPy best_dtypes = HPy_NULL;
    HPy best_resolver_info = HPy_NULL;
    HPy resolver_info = HPy_NULL;
    HPy curr_dtypes = HPy_NULL;

#if PROMOTION_DEBUG_TRACING
    printf("Promoting for '%s' promoters only: %d\n",
            ufunc_data->name ? ufunc_data->name : "<unknown>", (int)only_promoters);
    printf("    DTypes: ");
    HPy tmp = HPyArray_TupleFromItems(ctx, ufunc_data->nargs, op_dtypes, 1);
    HPy_Print(ctx, tmp, stdout, 0);
    printf("\n");
    HPy_Close(ctx, tmp);
#endif

    for (HPy_ssize_t res_idx = 0; res_idx < size; res_idx++) {
        /* Test all resolvers  */
        HPy_SETREF(ctx, resolver_info, HPy_GetItem_i(ctx, _loops, res_idx));

        HPy resolver_info_1 = HPy_GetItem_i(ctx, resolver_info, 1);
        if (only_promoters && HPyGlobal_TypeCheck(ctx,
                    resolver_info_1, HPyArrayMethod_Type)) {
            HPy_Close(ctx, resolver_info_1);
            continue;
        }
        HPy_Close(ctx, resolver_info_1);

        HPy_SETREF(ctx, curr_dtypes, HPy_GetItem_i(ctx, resolver_info, 0));
        /*
         * Test if the current resolver matches, it could make sense to
         * reorder these checks to avoid the IsSubclass check as much as
         * possible.
         */

        npy_bool matches = NPY_TRUE;
        /*
         * NOTE: We currently match the output dtype exactly here, this is
         *       actually only necessary if the signature includes.
         *       Currently, we rely that op-dtypes[nin:nout] is NULLed if not.
         */
        for (HPy_ssize_t i = 0; i < nargs; i++) {
            HPy given_dtype = op_dtypes[i];
            HPy /* (PyArray_DTypeMeta *) */ resolver_dtype = HPy_GetItem_i(ctx, curr_dtypes, i);
            assert(!HPy_Is(ctx, given_dtype, ctx->h_None));
            if (HPy_IsNull(given_dtype)) {
                if (i >= nin) {
                    /* Unspecified out always matches (see below for inputs) */
                    continue;
                }
                /*
                 * This is a reduce-like operation, which always have the form
                 * `(res_DType, op_DType, res_DType)`.  If the first and last
                 * dtype of the loops match, this should be reduce-compatible.
                 */
                HPy resolver_dtype_0 = HPy_GetItem_i(ctx, curr_dtypes, 0);
                HPy resolver_dtype_2 = HPy_GetItem_i(ctx, curr_dtypes, 2);
                int same = HPy_Is(ctx, resolver_dtype_0, resolver_dtype_2);
                HPy_Close(ctx, resolver_dtype_0);
                HPy_Close(ctx, resolver_dtype_2);
                if (same) {
                    continue;
                }
            }

            if (HPy_Is(ctx, resolver_dtype, ctx->h_None)) {
                HPy_Close(ctx, resolver_dtype);
                /* always matches */
                continue;
            }
            if (HPy_Is(ctx, given_dtype, resolver_dtype)) {
                HPy_Close(ctx, resolver_dtype);
                continue;
            }
            if (!NPY_DT_is_abstract(PyArray_DTypeMeta_AsStruct(ctx, resolver_dtype))) {
                HPy_Close(ctx, resolver_dtype);
                matches = NPY_FALSE;
                break;
            }

            // TODO HPY LABS PORT: PyObject_IsSubclass
            int subclass = HPyType_IsSubtype(ctx, given_dtype, resolver_dtype);
            HPy_Close(ctx, resolver_dtype);
            if (subclass < 0) {
                res = -1;
                goto finish;
            }
            if (!subclass) {
                matches = NPY_FALSE;
                break;
            }
            /*
             * TODO: Could consider allowing reverse subclass relation, i.e.
             *       the operation DType passed in to be abstract.  That
             *       definitely is OK for outputs (and potentially useful,
             *       you could enforce e.g. an inexact result).
             *       It might also be useful for some stranger promoters.
             */
        }
        if (!matches) {
            continue;
        }

        /* The resolver matches, but we have to check if it is better */
        if (!HPy_IsNull(best_dtypes)) {
            int current_best = -1;  /* -1 neither, 0 current best, 1 new */
            /*
             * If both have concrete and None in the same position and
             * they are identical, we will continue searching using the
             * first best for comparison, in an attempt to find a better
             * one.
             * In all cases, we give up resolution, since it would be
             * necessary to compare to two "best" cases.
             */
            for (HPy_ssize_t i = 0; i < nargs; i++) {
                if (i == ufunc_data->nin && current_best != -1) {
                    /* inputs prefer one loop and outputs have lower priority */
                    break;
                }

                int best;

                HPy prev_dtype = HPy_GetItem_i(ctx, best_dtypes, i);
                HPy new_dtype = HPy_GetItem_i(ctx, curr_dtypes, i);

                if (HPy_Is(ctx, prev_dtype, new_dtype)) {
                    HPy_Close(ctx, prev_dtype);
                    HPy_Close(ctx, new_dtype);
                    /* equivalent, so this entry does not matter */
                    continue;
                }
                if (HPy_IsNull(op_dtypes[i])) {
                    HPy_Close(ctx, prev_dtype);
                    HPy_Close(ctx, new_dtype);
                    /*
                     * If an a dtype is NULL it always matches, so there is no
                     * point in defining one as more precise than the other.
                     */
                    continue;
                }
                /* If either is None, the other is strictly more specific */
                if (HPy_Is(ctx, prev_dtype, ctx->h_None)) {
                    best = 1;
                }
                else if (HPy_Is(ctx, new_dtype, ctx->h_None)) {
                    best = 0;
                }
                /*
                 * If both are concrete and not identical, this is
                 * ambiguous.
                 */
                else if (!NPY_DT_is_abstract(PyArray_DTypeMeta_AsStruct(ctx, prev_dtype)) &&
                         !NPY_DT_is_abstract(PyArray_DTypeMeta_AsStruct(ctx, new_dtype))) {
                    /*
                     * Ambiguous unless they are identical (checked above),
                     * or one matches exactly.
                     */
                    if (HPy_Is(ctx, prev_dtype, op_dtypes[i])) {
                        best = 0;
                    }
                    else if (HPy_Is(ctx, new_dtype, op_dtypes[i])) {
                        best = 1;
                    }
                    else {
                        best = -1;
                    }
                }
                else if (!NPY_DT_is_abstract(PyArray_DTypeMeta_AsStruct(ctx, prev_dtype))) {
                    /* old is not abstract, so better (both not possible) */
                    best = 0;
                }
                else if (!NPY_DT_is_abstract(PyArray_DTypeMeta_AsStruct(ctx, new_dtype))) {
                    /* new is not abstract, so better (both not possible) */
                    best = 1;
                }
                /*
                 * TODO: This will need logic for abstract DTypes to decide if
                 *       one is a subclass of the other (And their subclass
                 *       relation is well defined).  For now, we bail out
                 *       in cas someone manages to get here.
                 */
                else {
                    HPyErr_SetString(ctx, ctx->h_NotImplementedError,
                            "deciding which one of two abstract dtypes is "
                            "a better match is not yet implemented.  This "
                            "will pick the better (or bail) in the future.");
                    HPy_Close(ctx, prev_dtype);
                    HPy_Close(ctx, new_dtype);
                    *out_info = HPy_NULL;
                    res = -1;
                    goto finish;
                }

                if (best == -1) {
                    HPy_Close(ctx, prev_dtype);
                    HPy_Close(ctx, new_dtype);
                    /* no new info, nothing to update */
                    continue;
                }
                if ((current_best != -1) && (current_best != best)) {
                    /*
                     * We need a clear best, this could be tricky, unless
                     * the signature is identical, we would have to compare
                     * against both of the found ones until we find a
                     * better one.
                     * Instead, only support the case where they are
                     * identical.
                     */
                    /* TODO: Document the above comment, may need relaxing? */
                    HPy_Close(ctx, prev_dtype);
                    HPy_Close(ctx, new_dtype);
                    current_best = -1;
                    break;
                }
                HPy_Close(ctx, prev_dtype);
                HPy_Close(ctx, new_dtype);
                current_best = best;
            }

            if (current_best == -1) {
                /*
                 * We could not find a best loop, but promoters should be
                 * designed in a way to disambiguate such scenarios, so we
                 * retry the whole lookup using only promoters.
                 * (There is a small chance we already got two promoters.
                 * We just redo it anyway for simplicity.)
                 */
                if (!only_promoters) {
                    return resolve_implementation_info(ctx, ufunc, ufunc_data,
                            op_dtypes, NPY_TRUE, out_info);
                }
                /*
                 * If this is already the retry, we are out of luck.  Promoters
                 * should be designed in a way that this cannot happen!
                 * (It should be noted, that the retry might not find anything
                 * and we still do a legacy lookup later.)
                 */
                HPy given = HPyArray_TupleFromItems(ctx,
                        ufunc_data->nargs, op_dtypes, 1);
                if (!HPy_IsNull(given)) {
                    PyErr_Format(PyExc_RuntimeError,
                            "Could not find a loop for the inputs:\n    %S\n"
                            "The two promoters %S and %S matched the input "
                            "equally well.  Promoters must be designed "
                            "to be unambiguous.  NOTE: This indicates an error "
                            "in NumPy or an extending library and should be "
                            "reported.",
                            given, best_dtypes, curr_dtypes);
                    HPy_Close(ctx, given);
                }
                *out_info = HPy_NULL;
                res = 0;
                goto finish;
            }
            else if (current_best == 0) {
                /* The new match is not better, continue looking. */
                continue;
            }
        }
        /* The new match is better (or there was no previous match) */
        best_dtypes = curr_dtypes;
        if (!HPy_IsNull(best_resolver_info)) {
            HPy_Close(ctx, best_resolver_info);
        }
        best_resolver_info = HPy_Dup(ctx, resolver_info);
    }
    if (HPy_IsNull(best_dtypes)) {
        /* The non-legacy lookup failed */
        *out_info = HPy_NULL;
        res = 0;
        goto finish;
    }

    *out_info = best_resolver_info;
    res = 0;
finish:
    HPy_Close(ctx, _loops);
    HPy_Close(ctx, resolver_info);
    HPy_Close(ctx, curr_dtypes);
    return res;
}


/*
 * A promoter can currently be either a C-Capsule containing a promoter
 * function pointer, or a Python function.  Both of these can at this time
 * only return new operation DTypes (i.e. mutate the input while leaving
 * those defined by the `signature` unmodified).
 */
static HPy
hpy_call_promoter_and_recurse(HPyContext *ctx,
        HPy /* PyUFuncObject * */ ufunc, PyUFuncObject * ufunc_struct,
        HPy promoter,
        HPy /* PyArray_DTypeMeta * */ op_dtypes[], 
        HPy /* PyArray_DTypeMeta * */ signature[],
        HPy /* PyArrayObject * */ const operands[])
{
    int nargs = ufunc_struct->nargs;
    HPy resolved_info = HPy_NULL;

    int promoter_result;
    HPy new_op_dtypes[NPY_MAXARGS]; // PyArray_DTypeMeta *

    HPy promoter_type = HPy_Type(ctx, promoter);
    int promoter_is_capsule = HPy_Is(ctx, promoter_type, ctx->h_CapsuleType);
    HPy_Close(ctx, promoter_type);
    if (promoter_is_capsule) {
        /* We could also go the other way and wrap up the python function... */
        promoter_function *promoter_function = HPyCapsule_GetPointer(ctx, promoter,
                "numpy._ufunc_promoter");
        if (promoter_function == NULL) {
            return HPy_NULL;
        }
        promoter_result = promoter_function(ctx, ufunc,
                op_dtypes, signature, new_op_dtypes);
    }
    else {
        HPyErr_SetString(ctx, ctx->h_NotImplementedError,
                "Calling python functions for promotion is not implemented.");
        return HPy_NULL;
    }
    if (promoter_result < 0) {
        return HPy_NULL;
    }
    /*
     * If none of the dtypes changes, we would recurse infinitely, abort.
     * (Of course it is nevertheless possible to recurse infinitely.)
     */
    int dtypes_changed = 0;
    for (int i = 0; i < nargs; i++) {
        if (!HPy_Is(ctx, new_op_dtypes[i], op_dtypes[i])) {
            dtypes_changed = 1;
            break;
        }
    }
    if (!dtypes_changed) {
        goto finish;
    }

    /*
     * Do a recursive call, the promotion function has to ensure that the
     * new tuple is strictly more precise (thus guaranteeing eventual finishing)
     */
    // if (Py_EnterRecursiveCall(" during ufunc promotion.") != 0) {
    //     goto finish;
    // }
    resolved_info = hpy_promote_and_get_info_and_ufuncimpl(ctx, ufunc,
            operands, signature, new_op_dtypes,
            /* no legacy promotion */ NPY_FALSE);

    // Py_LeaveRecursiveCall();

  finish:
    for (int i = 0; i < nargs; i++) {
        HPy_Close(ctx, new_op_dtypes[i]);
    }
    return resolved_info;
}


/*
 * Convert the DType `signature` into the tuple of descriptors that is used
 * by the old ufunc type resolvers in `ufunc_type_resolution.c`.
 *
 * Note that we do not need to pass the type tuple when we use the legacy path
 * for type resolution rather than promotion, since the signature is always
 * correct in that case.
 */
static int
_make_new_typetup(HPyContext *ctx,
        int nop, HPy /* (PyArray_DTypeMeta *) */ signature[], HPy *out_typetup) {
    HPyTupleBuilder builder = HPyTupleBuilder_New(ctx, nop);
    if (HPyTupleBuilder_IsNull(builder)) {
        return -1;
    }
    // *out_typetup = HPyTuple_New(ctx, nop);

    int none_count = 0;
    for (int i = 0; i < nop; i++) {
        HPy item;
        if (HPy_IsNull(signature[i])) {
            HPyTupleBuilder_Set(ctx, builder, i, ctx->h_None);
            none_count++;
        }
        else {
            PyArray_DTypeMeta *signature_i_data = PyArray_DTypeMeta_AsStruct(ctx, signature[i]);
            if (!NPY_DT_is_legacy(signature_i_data)
                    || NPY_DT_is_abstract(signature_i_data)) {
                /*
                 * The legacy type resolution can't deal with these.
                 * This path will return `None` or so in the future to
                 * set an error later if the legacy type resolution is used.
                 */
                HPyErr_SetString(ctx, ctx->h_RuntimeError,
                        "Internal NumPy error: new DType in signature not yet "
                        "supported. (This should be unreachable code!)");
                HPy_SETREF(ctx, *out_typetup, HPy_NULL);
                return -1;
            }
            item = hdtypemeta_get_singleton(ctx, signature[i]);
            HPyTupleBuilder_Set(ctx, builder, i, item);
            HPy_Close(ctx, item);
        }
        // Py_INCREF(item);
        // PyTuple_SET_ITEM(*out_typetup, i, item);
    }
    if (none_count == nop) {
        /* The whole signature was None, simply ignore type tuple */
        // Py_DECREF(*out_typetup);
        // *out_typetup = NULL;
        HPyTupleBuilder_Cancel(ctx, builder);
        *out_typetup = HPy_NULL;
    } else {
        *out_typetup = HPyTupleBuilder_Build(ctx, builder);
        if (HPy_IsNull(*out_typetup)) {
            return -1;
        }
    }
    return 0;
}


/*
 * Fills in the operation_DTypes with borrowed references.  This may change
 * the content, since it will use the legacy type resolution, which can special
 * case 0-D arrays (using value-based logic).
 */
static int
hpy_legacy_promote_using_legacy_type_resolver(HPyContext *ctx, HPy /* (PyUFuncObject *) */ ufunc,
        HPy /* (PyArrayObject *) */ const *ops, HPy /* (PyArray_DTypeMeta *) */ signature[],
        HPy /* (PyArray_DTypeMeta *) */ operation_DTypes[], int *out_cacheable)
{
    PyUFuncObject *ufunc_data = PyUFuncObject_AsStruct(ctx, ufunc);
    int nargs = ufunc_data->nargs;
    HPy out_descrs[NPY_MAXARGS] = {HPy_NULL}; /* (PyArray_Descr *) */

    HPy type_tuple = HPy_NULL;
    if (_make_new_typetup(ctx, nargs, signature, &type_tuple) < 0) {
        return -1;
    }

    /*
     * We use unsafe casting. This is of course not accurate, but that is OK
     * here, because for promotion/dispatching the casting safety makes no
     * difference.  Whether the actual operands can be casts must be checked
     * during the type resolution step (which may _also_ calls this!).
     */
    if (ufunc_data->hpy_type_resolver(ctx, ufunc,
            NPY_UNSAFE_CASTING, (HPy *)ops, type_tuple,
            out_descrs) < 0) {
        HPy_Close(ctx, type_tuple);
        /* Not all legacy resolvers clean up on failures: */
        for (int i = 0; i < nargs; i++) {
            HPy_Close(ctx, out_descrs[i]);
        }
        return -1;
    }
    HPy_Close(ctx, type_tuple);

    for (int i = 0; i < nargs; i++) {
        HPy_SETREF(ctx, operation_DTypes[i], HNPY_DTYPE(ctx, out_descrs[i]));
        HPy_Close(ctx, out_descrs[i]);
    }
    /*
     * The PyUFunc_SimpleBinaryComparisonTypeResolver has a deprecation
     * warning (ignoring `dtype=`) and cannot be cached.
     * All datetime ones *should* have a warning, but currently don't,
     * but ignore all signature passing also.  So they can also
     * not be cached, and they mutate the signature which of course is wrong,
     * but not doing it would confuse the code later.
     */
    for (int i = 0; i < nargs; i++) {
        if (!HPy_IsNull(signature[i]) && !HPy_Is(ctx, signature[i], operation_DTypes[i])) {
            HPy_SETREF(ctx, signature[i], HPy_Dup(ctx, operation_DTypes[i]));
            *out_cacheable = 0;
        }
    }
    return 0;
}

static int
legacy_promote_using_legacy_type_resolver(PyUFuncObject *ufunc,
        PyArrayObject *const *ops, PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *operation_DTypes[], int *out_cacheable)
{
    hpy_abort_not_implemented("legacy_promote_using_legacy_type_resolver");
    return -1;
}

/*
 * Note, this function *DOES NOT* return a borrowed reference to info.
 */
NPY_NO_EXPORT HPy
hpy_add_and_return_legacy_wrapping_ufunc_loop(HPyContext *ctx, HPy /* (PyUFuncObject *) */ ufunc,
        HPy /* (PyArray_DTypeMeta *) */ operation_dtypes[], int ignore_duplicate)
{
    PyUFuncObject *ufunc_data = PyUFuncObject_AsStruct(ctx, ufunc);
    HPy DType_tuple = HPyArray_TupleFromItems(ctx, ufunc_data->nargs,
            operation_dtypes, 0);
    if (HPy_IsNull(DType_tuple)) {
        return HPy_NULL;
    }

    // PyArrayMethodObject *
    HPy method = HPyArray_NewLegacyWrappingArrayMethod(ctx,
            ufunc_data, operation_dtypes);
    if (HPy_IsNull(method)) {
        HPy_Close(ctx, DType_tuple);
        return HPy_NULL;
    }
    HPy info = HPyTuple_Pack(ctx, 2, DType_tuple, method);
    HPy_Close(ctx, DType_tuple);
    HPy_Close(ctx, method);
    if (HPy_IsNull(info)) {
        return HPy_NULL;
    }
    if (HPyUFunc_AddLoop(ctx, ufunc, info, ignore_duplicate) < 0) {
        HPy_Close(ctx, info);
        return HPy_NULL;
    }
    // Py_DECREF(info);  /* now borrowed from the ufunc's list of loops */
    return info;
}

/*
 * Note, this function returns a BORROWED references to info since it adds
 * it to the loops.
 */
NPY_NO_EXPORT PyObject *
add_and_return_legacy_wrapping_ufunc_loop(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *operation_dtypes[], int ignore_duplicate)
{
    HPyContext *ctx = npy_get_context();
    HPy h_ufunc = HPy_FromPyObject(ctx, (PyObject *)ufunc);
    HPy *h_operation_dtypes = HPy_FromPyObjectArray(ctx, (PyObject **)operation_dtypes, ufunc->nargs);
    HPy h_res = hpy_add_and_return_legacy_wrapping_ufunc_loop(ctx, h_ufunc, h_operation_dtypes, ignore_duplicate);
    PyObject *res = HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_CloseAndFreeArray(ctx, h_operation_dtypes, ufunc->nargs);
    HPy_Close(ctx, h_ufunc);

    Py_DECREF(res);  /* now borrowed from the ufunc's list of loops */
    return res;
}


/*
 * The main implementation to find the correct DType signature and ArrayMethod
 * to use for a ufunc.  This function may recurse with `do_legacy_fallback`
 * set to False.
 *
 * If value-based promotion is necessary, this is handled ahead of time by
 * `promote_and_get_ufuncimpl`.
 */
static NPY_INLINE HPy
hpy_promote_and_get_info_and_ufuncimpl(HPyContext *ctx,
        HPy /* (PyUFuncObject *) */ ufunc,
        HPy /* (PyArrayObject *) */ const *ops,
        HPy /* (PyArray_DTypeMeta *) */ signature[],
        HPy /* (PyArray_DTypeMeta *) */ op_dtypes[],
        npy_bool allow_legacy_promotion)
{
    HPy res;
    PyUFuncObject *ufunc_data = PyUFuncObject_AsStruct(ctx, ufunc);
    PyArray_DTypeMeta **py_op_dtypes = NULL;
    /*
     * Fetch the dispatching info which consists of the implementation and
     * the DType signature tuple.  There are three steps:
     *
     * 1. Check the cache.
     * 2. Check all registered loops/promoters to find the best match.
     * 3. Fall back to the legacy implementation if no match was found.
     */
    HPy info = HPyArrayIdentityHash_GetItem(ctx, ufunc,
            ufunc_data->_dispatch_cache, op_dtypes);
    HPy array_method_type = HPyGlobal_Load(ctx, HPyArrayMethod_Type);
    HPy info1 = HPy_NULL;
    if (!HPy_IsNull(info)) {
        info1 = HPy_GetItem_i(ctx, info, 1);
        if (HPy_TypeCheck(ctx, info1, array_method_type)) {
            /* Found the ArrayMethod and NOT a promoter: return it */
            res = info;
            goto finish;
        }
    }

    /*
     * If `info == NULL`, loading from cache failed, use the full resolution
     * in `resolve_implementation_info` (which caches its result on success).
     */
    if (HPy_IsNull(info)) {
        if (resolve_implementation_info(ctx, ufunc, ufunc_data,
                op_dtypes, NPY_FALSE, &info) < 0) {
            res = HPy_NULL;
            goto finish;
        }

        if (!HPy_IsNull(info)) {
            HPy_SETREF(ctx, info1, HPy_GetItem_i(ctx, info, 1));
            if (HPy_TypeCheck(ctx, info1, array_method_type)) {
                /*
                 * Found the ArrayMethod and NOT promoter.  Before returning it
                 * add it to the cache for faster lookup in the future.
                 */
                if (HPyArrayIdentityHash_SetItem(ctx, ufunc,
                        ufunc_data->_dispatch_cache, op_dtypes, info, 0) < 0) {
                    res = HPy_NULL;
                    goto finish;
                }
                res = info;
                goto finish;
            }
        }
    }

    /*
     * At this point `info` is NULL if there is no matching loop, or it is
     * a promoter that needs to be used/called:
     */
    if (!HPy_IsNull(info)) {
        // HPy promoter = HPy_GetItem_i(ctx, info, 1);
        HPy promoter = info1;
        info = hpy_call_promoter_and_recurse(ctx, ufunc, ufunc_data,
                promoter, op_dtypes, signature, ops);
        if (HPy_IsNull(info) && HPyErr_Occurred(ctx)) {
            res = HPy_NULL;
            goto finish;
        }
        else if (!HPy_IsNull(info)) {
            /* Add result to the cache using the original types: */
            if (HPyArrayIdentityHash_SetItem(ctx, ufunc,
                    ufunc_data->_dispatch_cache, op_dtypes, info, 0) < 0) {
                res = HPy_NULL;
                goto finish;
            }
            res = info;
            goto finish;
        }
    }

    /*
     * Even using promotion no loop was found.
     * Using promotion failed, this should normally be an error.
     * However, we need to give the legacy implementation a chance here.
     * (it will modify `op_dtypes`).
     */
    if (!allow_legacy_promotion || ufunc_data->type_resolver == NULL ||
            (ufunc_data->ntypes == 0 && HPyField_IsNull(ufunc_data->userloops))) {
        /* Already tried or not a "legacy" ufunc (no loop found, return) */
        res = HPy_NULL;
        goto finish;
    }

    HPy new_op_dtypes[NPY_MAXARGS] = {HPy_NULL}; /* (PyArray_DTypeMeta *) */
    int cacheable = 1;  /* TODO: only the comparison deprecation needs this */
    if (hpy_legacy_promote_using_legacy_type_resolver(ctx, ufunc,
            ops, signature, new_op_dtypes, &cacheable) < 0) {
        res = HPy_NULL;
        goto finish;
    }
    info = hpy_promote_and_get_info_and_ufuncimpl(ctx, ufunc,
            ops, signature, new_op_dtypes, NPY_FALSE);
    for (int i = 0; i < ufunc_data->nargs; i++) {
        HPy_Close(ctx, new_op_dtypes[i]);
    }

    /* Add this to the cache using the original types: */
    if (cacheable && HPyArrayIdentityHash_SetItem(ctx, ufunc,
            ufunc_data->_dispatch_cache, op_dtypes, info, 0) < 0) {
        res = HPy_NULL;
        goto finish;
    }
    res = info;
finish:
    if (py_op_dtypes) {
        HPy_DecrefAndFreeArray(ctx, (PyObject **)py_op_dtypes, ufunc_data->nargs);
    }
    HPy_Close(ctx, array_method_type);
    HPy_Close(ctx, info1);
    return res;
}

static NPY_INLINE PyObject *
promote_and_get_info_and_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool allow_legacy_promotion)
{
    HPyContext *ctx = npy_get_context();
    int nargs = ufunc->nargs;
    HPy h_ufunc = HPy_FromPyObject(ctx, (PyObject *)ufunc);
    HPy *h_ops = HPy_FromPyObjectArray(ctx, (PyObject **)ops, nargs);
    HPy *h_signature = HPy_FromPyObjectArray(ctx, (PyObject **)signature, nargs);
    HPy *h_op_dtypes = HPy_FromPyObjectArray(ctx, (PyObject **)op_dtypes, nargs);

    HPy h_res = hpy_promote_and_get_info_and_ufuncimpl(ctx, h_ufunc, h_ops, h_signature, h_op_dtypes, allow_legacy_promotion);

    PyObject *res = HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);

    for (int i=0; i < nargs; i++) {
        Py_XSETREF(signature[i], (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_signature[i]));
        Py_XSETREF(op_dtypes[i], (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_op_dtypes[i]));
    }

    HPy_CloseAndFreeArray(ctx, h_op_dtypes, nargs);
    HPy_CloseAndFreeArray(ctx, h_signature, nargs);
    HPy_CloseAndFreeArray(ctx, h_ops, nargs);
    HPy_Close(ctx, h_ufunc);

    return res;
}


/**
 * The central entry-point for the promotion and dispatching machinery.
 *
 * It currently may work with the operands (although it would be possible to
 * only work with DType (classes/types).  This is because it has to ensure
 * that legacy (value-based promotion) is used when necessary.
 *
 * NOTE: The machinery here currently ignores output arguments unless
 *       they are part of the signature.  This slightly limits unsafe loop
 *       specializations, which is important for the `ensure_reduce_compatible`
 *       fallback mode.
 *       To fix this, the caching mechanism (and dispatching) can be extended.
 *       When/if that happens, the `ensure_reduce_compatible` could be
 *       deprecated (it should never kick in because promotion kick in first).
 *
 * @param ufunc The ufunc object, used mainly for the fallback.
 * @param ops The array operands (used only for the fallback).
 * @param signature As input, the DType signature fixed explicitly by the user.
 *        The signature is *filled* in with the operation signature we end up
 *        using.
 * @param op_dtypes The operand DTypes (without casting) which are specified
 *        either by the `signature` or by an `operand`.
 *        (outputs and the second input can be NULL for reductions).
 *        NOTE: In some cases, the promotion machinery may currently modify
 *        these including clearing the output.
 * @param force_legacy_promotion If set, we have to use the old type resolution
 *        to implement value-based promotion/casting.
 * @param ensure_reduce_compatible Must be set for reductions, in which case
 *        the found implementation is checked for reduce-like compatibility.
 *        If it is *not* compatible and `signature[2] != NULL`, we assume its
 *        output DType is correct (see NOTE above).
 *        If removed, promotion may require information about whether this
 *        is a reduction, so the more likely case is to always keep fixing this
 *        when necessary, but push down the handling so it can be cached.
 */
NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool force_legacy_promotion,
        npy_bool allow_legacy_promotion,
        npy_bool ensure_reduce_compatible)
{
    HPyContext *ctx = npy_get_context();
    int nargs = ufunc->nargs;
    HPy h_ufunc = HPy_FromPyObject(ctx, (PyObject *)ufunc);
    HPy *h_ops = HPy_FromPyObjectArray(ctx, (PyObject **)ops, nargs);
    HPy *h_signature = HPy_FromPyObjectArray(ctx, (PyObject **)signature, nargs);
    HPy *h_op_dtypes = HPy_FromPyObjectArray(ctx, (PyObject **)op_dtypes, nargs);

    HPy h_res = hpy_promote_and_get_ufuncimpl(ctx, h_ufunc, (HPy const *)h_ops, h_signature,
            h_op_dtypes, force_legacy_promotion, allow_legacy_promotion,
            ensure_reduce_compatible);

    PyArrayMethodObject *res = (PyArrayMethodObject *)HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);

    for (int i=0; i < nargs; i++) {
        Py_XSETREF(signature[i], (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_signature[i]));
        Py_XSETREF(op_dtypes[i], (PyArray_DTypeMeta *)HPy_AsPyObject(ctx, h_op_dtypes[i]));
    }

    HPy_CloseAndFreeArray(ctx, h_op_dtypes, nargs);
    HPy_CloseAndFreeArray(ctx, h_signature, nargs);
    HPy_CloseAndFreeArray(ctx, h_ops, nargs);
    HPy_Close(ctx, h_ufunc);

    return res;
}

NPY_NO_EXPORT HPy
hpy_promote_and_get_ufuncimpl(HPyContext *ctx,
        HPy /* (PyUFuncObject *) */ ufunc,
        HPy /* (PyArrayObject *) */ const ops[],
        HPy /* (PyArray_DTypeMeta *) */ signature[],
        HPy /* (PyArray_DTypeMeta *) */ op_dtypes[],
        npy_bool force_legacy_promotion,
        npy_bool allow_legacy_promotion,
        npy_bool ensure_reduce_compatible)
{
    PyUFuncObject *ufunc_data = PyUFuncObject_AsStruct(ctx, ufunc);
    int nin = ufunc_data->nin, nargs = ufunc_data->nargs;

    /*
     * Get the actual DTypes we operate with by mixing the operand array
     * ones with the passed signature.
     */
    for (int i = 0; i < nargs; i++) {
        if (!HPy_IsNull(signature[i])) {
            /*
             * ignore the operand input, we cannot overwrite signature yet
             * since it is fixed (cannot be promoted!)
             */
            // Py_INCREF(signature[i]);
            HPy_SETREF(ctx, op_dtypes[i], HPy_Dup(ctx, signature[i]));
            assert(i >= ufunc_data->nin || !NPY_DT_is_abstract(PyArray_DTypeMeta_AsStruct(ctx, signature[i])));
        }
        else if (i >= nin) {
            /*
             * We currently just ignore outputs if not in signature, this will
             * always give the/a correct result (limits registering specialized
             * loops which include the cast).
             * (See also comment in resolve_implementation_info.)
             */
            // TODO HPY LABS PORT: is that stealing the reference ?
            // Py_CLEAR(op_dtypes[i]);
            HPy_SETREF(ctx, op_dtypes[i], HPy_NULL);
        }
    }

    if (force_legacy_promotion) {
        /*
         * We must use legacy promotion for value-based logic. Call the old
         * resolver once up-front to get the "actual" loop dtypes.
         * After this (additional) promotion, we can even use normal caching.
         */
        int cacheable = 1;  /* unused, as we modify the original `op_dtypes` */
        if (hpy_legacy_promote_using_legacy_type_resolver(ctx, ufunc,
                ops, signature, op_dtypes, &cacheable) < 0) {
            return HPy_NULL;
        }
    }

    HPy info = hpy_promote_and_get_info_and_ufuncimpl(ctx, ufunc,
            ops, signature, op_dtypes, allow_legacy_promotion);

    if (HPy_IsNull(info)) {
        if (!HPyErr_Occurred(ctx)) {
            hpy_raise_no_loop_found_error(ctx, ufunc, op_dtypes);
        }
        return HPy_NULL;
    }

    // TODO HPY LABS PORT: seems like the returned ref is borrowed ??
    // PyArrayMethodObject *method = (PyArrayMethodObject *)PyTuple_GET_ITEM(info, 1);
    HPy method = HPy_GetItem_i(ctx, info, 1); /* (PyArrayMethodObject *) */

    /*
     * In certain cases (only the logical ufuncs really), the loop we found may
     * not be reduce-compatible.  Since the machinery can't distinguish a
     * reduction with an output from a normal ufunc call, we have to assume
     * the result DType is correct and force it for the input (if not forced
     * already).
     * NOTE: This does assume that all loops are "safe" see the NOTE in this
     *       comment.  That could be relaxed, in which case we may need to
     *       cache if a call was for a reduction.
     */
    HPy all_dtypes = HPy_GetItem_i(ctx, info, 0);
    if (ensure_reduce_compatible && HPy_IsNull(signature[0])) {
        HPy all_dtypes0 = HPy_GetItem_i(ctx, all_dtypes, 0);
        HPy all_dtypes2 = HPy_GetItem_i(ctx, all_dtypes, 2);
        int is_same = HPy_Is(ctx, all_dtypes0, all_dtypes2);
        HPy_Close(ctx, all_dtypes0);
        if (!is_same) {
            // signature[0] = (PyArray_DTypeMeta *)PyTuple_GET_ITEM(all_dtypes, 2);
            // Py_INCREF(signature[0]);
            HPy_Close(ctx, all_dtypes);
            signature[0] = all_dtypes2;
            return hpy_promote_and_get_ufuncimpl(ctx, ufunc,
                    ops, signature, op_dtypes,
                    force_legacy_promotion, allow_legacy_promotion, NPY_FALSE);
        }
        HPy_Close(ctx, all_dtypes2);
    }

    for (int i = 0; i < nargs; i++) {
        if (HPy_IsNull(signature[i])) {
            // signature[i] = (PyArray_DTypeMeta *)PyTuple_GET_ITEM(all_dtypes, i);
            // Py_INCREF(signature[i]);
            signature[i] = HPy_GetItem_i(ctx, all_dtypes, i);
        }
#ifndef NDEBUG
        else {
            HPy tmp = HPy_GetItem_i(ctx, all_dtypes, i);
            assert(HPy_Is(ctx, signature[i], tmp));
            HPy_Close(ctx, tmp);
        }
#endif
    }
    HPy_Close(ctx, all_dtypes);

    return method;
}


/*
 * Generic promoter used by as a final fallback on ufuncs.  Most operations are
 * homogeneous, so we can try to find the homogeneous dtype on the inputs
 * and use that.
 * We need to special case the reduction case, where op_dtypes[0] == NULL
 * is possible.
 */
NPY_NO_EXPORT int
default_ufunc_promoter(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    if (ufunc->hpy_type_resolver == &HPyUFunc_SimpleBinaryComparisonTypeResolver
            && signature[0] == NULL && signature[1] == NULL
            && signature[2] != NULL && signature[2]->type_num != NPY_BOOL) {
        /* bail out, this is _only_ to give future/deprecation warning! */
        return -1;
    }

    /* If nin < 2 promotion is a no-op, so it should not be registered */
    assert(ufunc->nin > 1);
    if (op_dtypes[0] == NULL) {
        assert(ufunc->nin == 2 && ufunc->nout == 1);  /* must be reduction */
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[0] = op_dtypes[1];
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[1] = op_dtypes[1];
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[2] = op_dtypes[1];
        return 0;
    }
    PyArray_DTypeMeta *common = NULL;
    /*
     * If a signature is used and homogeneous in its outputs use that
     * (Could/should likely be rather applied to inputs also, although outs
     * only could have some advantage and input dtypes are rarely enforced.)
     */
    for (int i = ufunc->nin; i < ufunc->nargs; i++) {
        if (signature[i] != NULL) {
            if (common == NULL) {
                Py_INCREF(signature[i]);
                common = signature[i];
            }
            else if (common != signature[i]) {
                Py_CLEAR(common);  /* Not homogeneous, unset common */
                break;
            }
        }
    }
    /* Otherwise, use the common DType of all input operands */
    if (common == NULL) {
        common = PyArray_PromoteDTypeSequence(ufunc->nin, op_dtypes);
        if (common == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();  /* Do not propagate normal promotion errors */
            }
            return -1;
        }
    }

    for (int i = 0; i < ufunc->nargs; i++) {
        PyArray_DTypeMeta *tmp = common;
        if (signature[i]) {
            tmp = signature[i];  /* never replace a fixed one. */
        }
        Py_INCREF(tmp);
        new_op_dtypes[i] = tmp;
    }
    for (int i = ufunc->nin; i < ufunc->nargs; i++) {
        Py_XINCREF(op_dtypes[i]);
        new_op_dtypes[i] = op_dtypes[i];
    }

    Py_DECREF(common);
    return 0;
}


/*
 * In some cases, we assume that there will only ever be object loops,
 * and the object loop should *always* be chosen.
 * (in those cases more specific loops should not really be registered, but
 * we do not check that.)
 *
 * We default to this for "old-style" ufuncs which have exactly one loop
 * consisting only of objects (during registration time, numba mutates this
 * but presumably).
 */
NPY_NO_EXPORT int
object_only_ufunc_promoter(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    PyArray_DTypeMeta *object_DType = PyArray_DTypeFromTypeNum(NPY_OBJECT);

    for (int i = 0; i < ufunc->nargs; i++) {
        if (signature[i] == NULL) {
            Py_INCREF(object_DType);
            new_op_dtypes[i] = object_DType;
        }
    }
    Py_DECREF(object_DType);
    return 0;
}

/*
 * Special promoter for the logical ufuncs.  The logical ufuncs can always
 * use the ??->? and still get the correct output (as long as the output
 * is not supposed to be `object`).
 */
static int
logical_ufunc_promoter(PyUFuncObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    /*
     * If we find any object DType at all, we currently force to object.
     * However, if the output is specified and not object, there is no point,
     * it should be just as well to cast the input rather than doing the
     * unsafe out cast.
     */
    int force_object = 0;

    if (signature[0] == NULL && signature[1] == NULL
            && signature[2] != NULL && signature[2]->type_num != NPY_BOOL) {
        /* bail out, this is _only_ to give future/deprecation warning! */
        return -1;
    }

    for (int i = 0; i < 3; i++) {
        PyArray_DTypeMeta *item;
        if (signature[i] != NULL) {
            item = signature[i];
            Py_INCREF(item);
            if (item->type_num == NPY_OBJECT) {
                force_object = 1;
            }
        }
        else {
            /* Always override to boolean */
            item = PyArray_DTypeFromTypeNum(NPY_BOOL);
            if (op_dtypes[i] != NULL && op_dtypes[i]->type_num == NPY_OBJECT) {
                force_object = 1;
            }
        }
        new_op_dtypes[i] = item;
    }

    if (!force_object || (op_dtypes[2] != NULL
                          && op_dtypes[2]->type_num != NPY_OBJECT)) {
        return 0;
    }
    /*
     * Actually, we have to use the OBJECT loop after all, set all we can
     * to object (that might not work out, but try).
     *
     * NOTE: Change this to check for `op_dtypes[0] == NULL` to STOP
     *       returning `object` for `np.logical_and.reduce(obj_arr)`
     *       which will also affect `np.all` and `np.any`!
     */
    for (int i = 0; i < 3; i++) {
        if (signature[i] != NULL) {
            continue;
        }
        Py_SETREF(new_op_dtypes[i], PyArray_DTypeFromTypeNum(NPY_OBJECT));
    }
    return 0;
}

static int
hpy_logical_ufunc_promoter(HPyContext *ctx, HPy /* PyUFuncObject * */ NPY_UNUSED(ufunc),
        HPy /* PyArray_DTypeMeta * */ op_dtypes[], 
        HPy /* PyArray_DTypeMeta * */ signature[],
        HPy /* PyArray_DTypeMeta * */ new_op_dtypes[])
{
    /*
     * If we find any object DType at all, we currently force to object.
     * However, if the output is specified and not object, there is no point,
     * it should be just as well to cast the input rather than doing the
     * unsafe out cast.
     */
    int force_object = 0;

    if (HPy_IsNull(signature[0]) && HPy_IsNull(signature[1])
            && !HPy_IsNull(signature[2]) 
            && PyArray_DTypeMeta_AsStruct(ctx, signature[2])->type_num != NPY_BOOL) {
        /* bail out, this is _only_ to give future/deprecation warning! */
        return -1;
    }

    for (int i = 0; i < 3; i++) {
        HPy item; // PyArray_DTypeMeta *
        if (!HPy_IsNull(signature[i])) {
            item = HPy_Dup(ctx, signature[i]);
            // Py_INCREF(item);
            if (PyArray_DTypeMeta_AsStruct(ctx, item)->type_num == NPY_OBJECT) {
                force_object = 1;
            }
        }
        else {
            /* Always override to boolean */
            item = HPyArray_DTypeFromTypeNum(ctx, NPY_BOOL);
            if (!HPy_IsNull(op_dtypes[i]) 
                    && PyArray_DTypeMeta_AsStruct(ctx, op_dtypes[i])->type_num == NPY_OBJECT) {
                force_object = 1;
            }
        }
        new_op_dtypes[i] = item;
    }

    if (!force_object || (!HPy_IsNull(op_dtypes[2])
                          && PyArray_DTypeMeta_AsStruct(ctx, op_dtypes[2])->type_num != NPY_OBJECT)) {
        return 0;
    }
    /*
     * Actually, we have to use the OBJECT loop after all, set all we can
     * to object (that might not work out, but try).
     *
     * NOTE: Change this to check for `op_dtypes[0] == NULL` to STOP
     *       returning `object` for `np.logical_and.reduce(obj_arr)`
     *       which will also affect `np.all` and `np.any`!
     */
    for (int i = 0; i < 3; i++) {
        if (!HPy_IsNull(signature[i])) {
            continue;
        }
        HPy_SETREF(ctx, new_op_dtypes[i], HPyArray_DTypeFromTypeNum(ctx, NPY_OBJECT));
    }
    return 0;
}

NPY_NO_EXPORT int
install_logical_ufunc_promoter(HPyContext *ctx, HPy ufunc)
{
    HPy ufunc_type = HPy_Type(ctx, ufunc);
    if (!HPyGlobal_Is(ctx, ufunc_type, HPyUFunc_Type)) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError,
                "internal numpy array, logical ufunc was not a ufunc?!");
        return -1;
    }
    HPy arraydescr_type = HPyGlobal_Load(ctx, HPyArrayDescr_Type);
    HPy dtype_tuple = HPyTuple_Pack(ctx, 3, arraydescr_type, arraydescr_type, arraydescr_type);
    HPy_Close(ctx, arraydescr_type);
    if (HPy_IsNull(dtype_tuple)) {
        return -1;
    }
    HPy promoter = HPyCapsule_New(ctx, &hpy_logical_ufunc_promoter,
            "numpy._ufunc_promoter", NULL);
    if (HPy_IsNull(promoter)) {
        HPy_Close(ctx, dtype_tuple);
        return -1;
    }

    HPy info = HPyTuple_Pack(ctx, 2, dtype_tuple, promoter);
    HPy_Close(ctx, dtype_tuple);
    HPy_Close(ctx, promoter);
    if (HPy_IsNull(info)) {
        return -1;
    }

    return HPyUFunc_AddLoop(ctx, ufunc, info, 0);
}

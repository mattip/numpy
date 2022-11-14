#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "common.h"
#include "mapping.h"

#include "sequence.h"
#include "calculation.h"

/*************************************************************************
 ****************   Implement Sequence Protocol **************************
 *************************************************************************/

/* Some of this is repeated in the array_as_mapping protocol.  But
   we fill it in here so that PySequence_XXXX calls work as expected
*/

HPyDef_SLOT(array_contains_slot, array_contains, HPy_sq_contains);
NPY_NO_EXPORT int
array_contains(HPyContext *ctx, HPy /* PyArrayObject * */ self, HPy el)
{
    /* equivalent to (self == el).any() */

    int ret;
    HPy res, any;
    HPy cmp = HPy_RichCompare(ctx, self, el, HPy_EQ);
    res = HPyArray_EnsureAnyArray(ctx, cmp);
    if (HPy_IsNull(res)) {
        return -1;
    }

    any = HPyArray_Any(ctx, res, NPY_MAXDIMS, HPy_NULL);
    HPy_Close(ctx, res);
    if (HPy_IsNull(any)) {
        return -1;
    }

    ret = HPy_IsTrue(ctx, any);
    HPy_Close(ctx, any);
    return ret;
}

HPyDef_SLOT(array_concat_slot, array_concat, HPy_sq_concat);
NPY_NO_EXPORT HPy
array_concat(HPyContext *ctx, HPy self, HPy other)
{
    /*
     * Throw a type error, when trying to concat NDArrays
     * NOTE: This error is not Thrown when running with PyPy
     */
    HPyErr_SetString(ctx, ctx->h_TypeError,
            "Concatenation operation is not implemented for NumPy arrays, "
            "use np.concatenate() instead. Please do not rely on this error; "
            "it may not be given on all Python implementations.");
    return HPy_NULL;
}



/****************** End of Sequence Protocol ****************************/


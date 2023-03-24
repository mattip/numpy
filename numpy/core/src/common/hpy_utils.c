#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "hpy_utils.h"

/*
    XXX: not sure if we can support this
*/
NPY_NO_EXPORT HPy
HPyLong_FromVoidPtr(HPyContext *ctx, void *p)
{
#if SIZEOF_VOID_P <= SIZEOF_LONG
    return HPyLong_FromUnsignedLong(ctx, (unsigned long)(uintptr_t)p);
#else

#if SIZEOF_LONG_LONG < SIZEOF_VOID_P
#   error "HPyLong_FromVoidPtr: sizeof(long long) < sizeof(void*)"
#endif
    return HPyLong_FromUnsignedLongLong(ctx, (unsigned long long)(uintptr_t)p);
#endif /* SIZEOF_VOID_P <= SIZEOF_LONG */

}

NPY_NO_EXPORT int
HPyFloat_CheckExact(HPyContext *ctx, HPy obj) {
    HPy obj_type = HPy_Type(ctx, obj);
    int res = HPy_Is(ctx, obj_type, ctx->h_FloatType);
    HPy_Close(ctx, obj_type);
    return res;
}

NPY_NO_EXPORT int
HPyComplex_CheckExact(HPyContext *ctx, HPy obj) {
    HPy obj_type = HPy_Type(ctx, obj);
    int res = HPy_Is(ctx, obj_type, ctx->h_ComplexType);
    HPy_Close(ctx, obj_type);
    return res;
}

NPY_NO_EXPORT HPy_ssize_t
HPyNumber_AsSsize_t(HPyContext *ctx, HPy item, HPy err)
{
    HPy_ssize_t result;
    HPy runerr;
    HPy value = HPy_Index(ctx, item);
    if (HPy_IsNull(value))
        return -1;

    result = HPyLong_AsSsize_t(ctx, value);
    if (result != -1 || !HPyErr_Occurred(ctx))
        goto finish;

    if (!HPyErr_ExceptionMatches(ctx, ctx->h_OverflowError))
        goto finish;

    HPyErr_Clear(ctx);
    if (HPy_IsNull(err)) {
        assert(HPyLong_Check(ctx, value));
        if (HPyLong_AsLong(ctx, value) < 0)
            result = PY_SSIZE_T_MIN;
        else
            result = PY_SSIZE_T_MAX;
    }
    else {
        /* Otherwise replace the error with caller's error object. */
        HPyErr_SetString(ctx, err,
                     "cannot fit into an index-sized integer");
    }

 finish:
    HPy_Close(ctx, value);
    return result;
}


// HPy PORT: this is not the actual implementation of PyUnicode_Concat
// TODO: PyUnicode_Concat should be included in HPy
NPY_NO_EXPORT HPy
HPyUnicode_Concat_t(HPyContext *ctx, HPy s1, HPy s2)
{
    if (!HPyUnicode_Check(ctx, s1) || !HPyUnicode_Check(ctx, s2)) {
        return HPy_NULL;
    }
    return HPy_Add(ctx, s1, s2);
}

NPY_NO_EXPORT int
HPyGlobal_Is(HPyContext *ctx, HPy obj, HPyGlobal expected)
{
    HPy h_expected = HPyGlobal_Load(ctx, expected);
    int is_dtype = HPy_Is(ctx, obj, h_expected);
    HPy_Close(ctx, h_expected);
    return is_dtype;
}

NPY_NO_EXPORT HPy
HPyFastcallToDict(HPyContext *ctx, const HPy *args, size_t nargs, HPy kwnames)
{
    HPy kw, kwname;
    HPy_ssize_t nkw, i;

    if (HPy_IsNull(kwnames))
        return HPy_NULL;

    kw = HPyDict_New(ctx);
    if (HPy_IsNull(kw))
        return HPy_NULL;

    nkw = HPy_Length(ctx, kwnames);
    if (nkw < 0)
        return HPy_NULL;

    for (i=0; i < nkw; i++)
    {
        kwname = HPy_GetItem_i(ctx, kwnames, i);
        if (HPy_IsNull(kwname))
        {
            HPy_Close(ctx, kw);
            return HPy_NULL;
        }
        if (HPy_SetItem(ctx, kw, kwname, args[nargs + i]) < 0)
        {
            HPy_Close(ctx, kwname);
            HPy_Close(ctx, kw);
            return HPy_NULL;
        }
        HPy_Close(ctx, kwname);
    }
    return kw;
}

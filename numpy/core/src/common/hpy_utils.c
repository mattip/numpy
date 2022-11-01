#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "hpy_utils.h"

#define NBUF 256

NPY_NO_EXPORT void
HPyErr_Format_p(HPyContext *ctx, HPy h_type, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    char buf[NBUF];
    char *buf_extended = NULL;

    int nprinted = vsnprintf(buf, NBUF, fmt, ap);
    if (nprinted >= NBUF)
    {
        buf_extended = (char *) calloc(nprinted, sizeof(char));
        vsnprintf(buf, nprinted*sizeof(char), fmt, ap);
    }
    va_end(ap);

    if (buf_extended)
    {
        HPyErr_SetString(ctx, h_type, buf_extended);
        free(buf_extended);
    }
    else
    {
        HPyErr_SetString(ctx, h_type, buf);

    }
}

NPY_NO_EXPORT HPy
HPyUnicode_FromFormat_p(HPyContext *ctx, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    char buf[NBUF];
    char *buf_extended = NULL;
    HPy ret = HPy_NULL;
    
    int nprinted = vsnprintf(buf, NBUF, fmt, ap);
    if (nprinted >= NBUF)
    {
        buf_extended = (char *) calloc(nprinted, sizeof(char));
        vsnprintf(buf, nprinted*sizeof(char), fmt, ap);
    }
    va_end(ap);

    if (buf_extended)
    {
        ret = HPyUnicode_FromString(ctx, buf_extended);
        free(buf_extended);
    }
    else
    {
        ret = HPyUnicode_FromString(ctx, buf);
    }
    return ret;
}

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


// HPy PORT: this is not the actual implementation of PyUnicode_Concat
// TODO: PyUnicode_Concat should be included in HPy
NPY_NO_EXPORT HPy
HPyUnicode_Concat_t(HPyContext *ctx, HPy s1, HPy s2)
{
    HPy ret = HPy_NULL;
    char *ss1, *ss2;
    HPy_ssize_t ls1, ls2;
    ss1 = HPyUnicode_AsUTF8AndSize(ctx, s1, &ls1);
    if (ls1 == 0) {
        return HPy_Dup(ctx, s2);
    }
    ss2 = HPyUnicode_AsUTF8AndSize(ctx, s2, &ls2);
    return HPyUnicode_FromString(ctx, strcat(ss1, ss2));
}

NPY_NO_EXPORT int
HPyGlobal_Is(HPyContext *ctx, HPy obj, HPyGlobal expected)
{
    HPy h_expected = HPyGlobal_Load(ctx, expected);
    int is_dtype = HPy_Is(ctx, obj, h_expected);
    HPy_Close(ctx, h_expected);
    return is_dtype;
}

NPY_NO_EXPORT int
HPyGlobal_TypeCheck(HPyContext *ctx, HPy obj, HPyGlobal type)
{
    HPy h_type = HPyGlobal_Load(ctx, type);
    int res = HPy_TypeCheck(ctx, obj, h_type);
    HPy_Close(ctx, h_type);
    return res;
}

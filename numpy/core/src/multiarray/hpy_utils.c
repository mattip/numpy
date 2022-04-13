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

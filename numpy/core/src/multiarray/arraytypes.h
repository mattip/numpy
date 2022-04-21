#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYTYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYTYPES_H_

#include "common.h"

NPY_NO_EXPORT int
set_typeinfo(HPyContext *ctx, HPy dict);

/* needed for blasfuncs */
NPY_NO_EXPORT void
FLOAT_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CFLOAT_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
DOUBLE_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CDOUBLE_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);


/* for _pyarray_correlate */
NPY_NO_EXPORT int
small_correlate(const char * d_, npy_intp dstride,
                npy_intp nd, enum NPY_TYPES dtype,
                const char * k_, npy_intp kstride,
                npy_intp nk, enum NPY_TYPES ktype,
                char * out_, npy_intp ostride);

void init_hpy_global_state(HPyContext *ctx);

NPY_NO_EXPORT HPy
HPyArray_DescrFromType(HPyContext *ctx, int type);

NPY_NO_EXPORT void
init_static_descrs_type_objs(HPyContext *ctx);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYTYPES_H_ */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"


typedef struct {
    int key_len;  /* number of identities used */
    /* Buckets stores: val1, key1[0], key1[1], ..., val2, key2[0], ... */
    HPy *buckets;
    npy_intp size;  /* current size */
    npy_intp nelem;  /* number of elements */
} PyArrayIdentityHash;


NPY_NO_EXPORT int
PyArrayIdentityHash_SetItem(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *value, int replace);

NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyArrayIdentityHash const *tb, PyObject *const *key);

NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len);

NPY_NO_EXPORT PyArrayIdentityHash *
HPyArrayIdentityHash_New(HPyContext *ctx, int key_len);

NPY_NO_EXPORT void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb);

NPY_NO_EXPORT int
HPyArrayIdentityHash_SetItem(HPyContext *ctx, PyArrayIdentityHash *tb,
        HPy const *key, HPy value, int replace);

NPY_NO_EXPORT HPy
HPyArrayIdentityHash_GetItem(HPyContext *ctx, PyArrayIdentityHash const *tb, HPy const *key);

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_NPY_HASHTABLE_H_ */

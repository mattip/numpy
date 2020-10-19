#ifndef _NPY_ARRAY_ALLOC_H_
#define _NPY_ARRAY_ALLOC_H_
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

#define NPY_TRACE_DOMAIN 389047

NPY_NO_EXPORT PyObject *
_set_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *enabled_obj);

NPY_NO_EXPORT void *
PyDataMem_UserNEW(npy_uintp sz, PyDataMem_AllocFunc *alloc);

NPY_NO_EXPORT void *
PyDataMem_UserNEW_ZEROED(size_t nmemb, size_t size, PyDataMem_ZeroedAllocFunc *zalloc);

NPY_NO_EXPORT void
PyDataMem_UserFREE(void * p, npy_uintp sd, PyDataMem_FreeFunc *func);

NPY_NO_EXPORT void *
PyDataMem_UserRENEW(void *ptr, size_t size, PyDataMem_ReallocFunc *func);

NPY_NO_EXPORT void *
npy_alloc_cache_dim(npy_uintp sz);

NPY_NO_EXPORT void
npy_free_cache_dim(void * p, npy_uintp sd);

static NPY_INLINE void
npy_free_cache_dim_obj(PyArray_Dims dims)
{
    npy_free_cache_dim(dims.ptr, dims.len);
}

static NPY_INLINE void
npy_free_cache_dim_array(PyArrayObject * arr)
{
    npy_free_cache_dim(PyArray_DIMS(arr), PyArray_NDIM(arr));
}

extern PyDataMem_Handler *current_allocator;
extern PyDataMem_Handler default_allocator;

#define PyArray_HANDLER(arr) ((PyArrayObject_fields*)(arr))->mem_handler

#endif

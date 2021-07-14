#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#include <pymem.h>
/* public api in 3.7 */
#if PY_VERSION_HEX < 0x03070000
#define PyTraceMalloc_Track _PyTraceMalloc_Track
#define PyTraceMalloc_Untrack _PyTraceMalloc_Untrack
#endif

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>
#include "numpy/arrayobject.h"
#include <numpy/npy_common.h>
#include "npy_config.h"
#include "alloc.h"


#include <assert.h>

#ifdef NPY_OS_LINUX
#include <sys/mman.h>
#ifndef MADV_HUGEPAGE
/*
 * Use code 14 (MADV_HUGEPAGE) if it isn't defined. This gives a chance of
 * enabling huge pages even if built with linux kernel < 2.6.38
 */
#define MADV_HUGEPAGE 14
#endif
#endif

#define NBUCKETS 1024 /* number of buckets for data*/
#define NBUCKETS_DIM 16 /* number of buckets for dimensions/strides */
#define NCACHE 7 /* number of cache entries per bucket */
/* this structure fits neatly into a cacheline */
typedef struct {
    npy_uintp available; /* number of cached pointers */
    void * ptrs[NCACHE];
} cache_bucket;
static cache_bucket datacache[NBUCKETS];
static cache_bucket dimcache[NBUCKETS_DIM];

static int _madvise_hugepage = 1;


/*
 * This function enables or disables the use of `MADV_HUGEPAGE` on Linux
 * by modifying the global static `_madvise_hugepage`.
 * It returns the previous value of `_madvise_hugepage`.
 *
 * It is exposed to Python as `np.core.multiarray._set_madvise_hugepage`.
 */
NPY_NO_EXPORT PyObject *
_set_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *enabled_obj)
{
    int was_enabled = _madvise_hugepage;
    int enabled = PyObject_IsTrue(enabled_obj);
    if (enabled < 0) {
        return NULL;
    }
    _madvise_hugepage = enabled;
    if (was_enabled) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}


/* as the cache is managed in global variables verify the GIL is held */

/*
 * very simplistic small memory block cache to avoid more expensive libc
 * allocations
 * base function for data cache with 1 byte buckets and dimension cache with
 * sizeof(npy_intp) byte buckets
 */
static NPY_INLINE void *
_npy_alloc_cache(npy_uintp nelem, npy_uintp esz, npy_uint msz,
                 cache_bucket * cache, void * (*alloc)(size_t))
{
    void * p;
    assert((esz == 1 && cache == datacache) ||
           (esz == sizeof(npy_intp) && cache == dimcache));
    assert(PyGILState_Check());
    if (nelem < msz) {
        if (cache[nelem].available > 0) {
            return cache[nelem].ptrs[--(cache[nelem].available)];
        }
    }
    p = alloc(nelem * esz);
    if (p) {
#ifdef _PyPyGC_AddMemoryPressure
        _PyPyPyGC_AddMemoryPressure(nelem * esz);
#endif
#ifdef NPY_OS_LINUX
        /* allow kernel allocating huge pages for large arrays */
        if (NPY_UNLIKELY(nelem * esz >= ((1u<<22u))) && _madvise_hugepage) {
            npy_uintp offset = 4096u - (npy_uintp)p % (4096u);
            npy_uintp length = nelem * esz - offset;
            /**
             * Intentionally not checking for errors that may be returned by
             * older kernel versions; optimistically tries enabling huge pages.
             */
            madvise((void*)((npy_uintp)p + offset), length, MADV_HUGEPAGE);
        }
#endif
    }
    return p;
}

/*
 * return pointer p to cache, nelem is number of elements of the cache bucket
 * size (1 or sizeof(npy_intp)) of the block pointed too
 */
static NPY_INLINE void
_npy_free_cache(void * p, npy_uintp nelem, npy_uint msz,
                cache_bucket * cache, void (*dealloc)(void *))
{
    assert(PyGILState_Check());
    if (p != NULL && nelem < msz) {
        if (cache[nelem].available < NCACHE) {
            cache[nelem].ptrs[cache[nelem].available++] = p;
            return;
        }
    }
    dealloc(p);
}


/*
 * array data cache, sz is number of bytes to allocate
 */
NPY_NO_EXPORT void *
npy_alloc_cache(npy_uintp sz)
{
    return _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &PyDataMem_NEW);
}

/* zero initialized data, sz is number of bytes to allocate */
NPY_NO_EXPORT void *
npy_alloc_cache_zero(size_t nmemb, size_t size)
{
    void * p;
    size_t sz = nmemb * size;
    NPY_BEGIN_THREADS_DEF;
    if (sz < NBUCKETS) {
        p = _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &PyDataMem_NEW);
        if (p) {
            memset(p, 0, sz);
        }
        return p;
    }
    NPY_BEGIN_THREADS;
    p = PyDataMem_NEW_ZEROED(nmemb, size);
    NPY_END_THREADS;
    return p;
}

NPY_NO_EXPORT void
npy_free_cache(void * p, npy_uintp sz)
{
    _npy_free_cache(p, sz, NBUCKETS, datacache, &PyDataMem_FREE);
}

/*
 * dimension/stride cache, uses a different allocator and is always a multiple
 * of npy_intp
 */
NPY_NO_EXPORT void *
npy_alloc_cache_dim(npy_uintp sz)
{
    /*
     * make sure any temporary allocation can be used for array metadata which
     * uses one memory block for both dimensions and strides
     */
    if (sz < 2) {
        sz = 2;
    }
    return _npy_alloc_cache(sz, sizeof(npy_intp), NBUCKETS_DIM, dimcache,
                            &PyArray_malloc);
}

NPY_NO_EXPORT void
npy_free_cache_dim(void * p, npy_uintp sz)
{
    /* see npy_alloc_cache_dim */
    if (sz < 2) {
        sz = 2;
    }
    _npy_free_cache(p, sz, NBUCKETS_DIM, dimcache,
                    &PyArray_free);
}


/* malloc/free/realloc hook */
NPY_NO_EXPORT PyDataMem_EventHookFunc *_PyDataMem_eventhook = NULL;
NPY_NO_EXPORT void *_PyDataMem_eventhook_user_data = NULL;

/*NUMPY_API
 * Sets the allocation event hook for numpy array data.
 * Takes a PyDataMem_EventHookFunc *, which has the signature:
 *        void hook(void *old, void *new, size_t size, void *user_data).
 *   Also takes a void *user_data, and void **old_data.
 *
 * Returns a pointer to the previous hook or NULL.  If old_data is
 * non-NULL, the previous user_data pointer will be copied to it.
 *
 * If not NULL, hook will be called at the end of each PyDataMem_NEW/FREE/RENEW:
 *   result = PyDataMem_NEW(size)        -> (*hook)(NULL, result, size, user_data)
 *   PyDataMem_FREE(ptr)                 -> (*hook)(ptr, NULL, 0, user_data)
 *   result = PyDataMem_RENEW(ptr, size) -> (*hook)(ptr, result, size, user_data)
 *
 * When the hook is called, the GIL will be held by the calling
 * thread.  The hook should be written to be reentrant, if it performs
 * operations that might cause new allocation events (such as the
 * creation/destruction numpy objects, or creating/destroying Python
 * objects which might cause a gc)
 */
NPY_NO_EXPORT PyDataMem_EventHookFunc *
PyDataMem_SetEventHook(PyDataMem_EventHookFunc *newhook,
                       void *user_data, void **old_data)
{
    PyDataMem_EventHookFunc *temp;
    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API
    temp = _PyDataMem_eventhook;
    _PyDataMem_eventhook = newhook;
    if (old_data != NULL) {
        *old_data = _PyDataMem_eventhook_user_data;
    }
    _PyDataMem_eventhook_user_data = user_data;
    NPY_DISABLE_C_API
    return temp;
}

/*NUMPY_API
 * Allocates memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW(size_t size)
{
    void *result;

    assert(size != 0);
    result = malloc(size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(NULL, result, size,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    return result;
}

/*NUMPY_API
 * Allocates zeroed memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW_ZEROED(size_t nmemb, size_t size)
{
    void *result;

    result = calloc(nmemb, size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(NULL, result, nmemb * size,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, nmemb * size);
    return result;
}

/*NUMPY_API
 * Free memory for array data.
 */
NPY_NO_EXPORT void
PyDataMem_FREE(void *ptr)
{
    PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    free(ptr);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(ptr, NULL, 0,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
}

/*NUMPY_API
 * Reallocate/resize memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_RENEW(void *ptr, size_t size)
{
    void *result;

    assert(size != 0);
    result = realloc(ptr, size);
    if (result != ptr) {
        PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(ptr, result, size,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    return result;
}

// The default data mem allocator malloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserNEW
// since itself does not handle eventhook and tracemalloc logic.
static NPY_INLINE void *
default_malloc(void *NPY_UNUSED(ctx), size_t size)
{
    return _npy_alloc_cache(size, 1, NBUCKETS, datacache, &malloc);
}

// The default data mem allocator calloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserNEW_ZEROED
// since itself does not handle eventhook and tracemalloc logic.
static NPY_INLINE void *
default_calloc(void *NPY_UNUSED(ctx), size_t nelem, size_t elsize)
{
    void * p;
    size_t sz = nelem * elsize;
    NPY_BEGIN_THREADS_DEF;
    if (sz < NBUCKETS) {
        p = _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &malloc);
        if (p) {
            memset(p, 0, sz);
        }
        return p;
    }
    NPY_BEGIN_THREADS;
    p = calloc(nelem, elsize);
    NPY_END_THREADS;
    return p;
}

// The default data mem allocator realloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserRENEW
// since itself does not handle eventhook and tracemalloc logic.
static NPY_INLINE void *
default_realloc(void *NPY_UNUSED(ctx), void *ptr, size_t new_size)
{
    return realloc(ptr, new_size);
}

// The default data mem allocator free routine does not make use of a ctx.
// It should be called only through PyDataMem_UserFREE
// since itself does not handle eventhook and tracemalloc logic.
static NPY_INLINE void
default_free(void *NPY_UNUSED(ctx), void *ptr, size_t size)
{
    _npy_free_cache(ptr, size, NBUCKETS, datacache, &free);
}

/* Memory handler global default */
PyDataMem_Handler default_handler = {
    "default_allocator",
    {
        NULL,            /* ctx */
        default_malloc,  /* malloc */
        default_calloc,  /* calloc */
        default_realloc, /* realloc */
        default_free     /* free */
    }
};

PyDataMem_Handler *current_handler = &default_handler;

int uo_index=0;   /* user_override index */

/* Wrappers for the default or any user-assigned PyDataMem_Handler */

NPY_NO_EXPORT void *
PyDataMem_UserNEW(size_t size, PyDataMemAllocator allocator)
{
    void *result;

    assert(size != 0);
    result = allocator.malloc(allocator.ctx, size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(NULL, result, size,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    return result;
}

NPY_NO_EXPORT void *
PyDataMem_UserNEW_ZEROED(size_t nmemb, size_t size, PyDataMemAllocator allocator)
{
    void *result;
    result = allocator.calloc(allocator.ctx, nmemb, size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(NULL, result, nmemb * size,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, nmemb * size);
    return result;
}

NPY_NO_EXPORT void
PyDataMem_UserFREE(void *ptr, size_t size, PyDataMemAllocator allocator)
{
    PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    allocator.free(allocator.ctx, ptr, size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(ptr, NULL, 0,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
}

NPY_NO_EXPORT void *
PyDataMem_UserRENEW(void *ptr, size_t size, PyDataMemAllocator allocator)
{
    void *result;

    assert(size != 0);
    result = allocator.realloc(allocator.ctx, ptr, size);
    if (result != ptr) {
        PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(ptr, result, size,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    return result;
}

/*NUMPY_API
 * Sets a new allocation policy. If the input value is NULL, will reset
 * the policy to the default. Returns the previous policy, NULL if the
 * previous policy was the default. We wrap the user-provided functions
 * so they will still call the python and numpy memory management callback
 * hooks.
 */
NPY_NO_EXPORT const PyDataMem_Handler *
PyDataMem_SetHandler(PyDataMem_Handler *handler)
{
    const PyDataMem_Handler *old = current_handler;
    if (handler) {
        current_handler = handler;
    }
    else {
        current_handler = &default_handler;
    }
    return old;
}

/*NUMPY_API
 * Return the const char name of the PyDataMem_Handler used by the
 * PyArrayObject or its base. If neither the PyArrayObject owns its own data
 * nor its base is a PyArrayObject which owns its own data return an empty string.
 * If NULL, return the name of the current global policy that
 * will be used to allocate data for the next PyArrayObject.
 */
NPY_NO_EXPORT const char *
PyDataMem_GetHandlerName(PyArrayObject *obj)
{
    if (obj == NULL) {
        return current_handler->name;
    }
    PyDataMem_Handler *handler;
    handler = PyArray_HANDLER(obj);
    if (handler != NULL) {
        return handler->name;
    }
    PyObject *base = PyArray_BASE(obj);
    if (base != NULL && PyArray_Check(base)) {
        handler = PyArray_HANDLER((PyArrayObject *) base);
        if (handler != NULL) {
             return handler->name;
        }
    }
    return "";
}

NPY_NO_EXPORT PyObject *
get_handler_name(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *arr=NULL;
    if (!PyArg_ParseTuple(args, "|O:get_handler_name", &arr)) {
        return NULL;
    }
    if (arr != NULL && !PyArray_Check(arr)) {
         PyErr_SetString(PyExc_ValueError, "if supplied, argument must be an ndarray");
         return NULL;
    }
    const char * name = PyDataMem_GetHandlerName((PyArrayObject *)arr);
    if (name == NULL) {
        return NULL;
    }
    else if (strlen(name) == 0) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString(name);
}

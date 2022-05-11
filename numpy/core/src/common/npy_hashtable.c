/*
 * This functionality is designed specifically for the ufunc machinery to
 * dispatch based on multiple DTypes.  Since this is designed to be used
 * as purely a cache, it currently does no reference counting.
 * Even though this is a cache, there is currently no maximum size.  It may
 * make sense to limit the size, or count collisions:  If too many collisions
 * occur, we could grow the cache, otherwise, just replace an old item that
 * was presumably not used for a long time.
 *
 * If a different part of NumPy requires a custom hashtable, the code should
 * be reused with care since specializing it more for the ufunc dispatching
 * case is likely desired.
 */

#include "templ_common.h"
#include "npy_hashtable.h"
#include "hpy_utils.h"



#if SIZEOF_PY_UHASH_T > 4
#define _NpyHASH_XXPRIME_1 ((Py_uhash_t)11400714785074694791ULL)
#define _NpyHASH_XXPRIME_2 ((Py_uhash_t)14029467366897019727ULL)
#define _NpyHASH_XXPRIME_5 ((Py_uhash_t)2870177450012600261ULL)
#define _NpyHASH_XXROTATE(x) ((x << 31) | (x >> 33))  /* Rotate left 31 bits */
#else
#define _NpyHASH_XXPRIME_1 ((Py_uhash_t)2654435761UL)
#define _NpyHASH_XXPRIME_2 ((Py_uhash_t)2246822519UL)
#define _NpyHASH_XXPRIME_5 ((Py_uhash_t)374761393UL)
#define _NpyHASH_XXROTATE(x) ((x << 13) | (x >> 19))  /* Rotate left 13 bits */
#endif

/*
 * This hashing function is basically the Python tuple hash with the type
 * identity hash inlined. The tuple hash itself is a reduced version of xxHash.
 *
 * Users cannot control pointers, so we do not have to worry about DoS attacks?
 */
static NPY_INLINE HPy_hash_t
identity_list_hash(HPyContext *ctx, HPy const *v, int len)
{
    Py_uhash_t acc = _NpyHASH_XXPRIME_5;
    for (int i = 0; i < len; i++) {
        /*
         * Lane is the single item hash, which for us is the rotated pointer.
         * Identical to the python type hash (pointers end with 0s normally).
         */
        // TODO HPY LABS PORT: we don't really have a way to compute the identity hash in HPy
        size_t y = (size_t)(HPy_IsNull(v[i]) ? 0 : HPy_Hash(ctx, v[i]));
        Py_uhash_t lane = (y >> 4) | (y << (8 * SIZEOF_VOID_P - 4));
        acc += lane * _NpyHASH_XXPRIME_2;
        acc = _NpyHASH_XXROTATE(acc);
        acc *= _NpyHASH_XXPRIME_1;
    }
    return acc;
}
#undef _NpyHASH_XXPRIME_1
#undef _NpyHASH_XXPRIME_2
#undef _NpyHASH_XXPRIME_5
#undef _NpyHASH_XXROTATE


static NPY_INLINE HPyField *
find_item(HPyContext *ctx, HPy cache_owner, PyArrayIdentityHash const *tb, HPy const *key)
{
    HPy_hash_t hash = identity_list_hash(ctx, key, tb->key_len);
    npy_uintp perturb = (npy_uintp)hash;
    npy_intp bucket;
    npy_intp mask = tb->size - 1 ;
    HPyField *item;

    bucket = (npy_intp)hash & mask;
    while (1) {
        item = &(tb->buckets[bucket * (tb->key_len + 1)]);

        if (HPyField_IsNull(item[0])) {
            /* The item is not in the cache; return the empty bucket */
            return item;
        }
        int eq = 1;
        for (int i=0; eq && i < tb->key_len; i++) {
            HPy h_item = HPyField_IsNull(item[i+1]) ? HPy_NULL :
                    HPyField_Load(ctx, cache_owner, item[i+1]);
            if (!HPy_Is(ctx, h_item, key[i])) {
                eq = 0;
            }
            HPy_Close(ctx, h_item);
        }
        if (eq) {
            /* This is a match, so return the item/bucket */
            return item;
        }
        // if (memcmp(item+1, key, tb->key_len * sizeof(HPy)) == 0) {
        //     /* This is a match, so return the item/bucket */
        //     return item;
        // }
        /* Hash collision, perturb like Python (must happen rarely!) */
        perturb >>= 5;  /* Python uses the macro PERTURB_SHIFT == 5 */
        bucket = mask & (bucket * 5 + perturb + 1);
    }
}


NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len)
{
    return HPyArrayIdentityHash_New(npy_get_context(), key_len);
}

NPY_NO_EXPORT PyArrayIdentityHash *
HPyArrayIdentityHash_New(HPyContext *ctx, int key_len)
{
    // TODO HPY LABS PORT: PyMem_Malloc
    // PyArrayIdentityHash *res = PyMem_Malloc(sizeof(PyArrayIdentityHash));
    PyArrayIdentityHash *res = malloc(sizeof(PyArrayIdentityHash));
    if (res == NULL) {
        HPyErr_NoMemory(ctx);
        return NULL;
    }

    assert(key_len > 0);
    res->key_len = key_len;
    res->size = 4;  /* Start with a size of 4 */
    res->nelem = 0;

    // TODO HPY LABS PORT: PyMem_Calloc
    res->buckets = calloc(4 * (key_len + 1), sizeof(HPyField));
    if (res->buckets == NULL) {
        HPyErr_NoMemory(ctx);
        // TODO HPY LABS PORT: PyMem_Free
        // PyMem_Free(res);
        free(res);
        return NULL;
    }
    return res;
}


NPY_NO_EXPORT void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb)
{
    // TODO HPY LABS PORT: PyMem_Free
    // PyMem_Free(tb->buckets);
    // PyMem_Free(tb);
    free(tb->buckets);
    free(tb);
}

NPY_NO_EXPORT int
HPyArrayIdentityHash_Traverse(PyArrayIdentityHash *tb, HPyFunc_visitproc visit, void *arg)
{
    /* Buckets stores: val1, key1[0], key1[1], ..., val2, key2[0], ... */
    const npy_intp nitems = tb->size * (tb->key_len + 1);
    for (npy_intp i = 0; i < nitems; i++) {
        HPy_VISIT(&tb->buckets[i]);
    }
    return 0;
}


static int
_resize_if_necessary(HPyContext *ctx, HPy cache_owner, PyArrayIdentityHash *tb)
{
    npy_intp new_size, prev_size = tb->size;
    HPyField *old_table = tb->buckets;
    assert(prev_size > 0);

    if ((tb->nelem + 1) * 2 > prev_size) {
        /* Double in size */
        new_size = prev_size * 2;
    }
    else {
        new_size = prev_size;
        while ((tb->nelem + 8) * 2 < new_size / 2) {
            /*
             * Should possibly be improved.  However, we assume that we
             * almost never shrink.  Still if we do, do not shrink as much
             * as possible to avoid growing right away.
             */
            new_size /= 2;
        }
        assert(new_size >= 4);
    }
    if (new_size == prev_size) {
        return 0;
    }

    npy_intp alloc_size;
    if (npy_mul_with_overflow_intp(&alloc_size, new_size, tb->key_len + 1)) {
        return -1;
    }
    // TODO HPY LABS PORT: PyMem_Calloc
    // tb->buckets = PyMem_Calloc(alloc_size, sizeof(PyObject *));
    tb->buckets = calloc(alloc_size, sizeof(HPyField));
    if (tb->buckets == NULL) {
        tb->buckets = old_table;
        HPyErr_NoMemory(ctx);
        return -1;
    }

    HPy *tmp = calloc(tb->key_len + 1, sizeof(HPy));
    tb->size = new_size;
    for (npy_intp i = 0; i < prev_size; i++) {
        /*
         * We need to fully re-insert the element. So, load all objects (i.e.
         * all key objects and the value object) and store HPy_NULL to the old
         * fields.
         */
        HPyField *item = &old_table[i * (tb->key_len + 1)];
        if (!HPyField_IsNull(item[0])) {
            for (int j = 0; j < tb->key_len + 1; j++) {
                tmp[j] = HPyField_Load(ctx, cache_owner, item[j]);
                HPyField_Store(ctx, cache_owner, item+j, HPy_NULL);
            }
            tb->nelem -= 1;  /* Decrement, setitem will increment again */
            HPyArrayIdentityHash_SetItem(ctx, cache_owner, tb, tmp+1, tmp[0], 1);
            for (int j = 0; j < tb->key_len + 1; j++) {
                HPy_Close(ctx, tmp[j]);
            }
        }
    }
    free(tmp);
    // TODO HPY LABS PORT: PyMem_Calloc
    // PyMem_Free(old_table);
    free(old_table);
    return 0;
}


/**
 * Add an item to the identity cache.  The storage location must not change
 * unless the cache is cleared.
 *
 * @param tb The mapping.
 * @param key The key, must be a C-array of pointers of the length
 *        corresponding to the mapping.
 * @param value Normally a Python object, no reference counting is done.
 *        use NULL to clear an item.  If the item does not exist, no
 *        action is performed for NULL.
 * @param replace If 1, allow replacements.
 * @returns 0 on success, -1 with a MemoryError or RuntimeError (if an item
 *        is added which is already in the cache).  The caller should avoid
 *        the RuntimeError.
 */
NPY_NO_EXPORT int
PyArrayIdentityHash_SetItem(PyArrayIdentityHash *tb, PyObject *cache_owner,
        PyObject *const *key, PyObject *value, int replace)
{
    HPyContext *ctx = npy_get_context();
    HPy h_cache_owner = HPy_FromPyObject(ctx, cache_owner);
    HPy const *h_key = (HPy const *)HPy_FromPyObjectArray(ctx, (PyObject **)key, tb->key_len);
    HPy h_value = HPy_FromPyObject(ctx, value);
    int res = HPyArrayIdentityHash_SetItem(ctx, h_cache_owner, tb, h_key, h_value, replace);
    HPy_Close(ctx, h_value);
    HPy_CloseAndFreeArray(ctx, (HPy *)h_key, tb->key_len);
    HPy_Close(ctx, h_cache_owner);
    return res;
}


NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyObject *cache_owner, PyArrayIdentityHash const *tb, PyObject *const *key)
{
    HPyContext *ctx = npy_get_context();
    HPy h_cache_owner = HPy_FromPyObject(ctx, cache_owner);
    HPy const *h_key = (HPy const *)HPy_FromPyObjectArray(ctx, (PyObject **)key, tb->key_len);
    HPy h_res = HPyArrayIdentityHash_GetItem(ctx, h_cache_owner, tb, h_key);
    PyObject *res = HPy_AsPyObject(ctx, h_res);
    HPy_Close(ctx, h_res);
    HPy_CloseAndFreeArray(ctx, (HPy *)h_key, tb->key_len);
    HPy_Close(ctx, h_cache_owner);
    return res;
}

NPY_NO_EXPORT int
HPyArrayIdentityHash_SetItem(HPyContext *ctx, HPy cache_owner, PyArrayIdentityHash *tb,
        HPy const *key, HPy value, int replace)
{
    if (!HPy_IsNull(value) && _resize_if_necessary(ctx, cache_owner, tb) < 0) {
        /* Shrink, only if a new value is added. */
        return -1;
    }

    HPyField *tb_item = find_item(ctx, cache_owner, tb, key);
    if (!HPy_IsNull(value)) {
        if (!HPyField_IsNull(tb_item[0]) && !replace) {
            HPyErr_SetString(ctx, ctx->h_RuntimeError,
                    "Identity cache already includes the item.");
            return -1;
        }
        HPyField_Store(ctx, cache_owner, tb_item, value);
        for (int i=0; i < tb->key_len; i++) {
            HPyField_Store(ctx, cache_owner, tb_item+1+i, key[i]);
        }
        tb->nelem += 1;
    }
    else {
        /* Clear the bucket -- just the value should be enough though. */
        for (int i=0; i < tb->key_len + 1; i++) {
            HPyField_Store(ctx, cache_owner, tb_item+i, HPy_NULL);
        }
    }

    return 0;
}

NPY_NO_EXPORT HPy
HPyArrayIdentityHash_GetItem(HPyContext *ctx, HPy cache_owner, PyArrayIdentityHash const *tb, HPy const *key)
{
    HPyField f_item = find_item(ctx, cache_owner, tb, key)[0];
    if (!HPyField_IsNull(f_item)) {
        return HPyField_Load(ctx, cache_owner, f_item);
    }
    return HPy_NULL;
}


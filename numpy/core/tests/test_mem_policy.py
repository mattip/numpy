import pathlib
import pytest
import tempfile
import numpy as np
from numpy.testing import extbuild

@pytest.fixture
def get_module(tmp_path):
    """ Add a memory policy that returns a false pointer 64 bytes into the
    actual allocation, and fill the prefix with some text. Then check at each
    memory manipulation that the prefix exists, to make sure all alloc/realloc/
    free/calloc go via the functions here.
    """
    functions = [(
        "test_prefix", "METH_O",
        """
            if (!PyArray_Check(args)) {
                PyErr_SetString(PyExc_ValueError,
                        "must be called with a numpy scalar or ndarray");
            }
            return PyUnicode_FromString(PyDataMem_GetHandlerName((PyArrayObject*)args));
        """
        ),
        ("set_new_policy", "METH_NOARGS",
        """
            const PyDataMem_Handler *old = PyDataMem_SetHandler(&new_handler);
            return PyUnicode_FromString(old->name);
        """),
        ("set_old_policy", "METH_NOARGS",
        """
            const PyDataMem_Handler *old = PyDataMem_SetHandler(NULL);
            return PyUnicode_FromString(old->name);
        """),
        ]
    prologue='''
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <numpy/arrayobject.h>
        NPY_NO_EXPORT void *
        shift_alloc(size_t sz) {
            char *real = (char *)malloc(sz + 64);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated %ld", sz);
            return (void *)(real + 64);
        }
        NPY_NO_EXPORT void *
        shift_zero(size_t sz, size_t cnt) {
            char *real = (char *)calloc(sz + 64, cnt);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated %ld", sz);
            return (void *)(real + 64);
        }
        NPY_NO_EXPORT void
        shift_free(void * p, npy_uintp sz) {
            if (p == NULL) {
                return ;
            }
            char *real = (char *)p - 64;
            if (strncmp(real, "originally allocated", 20) != 0) {
                fprintf(stdout, "uh-oh, unmatched shift_free\\n");
                /* Make gcc crash by calling free on the wrong address */
                free((char *)p + 10);
                /* free(p); */
            }
            else {
                free(real);
            }
        }
        NPY_NO_EXPORT void *
        shift_realloc(void * p, npy_uintp sz) {
            if (p != NULL) {
                char *real = (char *)p - 64;
                if (strncmp(real, "originally allocated", 20) != 0) {
                    fprintf(stdout, "uh-oh, unmatched shift_realloc\\n");
                    return realloc(p, sz);
                }
                return (void *)((char *)realloc(real, sz + 64) + 64);
            }
            else {
                char *real = (char *)realloc(p, sz + 64);
                if (real == NULL) {
                    return NULL;
                }
                snprintf(real, 64, "originally allocated (realloc) %ld", sz);
                return (void *)(real + 64);
            }
        }
        static PyDataMem_Handler new_handler = {
            "secret_data_allocator",
            shift_alloc,      /* alloc */
            shift_zero, /* zeroed_alloc */
            shift_free,       /* free */
            shift_realloc,      /* realloc */
            memcpy,               /* host2obj */
            memcpy,               /* obj2host */
            memcpy,               /* obj2obj */
        };
        '''
    more_init="import_array();"
    try:
        import mem_policy
        return mem_policy
    except ImportError:
        pass
    # if it does not exist, build and load it
    try:
        return extbuild.build_and_import_extension('mem_policy',
            functions, prologue=prologue, include_dirs=[np.get_include()],
            build_dir=tmp_path, more_init=more_init)
    except:
        raise
        pytest.skip("could not build module")


def test_set_policy(get_module):
    a = np.arange(10)
    orig_policy = get_module.test_prefix(a)
    assert get_module.set_new_policy() == orig_policy
    if orig_policy == 'default_allocator':
        get_module.set_old_policy()

@pytest.mark.slow
def test_new_policy(get_module):
    a = np.arange(10)
    orig_policy = get_module.test_prefix(a)
    assert get_module.set_new_policy() == orig_policy
    b = np.arange(10)
    assert get_module.test_prefix(b) == 'secret_data_allocator'

    # test array manipulation. This is slow
    if orig_policy == 'default_allocator':
        # when the test set recurses into this test, the policy will be set
        # so this "if" will be false, preventing infinite recursion
        #
        # if needed, debug this by setting extra_argv=['-vvx']
        np.core.test(verbose=0, extra_argv=[])
    get_module.set_old_policy()
    assert get_module.test_prefix(a) == orig_policy
    c = np.arange(10)
    assert get_module.test_prefix(c) == 'default_allocator'

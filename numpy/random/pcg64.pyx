from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

import numpy as np
cimport numpy as np

from .common cimport *
from .distributions cimport bitgen_t
from .entropy import random_entropy

np.import_array()

# IF PCG_EMULATED_MATH==1:
cdef extern from "src/pcg64/pcg64.h":
    # Use int as generic type, actual type read from pcg64.h and is platform dependent
    ctypedef int pcg64_random_t

    struct s_pcg64_state:
        pcg64_random_t *pcg_state
        int has_uint32
        uint32_t uinteger

    ctypedef s_pcg64_state pcg64_state

    uint64_t pcg64_next64(pcg64_state *state)  nogil
    uint32_t pcg64_next32(pcg64_state *state)  nogil
    void pcg64_jump(pcg64_state *state)
    void pcg64_advance(pcg64_state *state, uint64_t *step)
    void pcg64_set_seed(pcg64_state *state, uint64_t *seed, uint64_t *inc)
    void pcg64_get_state(pcg64_state *state, uint64_t *state_arr, int *has_uint32, uint32_t *uinteger)
    void pcg64_set_state(pcg64_state *state, uint64_t *state_arr, int has_uint32, uint32_t uinteger)

cdef uint64_t pcg64_uint64(void* st) nogil:
    return pcg64_next64(<pcg64_state *>st)

cdef uint32_t pcg64_uint32(void *st) nogil:
    return pcg64_next32(<pcg64_state *> st)

cdef double pcg64_double(void* st) nogil:
    return uint64_to_double(pcg64_next64(<pcg64_state *>st))

cdef class PCG64:
    u"""
    PCG64(seed=None, inc=0)

    Container for the PCG-64 pseudo-random number generator.

    PCG-64 is a 128-bit implementation of O'Neill's permutation congruential
    generator ([1]_, [2]_). PCG-64 has a period of :math:`2^{128}` and supports
    advancing an arbitrary number of steps as well as :math:`2^{127}` streams.

    ``PCG64`` exposes no user-facing API except ``generator``,``state``,
    ``cffi`` and ``ctypes``. Designed for use in a ``Generator`` object.

    **Compatibility Guarantee**

    ``PCG64`` makes a guarantee that a fixed seed will always produce the same
    results.

    Parameters
    ----------
    seed : {None, long}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**128] or ``None`` (the default).
        If `seed` is ``None``, then ``PCG64`` will try to read data
        from ``/dev/urandom`` (or the Windows analog) if available. If
        unavailable, a 64-bit hash of the time and process ID is used.
    inc : {None, int}, optional
        Stream to return.
        Can be an integer in [0, 2**128] or ``None`` (the default).  If `inc` is
        ``None``, then 0 is used.  Can be used with the same seed to
        produce multiple streams using other values of inc.

    Notes
    -----
    Supports the method advance to advance the RNG an arbitrary number of
    steps. The state of the PCG-64 RNG is represented by 2 128-bit unsigned
    integers.

    See ``PCG32`` for a similar implementation with a smaller period.

    **Parallel Features**

    ``PCG64`` can be used in parallel applications in one of two ways.
    The preferable method is to use sub-streams, which are generated by using the
    same value of ``seed`` and incrementing the second value, ``inc``.

    >>> from numpy.random import Generator, PCG64
    >>> rg = [Generator(PCG64(1234, i + 1)) for i in range(10)]

    The alternative method is to call ``advance`` with a different value on
    each instance to produce non-overlapping sequences.

    >>> rg = [Generator(PCG64(1234, i + 1)) for i in range(10)]
    >>> for i in range(10):
    ...     rg[i].bitgen.advance(i * 2**64)

    **State and Seeding**

    The ``PCG64`` state vector consists of 2 unsigned 128-bit values,
    which are represented externally as python longs (2.x) or ints (Python 3+).
    ``PCG64`` is seeded using a single 128-bit unsigned integer
    (Python long/int). In addition, a second 128-bit unsigned integer is used
    to set the stream.

    References
    ----------
    .. [1] "PCG, A Family of Better Random Number Generators",
           http://www.pcg-random.org/
    .. [2] O'Neill, Melissa E. "PCG: A Family of Simple Fast Space-Efficient
           Statistically Good Algorithms for Random Number Generation"
    """
    cdef pcg64_state *rng_state
    cdef bitgen_t *_bitgen
    cdef public object capsule
    cdef object _ctypes
    cdef object _cffi
    cdef object _generator
    cdef public object lock

    def __init__(self, seed=None, inc=0):
        self.rng_state = <pcg64_state *>malloc(sizeof(pcg64_state))
        self.rng_state.pcg_state = <pcg64_random_t *>malloc(sizeof(pcg64_random_t))
        self._bitgen = <bitgen_t *>malloc(sizeof(bitgen_t))
        self.seed(seed, inc)
        self.lock = Lock()

        self._bitgen.state = <void *>self.rng_state
        self._bitgen.next_uint64 = &pcg64_uint64
        self._bitgen.next_uint32 = &pcg64_uint32
        self._bitgen.next_double = &pcg64_double
        self._bitgen.next_raw = &pcg64_uint64

        self._ctypes = None
        self._cffi = None
        self._generator = None

        cdef const char *name = "BitGenerator"
        self.capsule = PyCapsule_New(<void *>self._bitgen, name, NULL)

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        from ._pickle import __bitgen_ctor
        return (__bitgen_ctor,
                (self.state['bitgen'],),
                self.state)

    def __dealloc__(self):
        if self.rng_state:
            free(self.rng_state)
        if self._bitgen:
            free(self._bitgen)

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def random_raw(self, size=None, output=True):
        """
        random_raw(self, size=None)

        Return randoms as generated by the underlying BitGenerator

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        output : bool, optional
            Output values.  Used for performance testing since the generated
            values are not returned.

        Returns
        -------
        out : uint or ndarray
            Drawn samples.

        Notes
        -----
        This method directly exposes the the raw underlying pseudo-random
        number generator. All values are returned as unsigned 64-bit
        values irrespective of the number of bits produced by the PRNG.

        See the class docstring for the number of bits returned.
        """
        return random_raw(self._bitgen, self.lock, size, output)

    def _benchmark(self, Py_ssize_t cnt, method=u'uint64'):
        return benchmark(self._bitgen, self.lock, cnt, method)

    def seed(self, seed=None, inc=0):
        """
        seed(seed=None, inc=0)

        Seed the generator.

        This method is called when ``PCG64`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``PCG64``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``PCG64``.
        inc : int, optional
            Increment to use for PCG stream

        Raises
        ------
        ValueError
            If seed values are out of range for the RNG.

        """
        cdef np.ndarray _seed, _inc
        ub = 2 ** 128
        if seed is None:
            try:
                _seed = <np.ndarray>random_entropy(4)
            except RuntimeError:
                _seed = <np.ndarray>random_entropy(4, 'fallback')
            _seed = <np.ndarray>_seed.view(np.uint64)
        else:
            err_msg = 'seed must be a scalar integer between 0 and ' \
                      '{ub}'.format(ub=ub)
            if not np.isscalar(seed):
                raise TypeError(err_msg)
            if int(seed) != seed:
                raise TypeError(err_msg)
            if seed < 0 or seed > ub:
                raise ValueError(err_msg)
            _seed = <np.ndarray>np.empty(2, np.uint64)
            _seed[0] = int(seed) // 2**64
            _seed[1] = int(seed) % 2**64

        if not np.isscalar(inc):
            raise TypeError('inc must be a scalar integer between 0 and {ub}'.format(ub=ub))
        if inc < 0 or inc > ub or int(inc) != inc:
            raise ValueError('inc must be a scalar integer between 0 and {ub}'.format(ub=ub))
        _inc = <np.ndarray>np.empty(2, np.uint64)
        _inc[0] = int(inc) // 2**64
        _inc[1] = int(inc) % 2**64

        pcg64_set_seed(self.rng_state, <uint64_t *>_seed.data, <uint64_t *>_inc.data)
        self._reset_state_variables()

    @property
    def state(self):
        """
        Get or set the RNG state

        Returns
        -------
        state : dict
            Dictionary containing the information required to describe the
            state of the RNG
        """
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger

        # state_vec is state.high, state.low, inc.high, inc.low
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        pcg64_get_state(self.rng_state, <uint64_t *>state_vec.data, &has_uint32, &uinteger)
        state = int(state_vec[0]) * 2**64 + int(state_vec[1])
        inc = int(state_vec[2]) * 2**64 + int(state_vec[3])
        return {'bitgen': self.__class__.__name__,
                'state': {'state': state, 'inc': inc},
                'has_uint32': has_uint32,
                'uinteger': uinteger}

    @state.setter
    def state(self, value):
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bitgen', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'RNG'.format(self.__class__.__name__))
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        state_vec[0] = value['state']['state'] // 2 ** 64
        state_vec[1] = value['state']['state'] % 2 ** 64
        state_vec[2] = value['state']['inc'] // 2 ** 64
        state_vec[3] = value['state']['inc'] % 2 ** 64
        has_uint32 = value['has_uint32']
        uinteger = value['uinteger']
        pcg64_set_state(self.rng_state, <uint64_t *>state_vec.data, has_uint32, uinteger)

    def advance(self, delta):
        """
        advance(delta)

        Advance the underlying RNG as-if delta draws have occurred.

        Parameters
        ----------
        delta : integer, positive
            Number of draws to advance the RNG. Must be less than the
            size state variable in the underlying RNG.

        Returns
        -------
        self : PCG64
            RNG advanced delta steps

        Notes
        -----
        Advancing a RNG updates the underlying RNG state as-if a given
        number of calls to the underlying RNG have been made. In general
        there is not a one-to-one relationship between the number output
        random values from a particular distribution and the number of
        draws from the core RNG.  This occurs for two reasons:

        * The random values are simulated using a rejection-based method
          and so, on average, more than one value from the underlying
          RNG is required to generate an single draw.
        * The number of bits required to generate a simulated value
          differs from the number of bits generated by the underlying
          RNG.  For example, two 16-bit integer values can be simulated
          from a single draw of a 32-bit RNG.

        Advancing the RNG state resets any pre-computed random numbers.
        This is required to ensure exact reproducibility.
        """
        cdef np.ndarray d = np.empty(2, dtype=np.uint64)
        d[0] = delta // 2**64
        d[1] = delta % 2**64
        pcg64_advance(self.rng_state, <uint64_t *>d.data)
        self._reset_state_variables()
        return self

    def jump(self, np.npy_intp iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**64 random numbers have been generated

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : PCG64
            RNG jumped iter times

        Notes
        -----
        Jumping the rng state resets any pre-computed random numbers. This is required
        to ensure exact reproducibility.
        """
        return self.advance(iter * 2**64)

    @property
    def ctypes(self):
        """
        ctypes interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing ctypes wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the BitGenerator struct
        """
        if self._ctypes is None:
            self._ctypes = prepare_ctypes(self._bitgen)

        return self._ctypes

    @property
    def cffi(self):
        """
        CFFI interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing CFFI wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the BitGenerator struct
        """
        if self._cffi is not None:
            return self._cffi
        self._cffi = prepare_cffi(self._bitgen)
        return self._cffi

    @property
    def generator(self):
        """
        Return a Generator object

        Returns
        -------
        gen : numpy.random.Generator
            Random generator using this instance as the core RNG
        """
        if self._generator is None:
            from .generator import Generator
            self._generator = Generator (self)
        return self._generator

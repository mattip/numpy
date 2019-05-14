.. _new-or-different:

.. currentmodule:: numpy.random

What's New or Different
-----------------------

.. warning::

  The Box-Muller method used to produce NumPy's normals is no longer available
  in `~.RandomGenerator`.  It is not possible to reproduce the exact random
  values using ``RandomGenerator`` for the normal distribution or any other
  distribution that relies on the normal such as the `numpy.random.gamma` or
  `numpy.random.standard_t`. If you require bitwise backward compatible
  streams, use `~.mtrand.RandomState`.

Quick comparison of legacy `mtrand <legacy>`_ to the new `generator
<RandomGenerator>`

====================== ==================== =============
Feature                Older Equivalent     Notes
---------------------- -------------------- -------------
`RandomGenerator`      `RandomState`        ``RandomGenerator`` requires a stream
                                            source, called a RandomNumberGenerator
                                            (RNG). A number of different basic
                                            `RNGs <bit_generator>`_ exist.
                                            ``RandomState`` uses only the Box-
                                            Muller method.
---------------------- -------------------- -------------
``np.random.``         ``np.random.``       Access the next values in an already-
``RandomGenerator().`` ``random_sample()``  instaniated RNG, convert them to
``random_sample()``                         ``float64`` in the interval ``[0.0.,``
                                            `` 1.0)`` In addition to the ``size``
                                            kwarg, now supports ``dtype='d'`` or
                                            ``dtype='f'``, and an ``out`` kwarg to
                                            fill a user-supplied array.

                                            Many other distributions are also
                                            supported.
---------------------- -------------------- -------------
``RandomGenerator().`` ``randint``,         Use the ``closed`` kwarg to adjust
``integers()``         ``random_integers``  the inclusion or exclution of the
                                            ``high`` interval endpoint
====================== ==================== =============

And in more detail:

* `~.entropy.random_entropy` provides access to the system
  source of randomness that is used in cryptographic applications (e.g.,
  ``/dev/urandom`` on Unix).
* Simulate from the complex normal distribution
  (`~.RandomGenerator.complex_normal`)
* The normal, exponential and gamma generators use 256-step Ziggurat
  methods which are 2-10 times faster than NumPy's default implementation in
  `~.RandomGenerator.standard_normal`,
  `~.RandomGenerator.standard_exponential` or
  `~.RandomGenerator.standard_gamma`.
* `~.RandomGenerator.integers` is now the canonical way to generate integer
  random numbers from a discrete uniform distribution. The ``rand`` and
  ``randn`` methods are only availabe through the legacy `~.RandomState`.
  This replaces both ``randint`` and the deprecated ``random_integers``.
* The Box-Muller used to produce NumPy's normals is no longer available.
* All bit generators can produce doubles, uint64s and
  uint32s via CTypes (`~.xoroshiro128.Xoroshiro128.
  ctypes`) and CFFI (`~.xoroshiro128.Xoroshiro128.cffi`).
  This allows these bit generators to be used in numba.
* The bit generators can be used in downstream projects via
  Cython.


.. ipython:: python

  from  numpy.random import RandomGenerator, Xoroshiro128
  import numpy.random
  rg = RandomGenerator(Xoroshiro128())
  %timeit rg.standard_normal(100000)
  %timeit numpy.random.standard_normal(100000)

.. ipython:: python

  %timeit rg.standard_exponential(100000)
  %timeit numpy.random.standard_exponential(100000)

.. ipython:: python

  %timeit rg.standard_gamma(3.0, 100000)
  %timeit numpy.random.standard_gamma(3.0, 100000)

* Optional ``dtype`` argument that accepts ``np.float32`` or ``np.float64``
  to produce either single or double prevision uniform random variables for
  select distributions

  * Uniforms (`~.RandomGenerator.random_sample` and
    `~.RandomGenerator.rand`)
  * Normals (`~.RandomGenerator.standard_normal` and
    `~.RandomGenerator.randn`)
  * Standard Gammas (`~.RandomGenerator.standard_gamma`)
  * Standard Exponentials (`~.RandomGenerator.standard_exponential`)

.. ipython:: python

  rg.bit_generator.seed(0)
  rg.random_sample(3, dtype='d')
  rg.bit_generator.seed(0)
  rg.random_sample(3, dtype='f')

* Optional ``out`` argument that allows existing arrays to be filled for
  select distributions

  * Uniforms (`~.RandomGenerator.random_sample`)
  * Normals (`~.RandomGenerator.standard_normal`)
  * Standard Gammas (`~.RandomGenerator.standard_gamma`)
  * Standard Exponentials (`~.RandomGenerator.standard_exponential`)

  This allows multithreading to fill large arrays in chunks using suitable
  PRNGs in parallel.

.. ipython:: python

  existing = np.zeros(4)
  rg.random_sample(out=existing[:2])
  print(existing)


.. currentmodule:: numpy.random

Random Generator
----------------
The `~RandomGenerator` provides access to
a wide range of distributions, and served as a replacement for
:class:`~numpy.random.RandomState`.  The main difference between
the two is that ``RandomGenerator`` relies on an additional bit generator to
manage state and generate the random bits, which are then transformed into
random values from useful distributions. The default bit generator used by
``RandomGenerator`` is :class:`~xoroshiro128.Xoroshiro128`.  The bit generator
can be changed by passing an instantized bit generator to ``RandomGenerator``.


.. autoclass:: RandomGenerator
	:exclude-members:

Accessing the RNG
=================
.. autosummary::
   :toctree: generated/

   ~RandomGenerator.brng

Simple random data
==================
.. autosummary::
   :toctree: generated/

   ~RandomGenerator.rand
   ~RandomGenerator.randn
   ~RandomGenerator.randint
   ~RandomGenerator.random_integers
   ~RandomGenerator.random_sample
   ~RandomGenerator.choice
   ~RandomGenerator.bytes

Permutations
============
.. autosummary::
   :toctree: generated/

   ~RandomGenerator.shuffle
   ~RandomGenerator.permutation

Distributions
=============
.. autosummary::
   :toctree: generated/

   ~RandomGenerator.beta
   ~RandomGenerator.binomial
   ~RandomGenerator.chisquare
   ~RandomGenerator.dirichlet
   ~RandomGenerator.exponential
   ~RandomGenerator.f
   ~RandomGenerator.gamma
   ~RandomGenerator.geometric
   ~RandomGenerator.gumbel
   ~RandomGenerator.hypergeometric
   ~RandomGenerator.laplace
   ~RandomGenerator.logistic
   ~RandomGenerator.lognormal
   ~RandomGenerator.logseries
   ~RandomGenerator.multinomial
   ~RandomGenerator.multivariate_normal
   ~RandomGenerator.negative_binomial
   ~RandomGenerator.noncentral_chisquare
   ~RandomGenerator.noncentral_f
   ~RandomGenerator.normal
   ~RandomGenerator.pareto
   ~RandomGenerator.poisson
   ~RandomGenerator.power
   ~RandomGenerator.rayleigh
   ~RandomGenerator.standard_cauchy
   ~RandomGenerator.standard_exponential
   ~RandomGenerator.standard_gamma
   ~RandomGenerator.standard_normal
   ~RandomGenerator.standard_t
   ~RandomGenerator.triangular
   ~RandomGenerator.uniform
   ~RandomGenerator.vonmises
   ~RandomGenerator.wald
   ~RandomGenerator.weibull
   ~RandomGenerator.zipf

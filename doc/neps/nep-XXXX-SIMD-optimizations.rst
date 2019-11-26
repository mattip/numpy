===============================================================
NEP XXXX — Using SIMD optimization instructions for performance
===============================================================

:Author: Sayed Adel, Matti Picus, Ralf Gommers
:Status: Draft
:Type: Standards
:Created: 2019-11-25
:Resolution: none


Abstract
--------

While compilers are getting better at using hardware-specific routines to
optimize code, they sometimes do not produce optimal results. Also, we would
like to be able to copy binary C-extension modules from one machine to another
with the same architecture without recompiling.

We have a mechanism in the ufunc machinery to `build alternative loops`_
indexed by CPU feature name. At import (in ``InitOperators``), the loop
function that matches the run-time CPU info `is chosen`_ from the candidates.This
NEP proposes a mechanism to build on that for many more features and
architectures.  The steps proposed are to:

- Establish a baseline of CPU features for minimal support
- Write explicit code to take advantage of well-defined, architecture-agnostic,
  universal intrisics which capture features available across architectures.
- Capture those universal intrisics in a set of C macros that at compile time
  would build code paths for sets of features from the baseline up to the maximum
  set of features available on that architecture, perhaps in only one step.
- At runtime, discover which CPU features are available, and choose from among
  the possible code paths accordingly.


Motivation and Scope
--------------------

Traditionally NumPy has counted on the compilers to generate optimal code
specifically for the target architecture.
However few users today compile NumPy locally for their machines. Most use the
binary packages which must then cater to the lowest-common denominator. Thus
users do not optimally use the more advanced features of their CPU processors.
Traditionally, these features have been exposed through `intrinsics`_ which are
compiler-specific instructions that map directly to assembly instructions.
Recently there were discussions about the effectiveness of adding more
intrinsics (e.g., `gh-11113`_ for AVX optimizations for floats).  In the past,
architecture-specific code was added to NumPy for `fast avx512 routines`_ in
various ufuncs, using the mechanism described above to choose the best loop
for the architecture. However the code is not generic and does not generalize
to other architectures.

Recently, OpenCV moved to using `universal intrinsics`_ in the Hardware
Abstraction Layer (HAL) which provided a nice abstraction for common shared
Single Instruction Multiple Data (SIMD) constructs. This NEP proposes a similar
mechanism for NumPy. There are three stages to using the mechanism:
- Infrastructure is provided in the code for abstract intrinsics. The ufunc
  machinery will be extended using sets of these abstract intrinsics, so that
  a single ufunc will be expressed as a set of loops, going from a minimal to
  a maximal set of possibly availabe intrinsics.
- At compile time, compiler macros and CPU detection are used to turn the
  abstract intrinsics into concrete intrinsic calls. Any intrinsics not
  available on the platform, either because the CPU does not support them
  (and so cannot be tested) or because the abstract intrinsic does not have a
  parallel concrete intrinsic on the platform will not error, rather the
  corresponding loop will not be produced and added to the set of
  possibilities.
- At runtime, the CPU detection code will further limit the set of loops
  available, and the optimal one will be chosen for the ufunc.

The current NEP proposes only to use the runtime feature detection and optimal
loop selection mechanism for ufuncs. Future NEPS may propose other uses for the
proposed solution.

Usage and Impact
----------------

The end user will be able to get a list of available intrinsics. Optionally,
the user may be able to specify which of the loops available at runtime will be
used, perhaps via an environment variable to enable benchmarking the impact of
the different loops. There should be no direct impact to naive end users, the
results of all the loops should be identical to within a small number (1-3)
ULPs. On the other hand, users with more powerful machines should notice a
performance boost.

Binary releases - wheels on PyPI and conda packages
```````````````````````````````````````````````````

The binaries released by this process will be larger since they include all
possible loops for the architecture. Some packagers may prefer to limit the
number of loops in order to limit the size of the binaries, we would hope they
would still support a wide range of families of architectures. Note this
problem already exists in the Intel MKL offering.

Source builds
`````````````
TBD
- Setting the baseline and set of runtime-dispatchable ISA extensions
- Behavior when compiler or hardware doesn't support a requested ISA extension


How to run benchmarks to assess performance benefits
````````````````````````````````````````````````````

Adding more code which use intrinsics will make the code harder to maintain.
Therefore, such code should only be added if it yields a significant
performance benefit. Assessing this performance benefit can be nontrivial.
To aid with this, the implementation for this NEP will add a way to select
which instruction sets can be used at *runtime* via environment variables.
(name TBD).


Diagnostics
```````````

A new dictionary `__cpu_features__` will be available to python. The keys are
the available features, the value is a boolean ``True``. Various new private
C functions will be used internally to query available features. These
might be exposed via specific c-extension modules for testing.


Workflow for adding a new CPU architecture-specific optimization
````````````````````````````````````````````````````````````````

NumPy will always have a baseline C implementation for any code that may be
a candidate for SIMD vectorization.  If a contributor wants to add SIMD
support for some architecture (typically the one of most interest to them),
this is the proposed workflow:

TODO (see https://github.com/numpy/numpy/pull/13516#issuecomment-558859638,
needs to be worked out more)

Reuse by other projects
```````````````````````

It would be nice if the universal intrinsics would be available to other
libraries like SciPy or Astropy that also build ufuncs, but that is not an
explicit goal of the first implementation of this NEP.

Backward compatibility
----------------------

There should be no impact on backwards compatibility.


Detailed description
--------------------

*This section should provide a detailed description of the proposed change.
It should include examples of how the new functionality would be used,
intended use-cases and pseudo-code illustrating its use.*

TODO: status today - what instructions are used at build time (SSE2/3 and
AVX/AVX512 for XXX functionality) and at runtime (some but less, see
``loops.c.src``)


Related Work
------------

- PIXMAX TBD: what is it?
- `Eigen`_ is a C++ template library for linear algebra: matrices, vectors,
  numerical solvers, and related algorithms. It is a higher level-abstraction
  than the intrinsics discussed here.
- `xsimd`_ is a header-only C++ library for x86 and ARM that implements the
  mathematical functions used in the algorithms of ``boost.SIMD``.
- OpenCV used to have the one-implementation-per-architecture design, but more
  recently moved to a design that is quite similar to what is proposed in this
  NEP. The top-level `dispatch code`_ includes a `generic header`_ that is
  `specialized at compile time`_ by the CMakefile system.


Implementation
--------------

Current PRs:

- `gh-13421 improve runtime detection of CPU features <https://github.com/numpy/numpy/pull/13421>`_
- `gh-13516: enable multi-platform SIMD compiler optimizations <https://github.com/numpy/numpy/pull/13516>`_

**Let's leave description of this out for now. Only do that once the questions
in the sections above are answered.**


Alternatives
------------

A proposed alternative in gh-13516_ is a per CPU architecture implementation of
SIMD code (e.g., have `loops.avx512.c.src`, `loops.avx2.c.src`, `loops.sse.c.src`,
`loops.vsx.c.src`, `loops.neon.c.src`, etc.). This is more similar to what
PIXMAX does. There's a lot of duplication here though, it is likely to be
much harder to maintain.


Discussion
----------

*This section may just be a bullet list including links to any discussions
regarding the NEP:

- This includes links to mailing list threads or relevant GitHub issues.*



References and Footnotes
------------------------

.. _`build alternative loops`: https://github.com/numpy/numpy/blob/v1.17.4/numpy/core/code_generators/generate_umath.py#L50
.. _`is chosen`: https://github.com/numpy/numpy/blob/v1.17.4/numpy/core/code_generators/generate_umath.py#L1038
.. _`gh-11113"`: https://github.com/numpy/numpy/pull/11113
.. _`fast avx512 routines`: https://github.com/numpy/numpy/pulls?q=is%3Apr+avx512+is%3Aclosed

.. [1] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. _`xsimd`: https://xsimd.readthedocs.io/en/latest/
.. _`Eigen`: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _`dispatch code`: https://github.com/opencv/opencv/blob/4.1.2/modules/core/src/arithm.dispatch.cpp
.. _`generic header`: https://github.com/opencv/opencv/blob/4.1.2/modules/core/src/arithm.simd.hpp
.. _`specialized at compile time`: https://github.com/opencv/opencv/blob/4.1.2/modules/core/CMakeLists.txt#L3-#L13
.. _`intrinsics`: https://software.intel.com/en-us/cpp-compiler-developer-guide-and-reference-intrinsics
.. _`universal intrinsics`: https://docs.opencv.org/master/df/d91/group__core__hal__intrin.html

Copyright
---------

This document has been placed in the public domain. [1]_

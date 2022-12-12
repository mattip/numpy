import os
import sys
import time


# Some options for running this below...

# multiplicator for the game of life matrix size
SIZE = 5
# use full numpy or just the multiarray_umath module directly
USE_FULL_NUMPY = False
# run with pyperf or just iterate a few times, printing the game of life to the terminal
USE_PYPERF = False


if not USE_FULL_NUMPY:
    sys.path.append(os.path.join(os.path.dirname(__file__), "numpy", "core"))

    # hack it so we can run this example with really minimal numpy import
    import struct
    start = time.time()
    import _multiarray_umath
    print("Import of", _multiarray_umath.__file__, "took", time.time() - start)
    # time.sleep(10)

    class np:
        @staticmethod
        def zeros(shape, dtype):
            return _multiarray_umath.zeros(shape, dtype)
            # return _multiarray_umath.ndarray(shape, dtype) * 0

        @staticmethod
        def argwhere(a):
            return _multiarray_umath.array(a.nonzero()).transpose()
            # return np.array(a.nonzero()).transpose()

        @staticmethod
        def array(nested_sequence):
            return _multiarray_umath.array(nested_sequence)
            # if not isinstance(nested_sequence, (list, tuple)):
            #     raise "hack only works for 2d-lists"
            # x = len(nested_sequence[0])
            # dtype = None
            # for ns in nested_sequence:
            #     assert len(ns) == x
            #     if dtype:
            #         assert isinstance(ns[0], dtype)
            #     elif len(ns) > 0:
            #         dtype = type(ns[0])
            #     else:
            #         dtype = int
            # y = len(nested_sequence)
            # ary = _multiarray_umath.ndarray((y, x), dtype)
            # for y,ns in enumerate(nested_sequence):
            #     for x,e in enumerate(ns):
            #         ary[y,x] = e
            # return ary

    def aryprint(a, indent=0):
        if USE_PYPERF:
            p = lambda *a, **k: None
        else:
            p = print
        shape = a.shape
        p(" " * indent, "[", sep="", end=("" if len(shape) == 1 else "\n"))
        for i in range(shape[0]):
            if len(shape) == 1:
                p(int(a[i]), " ", sep="", end="")
            else:
                aryprint(a[i], indent=indent + 1)
                p(",")
        p("]", end="")
else:
    start = time.time()
    import numpy as np
    print("Import of", np.__file__, "took", time.time() - start)

    def aryprint(a):
        if USE_PYPERF:
            p = lambda *a, **k: None
        else:
            p = print
        p(a)


def iterate(Z):
    N = np.zeros(Z.shape, int)
    N[1:-1,1:-1] += (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
                     Z[1:-1,0:-2]                + Z[1:-1,2:] +
                     Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
    N_ = N.ravel()
    Z_ = Z.ravel()

    R1 = np.argwhere( (Z_==1) & (N_ < 2) )
    R2 = np.argwhere( (Z_==1) & (N_ > 3) )
    R3 = np.argwhere( (Z_==1) & ((N_==2) | (N_==3)) )
    R4 = np.argwhere( (Z_==0) & (N_==3) )

    Z_[R1] = 0
    Z_[R2] = 0
    Z_[R3] = Z_[R3]
    Z_[R4] = 1

    Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = 0

    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z


def setup():
    Z = np.array([[0,0,0,0,0,0] * SIZE,
                  [0,0,0,1,0,0] * SIZE,
                  [0,1,0,1,0,0] * SIZE,
                  [0,0,1,1,0,0] * SIZE,
                  [0,0,0,0,0,0] * SIZE,
                  [0,0,0,0,0,0] * SIZE] * SIZE)
    return Z


def benchmark(Z, inner_loops=20):
    s = time.time()
    if not USE_PYPERF:
        print("\033c")
        print("\033[0;0H")
    aryprint(Z)
    for i in range(inner_loops):
        iterate(Z)
        if not USE_PYPERF:
            print("\033[0;0H")
        aryprint(Z)
    if not USE_PYPERF:
        print("done after", time.time() - s)


if not USE_PYPERF:
    Z = setup()
    print("start")
    for i in range(10):
        benchmark(Z)
else:
    import pyperf as perf
    inner_loops = 20
    runner = perf.Runner()
    runner.timeit(
        f"game-of-life{'-full' if USE_FULL_NUMPY else '-limited'}-{SIZE}",
        globals={
            "benchmark": benchmark,
            "np": np,
            "iterate": iterate,
            "inner_loops": inner_loops,
            "setup": setup,
        },
        inner_loops=inner_loops,
        setup="Z = setup()",
        stmt="benchmark(Z, inner_loops)"
    )

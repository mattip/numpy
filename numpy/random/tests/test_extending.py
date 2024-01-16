from importlib.util import spec_from_file_location, module_from_spec
import os
import pathlib
import pytest
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import warnings

import numpy as np
from numpy.testing import IS_WASM


try:
    import cffi
except ImportError:
    cffi = None

if sys.flags.optimize > 1:
    # no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1
    # cffi cannot succeed
    cffi = None

try:
    with warnings.catch_warnings(record=True) as w:
        # numba issue gh-4733
        warnings.filterwarnings('always', '', DeprecationWarning)
        import numba
except (ImportError, SystemError):
    # Certain numpy/numba versions trigger a SystemError due to a numba bug
    numba = None

try:
    import cython
    from Cython.Compiler.Version import version as cython_version
except ImportError:
    cython = None
else:
    from numpy._utils import _pep440
    # Cython 0.29.30 is required for Python 3.11 and there are
    # other fixes in the 0.29 series that are needed even for earlier
    # Python versions.
    # Note: keep in sync with the one in pyproject.toml
    required_version = '0.29.35'
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # too old or wrong cython, skip the test
        cython = None


@pytest.mark.skipif(
        sys.platform == "win32" and sys.maxsize < 2**32,
        reason="Failing in 32-bit Windows wheel build job, skip for now"
)
@pytest.mark.skipif(IS_WASM, reason="Can't start subprocess")
@pytest.mark.skipif(cython is None, reason="requires cython")
@pytest.mark.slow
def test_cython(tmp_path):
    import glob
    # build the examples in a temporary directory
    srcdir = os.path.join(os.path.dirname(__file__), '..')
    shutil.copytree(srcdir, tmp_path / 'random')
    build_dir = tmp_path / 'random' / '_examples' / 'cython'
    target_dir = build_dir / "build"
    os.makedirs(target_dir, exist_ok=True)
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release", 
                               "--vsenv", str(build_dir)],
                              cwd=target_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup", str(build_dir)],
                              cwd=target_dir
                              )
    subprocess.check_call(["meson", "compile", "-vv"], cwd=target_dir)

    # gh-16162: make sure numpy's __init__.pxd was used for cython
    # not really part of this test, but it is a convenient place to check

    g = glob.glob(str(target_dir / "*" / "extending.pyx.c"))
    init_txt = 'NumPy API declarations from "numpy/__init__'
    math_txt = 'NumPy API declarations from "numpy/math'
    found = set()
    with open(g[0]) as fid:
        for i, line in enumerate(fid):
            if init_txt in line:
                found.add("init")
            elif math_txt in line:
                found.add("math")
    assert "math" in found, (f"Could not find '{math_txt}' in C file, "
                           "numpy/math.pxd not used")
    assert "init" in found, (f"Could not find '{init_txt}' in C file, "
                           "numpy/__init__*.pxd not used")
    # import without adding the directory to sys.path
    suffix = sysconfig.get_config_var('EXT_SUFFIX')

    def load(modname):
        so = (target_dir / modname).with_suffix(suffix)
        spec = spec_from_file_location(modname, so)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # test that the module can be imported
    load("extending")
    load("extending_cpp")
    # actually test the cython c-extension
    extending_distributions = load("extending_distributions")
    from numpy.random import PCG64
    values = extending_distributions.uniforms_ex(PCG64(0), 10, 'd')
    assert values.shape == (10,)
    assert values.dtype == np.float64

@pytest.mark.skipif(numba is None or cffi is None,
                    reason="requires numba and cffi")
def test_numba():
    from numpy.random._examples.numba import extending  # noqa: F401

@pytest.mark.skipif(cffi is None, reason="requires cffi")
def test_cffi():
    from numpy.random._examples.cffi import extending  # noqa: F401

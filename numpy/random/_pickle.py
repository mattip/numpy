from .generator import RandomGenerator
from .dsfmt import DSFMT
from .mt19937 import MT19937
from .pcg32 import PCG32
from .pcg64 import PCG64
from .philox import Philox
from .threefry import ThreeFry
from .threefry32 import ThreeFry32
from .xoroshiro128 import Xoroshiro128
from .xorshift1024 import Xorshift1024
from .xoshiro256starstar import Xoshiro256StarStar
from .xoshiro512starstar import Xoshiro512StarStar
from .mtrand import RandomState

BitGeneratorS = {'MT19937': MT19937,
             'DSFMT': DSFMT,
             'PCG32': PCG32,
             'PCG64': PCG64,
             'Philox': Philox,
             'ThreeFry': ThreeFry,
             'ThreeFry32': ThreeFry32,
             'Xorshift1024': Xorshift1024,
             'Xoroshiro128': Xoroshiro128,
             'Xoshiro256StarStar': Xoshiro256StarStar,
             'Xoshiro512StarStar': Xoshiro512StarStar,
             }


def __generator_ctor(bitgen_name='mt19937'):
    """
    Pickling helper function that returns a RandomGenerator object

    Parameters
    ----------
    bitgen_name: str
        String containing the core BitGenerator

    Returns
    -------
    rg: RandomGenerator
        RandomGenerator using the named core BitGenerator
    """
    try:
        bitgen_name = bitgen_name.decode('ascii')
    except AttributeError:
        pass
    if bitgen_name in BitGeneratorS:
        bitgen = BitGeneratorS[bitgen_name]
    else:
        raise ValueError(str(bitgen_name) + ' is not a known BitGenerator module.')

    return RandomGenerator(bitgen())


def __bitgen_ctor(bitgen_name='mt19937'):
    """
    Pickling helper function that returns a bit generator object

    Parameters
    ----------
    bitgen_name: str
        String containing the name of the bit generator

    Returns
    -------
    bitgen: BitGenerator
        Basic RNG instance
    """
    try:
        bitgen_name = bitgen_name.decode('ascii')
    except AttributeError:
        pass
    if bitgen_name in BitGeneratorS:
        bitgen = BitGeneratorS[bitgen_name]
    else:
        raise ValueError(str(bitgen_name) + ' is not a known BitGenerator module.')

    return bitgen()


def __randomstate_ctor(bitgen_name='mt19937'):
    """
    Pickling helper function that returns a legacy RandomState-like object

    Parameters
    ----------
    bitgen_name: str
        String containing the core BitGenerator

    Returns
    -------
    rs: RandomState
        Legacy RandomState using the named core BitGenerator
    """
    try:
        bitgen_name = bitgen_name.decode('ascii')
    except AttributeError:
        pass
    if bitgen_name in BitGeneratorS:
        bitgen = BitGeneratorS[bitgen_name]
    else:
        raise ValueError(str(bitgen_name) + ' is not a known BitGenerator module.')

    return RandomState(bitgen())

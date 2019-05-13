from .generator import RandomGenerator
from .dsfmt import DSFMT
from .mt19937 import MT19937
from .philox import Philox
from .threefry import ThreeFry
from .threefry32 import ThreeFry32
from .xoroshiro128 import Xoroshiro128
from .xorshift1024 import Xorshift1024
from .xoshiro256starstar import Xoshiro256StarStar
from .xoshiro512starstar import Xoshiro512StarStar
from .mtrand import RandomState

BasicRNGS = {'MT19937': MT19937,
             'DSFMT': DSFMT,
             'Philox': Philox,
             'ThreeFry': ThreeFry,
             'ThreeFry32': ThreeFry32,
             'Xorshift1024': Xorshift1024,
             'Xoroshiro128': Xoroshiro128,
             'Xoshiro256StarStar': Xoshiro256StarStar,
             'Xoshiro512StarStar': Xoshiro512StarStar,
             }


def __generator_ctor(brng_name='mt19937'):
    """
    Pickling helper function that returns a RandomGenerator object

    Parameters
    ----------
    brng_name: str
        String containing the core BasicRNG

    Returns
    -------
    rg: RandomGenerator
        RandomGenerator using the named core BasicRNG
    """
    try:
        brng_name = brng_name.decode('ascii')
    except AttributeError:
        pass
    if brng_name in BasicRNGS:
        brng = BasicRNGS[brng_name]
    else:
        raise ValueError(str(brng_name) + ' is not a known BasicRNG module.')

    return RandomGenerator(brng())


def __brng_ctor(brng_name='mt19937'):
    """
    Pickling helper function that returns a basic RNG object

    Parameters
    ----------
    brng_name: str
        String containing the name of the Basic RNG

    Returns
    -------
    brng: BasicRNG
        Basic RNG instance
    """
    try:
        brng_name = brng_name.decode('ascii')
    except AttributeError:
        pass
    if brng_name in BasicRNGS:
        brng = BasicRNGS[brng_name]
    else:
        raise ValueError(str(brng_name) + ' is not a known BasicRNG module.')

    return brng()


def __randomstate_ctor(brng_name='mt19937'):
    """
    Pickling helper function that returns a legacy RandomState-like object

    Parameters
    ----------
    brng_name: str
        String containing the core BasicRNG

    Returns
    -------
    rs: RandomState
        Legacy RandomState using the named core BasicRNG
    """
    try:
        brng_name = brng_name.decode('ascii')
    except AttributeError:
        pass
    if brng_name in BasicRNGS:
        brng = BasicRNGS[brng_name]
    else:
        raise ValueError(str(brng_name) + ' is not a known BasicRNG module.')

    return RandomState(brng())

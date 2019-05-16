from .mtrand import RandomState
from .philox import Philox
from .threefry import ThreeFry
from .threefry32 import ThreeFry32
from .xoroshiro128 import Xoroshiro128
from .xorshift1024 import Xorshift1024
from .xoshiro256starstar import Xoshiro256StarStar
from .xoshiro512starstar import Xoshiro512StarStar

from .dsfmt import DSFMT
from .generator import Generator
from .mt19937 import MT19937

BitGeneratorS = {'MT19937': MT19937,
             'DSFMT': DSFMT,
             'Philox': Philox,
             'ThreeFry': ThreeFry,
             'ThreeFry32': ThreeFry32,
             'Xorshift1024': Xorshift1024,
             'Xoroshiro128': Xoroshiro128,
             'Xoshiro256StarStar': Xoshiro256StarStar,
             'Xoshiro512StarStar': Xoshiro512StarStar,
             }


def __generator_ctor(bit_generator_name='mt19937'):
    """
    Pickling helper function that returns a Generator object

    Parameters
    ----------
    bit_generator_name: str
        String containing the core BitGenerator

    Returns
    -------
    rg: Generator
        Generator using the named core BitGenerator
    """
    if bit_generator_name in BitGeneratorS:
        bit_generator = BitGeneratorS[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known BitGenerator module.')

    return Generator(bit_generator())


def __bit_generator_ctor(bit_generator_name='mt19937'):
    """
    Pickling helper function that returns a bit generator object

    Parameters
    ----------
    bit_generator_name: str
        String containing the name of the BitGenerator

    Returns
    -------
    bit_generator: BitGenerator
        BitGenerator instance
    """
    if bit_generator_name in BitGeneratorS:
        bit_generator = BitGeneratorS[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known BitGenerator module.')

    return bit_generator()


def __randomstate_ctor(bit_generator_name='mt19937'):
    """
    Pickling helper function that returns a legacy RandomState-like object

    Parameters
    ----------
    bit_generator_name: str
        String containing the core BitGenerator

    Returns
    -------
    rs: RandomState
        Legacy RandomState using the named core BitGenerator
    """
    if bit_generator_name in BitGeneratorS:
        bit_generator = BitGeneratorS[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known BitGenerator module.')

    return RandomState(bit_generator())

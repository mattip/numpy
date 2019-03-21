from __future__ import division, print_function

from os.path import join
import sys
import os
import platform
import struct
from distutils.dep_util import newer
from distutils.msvccompiler import get_build_version as get_msvc_build_version

def needs_mingw_ftime_workaround():
    # We need the mingw workaround for _ftime if the msvc runtime version is
    # 7.1 or above and we build with mingw ...
    # ... but we can't easily detect compiler version outside distutils command
    # context, so we will need to detect in randomkit whether we build with gcc
    msver = get_msvc_build_version()
    if msver and msver >= 8:
        return True

    return False


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    config = Configuration('random', parent_package, top_path)

    def generate_libraries(ext, build_dir):
        config_cmd = config.get_config_cmd()
        libs = get_mathlibs()
        if sys.platform == 'win32':
            libs.append('Advapi32')
        ext.libraries.extend(libs)
        return None

    # enable unix large file support on 32 bit systems
    # (64 bit off_t, lseek -> lseek64 etc.)
    if sys.platform[:3] == "aix":
        defs = [('_LARGE_FILES', None)]
    else:
        defs = [('_FILE_OFFSET_BITS', '64'),
                ('_LARGEFILE_SOURCE', '1'),
                ('_LARGEFILE64_SOURCE', '1')]
    if needs_mingw_ftime_workaround():
        defs.append(("NPY_NEEDS_MINGW_TIME_WORKAROUND", None))

    libs = []
    defs.append(('NPY_NO_DEPRECATED_API', 0))
    # Configure mtrand
    config.add_extension('mtrand',
                         sources=[join('mtrand', x) for x in
                                  ['mtrand.c', 'randomkit.c', 'initarray.c',
                                   'distributions.c']]+[generate_libraries],
                         libraries=libs,
                         depends=[join('mtrand', '*.h'),
                                  join('mtrand', '*.pyx'),
                                  join('mtrand', '*.pxi'),],
                         define_macros=defs,
                         )

    config.add_data_files(('.', join('mtrand', 'randomkit.h')))
    config.add_data_dir('tests')

    ##############################
    # randomgen
    ##############################

    # Make a guess as to whether SSE2 is present for now, TODO: Improve
    USE_SSE2 = False
    for k in platform.uname():
        for val in ('x86', 'i686', 'i386', 'amd64'):
            USE_SSE2 = USE_SSE2 or val in k.lower()
    print('Building with SSE?: {0}'.format(USE_SSE2))
    if '--no-sse2' in sys.argv:
        USE_SSE2 = False
        sys.argv.remove('--no-sse2')

    DEBUG = False
    PCG_EMULATED_MATH = False
    EXTRA_LINK_ARGS = []
    EXTRA_LIBRARIES = ['m'] if os.name != 'nt' else []
    EXTRA_COMPILE_ARGS = [] if os.name == 'nt' else [
        '-std=c99', '-U__GNUC_GNU_INLINE__']
    if os.name == 'nt':
        EXTRA_LINK_ARGS = ['/LTCG', '/OPT:REF', 'Advapi32.lib', 'Kernel32.lib']
        if DEBUG:
            EXTRA_LINK_ARGS += ['-debug']
            EXTRA_COMPILE_ARGS += ["-Zi", "/Od"]
        if sys.version_info < (3, 0):
            EXTRA_INCLUDE_DIRS += [join(MOD_DIR, 'src', 'common')]

    #
    # PCG64_DEFS = []
    # if sys.maxsize < 2 ** 32 or os.name == 'nt':
    #     # Force emulated mode here
    #     PCG_EMULATED_MATH = True
    #     PCG64_DEFS += [('PCG_FORCE_EMULATED_128BIT_MATH', '1')]
    # 
    # if struct.calcsize('P') < 8:
    #     PCG_EMULATED_MATH = True
    # defs.append(('PCG_EMULATED_MATH', int(PCG_EMULATED_MATH)))

    DSFMT_DEFS = [('DSFMT_MEXP', '19937')]
    if USE_SSE2:
        if os.name == 'nt':
            EXTRA_COMPILE_ARGS += ['/wd4146', '/GL']
            if struct.calcsize('P') < 8:
                EXTRA_COMPILE_ARGS += ['/arch:SSE2']
        else:
            EXTRA_COMPILE_ARGS += ['-msse2']
        DSFMT_DEFS += [('HAVE_SSE2', '1')]

    config.add_extension('randomgen.entropy',
                        sources=[join('randomgen', x) for x in
                            ['entropy.c', 'src/entropy/entropy.c']],
                        include_dirs=[join('randomgen', 'src', 'entropy')],
                        libraries=EXTRA_LIBRARIES,
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS,
                        depends=[join('randomgen', 'src', 'splitmix64', 'splitmix.h'),
                                 join('randomgen', 'src', 'entropy', 'entropy.h'),
                                 join('randomgen', 'entropy.pyx'),
                                ],
                        define_macros=defs,
                        )
    config.add_extension('randomgen.dsfmt',
                        sources=[join('randomgen', x) for x in
                            ['dsfmt.c', 'src/dsfmt/dSFMT.c',
                             'src/dsfmt/dSFMT-jump.c',
                             'src/aligned_malloc/aligned_malloc.c']],
                        include_dirs=[join('randomgen', 'src', 'dsfmt')],
                        libraries=EXTRA_LIBRARIES,
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS,
                        depends=[join('randomgen', 'src', 'dsfmt', 'dsfmt.h'),
                                 join('randomgen', 'dsfmt.pyx'),
                                ],
                        define_macros=defs + DSFMT_DEFS,
                        )
    config.add_extension('randomgen.legacy._legacy',
                        sources=[join('randomgen', x) for x in
                            ['legacy/_legacy.c',
                             'src/legacy/distributions-boxmuller.c',
                             'src/distributions/distributions.c' ]],
                        include_dirs=['randomgen', join('randomgen', 'legacy')],
                        libraries=EXTRA_LIBRARIES,
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS,
                        depends=[join('randomgen', 'legacy', '_legacy.pyx')],
                        define_macros=defs + DSFMT_DEFS,
                        )
    for gen in ['mt19937']:
        # gen.pyx, src/gen/gen.c, src/gen/gen-jump.c
        config.add_extension('randomgen.%s' % gen,
                        sources=[join('randomgen', x) for x in
                            ['{0}.c'.format(gen), 'src/{0}/{0}.c'.format(gen),
                             'src/{0}/{0}-jump.c'.format(gen)]],
                        include_dirs=[join('randomgen', 'src', gen)],
                        libraries=EXTRA_LIBRARIES,
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS,
                        depends=[join('randomgen', '%s.pyx' % gen)],
                        define_macros=defs,
                        )
    for gen in ['philox', 'threefry', 'threefry32',
                'xoroshiro128', 'xorshift1024', 'xoshiro256starstar',
                'xoshiro512starstar',
                # 'pcg64', 'pcg32',
               ]:
        # gen.pyx, src/gen/gen.c
        config.add_extension('randomgen.%s' % gen,
                        sources=[join('randomgen', x) for x in
                            ['{0}.c'.format(gen), 'src/{0}/{0}.c'.format(gen)]],
                        include_dirs=[join('randomgen', 'src', gen)],
                        libraries=EXTRA_LIBRARIES,
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS,
                        depends=[join('randomgen', '%s.pyx' % gen)],
                        define_macros=defs,
                        )
    for gen in ['common']:
        # gen.pyx
        config.add_extension('randomgen.%s' % gen,
                        sources=[join('randomgen', '{0}.c'.format(gen))],
                        libraries=EXTRA_LIBRARIES,
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS,
                        depends=[join('randomgen', '%s.pyx' % gen)],
                        define_macros=defs,
                        )
    for gen in ['generator', 'bounded_integers']:
        # gen.pyx, src/distributions/distributions.c
        config.add_extension('randomgen.%s' % gen,
                        sources=[join('randomgen', '{0}.c'.format(gen)),
                                 join('randomgen', 'src', 'distributions',
                                      'distributions.c')],
                        libraries=EXTRA_LIBRARIES,
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS,
                        depends=[join('randomgen', '%s.pyx' % gen)],
                        define_macros=defs,
                        )
    return config
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

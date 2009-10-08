#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    import ConfigParser
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    # Read relevant NLPy-specific configuration options.
    nlpy_config = ConfigParser.SafeConfigParser()
    nlpy_config.read(os.path.join(top_path, 'site.cfg'))
    hsl_dir = nlpy_config.get('HSL', 'hsl_dir')
    metis_dir = nlpy_config.get('HSL', 'metis_dir')
    metis_lib = nlpy_config.get('HSL', 'metis_lib')
    galahad_dir = nlpy_config.get('GALAHAD', 'galahad_dir')

    print 'hsl_dir = ', hsl_dir

    config = Configuration('linalg', parent_package, top_path)

    # Get info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        print 'No blas info found'

    #libhsl_dir = '/Users/dpo/local/linalg/hsl/g95'
    #hsllibname = 'hsl_g95' # Must be in LD_LIBRARY_PATH and called .so
    #libmetis_dir = '/Users/dpo/local/linalg/UMFPACK/metis-4.0'
    #metis_lib = 'metis'

    # Relevant files for building MA27 extension.
    ma27_src = ['fd05ad.f', 'id05ad.f', 'ma27ad.f']
    libma27_src = ['ma27_lib.c','ma27fact.f','nlpy_alloc.c']
    pyma27_src = ['_pyma27.c']

    # Relevant files for building MA57 extension.
    ma57_src = ['fd05ad.f', 'ma57ad.f', 'mc47ad.f', 'mc71ad.f', 'fd15ad.f',
                'mc21ad.f', 'mc59ad.f', 'mc34ad.f', 'mc64ad.f']
    libma57_src = ['ma57_lib.c','nlpy_alloc.c'] + ma57_src
    pyma57_src = ['_pyma57.c']

    # Build PyMA27
    ma27_sources  = [os.path.join('src',name) for name in libma27_src]
    ma27_sources += [os.path.join(hsl_dir,name) for name in ma27_src]

    config.add_library(
        name='ma27',
        sources=ma27_sources, #[os.path.join('src',name) for name in libma27_src],
        #libraries=[hsllibname],
        #library_dirs=[libhsl_dir],
        include_dirs=[hsl_dir,'src'],
        extra_info=blas_info,
        )

    config.add_extension(
        name='_pyma27',
        sources=[os.path.join('src',name) for name in pyma27_src],
        depends=[],
        libraries=['ma27'], #[hsllibname,'ma27'],
        #library_dirs=[libhsl_dir],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    # Build PyMA57
    config.add_library(
        name='ma57',
        sources=[os.path.join('src',name) for name in libma57_src],
        libraries=[metis_lib], #[hsllibname,metis_lib],
        library_dirs=[metis_dir],
        include_dirs=[hsl_dir,'src'],
        extra_info=blas_info,
        )

    config.add_extension(
        name='_pyma57',
        sources=[os.path.join('src',name) for name in pyma57_src],
        libraries=[metis_lib,'ma57'], #[hsllibname,metislib,'ma57'],
        library_dirs=[metis_dir],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.add_subpackage('scaling')

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

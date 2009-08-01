#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    config = Configuration('linalg', parent_package, top_path)

    # Get info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        print 'No blas info found'
    #libhsl = get_info('libhsl')
    #if not libhsl:
    #    raise NotFoundError, 'no libhsl resources found'
    #libhsl_dir = libhsl.get('libhsl_dir')
    #hsllibname = libhsl.get('hsllibname')
    libhsl_dir = '/Users/dpo/local/linalg/hsl/g95'
    hsllibname = 'hsl_g95' # Must be in LD_LIBRARY_PATH and called .so
    libmetis_dir = '/Users/dpo/local/linalg/UMFPACK/metis-4.0'
    metislib = 'metis'

    libma27_src = ['ma27_lib.c','ma27fact.f','nlpy_alloc.c']
    libma57_src = ['ma57_lib.c','nlpy_alloc.c']
    pyma27_src = ['_pyma27.c']
    pyma57_src = ['_pyma57.c']

    # Build PyMA27
    config.add_library(
        name='ma27',
        sources=[os.path.join('src',name) for name in libma27_src],
        libraries=[hsllibname],
        library_dirs=[libhsl_dir],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.add_extension(
        name='_pyma27',
        sources=[os.path.join('src',name) for name in pyma27_src],
        depends=[],
        libraries=[hsllibname,'ma27'],
        library_dirs=[libhsl_dir],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    # Build PyMA57
    config.add_library(
        name='ma57',
        sources=[os.path.join('src',name) for name in libma57_src],
        libraries=[hsllibname,metislib],
        library_dirs=[libhsl_dir,libmetis_dir],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.add_extension(
        name='_pyma57',
        sources=[os.path.join('src',name) for name in pyma57_src],
        libraries=[hsllibname,metislib,'ma57'],
        library_dirs=[libhsl_dir,libmetis_dir],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

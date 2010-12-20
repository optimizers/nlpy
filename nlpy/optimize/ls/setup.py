#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError
    import os

    config = Configuration('ls', parent_package, top_path)

    # Get info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        blas_info = get_info('blas',0)
        if not blas_info:
            print 'No blas info found'

    libcsrch_src = ['dcsrch.f', 'dcstep.f']
    pycsrch_src = ['_pycsrch.c']

    libmcsrch_src = ['mcsrch.f', 'mcstep.f']
    pymcsrch_src = ['_pymcsrch.c']

    config.add_library(
        name='nlpy_csrch',
        sources=[os.path.join('src',name) for name in libcsrch_src],
        libraries=[],
        library_dirs=[],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.add_extension(
        name='_pycsrch',
        sources=[os.path.join('src',name) for name in pycsrch_src],
        depends=[],
        libraries=['nlpy_csrch'],
        library_dirs=[],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.add_library(
        name='nlpy_mcsrch',
        sources=[os.path.join('src',name) for name in libmcsrch_src],
        libraries=[],
        library_dirs=[],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.add_extension(
        name='_pymcsrch',
        sources=[os.path.join('src',name) for name in pymcsrch_src],
        depends=[],
        libraries=['nlpy_mcsrch'],
        library_dirs=[],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

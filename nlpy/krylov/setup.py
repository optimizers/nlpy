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
    galahad_dir = nlpy_config.get('GALAHAD', 'galahad_dir')

    config = Configuration('krylov', parent_package, top_path)

    # Get BLAS info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        print 'No blas info found'
    lapack_info = get_info('lapack_opt',0)
    if not lapack_info:
        print 'No lapack info found'

    gdir = os.path.join(galahad_dir,'src')
    libgltr_src = [os.path.join(hsl_dir,'hsl_zd11d.f90'),
                   os.path.join(gdir,'auxiliary','norms.f90'),
                   os.path.join(gdir,'rand','rand.f90'),
                   os.path.join(gdir,'sym','symbols.f90'),
                   os.path.join(gdir,'smt','smt.f90'),
                   os.path.join(gdir,'space','space.f90'),
                   os.path.join(gdir,'spec','specfile.f90'),
                   os.path.join(gdir,'sort','sort.f90'),
                   os.path.join(gdir,'roots','roots.f90'),
                   os.path.join(gdir,'gltr','gltr.f90'),
                   os.path.join('src','pygltr.f90')]
    pygltr_src = ['_pygltr.c']

    # Build PyGLTR
    config.add_library(
        name='nlpy_gltr',
        sources=libgltr_src,
        extra_info=[blas_info, lapack_info],
        )

    config.add_extension(
        name='_pygltr',
        sources=[os.path.join('src',name) for name in pygltr_src],
        libraries=['nlpy_gltr'],
        include_dirs=['src'],
        extra_info=[blas_info, lapack_info],
        )

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

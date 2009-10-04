#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    config = Configuration('krylov', parent_package, top_path)

    # Get info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        print 'No blas info found'
    #libhsl = get_info('libhsl')
    #if not libhsl:
    #    raise NotFoundError, 'no libhsl resources found'
    #libhsl_dir = libhsl.get('libhsl_dir')
    #hsllibname = libhsl.get('hsllibname')
    hsl_src_dir = '/Users/dpo/local/linalg/hsl'
    #galahad = get_info('galahad')
    #galahad_dir = galahad.get('galahad_dir')
    galahad_dir = '/Users/dpo/local/dev/Galahad/galahad'

    gdir = os.path.join(galahad_dir,'src')
    libgltr_src = [os.path.join(hsl_src_dir,'hsl_zd11d.f90'),
                   os.path.join(gdir,os.path.join('auxiliary','norms.f90')),
                   os.path.join(gdir,os.path.join('rand','rand.f90')),
                   os.path.join(gdir,os.path.join('roots','roots.f90')),
                   os.path.join(gdir,os.path.join('sym','symbols.f90')),
                   os.path.join(gdir,os.path.join('smt','smt.f90')),
                   os.path.join(gdir,os.path.join('space','space.f90')),
                   os.path.join(gdir,os.path.join('spec','specfile.f90')),
                   os.path.join(gdir,os.path.join('gltr','gltr.f90')),
                   os.path.join('src','pygltr.f90')]
    pygltr_src = ['_pygltr.c']

    # Build PyGLTR
    config.add_library(
        name='gltr',
        sources=libgltr_src,
        extra_info=blas_info,
        )

    config.add_extension(
        name='_pygltr',
        sources=[os.path.join('src',name) for name in pygltr_src],
        libraries=['gltr'],
        include_dirs=['src'],
        extra_info=blas_info,
        )

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

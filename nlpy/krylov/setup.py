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

    config = Configuration('krylov', parent_package, top_path)

    # Get BLAS info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        print 'No blas info found'
    lapack_info = get_info('lapack_opt',0)
    if not lapack_info:
        print 'No lapack info found'

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

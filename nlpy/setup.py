#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    from numpy.distutils.misc_util import Configuration

    config = Configuration('nlpy', parent_package, top_path)

    # Create cache directory if necessary.
    cache_dir = os.path.join(top_path,'cache')
    if not os.access(cache_dir, os.F_OK):
        os.mkdir(cache_dir)

    config.add_subpackage('model')
    config.add_subpackage('linalg')
    config.add_subpackage('krylov')
    config.add_subpackage('precon')
    config.add_subpackage('optimize')
    config.add_subpackage('tools')
    #config.add_data_dir('tests')

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

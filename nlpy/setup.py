#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('nlpy', parent_package, top_path)

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

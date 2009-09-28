#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('solvers', parent_package, top_path)

    config.add_scripts(['nlpy_trunk.py',
                        'nlpy_lbfgs.py',
                        'nlpy_reglp.py',
                        'nlpy_regqp.py'])

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

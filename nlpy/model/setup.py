#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    import ConfigParser
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError



    config = Configuration('model', parent_package, top_path)

    libampl_dir = '/Users/dpo/local/dev/libampl'
    libampl_libdir = os.path.join(libampl_dir, 'Lib')
    libampl_include = os.path.join(libampl_dir, os.path.join('Src','solvers'))

    config.add_extension(
        name='_amplpy',
        sources=['src/_amplpy.c'],
        libraries=['ampl', 'funcadd0'],
        library_dirs=[libampl_libdir],
        include_dirs=['src', libampl_include],
        extra_link_args=[]
        )

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

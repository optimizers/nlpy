#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    import ConfigParser
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    from numpy.distutils.core import Extension

    # Read relevant NLPy-specific configuration options.
    nlpy_config = ConfigParser.SafeConfigParser()
    nlpy_config.read(os.path.join(top_path, 'site.cfg'))
    libampl_dir = nlpy_config.get('ASL', 'asl_dir')

    config = Configuration('model', parent_package, top_path)

    libampl_libdir = os.path.join(libampl_dir, 'lib')
    libampl_include = os.path.join(libampl_dir, os.path.join('include','asl'))

    amplpy_src = [os.path.join('src','_amplpy.c'),
                  os.path.join('src','amplutils.c')]

    config.add_extension(
        name='_amplpy',
        sources=amplpy_src,
        libraries=['asl'],
        library_dirs=[libampl_libdir],
        include_dirs=['src', libampl_include],
        )

    config.add_data_files(os.path.join('tests', 'rosenbrock.mod'),
                          os.path.join('tests', 'hs007.mod'))

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

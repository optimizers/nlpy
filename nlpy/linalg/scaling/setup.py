#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('scaling', parent_package, top_path)

    #libhsl = get_info('libhsl')
    #if not libhsl:
    #    raise NotFoundError, 'no libhsl resources found'
    #libhsl_dir = libhsl.get('libhsl_dir')
    #hsllibname = libhsl.get('hsllibname')
    libhsl_dir = '/Users/dpo/local/linalg/hsl/g95'
    hsllibname = 'hsl_g95' # Must be in LD_LIBRARY_PATH and called .so

    config.add_extension(
        name='mc29module',
        sources=['src/mc29.pyf'],
        libraries=[hsllibname],
        library_dirs=[libhsl_dir],
        include_dirs=['src'],
        extra_link_args=[])

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

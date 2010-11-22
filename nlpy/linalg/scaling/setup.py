#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    import ConfigParser
    import os
    from numpy.distutils.misc_util import Configuration

    # Read relevant NLPy-specific configuration options.
    nlpy_config = ConfigParser.SafeConfigParser()
    nlpy_config.read(os.path.join(top_path, 'site.cfg'))
    hsl_dir = nlpy_config.get('HSL', 'hsl_dir')

    config = Configuration('scaling', parent_package, top_path)

    mc29_sources = [os.path.join(hsl_dir,'mc29d','mc29d.f'),
                    os.path.join('src','mc29.pyf')]

    config.add_extension(
        name='mc29module',
        sources=mc29_sources,
        include_dirs=['src'],
        extra_link_args=[])

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

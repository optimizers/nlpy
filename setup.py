#!/usr/bin/env python
"""
NLPy: A Nonlinear Programming Environment in Python

NLPy is a programming environment that facilitates construction of optimization
algorithms by supplying a library of efficient building blocks.
D. Orban <dominique.orban@gerad.ca>
"""

DOCLINES = __doc__.split("\n")

import os
import sys

try:
    import setuptools   # To enable 'python setup.py develop'
except:
    pass

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: LGPL
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def configuration(parent_package='',top_path=None):
    import numpy
    import pysparse
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_include_dirs([numpy.get_include(), pysparse.get_include()])
    config.add_include_dirs('include')
    config.add_subpackage('nlpy')

    # Set config.version
    config.get_version(os.path.join('nlpy','version.py'))

    return config

def setup_package():

    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration
    from Cython.Distutils import build_ext

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)
    sys.path.insert(0,os.path.join(local_path,'nlpy')) # to retrieve version

    try:
        setup(
            name = 'nlpy',
            author = "Dominique Orban",
            author_email = "dominique.orban@gerad.ca",
            maintainer = "NLPy Developers",
            maintainer_email = "dominique.orban@gerad.ca",
            description = DOCLINES[0],
            long_description = "\n".join(DOCLINES[2:]),
            url = "",
            download_url = "",
            license = 'LGPL',
            classifiers=filter(None, CLASSIFIERS.split('\n')),
            platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
            configuration=configuration,
            cmdclass = {'build_ext': build_ext},
            )
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()

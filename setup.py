#!/usr/bin/env python
"""
NLPy: A Nonlinear Programming Environment in Python

NLPy is a programming environment that facilitates construction of optimization
algorithms by supplying a library of efficient building blocks.
D. Orban <dominique.orban@gerad.ca>
"""

DOCLINES = __doc__.split("\n")

#import setuptools   # To enable 'python setup.py develop'
import os
import sys

try:
    #import setuptools   # To enable 'python setup.py develop'
    pass
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


def fullsplit(path, result=None):
    """
    Split a pathname into components (the opposite of os.path.join) in a
    platform-neutral way.
    """
    if result is None:
        result = []
    head, tail = os.path.split(path)
    if head == '':
        return [tail] + result
    if head == path:
        return result
    return fullsplit(head, [tail] + result)
    

def setup_package():

    from numpy.distutils.core import setup, Extension
    from numpy.distutils.misc_util import Configuration

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)
    sys.path.insert(0,os.path.join(local_path,'nlpy')) # to retrieve version
    
    
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_path = local_path

    # Run build
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)
    
    # find all files that should be included
    packages, data_files = [], []
    for dirpath, dirnames, filenames in os.walk('nlpy'):
        # Ignore dirnames that start with '.'
        for i, dirname in enumerate(dirnames):
            if dirname.startswith('.'): del dirnames[i]
        if '__init__.py' in filenames:
            packages.append('.'.join(fullsplit(dirpath)))
        elif filenames:
            data_files.append([dirpath, [os.path.join(dirpath, f) for f in filenames]])

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
            packages = packages,
            classifiers=filter(None, CLASSIFIERS.split('\n')),
            platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
            configuration=configuration,
            )
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()

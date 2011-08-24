#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    import ConfigParser
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    # Imports for Cython extensions.
    #from distutils.extension import Extension
    from numpy.distutils.core import Extension
    #from Cython.Build import cythonize

    # Read relevant NLPy-specific configuration options.
    nlpy_config = ConfigParser.SafeConfigParser()
    nlpy_config.read(os.path.join(top_path, 'site.cfg'))
    libampl_dir = nlpy_config.get('LIBAMPL', 'libampl_dir')

    config = Configuration('model', parent_package, top_path)

    libampl_libdir = os.path.join(libampl_dir, 'Lib')
    libampl_include = os.path.join(libampl_dir, os.path.join('Src','solvers'))

### Code for auto Cythonize. Not working!
    ## # Cythonize AMPL module.
    ## amplpy_src = [os.path.join(top_path,'nlpy','model','src','_amplpy.pyx'),
    ##                os.path.join(top_path,'nlpy','model','src','amplutils.c')]
    ## amplpy_extension = Extension('_amplpy',
    ##                                amplpy_src,
    ##                                #libraries=['ampl','funcadd0'],
    ##                                #library_dirs=[libampl_libdir],
    ##                                include_dirs=['src',libampl_include],
    ##                               )
    ## print 'amplpy2_extension: ', amplpy_extension
    ## amplpy_cython_extension = cythonize(amplpy_extension)
    ## print 'top_path = ', top_path
    ## print 'Cython extension: ', amplpy_cython_extension

    ## # Add extension to extension list.
    ## for ext in amplpy_cython_extension:
    ##     config.ext_modules.append(ext)

    #config.add_extension(
    #    name='_amplpy2',
    #    sources=amplpy2_src,
    #    libraries=['ampl','funcadd0'],
    #    library_dirs=[libampl_libdir],
    #    include_dirs=['src', libampl_include],
    #    )

#   amplpy_src = [os.path.join('src','_amplpy.pyx'),
    amplpy_src = [os.path.join('src','_amplpy.c'),
                  os.path.join('src','amplutils.c')]
    
    config.add_extension(
        name='_amplpy',
        sources=amplpy_src,
        libraries=['ampl','funcadd0'],
        library_dirs=[libampl_libdir],
        include_dirs=['src', libampl_include],
        )

    ## config.add_extension(
    ##     name='_amplpy',
    ##     sources=amplpy_src,
    ##     libraries=['ampl', 'funcadd0'],
    ##     library_dirs=[libampl_libdir],
    ##     include_dirs=['src', libampl_include],
    ##     extra_link_args=[]
    ##     )

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

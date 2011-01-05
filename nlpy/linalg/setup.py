#!/usr/bin/env python
from urllib import urlretrieve
from tempfile import mkdtemp
import gzip, tarfile
import os, re

def tarzxf(archive):
    """
    This (oddly) named function performs the same tas as the ``tar zxf``
    command, i.e., uncompress and extract a compressed tar archive all
    at once. The uncompressed archive can subsequently be found in the
    newly-created directory named ``archive``, where ``archive.tar.gz``
    is the name of the original compressed tar archive.
    """
    archivetar_name = archive + '.tar'
    archivetargz_name = archivetar_name + '.gz'

    # Uncompress into regular tar archive.
    archivetargz = gzip.GzipFile(archivetargz_name, 'rb')
    archivetar = open(archivetar_name, mode='wb')
    for line in archivetargz:
        archivetar.write(line)
    archivetar.close()
    archivetargz.close()

    # Extract tar archive.
    archivetar = tarfile.open(archivetar_name)
    archivetar.extractall(path=archive)
    archivetar.close()

    return


def getoption(config, section, option):
    try:
        val = config.get(section,option)
    except:
        val = None
    return val


def configuration(parent_package='',top_path=None):
    import numpy
    import os
    import ConfigParser
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    # For debugging f2py extensions:
    f2py_options = []
    #f2py_options.append('--debug-capi')

    # Read relevant NLPy-specific configuration options.
    nlpy_config = ConfigParser.SafeConfigParser()
    nlpy_config.read(os.path.join(top_path, 'site.cfg'))
    hsl_dir = getoption(nlpy_config, 'HSL', 'hsl_dir')
    metis_dir = getoption(nlpy_config, 'HSL', 'metis_dir')
    metis_lib = getoption(nlpy_config, 'HSL', 'metis_lib')
    propack_dir = getoption(nlpy_config, 'PROPACK', 'propack_dir')

    config = Configuration('linalg', parent_package, top_path)
    cache_dir = os.path.join(top_path, 'cache')

    # Get info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        blas_info = get_info('blas',0)
        if not blas_info:
            print 'No blas info found'

    lapack_info = get_info('lapack_opt',0)
    if not lapack_info:
        lapack_info = get_info('lapack',0)
        if not lapack_info:
            print 'No lapack info found'

    if hsl_dir is not None:
        # Relevant files for building MA27 extension.
        ma27_src = ['fd05ad.f', 'ma27ad.f']
        libma27_src = ['ma27fact.f']
        pyma27_src = ['ma27_lib.c','nlpy_alloc.c','_pyma27.c']

        # Relevant files for building MA57 extension.
        ma57_src = ['ddeps.f', 'ma57d.f']
        pyma57_src = ['ma57_lib.c','nlpy_alloc.c','_pyma57.c']

        # Build PyMA27.
        ma27_sources  = [os.path.join(hsl_dir,name) for name in ma27_src]
        ma27_sources += [os.path.join('src',name) for name in libma27_src]

        config.add_library(
            name='nlpy_ma27',
            sources=ma27_sources,
            include_dirs=[hsl_dir,'src'],
            extra_info=blas_info,
            )

        config.add_extension(
            name='_pyma27',
            sources=[os.path.join('src',name) for name in pyma27_src],
            depends=[],
            libraries=['nlpy_ma27'],
            include_dirs=['src'],
            extra_info=blas_info,
            )

        # Prepare to build PyMA57.
        ma57_sources = [os.path.join(hsl_dir,'ma57d',name) for name in ma57_src]
        pyma57_sources = [os.path.join('src',name) for name in pyma57_src]

        # See if source files are present.
        build57 = True
        for src_file in ma57_sources:
            if not os.access(src_file, os.F_OK):
                build57 = False
                break

        if build57:

            if metis_dir is None or metis_lib is None:

                # Fetch and build METIS.
                libmetis_name = 'metis'
                src = 'http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-4.0.tar.gz'
                tmpdir = cache_dir #mkdtemp()
                localcopy = os.path.join(tmpdir, libmetis_name)

                # Fetch, uncompress and extract compressed tar archive.
                if not os.access(localcopy, os.F_OK):
                    print 'Downloading METIS'
                    urlretrieve(src, filename=localcopy + '.tar.gz')

                print 'Unarchiving METIS'
                tarzxf(localcopy)
                localcopy = os.path.join(localcopy, 'metis-4.0', 'Lib')

                # Change to unarchived directory and build header files.
                cwd = os.getcwd()
                print 'Changing to %s to build' % localcopy
                os.chdir(localcopy)

                # Read contents of Makefile.
                print 'Reading Makefile'
                fp = open('Makefile', 'r')
                makefile = fp.read()
                fp.close()
                os.chdir(cwd)

                # Extract list of source files.
                print 'Extracting source list'
                res = re.search(r'\nOBJS = ', makefile)
                k0 = k = res.start(0) + 1
                while makefile[k:k+2] != '\n\n':
                    k += 1
                lst = makefile[k0:k]
                lst = re.sub(r'[\\\n\t]', '', lst)  # Remove escape characters.
                lst = re.sub('\.o', '.c', lst)      # Change .o in .c.
                lst = lst[7:]                       # Remove 'OBJS = '.
                src_lst = lst.split()
                metis_sources = [os.path.join(localcopy,f) for f in src_lst]
                libmetis_include = localcopy
                metis_dir = ''
                metis_lib = 'metis'

                # Build METIS.
                config.add_library(
                    name=metis_lib,
                    sources=metis_sources,
                    include_dirs=[libmetis_include],
                )

            # Build PyMA57.
            config.add_library(
                name='nlpy_ma57',
                sources=ma57_sources,
                libraries=[metis_lib],
                library_dirs=[metis_dir],
                include_dirs=[hsl_dir,'src'],
                extra_info=blas_info,
            )

            config.add_extension(
                name='_pyma57',
                sources=pyma57_sources,
                libraries=['nlpy_ma57'],
                #libraries=[metis_lib,'nlpy_ma57'],
                #library_dirs=[metis_dir],
                include_dirs=['src'],
                extra_info=blas_info,
            )

    if propack_dir is not None:
        propack_src = ['dlanbpro.F', 'dreorth.F', 'dgetu0.F', 'dsafescal.F',
                       'dblasext.F', 'dlansvd.F', 'printstat.F', 'dgemm_ovwr.F',
                       'dlansvd_irl.F', 'dbsvd.F', 'dritzvec.F', 'dmgs.risc.F',
                       'second.F']

        propack_sources = [os.path.join(propack_dir, 'double', f) for f in propack_src]
        pypropack_sources = [os.path.join('src', 'propack.pyf')]

        config.add_library(
            name='nlpy_propack',
            sources=propack_sources,
            include_dirs=os.path.join(propack_dir, 'double'),
            extra_info=[blas_info, lapack_info],
            )

        config.add_extension(
            name='_pypropack',
            sources=pypropack_sources,
            libraries=['nlpy_propack'],
            extra_info=[blas_info, lapack_info],
            f2py_options=f2py_options,
        )

        config.add_subpackage('scaling')

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

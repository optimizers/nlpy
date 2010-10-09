====
NLPy
====

Welcome to the NLPy optimization toolkit for Python. If you read documentation,
the next section is for you. Otherwise, skip to the `Installation` section
below.


Documentation
-------------

The NLPy documentation is typeset using the Sphinx documentation system
`http://sphinx.pocoo.org`_ which is particularly suited to documenting Python
packages.

The manual in PDF format is in the `doc` subfolder. The HTML documentation is
accessible by pointing your browser to `doc/build/contents.html`.

Check the paper `NLPy---A Large-Scale Optimization Toolkit in Python` by
Dominique Orban available from `http://www.gerad.ca`_. 

If the user so desires, the documentation may be generated afresh as follows:

- To re-generate the HTML documentation, change to the `doc` subfolder and
  type::

        make html

- To re-generate the PDF documentation, change to the `doc` subfolder and
  type::

        make latex

  Then change to the `build/latex` subfolder and type::

       make all-pdf


Installation
------------

For now, just::

    cp site.template.cfg site.cfg
    # Edit site.cfg to adjust to your local settings.
    python setup.py build
    python setup.py install [--prefix=...]

To select another C and/or Fortran compiler::

    python setup.py config_fc --ccompiler=<name> --fcompiler=<name> build

To see a list of available Fortran compilers and their names::

    python setup.py config_fc --help-fcompiler

To see a list of available C compilers and their names::

    python setup.py config_fc --help-ccompiler

For example, you can force compilation with gfortran by specifying::

    --fcompiler=gnu95

and wit g95 by specifiying::

    --fcompiler=g95

(note the subtle name difference.)


Troubleshooting
---------------

-  On an OpenSuSE system I obtain the following error message with both
   gfortran and g95::

      ld: build/temp.linux-x86_64-2.6/libnlpy_ma27.a(ma27ad.o): relocation
      R_X86_64_32S against 'a local symbol' can not be used when making a shared
      object; recompile with -fPIC build/temp.linux-x86_64-2.6/libnlpy_ma27.a:
      could not read symbols: Bad value

   The source of the problem is that on this platform, distutils appears to be
   compiling a static library without the `-fPIC` flags, which generates
   relocatable objects. When subsequently linking against this library to build
   a shared object, the linker complains. This problem can be resolved by
   restarting the installation and adding the following flags to the `python
   setup.py` command line::

      --f77flags="-ffixed-form -fno-second-underscore -fPIC -O"
      --f90flags="-fno-second-underscore -fPIC -O"

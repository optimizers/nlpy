Installation
------------

For now, just::

    python setup.py build
    python setup.py install

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

(note the subtle difference.)


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
   relocatable objects. When subsequently linking against this library, the
   linker complains. This problem can be resolved by restarting the
   installation and adding the following flags to the `python setup.py`
   command line::

      --f77flags="-ffixed-form -fno-second-underscore -fPIC -O"
      --f90flags="-fno-second-underscore -fPIC -O"

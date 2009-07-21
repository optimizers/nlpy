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

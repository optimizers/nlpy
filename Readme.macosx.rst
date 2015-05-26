=====================================
Build instructions for Mac OS/X users
=====================================

Installing NLPy will be much easier if you use Homebrew
(http://brew.sh). Follow the instructions to install Homebrew.
Then, the following dependencies can be installed automatically in /usr/local::

    brew install gcc  # currently v4.9. Contains gfortran

    brew tap homebrew/science
    brew install adol-c                             # will also install Colpack
    brew install boost --with-mpi --without-single  # to use pycppad
    brew install cppad --with-adol-c --with-boost --cc=gcc-4.9
    brew install asl
    brew install metis

    pip install algopy
    pip install git+https://github.com/b45ch1/pycppad.git

Installing PyAdolc
------------------

    git clone https://github.com/b45ch1/pyadolc.git
    cd pyadolc
    cp setup.py.EXAMPLE setup.py
    $(EDITOR) setup.py

Modify ``setup.py`` so it looks like this:

    extra_compile_args = ['-ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB']
    include_dirs = [get_numpy_include_dirs()[0],'/usr/local/opt/adol-c/include','/usr/local/opt/include']
    library_dirs = ['/usr/local/opt/adol-c/lib','/usr/local/opt/colpack/lib']
    libraries = ['boost_python','adolc', 'ColPack']

Then install and test::

    python setup.py install
    cd
    python -c "import adolc; adolc.test()"


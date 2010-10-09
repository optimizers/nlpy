Tuning the METIS Makefile for OSX
=================================

The directions below should be easily adapted to Linux and other variants of
Unix.

Firstly, the top Makefile.in could look something like the following on OSX::

    # Which compiler to use
    CC = gcc
    
    # What optimization level to use
    OPTFLAGS = -O2 
    
    # What options to be used by the compiler
    COPTIONS = -fPIC
    
    # What options to be used by the loader
    LDOPTIONS =
    
    # What archiving to use
    AR = libtool -static -s -o
    
    # What to use for indexing the archive
    # Already done by '-s' flag to libtool.
    RANLIB = exec true

The changes from the original `Makefile.in` are as follows. Firstly, `gcc` is
selected as default compiler. If you have gotten this far, you surely have
XCode installed on your Mac and therefore you have `gcc`. Feel free to select a
different compiler, but make sure you remain consistent. Secondly, I added
`-fPIC` to the compiler flags. This is because the next topic will deal with
building a dynamic METIS library. Third, I changed the value of `AR` since
archiving is performed by `libtool` on OSX. The `-s` flag to `libtool` takes
care of creating an appropriate table of contents for the archive, which is why
we do not need to run `ranlib`. In fact, the rumor is that `ranlib` is bound to
disappear on OSX.


Building a Dynamic METIS Library
================================

Because we added the `-fPIC` compiler flag to `Makefile.in`, all objects have
been compiled as position independent and can thus participate in a dynamic
library. Add the following bit at the bottom of the top-level `Makefile`::

    # Create a dynamic library.
    libmetis.dylib: default
        libtool -dynamic -o $@ -lstdc++ -lm Lib/*.o

Note that the last line should begin with a <TAB> character, not just spaces.
You may need to change `-lstdc++` to something like `-lstdc++.6` depending on
what is in your `/usr/lib`. Do a `ls -l /usr/lib/libstdc++*.dylib` and see what
comes up. All you have to do next is type `make libmetis.dylib` in the
top-level directory of the METIS tree. NLPy will happily select the dynamic
library if it is present. If not, it will choose the static library.


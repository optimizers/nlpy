TROUBLESHOOTING
===============

1. On some Linux machines, compilation of METIS results in the error::

     .../nlpy/cache/metis/metis-4.0/Lib/proto.h:462:
     error: conflicting types for ‘__log2’
     /usr/include/bits/mathcalls.h:145: note: previous declaration of ‘__log2’ was here
     In file included from .../nlpy/cache/metis/metis-4.0/Lib/metis.h:36,
                      from .../nlpy/cache/metis/metis-4.0/Lib/mutil.c:14:
     .../nlpy/cache/metis/metis-4.0/Lib/proto.h:458:
     warning: function declaration isn’t a prototype
     .../nlpy/cache/metis/metis-4.0/Lib/proto.h:462:
     error: conflicting types for ‘__log2’
     /usr/include/bits/mathcalls.h:145: note: previous declaration of ‘__log2’ was here

   The solution is to apply the patch located in nlpy/extras/metis-4.0.patch to
   the METIS source tree. Change to the location of METIS (which may be in the
   `cache` directory if you elected to have NLPy download and build METIS for
   you) and type::

     patch -p1 < path/to/nlpy/extras/metis-4.0.patch

   Then rebuild::

     cd /path/to/nlpy
     rm -rf build
     python setup.py <options> build

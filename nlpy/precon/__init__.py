"""
A Module for Generic Preconditioners.
"""

from precon import *
try:
    from pycfs  import *
except:
    pass

__all__ = filter(lambda s:not s.startswith('_'), dir())

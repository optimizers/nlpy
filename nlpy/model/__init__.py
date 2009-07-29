"""
Python interface to the AMPL Solver Library.
"""

from nlp    import *
from amplpy import *

__all__ = filter(lambda s:not s.startswith('_'), dir())

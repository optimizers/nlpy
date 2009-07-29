"""
Python interface to the AMPL Solver Library.
"""

from amplpy import *

__all__ = filter(lambda s:not s.startswith('_'), dir())

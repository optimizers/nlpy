#!/usr/bin/env python
"""Simple AMPL driver for the derivative checker."""

from nlpy.model import PySparseAmplModel
from nlpy.tools import DerivativeChecker

import logging
import sys

if len(sys.argv) == 1:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
log = logging.getLogger('nlpy.der')
level = logging.INFO
log.setLevel(level)
fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

nlp = PySparseAmplModel(sys.argv[1])
dcheck = DerivativeChecker(nlp, nlp.x0)
dcheck.check()

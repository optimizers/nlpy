#!/usr/bin/env python

from nlpy.tool.pprof import parse_cmdline, PerformanceProfile
import sys

# Usage from the command line
(optlist, solvers) = parse_cmdline(sys.argv[1:])    
profile = PerformanceProfile(solvers, **optlist)
profile.show()

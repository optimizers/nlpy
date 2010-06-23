"NLPy: Nonlinear Programming in Python"

__docformat__ = 'restructuredtext'

# Imports
from numpy._import_tools import PackageLoader
from version import version as __version__

pkgload = PackageLoader()
pkgload(verbose=False,postpone=True)

if __doc__:
    __doc__ += """

Available subpackages
---------------------
"""
if __doc__:
    __doc__ += pkgload.get_pkgdocs()

__all__ = filter(lambda s: not s.startswith('_'), dir())
__all__ += '__version__'

__doc__ += """

Miscellaneous
-------------

    __version__  :  NLPy version string
"""

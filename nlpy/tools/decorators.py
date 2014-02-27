import warnings
import functools
import hashlib
import numpy as np


def deprecated(func):
  """
  This decorator can be used to mark functions as deprecated.
  It will emit a warning when the function is called.

  From http://wiki.python.org/moin/PythonDecoratorLibrary
  """

  @functools.wraps(func)
  def new_func(*args, **kwargs):
      warnings.warn_explicit(
          "Call to deprecated function/method {}.".format(func.__name__),
          category=DeprecationWarning,
          filename=func.func_code.co_filename,
          lineno=func.func_code.co_firstlineno + 1
      )
      return func(*args, **kwargs)
  return new_func


def counter(func):
  """
  This decorator counts the number of calls to the wrapped
  function or method.
  """

  @functools.wraps(func)
  def _counted(*args, **kwargs):
      _counted.ncalls += 1
      func(*args, **kwargs)
  _counted.ncalls = 0
  return _counted


def get_signature(x):
    """
    Return signature of argument.
    The signature is the value of the argument or the sha1 digest if the
    argument is a numpy array.
    """
    if isinstance(x, np.ndarray):
        _x = x.view(np.uint8)
        return hashlib.sha1(_x).hexdigest()
    return x


def memoize_full(fcn):
    """
    Decorator used to cache the all values of a function or method
    based on the sha1 signature of its arguments. If any single argument
    changes, the function or method is evaluated afresh.
    """
    _cache = {}

    @functools.wraps(fcn)
    def _memoized_fcn(*args, **kwargs):
        args_signature = tuple(map(get_signature,
                                   list(args) + kwargs.values()))
        if args_signature not in _cache:
            _cache[args_signature] = fcn(*args, **kwargs)
        print _cache
        return _cache[args_signature]

    return _memoized_fcn


def memoize_cheap(fcn):
    """
    Decorator used to cache the most recent value of a function or method
    based on the sha1 signature of its arguments. If any single argument
    changes, the function or method is evaluated afresh.
    """
    _cached_signature = [None]  # Must be mutable.
    _cached_value = [None]

    @functools.wraps(fcn)
    def _memoized_fcn(*args, **kwargs):
        args_signature = map(get_signature, list(args) + kwargs.values())
        if args_signature != _cached_signature[0]:
            _cached_signature[0] = args_signature
            _cached_value[0] = fcn(*args, **kwargs)
        return _cached_value[0]

    return _memoized_fcn

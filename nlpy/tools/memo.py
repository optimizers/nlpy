import functools
import hashlib
import numpy as np

class Memoized(object):
    """
    Decorator class used to cache the most recent value of a function or method
    based on the signature of its arguments. If any single argument changes,
    the function or method is evaluated afresh.
    """

    def __init__(self, callable):
        """
        The only argument, `callable`, can be a plain function or a class
        method.
        """
        self._callable = callable
        self._callable_is_method = False
        self.value = None            # Cached value or derivative.
        self._args_signatures = {}
        return


    def __get_signature(self, x):
        """
        Return signature of argument. The signature is the value of the
        argument or the sha1 digest if the argument is a numpy array.
        Subclass to implement other digests.
        """
        if isinstance(x, np.ndarray):
            _x = x.view(np.uint8)
            return hashlib.sha1(_x).hexdigest()
        return x


    def __call__(self, *args, **kwargs):
        # The method will be called if any single argument is new or changed.

        callable = self._callable
        evaluate = False

        # If we're memoizing a class method, the first argument will be 'self'
        # and need not be memoized.
        firstarg = 1 if self._callable_is_method else 0

        print 'Before: ', self._args_signatures

        # Get signature of all arguments.
        nargs = callable.func_code.co_argcount   # Positional arguments only.
        argnames = callable.func_code.co_varnames[firstarg:nargs]
        argvals = args[firstarg:]

        for (argname,argval) in zip(argnames,argvals) + kwargs.items():

            _arg_signature = self.__get_signature(argval)
            print '%s = %s' % (argname, _arg_signature)

            try:
                cached_arg_sig = self._args_signatures[argname]
                if cached_arg_sig != _arg_signature:
                    self._args_signatures[argname] = _arg_signature
                    evaluate = True

            except KeyError:
                self._args_signatures[argname] = _arg_signature
                evaluate = True

        print 'After: ', self._args_signatures
        print 'Verdict: evaluate = ', evaluate

        # If all arguments are unchanged, return cached value.
        if evaluate:
            print 'Calling method with:'
            print '  args = ', args
            print '  kwargs = ', kwargs
            self.value = callable(*args, **kwargs)

        return self.value

    def __get__(self, obj, objtype):
        "Support instance methods."
        self._callable_is_method = True
        return functools.partial(self.__call__, obj)


    def __repr__(self):
        "Return the wrapped function or method's docstring."
        return self.method.__doc__

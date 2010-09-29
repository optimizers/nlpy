import numpy as np

import logging
import logging.handlers

__docformat__ = 'restructuredtext'

class LinearOperator:
    """
    A linear operator is a linear mapping x -> A(x) such that the size of the
    input vector x is `nargin` and the size of the output is `nargout`. It can
    be visualized as a matrix of shape (`nargout`, `nargin`).
    """

    def __init__(self, nargin, nargout, **kwargs):
        self.nargin = nargin
        self.nargout = nargout
        self.shape = (nargout, nargin)

        # Log activity.
        self.log = kwargs.get('log', False)
        if self.log:
            self.logger = logging.getLogger('LINOP')
            self.logger.setLevel(logging.DEBUG)
            ch = logging.handlers.RotatingFileHandler('linop.log')
            ch.setLevel(logging.DEBUG)
            fmtr = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s %(message)s')
            ch.setFormatter(fmtr)
            self.logger.addHandler(ch)
            self.logger.info('New linear operator with shape ' + str(self.shape))
        else:
            self.logger = None

        return


    def get_shape(self):
        return self.shape


    def __call__(self, *args, **kwargs):
        # An alias for __mul__.
        return self.__mul__(*args, **kwargs)


    def __mul__(self, x):
        raise NotImplementedError, 'Please subclass to implement __mul__.'



class SimpleLinearOperator(LinearOperator):
    """
    A linear operator constructed from a matvec and (possibly) a matvec_transp
    function.
    """

    def __init__(self, nargin, nargout, matvec,
                 matvec_transp=None, symmetric=False, **kwargs):
        LinearOperator.__init__(self, nargin, nargout, **kwargs)
        self.symmetric = symmetric
        self.transposed = kwargs.get('transposed', False)
        transpose_of = kwargs.get('transpose_of', None)

        self.__mul__ = matvec

        if symmetric:
            self.T = self
        else:
            if transpose_of is None:
                if matvec_transp is not None:
                    # Create 'pointer' to transpose operator.
                    self.T = SimpleLinearOperator(nargout, nargin,
                                                  matvec_transp,
                                                  matvec_transp=matvec,
                                                  transposed=not self.transposed,
                                                  transpose_of=self,
                                                  log=self.log)
                else:
                    self.T = None
            else:
                # Use operator supplied as transpose operator.
                if isinstance(transpose_of, LinearOperator):
                    self.T = transpose_of
                else:
                    msg = 'kwarg transposed_of must be a LinearOperator.'
                    msg += ' Got ' + str(transpose_of.__class__)
                    raise ValueError, msg



class PysparseLinearOperator(LinearOperator):
    """
    A linear operator constructed from any object implementing either `__mul__`
    or `matvec` and either `__rmul__` or `matvec_transp`, such as a `ll_mat`
    object or a `PysparseMatrix` object.
    """

    def __init__(self, A, symmetric=False, **kwargs):
        m, n = A.shape
        self.A = A
        self.symmetric = symmetric
        self.transposed = kwargs.get('transposed', False)
        transpose_of = kwargs.get('transpose_of', None)

        if self.transposed:

            LinearOperator.__init__(self, m, n, **kwargs)
            #self.shape = (self.nargin, self.nargout)
            if hasattr(A, '__rmul__'):
                self.__mul__ = A.__rmul__
            else:
                self.__mul__ = self._rmul

        else:

            LinearOperator.__init__(self, n, m, **kwargs)
            if hasattr(A, '__mul__'):
                self.__mul__ = A.__mul__
            else:
                self.__mul__ = self._mul

        if self.log:
            self.logger.info('New linop has transposed = ' + str(self.transposed))

        if symmetric:
            self.T = self
        else:
            if transpose_of is None:
                # Create 'pointer' to transpose operator.
                self.T = PysparseLinearOperator(self.A,
                                                transposed=not self.transposed,
                                                transpose_of=self,
                                                log=self.log)
            else:
                # Use operator supplied as transpose operator.
                if isinstance(transpose_of, LinearOperator):
                    self.T = transpose_of
                else:
                    msg = 'kwarg transposed_of must be a LinearOperator.'
                    msg += ' Got ' + str(transpose_of.__class__)
                    raise ValueError, msg

        return


    def _mul(self, x):
        # Make provision for the case where A does not implement __mul__.
        if x.shape != (self.nargin,):
            msg = 'Input has shape ' + str(x.shape)
            msg += ' instead of (%d,)' % self.nargin
            raise ValueError, msg
        Ax = np.empty(self.nargout)
        self.A.matvec(x, Ax)
        return Ax


    def _rmul(self, y):
        # Make provision for the case where A does not implement __rmul__.
        # This method is only relevant when transposed=True.
        if y.shape != (self.nargin,):  # This is the transposed op's nargout!
            msg = 'Input has shape ' + str(y.shape)
            msg += ' instead of (%d,)' % self.nargin
            raise ValueError, msg
        ATy = np.empty(self.nargout)   # This is the transposed op's nargin!
        self.A.matvec_transp(y, ATy)
        return ATy

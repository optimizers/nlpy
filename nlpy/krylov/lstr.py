# LSTR: Solve a linear least-squares problem with elliptical trust-region
# constraint using the truncated conjugate-gradient method.

from nlpy.krylov.linop import PysparseLinearOperator, SquaredLinearOperator
from nlpy.krylov.pcg import TruncatedCG

__docformat__ = 'restructuredtext'

class LSTRFramework(TruncatedCG):
    """
    Solve the constrained linear least-squares problem

    |      minimize{in n} |c + J n|  subject to  |n| <= radius,

    where J is a matrix and |.| is the Euclidian norm. The method consists in
    using the equivalent formulation

    |      minimize (J'c)'n + 1/2 n' (J'J) n  subject to  |n| <= radius

    and in applying the trucated conjugate gradient method.
    """

    def __init__(self, J, c, radius=None, transposed=False, **kwargs):
        """
        :parameters:
            :J: coefficient matrix (may be rectangular)
            :c: constant vector (numpy array)
            :radius: positive real number or None (default: None)
            :transpose: if set to True, replace J with its transpose.

        `J` should be a linear operator.
        Additional keyword arguments are passed directly to `TruncatedCG`.

        Upon completion, the member `step` is set to the most recent solution
        estimate. See the documentation of `TruncatedCG` for more
        information.
        """
        self.op = SquaredLinearOperator(J, transposed=transposed)
        TruncatedCG.__init__(self, J.T * c, self.op, radius=radius, **kwargs)
        return


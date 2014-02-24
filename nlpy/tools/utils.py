# Various utilities.
import numpy as np
import logging
from math import copysign, sqrt


def Max(a):
    """
    A safeguarded max function. Returns -infinity for empty arrays.
    """
    return np.max(a) if a.size > 0 else -np.inf


def Min(a):
    """
    A safeguarded min function. Returns +infinity for empty arrays.
    """
    return np.min(a) if a.size > 0 else np.inf


class NullHandler(logging.Handler):
    '''
    A simple implementation of the null handler for Python 2.6.x (and older?)
    Useful for compatibility with older versions of Python.
    '''

    def emit(self, record):
        pass

    def handle(self, record):
        pass

    def createLock(self):
        return None


# Helper functions.
def identical(a, b):
    """
    Check that two arrays or lists are identical. Must be cautious because
    of Numpy's strange behavior:
    >>> a = np.array([]) ; b = np.array([0])
    >>> np.all(a==b)
    True
    """
    if a.shape == b.shape:
        return np.all(a == b)
    return False


def where(cond):
    "Bypass Numpy's annoyances. Gee does someone need to write a proper Numpy!"
    return np.where(cond)[0]


def roots_quadratic(q2, q1, q0, tol=1.0e-8, nitref=1):
    """
    Find the real roots of the quadratic q(x) = q2 * x^2 + q1 * x + q0.
    The numbers q0, q1 and q0 must be real.

    This function takes after the GALAHAD function of the same name.
    See http://galahad.rl.ac.uk.
    """
    a2 = float(q2); a1 = float(q1); a0 = float(q0)

    # Case of a linear function.
    if a2 == 0.0:
        if a1 == 0.0:
            if a0 == 0.0:
                return [0.0]
            else:
                return []
        else:
            roots = [-a0/a1]
    else:
        # Case of a quadratic.
        rhs = tol * a1 * a1
        if abs(a0*a2) > rhs:
            rho = a1 * a1 - 4.0 * a2 * a0
            if rho < 0.0:
                return []
            # There are two real roots.
            d = -0.5 * (a1 + copysign(sqrt(rho), a1))
            roots = [d/a2, a0/d]
        else:
            # Ill-conditioned quadratic.
            roots = [-a1/a2, 0.0]

    # Perform a few Newton iterations to improve accuracy.
    new_roots = []
    for root in roots:
        for it in range(nitref):
            val = (a2 * root + a1) * root + a0
            der = 2.0 * a2 * root + a1
            if der == 0.0:
                continue
            else:
                root = root - val/der
        new_roots.append(root)

    return new_roots


if __name__ == '__main__':
    roots = roots_quadratic(2.0e+20, .1, -4)
    print 'Received: ', roots

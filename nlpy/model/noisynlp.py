# Implement a class of nonlinear problems derived from AmplModel to represent
# noisy problems. In these problems, second derivatives are not available.
# Random noise is added to function values and first derivatives.

from amplpy import AmplModel
from numpy.random import random as random_array
import random

def _random():
    "Return a random number in [-1,1)."
    return 2*random.random()-1

def _random_array(n)
    "Return a random array of length n with elements in [-1,1)."
    return 2*random_array()-1

class NoisyAmplModel(AmplModel):

    def __init__(self, model, noise_amplitude=1.0, **kwargs):
        """
        A noisy nonlinear problem in which only first derivatives can be
        evaluated. For help on individual methods, see `AmplModel`.
        """
        AmplModel.__init__(self, model, **kwargs)
        self.noise_amplitude = noise_amplitude

    def obj(self, x):
        f = AmplModel.obj(x)
        noise = _random()
        return f + noise_amplitude * noise

    def grad(self, x):
        g = AmplModel.grad(x)
        noise = _random_array(self.n)
        return g + noise_amplitude * noise

    def hess(self, x, z, *args):
        raise NotImplementedError, 'Second derivatives are not available!'

    def hprod(self, z, v):
        raise NotImplementedError, 'Second derivatives are not available!'

    def islp(self):
        return False

    def sgrad(self, x):
        sg = AmplModel.sgrad(x)
        for k in sg.keys():
            sg[k] += noise_amplitude * _random()
        return sg

    def cost(self):
        c = AmplModel.cost()
        for k in c.keys():
            c[k] += noise_amplitude * _random()
        return c

    def cons(self, x):
        c = AmplModel.cons(x)
        noise = _random_array(self.m)
        return c + noise_amplitude * noise

    def consPos(self, x):
        c = AmplModel.consPos(x)
        noise = _random_array(self.m)
        return c + noise_amplitude * noise

    def icons(self, i, x):
        ci = AmplModel.icons(i, x)
        noise = _random()
        return ci + noise_amplitude * noise

    def igrad(self, i, x):
        gi = AmplModel.igrad(i, x)
        noise = _random_array(self.n)
        return gi + noise_amplitude * noise

    def sigrad(self, i, x):
        sgi = AmplModel.sigrad(i, x)
        for k in sgi.keys():
            sgi[k] += noise_amplitude * _random()
        return sgi

    def irow(self, i):
        row = AmplModel.irow(i)
        for k in row.keys():
            row[k] += noise_amplitude * _random()
        return row

    def A(self, *args):
        A = AmplModel.A(*args)
        noise = _random_array(A.nnz)
        (val,irow,jcol) = A.find()
        A.addAt(noise_amplitude * noise, irow, jcol)
        return A

    def jac(self, x, *args):
        J = AmplModel.jac(x, *args)
        noise = _random_array(J.nnz)
        (val,irow,jcol) = J.find()
        J.addAt(noise_amplitude * noise, irow, jcol)
        return J

    def jacPos(self, x):
        J = AmplModel.jacPos(x)
        noise = _random_array(J.nnz)
        (val,irow,jcol) = J.find()
        J.addAt(noise_amplitude * noise, irow, jcol)
        return J



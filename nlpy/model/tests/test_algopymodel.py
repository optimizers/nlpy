from numpy.testing import *
from nlpy.model.algopymodel import AlgopyModel

import numpy as np

class Test_RosenbrockModel(TestCase):
    
    def setUp(self):
        
        class Rosenbrock(AlgopyModel):
    
            def obj(self, x, **kwargs):
                return np.sum( 100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2 )
    
        nvar = 10
        nlp = Rosenbrock(n=nvar, name='Rosenbrock', x0=-np.ones(nvar))
        
        self.nlp = nlp


    def test_gradient(self):
        nlp = self.nlp
        g = nlp.grad(nlp.x0)
        
        assert_array_almost_equal(g,[1,2])

        


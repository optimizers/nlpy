# Tests relative to algorithmic differentiation with AMPL.
#from numpy.testing import *

try:
    from nlpy.model import AmplModel, PySparseAmplModel, SciPyAmplModel
    from nlpy.model.tests.helper import *
    import numpy as np
    import os

    this_path = os.path.dirname(os.path.realpath(__file__))


    class Test_AmplRosenbrock(TestCase, Rosenbrock):  # Test def'd in Rosenbrock

        def setUp(self):
            model = os.path.join(this_path, 'rosenbrock.nl')
            self.nlp = AmplModel(model)    # x0 = (-1, ..., -1)


    class Test_PySparseAmplRosenbrock(Test_AmplRosenbrock):

        def get_derivatives(self, nlp):
            return get_derivatives_llmat(nlp)

        def setUp(self):
            model = os.path.join(this_path, 'rosenbrock.nl')
            self.nlp = PySparseAmplModel(model)    # x0 = (-1, ..., -1)


    class Test_SciPyAmplRosenbrock(Test_AmplRosenbrock):

        def get_derivatives(self, nlp):
            return get_derivatives_scipy(nlp)

        def setUp(self):
            model = os.path.join(this_path, 'rosenbrock.nl')
            self.nlp = SciPyAmplModel(model)    # x0 = (-1, ..., -1)


    class Test_AmplHS7(TestCase, Hs7):    # Test defined in Hs7

        def setUp(self):
            model = os.path.join(this_path, 'hs007.nl')
            self.nlp = AmplModel(model)    # x0 = (2, 2)
            self.nlp.pi0 = np.ones(1)


    class Test_PySparseAmplHS7(Test_AmplHS7):

        def get_derivatives(self, nlp):
            return get_derivatives_llmat(nlp)

        def setUp(self):
            model = os.path.join(this_path, 'hs007.nl')
            self.nlp = PySparseAmplModel(model)
            self.nlp.pi0 = np.ones(1)


    class Test_SciPyAmplHS7(Test_AmplHS7):

        def get_derivatives(self, nlp):
            return get_derivatives_scipy(nlp)

        def setUp(self):
            model = os.path.join(this_path, 'hs007.nl')
            self.nlp = SciPyAmplModel(model)
            self.nlp.pi0 = np.ones(1)

except:
    pass

from nlpy.model.amplpy import AmplModel
from nlpy.krylov.ppcg import ProjectedCG as Ppcg
from nlpy.tools import norms
import numpy
import sys

if len(sys.argv) < 2:
        print 'This demo reads the nl file of an AMPL-decoded model and'
        print 'demonstrates the projected conjugate gradient by solving the'
        print 'following quadratic program with trust-region constraint:'
        print '     min   <g, x> + 1/2 <x, H x>'
        print '     s.t.  J d = 0,  |d| <= radius,'
        print
        print ' where g = grad(x0), H = Hess(x0), J = jac(x0), radius = |g|/10'
        print 'Please supply a constrained problem as input'
        sys.exit(0)

problem = sys.argv[1]
nlp = AmplModel(problem)

if nlp.m == 0:
    sys.stderr.write( 'Problem should have general constraints\n' )
    nlp.close()
    sys.exit(1)

x0 = numpy.ones(nlp.n) #nlp.x0            # Initial point
g  = numpy.random.random(nlp.n) #nlp.grad(x0)      # Objective gradient
J  = nlp.jac(x0)       # Obtain constraints Jacobian
delta = 1000 #max(1.0, 100 * norms.norm_infty(g))

# You can test constraints with a nonzero rhs by uncommenting the following,
# adding the 'rhs' and removing the 'radius' keywords in Ppcg below.
# Also make sure to adjust the computation of 'feas' at the bottom of the file.
#e = numpy.ones(nlp.n)
#c = numpy.empty(nlp.m, 'd')
#J.matvec(e, c)

# Call projected CG to solve problem
#   min  <g,d> + 1/2 <d, Hd>
#   s.t  Jd = 0,  |d| <= delta
hprod = lambda p: nlp.hprod(nlp.pi0,p)
CG = Ppcg(g, A=J, matvec=hprod, radius=delta, debug=True)
#CG = Ppcg(g, A=J, rhs=-c, matvec=hprod, debug=True)
CG.Solve()

Jd = numpy.empty(nlp.m, 'd')
J.matvec(CG.x, Jd)
feas = norms.norm_infty(Jd)
#feas = norms.norm_infty( c + Jd )  # for nonzero rhs constraint
snorm = norms.norm2(CG.x) # The trust-region is in the l2 norm

sys.stdout.write( 'Number of variables      : %-d\n' % nlp.n )
sys.stdout.write( 'Number of constraints    : %-d\n' % nlp.m )
sys.stdout.write( 'Converged                : %-s\n' % repr(CG.converged) )
sys.stdout.write( 'Trust-region radius      : %8.1e\n' % delta )
sys.stdout.write( 'Solution norm            : %8.1e\n' % snorm )
sys.stdout.write( 'Final/Initial Residual   : ')
sys.stdout.write( '%8.1e / %8.1e\n' % (CG.residNorm,CG.residNorm0) )
sys.stdout.write( 'Feasibility error        : %8.1e\n' % feas )
sys.stdout.write( 'Number of iterations     : %-d\n' % CG.iter )

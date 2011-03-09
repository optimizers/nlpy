.. Description of optimize module
.. _optimize-page:

******************
Optimization Tools
******************

==================
Linesearch Methods
==================

The `linesearch` module in NLPy implements a few typical line searches, i.e.,
given the current iterate :math:`x_k` along with the value :math:`f(x_k)`, the
gradient :math:`\nabla f(x_k)` and the search direction :math:`d_k`, they
return a stepsize :math:`\alpha_k > 0` such that :math:`x_k + \alpha_k d_k`
is an improved iterate in a certain sense. Typical conditions to be satisfied
by :math:`\alpha_k` include the ``Armijo condition``

.. math::

    f(x_k + \alpha_k d_k) \leq f(x_k) + \beta \alpha_k \nabla f(x_k)^T d_k

for some :math:`\beta \in (0,1)`, the ``Wolfe conditions``, which consist in
the Armijo condition supplemented with

.. math::

    \nabla f(x_k + \alpha_k d_k)^T d_k \geq \gamma \nabla f(x_k)^T d_k

for some :math:`\gamma \in (0,\beta)`, and the ``strong Wolfe conditions``,
which consist in the Armijo condition supplemented with

.. math::

    |\nabla f(x_k + \alpha_k d_k)^T d_k| \leq \gamma |\nabla f(x_k)^T d_k|

for some :math:`\gamma \in (0,\beta)`.

The :mod:`linesearch` Module
============================

.. _linesearch-section:

.. automodule:: linesearch

.. autoclass:: LineSearch
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: ArmijoLineSearch
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


The :mod:`pyswolfe` Module
==========================

.. _pyswolfe-section:

.. automodule:: pyswolfe

.. autoclass:: StrongWolfeLineSearch
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


The :mod:`pymswolfe` Module
===========================

.. _pymswolfe-section:

.. automodule:: pymswolfe

.. autoclass:: StrongWolfeLineSearch
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


====================
Trust-Region Methods
====================

Trust-region methods are an alternative to linesearch methods as a mechanism to
enforce global convergence, i.e., :math:`lim \nabla f(x_k) = 0`. In
trust-region methods, a quadratic model is approximately minimized at each
iteration subject to a trust-region constraint:

.. math::

    \begin{align*}
        \min_{d \in \mathbb{R}^n} & g_k^T d + \tfrac{1}{2} d^T H_k d \\
        \text{s.t.} & \|d\| \leq \Delta_k,
    \end{align*}

where :math:`g_k = \nabla f(x_k)`, :math:`H_k \approx \nabla^2 f(x_k)` and
:math:`\Delta_k > 0` is the current trust-region radius.

The :mod:`trustregion` Module
=============================

.. _trustregion-section:

.. automodule:: trustregion

.. autoclass:: TrustRegionFramework
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: TrustRegionSolver
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: TrustRegionCG
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: TrustRegionPCG
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: TrustRegionGLTR
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


================
Complete Solvers
================

.. _solvers-section:

Linear Least-Squares Problems
=============================

Linear least-squares problems may be stated as

.. math::

    \min_{x \in \mathbb{R}^n} \ \tfrac{1}{2} \|Ax-b\|_G^2,

for some matrix or linear operator :math:`A` and positive definite
preconditioner :math:`G`. See :ref:`linop` for more on linear operators.

.. automodule:: lsqr

.. autoclass:: LSQRFramework
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Linear Programming
==================

The linear programming problem can be stated in standard form as

.. math::

    \min_{x \in \mathbb{R}^n} \ c^T x \quad \text{subject to} \ Ax=b, \ x \geq
    0,

for some matrix :math:`A`. It is typical to reformulate arbitrary linear
programs as an equivalent linear program in standard form. However, in the next
solver, they are reformulated in so-called ``slack form`` using the
`SlackFramework` module. See :ref:`slacks-section`.

.. automodule:: lp

.. autoclass:: RegLPInteriorPointSolver
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: RegLPInteriorPointSolver29
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Convex Quadratic Programming
============================

The convex quadratic programming problem can be stated in standard form as

.. math::

    \min_{x \in \mathbb{R}^n} \ c^T x + \tfrac{1}{2} x^T Q x \quad \text{subject to} \ Ax=b, \ x \geq
    0,

for some matrix :math:`A` and some square symmetric positive semi-definite
matrix :math:`Q`. It is typical to reformulate arbitrary quadratic
programs as an equivalent quadratic program in standard form. However, in the next
solver, they are reformulated in so-called ``slack form`` using the
`SlackFramework` module. See :ref:`slacks-section`.

.. automodule:: cqp

.. autoclass:: RegQPInteriorPointSolver
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: RegQPInteriorPointSolver29
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Unconstrained Programming
=========================

The unconstrained programming problem can be stated as

.. math::

    \min_{x \in \mathbb{R}^n} \ f(x)

for some smooth function :math:`f: \mathbb{R}^n \to \R`. Typically, :math:`f`
is required to be twice continuously differentiable.

The `trunk` solver requires access to exact first and second derivatives of
:math:`f`. If minimizes :math:`f` by solving a sequence of trust-region
subproblems, i.e., problems of the form

.. math::

    \min_{d \in \mathbb{R}^n} \ g_k^T d + \tfrac{1}{2} d^T H_k d \quad
    \text{s.t.} \ \|d\| \leq \Delta_k,

where :math:`g_k = \nabla f(x_k)`, :math:`H_k = \nabla^2 f(x_k)` and
:math:`\Delta_k > 0` is the current trust-region radius.

.. automodule:: trunk

.. autoclass:: TrunkFramework
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: TrunkLbfgsFramework
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


The inverse L-BFGS method only requires access to exact first derivatives of
:math:`f` and maintains its own approximation to the second derivatives.

.. automodule:: lbfgs

.. autoclass:: InverseLBFGS
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: LBFGSFramework
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Bound-Constrained Programming
=============================

The unconstrained programming problem can be stated as

.. math::

    \min_{x \in \mathbb{R}^n} \ f(x) \quad \text{s.t.} \ x_i \geq 0 \ (i \in
    \mathcal{B})

for some smooth function :math:`f: \mathbb{R}^n \to \R`. Typically, :math:`f`
is required to be twice continuously differentiable.

.. automodule:: pdmerit

.. autoclass:: PrimalDualInteriorPointFramework
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

General Nonlinear Programming
=============================

The general nonlinear programming problem can be stated as

.. math::

    \begin{array}{ll}
      \min_{x \in \mathbb{R}^n} & f(x) \\
      \text{s.t.} & h(x) = 0, \\
                  & c(x) \geq 0,
    \end{array}

for smooth functions :math:`f`, :math:`h` and :math:`c`.

.. todo::

   Insert this module.

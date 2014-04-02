.. Description of modeling module
.. _model-page:

================
Modeling in NLPy
================

.. _nlp:

The general problem consists in minimizing an objective :math:`f(x)` subject to
general constraints and bounds:

.. math::

    c_i(x) = a_i,                 & \qquad i = 1, \ldots, m, \\
    g_j^L \leq g_j(x) \leq g_j^U, & \qquad j = 1, \ldots, p, \\
    x_k^L \leq x_k \leq x_k^U,    & \qquad k = 1, \ldots, n,

where some or all lower bounds :math:`g_j^L` and :math:`x_k^L` may be equal to
:math:`-\infty`, and some or all upper bounds :math:`g_j^U` and :math:`x_k^U`
may be equal to :math:`+\infty`.


.. _nlp-section:

--------------------------------
Modeling with `NLPModel` Objects
--------------------------------

`NLPModel` objects are the most basic type of model. They require that functions and their derivatives be coded explicitly.

The :mod:`nlp` Module
=====================

.. automodule:: nlp

.. autoclass:: NLPModel
   :show-inheritance:
   :members:
   :undoc-members:


Example
=======

.. todo:: Insert example.


.. _amplpy-section:

------------------------------------
Models in the AMPL Modeling Language
------------------------------------

`AmplModel` objects are convenient in that the model is written in the AMPL modeling language, and the AMPL Solver Library takes care of evaluating derivatives for us.

See the `AMPL home page <http://www.ampl.com>`_ for more information on the
AMPL algebraic modeling language.

The :mod:`amplpy` Module
========================

.. automodule:: amplpy

.. autoclass:: AmplModel
   :show-inheritance:
   :members:
   :undoc-members:


Example
=======

.. literalinclude:: ../../examples/demo_amplpy.py
   :linenos:


--------------------------------------------
Python Models with Automatic Differentiation
--------------------------------------------

ADOLC
=====

.. todo:: Write this section

CppAD
=====

.. todo:: Write this section

AlgoPy
======

.. todo:: Write this section


--------------------
Reformulating Models
--------------------

.. _slacks-section:

Using the Slack Formulation of a Model
--------------------------------------

The :mod:`snlp` Module
========================

.. automodule:: snlp


`SlackFramework` is a general framework for converting the :ref:`general
nonlinear optimization problem <nlp>` to a form using slack variables.

The transformed problem is in the variables `x`, `s` and `t` and its
constraints have the form

.. math::

    c_i(x) - a_i = 0, & \qquad i = 1, \ldots, m, \\
    g_j(x) - g_j^L - s_j^L = 0, & \qquad j = 1, \ldots, p,
    \qquad \text{for which} \quad  g_j^L > -\infty, \\
    s_j^L \geq 0, & \qquad j = 1, \ldots, p,
    \qquad \text{for which} \quad g_j^L > -\infty, \\
    g_j^U - g_j(x) - s_j^U = 0, & \qquad j = 1, \ldots, p,
    \qquad \text{for which} \quad g_j^U < +\infty, \\
    s_j^U \geq 0, & \qquad j = 1, \ldots, p,
    \qquad \text{for which} \quad g_j^U < +\infty, \\
    x_k - x_k^L - t_k^L = 0, & \qquad k = 1, \ldots, n,
    \qquad \text{for which} \quad x_k^L > -\infty, \\
    t_k^L \geq 0, & \qquad k = 1, \ldots, n,
    \qquad \text{for which} \quad x_k^L > -\infty, \\
    x_k^U - x_k - t_k^U = 0, & \qquad k = 1, \ldots, n,
    \qquad \text{for which} \quad x_k^U < +\infty, \\
    t_k^U \geq 0, & \qquad k = 1, \ldots, n,
    \qquad \text{for which} \quad x_k^U < +\infty.

In the latter problem, the only inequality constraints are bounds on
the slack variables. The other constraints are (typically) nonlinear
equalities.

The order of variables in the transformed problem is as follows:

[  x  |  sL  |  sU  |  tL  |  tU  ]

where:

- sL = [ sLL | sLR ], sLL being the slack variables corresponding to
  general constraints with a lower bound only, and sLR being the slack
  variables corresponding to the 'lower' side of range constraints.

- sU = [ sUU | sUR ], sUU being the slack variables corresponding to
  general constraints with an upper bound only, and sUR being the slack
  variables corresponding to the 'upper' side of range constraints.

- tL = [ tLL | tLR ], tLL being the slack variables corresponding to
  variables with a lower bound only, and tLR being the slack variables
  corresponding to the 'lower' side of two-sided bounds.

- tU = [ tUU | tUR ], tUU being the slack variables corresponding to
  variables with an upper bound only, and tLR being the slack variables
  corresponding to the 'upper' side of two-sided bounds.

This framework initializes the slack variables sL, sU, tL, and tU to
zero by default.

Note that the slack framework does not update all members of AmplModel,
such as the index set of constraints with an upper bound, etc., but
rather performs the evaluations of the constraints for the updated
model implicitly.

.. autoclass:: SlackModel
   :show-inheritance:
   :members:
   :undoc-members:


Example
=======

.. todo:: Insert example.

Inheritance Diagrams
====================

.. inheritance-diagram:: nlpy.model.nlp nlpy.model.amplpy nlpy.model.snlp nlpy.model.adolcmodel nlpy.model.cppadmodel nlpy.model.algopymodel nlpy.model.qnmodel nlpy.model.augmented_lagrangian nlpy.model.l1
  :parts: 1


.. inheritance-diagram:: nlpy.model.lbfgs nlpy.model.kkt
  :parts: 1

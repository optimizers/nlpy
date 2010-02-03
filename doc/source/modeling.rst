.. Description of modeling module
.. _model-page:

================
Modeling in NLPy
================

.. _amplpy-section:

------------------------------------------
Using Models in the AMPL Modeling Language
------------------------------------------

The :mod:`amplpy` Module
========================

.. automodule:: amplpy

.. autoclass:: AmplModel
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Example
=======

.. literalinclude:: ../../examples/demo_amplpy.py
   :linenos:


.. _nlp-section:

--------------------------------
Modeling with `NLPModel` Objects
--------------------------------

The :mod:`nlp` Module
=====================

.. automodule:: nlp

.. autoclass:: NLPModel
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Example
=======

.. todo:: Insert example.

.. _slacks-section:

--------------------------------------
Using the Slack Formulation of a Model
--------------------------------------

The :mod:`slacks` Module
========================

.. automodule:: slacks


`SlackFramework` is a general framework for converting a nonlinear optimization
problem to a form using slack variables.

The initial problem has the form

.. math::

    \text{minimize}   & f(x) & \\
    \text{subject to} & c_i(x) = a_i,                 & i = 1, \ldots, m, \\
                      & g_j^L \leq g_j(x) \leq g_j^U, & j = 1, \ldots, p, \\
                      & x_k^L \leq x_k \leq x_k^U,    & k = 1, \ldots, n,

where some or all lower bounds :math:`g_j^L` and :math:`x_k^L` may be equal to
:math:`-\infty`, and some or all upper bounds :math:`g_j^U` and :math:`x_k^U`
may be equal to :math:`+\infty`.

The transformed problem is

.. math::

    \text{minimize}   & f(x) & & \\
    \text{subject to} & c_i(x) - a_i = 0, & i = 1, \ldots, m, & \\
    & g_j(x) - g_j^L - s_j^L = 0, & j = 1, \ldots, p, &
    \text{ for which } g_j^L > -\infty, \\
    & s_j^L \geq 0, & j = 1, \ldots, p, & \text{ for which } g_j^L > -\infty, \\
    & g_j^U - g_j(x) - s_j^U = 0, & j = 1, \ldots, p, & \text{ for which }
    g_j^U < +\infty, \\
    & s_j^U >= 0, & j = 1, \ldots, p, & \text{ for which } g_j^U < +\infty, \\
    & x_k - x_k^L - t_k^L = 0, & k = 1, \ldots, n, & \text{ for which }
    x_k^L > -\infty, \\
    & t_k^L >= 0, & k = 1, \ldots, n, & \text{ for which } x_k^L > -\infty, \\
    & x_k^U - x_k - t_k^U = 0, & k = 1, \ldots, n, & \text{ for which }
    x_k^U < +\infty, \\
    & t_k^U >= 0, & k = 1, \ldots, n, & \text{ for which } x_k^U < +\infty.

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

.. .. autoclass:: SlackFramework
..    :show-inheritance:
..    :members:
..    :inherited-members:
..    :undoc-members:


Example
=======

.. todo:: Insert example.

Inheritance Diagram
===================

.. inheritance-diagram:: nlpy.model.slacks

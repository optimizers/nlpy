.. Description of krylov module
.. _krylov-page:

=================================================
Iterative Solution of Systems of Linear Equations
=================================================

.. _linop:

----------------
Linear Operators
----------------

An important characteristic of Krylov-type methods for the solution of systems
of linear equations is that they do not require explicit knowledge of the
coefficient matrix. Instead, they only require to be able to compute
matrix-vector products and, sometimes, matrix-vector products with the
transpose matrix. This opens the door to solving systems in which the
coefficient matrix is not known explicitly or is computationally expensive to
evaluate, form and/or store in memory. Whenever the coefficient matrix is used
in this way, we say that it is used as an `operator` or more precisely as a
`linear operator`.

Conceptually, if **A** is a matrix, the linear operator defined by **A** is the
function **L(x)** := **A*x** defined for any vector **x** of appropriate size.
In NLPy, linear operators are objects which may also encapsulate data related
to the linear operator.

In more technical terms, a``linear operator`` is an object which
implements ``__mul__()``. In NLPy, the ``__call__()`` method of such objects is
also defined as an alias to ``__mul__()``.

Among the members of a linear operator object is the `transpose` linear
operator if the latter is available. If ``L`` is a linear operator, its
transpose,
if it is available, is accessible via ``L.T`` which is a mnemonic for `L
transpose`. This allows to write expressions such as ``y = L * x`` and
``z = L.T * w``, but also ``y = L(x)`` and ``z = L.T(w)``.

Linear operators may be constructed in several ways:

1. By specifying a function which computes the result of the application of the
   linear operator to an input vector and, possibly, a second function which
   specifies the transpose linear operator.
2. By specifying any object which implements ``__mul__()`` such as a PySparse
   sparse matrix, a SciPy sparse matrix, a NumPy array, etc.
3. (The hard way) by subclassing the ``LinearOperator`` class.

Typically, a Krylov method only requires a linear operator as input. Whether or
not the transpose of this operator should be available via the ``T`` member
depends on the particular method used (and is only needed when solving
unsymmetric linear systems).

Examples
========

The first example builds a linear operator from a PySparse matrix in ``ll_mat``
format. In this case we use ``PysparseLinearOperator``::

    from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
    from nlpy.model import AmplModel
    import numpy as np

    nlp = AmplModel(sys.argv[1])
    J = nlp.jac(nlp.x0)
    e1 = np.ones(J.shape[0])
    e2 = np.ones(J.shape[1])

    op = PysparseLinearOperator(J)
    print 'op.shape = ', op.shape
    print 'op.T.shape = ', op.T.shape
    print 'op * e2 = ', op * e2
    print "op.T * e1 = ", op.T * e1
    print 'op.T.T * e2 = ', op.T.T * e2      # The same as op * e2.
    print 'op.T.T.T * e1 = ', op.T.T.T * e1  # The same as op.T * e1.
    print 'With call:'
    print 'op(e2) = ', op(e2)
    print 'op.T(e1) = ', op.T(e1)
    print 'op.T.T is op : ', (op.T.T is op)  # Returns True.

The second example uses functions to build the linear operator. The appropriate
object is then ``SimpleLinearOperator``::

    J = sp(matrix=nlp.jac(nlp.x0))
    op = SimpleLinearOperator(J.shape[1], J.shape[0],
                              lambda v: J*v,
                              matvec_transp=lambda u: u*J)
    print 'op.shape = ', op.shape
    print 'op.T.shape = ', op.T.shape
    print 'op * e2 = ', op * e2
    print 'e1.shape = ', e1.shape
    print 'op.T * e1 = ', op.T * e1
    print 'op.T.T * e2 = ', op.T.T * e2
    print 'op(e2) = ', op(e2)
    print 'op.T(e1) = ', op.T(e1)
    print 'op.T.T is op : ', (op.T.T is op)

Special linear operators are those that represent the matrices ``A * A.T`` and
``A.T * A``. Such operators arise, for example, when solving linear
least-squares problems via the normal equations. They can be constructed using
``SquaredLinearOperator`` objects::

    op2 = SquaredLinearOperator(J)                  # Represents J'*J
    print 'op2 * e2 = ', op2 * e2
    print 'op.T * (op * e2) = ', op.T * (op * e2)   # Mind the parentheses!
    op3 = SquaredLinearOperator(J, transpose=True)  # Represents J*J'
    print 'op3 * e1 = ', op3 * e1
    print 'op * (op.T * e1) = ', op * (op.T * e1)

The :mod:`linop` Module
=======================

.. _linop-section:

.. automodule:: linop

.. autoclass:: LinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: PysparseLinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: SimpleLinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: SquaredLinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


------------------------------
Symmetric Systems of Equations
------------------------------

The :mod:`minres` Module
========================

.. _minres-section:

.. automodule:: minres

.. autoclass:: Minres
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


The :mod:`pcg` Module
=====================

.. _pcg-section:

.. automodule:: pcg

.. autoclass:: TruncatedCG
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

The :mod:`pygltr` Module
========================

.. _pygltr-section:

.. automodule:: pygltr

.. autoclass:: PyGltrContext
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


The :mod:`projKrylov` Module
============================

.. _projKrylov-section:

.. automodule:: projKrylov

.. autoclass:: ProjectedKrylov
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

The :mod:`ppcg` Module
======================

.. _ppcg-section:

.. automodule:: ppcg

.. autoclass:: ProjectedCG
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


----------------------------------
Non-Symmetric Systems of Equations
----------------------------------

The :mod:`pbcgstab` Module
==========================

.. _pbcgstab-section:

.. automodule:: pbcgstab

.. autoclass:: ProjectedBCGSTAB
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

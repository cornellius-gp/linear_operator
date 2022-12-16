.. role:: hidden
    :class: hidden-section

.. currentmodule:: linear_operator


Functions
==============================

LinearOperator objects are designed to work seamlessly with the torch API.
For example:

    >>> linear_op = linear_operators.operators.ToeplitzLinearOperator(
    >>>    torch.tensor([1., 2., 3., 4.])
    >>> )
    >>> # Represents a Toeplitz matrix:
    >>> # [[1., 2., 3., 4.],
    >>> #  [2., 1., 2., 3.],
    >>> #  [3., 2., 1., 2.],
    >>> #  [4., 3., 2., 1.]]
    >>> torch.matmul(linear_op, torch.tensor([0.5, 0.1, 0.2, 0.4]))
    >>> # Returns: tensor([2.9, 2.7, 2.7, 3.1])

The :attr:`linear_operator` module also includes some functions taht are not implemented as part of Torch.
These functions are designed to work on :class:`~linear_operator.operators.LinearOperator` and :class:`~torch.Tensor`
objects alike.


.. automodule:: linear_operator

.. autofunction:: add_diagonal

.. autofunction:: add_jitter

.. autofunction:: dsmm

.. autofunction:: diagonalization

.. autofunction:: inv_quad

.. autofunction:: inv_quad_logdet

.. autofunction:: pivoted_cholesky

.. autofunction:: root_decomposition

.. autofunction:: root_inv_decomposition

.. autofunction:: solve

.. autofunction:: sqrt_inv_matmul

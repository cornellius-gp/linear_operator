.. role:: hidden
    :class: hidden-section

Writing Your Own LinearOpeators
===================================

In order to define a new LinearOperator class, a user must define
at a minimum the following methods (in each example, :math:`\mathbf A` denotes
the matrix that the LinearOperator represents)

* :meth:`~linear_operator.operators.LinearOperator._matmul`, which performs a
  matrix multiplication :math:`\mathbf {AB}`
* :meth:`~linear_operator.operators.LinearOperator._size`, which returns a
  :class:`torch.Size` containing the dimensions of :math:`\mathbf A`.
* :meth:`~linear_operator.operators.LinearOperator._transpose_nonbatch`, which
  returns a transposed version of the LinearOperator

In addition to these, the following methods should be implemented for maximum efficiency

* :meth:`~linear_operator.operators.LinearOperator._quad_form_derivative`,
  which computes the derivative of a quadratic form with the LinearOperator
  (e.g. :math:`d (\mathbf b^T \mathbf A \mathbf c) / d \mathbf A`).
* :meth:`~linear_operator.operators.LinearOperator._get_indices`, which returns
  a :class:`torch.Tensor` containing elements that are given by various tensor indices.
* :meth:`~linear_operator.operators.LinearOperator._expand_batch`, which
  expands the batch dimensions of LinearOperators.
* :meth:`~linear_operator.operators.LinearOperator._check_args`, which performs
  error checking on the arguments supplied to the LinearOperator constructor.

In addition to these, a LinearOperator *may* need to define the following functions if it does anything interesting
with the batch dimensions (e.g. sums along them, adds additional ones, etc):
:func:`~linear_operator.operators.LinearOperator._unsqueeze_batch`,
:func:`~linear_operator.operators.LinearOperator._getitem`, and
:func:`~linear_operator.operators.LinearOperator._permute_batch`.
See the documentation for these methods for details.

.. note::
    The base LinearOperator class provides default implementations of many
    other operations in order to mimic the behavior of a standard tensor as
    closely as possible. For example, we provide default implementations of
    :func:`~linear_operator.operators.LinearOperator.__getitem__`,
    :func:`~linear_operator.operators.LinearOperator.__add__`, etc that either
    make use of other linear operators or exploit the functions that **must**
    be defined above.

    Rather than overriding the public methods, we recommend that you override
    the private versions associated with these methods (e.g. - write a custom
    :meth:`_getitem` verses a custom :meth:`__getitem__`). This is because the public
    methods do quite a bit of error checking and casing that doesn't need to be
    repeated.

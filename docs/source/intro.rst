.. role:: hidden
    :class: hidden-section

The LinearOperator Abstraction
===================================

A :obj:`~linear_operator.LinearOperator` is an object that represents a tensor
object, similar to :obj:`torch.tensor`, but typically differs in two ways:

#. A tensor represented by a LinearOperator can typically be represented more
   efficiently than storing a full matrix.  For example, a LinearOperator
   representing :math:`\mathbf A= \mathbf{XX}^{\top}` where :math:`\mathbf A` is :math:`N \times N` but
   :math:`\mathbf X` is :math:`N \times D` might store :math:`\mathbf X`
   instead of :math:`\mathbf A` directly.
#. A LinearOperator typically defines a matmul routine that performs
   :math:`\mathbf {AM}` that is more efficient than storing the full matrix.
   Using the above example, performing
   :math:`\mathbf{AM}=\mathbf X(\mathbf X^{\top}\mathbf M)` requires only :math:`O(ND)` time,
   rather than the :math:`O(N^2)` time required if we were storing :math:`\mathbf A` directly.

.. note::
    LinearOperators are designed by default to optionally represent batches of matrices. Thus, the size of a
    LinearOperator may be (for example) :math:`B_1 \times \ldots \times B_K \times N \times N`. Many of
    the methods are designed to efficiently operate on these batches if present.

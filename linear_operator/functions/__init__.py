#!/usr/bin/env python3

from ._dsmm import DSMM


def dsmm(sparse_mat, dense_mat):
    """
    Performs the (batch) matrix multiplication S x D
    where S is a sparse matrix and D is a dense matrix

    Args:
        - sparse_mat (matrix (b x)mxn) - Tensor wrapping sparse matrix
        - dense_mat (matrix (b x)nxo) - Tensor wrapping dense matrix

    Returns:
        - matrix (b x)mxo - Result
    """
    return DSMM.apply(sparse_mat, dense_mat)


def matmul(mat, rhs):
    """
    Computes a matrix multiplication between a matrix (mat) and a right hand side (rhs).
    If mat is a tensor, then this is the same as torch.matmul.
    This function can work on lazy tensors though

    Args:
        - mat (matrix nxn) - left hand size matrix
        - rhs (matrix nxk) - rhs matrix or vector

    Returns:
        - matrix nxk
    """
    return mat.matmul(rhs)


def inv_matmul(mat, right_tensor, left_tensor=None):
    r"""
    Computes a linear solve (w.r.t :attr:`mat` = :math:`A`) with several right hand sides :math:`R`.
    I.e. computes

    ... math::

        \begin{equation}
            A^{-1} R,
        \end{equation}

    where :math:`R` is :attr:`right_tensor` and :math:`A` is :attr:`mat`.

    If :attr:`left_tensor` is supplied, computes

    ... math::

        \begin{equation}
            L A^{-1} R,
        \end{equation}

    where :math:`L` is :attr:`left_tensor`. Supplying this can reduce the number of
    CG calls required.

    Args:
        - :obj:`torch.tensor` (n x k) - Matrix :math:`R` right hand sides
        - :obj:`torch.tensor` (m x n) - Optional matrix :math:`L` to perform left multiplication with

    Returns:
        - :obj:`torch.tensor` - :math:`A^{-1}R` or :math:`LA^{-1}R`.
    """
    from ..operators import to_linear_operator

    return to_linear_operator(mat).inv_matmul(right_tensor, left_tensor)


def inv_quad(mat, tensor):
    """
    Computes an inverse quadratic form (w.r.t mat) with several right hand sides.
    I.e. computes tr( tensor^T mat^{-1} tensor )

    Args:
        - tensor (tensor nxk) - Vector (or matrix) for inverse quad

    Returns:
        - tensor - tr( tensor^T (mat)^{-1} tensor )
    """
    res, _ = inv_quad_logdet(mat, inv_quad_rhs=tensor, logdet=False)
    return res


def inv_quad_logdet(mat, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
    """
    Computes an inverse quadratic form (w.r.t mat) with several right hand sides.
    I.e. computes tr( tensor^T mat^{-1} tensor )
    In addition, computes an (approximate) log determinant of the the matrix

    Args:
        - tensor (tensor nxk) - Vector (or matrix) for inverse quad

    Returns:
        - scalar - tr( tensor^T (mat)^{-1} tensor )
        - scalar - log determinant
    """
    from ..operators import to_linear_operator

    return to_linear_operator(mat).inv_quad_logdet(inv_quad_rhs, logdet, reduce_inv_quad=reduce_inv_quad)


def logdet(mat):
    """
    Computes an (approximate) log determinant of the matrix

    Returns:
        - scalar - log determinant
    """
    _, res = inv_quad_logdet(mat, inv_quad_rhs=None, logdet=True)
    return res


def pivoted_cholesky(mat, rank, error_tol=None, return_pivots=None):
    r"""
    Performs a partial pivoted Cholesky factorization of the (positive definite) matrix.
    :math:`\mathbf L \mathbf L^\top = \mathbf K`.
    The partial pivoted Cholesky factor :math:`\mathbf L \in \mathbb R^{N \times \text{rank}}`
    forms a low rank approximation to the matrix.

    The pivots are selected greedily, corresponding to the maximum diagonal element in the
    residual after each Cholesky iteration. See `Harbrecht et al., 2012`_.

    :param mat: The matrix :math:`\mathbf K` to decompose
    :type mat: ~linear_operator.lazy.LinearOperator or ~torch.Tensor
    :param int rank: The size of the partial pivoted Cholesky factor.
    :param error_tol: Defines an optional stopping criterion.
        If the residual of the factorization is less than :attr:`error_tol`, then the
        factorization will exit early. This will result in a :math:`\leq \text{ rank}` factor.
    :type error_tol: float, optional
    :param bool return_pivots: (default: False) Whether or not to return the pivots alongside
        the partial pivoted Cholesky factor.
    :return: the `... x N x rank` factor (and optionally the `... x N` pivots)
    :rtype: torch.Tensor or tuple(torch.Tensor, torch.Tensor)

    .. _Harbrecht et al., 2012:
        https://www.sciencedirect.com/science/article/pii/S0168927411001814
    """
    from ..operators import to_linear_operator

    return to_linear_operator(mat).pivoted_cholesky(rank=rank, error_tol=error_tol, return_pivots=return_pivots)


def root_decomposition(mat):
    """
    Returns a (usually low-rank) root decomposotion lazy tensor of a PSD matrix.
    This can be used for sampling from a Gaussian distribution, or for obtaining a
    low-rank version of a matrix
    """
    from ..operators import to_linear_operator

    return to_linear_operator(mat).root_decomposition()


def root_inv_decomposition(mat, initial_vectors=None, test_vectors=None):
    """
    Returns a (usually low-rank) root decomposotion lazy tensor of a PSD matrix.
    This can be used for sampling from a Gaussian distribution, or for obtaining a
    low-rank version of a matrix
    """
    from ..operators import to_linear_operator

    return to_linear_operator(mat).root_inv_decomposition(initial_vectors, test_vectors)


__all__ = [
    "dsmm",
    "inv_matmul",
    "inv_quad",
    "inv_quad_logdet",
    "logdet",
    "matmul",
    "normal_cdf",
    "pivoted_cholesky",
    "root_decomposition",
    "root_inv_decomposition",
]

#!/usr/bin/env python3

import operator
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from .. import settings
from ..utils.broadcasting import _matmul_broadcast_shape
from ..utils.memoize import cached
from ._linear_operator import IndexType, LinearOperator
from .dense_linear_operator import to_linear_operator
from .diag_linear_operator import ConstantDiagLinearOperator, DiagLinearOperator
from .triangular_linear_operator import _TriangularLinearOperatorBase, TriangularLinearOperator


def _kron_diag(*lts) -> Tensor:
    """Compute diagonal of a KroneckerProductLinearOperator from the diagonals of the constituiting tensors"""
    lead_diag = lts[0]._diagonal()
    if len(lts) == 1:  # base case:
        return lead_diag
    trail_diag = _kron_diag(*lts[1:])
    diag = lead_diag.unsqueeze(-2) * trail_diag.unsqueeze(-1)
    return diag.mT.reshape(*diag.shape[:-2], -1)


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _matmul(linear_ops, kp_shape, rhs):
    output_shape = _matmul_broadcast_shape(kp_shape, rhs.shape)
    output_batch_shape = output_shape[:-2]

    res = rhs.contiguous().expand(*output_batch_shape, *rhs.shape[-2:])
    num_cols = rhs.size(-1)
    for linear_op in linear_ops:
        res = res.view(*output_batch_shape, linear_op.size(-1), -1)
        factor = linear_op._matmul(res)
        factor = factor.view(*output_batch_shape, linear_op.size(-2), -1, num_cols).transpose(-3, -2)
        res = factor.reshape(*output_batch_shape, -1, num_cols)
    return res


def _t_matmul(linear_ops, kp_shape, rhs):
    kp_t_shape = (*kp_shape[:-2], kp_shape[-1], kp_shape[-2])
    output_shape = _matmul_broadcast_shape(kp_t_shape, rhs.shape)
    output_batch_shape = torch.Size(output_shape[:-2])

    res = rhs.contiguous().expand(*output_batch_shape, *rhs.shape[-2:])
    num_cols = rhs.size(-1)
    for linear_op in linear_ops:
        res = res.view(*output_batch_shape, linear_op.size(-2), -1)
        factor = linear_op._t_matmul(res)
        factor = factor.view(*output_batch_shape, linear_op.size(-1), -1, num_cols).transpose(-3, -2)
        res = factor.reshape(*output_batch_shape, -1, num_cols)
    return res


class KroneckerProductLinearOperator(LinearOperator):
    r"""
    Given linearOperators :math:`\boldsymbol K_1, \ldots, \boldsymbol K_P`,
    this LinearOperator represents the Kronecker product :math:`\boldsymbol K_1 \otimes \ldots \otimes \boldsymbol K_P`.

    :param linear_ops: :math:`\boldsymbol K_1, \ldots, \boldsymbol K_P`: the LinearOperators in the Kronecker product.
    """

    def __init__(self, *linear_ops: Union[Float[Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"]]):
        try:
            linear_ops = tuple(to_linear_operator(linear_op) for linear_op in linear_ops)
        except TypeError:
            raise RuntimeError("KroneckerProductLinearOperator is intended to wrap lazy tensors.")

        # Make batch shapes the same for all operators
        try:
            batch_broadcast_shape = torch.broadcast_shapes(*(linear_op.batch_shape for linear_op in linear_ops))
        except RuntimeError:
            raise RuntimeError(
                "Batch shapes of LinearOperators "
                f"({', '.join([str(tuple(linear_op.shape)) for linear_op in linear_ops])}) "
                "are incompatible for a Kronecker product."
            )

        if len(batch_broadcast_shape):  # Otherwise all linear_ops are non-batch, and we don't need to expand
            # NOTE: we must explicitly call requires_grad on each of these arguments
            # for the automatic _bilinear_derivative to work in torch.autograd.Functions
            linear_ops = tuple(
                linear_op._expand_batch(batch_broadcast_shape).requires_grad_(linear_op.requires_grad)
                for linear_op in linear_ops
            )

        super().__init__(*linear_ops)
        self.linear_ops = linear_ops

    def __add__(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"], float],
    ) -> Union[Float[LinearOperator, "... M N"], Float[Tensor, "... M N"]]:
        if isinstance(other, (KroneckerProductDiagLinearOperator, ConstantDiagLinearOperator)):
            from .kronecker_product_added_diag_linear_operator import KroneckerProductAddedDiagLinearOperator

            return KroneckerProductAddedDiagLinearOperator(self, other)
        if isinstance(other, KroneckerProductLinearOperator):
            from .sum_kronecker_linear_operator import SumKroneckerLinearOperator

            return SumKroneckerLinearOperator(self, other)
        if isinstance(other, DiagLinearOperator):
            return self.add_diagonal(other._diagonal())
        return super().__add__(other)

    def add_diagonal(
        self: Float[LinearOperator, "*batch N N"],
        diag: Union[Float[torch.Tensor, "... N"], Float[torch.Tensor, "... 1"], Float[torch.Tensor, ""]],
    ) -> Float[LinearOperator, "*batch N N"]:
        from .kronecker_product_added_diag_linear_operator import KroneckerProductAddedDiagLinearOperator

        if not self.is_square:
            raise RuntimeError("add_diag only defined for square matrices")

        diag_shape = diag.shape
        if len(diag_shape) == 0:
            # interpret scalar tensor as constant diag
            diag_tensor = ConstantDiagLinearOperator(diag.unsqueeze(-1), diag_shape=self.shape[-1])
        elif diag_shape[-1] == 1:
            # interpret single-trailing element as constant diag
            diag_tensor = ConstantDiagLinearOperator(diag, diag_shape=self.shape[-1])
        else:
            try:
                expanded_diag = diag.expand(self.shape[:-1])
            except RuntimeError:
                raise RuntimeError(
                    "add_diag for LinearOperator of size {} received invalid diagonal of size {}.".format(
                        self.shape, diag_shape
                    )
                )
            diag_tensor = DiagLinearOperator(expanded_diag)

        return KroneckerProductAddedDiagLinearOperator(self, diag_tensor)

    def diagonalization(
        self: Float[LinearOperator, "*batch N N"], method: Optional[str] = None
    ) -> Tuple[Float[Tensor, "*batch M"], Optional[Float[LinearOperator, "*batch N M"]]]:
        if method is None:
            method = "symeig"
        return super().diagonalization(method=method)

    @cached
    def inverse(self: Float[LinearOperator, "*batch N N"]) -> Float[LinearOperator, "*batch N N"]:
        # here we use that (A \kron B)^-1 = A^-1 \kron B^-1
        # TODO: Investigate under what conditions computing individual individual inverses makes sense
        inverses = [lt.inverse() for lt in self.linear_ops]
        return self.__class__(*inverses)

    def inv_quad_logdet(
        self: Float[LinearOperator, "*batch N N"],
        inv_quad_rhs: Optional[Union[Float[Tensor, "*batch N M"], Float[Tensor, "*batch N"]]] = None,
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[
        Optional[Union[Float[Tensor, "*batch M"], Float[Tensor, " *batch"], Float[Tensor, " 0"]]],
        Optional[Float[Tensor, "..."]],
    ]:
        if inv_quad_rhs is not None:
            inv_quad_term, _ = super().inv_quad_logdet(
                inv_quad_rhs=inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
            )
        else:
            inv_quad_term = None
        logdet_term = self._logdet() if logdet else None
        return inv_quad_term, logdet_term

    @cached(name="cholesky")
    def _cholesky(
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        chol_factors = [lt.cholesky(upper=upper) for lt in self.linear_ops]
        return KroneckerProductTriangularLinearOperator(*chol_factors, upper=upper)

    def _diagonal(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
        return _kron_diag(*self.linear_ops)

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: Union[torch.Size, List[int]]
    ) -> Float[LinearOperator, "... M N"]:
        return self.__class__(*[linear_op._expand_batch(batch_shape) for linear_op in self.linear_ops])

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        row_factor = self.size(-2)
        col_factor = self.size(-1)

        res = None
        for linear_op in self.linear_ops:
            sub_row_size = linear_op.size(-2)
            sub_col_size = linear_op.size(-1)

            row_factor //= sub_row_size
            col_factor //= sub_col_size
            sub_res = linear_op._get_indices(
                torch.div(row_index, row_factor, rounding_mode="floor").fmod(sub_row_size),
                torch.div(col_index, col_factor, rounding_mode="floor").fmod(sub_col_size),
                *batch_indices,
            )
            res = sub_res if res is None else (sub_res * res)

        return res

    def _solve(
        self: Float[LinearOperator, "... N N"],
        rhs: Float[torch.Tensor, "... N C"],
        preconditioner: Optional[Callable[[Float[torch.Tensor, "... N C"]], Float[torch.Tensor, "... N C"]]] = None,
        num_tridiag: Optional[int] = 0,
    ) -> Union[
        Float[torch.Tensor, "... N C"],
        Tuple[
            Float[torch.Tensor, "... N C"],
            Float[torch.Tensor, "..."],  # Note that in case of a tuple the second term size depends on num_tridiag
        ],
    ]:
        # Computes solve by exploiting the identity (A \kron B)^-1 = A^-1 \kron B^-1
        # we perform the solve first before worrying about any tridiagonal matrices

        tsr_shapes = [q.size(-1) for q in self.linear_ops]
        n_rows = rhs.size(-2)
        batch_shape = torch.broadcast_shapes(self.shape[:-2], rhs.shape[:-2])
        perm_batch = tuple(range(len(batch_shape)))
        y = rhs.clone().expand(*batch_shape, *rhs.shape[-2:])
        for n, q in zip(tsr_shapes, self.linear_ops):
            # for KroneckerProductTriangularLinearOperator this solve is very cheap
            y = q.solve(y.reshape(*batch_shape, n, -1))
            y = y.reshape(*batch_shape, n, n_rows // n, -1).permute(*perm_batch, -2, -3, -1)
        res = y.reshape(*batch_shape, n_rows, -1)

        if num_tridiag == 0:
            return res
        else:
            # we need to return the t mat, so we return the eigenvalues
            # in general, this should not be called because log determinant estimation
            # is closed form and is implemented in _logdet
            # TODO: make this more efficient
            evals, _ = self.diagonalization()
            evals_repeated = evals.unsqueeze(0).repeat(num_tridiag, *[1] * evals.ndim)
            lazy_evals = DiagLinearOperator(evals_repeated)
            batch_repeated_evals = lazy_evals.to_dense()
            return res, batch_repeated_evals

    def _inv_matmul(self, right_tensor, left_tensor=None):
        # if _inv_matmul is called, we ignore the eigenvalue handling
        # this is efficient because of the structure of the lazy tensor
        res = self._solve(rhs=right_tensor)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def _logdet(self: Float[LinearOperator, "*batch M N"]) -> Float[Tensor, " *batch"]:
        evals, _ = self.diagonalization()
        logdet = evals.clamp(min=1e-7).log().sum(-1)
        return logdet

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _matmul(self.linear_ops, self.shape, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    @cached(name="root_decomposition")
    def root_decomposition(
        self: Float[LinearOperator, "*batch N N"], method: Optional[str] = None
    ) -> Float[LinearOperator, "*batch N N"]:
        from linear_operator.operators import RootLinearOperator

        # return a dense root decomposition if the matrix is small
        if self.shape[-1] <= settings.max_cholesky_size.value():
            return super().root_decomposition(method=method)

        root_list = [lt.root_decomposition(method=method).root for lt in self.linear_ops]
        kronecker_root = KroneckerProductLinearOperator(*root_list)
        return RootLinearOperator(kronecker_root)

    @cached(name="root_inv_decomposition")
    def root_inv_decomposition(
        self: Float[LinearOperator, "*batch N N"],
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
        method: Optional[str] = None,
    ) -> Union[Float[LinearOperator, "... N N"], Float[Tensor, "... N N"]]:
        from linear_operator.operators import RootLinearOperator

        # return a dense root decomposition if the matrix is small
        if self.shape[-1] <= settings.max_cholesky_size.value():
            return super().root_inv_decomposition()

        root_list = [lt.root_inv_decomposition().root for lt in self.linear_ops]
        kronecker_root = KroneckerProductLinearOperator(*root_list)
        return RootLinearOperator(kronecker_root)

    @cached(name="size")
    def _size(self) -> torch.Size:
        left_size = _prod(linear_op.size(-2) for linear_op in self.linear_ops)
        right_size = _prod(linear_op.size(-1) for linear_op in self.linear_ops)
        return torch.Size((*self.linear_ops[0].batch_shape, left_size, right_size))

    @cached(name="svd")
    def _svd(
        self: Float[LinearOperator, "*batch N N"]
    ) -> Tuple[Float[LinearOperator, "*batch N N"], Float[Tensor, "... N"], Float[LinearOperator, "*batch N N"]]:
        U, S, V = [], [], []
        for lt in self.linear_ops:
            U_, S_, V_ = lt.svd()
            U.append(U_)
            S.append(S_)
            V.append(V_)
        S = KroneckerProductLinearOperator(*[DiagLinearOperator(S_) for S_ in S])._diagonal()
        U = KroneckerProductLinearOperator(*U)
        V = KroneckerProductLinearOperator(*V)
        return U, S, V

    def _symeig(
        self: Float[LinearOperator, "*batch N N"],
        eigenvectors: bool = False,
        return_evals_as_lazy: Optional[bool] = False,
    ) -> Tuple[Float[Tensor, "*batch M"], Optional[Float[LinearOperator, "*batch N M"]]]:
        # return_evals_as_lazy is a flag to return the eigenvalues as a lazy tensor
        # which is useful for root decompositions here (see the root_decomposition
        # method above)
        evals, evecs = [], []
        for lt in self.linear_ops:
            evals_, evecs_ = lt._symeig(eigenvectors=eigenvectors)
            evals.append(evals_)
            evecs.append(evecs_)
        evals = KroneckerProductDiagLinearOperator(*[DiagLinearOperator(evals_) for evals_ in evals])

        if not return_evals_as_lazy:
            evals = evals._diagonal()

        if eigenvectors:
            evecs = KroneckerProductLinearOperator(*evecs)
        else:
            evecs = None
        return evals, evecs

    def _t_matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[Tensor, "*batch2 M P"], Float[LinearOperator, "*batch2 M P"]],
    ) -> Union[Float[LinearOperator, "... N P"], Float[Tensor, "... N P"]]:
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _t_matmul(self.linear_ops, self.shape, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return self.__class__(*(linear_op._transpose_nonbatch() for linear_op in self.linear_ops), **self._kwargs)


class KroneckerProductTriangularLinearOperator(KroneckerProductLinearOperator, _TriangularLinearOperatorBase):
    def __init__(self, *linear_ops, upper=False):
        if not all(isinstance(lt, TriangularLinearOperator) for lt in linear_ops):
            raise RuntimeError(
                "Components of KroneckerProductTriangularLinearOperator must be TriangularLinearOperator."
            )
        super().__init__(*linear_ops)
        self.upper = upper

    @cached
    def inverse(self: Float[LinearOperator, "*batch N N"]) -> Float[LinearOperator, "*batch N N"]:
        # here we use that (A \kron B)^-1 = A^-1 \kron B^-1
        inverses = [lt.inverse() for lt in self.linear_ops]
        return self.__class__(*inverses, upper=self.upper)

    @cached(name="cholesky")
    def _cholesky(
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        raise NotImplementedError("_cholesky not applicable to triangular lazy tensors")

    def _cholesky_solve(
        self: Float[LinearOperator, "*batch N N"],
        rhs: Union[Float[LinearOperator, "*batch2 N M"], Float[Tensor, "*batch2 N M"]],
        upper: Optional[bool] = False,
    ) -> Union[Float[LinearOperator, "... N M"], Float[Tensor, "... N M"]]:
        if upper:
            # res = (U.T @ U)^-1 @ v = U^-1 @ U^-T @ v
            w = self._transpose_nonbatch().solve(rhs)
            res = self.solve(w)
        else:
            # res = (L @ L.T)^-1 @ v = L^-T @ L^-1 @ v
            w = self.solve(rhs)
            res = self._transpose_nonbatch().solve(w)
        return res

    def _symeig(
        self: Float[LinearOperator, "*batch N N"],
        eigenvectors: bool = False,
        return_evals_as_lazy: Optional[bool] = False,
    ) -> Tuple[Float[Tensor, "*batch M"], Optional[Float[LinearOperator, "*batch N M"]]]:
        raise NotImplementedError("_symeig not applicable to triangular lazy tensors")

    def solve(
        self: Float[LinearOperator, "... N N"],
        right_tensor: Union[Float[Tensor, "... N P"], Float[Tensor, " N"]],
        left_tensor: Optional[Float[Tensor, "... O N"]] = None,
    ) -> Union[Float[Tensor, "... N P"], Float[Tensor, "... N"], Float[Tensor, "... O P"], Float[Tensor, "... O"]]:
        # For triangular components, using triangular-triangular substition should generally be good
        return self._inv_matmul(right_tensor=right_tensor, left_tensor=left_tensor)


class KroneckerProductDiagLinearOperator(DiagLinearOperator, KroneckerProductTriangularLinearOperator):
    r"""
    Represents the kronecker product of multiple DiagonalLinearOperators
    (i.e. :math:`\mathbf D_1 \otimes \mathbf D_2 \otimes \ldots \mathbf D_\ell`)
    Supports arbitrary batch sizes.

    :param linear_ops: Diagonal linear operators (:math:`\mathbf D_1, \mathbf D_2, \ldots \mathbf D_\ell`).
    """

    def __init__(self, *linear_ops: Tuple[DiagLinearOperator, ...]):
        if not all(isinstance(lt, DiagLinearOperator) for lt in linear_ops):
            raise RuntimeError("Components of KroneckerProductDiagLinearOperator must be DiagLinearOperator.")
        super(KroneckerProductTriangularLinearOperator, self).__init__(*linear_ops)
        self.upper = False

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        return KroneckerProductTriangularLinearOperator._bilinear_derivative(self, left_vecs, right_vecs)

    @cached(name="cholesky")
    def _cholesky(
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        chol_factors = [lt.cholesky(upper=upper) for lt in self.linear_ops]
        return KroneckerProductDiagLinearOperator(*chol_factors)

    @property
    def _diag(self: Float[LinearOperator, "*batch N N"]) -> Float[Tensor, "*batch N"]:
        return _kron_diag(*self.linear_ops)

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: Union[torch.Size, List[int]]
    ) -> Float[LinearOperator, "... M N"]:
        return KroneckerProductTriangularLinearOperator._expand_batch(self, batch_shape)

    def _mul_constant(
        self: Float[LinearOperator, "*batch M N"], other: Union[float, torch.Tensor]
    ) -> Float[LinearOperator, "*batch M N"]:
        return DiagLinearOperator(self._diag * other.unsqueeze(-1))

    def _size(self) -> torch.Size:
        # Though the super._size method works, this is more efficient
        diag_shapes = [linear_op._diag.shape for linear_op in self.linear_ops]
        batch_shape = torch.broadcast_shapes(*[diag_shape[:-1] for diag_shape in diag_shapes])
        N = 1
        for diag_shape in diag_shapes:
            N *= diag_shape[-1]
        return torch.Size([*batch_shape, N, N])

    def _symeig(
        self: Float[LinearOperator, "*batch N N"],
        eigenvectors: bool = False,
        return_evals_as_lazy: Optional[bool] = False,
    ) -> Tuple[Float[Tensor, "*batch M"], Optional[Float[LinearOperator, "*batch N M"]]]:
        # return_evals_as_lazy is a flag to return the eigenvalues as a lazy tensor
        # which is useful for root decompositions here (see the root_decomposition
        # method above)
        evals, evecs = [], []
        for lt in self.linear_ops:
            evals_, evecs_ = lt._symeig(eigenvectors=eigenvectors)
            evals.append(evals_)
            evecs.append(evecs_)
        evals = KroneckerProductDiagLinearOperator(*[DiagLinearOperator(evals_) for evals_ in evals])

        if not return_evals_as_lazy:
            evals = evals._diagonal()

        if eigenvectors:
            evecs = KroneckerProductDiagLinearOperator(*evecs)
        else:
            evecs = None
        return evals, evecs

    def abs(self) -> LinearOperator:
        """
        Returns a DiagLinearOperator with the absolute value of all diagonal entries.
        """
        return self.__class__(*[lt.abs() for lt in self.linear_ops])

    def exp(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        raise NotImplementedError(f"torch.exp({self.__class__.__name__}) is not implemented.")

    @cached
    def inverse(self: Float[LinearOperator, "*batch N N"]) -> Float[LinearOperator, "*batch N N"]:
        """
        Returns the inverse of the DiagLinearOperator.
        """
        # here we use that (A \kron B)^-1 = A^-1 \kron B^-1
        inverses = [lt.inverse() for lt in self.linear_ops]
        return self.__class__(*inverses)

    def log(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        raise NotImplementedError(f"torch.log({self.__class__.__name__}) is not implemented.")

    def sqrt(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        """
        Returns a DiagLinearOperator with the square root of all diagonal entries.
        """
        return self.__class__(*[lt.sqrt() for lt in self.linear_ops])

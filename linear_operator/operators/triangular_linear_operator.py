#!/usr/bin/env python3

from typing import Callable, List, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from linear_operator.utils.errors import NotPSDError
from linear_operator.utils.memoize import cached
from linear_operator.operators._linear_operator import IndexType, LinearOperator
from linear_operator.operators.batch_repeat_linear_operator import BatchRepeatLinearOperator
from linear_operator.operators.dense_linear_operator import DenseLinearOperator

Allsor = Union[Tensor, LinearOperator]


class _TriangularLinearOperatorBase:
    """Base class that all triangular lazy tensors are derived from."""

    pass


class TriangularLinearOperator(LinearOperator, _TriangularLinearOperatorBase):
    r"""
    A wrapper for LinearOperators when we have additional knowledge that it
    represents a lower- or upper-triangular matrix (or batch of matrices).

    :param tensor: A `... x N x N` Tensor, representing a (batch of)
        `N x N` triangular matrix.
    :param upper: If True, the tensor is considered to be upper-triangular, otherwise lower-triangular.
    """

    def __init__(self, tensor: Allsor, upper: bool = False) -> None:
        if isinstance(tensor, TriangularLinearOperator):
            # this is a null-op, we can just use underlying tensor directly.
            tensor = tensor._tensor
            # TODO: Use a metaclass to create a DiagLinearOperator if tensor is diagonal
        elif isinstance(tensor, BatchRepeatLinearOperator):
            # things get kind of messy when interleaving repeats and triangualrisms
            if not isinstance(tensor.base_linear_op, TriangularLinearOperator):
                tensor = tensor.__class__(
                    TriangularLinearOperator(tensor.base_linear_op, upper=upper),
                    batch_repeat=tensor.batch_repeat,
                )
        if torch.is_tensor(tensor):
            tensor = DenseLinearOperator(tensor)
        super().__init__(tensor, upper=upper)
        self.upper = upper
        self._tensor = tensor

    def __add__(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"], float],
    ) -> Union[Float[LinearOperator, "... M N"], Float[Tensor, "... M N"]]:
        from linear_operator.operators.diag_linear_operator import DiagLinearOperator

        if isinstance(other, DiagLinearOperator):
            from linear_operator.operators.added_diag_linear_operator import AddedDiagLinearOperator

            return self.__class__(AddedDiagLinearOperator(self._tensor, other), upper=self.upper)
        if isinstance(other, TriangularLinearOperator) and not self.upper ^ other.upper:
            return self.__class__(self._tensor + other._tensor, upper=self.upper)
        return self._tensor + other

    def _cholesky(
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        raise NotPSDError("TriangularLinearOperator does not allow a Cholesky decomposition")

    def _cholesky_solve(
        self: Float[LinearOperator, "*batch N N"],
        rhs: Union[Float[LinearOperator, "*batch2 N M"], Float[Tensor, "*batch2 N M"]],
        upper: Optional[bool] = False,
    ) -> Union[Float[LinearOperator, "... N M"], Float[Tensor, "... N M"]]:
        # use custom method if implemented
        try:
            res = self._tensor._cholesky_solve(rhs=rhs, upper=upper)
        except NotImplementedError:
            if upper:
                # res = (U.T @ U)^-1 @ v = U^-1 @ U^-T @ v
                w = self._transpose_nonbatch().solve(rhs)
                res = self.solve(w)
            else:
                # res = (L @ L.T)^-1 @ v = L^-T @ L^-1 @ v
                w = self.solve(rhs)
                res = self._transpose_nonbatch().solve(w)
        return res

    def _diagonal(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
        return self._tensor._diagonal()

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: Union[torch.Size, List[int]]
    ) -> Float[LinearOperator, "... M N"]:
        if len(batch_shape) == 0:
            return self
        return self.__class__(tensor=self._tensor._expand_batch(batch_shape), upper=self.upper)

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        return self._tensor._get_indices(row_index, col_index, *batch_indices)

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return self._tensor.matmul(rhs)

    def _mul_constant(
        self: Float[LinearOperator, "*batch M N"], other: Union[float, torch.Tensor]
    ) -> Float[LinearOperator, "*batch M N"]:
        return self.__class__(self._tensor * other.unsqueeze(-1), upper=self.upper)

    def _root_decomposition(
        self: Float[LinearOperator, "... N N"]
    ) -> Union[Float[torch.Tensor, "... N N"], Float[LinearOperator, "... N N"]]:
        raise NotPSDError("TriangularLinearOperator does not allow a root decomposition")

    def _root_inv_decomposition(
        self: Float[LinearOperator, "*batch N N"],
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[Float[LinearOperator, "... N N"], Float[Tensor, "... N N"]]:
        raise NotPSDError("TriangularLinearOperator does not allow an inverse root decomposition")

    def _size(self) -> torch.Size:
        return self._tensor.shape

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
        # already triangular, can just call solve for the solve
        return self.solve(rhs)

    def _sum_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self._tensor._sum_batch(dim), upper=self.upper)

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return self.__class__(self._tensor._transpose_nonbatch(), upper=not self.upper)

    def abs(self) -> LinearOperator:
        """
        Returns a TriangleLinearOperator with the absolute value of all diagonal entries.
        """
        return self.__class__(self._tensor.abs(), upper=self.upper)

    def add_diagonal(
        self: Float[LinearOperator, "*batch N N"],
        diag: Union[Float[torch.Tensor, "... N"], Float[torch.Tensor, "... 1"], Float[torch.Tensor, ""]],
    ) -> Float[LinearOperator, "*batch N N"]:
        added_diag_lt = self._tensor.add_diagonal(diag)
        return self.__class__(added_diag_lt, upper=self.upper)

    def to_dense(self: Float[LinearOperator, "*batch M N"]) -> Float[Tensor, "*batch M N"]:
        return self._tensor.to_dense()

    def exp(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        """
        Returns a TriangleLinearOperator with all diagonal entries exponentiated.
        """
        return self.__class__(self._tensor.exp(), upper=self.upper)

    def inv_quad_logdet(
        self: Float[LinearOperator, "*batch N N"],
        inv_quad_rhs: Optional[Union[Float[Tensor, "*batch N M"], Float[Tensor, "*batch N"]]] = None,
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[
        Optional[Union[Float[Tensor, "*batch M"], Float[Tensor, " *batch"], Float[Tensor, " 0"]]],
        Optional[Float[Tensor, "..."]],
    ]:
        if inv_quad_rhs is None:
            inv_quad_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            # triangular, solve is cheap
            inv_quad_term = (inv_quad_rhs * self.solve(inv_quad_rhs)).sum(dim=-2)
        if logdet:
            diag = self._diagonal()
            logdet_term = self._diagonal().abs().log().sum(-1)
            if torch.sign(diag).prod(-1) < 0:
                logdet_term = torch.full_like(logdet_term, float("nan"))
        else:
            logdet_term = torch.empty(0, dtype=self.dtype, device=self.device)
        if inv_quad_term.numel() and reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term, logdet_term

    @cached
    def inverse(self: Float[LinearOperator, "*batch N N"]) -> Float[LinearOperator, "*batch N N"]:
        """
        Returns the inverse of the DiagLinearOperator.
        """
        eye = torch.eye(self._tensor.size(-1), device=self._tensor.device, dtype=self._tensor.dtype)
        inv = self.solve(eye)
        return self.__class__(inv, upper=self.upper)

    def solve(
        self: Float[LinearOperator, "... N N"],
        right_tensor: Union[Float[Tensor, "... N P"], Float[Tensor, " N"]],
        left_tensor: Optional[Float[Tensor, "... O N"]] = None,
    ) -> Union[Float[Tensor, "... N P"], Float[Tensor, "... N"], Float[Tensor, "... O P"], Float[Tensor, "... O"]]:
        squeeze = False
        if right_tensor.dim() == 1:
            right_tensor = right_tensor.unsqueeze(-1)
            squeeze = True

        if isinstance(self._tensor, DenseLinearOperator):
            res = torch.linalg.solve_triangular(self.to_dense(), right_tensor, upper=self.upper)
        elif isinstance(self._tensor, BatchRepeatLinearOperator):
            res = self._tensor.base_linear_op.solve(right_tensor, left_tensor)
            # TODO: Proper broadcasting
            res = res.expand(self._tensor.batch_repeat + res.shape[-2:])
        else:
            # TODO: Can we be smarter here?
            res = self._tensor.solve(right_tensor=right_tensor, left_tensor=left_tensor)

        if squeeze:
            res = res.squeeze(-1)

        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def solve_triangular(
        self, rhs: torch.Tensor, upper: bool, left: bool = True, unitriangular: bool = False
    ) -> torch.Tensor:
        if upper != self.upper:
            raise RuntimeError(
                f"Incompatible argument: {self.__class__.__name__}.solve_triangular called with `upper={upper}`, "
                f"but `LinearOperator` has `upper={self.upper}`."
            )
        if not left:
            raise NotImplementedError(
                f"Argument `left=False` not yet supported for {self.__class__.__name__}.solve_triangular."
            )
        if unitriangular:
            raise NotImplementedError(
                f"Argument `unitriangular=True` not yet supported for {self.__class__.__name__}.solve_triangular."
            )
        return self.solve(right_tensor=rhs)

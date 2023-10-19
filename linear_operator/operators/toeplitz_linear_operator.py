#!/usr/bin/env python3
from typing import Callable, List, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.errors import NotPSDError
from ..utils.toeplitz import toeplitz_derivative_quadratic_form, toeplitz_matmul, toeplitz_solve_ld, toeplitz_inverse
from ._linear_operator import IndexType, LinearOperator


class ToeplitzLinearOperator(LinearOperator):
    def __init__(self, column, row=None):
        """
        Construct a Toeplitz matrix.
        The Toeplitz matrix has constant diagonals, with `column` as its first
        column and `row` as its first row. If `row` is not given, 
        `row == conjugate(column)` is assumed.
        
        Args:
            :attr: `column` (Tensor)
                First column of the matrix. If `column` is a 1D Tensor of length `n`, this represents a
                Toeplitz matrix with `column` as its first column.
                If `column` is `b_1 x b_2 x ... x b_k x n`, then this represents a batch
                `b_1 x b_2 x ... x b_k` of Toeplitz matrices.
            :attr: `row` (Tensor)
                First row of the matrix If `row` is a 1D Tensor of length `n`, this represents a
                Toeplitz matrix with `row` as its row column. 
                `row` tensor must have the same size as `column`, with `column[...,0]`
                equal to `row[...,0]`.
                If `row` is `None` or is not supplied, assumes `row == conjugate(column)`.
                If `row[0]` is real, the result is a Hermitian matrix. 
        """
        self.column = column
        if row is None:
            super(ToeplitzLinearOperator, self).__init__(column)
            self.sym = True
            myrow = column.conj()
            myrow.data[...,0] = column[...,0]
            self.row = myrow
        else:
            super(ToeplitzLinearOperator, self).__init__(column, row)
            self.sym = False
            self.row = row
            if torch.any(row[...,0] != column[...,0]):
                raise ValueError("The first elements in column does not match the first values in row")
            if torch.allclose(row, column.conj()):
                self.sym = True
    
    def _cholesky(
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        if not self.sym:
            #Cholesky decompositions are for Hermitian matrix
            raise NotPSDError("Non-symmetric ToeplitzLinearOperator does not allow a Cholesky decomposition")
        return super(ToeplitzLinearOperator, self)._cholesky(upper)

    def _diagonal(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
        diag_term = self.column[..., 0]
        size = min(self.column.size(-1), self.row.size(-1))
        if self.column.ndimension() > 1:
            diag_term = diag_term.unsqueeze(-1)
        return diag_term.expand(*self.column.size()[:-1], size)

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: Union[torch.Size, List[int]]
    ) -> Float[LinearOperator, "... M N"]:
        #return self.__class__(self.column.expand(*batch_shape, self.column.size(-1)))
        if self.sym:
            return self.__class__(self.column.expand(*batch_shape, self.column.size(-1)))
        else:
            return self.__class__(
                self.column.expand(*batch_shape, self.column.size(-1)),
                self.row.expand(*batch_shape, self.row.size(-1)),
            )

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        toeplitz_indices = (row_index - col_index).fmod(self.size(-1)).abs().long()
        res = torch.where(row_index > col_index, self.column[(*batch_indices, toeplitz_indices)], self.row[(*batch_indices, toeplitz_indices)])
        return res #self.column[(*batch_indices, toeplitz_indices)]

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return toeplitz_matmul(self.column, self.row, rhs)

    def _t_matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[Tensor, "*batch2 M P"], Float[LinearOperator, "*batch2 M P"]],
    ) -> Union[Float[LinearOperator, "... N P"], Float[Tensor, "... N P"]]:
        return toeplitz_matmul(self.row, self.column, rhs)
        
    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        
        res_c, res_r = toeplitz_derivative_quadratic_form(left_vecs, right_vecs)

        # Collapse any expanded broadcast dimensions
        if res_c.dim() > self.column.dim():
            res_c = res_c.view(-1, *self.column.shape).sum(0)
        if res_r.dim() > self.row.dim():
            res_r = res_r.view(-1, *self.row.shape).sum(0)
        
        res_r[...,0] = 0. #set it to zero as already in res_c[...,0]
        
        if self.sym:
            return (res_c + res_r,)
        else:
            return (res_c, res_r,)

    def _root_decomposition(
        self: Float[LinearOperator, "... N N"]
    ) -> Union[Float[torch.Tensor, "... N N"], Float[LinearOperator, "... N N"]]:
        if not self.sym:
            raise NotPSDError("Non-symmetric ToeplitzLinearOperator does not allow a root decomposition")
        return super(ToeplitzLinearOperator, self)._root_decomposition()

    def _root_inv_decomposition(
        self: Float[LinearOperator, "*batch N N"],
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[Float[LinearOperator, "... N N"], Float[Tensor, "... N N"]]:
        if not self.sym:
            raise NotPSDError("Non-symmetric ToeplitzLinearOperator does not allow an inverse root decomposition")
        return super(ToeplitzLinearOperator, self)._root_inv_decomposition(initial_vectors, test_vectors)

    def _size(self) -> torch.Size:
        return torch.Size((*self.row.shape, self.column.size(-1)))
    
    def solve(
        self: Float[LinearOperator, "... N N"],
        right_tensor: Union[Float[Tensor, "... N P"], Float[Tensor, " N"]],
        left_tensor: Optional[Float[Tensor, "... O N"]] = None,
    ) -> Union[Float[Tensor, "... N P"], Float[Tensor, "... N"], Float[Tensor, "... O P"], Float[Tensor, "... O"]]:
        squeeze = False
        if right_tensor.dim() == 1:
            rhs_ = right_tensor.unsqueeze(-1)
            squeeze = True
        else:
            rhs_ = right_tensor
        res = toeplitz_solve_ld(self.column, self.row, rhs_)
        if squeeze:
            res = res.squeeze(-1)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        if self.sym:
            return ToeplitzLinearOperator(self.column)
        else:
            myrow = torch.cat([self.column[...,0].unsqueeze(-1), self.row[...,1:]], dim=-1)
            mycol = torch.clone(self.column)
            myrow.data[...,0] = mycol[...,0]
            return ToeplitzLinearOperator(myrow, mycol)

    def add_jitter(
        self: Float[LinearOperator, "*batch N N"], jitter_val: float = 1e-3
    ) -> Float[LinearOperator, "*batch N N"]:
        jitter = torch.zeros_like(self.column)
        jitter.narrow(-1, 0, 1).fill_(jitter_val)
        if self.sym:
            return ToeplitzLinearOperator(self.column.add(jitter))
        else:
            return ToeplitzLinearOperator(self.column.add(jitter), self.row.add(jitter))

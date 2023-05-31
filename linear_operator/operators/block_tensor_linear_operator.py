from typing import List, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ._linear_operator import LinearOperator
from .dense_linear_operator import to_linear_operator


class BlockTensorLinearOperator(LinearOperator):
    def __init__(self, linear_operators: List[List[LinearOperator]]) -> None:
        assert len(linear_operators) > 0, "must have nested list"
        assert len(linear_operators[0]) == len(linear_operators), "must be square over block dimensions"

        super().__init__(linear_operators)

        self.linear_operators = linear_operators
        self.num_tasks = len(self.linear_operators)
        self.block_rows = linear_operators[0][0].shape[0]
        self.block_cols = linear_operators[0][0].shape[1]

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:

        T = self.num_tasks

        # A is block [N * T1, M * T2] and B is block [O * S1, P * S2]. If A and B have conformal block counts
        # ie T2==S1 as well as M==O then use the blockwise algorithm. Else use to_dense()
        if isinstance(rhs, self.__class__) and self.num_tasks == rhs.num_tasks and self.block_cols == rhs.block_rows:
            output = []
            for i in range(T):
                tmp = []
                for j in range(T):
                    tmp.append([])
                output.append(tmp)
            for i in range(T):
                for j in range(T):
                    out_ij = self.linear_operators[i][0] @ rhs.linear_operators[0][j]
                    for k in range(1, T):
                        out_ij += self.linear_operators[i][k] @ rhs.linear_operators[k][j]
                    output[i][j] = out_ij
            return self.__class__(output)
        elif isinstance(rhs, Tensor):
            # Check both matrix dims divisible by T,
            # reshape to (T, T, ), call block multiplication
            if rhs.size(0) % T == 0 and rhs.size(1) % T == 0:
                # A is block [N * T, M * T] and B is a general tensor/operator of shape [O, P].
                # If O and P are both divisible by T,
                # then interpret B as a [O//T * T, P//T * T] block matrix
                O_T = rhs.size(0) // T
                P_T = rhs.size(1) // T
                rhs_blocks_raw = rhs.reshape(T, O_T, T, P_T)
                rhs_blocks = rhs_blocks_raw.permute(0, 2, 1, 3)
                rhs_op = BlockTensorLinearOperator.from_tensor(rhs_blocks, T)
                return self._matmul(rhs_op)

        A = self.to_dense()
        B = rhs.to_dense()
        res = A @ B
        return res

    def to_dense(self: Float[LinearOperator, "*batch M N"]) -> Float[Tensor, "*batch M N"]:
        out = []
        for i in range(self.num_tasks):
            rows = []
            for j in range(self.num_tasks):
                rows.append(self.linear_operators[i][j].to_dense())
            out.append(torch.concat(rows, axis=1))
        return torch.concat(out, axis=0)

    def _size(self) -> torch.Size:
        sz = self.linear_operators[0][0].size()
        return torch.Size([self.num_tasks * sz[0], self.num_tasks * sz[1]])

    def _diag(self):
        out = []
        for i in range(self.num_tasks):
            diagonal = self.linear_operators[i][i].diagonal()
            out.append(diagonal)
        return torch.concat(out, axis=1)

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return self  # Diagonal matrices are symmetric

    @classmethod
    def from_tensor(cls, tensor: Tensor, num_tasks: int):
        linear_ops = [
            [to_linear_operator(t[0]) for t in list(torch.tensor_split(tensor[i], num_tasks))] for i in range(num_tasks)
        ]
        return cls(linear_ops)

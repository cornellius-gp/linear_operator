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

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:

        T = self.num_tasks
        output = []
        for i in range(T):
            tmp = []
            for j in range(T):
                tmp.append([])
            output.append(tmp)

        if isinstance(rhs, self.__class__):
            # TO DO: Check size is the same
            for i in range(T):
                for j in range(T):
                    out_ij = to_linear_operator(
                        torch.zeros(self.linear_operators[0][0].shape[0], rhs.linear_operators[0][0].shape[1])
                    )
                    for k in range(T):
                        out_ij += self.linear_operators[i][k] @ rhs.linear_operators[k][j]
                    output[i][j] = out_ij
        elif isinstance(rhs, Tensor):
            # Check both matrix dims divisible by T,
            # reshape to (T, T, ), call .from_tensor
            pass

        elif isinstance(rhs, LinearOperator):
            pass

        else:
            raise Exception("")

        return self.__class__(output)

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

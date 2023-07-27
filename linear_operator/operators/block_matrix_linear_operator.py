import math
from typing import List, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor

from .. import settings
from ._linear_operator import IndexType, LinearOperator, to_dense
from .dense_linear_operator import DenseLinearOperator
from .zero_linear_operator import ZeroLinearOperator


class BlockMatrixLinearOperator(LinearOperator):
    """
    A TxT block matrix of LinearOperators.

    Idea. Represent [TN, TM] tensors by TxT blocks of NxM lazy tensors.

    Implementation. A block linear operator class that can keep track of the [T, T] block structure,
    represented as T^2 lazy tensors of the same shape. Implement matrix multiplication between block matrices as
    the appropriate linear operators on the blocks.

    :param linear_operators: A T^2 (flattened) list of linear operators representing a 2-D TxT block matrix.
        The list of linear operators should be flattened into a concatenation of block-rowsa.
    """

    def __init__(self, *flattened_linear_operators: LinearOperator) -> None:
        self.num_tasks = int(math.sqrt(len(flattened_linear_operators)))

        if settings.debug.on():
            assert len(flattened_linear_operators) > 0, "must have non-empty list"
            assert self.num_tasks**2 == len(flattened_linear_operators)

        super().__init__(*flattened_linear_operators)

        self.linear_operators = tuple(
            flattened_linear_operators[i * self.num_tasks : (i + 1) * self.num_tasks] for i in range(self.num_tasks)
        )
        self.block_rows = self.linear_operators[0][0].shape[0]
        self.block_cols = self.linear_operators[0][0].shape[1]

    @staticmethod
    def create_square_ops_output(T: int) -> List[List[LinearOperator]]:
        """Return an empty (square) list of operators of shape TxT"""
        ops = []
        for i in range(T):
            tmp = []
            for j in range(T):
                tmp.append([])
            ops.append(tmp)
        return ops

    def _matmul_two_block_matrix_linear_operators(
        self: "BlockMatrixLinearOperator",
        other: "BlockMatrixLinearOperator",
    ) -> "BlockMatrixLinearOperator":
        assert self.num_tasks == other.num_tasks
        assert self.block_cols == other.block_rows

        T = self.num_tasks
        output = []
        for i in range(T):
            for j in range(T):
                out_ij = self.linear_operators[i][0] @ other.linear_operators[0][j]
                for k in range(1, T):
                    out_ij += self.linear_operators[i][k] @ other.linear_operators[k][j]
                output.append(out_ij)
        return self.__class__(*output)

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        T = self.num_tasks

        if isinstance(rhs, Tensor) and rhs.ndim == 2:
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
                rhs_op = BlockMatrixLinearOperator.from_tensor(rhs_blocks, T)
                return self._matmul_two_block_matrix_linear_operators(rhs_op).to_dense()

        # Failover implementation. Convert to dense and multiply matricies
        # Batch logic is not supported for now
        assert rhs.dim() <= 2
        A = self.to_dense()
        B = to_dense(rhs)

        res = A @ B
        return res

    def matmul(
        self: Float[LinearOperator, "*batch M N"],
        other: Union[Float[Tensor, "*batch2 N P"], Float[Tensor, "*batch2 N"], Float[LinearOperator, "*batch2 N P"]],
    ) -> Union[Float[Tensor, "... M P"], Float[Tensor, "... M"], Float[LinearOperator, "... M P"]]:
        # A is block [N * T1, M * T2] and B is block [O * S1, P * S2]. If A and B have conformal block counts
        # ie T2==S1 as well as M==O then use the blockwise algorithm. Else use to_dense()
        if isinstance(other, self.__class__):
            if self.num_tasks == other.num_tasks and self.block_cols == other.block_rows:
                return self._matmul_two_block_matrix_linear_operators(other)
        elif isinstance(other, LinearOperator):
            from .matmul_linear_operator import MatmulLinearOperator

            return MatmulLinearOperator(self, other)

        # The base method wants to perform a matmul via broadcasting and a
        # representation tree which this operator doesn't support.
        return self._matmul(other)

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

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self.linear_operators[0][0].dtype

    @property
    def device(self) -> Optional[torch.device]:
        return self.linear_operators[0][0].device

    def _diag(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
        out = []
        for i in range(self.num_tasks):
            # The underlying operators will test if they are square
            diagonal = self.linear_operators[i][i].diagonal()
            out.append(diagonal)
        return torch.concat(out, axis=1)

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        out = []
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                out.append(self.linear_operators[j][i].mT)
        return BlockMatrixLinearOperator(*out)

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        # Perform the __getitem__
        tsr = self.to_dense()
        res = tsr[(*batch_indices, row_index, col_index)]
        return DenseLinearOperator(res)

    @classmethod
    def from_tensor(cls, tensor: Tensor, num_tasks: int) -> "BlockMatrixLinearOperator":
        def tensor_to_linear_op(t: Tensor) -> LinearOperator:
            if torch.count_nonzero(t) > 0:
                return DenseLinearOperator(t)
            return ZeroLinearOperator(*t.size(), dtype=t.dtype, device=t.device)

        linear_ops = [
            tensor_to_linear_op(t[0]) for i in range(num_tasks) for t in list(torch.tensor_split(tensor[i], num_tasks))
        ]
        return cls(*linear_ops)

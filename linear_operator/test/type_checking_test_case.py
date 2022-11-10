# These are tests to directly check torchtyping signatures as extended to LinearOperator.
# The idea is to verify that dimension tests are working as expected.
import unittest
from typing import Union

import torch
from jaxtyping import Float, jaxtyped

# Use your favourite typechecker: usually one of the two lines below.
from typeguard import typechecked as typechecker

from linear_operator.operators import DenseLinearOperator, LinearOperator


@jaxtyped
@typechecker
def linop_matmul_fn(
    lo: Float[LinearOperator, "*batch M N"],
    vec: Union[Float[torch.Tensor, "*batch N C"], Float[torch.Tensor, "*batch N"]],
) -> Union[Float[torch.Tensor, "*batch M C"], Float[torch.Tensor, "*batch M"]]:
    r"""
    Performs a matrix multiplication :math:`\mathbf KM` with the (... x M x N) matrix :math:`\mathbf K`
    that lo represents. Should behave as
    :func:`torch.matmul`. If the LinearOperator represents a batch of
    matrices, this method should therefore operate in batch mode as well.

    :param lo: the K = MxN left hand matrix
    :param vec: the matrix :math:`\mathbf M` to multiply with (... x N x C).
    :return: :math:`\mathbf K \mathbf M` (... x M x C)
    """
    res = lo.matmul(vec)
    return res


@jaxtyped
@typechecker
def linop_matmul_fn_bad_lo(
    lo: Float[LinearOperator, " N"],
    vec: Union[Float[torch.Tensor, "*batch N C"], Float[torch.Tensor, "*batch N"]],
) -> Union[Float[torch.Tensor, "*batch M C"], Float[torch.Tensor, "*batch M"]]:
    r"""
    As above, but with bad size array for lo
    """
    res = lo.matmul(vec)
    return res


@jaxtyped
@typechecker
def linop_matmul_fn_bad_vec(
    lo: Float[LinearOperator, "*batch M N"], vec: Float[torch.Tensor, "*batch N C"]
) -> Union[Float[torch.Tensor, "*batch M C"], Float[torch.Tensor, "*batch M"]]:
    r"""
    As above, but with bad size array for vec
    """
    res = lo.matmul(vec)
    return res


@jaxtyped
@typechecker
def linop_matmul_fn_bad_retn(
    lo: Float[LinearOperator, "*batch M N"],
    vec: Union[Float[torch.Tensor, "*batch N C"], Float[torch.Tensor, "*batch N"]],
) -> Float[torch.Tensor, "*batch M C"]:
    r"""
    As above, but with bad return size
    """
    res = lo.matmul(vec)
    return res


class TestTypeChecking(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mat = torch.tensor([[3, -1, 0], [-1, 3, 0], [0, 0, 3]], dtype=torch.float)
        self.vec = torch.randn(3)
        self.lo = DenseLinearOperator(mat)

    def test_linop_matmul_fn(self):
        linop_matmul_fn(self.lo, self.vec)

    def test_linop_matmul_fn_bad_lo(self):
        with self.assertRaises(TypeError):
            linop_matmul_fn_bad_lo(self.lo, self.vec)

    def test_linop_matmul_fn_bad_vec(self):
        with self.assertRaises(TypeError):
            linop_matmul_fn_bad_vec(self.lo, self.vec)

    def test_linop_matmul_fn_bad_retn(self):
        with self.assertRaises(TypeError):
            linop_matmul_fn_bad_retn(self.lo, self.vec)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import (
    BlockDiagLinearOperator,
    BlockInterleavedLinearOperator,
    DenseLinearOperator,
    SumBatchLinearOperator,
)


class TestBlockLinearOperatorBatchScaling(unittest.TestCase):
    def test_batch_scaling_forward_and_gradients(self) -> None:
        cases = [
            ((2,), 3, (2,)),
            ((3,), 3, (3,)),
            ((2,), 1, (2,)),
            ((2, 3), 4, (2, 1)),
            ((2, 3), 3, (1, 3)),
        ]
        for operator in (SumBatchLinearOperator, BlockDiagLinearOperator, BlockInterleavedLinearOperator):
            for dtype in (torch.float32, torch.float64):
                for batch_shape, n_blocks, scale_shape in cases:
                    with self.subTest(
                        operator=operator.__name__, dtype=dtype, case=(batch_shape, n_blocks, scale_shape)
                    ):
                        shape = (*batch_shape, n_blocks, 2, 2)
                        blocks = torch.arange(1, torch.Size(shape).numel() + 1, dtype=dtype).reshape(shape)
                        blocks.requires_grad_(True)
                        scales = torch.linspace(-2, 3, torch.Size(scale_shape).numel(), dtype=dtype).reshape(
                            scale_shape
                        )
                        scales.requires_grad_(True)
                        linear_op = operator(DenseLinearOperator(blocks))

                        # Assemble the reference without using a block LinearOperator.
                        if operator is SumBatchLinearOperator:
                            dense = blocks.sum(-3)
                        else:
                            identity = torch.eye(n_blocks, dtype=dtype)
                            if operator is BlockDiagLinearOperator:
                                dense = torch.einsum("...bij,bc->...bicj", blocks, identity)
                            else:
                                dense = torch.einsum("...bij,bc->...ibjc", blocks, identity)
                            dense = dense.reshape(*batch_shape, 2 * n_blocks, 2 * n_blocks)
                        dense = dense * scales[..., None, None]

                        scaled_op = linear_op * scales[..., None, None]
                        self.assertIsInstance(scaled_op, operator)
                        torch.testing.assert_close(scaled_op.to_dense(), dense)

                        rhs = torch.linspace(-1, 1, dense.numel(), dtype=dtype).reshape(dense.shape)
                        rhs.requires_grad_(True)
                        actual = scaled_op @ rhs
                        expected = dense @ rhs
                        torch.testing.assert_close(actual, expected)

                        weights = torch.linspace(1, 2, actual.numel(), dtype=dtype).reshape(actual.shape)
                        inputs = (blocks, scales, rhs)
                        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
                        expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
                        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()

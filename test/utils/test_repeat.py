import unittest

import torch

from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.test.utils import approx_equal


class TestRepeat(unittest.TestCase):
    def make_example(self):
        return DenseLinearOperator(torch.randn(3, 3))

    def test_repeat(self):
        example = self.make_example()
        repeated = example.repeat(2, 1, 1)
        repeated_dense = example.to_dense().repeat(2, 1, 1)
        self.assertTrue(approx_equal(repeated.to_dense(), repeated_dense))

    def test_repeat_noop(self):
        example = self.make_example()
        repeated = example.repeat(1, 1)
        self.assertTrue(approx_equal(repeated.to_dense(), example.to_dense()))
        self.assertIsInstance(repeated, DenseLinearOperator)  # ensure that fast path is taken

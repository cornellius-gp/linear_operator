#!/usr/bin/env python3

import math
import unittest
from unittest.mock import MagicMock, patch

import torch

import linear_operator
from linear_operator.operators import LowRankRootAddedDiagLinearOperator, LowRankRootLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestLowRankRootAddedDiagLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_op(self):
        tensor = torch.randn(5, 2)
        diag = torch.tensor([1.0, 2.0, 4.0, 2.0, 3.0])
        lt = LowRankRootLinearOperator(tensor).add_diagonal(diag)
        assert isinstance(lt, LowRankRootAddedDiagLinearOperator)
        return lt

    def evaluate_linear_op(self, linear_op):
        diag = linear_op._diag_tensor._diag
        root = linear_op._linear_op.root.tensor
        return root @ root.mT + diag.diag_embed(dim1=-2, dim2=-1)

    def test_root_decomposition_cholesky(self):
        self.test_root_decomposition(cholesky=True)


class TestLowRankRootAddedDiagLinearOperatorBatch(TestLowRankRootAddedDiagLinearOperator):
    seed = 4
    should_test_sample = True

    def create_linear_op(self):
        tensor = torch.randn(3, 5, 2)
        diag = torch.tensor([[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]])
        lt = LowRankRootLinearOperator(tensor).add_diagonal(diag)
        assert isinstance(lt, LowRankRootAddedDiagLinearOperator)
        return lt

    def evaluate_linear_op(self, linear_op):
        diag = linear_op._diag_tensor._diag
        root = linear_op._linear_op.root.tensor
        return root @ root.mT + diag.diag_embed(dim1=-2, dim2=-1)


class TestLowRankRootAddedDiagLinearOperatorMultiBatch(TestLowRankRootAddedDiagLinearOperator):
    seed = 4
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        tensor = torch.randn(4, 3, 5, 2)
        diag = torch.tensor([[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]]).repeat(
            4, 1, 1
        )
        lt = LowRankRootLinearOperator(tensor).add_diagonal(diag)
        assert isinstance(lt, LowRankRootAddedDiagLinearOperator)
        return lt

    def evaluate_linear_op(self, linear_op):
        diag = linear_op._diag_tensor._diag
        root = linear_op._linear_op.root.tensor
        return root @ root.mT + diag.diag_embed(dim1=-2, dim2=-1)


if __name__ == "__main__":
    unittest.main()

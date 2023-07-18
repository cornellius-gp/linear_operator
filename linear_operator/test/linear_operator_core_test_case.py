#!/usr/bin/env python3

from abc import abstractmethod

import torch

import linear_operator
from linear_operator.operators import DiagLinearOperator, to_dense
from .base_test_case import BaseTestCase

"""
From the project description, a LinearOperator is a class that:

- specifies the tensor(s) needed to define the LinearOperator,
- specifies a _matmul function (how the LinearOperator is applied to a vector),
- specifies a _size function (how big is the LinearOperator if it is represented as a matrix, or batch of matrices), and
- specifies a _transpose_nonbatch function (the adjoint of the LinearOperator).
- (optionally) defines other functions (e.g. logdet, eigh, etc.) to accelerate computations for which efficient
  sturcture-exploiting routines exist.

What follows is a class to test these core LinearOperator operations.
Note that batch operations are excluded here since they are not part of the core definition.
"""


class CoreLinearOperatorTestCase(BaseTestCase):
    """Test the core operations for a LinearOperator"""

    tolerances = {
        "matmul": {"rtol": 1e-3},
        "transpose": {"rtol": 1e-4, "atol": 1e-5},
    }

    @abstractmethod
    def create_linear_op(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_linear_op(self):
        raise NotImplementedError()

    def _test_matmul(self, rhs):
        linear_op = self.create_linear_op().detach().requires_grad_(True)
        linear_op_copy = torch.clone(linear_op).detach().requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)
        rhs_evaluated = to_dense(rhs)

        # Test operator
        res = linear_op @ rhs
        actual = evaluated.matmul(rhs_evaluated)
        res_evaluated = to_dense(res)
        self.assertAllClose(res_evaluated, actual)

        # Test __torch_function__
        res = torch.matmul(linear_op, rhs)
        actual = evaluated.matmul(rhs)
        self.assertAllClose(to_dense(res), actual)

    def test_transpose_nonbatch(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        res = linear_op._transpose_nonbatch()
        actual = evaluated.mT
        res_evaluated = to_dense(res)
        self.assertAllClose(res_evaluated, actual, **self.tolerances["transpose"])

    def _test_rmatmul(self, lhs):
        # Note. transpose_nonbatch is tested implicitly here because
        # the base linear operator class defines
        # def rmatmul(other):
        #     return self.mT.matmul(other.mT).mT
        linear_op = self.create_linear_op().detach().requires_grad_(True)
        linear_op_copy = torch.clone(linear_op).detach().requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)

        # Test operator
        res = lhs @ linear_op
        res_evaluated = to_dense(res)
        actual = lhs @ evaluated
        self.assertAllClose(res_evaluated, actual)

        # Test __torch_function__
        res = torch.matmul(lhs, linear_op)
        res_evaluated = to_dense(res)
        actual = torch.matmul(lhs, evaluated)
        self.assertAllClose(res_evaluated, actual)

    def test_add(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        rhs = torch.randn(linear_op.shape)
        # Test operator functionality
        a = (linear_op + rhs).to_dense()
        b = evaluated + rhs
        self.assertAllClose(a, b)
        self.assertAllClose((linear_op + rhs).to_dense(), evaluated + rhs)
        self.assertAllClose((rhs + linear_op).to_dense(), evaluated + rhs)
        # Test __torch_function__ functionality
        self.assertAllClose(torch.add(linear_op, rhs).to_dense(), evaluated + rhs)
        self.assertAllClose(torch.add(rhs, linear_op).to_dense(), evaluated + rhs)

        rhs = torch.randn(linear_op.matrix_shape)
        self.assertAllClose((linear_op + rhs).to_dense(), evaluated + rhs)

        # rhs = torch.randn(2, *linear_op.shape)
        # self.assertAllClose((linear_op + rhs).to_dense(), evaluated + rhs)

        self.assertAllClose((linear_op + linear_op).to_dense(), evaluated * 2)
        return linear_op, evaluated

    def test_matmul_vec(self):
        linear_op = self.create_linear_op()

        # We skip this test if we're dealing with batch LinearOperators
        # They shouldn't multiply by a vec
        if linear_op.ndimension() > 2:
            return

        rhs = torch.randn(linear_op.size(-1))
        return self._test_matmul(rhs)

    def test_constant_mul(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        # Test operator functionality
        self.assertAllClose((linear_op * 5.0).to_dense(), evaluated * 5.0)
        self.assertAllClose((linear_op * torch.tensor(5.0)).to_dense(), evaluated * 5.0)
        self.assertAllClose((5.0 * linear_op).to_dense(), evaluated * 5.0)
        self.assertAllClose((torch.tensor(5.0) * linear_op).to_dense(), evaluated * 5.0)

        # Test __torch_function__ functionality
        self.assertAllClose(torch.mul(linear_op, torch.tensor(5.0)).to_dense(), evaluated * 5.0)
        self.assertAllClose(torch.mul(torch.tensor(5.0), linear_op).to_dense(), evaluated * 5.0)

    def test_constant_mul_neg(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)
        self.assertAllClose((linear_op * -5.0).to_dense(), evaluated * -5.0)

    def test_constant_div(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        # Test operator functionality
        self.assertAllClose((linear_op / 5.0).to_dense(), evaluated / 5.0)
        self.assertAllClose((linear_op / torch.tensor(5.0)).to_dense(), evaluated / 5.0)

        # Test __torch_function__ functionality
        self.assertAllClose(torch.div(linear_op, torch.tensor(5.0)).to_dense(), evaluated / 5.0)

    def test_to_dense(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)
        self.assertAllClose(linear_op.to_dense(), evaluated)

    def test_getitem(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        # Non-batch case
        if linear_op.ndimension() == 2:
            res = linear_op[1]
            actual = evaluated[1]
            self.assertAllClose(res, actual)
            res = linear_op[0:2].to_dense()
            actual = evaluated[0:2]
            self.assertAllClose(res, actual)
            res = linear_op[:, 0:2].to_dense()
            actual = evaluated[:, 0:2]
            self.assertAllClose(res, actual)
            res = linear_op[0:2, :].to_dense()
            actual = evaluated[0:2, :]
            self.assertAllClose(res, actual)
            res = linear_op[..., 0:2].to_dense()
            actual = evaluated[..., 0:2]
            self.assertAllClose(res, actual)
            res = linear_op[0:2, ...].to_dense()
            actual = evaluated[0:2, ...]
            self.assertAllClose(res, actual)
            res = linear_op[..., 0:2, 2]
            actual = evaluated[..., 0:2, 2]
            self.assertAllClose(res, actual)
            res = linear_op[0:2, ..., 2]
            actual = evaluated[0:2, ..., 2]
            self.assertAllClose(res, actual)

    def test_getitem_tensor_index(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        # Non-batch case
        if linear_op.ndimension() == 2:
            index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
            res, actual = linear_op[index], evaluated[index]
            self.assertAllClose(res, actual)
            index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
            res, actual = linear_operator.to_dense(linear_op[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
            res, actual = linear_operator.to_dense(linear_op[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (torch.tensor([0, 0, 1, 2]), Ellipsis)
            res, actual = linear_operator.to_dense(linear_op[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (Ellipsis, torch.tensor([0, 0, 1, 2]))
            res, actual = linear_operator.to_dense(linear_op[index]), evaluated[index]
            self.assertAllClose(res, actual)
            index = (Ellipsis, torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
            res, actual = linear_op[index], evaluated[index]
            self.assertAllClose(res, actual)

    def test_getitem_broadcasted_tensor_index(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        # Non-batch case
        if linear_op.ndimension() == 2:
            index = (
                torch.tensor([0, 0, 1, 2]).unsqueeze(-1),
                torch.tensor([0, 1, 0, 2]).unsqueeze(-2),
            )
            res, actual = linear_op[index], evaluated[index]
            self.assertAllClose(res, actual)
            index = (
                Ellipsis,
                torch.tensor([0, 0, 1, 2]).unsqueeze(-2),
                torch.tensor([0, 1, 0, 2]).unsqueeze(-1),
            )
            res, actual = linear_op[index], evaluated[index]
            self.assertAllClose(res, actual)

    def test_permute(self):
        linear_op = self.create_linear_op()
        if linear_op.dim() >= 4:
            evaluated = self.evaluate_linear_op(linear_op)
            dims = torch.randperm(linear_op.dim() - 2).tolist()

            # Call using __torch_function__
            res = torch.permute(linear_op, (*dims, -2, -1)).to_dense()
            actual = torch.permute(evaluated, (*dims, -2, -1))
            self.assertAllClose(res, actual)

            # Call using method
            res = linear_op.permute(*dims, -2, -1).to_dense()
            actual = torch.permute(evaluated, (*dims, -2, -1))
            self.assertAllClose(res, actual)

    def test_rmatmul_vec(self):
        linear_op = self.create_linear_op()

        # We skip this test if we're dealing with batch LinearOperators
        # They shouldn't multiply by a vec
        if linear_op.ndimension() > 2:
            return

        lhs = torch.randn(linear_op.size(-2))
        return self._test_rmatmul(lhs)

    def test_matmul_matrix(self):
        linear_op = self.create_linear_op()
        rhs = torch.randn(*linear_op.batch_shape, linear_op.size(-1), 4)
        return self._test_matmul(rhs)

    def test_rmatmul_matrix(self):
        linear_op = self.create_linear_op()
        lhs = torch.randn(*linear_op.batch_shape, 4, linear_op.size(-2))
        return self._test_rmatmul(lhs)

    def test_matmul_diag_matrix(self):
        linear_op = self.create_linear_op()
        diag = torch.rand(*linear_op.batch_shape, linear_op.size(-1))
        rhs = DiagLinearOperator(diag)
        return self._test_matmul(rhs)

    def test_rsub(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        rhs = torch.randn(linear_op.shape)
        # Test operator functionality
        self.assertAllClose((rhs - linear_op).to_dense(), rhs - evaluated)
        # Test __torch_function__ functionality
        self.assertAllClose(torch.sub(rhs, linear_op).to_dense(), rhs - evaluated)

    def test_sub(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        rhs = torch.randn(linear_op.shape)
        # Test operator functionality
        self.assertAllClose((linear_op - rhs).to_dense(), evaluated - rhs)
        # Test __torch_function__ functionality
        self.assertAllClose(torch.sub(linear_op, rhs).to_dense(), evaluated - rhs)

    def test_sum(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)

        self.assertAllClose(torch.sum(linear_op, -1), torch.sum(evaluated, -1))
        self.assertAllClose(torch.sum(linear_op, -2), torch.sum(evaluated, -2))
        if linear_op.ndimension() > 2:
            self.assertAllClose(torch.sum(linear_op, -3).to_dense(), torch.sum(evaluated, -3))
        if linear_op.ndimension() > 3:
            self.assertAllClose(torch.sum(linear_op, -4).to_dense(), torch.sum(evaluated, -4))

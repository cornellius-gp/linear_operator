#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import ZeroLinearOperator
from linear_operator.test.utils import approx_equal


class TestZeroLinearOperator(unittest.TestCase):
    def test_to_dense(self):
        lv = ZeroLinearOperator(5, 4, 3)
        actual = torch.zeros(5, 4, 3)
        res = lv.to_dense()
        self.assertLess(torch.norm(res - actual), 1e-4)

    def test_getitem(self):
        lv = ZeroLinearOperator(5, 4, 3)

        res_one = lv[0].to_dense()
        self.assertLess(torch.norm(res_one - torch.zeros(4, 3)), 1e-4)
        res_two = lv[:, 1, :]
        self.assertLess(torch.norm(res_two - torch.zeros(5, 3)), 1e-4)
        res_three = lv[:, :, 2]
        self.assertLess(torch.norm(res_three - torch.zeros(5, 4)), 1e-4)

    def test_getitem_complex(self):
        lv = ZeroLinearOperator(5, 4, 3)

        res_one = lv[[0, 1]].to_dense()
        self.assertLess(torch.norm(res_one - torch.zeros(2, 4, 3)), 1e-4)
        res_two = lv[:, [0, 1], :].to_dense()
        self.assertLess(torch.norm(res_two - torch.zeros(5, 2, 3)), 1e-4)
        res_three = lv[:, :, [0, 2]].to_dense()
        self.assertLess(torch.norm(res_three - torch.zeros(5, 4, 2)), 1e-4)

    def test_getitem_ellipsis(self):
        lv = ZeroLinearOperator(5, 4, 3)

        res_one = lv[[0, 1]].to_dense()
        self.assertLess(torch.norm(res_one - torch.zeros(2, 4, 3)), 1e-4)
        res_two = lv[:, [0, 1], ...].to_dense()
        self.assertLess(torch.norm(res_two - torch.zeros(5, 2, 3)), 1e-4)
        res_three = lv[..., [0, 2]].to_dense()
        self.assertLess(torch.norm(res_three - torch.zeros(5, 4, 2)), 1e-4)

    def test_get_item_tensor_index(self):
        # Tests the default LV.__getitem__ behavior
        linear_op = ZeroLinearOperator(5, 5)
        evaluated = linear_op.to_dense()

        index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(linear_op[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(linear_op[index].to_dense(), evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(linear_op[index].to_dense(), evaluated[index]))
        index = (Ellipsis, slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(linear_op[index].to_dense(), evaluated[index]))
        index = (Ellipsis, torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(linear_op[index].to_dense(), evaluated[index]))

    def test_get_item_tensor_index_on_batch(self):
        # Tests the default LV.__getitem__ behavior
        linear_op = ZeroLinearOperator(3, 5, 5)
        evaluated = linear_op.to_dense()

        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
        self.assertTrue(approx_equal(linear_op[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(linear_op[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1]), slice(None, None, None), torch.tensor([0, 1, 2]))
        self.assertTrue(approx_equal(linear_op[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(linear_op[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 1]), slice(None, None, None), slice(None, None, None))
        self.assertTrue(approx_equal(linear_op[index].to_dense(), evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(linear_op[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(linear_op[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 0]), slice(None, None, None), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(linear_op[index], evaluated[index]))
        index = (Ellipsis, torch.tensor([0, 1, 1, 0]))
        self.assertTrue(approx_equal(linear_op[index].to_dense(), evaluated[index]))

    def test_add_diagonal(self):
        diag = torch.tensor(1.5)
        res = ZeroLinearOperator(5, 5).add_diagonal(diag).to_dense()
        actual = torch.eye(5).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5])
        res = ZeroLinearOperator(5, 5).add_diagonal(diag).to_dense()
        actual = torch.eye(5).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5, 1.3, 1.2, 1.1, 2.0])
        res = ZeroLinearOperator(5, 5).add_diagonal(diag).to_dense()
        actual = torch.diag_embed(diag)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor(1.5)
        res = ZeroLinearOperator(2, 5, 5).add_diagonal(diag).to_dense()
        actual = torch.eye(5).repeat(2, 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5])
        res = ZeroLinearOperator(2, 5, 5).add_diagonal(diag).to_dense()
        actual = torch.eye(5).repeat(2, 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5, 1.3, 1.2, 1.1, 2.0])
        res = ZeroLinearOperator(2, 5, 5).add_diagonal(diag).to_dense()
        actual = torch.diag_embed(diag).repeat(2, 1, 1)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([[1.5, 1.3, 1.2, 1.1, 2.0], [0, 1, 2, 1, 1]])
        res = ZeroLinearOperator(2, 5, 5).add_diagonal(diag).to_dense()
        actual = torch.diag_embed(diag)
        self.assertTrue(approx_equal(res, actual))

    def test_matmul(self):
        zero = ZeroLinearOperator(5, 4, 3)
        lazy_square = ZeroLinearOperator(5, 3, 3)
        actual = torch.zeros(5, 4, 3)
        product = zero.matmul(lazy_square)
        self.assertTrue(approx_equal(product, actual))

        tensor_square = torch.eye(3, dtype=int).repeat(5, 1, 1)
        product = zero._matmul(tensor_square)
        self.assertTrue(approx_equal(product, actual))
        self.assertEqual(product.dtype, tensor_square.dtype)

        tensor_square = torch.eye(4).repeat(5, 1, 1)
        actual = torch.zeros(5, 3, 4)
        product = zero._t_matmul(tensor_square)
        self.assertTrue(approx_equal(product, actual))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import PermutationLinearOperator, TransposePermutationLinearOperator


class TestPermutationLinearOperator(unittest.TestCase):
    def test_permutation_linear_operator(self):
        with self.assertRaisesRegex(ValueError, "perm is not a Tensor."):
            PermutationLinearOperator([1, 3, 5])

        with self.assertRaisesRegex(ValueError, "Invalid perm*"):
            PermutationLinearOperator(torch.tensor([1, 3, 5]))

        with self.assertRaisesRegex(ValueError, "Invalid perm*"):
            PermutationLinearOperator(torch.tensor([0, 2, 1]), torch.tensor([0, 1, 2]))

        n = 3
        P = PermutationLinearOperator(torch.randperm(n))

        b1 = 2
        b2 = 5
        batch_shapes = [(), (b1,), (b2, b1)]

        permutations = [
            torch.randperm(n),
            torch.cat(tuple(torch.randperm(n).unsqueeze(0) for _ in range(b1)), dim=0),
        ]
        operators = [PermutationLinearOperator(perm) for perm in permutations]
        right_hand_sides = [torch.randn(n)] + [torch.randn(*batch_shape, n, 4) for batch_shape in batch_shapes]

        for P in operators:
            if torch.__version__ > "1.12":
                D = P.to_dense()
                S = P.to_sparse()
                self.assertTrue(isinstance(S, torch.Tensor))
                self.assertTrue(S.layout == torch.sparse_csr)
                self.assertTrue(torch.equal(D, S.to_dense()))

            for x in right_hand_sides:
                batch_shape = torch.broadcast_shapes(P.batch_shape, x.shape[:-2])
                expanded_x = x.expand(*batch_shape, *x.shape[-2:]).contiguous()
                self.assertTrue(P._matmul_batch_shape(x) == batch_shape)
                y = P @ x

                # computed inverse permutation field sorts the permutation
                perm_batch_indices = P._batch_indexing_helper(P.batch_shape)
                self.assertTrue((P.perm[perm_batch_indices + (P.inv_perm,)] == torch.arange(n)).all())

                # application of permutation operator correctly permutes the input
                batch_indices = P._batch_indexing_helper(batch_shape)
                indices = batch_indices + (P.perm, slice(None))
                if x.ndim == 1:
                    expanded_x = expanded_x.unsqueeze(-1)
                    y = y.unsqueeze(-1)

                xp = expanded_x[indices]
                self.assertTrue(torch.equal(y, xp))

                # inverse of permutation operator
                P_inv = torch.inverse(P)
                self.assertTrue(torch.equal(P_inv @ y, expanded_x))

                # transpose of permutation operator is equal to its inverse
                self.assertTrue(torch.equal(P.transpose(-1, -2).perm, P_inv.perm))


class TestTransposePermutationLinearOperator(unittest.TestCase):
    def test_transpose_permutation_linear_operator(self):
        m = 0
        msg = "m*has to be a positive integer."
        with self.assertRaisesRegex(ValueError, msg):
            TransposePermutationLinearOperator(m)

        m = 3
        P = TransposePermutationLinearOperator(m)
        n = m**2
        self.assertTrue(P.shape == (n, n))

        batch_shapes = [(), (2,), (5, 2)]
        right_hand_sides = [torch.randn(n)] + [torch.randn(*batch_shape, n, 3) for batch_shape in batch_shapes]

        for x in right_hand_sides:
            flat_i = -2 if x.ndim > 1 else -1
            X = x.unflatten(flat_i, (m, m))
            Xt = X.transpose(flat_i - 1, flat_i)
            xt = Xt.flatten(start_dim=flat_i - 1, end_dim=flat_i)
            y = P @ x
            self.assertTrue(torch.equal(y, xt))
            self.assertTrue(P is P.inverse())
            self.assertTrue((P @ y - x).abs().max() == 0)

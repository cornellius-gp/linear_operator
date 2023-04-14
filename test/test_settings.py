#!/usr/bin/env python3

import unittest

import torch

from linear_operator.settings import cholesky_jitter
from linear_operator.test.base_test_case import BaseTestCase


class TestSettings(BaseTestCase, unittest.TestCase):
    def test_cholesky_jitter(self):
        # Test value and defaults.
        self.assertEqual(cholesky_jitter.value(dtype=torch.float), 1e-6)
        self.assertEqual(cholesky_jitter.value(dtype=torch.ones(1, dtype=torch.float)), 1e-6)
        self.assertEqual(cholesky_jitter.value(dtype=torch.double), 1e-8)
        self.assertEqual(cholesky_jitter.value(dtype=torch.half), None)
        with self.assertRaisesRegex(RuntimeError, "Unsupported dtype"):
            cholesky_jitter.value(dtype=None)

        # Test init/enter/exit/set_value.
        with cholesky_jitter(float_value=0.1, double_value=0.01):
            self.assertEqual(cholesky_jitter.value(dtype=torch.float), 0.1)
            self.assertEqual(cholesky_jitter.value(dtype=torch.double), 0.01)
        self.assertEqual(cholesky_jitter.value(dtype=torch.float), 1e-6)
        self.assertEqual(cholesky_jitter.value(dtype=torch.double), 1e-8)

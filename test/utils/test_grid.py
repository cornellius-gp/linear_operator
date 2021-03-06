#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from linear_operator import utils


class TestGrid(unittest.TestCase):
    def test_scale_to_bounds(self):
        """
        """
        x = torch.randn(100) * 50
        res = utils.grid.scale_to_bounds(x, -1, 1)
        self.assertGreater(res.min().item(), -1)
        self.assertLess(res.max().item(), 1)

    def test_choose_grid_size(self):
        """
        """
        x = torch.randn(100)
        grid_size = utils.grid.choose_grid_size(x, ratio=2.0)
        self.assertEqual(grid_size, 200)

        x = torch.randn(100, 1)
        grid_size = utils.grid.choose_grid_size(x, ratio=2.0)
        self.assertEqual(grid_size, 200)

        x = torch.randn(10000, 2)
        grid_size = utils.grid.choose_grid_size(x, ratio=2.0)
        self.assertEqual(grid_size, 200)

        x = torch.randn(16, 10000, 4)
        grid_size = utils.grid.choose_grid_size(x, ratio=2.0)
        self.assertEqual(grid_size, 20)

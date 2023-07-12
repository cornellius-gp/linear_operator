#!/usr/bin/env python3

from abc import ABC, abstractmethod


class ColaLinearOperator(ABC):
    def __init__(self, lo):
        self._lo = lo

    @abstractmethod
    def _generate_cola_lo(self):
        raise NotImplementedError

    @property
    def _cola_lo(self):
        return self._generate_cola_lo()

    @property
    def batch_shape(self):
        # COLAIFY
        return self._lo.batch_shape

    def detach(self):
        # COLAIFY
        return self._lo.detach()

    def matmul(self, rhs):
        print("Using CoLA for matmul")
        return self._col_lo @ rhs

    def size(self, dim):
        shape = self._cola_lo.shape
        if dim is not None:
            return shape[dim]
        return shape

    def __matmul__(self, rhs):
        print("Using CoLA for matmul")
        return self.matmul(rhs)

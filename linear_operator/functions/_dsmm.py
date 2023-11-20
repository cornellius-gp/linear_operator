#!/usr/bin/env python3

from torch.autograd import Function

from linear_operator.utils.sparse import bdsmm


class DSMM(Function):
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.sparse = sparse
        return bdsmm(ctx.sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        return None, bdsmm(ctx.sparse.mT, grad_output)

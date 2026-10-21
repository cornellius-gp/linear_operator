#!/usr/bin/env python3
from __future__ import annotations

from torch.autograd import Function

from linear_operator import settings


class Matmul(Function):
    @staticmethod
    def forward(ctx, representation_tree, rhs, *matrix_args):
        ctx.representation_tree = representation_tree
        orig_rhs = rhs

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        linear_op = ctx.representation_tree(*matrix_args)
        res = linear_op._matmul(rhs)

        to_save = [orig_rhs] + list(matrix_args)
        ctx.save_for_backward(*to_save)
        if settings.memory_efficient.off():
            ctx._linear_op = linear_op

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        rhs = ctx.saved_tensors[0]
        matrix_args = ctx.saved_tensors[1:]
        rhs_shape = rhs.shape
        is_vector = rhs.ndimension() == 1
        rhs_matrix = rhs.unsqueeze(-1) if is_vector else rhs
        grad_output_matrix = grad_output.unsqueeze(-1) if is_vector else grad_output

        rhs_grad = None
        arg_grads = [None] * len(matrix_args)

        # input_1 gradient
        if any(ctx.needs_input_grad[2:]):
            arg_grads = ctx.representation_tree(*matrix_args)._bilinear_derivative(grad_output_matrix, rhs_matrix)

        # input_2 gradient
        if ctx.needs_input_grad[1]:
            if hasattr(ctx, "_linear_op"):
                linear_op = ctx._linear_op
            else:
                linear_op = ctx.representation_tree(*matrix_args)

            # The original right hand side determines vector semantics. A
            # batched operator produces a batched vector gradient, even when
            # that input was one-dimensional.
            rhs_grad = linear_op._t_matmul(grad_output_matrix)
            if is_vector:
                rhs_grad = rhs_grad.squeeze(-1)

            # For broadcasting
            if rhs_grad.dim() > len(rhs_shape):
                rhs_grad = rhs_grad.reshape(-1, *rhs_shape).sum(0)

        return tuple([None] + [rhs_grad] + list(arg_grads))

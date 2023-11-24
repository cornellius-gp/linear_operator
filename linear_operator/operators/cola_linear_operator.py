#!/usr/bin/env python3

from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import cola
from cola.linalg.decompositions.lanczos import lanczos


_HANDLED_FUNCTIONS = {}
_HANDLED_SECOND_ARG_FUNCTIONS = {}


def _implements(torch_function: Callable) -> Callable:
    """
    Register a torch function override for LinearOperator
    """

    @functools.wraps(torch_function)
    def decorator(func):
        # Hack: we store the name of the function, not the actual function
        # This makes it so that torch_function can map to subclass versions of functions,
        #   rather than always mapping to the superclass function
        _HANDLED_FUNCTIONS[torch_function] = func.__name__
        return func

    return decorator


def _implements_second_arg(torch_function: Callable) -> Callable:
    """
    Register a torch function override for LinearOperator,
    where the first argument of the function is a torch.Tensor and the
    second argument is a LinearOperator

    Examples of this include :meth:`torch.cholesky_solve`, `torch.solve`, or `torch.matmul`.
    """

    @functools.wraps(torch_function)
    def decorator(func):
        # Hack: we store the name of the function, not the actual function
        # This makes it so that torch_function can map to subclass versions of functions,
        #   rather than always mapping to the superclass function
        _HANDLED_SECOND_ARG_FUNCTIONS[torch_function] = func.__name__
        return func

    return decorator


def _implements_symmetric(torch_function: Callable) -> Callable:
    """
    Register a torch function override for LinearOperator
    """

    @functools.wraps(torch_function)
    def decorator(func):
        # Hack: we store the name of the function, not the actual function
        # This makes it so that torch_function can map to subclass versions of functions,
        #   rather than always mapping to the superclass function
        _HANDLED_FUNCTIONS[torch_function] = func.__name__
        _HANDLED_SECOND_ARG_FUNCTIONS[torch_function] = func.__name__
        return func

    return decorator


class ColaLinearOperator(ABC):
    def __init__(self, *args, **kwargs):
        self._cola_lo = self._generate_cola_lo(*args, **kwargs)
        self._orig_lo = self._generate_orig_lo(*args, **kwargs)
        self.shape = self._cola_lo.shape
        self.dtype = self._cola_lo.dtype
        self.matrix_shape = self.shape[-2:]

    @abstractmethod
    def _generate_cola_lo(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _generate_orig_lo(self, *args, **kwargs):
        raise NotImplementedError

    @_implements(torch.abs)
    def abs(self):
        output = self._cola_lo.abs()
        return output

    @_implements_symmetric(torch.add)
    def add(self, other, alpha=None):
        if alpha is None:
            return self + other
        else:
            return self + alpha * other

    def add_diagonal(self, diag):
        shape, dtype = self._cola_lo.shape, self._cola_lo.dtype
        SOp = cola.ops.ScalarMul(
            diag.clone().detach(), shape=shape, dtype=dtype)
        output = self._cola_lo + SOp
        params, unflatten = output.flatten()
        return ColaWrapperLinearOperator(unflatten, *params)

    def add_jitter(self, val):
        Id = cola.ops.I_like(self._cola_lo)
        output = self._cola_lo + val * Id
        params, unflatten = output.flatten()
        return ColaWrapperLinearOperator(unflatten, *params)

    def add_low_rank(self, V):
        VOp = cola.ops.Dense(V)
        output = self._cola_lo + VOp @ VOp.T
        params, unflatten = output.flatten()
        return ColaWrapperLinearOperator(unflatten, *params)

    @property
    def batch_shape(self):
        # COLAIFY
        return self._orig_lo.batch_shape

    @_implements(torch.clone)
    def clone(self):
        # COLAIFY
        cloned_lo = self._orig_lo.clone()
        return self.__class__(*cloned_lo._args, **cloned_lo._kwargs)

    def detach(self):
        # COLAIFY
        detached_lo = self._orig_lo.detach()
        return self.__class__(*detached_lo._args, **detached_lo._kwargs)

    def dim(self):
        return len(self._cola_lo.shape)

    @_implements(torch.div)
    def div(self, rhs):
        print("Using CoLA for div")
        return self._cola_lo / rhs

    @_implements(torch.matmul)
    def matmul(self, rhs):
        print("Using CoLA for matmul")
        return self._cola_lo @ rhs

    @_implements_symmetric(torch.mul)
    def mul(self, rhs):
        print("Using CoLA for mul")
        return self._cola_lo * rhs

    def ndimension(self):
        return len(self._cola_lo.shape)

    def representation(self):
        res, _ = self._cola_lo.flatten()
        return tuple(*res)

    def requires_grad_(self, value):
        # COLAIFY
        self._orig_lo.requires_grad_(value)
        return self

    def root_decomposition(self, method=None):
        # TODO: AP update lanczos for gradients
        Q, T, *_ = lanczos(self._cola_lo)
        L = torch.linalg.cholesky(T.to_dense())
        output = cola.ops.Dense(Q.to_dense() @ L)
        output = output @ output.T

        params, unflatten = output.flatten()
        return ColaWrapperLinearOperator(unflatten, *params)

    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None, method=None):
        # TODO: AP update lanczos for gradients
        Q, T, *_ = lanczos(self._cola_lo)
        L = torch.linalg.cholesky(T.to_dense())
        output = torch.linalg.solve_triangular(L, Q.to_dense().T, upper=False)
        output = cola.ops.Dense(output)
        output = output.T @ output

        params, unflatten = output.flatten()
        return ColaWrapperLinearOperator(unflatten, *params)

    def size(self, dim):
        shape = self._cola_lo.shape
        if dim is not None:
            return shape[dim]
        return shape

    def to_dense(self):
        return self._cola_lo.to_dense()

    def __add__(self, other):
        # COLAIFY
        if len(other.shape) >= 3:
            return self._orig_lo + other
        else:
            if isinstance(other, ColaLinearOperator):
                other = other._cola_lo
            return self._cola_lo + cola.fns.lazify(other)

    def __matmul__(self, rhs):
        return self.matmul(rhs)

    def __mul__(self, x):
        return self._cola_lo * x

    @_implements_second_arg(torch.Tensor.add)
    def __radd__(self, other):
        return self + cola.fns.lazify(other)

    def __rmul__(self, x):
        return x * self._cola_lo

    @classmethod
    def __torch_function__(
        cls, func: Callable, types: Tuple[type, ...], args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = None
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if not isinstance(args[0], cls):
            if func not in _HANDLED_SECOND_ARG_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, ColaLinearOperator)) for t in types
            ):
                name = func.__name__.replace("linalg_", "linalg.")
                arg_classes = ", ".join(arg.__class__.__name__ for arg in args)
                kwarg_classes = ", ".join(
                    f"{key}={val.__class__.__name__}" for key, val in kwargs.items())
                raise NotImplementedError(
                    f"torch.{name}({arg_classes}, {kwarg_classes}) is not implemented.")
            # Hack: get the appropriate class function based on its name
            # As a result, we will call the subclass method (when applicable) rather than the superclass method
            func = getattr(cls, _HANDLED_SECOND_ARG_FUNCTIONS[func])
            return func(args[1], args[0], *args[2:], **kwargs)
        else:
            if func not in _HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, ColaLinearOperator)) for t in types):
                name = func.__name__.replace("linalg_", "linalg.")
                arg_classes = ", ".join(arg.__class__.__name__ for arg in args)
                kwarg_classes = ", ".join(
                    f"{key}={val.__class__.__name__}" for key, val in kwargs.items())
                raise NotImplementedError(
                    f"torch.{name}({arg_classes}, {kwarg_classes}) is not implemented.")
            # Hack: get the appropriate class function based on its name
            # As a result, we will call the subclass method (when applicable) rather than the superclass method
            func = getattr(cls, _HANDLED_FUNCTIONS[func])
            return func(*args, **kwargs)

    def __truediv__(self, x):
        return self._cola_lo / x


class ColaWrapperLinearOperator(ColaLinearOperator):
    def __init__(self, unflatten, *args, **kwargs):
        self._unflatten = unflatten
        super().__init__(*args, **kwargs)

    def _generate_cola_lo(self, *args):
        return self._unflatten(args)

    def _generate_orig_lo(self, *args):
        return None

#!/usr/bin/env python3

from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch


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

    @abstractmethod
    def _generate_cola_lo(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _generate_orig_lo(self, *args, **kwargs):
        raise NotImplementedError

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

    @_implements(torch.matmul)
    def matmul(self, rhs):
        print("Using CoLA for matmul")
        return self._cola_lo @ rhs

    def representation(self):
        res, _ = self._cola_lo.flatten()
        return tuple(*res)

    def requires_grad_(self, value):
        # COLAIFY
        self._orig_lo.requires_grad_(value)
        return self

    def size(self, dim):
        shape = self._cola_lo.shape
        if dim is not None:
            return shape[dim]
        return shape

    def __matmul__(self, rhs):
        return self.matmul(rhs)

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
                kwarg_classes = ", ".join(f"{key}={val.__class__.__name__}" for key, val in kwargs.items())
                raise NotImplementedError(f"torch.{name}({arg_classes}, {kwarg_classes}) is not implemented.")
            # Hack: get the appropriate class function based on its name
            # As a result, we will call the subclass method (when applicable) rather than the superclass method
            func = getattr(cls, _HANDLED_SECOND_ARG_FUNCTIONS[func])
            return func(args[1], args[0], *args[2:], **kwargs)
        else:
            if func not in _HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, ColaLinearOperator)) for t in types):
                name = func.__name__.replace("linalg_", "linalg.")
                arg_classes = ", ".join(arg.__class__.__name__ for arg in args)
                kwarg_classes = ", ".join(f"{key}={val.__class__.__name__}" for key, val in kwargs.items())
                raise NotImplementedError(f"torch.{name}({arg_classes}, {kwarg_classes}) is not implemented.")
            # Hack: get the appropriate class function based on its name
            # As a result, we will call the subclass method (when applicable) rather than the superclass method
            func = getattr(cls, _HANDLED_FUNCTIONS[func])
            return func(*args, **kwargs)

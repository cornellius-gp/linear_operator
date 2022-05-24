#!/usr/bin/env python3
from . import beta_features, operators, settings, utils
from .functions import (  # Deprecated
    dsmm,
    inv_matmul,
    inv_quad,
    inv_quad_logdet,
    logdet,
    matmul,
    pivoted_cholesky,
    root_decomposition,
    root_inv_decomposition,
)
from .operators import LinearOperator, cat, to_dense, to_linear_operator

__version__ = "0.0.1"

__all__ = [
    # Base class
    "LinearOperator",
    # Submodules
    "operators",
    "utils",
    # Functions
    "cat",
    "dsmm",
    "inv_matmul",
    "inv_quad",
    "inv_quad_logdet",
    "logdet",
    "matmul",
    "pivoted_cholesky",
    "root_decomposition",
    "root_inv_decomposition",
    "to_dense",
    "to_linear_operator",
    # Context managers
    "beta_features",
    "settings",
    # Other
    "__version__",
]

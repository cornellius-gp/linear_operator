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
from .operators import cat, delazify, lazify

__version__ = "0.0.1"

__all__ = [
    # Submodules
    "operators",
    "utils",
    # Functions
    "cat",
    "delazify",
    "dsmm",
    "inv_matmul",
    "inv_quad",
    "inv_quad_logdet",
    "lazify",
    "logdet",
    "log_normal_cdf",
    "matmul",
    "pivoted_cholesky",
    "root_decomposition",
    "root_inv_decomposition",
    # Context managers
    "beta_features",
    "settings",
    # Other
    "__version__",
]

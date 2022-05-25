#!/usr/bin/env python3

import torch
from torch import Tensor

from .qr import stable_qr


def stable_pinverse(A: Tensor) -> Tensor:
    """Compute a pseudoinverse of a matrix. Employs a stabilized QR decomposition."""
    if A.shape[-2] >= A.shape[-1]:
        # skinny (or square) matrix
        Q, R = stable_qr(A)
        return torch.triangular_solve(Q.mT, R).solution
    else:
        # fat matrix
        Q, R = stable_qr(A.mT)
        return torch.triangular_solve(Q.mT, R).solution.mT

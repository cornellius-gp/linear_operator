#!/usr/bin/env python3

import torch
from torch.fft import fft, ifft

from ..utils import broadcasting


def toeplitz(toeplitz_column, toeplitz_row):
    """
    Constructs tensor version of toeplitz matrix from column vector
    Args:
        - toeplitz_column (vector n) - column of toeplitz matrix
        - toeplitz_row (vector n-1) - row of toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
    """
    if toeplitz_column.ndimension() != 1:
        raise RuntimeError("toeplitz_column must be a vector.")

    if toeplitz_row.ndimension() != 1:
        raise RuntimeError("toeplitz_row must be a vector.")

    if toeplitz_column[0] != toeplitz_row[0]:
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0], toeplitz_row[0])
        )

    if len(toeplitz_column) != len(toeplitz_row):
        raise RuntimeError("c and r should have the same length " "(Toeplitz matrices are necessarily square).")

    if type(toeplitz_column) != type(toeplitz_row):
        raise RuntimeError("toeplitz_column and toeplitz_row should be the same type.")

    if len(toeplitz_column) == 1:
        return toeplitz_column.view(1, 1)

    res = torch.empty(
        len(toeplitz_column),
        len(toeplitz_column),
        dtype=toeplitz_column.dtype,
        device=toeplitz_column.device,
    )
    for i, val in enumerate(toeplitz_column):
        for j in range(len(toeplitz_column) - i):
            res[j + i, j] = val
    for i, val in list(enumerate(toeplitz_row))[1:]:
        for j in range(len(toeplitz_row) - i):
            res[j, j + i] = val
    return res


def sym_toeplitz(toeplitz_column):
    """
    Constructs tensor version of symmetric toeplitz matrix from column vector
    Args:
        - toeplitz_column (vector n) - column of Toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
    """
    return toeplitz(toeplitz_column, toeplitz_column)


def toeplitz_getitem(toeplitz_column, toeplitz_row, i, j):
    """
    Gets the (i,j)th entry of a Toeplitz matrix T.
    Args:
        - toeplitz_column (vector n) - column of Toeplitz matrix
        - toeplitz_row (vector n) - row of Toeplitz matrix
        - i (scalar) - row of entry to get
        - j (scalar) - column of entry to get
    Returns:
        - T[i,j], where T is the Toeplitz matrix specified by c and r.
    """
    index = i - j
    if index < 0:
        return toeplitz_row[abs(index)]
    else:
        return toeplitz_column[index]


def sym_toeplitz_getitem(toeplitz_column, i, j):
    """
    Gets the (i,j)th entry of a symmetric Toeplitz matrix T.
    Args:
        - toeplitz_column (vector n) - column of symmetric Toeplitz matrix
        - i (scalar) - row of entry to get
        - j (scalar) - column of entry to get
    Returns:
        - T[i,j], where T is the Toeplitz matrix specified by c and r.
    """
    return toeplitz_getitem(toeplitz_column, toeplitz_column, i, j)


def toeplitz_matmul(toeplitz_column, toeplitz_row, tensor):
    """
    Performs multiplication T * M where the matrix T is Toeplitz.
    Args:
        - toeplitz_column (vector n or b x n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n or b x n) - First row of the Toeplitz matrix T.
        - tensor (matrix n x p or b x n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor (n x p or b x n x p) - The result of the matrix multiply T * M.
    """
    toeplitz_input_check(toeplitz_column, toeplitz_row)

    toeplitz_shape = torch.Size((*toeplitz_column.shape, toeplitz_row.size(-1)))
    output_shape = broadcasting._matmul_broadcast_shape(toeplitz_shape, tensor.shape)
    broadcasted_t_shape = output_shape[:-1] if tensor.dim() > 1 else output_shape

    if tensor.ndimension() == 1:
        tensor = tensor.unsqueeze(-1)
    toeplitz_column = toeplitz_column.expand(*broadcasted_t_shape)
    toeplitz_row = toeplitz_row.expand(*broadcasted_t_shape)
    tensor = tensor.expand(*output_shape)

    if not torch.equal(toeplitz_column[..., 0], toeplitz_row[..., 0]):
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first element, otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0], toeplitz_row[0])
        )

    if type(toeplitz_column) != type(toeplitz_row) or type(toeplitz_column) != type(tensor):
        raise RuntimeError("The types of all inputs to ToeplitzMV must match.")

    *batch_shape, orig_size, num_rhs = tensor.size()
    r_reverse = toeplitz_row[..., 1:].flip(dims=(-1,))

    c_r_rev = torch.zeros(*batch_shape, orig_size + r_reverse.size(-1), dtype=tensor.dtype, device=tensor.device)
    c_r_rev[..., :orig_size] = toeplitz_column
    c_r_rev[..., orig_size:] = r_reverse

    temp_tensor = torch.zeros(
        *batch_shape, 2 * orig_size - 1, num_rhs, dtype=toeplitz_column.dtype, device=toeplitz_column.device
    )
    temp_tensor[..., :orig_size, :] = tensor

    fft_M = fft(temp_tensor.mT.contiguous())
    fft_c = fft(c_r_rev).unsqueeze(-2).expand_as(fft_M)
    fft_product = fft_M.mul_(fft_c)

    output = ifft(fft_product).real.mT
    output = output[..., :orig_size, :]
    return output


def sym_toeplitz_matmul(toeplitz_column, tensor):
    """
    Performs a matrix-matrix multiplication TM where the matrix T is symmetric Toeplitz.
    Args:
        - toeplitz_column (vector n) - First column of the symmetric Toeplitz matrix T.
        - matrix (matrix n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor
    """
    return toeplitz_matmul(toeplitz_column, toeplitz_column, tensor)


def sym_toeplitz_solve_ld(toeplitz_column, right_vectors):
    """
    Solve the linear system Tx=b where T is a symmetric Toeplitz matrix and b the right
    hand side of the equation using the Levinson-Durbin recursion, which run in O(n^2) time.
    Args:
        - toeplitz_column (vector n or b x n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n or b x n) - First row of the Toeplitz matrix T.
        - right_vectors (matrix n x p or b x n x p) - Right hand side in T x = b
    Returns:
        - tensor (n x p or b x n x p) - The solution to the system T x = b.
            Shape of return matches shape of b.
    """
    return toeplitz_solve_ld(toeplitz_column, toeplitz_column, right_vectors)


def toeplitz_solve_ld(toeplitz_column, toeplitz_row, right_vectors):
    """
    Solve the linear system Tx=b where T is a general Toeplitz matrix and b the right
    hand side of the equation. Use the Levinson-Durbin recursion, which run in O(n^2) time,
    but may exhibit numerical stability issues.
    Args:
        - toeplitz_column (vector n or b x n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n or b x n) - First row of the Toeplitz matrix T.
        - right_vectors (matrix n x p or b x n x p) - Right hand side in T x = b
    Returns:
        - tensor (n x p or b x n x p) - The solution to the system T x = b.
            Shape of return matches shape of b.
    """
    # check input
    toeplitz_input_check(toeplitz_column, toeplitz_row)
    if right_vectors.ndimension() == 1:
        if toeplitz_row.shape[-1] != len(right_vectors):
            raise RuntimeError(f"Incompatible size betwen the Toeplitz matrix and the right vector: {toeplitz_column.shape} and {right_vectors.shape}")
    else:
        if toeplitz_row.shape[-1] != right_vectors.size(-2):
            raise RuntimeError(f"Incompatible size betwen the Toeplitz matrix and the right vector: {toeplitz_column.shape} and {right_vectors.shape}")
    
    output_shape = torch.broadcast_shapes(toeplitz_row.shape, right_vectors.shape[:-1])
    broadcasted_t_shape = output_shape#[:-1] if right_vectors.dim() > 1 else output_shape
    unsqueezed_vec = False
    if right_vectors.ndimension() == 1:
        right_vectors = right_vectors.unsqueeze(-1)
        unsqueezed_vec = True
    toeplitz_column = toeplitz_column.expand(*broadcasted_t_shape)
    toeplitz_row = toeplitz_row.expand(*broadcasted_t_shape)
    N = toeplitz_column.size(-1)
    
    # f = forward vector , b = backward vector
    # xi = vector at iterator i, xim = vector at iteration i-1
    flipped_toeplitz_column = toeplitz_column[..., 1:].flip(dims=(-1,))
    xi = torch.zeros_like(right_vectors).expand(*broadcasted_t_shape, right_vectors.shape[-1]).clone()
    fi = torch.zeros_like(xi)
    bi = torch.zeros_like(xi)
    bim = torch.zeros_like(xi)

    # iteration 0
    fi[...,0,:] = 1/toeplitz_column[...,0,None]
    bi[...,N-1,:] = 1/toeplitz_column[...,0,None]
    xi[...,0,:] = right_vectors[...,0,:]/toeplitz_column[...,0,None]
    if N == 1: return xi

    for i in range(1,N):
        #update
        bim = bi.clone()
        #compute the new forward and backward vector
        efi = torch.matmul(flipped_toeplitz_column[...,N-i-1:N-1,None].mT, fi.clone()[...,:i,:])
        ebi = torch.matmul(toeplitz_row[...,1:i+1,None].mT, bim[...,N-i:,:])
        coeff = 1/(1-ebi*efi)
        bi[...,N-i-1:,:] = coeff * (bim[...,N-i-1:,:] - ebi * fi.clone()[...,:i+1,:])
        fi[...,:i+1,:] = coeff * (fi[...,:i+1,:] - efi * bim[...,N-i-1:,:])
        #update solution
        exim = torch.matmul(flipped_toeplitz_column[...,N-i-1:N-1,None].mT, xi.clone()[...,:i,:])
        xi[...,:i+1,:] = xi[...,:i+1,:] + bi.clone()[...,N-i-1:,:] * (right_vectors[...,i,:,None].mT - exim)
    
    if unsqueezed_vec == 1:
        xi = xi.squeeze()

    return xi


def toeplitz_inverse(toeplitz_column, toeplitz_row):
    """
    Calculate the Toeplitz matrix inverse following the Trench algorithm.
    (See: Shalhav Zohar (1969) - Toeplitz Matrix Inversion: The Algorithm of W. F. Trench)
    Args:
        - toeplitz_column (vector n or b x n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n or b x n) - First row of the Toeplitz matrix T.
    Returns:
        - tensor (m x m or s x m x m) - The inverse of the Toeplitz matrices.
    """
    # Algorithm taken from:
    # https://dl.acm.org/doi/pdf/10.1145/321541.321549
    if toeplitz_column[0] == 0.:
        raise ValueError("The main diagonal term (i.e. first column and row element) must be non-zero")
    raise NotImplementedError("To be implemented")
    return inv


def sym_toeplitz_derivative_quadratic_form(left_vectors, right_vectors):
    r"""
    Given a left vector v1 and a right vector v2, computes the quadratic form:
                                v1'*(dT/dc_i)*v2
    for all i, where dT/dc_i is the derivative of the Toeplitz matrix with respect to
    the ith element of its first column. Note that dT/dc_i is the same for any symmetric
    Toeplitz matrix T, so we do not require it as an argument.

    In particular, dT/dc_i is given by:
                                [0 0; I_{m-i+1} 0] + [0 I_{m-i+1}; 0 0]
    where I_{m-i+1} is the (m-i+1) dimensional identity matrix. In other words, dT/dc_i
    for i=1..m is the matrix with ones on the ith sub- and superdiagonal.

    Args:
        - left_vectors (vector m or matrix s x m) - s left vectors u[j] in the quadratic form.
        - right_vectors (vector m or matrix s x m) - s right vectors v[j] in the quadratic form.
    Returns:
        - vector m - a vector so that the ith element is the result of \sum_j(u[j]*(dT/dc_i)*v[j])
    """
    if left_vectors.ndimension() == 1:
        left_vectors = left_vectors.unsqueeze(1)
        right_vectors = right_vectors.unsqueeze(1)

    batch_shape = left_vectors.shape[:-2]
    toeplitz_size = left_vectors.size(-2)
    num_vectors = left_vectors.size(-1)

    left_vectors = left_vectors.mT.contiguous()
    right_vectors = right_vectors.mT.contiguous()

    columns = torch.zeros_like(left_vectors)
    columns[..., 0] = left_vectors[..., 0]
    res = toeplitz_matmul(columns, left_vectors, right_vectors.unsqueeze(-1))
    rows = left_vectors.flip(dims=(-1,))
    columns[..., 0] = rows[..., 0]
    res += toeplitz_matmul(columns, rows, torch.flip(right_vectors, dims=(-1,)).unsqueeze(-1))

    res = res.reshape(*batch_shape, num_vectors, toeplitz_size).sum(-2)
    res[..., 0] -= (left_vectors * right_vectors).view(*batch_shape, -1).sum(-1)

    return res


def toeplitz_derivative_quadratic_form(left_vectors, right_vectors):
    r"""
    Given a left vector v1 and a right vector v2, computes the quadratic form:
                                v1'*(dT/dc_i)*v2
    for all i, where dT/dc_i is the derivative of the Toeplitz matrix with respect to
    the ith element of its first column. Note that dT/dc_i is the same for any
    Toeplitz matrix T, so we do not require it as an argument.

    In particular, dT/dc_i for i=-m..0..m is the matrix with ones on the ith sub- (i>0) and superdiagonal (i<0).

    Args:
        - left_vectors (vector m or matrix s x m) - s left vectors u[j] in the quadratic form.
        - right_vectors (vector m or matrix s x m) - s right vectors v[j] in the quadratic form.
    Returns:
        - vector d_column - a vector so that the ith element is the result of \sum_j(u[j]*(dT/dc_i)*v[j]) 
            (i<0, corresponding to derivative relative to column entries)
        - vector d_row - a vector so that the ith element is the result of \sum_j(u[j]*(dT/dc_i)*v[j]) (i>0)
            (i>0, corresponding to derivative relative to row entries)
    """
    if left_vectors.ndimension() == 1:
        left_vectors = left_vectors.unsqueeze(1)
        right_vectors = right_vectors.unsqueeze(1)

    batch_shape = left_vectors.shape[:-2]
    toeplitz_size = left_vectors.size(-2)
    num_vectors = left_vectors.size(-1)

    left_vectors = left_vectors.mT.contiguous()
    right_vectors = right_vectors.mT.contiguous()

    columns = torch.zeros_like(left_vectors)
    columns[..., 0] = left_vectors[..., 0]
    
    res_r = toeplitz_matmul(columns, left_vectors, right_vectors.unsqueeze(-1))
    rows = left_vectors.flip(dims=(-1,))
    columns[..., 0] = rows[..., 0]
    res_c = toeplitz_matmul(columns, rows, torch.flip(right_vectors, dims=(-1,)).unsqueeze(-1))

    res_c = res_c.reshape(*batch_shape, num_vectors, toeplitz_size).sum(-2)
    res_r = res_r.reshape(*batch_shape, num_vectors, toeplitz_size).sum(-2)
    
    return [res_c, res_r]


def toeplitz_input_check(toeplitz_column, toeplitz_row):
    """
    Helper routine to check if the input Toeplitz matrix is well defined.
    """
    if toeplitz_column.size() != toeplitz_row.size():
        raise RuntimeError("c and r should have the same length (Toeplitz matrices are necessarily square).")
    if not torch.equal(toeplitz_column[..., 0], toeplitz_row[..., 0]):
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first element, otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0], toeplitz_row[0])
        )
    if type(toeplitz_column) != type(toeplitz_row):
        raise RuntimeError("toeplitz_column and toeplitz_row should be the same type.")
    return True

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from jaxtyping import Float
from torch import Tensor

from ..utils.broadcasting import _pad_with_singletons
from ..utils.getitem import _noop_index, IndexType
from ..utils.memoize import cached
from ._linear_operator import LinearOperator, to_dense


def _x_getitem(x, batch_indices, data_index):
    """
    Helper function to compute x[*batch_indices, data_index, :] in an efficient way.
    (Sometimes x needs to be expanded before calling x[*batch_indices, data_index, :]; i.e. if
    the batch_indices broadcast. We try to prevent this expansion if possible.
    """
    try:
        x = x[(*batch_indices, data_index, _noop_index)]
    # We're going to handle multi-batch indexing with a try-catch loop
    # This way - in the default case, we can avoid doing expansions of x1 which can be timely
    except IndexError:
        if isinstance(batch_indices, slice):
            x = x.expand(1, *x.shape[-2:])
            x = [(*batch_indices, data_index, _noop_index)]
        elif isinstance(batch_indices, tuple):
            if any(not isinstance(bi, slice) for bi in batch_indices):
                raise RuntimeError(
                    "Attempting to tensor index a non-batch matrix's batch dimensions. "
                    f"Got batch index {batch_indices} but my shape was {x.shape}"
                )
            x = x.expand(*([1] * len(batch_indices)), *x.shape[-2:])
            x = x[(*batch_indices, data_index, _noop_index)]
    return x


class KernelLinearOperator(LinearOperator):
    r"""
    Represents the kernel matrix :math:`\boldsymbol K`
    of data :math:`\boldsymbol X_1 \in \mathbb R^{M \times D}`
    and :math:`\boldsymbol X_2 \in \mathbb R^{N \times D}`
    under the covariance function :math:`k_{\boldsymbol \theta}(\cdot, \cdot)`
    (parameterized by hyperparameters :math:`\boldsymbol \theta`
    so that :math:`\boldsymbol K_{ij} = k_{\boldsymbol \theta}([\boldsymbol X_1]_i, [\boldsymbol X_2]_j)`.

    The output of :math:`k_{\boldsymbol \theta}(\cdot,\cdot)` (`covar_func`) can either be a torch.Tensor
    or a LinearOperator.

    .. note ::

        All hyperparameters have some number of batch dimensions (which broadcast with the
        batch dimensions of x1 and x2) and some number of non-batch dimensions
        (dimensions that would exist if we were computing a single covariance matrix).

        By default, each hyperparameter is assumed to have 2 (potentially singleton) non-batch
        dimensions. However, the number of non_batch dimensions can be specified on a
        per-hyperparameter through the optional `num_nonbatch_dimensions` dictionary argument.

        For example, to implement the RBF kernel

        .. math::

            o^2 \exp\left(
                -\tfrac{1}{2} (\boldsymbol x_1 - \boldsymbol x2)^\top \boldsymbol D_\ell^{-2}
                (\boldsymbol x_1 - \boldsymbol x2)
            \right),

        where :math:`o` is an `outputscale` parameter and :math:`D_\ell` is a diagonal `lengthscale` matrix,
        we would expect the following shapes:

        - `x1`: `(*batch_shape x N x D)`
        - `x2`: `(*batch_shape x M x D)`
        - `lengthscale`: `(*batch_shape x 1 x D)`
        - `outputscale`: `(*batch_shape)`  # Note this parameter does not have non-batch dimensions

        We would then supply the dictionary `num_nonbatch_dimensions = {"outputscale": 0}`.
        (We do not need to include lengthscale in the dictionary since it has 2 non-batch dimensions.)

    .. code-block:: python

        # NOTE: _covar_func intentionally does not close over any parameters
        def _covar_func(x1, x2, lengthscale, outputscale):
            # RBF kernel function
            # x1: ... x N x D
            # x2: ... x M x D
            # lengthscale: ... x 1 x D
            # outputscale: ...
            x1 = x1.div(lengthscale)
            x2 = x2.div(lengthscale)
            sq_dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(dim=-1)
            kern = sq_dist.div(-2.0).exp().mul(outputscale[..., None, None].square())
            return kern


        # Batches of data
        x1 = torch.randn(3, 5, 6)
        x2 = torch.randn(3, 4, 6)
        # Broadcasting lengthscale and output parameters
        lengthscale = torch.randn(2, 1, 1, 6)  # Batch shape is 2 x 1, with 2 non-batch dimensions
        outputscale = torch.randn(2, 1)  # Batch shape is 2 x 1, no non-batch dimensions
        kern = KernelLinearOperator(
            x1, x2, lengthscale=lengthscale, outputscale=outputscale,
            covar_func=covar_func, num_nonbatch_dimensions={"outputscale": 0}
        )

        # kern is of size 2 x 3 x 5 x 4

    .. warning ::

        `covar_func` should not close over any parameters. Any parameters that are closed over will not have
        propagated gradients.

        See the example above: the lengthscale and outputscale of _covar_func are passed in as arguments,
        rather than being externally defined variables.

    :param x1: The data :math:`\boldsymbol X_1.`
    :param x2: The data :math:`\boldsymbol X_2.`
    :param covar_func: The covariance function :math:`k_{\boldsymbol \theta}(\cdot, \cdot)`.
        Its arguments should be `x1`, `x2`, `**params`, and it should output the covariance matrix
        between :math:`\boldsymbol X_1` and :math:`\boldsymbol X_2`.
    :param num_outputs_per_input: The number of outputs per data point.
        This parameter should be 1 for most kernels, but will be >1 for multitask kernels,
        gradient kernels, and any other kernels that require cross-covariance terms for multiple domains.
        If a tuple is passed, there will be a different number of outputs per input dimension
        for the rows/cols of the kernel matrix.
    :param params: Additional hyperparameters (:math:`\boldsymbol \theta`) or keyword arguments passed into covar_func.
    """

    def __init__(
        self,
        x1: Float[Tensor, "... M D"],
        x2: Float[Tensor, "... N D"],
        covar_func: Callable[..., Float[Union[Tensor, LinearOperator], "... M N"]],
        num_outputs_per_input: Tuple[int, int] = (1, 1),
        num_nonbatch_dimensions: Optional[Dict[str, int]] = None,
        **params: Union[Tensor, Any],
    ):
        # Change num_nonbatch_dimensions into a default dict
        if num_nonbatch_dimensions is None:
            num_nonbatch_dimensions = defaultdict(lambda: 2)
        else:
            num_nonbatch_dimensions = defaultdict(lambda: 2, **num_nonbatch_dimensions)

        # Divide params into tensors and non-tensors
        tensor_params = dict()
        nontensor_params = dict()
        for name, val in params.items():
            if torch.is_tensor(val):
                tensor_params[name] = val
            else:
                nontensor_params[name] = val

        # Compute param_batch_shapes
        param_batch_shapes = dict()
        param_nonbatch_shapes = dict()
        for name, val in tensor_params.items():
            if num_nonbatch_dimensions[name] == 0:
                param_batch_shapes[name] = val.shape
                param_nonbatch_shapes[name] = torch.Size([])
            else:
                nonbatch_dim = num_nonbatch_dimensions[name]
                param_batch_shapes[name] = val.shape[:-nonbatch_dim]
                param_nonbatch_shapes[name] = val.shape[-nonbatch_dim:]

        # Ensure that x1, x2, and params can broadcast together
        try:
            batch_broadcast_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2], *param_batch_shapes.values())
        except RuntimeError:
            # Check if the issue is with x1 and x2
            try:
                x1_nodata_shape = torch.Size([*x1.shape[:-2], 1, x1.shape[-1]])
                x2_nodata_shape = torch.Size([*x2.shape[:-2], 1, x2.shape[-1]])
                torch.broadcast_shapes(x1_nodata_shape, x2_nodata_shape)
            except RuntimeError:
                raise RuntimeError(
                    "Incompatible data shapes for a kernel matrix: "
                    f"x1.shape={tuple(x1.shape)}, x2.shape={tuple(x2.shape)}."
                )

            # If we've made here, this means that the parameter shapes aren't compatible with x1 and x2
            raise RuntimeError(
                "Shape of kernel parameters "
                f"({', '.join([str(tuple(param.shape)) for param in tensor_params.values()])}) "
                f"is incompatible with data shapes x1.shape={tuple(x1.shape)}, x2.shape={tuple(x2.shape)}.\n"
                "Recall that parameters passed to KernelLinearOperator should have dimensionality compatible "
                "with the data (see documentation)."
            )

        # Create a version of each argument that is expanded to the broadcast batch shape
        #
        # NOTE: we must explicitly call requires_grad on each of these arguments
        # for the automatic _bilinear_derivative to work in torch.autograd.Functions
        if len(batch_broadcast_shape):  # Otherwise all tensors are non-batch, and we don't need to expand
            x1 = x1.expand(*batch_broadcast_shape, *x1.shape[-2:]).contiguous().requires_grad_(x1.requires_grad)
            x2 = x2.expand(*batch_broadcast_shape, *x2.shape[-2:]).contiguous().requires_grad_(x2.requires_grad)
            tensor_params = {
                name: val.expand(*batch_broadcast_shape, *param_nonbatch_shapes[name]).requires_grad_(val.requires_grad)
                for name, val in tensor_params.items()
            }
        # Everything should now have the same batch shape

        # Standard constructor
        super().__init__(
            x1,
            x2,
            covar_func=covar_func,
            num_outputs_per_input=num_outputs_per_input,
            num_nonbatch_dimensions=num_nonbatch_dimensions,
            **tensor_params,
            **nontensor_params,
        )
        self.batch_broadcast_shape = batch_broadcast_shape
        self.x1 = x1
        self.x2 = x2
        self.tensor_params = tensor_params
        self.nontensor_params = nontensor_params
        self.covar_func = covar_func
        self.num_outputs_per_input = num_outputs_per_input
        self.num_nonbatch_dimensions = num_nonbatch_dimensions

    @cached(name="kernel_diag")
    def _diagonal(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
        # Explicitly compute kernel diag via covar_func when it is needed rather than relying on lazy tensor ops.
        # We will do this by shoving all of the data into a batch dimension (i.e. compute a N x ... x  1 x 1 kernel
        # or a N x ... x num_outs-per_in x num_outs_per_in kernel)
        # and then squeeze out the batch dimensions
        x1 = self.x1.unsqueeze(0).transpose(0, -2)
        x2 = self.x2.unsqueeze(0).transpose(0, -2)
        tensor_params = {name: val.unsqueeze(0) for name, val in self.tensor_params.items()}
        diag_mat = to_dense(self.covar_func(x1, x2, **tensor_params, **self.nontensor_params))
        assert diag_mat.shape[-2:] == torch.Size(self.num_outputs_per_input)

        # Easy case: the kernel only has one output per input (standard kernels)
        if self.num_outputs_per_input == (1, 1):
            return diag_mat.transpose(0, -2)[0, ..., 0]
        # Complicated case: the kernel only has multiple output per input (e.g. multitask kernels)
        else:
            # First: reshape the matrix to be ... x N x num_outputs_per_input x num_outputs_per_input
            diag_mat = diag_mat.permute(*range(1, diag_mat.dim() - 2), 0, -2, -1)
            # Next: get the diagonal vector, so that we have ... x N x num_outputs_per_input
            unflattened_diag = diag_mat.diagonal(dim1=-1, dim2=-2)
            # Finally: flatten the diagonal vector, so that we have ... x (N * num_outputs_per_input)
            return unflattened_diag.reshape(*unflattened_diag.shape[:-2], -1)

    @property
    @cached(name="covar_mat")
    def covar_mat(self: Float[LinearOperator, "... M N"]) -> Float[Union[Tensor, LinearOperator], "... M N"]:
        return self.covar_func(self.x1, self.x2, **self.tensor_params, **self.nontensor_params)

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        # Similar to diagonal will do this by shoving all of the data into a batch dimension
        # (i.e. compute a N x ... x  1 x 1 kernel or a N x ... x num_outs_per_in x num_outs_per_in kernel)
        # and then squeeze out the batch dimensions
        num_outs_per_in_rows, num_outs_per_in_cols = self.num_outputs_per_input
        x1_ = self.x1[(*batch_indices, row_index.div(num_outs_per_in_rows, rounding_mode="floor"))].unsqueeze(
            -2
        )  # x1 will have shape ... x 1 x 1
        x2_ = self.x2[(*batch_indices, col_index.div(num_outs_per_in_rows, rounding_mode="floor"))].unsqueeze(
            -2
        )  # x2 will have shape ... x 1 x 1
        tensor_params_ = {name: val[batch_indices] for name, val in self.tensor_params.items()}  # will have shape ...
        indices_mat = to_dense(self.covar_func(x1_, x2_, **tensor_params_, **self.nontensor_params))
        assert indices_mat.shape[-2:] == torch.Size(self.num_outputs_per_input)
        # Easy case: the kernel only has one output per input (standard kernels)
        if self.num_outputs_per_input == (1, 1):
            return indices_mat[..., 0, 0]
        # Complicated case: the kernel only has multiple output per input (e.g. multitask kernels)
        else:
            # The current shape of indices mat is ... x num_outs_per_in_row x num_outs_per_in_col
            # And we want the final shape to be ...
            # Therefore, figure out which of outputs we want to keep
            row_output_index = row_index % num_outs_per_in_rows
            col_output_index = col_index % num_outs_per_in_cols
            # Now we select those specific outputs
            # We neeed iterative tensors to select the appropriate elements from the batch dimensions
            # of indices_mat
            batch_indices = [
                _pad_with_singletons(
                    torch.arange(size, device=indices_mat.device),
                    num_singletons_before=i,
                    num_singletons_after=(indices_mat.dim() - 3 - i),
                )
                for i, size in enumerate(indices_mat.shape[:-2])
            ]
            return indices_mat[(*batch_indices, row_output_index, col_output_index)]

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        # If we have multiple outputs per input, then the indices won't directly
        # correspond to the entries of row/col. We'll have to do a little pre-processing
        num_outs_per_in_rows, num_outs_per_in_cols = self.num_outputs_per_input
        if num_outs_per_in_rows != 1 or num_outs_per_in_cols != 1:
            if not isinstance(row_index, slice) or not isinstance(col_index, slice):
                # It's too complicated to deal with tensor indices in this case - we'll use the super method
                try:
                    return self.covar_mat._getitem(row_index, col_index, *batch_indices)
                except Exception:
                    raise TypeError(
                        f"{self.__class__.__name__} does not accept non-slice indices. "
                        f"Got {','.join(type(t) for t in [*batch_indices, row_index, col_index])}"
                    )

            # Now we know that x1 and x2 are slices
            # Let's make sure that the slice dimensions perfectly correspond with the number of
            # outputs per input that we have
            *batch_shape, num_rows, num_cols = self._size()
            row_start, row_end, row_step = (
                row_index.start if row_index.start is not None else 0,
                row_index.stop if row_index.stop is not None else num_rows,
                row_index.step if row_index.step is not None else 1,
            )
            col_start, col_end, col_step = (
                col_index.start if col_index.start is not None else 0,
                col_index.stop if col_index.stop is not None else num_cols,
                col_index.step if col_index.step is not None else 1,
            )
            if row_step is not None or col_step is not None:
                # It's too complicated to deal with tensor indices in this case - we'll try to evaluate the kernel
                # and use the super method
                try:
                    return self.covar_mat._getitem(row_index, col_index, *batch_indices)
                except Exception:
                    raise TypeError(f"{self.covar_mat.__class__.__name__} does not accept slices with steps.")
            if (
                (row_start % num_outs_per_in_rows)
                or (col_start % num_outs_per_in_cols)
                or (row_end % num_outs_per_in_rows)
                or (col_end % num_outs_per_in_cols)
            ):
                # It's too complicated to deal with tensor indices in this case - we'll try to evaluate the kernel
                # and use the super method
                try:
                    return self.covar_mat._getitem(row_index, col_index, *batch_indices)
                except Exception:
                    raise TypeError(
                        f"{self.covar_mat.__class__.__name__} received an invalid slice. "
                        "Since the covariance function produces multiple outputs for input, the slice "
                        "should perfectly correspond with the number of outputs per input."
                    )

            # Otherwise - let's divide the slices by the number of outputs per input
            row_index = slice(row_start // num_outs_per_in_rows, row_end // num_outs_per_in_rows, None)
            col_index = slice(col_start // num_outs_per_in_cols, col_end // num_outs_per_in_cols, None)

        # Get the indices of x1 and x2 that matter for the kernel
        # Call x1[*batch_indices, row_index, :] and x2[*batch_indices, col_index, :]
        x1 = _x_getitem(self.x1, batch_indices, row_index)
        x2 = _x_getitem(self.x2, batch_indices, col_index)

        # Call params[*batch_indices, :, :]
        tensor_params = {
            name: val[(*batch_indices, *([_noop_index] * self.num_nonbatch_dimensions[name]))]
            for name, val in self.tensor_params.items()
        }

        # Now construct a kernel with those indices
        return self.__class__(
            x1,
            x2,
            covar_func=self.covar_func,
            num_outputs_per_input=self.num_outputs_per_input,
            num_nonbatch_dimensions=self.num_nonbatch_dimensions,
            **tensor_params,
            **self.nontensor_params,
        )

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return self.covar_mat @ rhs.contiguous()

    def _permute_batch(self, *dims: int) -> LinearOperator:
        x1 = self.x1.permute(*dims, -2, -1)
        x2 = self.x2.permute(*dims, -2, -1)
        tensor_params = {
            name: val.permute(*dims, *range(-self.num_nonbatch_dimensions[name], 0))
            for name, val in self.tensor_params.items()
        }
        return self.__class__(
            x1,
            x2,
            covar_func=self.covar_func,
            num_outputs_per_input=self.num_outputs_per_input,
            num_nonbatch_dimensions=self.num_nonbatch_dimensions,
            **tensor_params,
            **self.nontensor_params,
        )

    def _size(self) -> torch.Size:
        num_outs_per_in_rows, num_outs_per_in_cols = self.num_outputs_per_input
        return torch.Size(
            [
                *self.batch_broadcast_shape,
                self.x1.shape[-2] * num_outs_per_in_rows,
                self.x2.shape[-2] * num_outs_per_in_cols,
            ]
        )

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return self.__class__(
            self.x2,
            self.x1,
            covar_func=self.covar_func,
            num_outputs_per_input=self.num_outputs_per_input,
            num_nonbatch_dimensions=self.num_nonbatch_dimensions,
            **self.tensor_params,
            **self.nontensor_params,
        )

    def _unsqueeze_batch(self, dim: int) -> LinearOperator:
        x1 = self.x1.unsqueeze(dim)
        x2 = self.x2.unsqueeze(dim)
        tensor_params = {name: val.unsqueeze(dim) for name, val in self.tensor_params.items()}
        return self.__class__(
            x1,
            x2,
            covar_func=self.covar_func,
            num_outputs_per_input=self.num_outputs_per_input,
            num_nonbatch_dimensions=self.num_nonbatch_dimensions,
            **tensor_params,
            **self.nontensor_params,
        )

from typing import Any, Callable, Union

import torch

from jaxtyping import Float
from torch import Tensor

from ..utils.getitem import _noop_index, IndexType
from ..utils.memoize import cached
from ._linear_operator import LinearOperator, to_dense


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

        Each of the passed parameters should have a minimum of 2 (likely singleton) dimensions.
        Any additional dimension will be considered a batch dimension that broadcasts with the batch dimensions
        of x1 and x2.

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
        - `outputscale`: `(*batch_shape x 1 x 1)`

        Not adding the appropriate singleton dimensions to `lengthscale` and `outputscale` will lead to erroneous
        kernel matrix outputs.

    .. code-block:: python

        # NOTE: _covar_func intentionally does not close over any parameters
        def _covar_func(x1, x2, lengthscale, outputscale):
            # RBF kernel function
            # x1: ... x N x D
            # x2: ... x M x D
            # lengthscale: ... x 1 x D
            # outputscale: ... x 1 x 1
            x1 = x1.div(lengthscale)
            x2 = x2.div(lengthscale)
            sq_dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(dim=-1)
            kern = sq_dist.div(-2.0).exp().mul(outputscale.square())
            return kern


        # Batches of data
        x1 = torch.randn(3, 5, 6)
        x2 = torch.randn(3, 4, 6)
        # Broadcasting lengthscale and output parameters
        lengthscale = torch.randn(2, 1, 1, 6)
        outputscale = torch.randn(2, 1, 1, 1)
        kern = KernelLinearOperator(x1, x2, lengthscale, outputscale, covar_func=covar_func)

        # kern is of size 2 x 3 x 5 x 4

    .. warning ::

        `covar_func` should not close over any parameters. Any parameters that are closed over will not have
        propagated gradients.

    :param x1: The data :math:`\boldsymbol X_1.`
    :param x2: The data :math:`\boldsymbol X_2.`
    :param params: Additional hyperparameters (:math:`\boldsymbol \theta`) passed into covar_func.
    :param covar_func: The covariance function :math:`k_{\boldsymbol \theta}(\cdot, \cdot)`.
        Its arguments should be `x1`, `x2`, `*params`, `**kwargs`, and it should output the covariance matrix
        between :math:`\boldsymbol X_1` and :math:`\boldsymbol X_2`.
    :param kwargs: Any additional (non-hyperparameter) kwargs to pass into `covar_func`.
    """

    def __init__(
        self,
        x1: Float[Tensor, "... M D"],
        x2: Float[Tensor, "... N D"],
        *params: Float[Tensor, "... #P #D"],
        covar_func: Callable[..., Float[Union[Tensor, LinearOperator], "... M N"]],
        **kwargs: Any,
    ):
        # Ensure that x1, x2, and params can broadcast together
        try:
            batch_broadcast_shape = torch.broadcast_shapes(
                x1.shape[:-2], x2.shape[:-2], *[param.shape[:-2] for param in params]
            )
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
                f"Shape of kernel parameters ({', '.join([tuple(param.shape) for param in params])}) "
                f"is incompatible with data shapes x1.shape={tuple(x1.shape)}, x2.shape={tuple(x2.shape)}.\n"
                "Recall that parameters passed to KernelLinearOperator should have dimensionality compatible "
                "with the data (see documentation)."
            )

        # Create a version of each argument that is expanded to the broadcast batch shape
        x1 = x1.expand(*batch_broadcast_shape, *x1.shape[-2:]).contiguous()
        x2 = x2.expand(*batch_broadcast_shape, *x2.shape[-2:]).contiguous()
        params = [param.expand(*batch_broadcast_shape, *param.shape[-2:]) for param in params]

        # Standard constructor
        super().__init__(x1, x2, *params, covar_func=covar_func, **kwargs)
        self.batch_broadcast_shape = batch_broadcast_shape
        self.x1 = x1
        self.x2 = x2
        self.params = params
        self.covar_func = covar_func
        self.kwargs = kwargs

    @cached(name="kernel_diag")
    def _diagonal(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
        # Explicitly compute kernel diag via covar_func when it is needed rather than relying on lazy tensor ops.
        # We will do this by shoving all of the data into a batch dimension (i.e. compute a ... x N x 1 x 1 kernel)
        # and then squeeze out the batch dimensions
        x1 = self.x1.unsqueeze(-2)
        x2 = self.x2.unsqueeze(-2)
        params = [param.unsqueeze(-3) for param in self.params]
        diag_mat = to_dense(self.covar_func(x1, x2, *params, **self.kwargs))
        assert diag_mat.shape[-2:] == torch.Size([1, 1])
        return diag_mat[..., 0, 0]

    @property
    @cached(name="covar_mat")
    def covar_mat(self: Float[LinearOperator, "... M N"]) -> Float[Union[Tensor, LinearOperator], "... M N"]:
        return self.covar_func(self.x1, self.x2, *self.params, **self.kwargs)

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return self.covar_mat @ rhs.contiguous()

    def _size(self) -> torch.Size:
        return torch.Size([*self.batch_broadcast_shape, self.x1.shape[-2], self.x2.shape[-2]])

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return self.__class__(self.x2, self.x1, *self.params, covar_func=self.covar_func, **self.kwargs)

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        x1_ = self.x1[(*batch_indices, row_index)].unsqueeze(-2)
        x2_ = self.x2[(*batch_indices, col_index)].unsqueeze(-2)
        params_ = [param[batch_indices] for param in self.params]
        indices_mat = to_dense(self.covar_func(x1_, x2_, *params_, **self.kwargs))
        assert indices_mat.shape[-2:] == torch.Size([1, 1])
        return indices_mat[..., 0, 0]

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        dim_index = _noop_index

        # Get the indices of x1 and x2 that matter for the kernel
        # Call x1[*batch_indices, row_index, :]
        try:
            x1 = self.x1[(*batch_indices, row_index, dim_index)]
        # We're going to handle multi-batch indexing with a try-catch loop
        # This way - in the default case, we can avoid doing expansions of x1 which can be timely
        except IndexError:
            if isinstance(batch_indices, slice):
                x1 = self.x1.expand(1, *self.x1.shape[-2:])[(*batch_indices, row_index, dim_index)]
            elif isinstance(batch_indices, tuple):
                if any(not isinstance(bi, slice) for bi in batch_indices):
                    raise RuntimeError(
                        "Attempting to tensor index a non-batch matrix's batch dimensions. "
                        f"Got batch index {batch_indices} but my shape was {self.shape}"
                    )
                x1 = self.x1.expand(*([1] * len(batch_indices)), *self.x1.shape[-2:])
                x1 = x1[(*batch_indices, row_index, dim_index)]

        # Call x2[*batch_indices, col_index, :]
        try:
            x2 = self.x2[(*batch_indices, col_index, dim_index)]
        # We're going to handle multi-batch indexing with a try-catch loop
        # This way - in the default case, we can avoid doing expansions of x1 which can be timely
        except IndexError:
            if isinstance(batch_indices, slice):
                x2 = self.x2.expand(1, *self.x2.shape[-2:])[(*batch_indices, row_index, dim_index)]
            elif isinstance(batch_indices, tuple):
                if any([not isinstance(bi, slice) for bi in batch_indices]):
                    raise RuntimeError(
                        "Attempting to tensor index a non-batch matrix's batch dimensions. "
                        f"Got batch index {batch_indices} but my shape was {self.shape}"
                    )
                x2 = self.x2.expand(*([1] * len(batch_indices)), *self.x2.shape[-2:])
                x2 = x2[(*batch_indices, row_index, dim_index)]

        # Call params[*batch_indices, :, :]
        params = [param[(*batch_indices, _noop_index, _noop_index)] for param in self.params]

        # Now construct a kernel with those indices
        return self.__class__(x1, x2, *params, covar_func=self.covar_func, **self.kwargs)

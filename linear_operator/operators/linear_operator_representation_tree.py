#!/usr/bin/env python3


class LinearOperatorRepresentationTree(object):
    def __init__(self, linear_op):
        self._cls = linear_op.__class__
        self._differentiable_kwarg_names = linear_op._differentiable_kwarg_names
        self._nondifferentiable_kwargs = linear_op._nondifferentiable_kwargs

        counter = 0
        self.children = []
        for arg in list(linear_op._args) + list(linear_op._differentiable_kwarg_vals):
            if hasattr(arg, "representation") and callable(arg.representation):  # Is it a lazy tensor?
                representation_size = len(arg.representation())
                self.children.append((slice(counter, counter + representation_size, None), arg.representation_tree()))
                counter += representation_size
            else:
                self.children.append((counter, None))
                counter += 1

    def __call__(self, *flattened_representation):
        unflattened_representation = []

        for index, subtree in self.children:
            if subtree is None:
                unflattened_representation.append(flattened_representation[index])
            else:
                sub_representation = flattened_representation[index]
                unflattened_representation.append(subtree(*sub_representation))

        if len(self._differentiable_kwarg_names):
            args = unflattened_representation[: -len(self._differentiable_kwarg_names)]
            differentiable_kwargs = dict(
                zip(
                    self._differentiable_kwarg_names,
                    unflattened_representation[-len(self._differentiable_kwarg_names) :],
                )
            )
            return self._cls(*args, **differentiable_kwargs, **self._nondifferentiable_kwargs)
        else:
            return self._cls(*unflattened_representation, **self._nondifferentiable_kwargs)

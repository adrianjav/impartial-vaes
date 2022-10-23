from typing import Optional, Tuple, Any, Callable, List, Sequence

import torch
import torch.nn as nn

from .utils import batch_product


class MOOForLoop(nn.Module):

    inputs: Optional[torch.Tensor]

    def __init__(self, num_heads: int, moo_method: Optional[nn.Module] = None):
        super().__init__()

        # The user should explicitly set up the optimizers for the algorithms that have learnable parameters
        self._moo_method = [moo_method]
        self.num_heads = num_heads
        self.inputs = None
        self.outputs = None

        if self.moo_method is not None:
            self.register_full_backward_hook(MOOForLoop._hook)

    @property
    def moo_method(self):
        return self._moo_method[0]

    # @typechecked
    def _hook(self, grads_input: Tuple[torch.Tensor], grads_output: Any) -> Tuple[torch.Tensor]:
        moo_directions = self.moo_method(
            grads_output[0], self.inputs, self.outputs)
        self.outputs = None

        # we scale the gradients so that they have the same magnitude as the unmodified ones
        original_norm = grads_output[0].sum(dim=0).norm(p=2)
        moo_norm = moo_directions.sum(dim=0).norm(p=2).clamp_min(1e-10)
        moo_directions.mul_(original_norm / moo_norm)

        return moo_directions.sum(dim=0),

    def forward(self, z):
        extended_shape = [self.num_heads] + [-1 for _ in range(z.ndim)]
        if self.moo_method.requires_input and z.requires_grad:
            self.inputs = z.detach()
        extended_z = z.unsqueeze(0).expand(extended_shape)
        return extended_z


class MultiMOOForLoop(nn.Module):
    def __init__(self, num_heads: int, moo_methods: Sequence[Optional[nn.Module]] = None):
        super().__init__()

        self.num_inputs = len(moo_methods)
        self.loops = [MOOForLoop(num_heads, method) for method in moo_methods]

    def forward(self, *args):
        assert len(args) == self.num_inputs
        return (loop(z) for z, loop in zip(args, self.loops))


class MOOModule(nn.Module):

    inputs: Optional[torch.Tensor]

    def __init__(self, module: nn.Module, num_heads: int, moo_method: Optional[nn.Module]):
        super().__init__()

        # The user should explicitly set up the optimizers for the algorithms that have learnable parameters
        self.module = module
        self.num_heads = num_heads
        self.inputs = None
        self.outputs = None
        self._moo_method = [moo_method]
        self.fire = [False]

        self.module_dismantle()

    @property
    def moo_method(self):
        return self._moo_method[0]

    def module_dismantle(self):
        def extend(param):
            extended_shape = [self.num_heads] + [-1 for _ in range(param.ndim)]
            extended_z = param.unsqueeze(0).expand(extended_shape)
            extended_z.retain_grad()
            return extended_z

        # https://github.com/pytorch/pytorch/issues/50292
        def __getattribute__(self, name):
            try:
                return super(nn.Module, self).__getattribute__(name)
            except AttributeError as e:
                if '_parameters' in self.__dict__:
                    _parameters = self.__dict__['_parameters']
                    if name in _parameters:
                        value = _parameters[name]
                        if self.training and getattr(self, 'fire', [False])[0]:
                            if self.counter[0] == 0:
                                value = extend(value)
                                value.retain_grad()
                                setattr(self, f'_{name}_extended', value)
                            else:
                                value = getattr(self, f'_{name}_extended')

                            return value[self.counter[0]]

                        return value
                if '_buffers' in self.__dict__:
                    _buffers = self.__dict__['_buffers']
                    if name in _buffers:
                        return _buffers[name]
                if '_modules' in self.__dict__:
                    modules = self.__dict__['_modules']
                    if name in modules:
                        return modules[name]

                    # Note: Purposely reuse the specific error thrown from the original attempt to
                    # access the property. This ensures that the error message is correct for the
                    # case where a property method attempts to call a non-existent property, for example.
                raise e

        self.module.counter = [0]
        self.module.fire = [False]

        def dismantle(module):
            module.counter = self.module.counter
            module.fire = self.module.fire

            setattr(module.__class__, '__getattribute__', __getattribute__)

        self.module.apply(dismantle)

    def grad_step(self) -> None:
        grads = self._retrieve_grads()
        grads = torch.cat([g.flatten(start_dim=1)
                          for i, g in enumerate(grads)], dim=1)

        new_grad = self.moo_method(grads, self.inputs, self.outputs)
        self.outputs = None

        original_norm = grads.sum(dim=0).norm(p=2)
        moo_norm = new_grad.sum(dim=0).norm(p=2).clamp_min(1e-10)
        new_grad.mul_(original_norm / moo_norm)

        new_grad = new_grad.sum(dim=0)
        self._unflatten_and_set_grad(new_grad)

    def _retrieve_grads(self):
        grad = []

        def _retrieve_grads_(module):
            for n, p in module.named_parameters(recurse=False):
                param = getattr(module, '_' + n + '_extended')
                if param.grad is not None:
                    grad.append(param.grad)

        self.module.apply(_retrieve_grads_)
        return grad

    @staticmethod
    def _flatten_grad(grad: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.flatten(start_dim=1) for i, g in enumerate(grad)], dim=1)

    def _unflatten_and_set_grad(self, grad: torch.Tensor) -> None:
        pos_grad = 0

        def _unflatten_and_set_grad_(module):
            nonlocal pos_grad

            for n, p in module.named_parameters(recurse=False):
                param = getattr(module, n)
                if param.grad is not None:
                    param.grad = grad[pos_grad: pos_grad +
                                      p.numel()].view_as(p)
                    pos_grad += p.numel()

        self.module.apply(_unflatten_and_set_grad_)

    def forward(self, x):
        y = self.module(x)
        if self.training and getattr(self.module, 'fire', [False])[0]:
            self.inputs = x.detach()
            self.module.counter[0] = (getattr(self.module, 'counter', [0])[
                                      0] + 1) % self.num_heads
            if self.module.counter[0] == 0:
                self.module.fire = [False]
        return y

    def moo(self):
        if self.training:
            self.module.fire[0] = True


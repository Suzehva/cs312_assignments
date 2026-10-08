"""Explicit initialization, forward and optimizer policies for P4.1.

The supplied stress runner remains responsible for precision and measurements.
"""
from dataclasses import dataclass
import math
from torch import nn
from modeling import LlamaRMSNorm


@dataclass(frozen=True)
class Policy:
    name: str = 'mup'
    reference_width: int = 512
    reference_depth: int = 2

    def __post_init__(self):
        if self.name not in ('kaiming', 'mup', 'depth_mup', 'completep'):
            raise ValueError(self.name)
        if self.reference_width <= 0 or self.reference_depth <= 0:
            raise ValueError('Reference dimensions must be positive')

    def depth_factors(self, depth):
        r = depth / self.reference_depth
        if self.name == 'depth_mup':
            return r ** -.5, r ** -.5, r ** -.5
        if self.name == 'completep':
            return r ** -1, 1., r ** -1
        return 1., 1., 1.

    def initialize(self, model):
        n = model.config.width
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=1.)
            elif isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                if module is model.head and self.name != 'kaiming':
                    fan_in = self.reference_width
                nn.init.normal_(module.weight, std=1 / math.sqrt(fan_in))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, LlamaRMSNorm):
                module.weight.fill_(1.)
        model.output_multiplier = 1. if self.name == 'kaiming' else self.reference_width / n
        residual, _, _ = self.depth_factors(model.config.depth)
        for block in model.blocks:
            block.residual_multiplier = residual

    def parameter_groups(self, model, base_lr):
        _, block_lr, block_eps = self.depth_factors(model.config.depth)
        hidden_ids = {id(p) for module in model.blocks.modules()
                      if isinstance(module, nn.Linear) for p in module.parameters(recurse=False)}
        grouped = {}
        for name, param in model.named_parameters():
            lr = base_lr
            if id(param) in hidden_ids and self.name != 'kaiming':
                lr *= self.reference_width / model.config.width
            eps = 1e-8
            if name.startswith('blocks.'):
                lr *= block_lr
                eps *= block_eps
            grouped.setdefault((lr, eps), []).append(param)
        return [dict(params=params, lr=lr, eps=eps)
                for (lr, eps), params in grouped.items()]

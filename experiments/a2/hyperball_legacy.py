"""P1(d)'s Hyperball optimizer with Kaiyue Wen's AdamH core update.

Reference (``scale_invariant_update_`` and ``AdamH``):
https://github.com/KellerJordan/modded-nanogpt/pull/272
https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/results/20260430_adamh/7533dd87-107f-4a4f-8229-acbec0fb00ac.txt

Parameter groups, LR ratio, and scheduling follow the assignment handout,
not the reference benchmark's training configuration.
"""

import torch


ADAM_LR_RATIO = 0.000656 / 0.00630


@torch.no_grad()
def scale_invariant_update_(param, update, lr, eps=1e-10):
    """Kaiyue's norm-scaled step and projection to the pre-step norm."""
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)


class Hyperball(torch.optim.Optimizer):
    """Adam directions with relative-norm steps and fixed linear-weight norms.

    Linear-layer matrices use W <- W - lr * ||W||/||U|| * U followed by
    projection back to their pre-step Frobenius norm. Other parameters
    use ordinary Adam. Group LRs are scheduled by the shared trainer.
    """

    def __init__(self, groups, *, lr, betas, eps=1e-8):
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Hyperball requires dense gradients.")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                mean, variance = state["exp_avg"], state["exp_avg_sq"]
                mean.mul_(beta1).add_(gradient, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                direction = (mean / bc1) / ((variance / bc2).sqrt() + group["eps"])
                if group["hyperball"]:
                    scale_invariant_update_(parameter, direction, group["lr"])
                else:
                    parameter.add_(direction, alpha=-group["lr"])
        return loss


def build_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2):
    """Select all linear weights, including readout; leave other params in Adam."""
    if optimizer_name != "adamh" or weight_decay != 0:
        raise ValueError("P1(d) requires optimizer_name='adamh' and weight_decay=0.")
    linear_ids = {
        id(module.weight) for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    }
    matrices, ordinary = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (matrices if id(parameter) in linear_ids else ordinary).append(parameter)
    groups = []
    if matrices:
        groups.append(dict(params=matrices, lr=learning_rate, hyperball=True))
    if ordinary:
        groups.append(dict(params=ordinary, lr=learning_rate * ADAM_LR_RATIO, hyperball=False))
    return Hyperball(groups, lr=learning_rate, betas=(beta1, beta2))

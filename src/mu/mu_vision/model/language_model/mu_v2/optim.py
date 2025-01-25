import torch


class OrthogonalNesterov(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True, zeropower_iters=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, zeropower_iters=zeropower_iters)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeroth_power_via_newtonschulz5(g, steps=group["zeropower_iters"])
                scale = update.numel() ** 0.5 / update.norm()
                p.data.add_(update, alpha=-lr * scale)


@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2, "Newton-Schulz optimizer only works for 2D parameters"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)
        self.base_lrs = [group["lr"] for opt in self.optimizers for group in opt.param_groups]

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(**kwargs)

    def scale_lrs(self, lr_scale):
        for base_lr, group in zip(self.base_lrs, self.param_groups):
            group["lr"] = base_lr * lr_scale

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict):
        for opt, opt_state in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(opt_state)

    def __repr__(self):
        return f"CombinedOptimizer({self.optimizers})"

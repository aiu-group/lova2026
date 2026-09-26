from __future__ import annotations
from typing import Tuple, Callable

import torch
from torch.optim.optimizer import Optimizer
import math


def exists(val):
    return val is not None


# minimum memory footprint version where updates are with p.grad in place
class Lova(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        eps: float = 1.0e-8,  # 10-5 in reinforcement learning is better, similar to adam
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

        super(Lova, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr, wd, beta1, beta2, eps = (
                group["lr"],
                group["weight_decay"],
                *group["betas"],
                group["eps"],
            )

            for p in filter(lambda p: exists(p.grad), group["params"]):

                grad, state = p.grad, self.state[p]

                # first do weight decay
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))

                # log grad information before this line, otherwise misleading
                grad.sub_(exp_avg).pow_(2.0).mul_(1.0 - beta2)
                grad.addcmul_(exp_avg, exp_avg, value=beta2)
                grad.sqrt_().add_(eps)

                p.data.addcdiv_(exp_avg, grad, value=-lr)

        return loss

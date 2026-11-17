"""Token-based LR schedule and owner-sharded AdamW with FP32 master weights and moments (not FSDP)."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist

from ..common.rng import restore_rng, rng_state, seed_all  # noqa: F401  (re-exported for the trainer)


def token_lr(tokens, budget, schedule):
    peak, floor = schedule["peak_lr"], schedule["min_lr"]
    warmup = budget * schedule["warmup_fraction"]
    if tokens < warmup:
        return peak * max(0.0, tokens / max(warmup, 1))
    fraction = min(1.0, max(0.0, (tokens - warmup) / max(budget - warmup, 1)))
    return floor + (peak - floor) * (1 + math.cos(math.pi * fraction)) / 2


class ShardedAdamW:
    """FP32 accumulators; owner-sharded FP32 master weights and moments.

    Gradients are SUM-reduced once per effective batch. The caller divides
    local token loss sums by the global response-token count; no rank or
    microbatch mean averaging is performed. Supports a fixed-world-size resume.
    """
    def __init__(self, model, config, rank=0, world_size=1):
        self.rank, self.world_size = rank, world_size
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        loads, owners = [0] * world_size, {}
        for name, parameter in sorted(self.named, key=lambda x: (-x[1].numel(), x[0])):
            owner = min(range(world_size), key=lambda x: (loads[x], x))
            owners[name] = owner
            loads[owner] += parameter.numel()
        self.owners = [owners[n] for n, _ in self.named]
        self.grads = [torch.zeros_like(p, dtype=torch.float32) for _, p in self.named]
        self.masters = {n: torch.nn.Parameter(p.detach().float().clone())
                        for (n, p), owner in zip(self.named, self.owners) if owner == rank}
        decay, no_decay = [], []
        for name, master in self.masters.items():
            (decay if master.ndim >= 2 else no_decay).append(master)
        groups = [{"params": decay, "weight_decay": config["weight_decay"]},
                  {"params": no_decay, "weight_decay": 0.0}]
        self.optimizer = torch.optim.AdamW(groups, lr=1e-5,
                                          betas=(config["beta1"], config["beta2"]),
                                          eps=config["eps"],
                                          fused=self.named[0][1].is_cuda)
        self.max_grad_norm = config["max_grad_norm"]

    def accumulate(self):
        for (_, parameter), gradient in zip(self.named, self.grads):
            if parameter.grad is not None:
                gradient.add_(parameter.grad.detach())
                parameter.grad = None

    def step(self, lr):
        # Chunk large embedding/head matrices to keep collective temporaries small.
        if self.world_size > 1:
            for gradient in self.grads:
                for chunk in gradient.view(-1).split(4 * 1024 * 1024):
                    dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
        device = self.grads[0].device
        norm_sq = torch.zeros((), dtype=torch.float64, device=device)
        finite = torch.ones((), dtype=torch.bool, device=device)
        for gradient in self.grads:
            for chunk in gradient.view(-1).split(4 * 1024 * 1024):
                finite &= torch.isfinite(chunk).all()
                norm_sq += chunk.double().square().sum()
        norm = norm_sq.sqrt()
        if not finite.item() or not torch.isfinite(norm).item():
            raise FloatingPointError("nonfinite global gradient; optimizer update was not applied")
        factor = (self.max_grad_norm / (norm + 1e-12)).clamp(max=1).float()
        for ((name, _), owner, gradient) in zip(self.named, self.owners, self.grads):
            if owner == self.rank:
                gradient.mul_(factor)
                self.masters[name].grad = gradient
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for ((name, parameter), owner) in zip(self.named, self.owners):
                if owner == self.rank:
                    parameter.copy_(self.masters[name])
                if self.world_size > 1:
                    dist.broadcast(parameter, src=owner)
            for gradient in self.grads:
                gradient.zero_()
        return float(norm)

    def state_dict(self):
        return {"rank": self.rank, "world_size": self.world_size,
                "names": [n for n, _ in self.named], "owners": self.owners,
                "masters": {n: p.detach().cpu() for n, p in self.masters.items()},
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        if (state["rank"], state["world_size"], state["names"], state["owners"]) != (
                self.rank, self.world_size, [n for n, _ in self.named], self.owners):
            raise ValueError("resume requires the same model, parameter order, rank and world size")
        with torch.no_grad():
            for name, master in self.masters.items():
                master.copy_(state["masters"][name])
        self.optimizer.load_state_dict(state["optimizer"])
        # AdamW state tensors must remain FP32, including after deserialization.
        if any(t.dtype != torch.float32 for s in self.optimizer.state.values()
               for k, t in s.items() if k != "step" and isinstance(t, torch.Tensor)):
            raise ValueError("optimizer moments must be FP32")

"""Classical baselines: real zero-fill and primal-dual isotropic TV.

TV minimizes .5 ||A x-y||² + weight * sum sqrt(Dx²+Dy²), unconstrained real
images, periodic finite differences. PDHG uses prox of quadratic in dual.
"""
import torch


def gradient(x):
    return torch.stack((torch.roll(x, -1, -2)-x, torch.roll(x, -1, -1)-x), dim=0)


def gradient_adjoint(p):
    return torch.roll(p[0], 1, -2)-p[0] + torch.roll(p[1], 1, -1)-p[1]


@torch.no_grad()
def zero_filled(y, operator_assumed, prior=None, config=None):
    return operator_assumed.real_adjoint(y)


@torch.no_grad()
def tv(y, operator_assumed, prior=None, config=None):
    c = config or {}
    weight = c.get("tv_weight", .02)
    if weight <= 0:
        raise ValueError("TV weight must be positive")
    x = operator_assumed.real_adjoint(y)
    xbar = x.clone()
    q, p = torch.zeros_like(y), torch.zeros_like(gradient(x))
    step = .99/(operator_assumed.sensitivity.abs().max().square()+8).sqrt()
    for _ in range(c.get("tv_iterations", 200)):
        q = (q + step*(operator_assumed(xbar)-y))/(1+step)
        p = p + step*gradient(xbar)
        p /= (p.square().sum(0).sqrt()/weight).clamp_min(1).unsqueeze(0)
        new = x-step*(operator_assumed.real_adjoint(q)+gradient_adjoint(p))
        xbar, x = 2*new-x, new
    return x

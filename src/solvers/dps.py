"""DPS: Chung et al., arXiv:2209.14687, Eqs. (10),(15), Algorithm 1.

Paper x_i -> x; xhat_0 -> x0; A -> op; zeta_i -> zeta/norm.
Requested variant: grad ||r|| with zeta/norm (dps_loss='norm').
The paper's Algorithm 1 instead uses grad ||r||²; select 'squared' for that
variant. Both are labeled in provenance; they are NOT algebraically identical.
Network weights are frozen, but the full epsilon input Jacobian is retained.
"""
import torch
from .common import tweedie, ddim_step, run_guided


def update(x, t, a, ap, y, op, prior, c, generator):
    with torch.enable_grad():
        x = x.requires_grad_(True)
        eps = prior.predict_eps(x, t)
        x0 = tweedie(x, eps, a)
        norm = (y-op(x0)).abs().square().flatten(1).sum(1).clamp_min(1e-16).sqrt()
        kind = c.get("dps_loss", "norm")
        if kind not in ("norm", "squared"):
            raise ValueError("dps_loss must be norm or squared")
        loss = norm if kind == "norm" else norm.square()
        grad = torch.autograd.grad(loss.sum(), x)[0]
        unconditional = ddim_step(x0, eps, a, ap, c.get("eta", 0), generator)
        scale = c.get("zeta", .1) / norm.detach().clamp_min(1e-8)
        return unconditional - scale[:, None, None, None] * grad


def solve(y, operator_assumed, prior, config):
    return run_guided(y, operator_assumed, prior, config, update)

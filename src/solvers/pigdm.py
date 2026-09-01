"""PiGDM: Song et al., ICLR 2023, Eq. (7) noisy VJP, Eq. (8) noiseless.

Paper xhat_t -> x0; H -> op; r_t² -> (1-a)/a; sigma_y -> op.sigma.
The requested VE-style variance approximation is used. For a real latent,
apply the exact real normal-operator diagonal via push-through; treating
complex AA^H as identity would ignore conjugate redundancy. One autograd.grad
per timestep computes J_x0(x)^T v, never a full Jacobian. Conditional score
s=s_prior+g is integrated with DDIM by eps_cond=eps-sqrt(1-a)*g.
"""
import torch
from .common import tweedie, ddim_step, run_guided


def vjp(x0, x, vector):
    return torch.autograd.grad(x0, x, grad_outputs=vector.detach())[0]


def update(x, t, a, ap, y, op, prior, c, generator):
    with torch.enable_grad():
        x = x.requires_grad_(True)
        eps = prior.predict_eps(x, t)
        x0 = tweedie(x, eps, a)
        with torch.no_grad():
            vector = op.covariance_backproject(y-op(x0), (1-a)/a)
        guidance = vjp(x0, x, vector)
        eps_cond = eps.detach() - c.get("pigdm_scale", 1.)*(1-a).sqrt()*guidance
        x0_cond = tweedie(x.detach(), eps_cond, a)
        return ddim_step(x0_cond, eps_cond, a, ap, c.get("eta", 0.), generator)


def solve(y, operator_assumed, prior, config):
    operator_assumed.require_unit_sensitivity()
    return run_guided(y, operator_assumed, prior, config, update)

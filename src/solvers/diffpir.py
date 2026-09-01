"""DiffPIR: Zhu et al., arXiv:2305.08995, Eq. (12b), Algorithm 1 lines 4–7.

x_0^(t) -> x0; xhat_0^(t) -> z; sigma_bar_t² -> (1-a)/a;
rho_t -> rho=lambda*sigma²/sigma_bar_t²; paper zeta -> config eta.
Real-domain Fourier proximal is exact, including asymmetric masks. No CG or
network gradients. Effective noise is recomputed after the proximal update.
"""
import torch
from .common import tweedie, run_guided


@torch.no_grad()
def update(x, t, a, ap, y, op, prior, c, generator):
    eps = prior.predict_eps(x, t)
    x0 = tweedie(x, eps, a)
    rho = c.get("lambda", 1.) * op.sigma**2 / ((1-a)/a).clamp_min(1e-12)
    z = op.prox(x0, y, rho)
    effective_eps = (x-a.sqrt()*z)/(1-a).sqrt()
    eta = c.get("eta", 0.)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return ap.sqrt()*z + (1-ap).sqrt()*((1-eta)**.5*effective_eps + eta**.5*noise)


def solve(y, operator_assumed, prior, config):
    operator_assumed.require_unit_sensitivity()
    return run_guided(y, operator_assumed, prior, config, update)

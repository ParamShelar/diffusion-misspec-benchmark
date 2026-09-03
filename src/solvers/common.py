"""Shared DDIM state evolution and explicit image/measurement scaling."""
import time
import torch
from src.operators.fourier import FourierOperator


def tweedie(x, eps, abar):
    return (x - (1-abar).sqrt()*eps) / abar.sqrt()


def bounded_tweedie(x, eps, abar):
    """Predict the clean signed image and enforce its training-domain bounds.

    At the high-noise end of a cosine schedule, ``abar`` is very small.  Even
    a small epsilon-prediction error can therefore make the algebraic Tweedie
    estimate enormous.  The unconditional sampler already uses this bounded
    estimate; guided samplers must use the same estimate before applying a
    data-consistency update.
    """
    return tweedie(x, eps, abar).clamp(-1, 1)


def ddim_step(x0, eps, abar, previous, eta, generator):
    variance = ((1-previous)/(1-abar) * (1-abar/previous)).clamp_min(0)
    sigma = eta * variance.sqrt()
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    return previous.sqrt()*x0 + (1-previous-sigma.square()).clamp_min(0).sqrt()*eps + sigma*noise


def timesteps(prior, steps, start=None):
    top = len(prior.alphas_cumprod)-1 if start is None else start
    if not 1 <= steps <= top+1:
        raise ValueError("steps out of schedule range")
    return torch.linspace(top, 0, steps).round().long().tolist()


def run_guided(y, operator, prior, config, update):
    """x_t is in the prior's signed coordinates u=2x-1.

    Transform both measurement and noise exactly: y_u=2y-A(1), sigma_u=2sigma.
    Guidance never observes the true operator or the reference image.
    """
    op = FourierOperator(operator.mask, operator.sensitivity, 2*operator.sigma)
    shape = y.shape
    generator = torch.Generator(device=y.device).manual_seed(config.get("seed", 0))
    x = torch.randn(shape, device=y.device, dtype=y.real.dtype, generator=generator)
    yn = 2*y - op(torch.ones_like(x))
    ts = timesteps(prior, config["steps"])
    for i, t in enumerate(ts):
        abar = prior.alphas_cumprod[t].to(x)
        previous = prior.alphas_cumprod[ts[i+1]].to(x) if i+1 < len(ts) else x.new_tensor(1.)
        x = update(x.detach(), t, abar, previous, yn, op, prior, config, generator).detach()
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"Nonfinite sampler output at t={t}; inspect guidance strength")
    result = (x + 1) / 2
    if config.get("final_consistency", False):
        if operator.sigma != 0:
            raise ValueError("Hard final consistency is only allowed when assumed sigma=0")
        result = operator.prox(result, y, 0.)
    return result


@torch.no_grad()
def sample(prior, count, steps, device, seed=0, x=None, start=None, deadline=None):
    generator = torch.Generator(device=device).manual_seed(seed)
    if x is None:
        x = torch.randn((count, 1, 64, 64), generator=generator, device=device)
    ts = timesteps(prior, steps, start)
    for i, t in enumerate(ts):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Sampling stopped at the wall-clock deadline")
        a = prior.alphas_cumprod[t].to(x)
        ap = prior.alphas_cumprod[ts[i+1]].to(x) if i+1 < len(ts) else x.new_tensor(1.)
        eps = prior.predict_eps(x, t)
        # Prior sampling follows standard DDIM with a clipped denoised estimate.
        x0 = bounded_tweedie(x, eps, a)
        x = ddim_step(x0, eps, a, ap, 0., generator)
    return (x+1)/2

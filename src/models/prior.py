"""Epsilon priors in [-1,1] image coordinates; sampling always uses EMA weights.

CIFAR fallback is explicitly an out-of-domain adapter: each 64x64 grayscale
image is four independent 32x32 tiles, replicated to RGB; predicted epsilon
is averaged across RGB. No resizing of the measured image or Fourier grid.
"""
from pathlib import Path
import hashlib
import torch
from src.runtime import digest, file_digest


class Prior:
    def __init__(self, model, alphas_cumprod, identity, tiled=False):
        self.model = model.eval().requires_grad_(False)
        self.alphas_cumprod = alphas_cumprod.to(next(model.parameters()).device)
        self.identity = identity
        self.tiled = tiled

    def predict_eps(self, x, t):
        if self.tiled:
            rows = []
            for row in range(0, x.shape[-2], 32):
                columns = []
                for col in range(0, x.shape[-1], 32):
                    tile = x[..., row:row+32, col:col+32].expand(-1, 3, -1, -1)
                    columns.append(self.model(tile, t).sample.mean(dim=1, keepdim=True))
                rows.append(torch.cat(columns, dim=-1))
            return torch.cat(rows, dim=-2)
        return self.model(x, t).sample


def weights_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_prior(config, device):
    from diffusers import UNet2DModel, DDPMScheduler
    p = config["prior"]
    if p["kind"] == "cifar":
        kwargs = dict(cache_dir=p.get("cache_dir"), revision=p.get("revision", "main"))
        model = UNet2DModel.from_pretrained(p["model_id"], **kwargs).to(device)
        scheduler = DDPMScheduler.from_pretrained(p["model_id"], **kwargs)
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("Only epsilon-prediction priors are supported")
        identity = {"kind": "cifar", "model": p["model_id"], "weights_sha256": weights_digest(model),
                    "adapter": "four-32px-grayscale-tiles-v1", "schedule": scheduler.alphas_cumprod.tolist()}
        return Prior(model, scheduler.alphas_cumprod, digest(identity), tiled=True)
    if p["kind"] == "trained":
        checkpoint = Path(p["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "ema" not in payload or "model_config" not in payload:
            raise ValueError("Expected an EMA training checkpoint, not raw model weights")
        from src.models.unet import make_model
        model = make_model(payload["model_config"])
        model.load_state_dict(payload["ema"], strict=True)
        return Prior(model.to(device), payload["alphas_cumprod"], file_digest(checkpoint))
    raise ValueError("prior.kind must be cifar or trained")

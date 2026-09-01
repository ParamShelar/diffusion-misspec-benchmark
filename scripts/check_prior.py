"""Prior gate. Research PASS requires PSNR>=15 dB, FID<=200, FID-floor<=150.

These are fixed engineering sanity thresholds, not validated medical criteria.
Full: 2000 generated vs 2000 validation images; floor uses two DISJOINT groups
of 2000 validation images. Never train on either. Smoke evaluates 4 samples,
20 DDIM steps, genuine 2048-D Inception FID; SMOKE_PASS cannot authorize research.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torchvision.utils import save_image
from src.data.datasets import from_config, tensor_digest
from src.models.prior import load_prior
from src.metrics.quality import InceptionFeatures, fid, psnr
from src.solvers.common import sample
from src.runtime import load_config, device_for, seed_all, atomic_json, source_digest, SmokeDeadline, environment_info

THRESHOLDS = {"psnr_min": 15., "fid_max": 200., "fid_excess_max": 150.}


def run_gate(config, prior=None, deadline=None):
    seed_all(config["seed"])
    device = device_for(config)
    prior = prior or load_prior(config, device)
    held = from_config(config, "val")
    smoke = config.get("smoke", False)
    count = 4 if smoke else 2000
    steps = config["sampling"]["steps"]
    if len(held) < 2*count:
        raise ValueError(f"Gate needs {2*count} distinct validation slices; found {len(held)}")
    order = torch.randperm(len(held),generator=torch.Generator().manual_seed(config["seed"]+882))
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    output_gate = out / "prior_gate.json"
    # A crash or failed retry must not leave a stale PASS behind.
    atomic_json(output_gate, {"status": "RUNNING", "prior_id": prior.identity})
    started = time.monotonic()
    inception = InceptionFeatures(device)
    generated, real_a, real_b, grid = [], [], [], []
    batch_size = 1 if smoke else config.get("quality", {}).get("batch_size", 16)
    if not 1 <= batch_size <= 16:
        raise ValueError("Quality-gate batch_size must be between 1 and 16")
    for i in range(0, count, batch_size):
        if deadline is not None and time.monotonic() > deadline:
            atomic_json(output_gate, {"status": "FAIL", "reason": "wall-clock budget exhausted", "prior_id": prior.identity})
            return json.loads(output_gate.read_text())
        b = min(batch_size, count-i)
        try:
            image = sample(prior, b, steps, device, config["seed"]+10000+i, deadline=deadline)
        except TimeoutError:
            result = {"status":"FAIL","reason":"wall-clock budget exhausted during quality sampling", "prior_id":prior.identity}
            atomic_json(output_gate,result)
            return result
        generated.append(inception(image))
        real_a.append(inception(held[order[i:i+b]]))
        real_b.append(inception(held[order[count+i:count+i+b]]))
        if i < 64:
            grid.append(image[:64-i].cpu())
        if smoke or i//100 != (i+b)//100 or i+b==count:
            print(f"Prior gate samples: {i+b}/{count}", flush=True)
    save_image(torch.cat(grid), out / "prior_samples.png", nrow=2 if smoke else 8)
    a = prior.alphas_cumprod[250]
    g = torch.Generator(device=device).manual_seed(config["seed"]+20000)
    original = held[order[:1]].to(device)
    noisy = a.sqrt()*(2*original-1)+(1-a).sqrt()*torch.randn(original.shape, device=device, generator=g)
    try:
        denoised = sample(prior, 1, min(steps, 251), device, x=noisy, start=250, deadline=deadline)
    except TimeoutError:
        result = {"status":"FAIL","reason":"wall-clock budget exhausted during denoising", "prior_id":prior.identity}
        atomic_json(output_gate,result)
        return result
    reconstruction_psnr = float(psnr(original, denoised))
    save_image(torch.cat((original, ((noisy+1)/2).clamp(0,1), denoised)), out / "noise_then_denoise.png", nrow=3)
    gen, ra, rb = map(np.concatenate, (generated, real_a, real_b))
    score, floor = fid(gen, ra), fid(ra, rb)
    numerical = all(np.isfinite(v) for v in (score, floor, reconstruction_psnr))
    in_budget = deadline is None or time.monotonic() <= deadline
    passed = numerical and in_budget and reconstruction_psnr >= THRESHOLDS["psnr_min"] and score <= THRESHOLDS["fid_max"] and score-floor <= THRESHOLDS["fid_excess_max"]
    result = dict(status=("SMOKE_PASS" if numerical and in_budget else "FAIL") if smoke else ("PASS" if passed else "FAIL"),
                  scientific_pass=bool(passed and not smoke), smoke=smoke, prior_id=prior.identity,
                  validation_hash=tensor_digest(held), data=config["data"], source_hash=source_digest(),
                  environment=environment_info(), steps=steps, samples=count, psnr=reconstruction_psnr, fid=score, fid_floor=floor,
                  fid_excess=score-floor, thresholds=THRESHOLDS, elapsed_seconds=time.monotonic()-started)
    atomic_json(output_gate, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    timer = SmokeDeadline(args.smoke)
    config = load_config(args.config, args.smoke)
    if args.device:
        config["device"] = args.device
    try:
        result = run_gate(config)
        return 0 if result["status"] in ("PASS", "SMOKE_PASS") else 1
    finally:
        timer.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Paired misspecification sweeps; resume by immutable experiment/row identity."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import load_config, device_for, seed_all, digest, source_digest, git_commit, atomic_json, SmokeDeadline, environment_info
from src.data.datasets import from_config, tensor_digest
from src.models.prior import load_prior
from src.operators.fourier import FourierOperator, make_mask, mismatch_mask, sensitivity_map
from src.solvers import SOLVERS
from src.metrics.quality import Metrics
from src.ledger import Ledger, row_key


def validate_gate(config, prior, validation_hash):
    path = Path(config["output"]) / "prior_gate.json"
    if not path.exists():
        raise RuntimeError("Prior gate missing. Run scripts/check_prior.py with the same config and --smoke mode first.")
    gate = json.loads(path.read_text())
    expected = "SMOKE_PASS" if config.get("smoke") else "PASS"
    if gate.get("status") != expected or gate.get("prior_id") != prior.identity or gate.get("validation_hash") != validation_hash or gate.get("source_hash") != source_digest() or gate.get("steps") != config["sampling"]["steps"] or gate.get("environment") != environment_info():
        raise RuntimeError("Prior gate failed, is stale, or belongs to different weights/data/code/sampling settings. Rerun check_prior.py.")
    return gate


def operators(config, axis, level, seed, device):
    c = config["operator"]
    mask = make_mask(64, c["mask"], c["acceleration"], c["acs_lines"], seed)
    assumed_mask = mismatch_mask(mask, level, c["acs_lines"], seed+73) if axis == "mask" else mask.clone()
    sigma = c["sigma"]
    if axis == "noise" and sigma == 0:
        raise ValueError("log(sigma_hat/sigma) noise sweep requires positive true sigma")
    sigma_hat = sigma * math.exp(level) if axis == "noise" else sigma
    sensitivity = sensitivity_map(64, level if axis == "sensitivity" else 0., device)
    true = FourierOperator(mask.to(device), sensitivity, sigma)
    assumed = FourierOperator(assumed_mask.to(device), sigma=sigma_hat)
    disagreement = float((mask != assumed_mask).float().mean())
    return true, assumed, disagreement


def run(config):
    seed_all(config["seed"])
    device = device_for(config)
    prior = load_prior(config, device)
    validate_gate(config, prior, tensor_digest(from_config(config, "val")))
    dataset = from_config(config)
    data_hash = tensor_digest(dataset)
    code_hash, commit = source_digest(), git_commit()
    identity_config = copy.deepcopy(config)
    identity_config.pop("output", None)
    config_hash = digest(dict(config=identity_config, prior_id=prior.identity, data_hash=data_hash,
                              source_hash=code_hash, git_commit=commit, environment=environment_info(), device=str(device)))
    out = Path(config["output"])
    artifacts = out / "reconstructions" / config_hash
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_json(out / f"config-{config_hash}.json", dict(config=config, config_hash=config_hash,
                source_hash=code_hash, git_commit=commit, prior_id=prior.identity, data_hash=data_hash, environment=environment_info()))
    ledger = Ledger(out / "metrics.csv")
    seen = {row_key(row) for row in ledger.rows()}
    names = config["sweep"]["solvers"]
    if set(names) != set(SOLVERS) or len(names) != len(SOLVERS):
        raise ValueError("Every benchmark sweep must include each of zero_filled, tv, dps, diffpir, pigdm exactly once")
    metric = Metrics(device)
    written = 0
    for axis, levels in config["sweep"]["axes"].items():
        if axis not in ("mask", "noise", "sensitivity"):
            raise ValueError(f"Unknown axis {axis}")
        for level in levels:
            for seed in config["sweep"]["seeds"]:
                truth, assumed, disagreement = operators(config, axis, level, seed, device)
                for image_id in config["sweep"]["image_ids"]:
                    if not 0 <= image_id < len(dataset):
                        raise ValueError(f"image_id {image_id} outside provided split")
                    reference = dataset[image_id:image_id+1].to(device)
                    # Paired noise and sampler initialization across all axes/levels/solvers.
                    paired_seed = seed*100003 + image_id
                    generator = torch.Generator(device=device).manual_seed(paired_seed)
                    y = truth.measure(reference, generator)
                    for name in names:
                        row = dict(config_hash=config_hash, solver=name, axis=axis, level=float(level), seed=seed, image_id=image_id)
                        if row_key(row) in seen:
                            continue
                        seed_all(paired_seed)
                        solver_config = dict(config["sampling"], seed=paired_seed,
                                             tv_weight=config["sweep"]["tv_weight"], tv_iterations=config["sweep"]["tv_iterations"])
                        estimate = SOLVERS[name](y, assumed, prior, solver_config)
                        metrics = metric(reference, estimate, y, assumed, truth)
                        path = artifacts / f"{name}-{axis}-{level}-{seed}-{image_id}.pt"
                        tmp = path.with_suffix(".tmp")
                        torch.save(dict(reference=reference.cpu(), reconstruction=estimate.cpu()), tmp)
                        os.replace(tmp, path)
                        row.update(git_commit=commit, source_hash=code_hash, prior_id=prior.identity, data_hash=data_hash,
                                   smoke=config["smoke"], sigma_true=truth.sigma, sigma_assumed=assumed.sigma,
                                   mask_disagreement=disagreement, mask_density=truth.mask.mean().item(),
                                   artifact=str(path.relative_to(out)), **metrics)
                        ledger.append(row)
                        seen.add(row_key(row))
                        written += 1
                        print(f"{name:11s} {axis:11s} level={level:g} seed={seed} image={image_id} PSNR={metrics['psnr']:.2f}", flush=True)
    print(f"Sweep complete: {written} new rows; {out / 'metrics.csv'}", flush=True)
    return out / "metrics.csv"


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
        run(config)
    finally:
        timer.close()


if __name__ == "__main__":
    main()

"""Degradation curves and low/middle/high grids with both classical baselines."""
import argparse
import collections
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
os.environ.setdefault("MPLCONFIGDIR",str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.ledger import Ledger
from src.runtime import load_config, SmokeDeadline

NAMES = ["zero_filled", "tv", "dps", "diffpir", "pigdm"]
LABELS = ["Zero-filled", "TV", "DPS", "DiffPIR", "PiGDM"]
METRICS = ["psnr", "ssim", "lpips", "residual_assumed", "residual_true", "residual_gap"]


def make_figures(config, selected_hash=None):
    out = Path(config["output"])
    all_rows = Ledger(out / "metrics.csv").rows()
    hashes = sorted({r["config_hash"] for r in all_rows})
    if selected_hash is None:
        if len(hashes) != 1:
            raise ValueError("Select --config-hash when CSV contains multiple experiments (or no completed rows)")
        selected_hash = hashes[0]
    rows = [r for r in all_rows if r["config_hash"] == selected_hash]
    if not rows:
        raise ValueError("No rows for the requested experiment")
    saved = json.loads((out / f"config-{selected_hash}.json").read_text())["config"]
    expected = len(saved["sweep"]["solvers"])*len(saved["sweep"]["seeds"])*len(saved["sweep"]["image_ids"])*sum(map(len, saved["sweep"]["axes"].values()))
    if len(rows) != expected:
        raise ValueError(f"Sweep incomplete ({len(rows)}/{expected} rows). Resume it before generating figures.")
    figdir = out / "figures" / selected_hash
    figdir.mkdir(parents=True, exist_ok=True)
    smoke = rows[0]["smoke"] == "True"
    prefix = "SMOKE ONLY · " if smoke else ""
    for axis in sorted({r["axis"] for r in rows}):
        subset = [r for r in rows if r["axis"] == axis]
        if not {"zero_filled", "tv"}.issubset({r["solver"] for r in subset}):
            raise ValueError("Classical baselines missing")
        levels = sorted({float(r["level"]) for r in subset})
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for ax, metric in zip(axes.flat, METRICS):
            for name, label in zip(NAMES, LABELS):
                means, errors = [], []
                for level in levels:
                    by_seed = collections.defaultdict(list)
                    for r in subset:
                        if r["solver"] == name and float(r["level"]) == level:
                            by_seed[r["seed"]].append(float(r[metric]))
                    seed_means = [np.mean(v) for _, v in sorted(by_seed.items())]
                    means.append(np.mean(seed_means))
                    errors.append(np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.)
                ax.errorbar(levels, means, yerr=errors, label=label, marker="o", capsize=3)
            ax.set(xlabel="log(sigma assumed / sigma true)" if axis == "noise" else f"{axis} level", ylabel=metric.replace("_", " "))
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{prefix}{axis} misspecification · mean over images, ±1 SD over seeds")
        fig.savefig(figdir / f"{axis}_curves.png", dpi=150)
        plt.close(fig)
        chosen = [levels[0], levels[len(levels)//2], levels[-1]]
        fig, axs = plt.subplots(3, 6, figsize=(12, 7), constrained_layout=True)
        seed = min(int(r["seed"]) for r in subset)
        image_id = min(int(r["image_id"]) for r in subset)
        for j, level in enumerate(chosen):
            for k, name in enumerate(NAMES):
                r = next(r for r in subset if r["solver"] == name and float(r["level"]) == level and int(r["seed"]) == seed and int(r["image_id"]) == image_id)
                payload = torch.load(out / r["artifact"], weights_only=True)
                if k == 0:
                    axs[j, 0].imshow(payload["reference"][0,0], cmap="gray", vmin=0, vmax=1)
                    axs[j, 0].set_title(f"Reference · {level:g}")
                axs[j, k+1].imshow(payload["reconstruction"][0,0].clamp(0,1), cmap="gray", vmin=0, vmax=1)
                axs[j, k+1].set_title(f"{LABELS[k]}\n{float(r['psnr']):.1f} dB", fontsize=9)
        for ax in axs.flat:
            ax.axis("off")
        fig.suptitle(f"{prefix}{axis} · low / middle / high parameter · seed {seed}, image {image_id}")
        fig.savefig(figdir / f"{axis}_qualitative.png", dpi=150)
        plt.close(fig)
    print(f"Figures: {figdir}")
    return figdir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--config-hash")
    args = p.parse_args()
    timer = SmokeDeadline(args.smoke)
    try:
        make_figures(load_config(args.config, args.smoke), args.config_hash)
    finally:
        timer.close()


if __name__ == "__main__":
    main()

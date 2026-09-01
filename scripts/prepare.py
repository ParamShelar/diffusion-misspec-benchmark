"""Automatically fetch data, prior and metric weights before timed smoke runs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import load_config, seed_all
from src.data.datasets import from_config
from src.models.prior import load_prior
from src.metrics.quality import Metrics, InceptionFeatures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--smoke", action="store_true", help="Accept local tier; downloads are not time-bounded")
    args = p.parse_args()
    c = load_config(args.config, args.smoke)
    seed_all(c["seed"])
    torch.hub.set_dir(str(Path(".cache/torch").resolve()))
    print("Downloading/preprocessing official data splits...", flush=True)
    x = from_config(c)
    print(f"Selected split: {tuple(x.shape)}", flush=True)
    prior = load_prior(c, "cpu")
    print(f"Prior fingerprint: {prior.identity}", flush=True)
    del prior
    Metrics("cpu")
    InceptionFeatures("cpu")
    print("Caches prepared. Timed smoke commands can run without downloads.", flush=True)


if __name__ == "__main__":
    main()

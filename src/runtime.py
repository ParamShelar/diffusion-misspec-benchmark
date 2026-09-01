"""Configuration, reproducibility, and conservative hardware admission checks."""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def source_digest():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): file_digest(p)
                   for folder in ("src", "scripts") for p in sorted((root / folder).rglob("*.py"))})


def environment_info():
    packages = ("torch", "torchvision", "diffusers", "numpy", "scipy", "scikit-image", "lpips", "torch-fidelity")
    return {name: importlib.metadata.version(name) for name in packages}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        return "uncommitted"


def seed_all(seed):
    torch.hub.set_dir(str(Path(".cache/torch").resolve()))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def load_config(path, smoke=False):
    with open(path) as f:
        c = yaml.safe_load(f)
    c = copy.deepcopy(c)
    if c["tier"] not in ("local", "colab"):
        raise ValueError("tier must be local or colab")
    c["smoke"] = smoke
    if smoke:
        c["sampling"].update(steps=20, batch_size=1)
        c["sweep"].update(seeds=[0], image_ids=[0], tv_iterations=8)
        c["output"] = str(Path(c["output"]) / "smoke")
    elif c["tier"] == "local":
        raise ValueError("Local tier is for --smoke only; use configs/colab.yaml for research sweeps.")
    steps = c["sampling"]["steps"]
    if not 1 <= steps <= 1000 or c["sampling"]["batch_size"] != 1:
        raise ValueError("Sweep supports batch_size=1 and 1..1000 sampling steps.")
    if not 0 <= c["sampling"]["eta"] <= 1:
        raise ValueError("eta must be in [0,1]")
    if c["operator"]["sigma"] < 0:
        raise ValueError("sigma must be nonnegative")
    return c


def device_for(config, training=False):
    requested = config.get("device", "auto")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if training and (config["tier"] != "colab" or dev.type != "cuda"):
        raise RuntimeError("Training requires the Colab tier with an L4-class GPU.")
    if not config.get("smoke") and dev.type != "cuda":
        raise RuntimeError("Full runs require a GPU; use --smoke for CPU validation.")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Fix the GPU runtime or use --smoke --device cpu.")
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        free = torch.cuda.mem_get_info(dev)[0] / 2**30
        budget = min(4 if config["tier"] == "local" else 24, total)
        # Conservative planning estimate, not a guarantee about allocator fragmentation.
        batch = config.get("training", {}).get("batch_size", 128) if training else config["sampling"]["batch_size"]
        estimated = 3.0 + .12 * batch if training else 1.8 + .3 * batch
        print(f"GPU: {torch.cuda.get_device_name(dev)}, {total:.1f} GiB; estimated need {estimated:.1f} GiB")
        if estimated > min(budget, free) * .95:
            raise RuntimeError(f"Config exceeds VRAM budget: estimated {estimated:.1f} GiB, tier/device/free limit {min(budget, free):.1f} GiB. Reduce batch size.")
    return dev


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


class SmokeDeadline:
    """Bound compute/download stalls on Linux; never turn a timeout into a PASS."""
    def __init__(self, smoke, seconds=175):
        self.enabled = smoke and hasattr(signal, "SIGALRM")
        if self.enabled:
            signal.signal(signal.SIGALRM, self.expired)
            signal.alarm(seconds)

    @staticmethod
    def expired(*_):
        raise TimeoutError("Smoke exceeded 175 seconds. Warm download caches first; no successful run was claimed.")

    def close(self):
        if self.enabled:
            signal.alarm(0)

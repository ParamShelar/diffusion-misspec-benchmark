"""Atomic Drive checkpoints with complete RNG state and bounded retention."""
import os
from pathlib import Path
import random
import time
import numpy as np
import torch


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(payload,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    with open(temp,"wb") as f:
        torch.save(payload,f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)


def prune_checkpoints(directory, keep=3):
    checkpoints = sorted(Path(directory).glob("step-*.pt"))
    for p in checkpoints[:-keep]:
        p.unlink()


def latest_checkpoint(directory):
    """Try newest regular checkpoint first, then older copies and best.

    A completed atomic rename defines a usable candidate. Corrupt files are
    reported and skipped; partial *.tmp files never become resume candidates.
    """
    paths = sorted(Path(directory).glob("step-*.pt"),reverse=True)
    best = Path(directory)/"best.pt"
    if best.exists():
        paths.append(best)
    for path in paths:
        try:
            payload = torch.load(path,map_location="cpu",weights_only=False)
            required = {"model","ema","optimizer","step","rng","model_config","alphas_cumprod"}
            if not required.issubset(payload):
                raise ValueError("Incomplete checkpoint")
            return path,payload
        except (OSError,RuntimeError,EOFError,ValueError) as error:
            print(f"Cannot restore {path.name}: {error}",flush=True)
    if paths:
        raise RuntimeError("Existing checkpoints are unusable; refusing to silently restart training")
    return None,None

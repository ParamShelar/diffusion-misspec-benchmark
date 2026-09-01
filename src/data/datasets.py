"""Official splits, one float32 cache per source, interchangeable dataset backends."""
import os
from pathlib import Path
import numpy as np
import torch
from filelock import FileLock
from src.runtime import digest, file_digest

CACHE_VERSION = "percentile99-v1"
SPLITS = ("train", "val", "test")


def preprocess(images):
    x = torch.as_tensor(np.asarray(images).copy())
    if x.ndim == 3:
        x = x[:, None]
    if x.ndim != 4 or tuple(x.shape[1:]) != (1, 64, 64):
        raise ValueError(f"Expected N x 1 x 64 x 64 slices, got {tuple(x.shape)}")
    is_uint8 = x.dtype == torch.uint8
    x = x.float()
    if is_uint8:
        x /= 255
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("npy slices must be uint8 or finite floats in [0,1]")
    # Chunk to avoid a second dataset-sized quantile allocation.
    for chunk in x.split(256):
        q = torch.quantile(chunk.flatten(1), .99, dim=1)[:, None, None, None]
        chunk.copy_(torch.minimum(chunk, q) / q.clamp_min(1e-8))
    return x.contiguous()


def get_dataset(name="organamnist64", split="train", root="data", npy_dir=None):
    if split not in SPLITS:
        raise ValueError(f"Use official train, val or test split, got {split}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if name == "organamnist64":
        source = "medmnist/organamnist_64/official-splits"
    elif name == "npy_dir":
        base = Path(npy_dir or root).resolve()
        paths = {s: sorted((base / s).glob("*.npy")) for s in SPLITS}
        if any(not p for p in paths.values()):
            raise ValueError("npy_dir requires explicit nonempty train/, val/, test/ directories; no automatic re-split.")
        source = digest({str(p): file_digest(p) for ps in paths.values() for p in ps})
    else:
        raise ValueError(f"Unknown dataset: {name}")
    cache = root / f"{name}-{CACHE_VERSION}-{digest(source)[:12]}.pt"
    with FileLock(str(cache) + ".lock"):
        if not cache.exists():
            splits = {}
            if name == "organamnist64":
                from medmnist import OrganAMNIST
                for s in SPLITS:
                    ds = OrganAMNIST(split=s, size=64, download=True, root=str(root))
                    splits[s] = preprocess(ds.imgs)
            else:
                for s in SPLITS:
                    splits[s] = preprocess(np.stack([np.load(p, allow_pickle=False).squeeze() for p in paths[s]]))
            tmp = cache.with_suffix(".tmp")
            torch.save({"version": CACHE_VERSION, "source": source, **splits}, tmp)
            os.replace(tmp, cache)
    payload = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    return payload[split]


def from_config(config, split=None):
    d = config["data"]
    return get_dataset(d["name"], split or d["split"], d["root"], d.get("npy_dir"))


def tensor_digest(x):
    # Content fingerprint distinguishes datasets even after copying to another directory.
    import hashlib
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()

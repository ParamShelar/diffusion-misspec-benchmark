"""Locked, fsynced, append-safe CSV with deterministic numerical serialization."""
import csv
import io
import os
from pathlib import Path
from filelock import FileLock

FIELDS = ["config_hash", "git_commit", "source_hash", "prior_id", "data_hash", "smoke",
          "solver", "axis", "level", "seed", "image_id", "sigma_true", "sigma_assumed",
          "mask_disagreement", "mask_density", "psnr", "ssim", "lpips",
          "residual_assumed", "residual_true", "residual_gap", "artifact"]
KEY = ["config_hash", "solver", "axis", "level", "seed", "image_id"]


def canonical(value):
    return format(value, ".12g") if isinstance(value, float) else str(value)


def row_key(row):
    return tuple(canonical(row[k]) for k in KEY)


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock")

    def _read(self):
        if not self.path.exists():
            return []
        # Only a torn trailing write is repaired. Interior corruption fails loudly.
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raw = raw[:raw.rfind(b"\n")+1]
            self.path.write_bytes(raw)
        if not raw:
            return []
        reader = csv.DictReader(io.StringIO(raw.decode()))
        if reader.fieldnames != FIELDS:
            raise ValueError("CSV schema mismatch; use a new output directory")
        rows = list(reader)
        if any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError("CSV contains a malformed interior row")
        return rows

    def rows(self):
        with self.lock:
            return self._read()

    def append(self, row):
        if set(row) != set(FIELDS):
            raise ValueError(f"Wrong CSV fields: {set(row)^set(FIELDS)}")
        with self.lock:
            rows = self._read()
            if row_key(row) in {row_key(r) for r in rows}:
                return False
            new = not self.path.exists() or not self.path.stat().st_size
            with self.path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
                if new:
                    writer.writeheader()
                writer.writerow({k: canonical(row[k]) for k in FIELDS})
                f.flush()
                os.fsync(f.fileno())
        return True

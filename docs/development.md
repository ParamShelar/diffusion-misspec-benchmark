# Development notes

Engineering detail for working on this repository. Results and solver findings are in the
[README](../README.md).

## Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m pytest -q
```

The CUDA build above is tested against an NVIDIA driver supporting CUDA 12.6. Colab supplies its
own CUDA PyTorch; skip that first install there. `requirements-tested.txt` pins the full
environment used for the validation run.

`device_for` estimates activation memory and rejects a configuration whose estimate exceeds 95%
of the smaller of the tier budget and free device memory. The budget is 4 GiB on the `local`
tier and 24 GiB on `colab`. The estimate is conservative and does not account for fragmentation
or memory held by other processes.

## Smoke path

`configs/local.yaml` runs on a 4 GB card at batch 1 with `--smoke`. Every script accepts the flag.

```bash
python scripts/prepare.py      --config configs/local.yaml --smoke
python scripts/check_prior.py  --config configs/local.yaml --smoke
python scripts/run_sweep.py    --config configs/local.yaml --smoke
python scripts/make_figures.py --config configs/local.yaml --smoke
```

The smoke sweep covers three axes at three levels, all five solvers, one image and one seed, at
20 sampling steps: 45 rows. Output lands in `results/local/smoke/` as `metrics.csv`, an immutable
config record, reconstruction tensors, the prior report and sample PNGs, and six figures under
`figures/<config hash>/`. Every smoke figure is watermarked **SMOKE ONLY**. Twenty steps and one
seed give no error bars and support no conclusion about a solver.

The three benchmark scripts run under a 175-second budget in smoke mode (`src/runtime.py`,
`Deadline`) and raise `TimeoutError` on expiry. Downloads and the one-time preprocessing cache
sit outside that budget; the first automatic dataset download alone took about 231 seconds.

The `local` tier is smoke-only by construction: `load_config` rejects a non-smoke local run, and
`device_for` refuses to train off the `colab` tier. Frozen copies of a smoke run are committed
under `examples/smoke/` so the layout is inspectable without running anything.

### Warm-cache smoke timings

Measured on a GTX 1650 (4 GB), Python 3.12.3, PyTorch 2.7.1+cu126, driver 580.173.02.

| Command | Seconds |
| --- | ---: |
| `check_prior` | 13.105 |
| `run_sweep` | 73.532 |
| `make_figures` | 5.565 |
| resume | 6.592 |

An independent rerun completed in 67.369 seconds and produced 45 byte-identical CSV rows. The
resume check added no rows and left the CSV bytes unchanged.

Two-step memory profiling measured peak PyTorch allocations of roughly 385 MiB for DPS, 220 MiB
for DiffPIR and 385 MiB for PiGDM. These are solver allocations, not total device usage, and say
nothing about training memory.

## CIFAR fallback prior

`prior.kind: cifar` loads `google/ddpm-cifar10-32` at its native 32x32 and applies it
sequentially to four tiles of the 64x64 image, replicating grayscale to RGB and averaging the
predicted RGB epsilon. The Fourier problem stays at 64x64. This is an out-of-domain,
tile-independent score approximation: it can show tile seams and it is not a medical-image prior.
Network weights are frozen without disabling the input gradients DPS and PiGDM need.

It exists so the pipeline can be exercised end to end without a trained checkpoint. It cannot
pass the research quality gate, and the gate does not fabricate a pass to let a sweep proceed.

## Quality gate

`check_prior.py` generates 2000 samples and a 64-image grid, denoises a held-out image starting
at training timestep 250, and computes 2048-dimensional Inception FID against 2000 validation
images. A disjoint second set of 2000 validation images gives the held-out-vs-held-out FID floor.
Thresholds: PSNR >= 15 dB, FID <= 200, FID minus floor <= 150. They are engineering sanity
bounds, not clinical cutoffs.

`SMOKE_PASS` means the reduced path completed with finite outputs; it cannot authorize a research
sweep. A research sweep requires `PASS` for the same weights, validation data, source files and
sampling-step count. Any change invalidates the gate. Trained checkpoints must carry EMA weights,
and all inference from a trained prior uses EMA only.

## Determinism and the CSV ledger

Sweep rows are paired: the same image, mask seed, complex noise realization and sampler
initialization are reused across every solver and mismatch level, so differences between rows are
attributable to the solver and the mismatch.

`src/ledger.py` writes `metrics.csv` under a `filelock` file lock, flushed and `fsync`ed. On
resume, a torn trailing line is discarded and rewritten; interior corruption raises rather than
being repaired. Image artifacts are written before their CSV rows, so a row always has its tensor.
Completed rows are skipped on re-run. No elapsed times or timestamps enter a metric row.

Each row carries config, source, data and actual prior-weight fingerprints plus the Git commit,
so runs are distinguishable after the fact. `make_figures.py` refuses an incomplete sweep and
refuses to pool rows from different config hashes; pass `--config-hash` to select one when a
single CSV holds several runs.

Bitwise reproducibility holds on the same hardware and software stack. It does not hold across
CUDA versions or GPU models.

## Training, checkpoints and resume

Training runs on a Colab L4 through `notebooks/train_colab.ipynb`. The model has 22,809,921
parameters: output widths 64/128/128/128, two residual blocks per encoder and decoder level, an
additional two-block bottleneck, and attention at 16x16 only. Each residual block expands
internally by four. Gradient checkpointing is on so batch 128 fits.

1000 cosine-schedule timesteps, epsilon MSE, bf16 autocast, channels-last, TF32, AdamW at 2e-4,
500-step linear warmup, gradient clipping at 1.0, EMA decay 0.9995. `torch.compile` is opt-in via
`--compile`. The preprocessed training split is re-quantized once to uint8 and held as a single
GPU tensor; batches are gathered with GPU indices, with no DataLoader and no host-to-device copies
in the loop. Iterations per second and projected step counts print at updates 10, 50 and 100.

`TRAIN_ARGS="--max-minutes 90"` bounds training, monitoring, checkpointing and the final gate
together, with 45 minutes reserved by default for the gate. Setup-cell downloads are outside it.
Sampling checks the deadline each denoising step; training checks between optimizer updates, so a
running CUDA kernel or a final Drive flush can overrun by a few seconds. Use
`--max-minutes 2 --smoke` in a separate Drive experiment for a two-update notebook check.

Checkpoints are written every 1000 steps or ten minutes, whichever comes first, and on clean stop
or interrupt. The last three regular checkpoints and `best.pt` are kept; `best` is ranked by EMA
MSE on fixed held-out noise and timesteps. A 16-image EMA grid is written every 2000 steps when it
fits before the deadline. Partial `.tmp` checkpoints are ignored. On restart the latest readable
checkpoint restores model, EMA, optimizer, step count and the Python, NumPy, PyTorch and CUDA RNG
states; if every checkpoint is unreadable, training raises rather than silently restarting from
scratch. All checkpoints, JSONL logs, grids, gate results and the generated sweep config live
under the mounted Drive experiment.

A gate `PASS` writes `trained_sweep.yaml`, which the final notebook cell feeds to the same
sweep and figure scripts. `configs/colab_trained.yaml` is the checked-in equivalent.

## Data

`get_dataset(name, split, root=..., npy_dir=...)` returns float32 `(N,1,64,64)` in `[0,1]`. The
`organamnist64` backend uses MedMNIST's own train/val/test partitions without re-splitting. Each
image is scaled from uint8, clipped at its own 99th percentile and divided by it; zero images stay
zero. All three processed splits share one `.pt` cache per source fingerprint. Training may
re-quantize that cache to uint8 for GPU residence, costing at most half a quantization bin.

For `data.name: npy_dir`, supply explicit `train/`, `val/` and `test/` folders under
`data.npy_dir`, each holding 64x64 `.npy` arrays of uint8 or finite floats in `[0,1]`. There is no
implicit resizing, complex-to-magnitude conversion or split generation. Changing source content
invalidates the cache.

Radial masks are rasterized spokes on the Cartesian grid with a square ACS region. They are not
non-Cartesian trajectories and there is no NUFFT; density is approximate to within a spoke. The
mask mismatch axis is line-based and therefore Cartesian-only; radial supports the noise and
sensitivity axes.

# Validation record

Validated on 2026-09-01 using the user’s NVIDIA GeForce GTX 1650 (4 GB) outside the sandbox.

- Python 3.12.3, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126; NVIDIA driver 580.173.02.
- 42 automated tests passed. Notebook schema and every code cell compiled successfully.
- Tests cover complex and real adjoints across all masks/sensitivities; Cartesian density/ACS; radial density; real Fourier proximal versus dense least squares; PiGDM covariance versus a dense real inverse; VJP finite differences; DPS network Jacobian; deterministic sampling/CSV serialization; dataset caching; EMA-only loading; architecture and checkpoint/RNG recovery.
- The borrowed-prior pipeline produced figures and passed its initial tests before training code was added.
- No 90-minute L4 training job or full 2,000-sample scientific quality gate was run. The notebook is implemented and locally checked, not empirically certified on L4.

## Final warm-cache smoke timings

| Command | Seconds |
| --- | ---: |
| check_prior | 13.105 |
| run_sweep | 73.532 |
| make_figures | 5.565 |
| resume | 6.592 |

All three benchmark scripts completed below three minutes. The final sweep used 20 sampling steps, batch 1, one image, one seed, all five methods and all nine axis/level combinations (45 rows). The download/preprocessing setup is excluded; the first automatic dataset download alone took about 231 seconds.

An independent rerun completed in 67.369 seconds and produced **45 byte-identical CSV rows**. The resume check added no rows and preserved the CSV bytes. Six curve/qualitative figures were generated; representative curve and grid layouts were visually inspected.

Separate two-step memory profiling measured peak PyTorch allocations of approximately 385 MiB for DPS, 220 MiB for DiffPIR and 385 MiB for PiGDM. These are solver allocations, not total display/driver memory, and are not a claim about training memory.

## Quality result and limitations

Gate status: **SMOKE_PASS**, scientific pass: **false**. The 4-sample smoke FID is 454.831, held-out floor 449.322, and noise-then-denoise PSNR 11.557 dB. Such a small FID estimate is unstable and cannot validate a prior. The full quality thresholds remain enforced for research sweeps.

On the single smoke image, with matched mask/noise/sensitivity at R=4, the recorded PSNRs were:

| Method | PSNR (dB) |
| --- | ---: |
| zero_filled | 22.659 |
| tv | 23.138 |
| dps | 2.336 |
| diffpir | 7.249 |
| pigdm | 7.854 |

The CIFAR tile adapter remains substantially worse than the classical baselines in this diagnostic. These results establish a working, reproducible benchmark path, not competitive CT or MRI reconstruction quality.

The README documents the real-domain Fourier corrections, the requested DPS variant’s difference from the paper, and why exact recovery for arbitrary priors requires an explicit final consistency projection. That projection is off in the benchmark and is tested separately.

## Provenance

- Experiment hash: `e75ec94982975b1bd629f4e6e9fe889b9ba3832874da19662175dc2e30465e0d`
- Source hash: `795a99180e3da4d679d7fbf79b5646a6d30b5205e2278b687902ff70598fefb4`
- Prior fingerprint: `b5b3e9045b7173ecf10cee4f23e019c44afbf16f75553c483c60fbfb264f4d3f`
- Git commit: `uncommitted`. This records the smoke run before the initial Git commit. Source and actual weight hashes preserve that tested state; publication does not rewrite the historical results.
- `requirements-tested.txt` records the complete validation environment.
- Frozen outputs are in `examples/smoke/`. Reproducible live outputs are in `results/local/smoke/` on the supplied machine.

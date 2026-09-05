# Validation record

Test coverage and run provenance. Engineering detail is in
[docs/development.md](docs/development.md); results are in the [README](README.md).

## Test coverage

45 automated tests pass at commit `3edb09d`. The notebook schema parses and every code
cell compiles.

- **Operators.** Complex and real adjoint inner-product conventions across all mask kinds and
  nonconstant sensitivities; Cartesian line count and ACS preservation at R = 1, 2, 4, 8; radial
  density and ACS; mask-axis feasibility, nesting and pairing.
- **Solver algebra.** DiffPIR's real Fourier proximal against a dense least-squares solve on odd
  and even grids with asymmetric masks; the complex closed form on conjugate-symmetric masks;
  PiGDM's `covariance_backproject` against a dense real inverse; the PiGDM VJP against finite
  differences through the network; DPS network-Jacobian retention; TV gradient adjoint.
- **Samplers.** Guided solvers bound their high-noise clean estimates; solvers are deterministic
  and consume their measurements; sampling honours an expired time budget.
- **Pipeline.** Deterministic metric serialization and CSV round-tripping; dataset caching and
  split handling.
- **Training.** Architecture and parameter count; checkpoint retention policy; exact optimizer,
  EMA and RNG restoration; EMA-only loading of trained priors.

### Known coverage gap

`tests/test_math.py::test_all_solvers_full_noiseless_with_explicit_projection` passes
`final_consistency=True`, so the final zero-noise projection recovers the image on a fully
sampled noiseless operator regardless of what the sampler produced. The test cannot detect broken
guidance, and it does not: all three solvers pass it while DPS, DiffPIR and PiGDM fail as
described in the README. No test currently exercises guidance quality end to end.

`final_consistency` is disabled in every benchmark config and is rejected for nonzero assumed
sigma.

## Trained run

- Config hash: `d4edbcc97ec00f079508e13004cf8c0e01806e2a1c89e0eb93c0e8df45158196`
- Source hash: `cff553233941e75169564d196f318bbdd728713e0809f049f5ada55d92b57e1e`
- Prior fingerprint: `7e25783c0a7c26e9b9f205121679098986989cb4fa109a54923972afe75d60ce`
- Data hash: `ac6168d8c51a9220ff3b88b5de5c6038c357dcd1c8a36c2a1f201e2293e484e8`
- Git commit: `3edb09d41378dfae155626fe72b75a2c4bde6a58`
- Checkpoint: `step-000012033.pt` (EMA)
- Gate: `PASS`, scientific pass true. FID 50.283, floor 16.411, excess 33.872, PSNR 22.045 dB,
  2000 samples, 100 steps.
- Environment: torch 2.11.0+cu128, torchvision 0.26.0+cu128, diffusers 0.40.0, numpy 2.1.3,
  scipy 1.16.3, scikit-image 0.25.2, lpips 0.1.4, torch-fidelity 0.3.0.
- Artifacts: `results/trained/`. 1800 rows.

## CIFAR smoke run

Validated 2026-09-01. Python 3.12.3, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126.

- Experiment hash: `e75ec94982975b1bd629f4e6e9fe889b9ba3832874da19662175dc2e30465e0d`
- Source hash: `795a99180e3da4d679d7fbf79b5646a6d30b5205e2278b687902ff70598fefb4`
- Prior fingerprint: `b5b3e9045b7173ecf10cee4f23e019c44afbf16f75553c483c60fbfb264f4d3f`
- Git commit: `uncommitted`. This run predates the first commit; the source and weight hashes
  preserve the tested state.
- Gate: `SMOKE_PASS`, scientific pass false. 4-sample FID 454.831, floor 449.322,
  noise-then-denoise PSNR 11.557 dB. A 4-sample FID estimate cannot validate a prior.
- Artifacts: `examples/smoke/` (frozen), `results/local/smoke/` (regenerated locally).
- `requirements-tested.txt` records the environment.

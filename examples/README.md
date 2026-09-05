# CIFAR-prior diagnostic run

Frozen outputs from the borrowed-prior smoke path: `google/ddpm-cifar10-32` applied as a 32x32
tile adapter, 20 DDIM steps, one official test image, one seed, five solvers, three axes at three
levels. Figures are watermarked **SMOKE ONLY**.

This run exists to show the pipeline executing end to end. The prior is out of domain and the
sample size supports no conclusion about any solver. The benchmark results are the trained run in
[`../results/trained/`](../results/trained/); see the [README](../README.md).

Reproduce with the smoke commands in [`../docs/development.md`](../docs/development.md); new
outputs are written to `results/local/smoke/`.

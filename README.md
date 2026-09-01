# Forward-model misspecification benchmark

Repository: [ParamShelar/diffusion-misspec-benchmark](https://github.com/ParamShelar/diffusion-misspec-benchmark) (private).

Compare DPS, DiffPIR and PiGDM with zero-filled Fourier reconstruction and isotropic TV under mask, noise and sensitivity mismatch. This is a research implementation with an explicitly labeled local diagnostic path, not an MRI reconstruction product.

## Run the borrowed-prior pipeline first

Python 3.10+ is required. For the GTX 1650 (4 GB), use batch 1 and `--smoke`. The CUDA install below is tested with an NVIDIA driver compatible with CUDA 12.6. Colab already supplies CUDA PyTorch; skip that install there.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/prepare.py --config configs/local.yaml --smoke
python -m pytest -q
python scripts/check_prior.py --config configs/local.yaml --smoke
python scripts/run_sweep.py --config configs/local.yaml --smoke
python scripts/make_figures.py --config configs/local.yaml --smoke
```

`prepare.py` automatically downloads the official data archive, pretrained diffusion checkpoint, AlexNet LPIPS weights, and TensorFlow-compatible Inception weights. No manual data downloads or accounts are required. The OrganAMNIST64 archive currently fetched by MedMNIST is approximately 200 MB. Downloads and the one-time preprocessing cache are setup work and **cannot have a network-independent three-minute guarantee**. Every script accepts `--smoke`; the three benchmark scripts have a 175-second deadline and fail explicitly on timeout. A timeout is never called a pass. Warm-cache runtimes are recorded in `VALIDATION.md` when measured.

The smoke pipeline evaluates all three axes at three levels, all five methods, one image and one seed, with 20 sampling steps. Its output is `results/local/smoke/`: `metrics.csv`, immutable configuration records, reconstruction tensors, the prior report and sample PNGs, and six figures under `figures/<config hash>/`. Every figure visibly says **SMOKE ONLY**. Twenty sampling steps and one seed do not support scientific conclusions or useful error bars.

The CIFAR fallback remains available through `prior.kind: cifar`. It uses `google/ddpm-cifar10-32` at its native 32×32 size, sequentially on four tiles of a 64×64 image, replicating grayscale to RGB and averaging predicted RGB epsilon. This is an out-of-domain, tile-independent score approximation; it can show seams and is not a trained medical-image prior. The Fourier problem itself stays at 64×64. Network weights are frozen without disabling the input gradients needed by DPS/PiGDM.

## Research sweeps and quality gate

On an L4 Colab runtime with Drive mounted:

```bash
python scripts/prepare.py --config configs/colab.yaml
python scripts/check_prior.py --config configs/colab.yaml
python scripts/run_sweep.py --config configs/colab.yaml
python scripts/make_figures.py --config configs/colab.yaml
```

The full quality gate generates 2,000 samples and a 64-image sample grid, denoises a held-out image starting at training timestep 250, and computes 2,048-dimensional Inception FID against 2,000 validation images. A disjoint second set of 2,000 validation images provides the held-out-vs-held-out FID floor. Fixed sanity thresholds are PSNR ≥15 dB, FID ≤200, and FID minus floor ≤150. They are engineering thresholds, not clinically validated cutoffs. A borrowed CIFAR prior can fail this gate; the implementation never fabricates a pass to permit a sweep.

`SMOKE_PASS` means only that the reduced numerical path completed with finite outputs. It cannot authorize a research sweep. Research requires `PASS` for the same weights, validation-data contents, source files, and sampling steps. Any change invalidates the gate. Trained checkpoints must contain EMA weights. All inference from trained priors uses EMA only.

Sweep rows are paired: the same image, mask seed, complex noise realization and sampler initialization are reused across methods and mismatch levels. Re-running the sweep skips completed rows. CSV writes are locked, flushed and synced; a torn final line is discarded on resume, while interior corruption fails loudly. Image artifacts are saved before their CSV rows. Config, source, data and actual prior weight fingerprints distinguish runs; the Git commit is also recorded. No elapsed times or timestamps enter deterministic metric rows. Bitwise reproducibility applies on the same hardware/software stack, not across CUDA versions or GPU models.

Figures show means across images, then mean ±1 sample standard deviation across seeds. Both classical baselines appear in every curve panel and qualitative grid. With one seed, the displayed error bar is zero and is not an uncertainty estimate. Figures refuse incomplete sweeps and refuse to pool different configuration hashes; use `--config-hash` to select one if multiple runs share a CSV.

## Data and operators

`get_dataset(name, split, root=..., npy_dir=...)` returns float32 `(N,1,64,64)` in `[0,1]`. The `organamnist64` backend uses MedMNIST's provided train/val/test partitions without re-splitting. Each image is scaled from uint8, clipped at its own 99th percentile and divided by that percentile; zero images remain zero. All three processed splits share one `.pt` cache per source fingerprint. Training may re-quantize this processed cache to uint8 for GPU residence, introducing at most half a quantization bin of rounding error.

For `data.name: npy_dir`, provide explicit `train/`, `val/`, and `test/` folders under `data.npy_dir`, each containing `.npy` arrays with shape 64×64. Acceptable values are uint8 or finite floats in `[0,1]`; there is no implicit resizing, complex-to-magnitude conversion, or split generation. Source-content changes invalidate the cache.

The noiseless operator is `A(x)=M F(S*x)`, with centered orthonormal FFT and complex sensitivity map. Images are **real**, single channel. Measurements are stored as full complex grids with zeros off the true mask; each measured real and imaginary component receives independent `N(0,sigma²)` noise. Thus `E|noise|²=2 sigma²`. The complex adjoint is `conj(S)*F^-1(M*y)`; the real reconstruction-space adjoint is its real part. Tests cover both inner-product conventions, including nonconstant S.

Cartesian random and equispaced masks have exactly `round(64/R)` lines, including ACS; impossible ACS/R combinations are rejected. Equispaced means equally spaced ranks among non-ACS candidates, preserving the exact line count. Radial masks are rasterized spokes on the Cartesian grid with a square ACS region: they are **not** non-Cartesian MRI trajectories or a NUFFT, and density is approximate to within a spoke.

Mismatch axes:

- **Mask:** true mask B is fixed; assumed A exchanges equal numbers of selected and unselected non-ACS lines. Level is the exact fraction of all lines that disagree. Levels must be feasible multiples of `2/64`; the default R=4/ACS=8 maximum is 0.25. Changed sets are nested across levels. This line-based axis applies to Cartesian masks only; radial can use noise and sensitivity axes.
- **Noise:** truth sigma is fixed and assumed sigma is `sigma*exp(level)`. Negative levels underestimate noise; positive levels overestimate it. Zero is perfect specification.
- **Sensitivity:** truth has a fixed smooth polynomial phase and log-magnitude field multiplied by level; assumed S is one. Level zero is exactly one. Field perturbations increase with level, although reconstruction error need not increase monotonically.

The solvers see only the assumed operator. Metrics report `||A_assumed*x-y||/||y||`, `||A_true*x-y||/||y||`, and their signed difference `true-assumed`. Both residuals use the same full observed grid, including measurements the assumed mask cannot explain. PSNR and SSIM use the raw reconstruction with data range one; LPIPS and display PNGs clip to `[0,1]`. The raw reconstruction remains available in its tensor artifact.

## Guidance math and deliberate qualifications

The diffusion latent uses `u=2*x-1`. Before guidance, measurements are transformed exactly to `2*y-A(1)` and sigma to `2*sigma`; results are converted back once. Every solver uses the same timestep selection and Tweedie estimate.

**DPS:** the requested default subtracts `zeta/||r|| * grad ||r||` after a DDIM step, with autograd through epsilon prediction. This differs from the squared-residual gradient in [Chung et al., Algorithm 1](https://arxiv.org/html/2209.14687v2). Set `sampling.dps_loss: squared` for that variant. The requested normalized update does not explicitly use assumed sigma, so its noise-mismatch curve is flat; inventing sigma dependence would change the requested algorithm. Both variants are approximate samplers.

**DiffPIR:** implements the proximal and noise-blending updates of [Zhu et al., Eq. (12b) and Algorithm 1](https://arxiv.org/pdf/2305.08995), with `rho=lambda*sigma²/((1-abar)/abar)` and `eta` mapped to the paper's noise-blending zeta. There is no network backward pass or conjugate-gradient loop. For real images and asymmetric masks, the brief's complex minimizer followed by `real()` is generally not the real constrained minimizer. Define `mbar(k)=(M(k)+M(-k))/2` and `b=F(real(A^H y))`; then the exact real solution is `F^-1((b+rho*F(x0))/(mbar+rho)).real`. It reduces to the simpler formula for conjugate-symmetric masks. Dense real least-squares tests cover odd/even grids and asymmetric masks. At rho=0 the prior estimate is preserved in unobserved Fourier directions.

**PiGDM:** uses the VJP guidance in [Song et al., Eqs. (7–8)](https://openreview.net/forum?id=9_gsMA8MRKQ), with the brief's `r²=(1-abar)/abar` approximation. For a real reconstruction variable, push through the covariance to obtain `(r² A* A+sigma² I)^-1 A* r`, then divide in Fourier space by `r²*mbar+sigma²`. This exactly handles conjugate redundancy for assumed S=1; blindly using complex `AA^H=I` would not. A single `torch.autograd.grad` computes each VJP. Conditional epsilon is `eps-sqrt(1-abar)*guidance`, followed by the shared DDIM step. Dense covariance and finite-difference tests verify this operation.

DiffPIR and PiGDM explicitly reject nonconstant **assumed** sensitivities because these closed forms require S=1. This does not restrict the specified sensitivity experiment, where only the true S varies. DPS and TV can use general S.

Finite-step approximate posterior samplers cannot universally guarantee high PSNR on fully sampled noiseless data independent of prior quality. `sampling.final_consistency: true` exposes an optional final zero-noise projection; it is rejected for nonzero assumed sigma and is disabled in all benchmark configs. The full-sampling exact-recovery tests explicitly test this wrapper, separately from raw guidance/Jacobian tests. They are not evidence that an untrained or unrelated prior is good.

## Assumptions and limitations

- **OrganAMNIST is CT, not MRI.** Fourier k-space is simulated from magnitude images as a compressed-sensing benchmark; these are not acquired MR measurements.
- The reconstruction is real and single channel; sensitivity/phase is introduced only in the forward model. There is one synthetic coil, no coil combination, no motion, no non-Cartesian acquisition physics and no learned sensitivity estimator.
- CIFAR RGB-to-grayscale tiling is a diagnostic fallback with a substantial domain mismatch. A trained medical prior is still subject to the quality gate.
- LPIPS and Inception FID use natural-image networks. Neither measures clinical validity. FID at 2,000 samples is biased/noisy; the held-out floor contextualizes it without removing that limitation.
- VRAM admission uses conservative estimates and available-device memory. It rejects over-budget configurations early but cannot mathematically guarantee against fragmentation or memory consumed by other applications. The local tier never trains and never silently reduces requested research settings.
- Cold downloads, scientific sweeps and 2,000-sample quality gates can take substantially longer than a smoke run. Full training and sweeps belong on Colab L4, not the 4 GB GTX 1650.

## Colab training and recovery

Open `notebooks/train_colab.ipynb` in Colab after inspecting the CIFAR smoke figures. Cell 1 already points to this repository. Because it is private, a fresh Colab session asks for a GitHub token with read access when needed; the token is entered invisibly and is not stored in the remote URL. Cell 1 clones it and installs dependencies, cell 2 mounts Drive, and the next cell prints and asserts an L4 GPU. T4 gets an explicit bf16 warning and stops.

The model has **22,809,921 parameters**: output widths 64/128/128/128, two residual blocks per encoder and decoder level, an additional two-block bottleneck, and attention only at 16×16. Each residual block expands internally by four; this explicit design choice satisfies the requested parameter budget without changing the level widths. Gradient checkpointing is enabled for training to fit batch 128. Training uses 1,000 cosine-schedule timesteps, epsilon MSE, bf16 autocast, channels-last tensors, TF32, AdamW at 2e-4, 500-step linear warmup, gradient clipping at 1.0 and EMA decay 0.9995. Compilation is disabled unless `--compile` is supplied.

The complete preprocessed training split is re-quantized once and loaded as a single GPU uint8 tensor. Batches are selected with GPU indices, with no DataLoader or host-to-device copies in the training loop. The actual official dataset counts are printed; the implementation does not assume the approximate count in the brief. Measured iterations/second and projected step counts are printed at updates 10, 50 and 100 of each session.

Set `TRAIN_ARGS="--max-minutes 90"` in the notebook. The timer covers the training function, monitoring, checkpoints and final gate, with a default 45-minute tail reserved for the expensive gate. This means up to about 45 minutes of optimizer updates in a 90-minute run. Dataset/model downloads in prior setup cells are outside that compute budget. Sampling checks the deadline at each denoising step; training checks it between optimizer updates. A running CUDA kernel or final Drive flush may overrun a boundary by a few seconds. If the complete 2,000-sample gate cannot finish, it writes FAIL and blocks the sweep. The code cannot guarantee model quality or a successful gate within a fixed training time.

Checkpoints save every 1,000 steps or ten minutes, whichever comes first, and on clean stop/interruption. The last three regular checkpoints and `best.pt` are retained; best is ranked by MSE from the EMA model on fixed held-out noise/timesteps. A 16-image EMA grid is written every 2,000 steps when it fits before the deadline. All checkpoints, JSONL logs, grids, quality results and generated sweep configuration are stored under the mounted Drive experiment. Partial `.tmp` checkpoints are ignored. On restart, the latest readable checkpoint restores model, EMA, optimizer, step count and Python/NumPy/PyTorch/CUDA RNG states. If all existing checkpoints are corrupt, training refuses to silently restart.

Use `--max-minutes 2 --smoke` for a two-update **L4 notebook** validation in a separate Drive experiment. The GTX 1650 local tier does not train. A successful full gate writes `trained_sweep.yaml`; the final notebook cell uses that configuration to run the same sweep/figure pipeline against EMA weights. Local tests cover architecture, checkpoint retention, exact optimizer/EMA/RNG restoration and notebook syntax; the full L4 training job must still be executed on Colab.

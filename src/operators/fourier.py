"""Centered orthonormal Fourier measurements of real single-channel images.

Complex extension: A^H y = conj(S) F^H(M y).
Real reconstruction space: A^* y = real(A^H y), using Re <.,.>.
sigma is the standard deviation of EACH real/imaginary noise component.
"""
import math
import torch


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def reflect_frequency(x):
    """Value at -k in centered FFT coordinates (even AND odd sizes)."""
    z = torch.fft.ifftshift(x, dim=(-2, -1))
    for dim in (-2, -1):
        ix = (-torch.arange(z.shape[dim], device=z.device)) % z.shape[dim]
        z = z.index_select(dim, ix)
    return torch.fft.fftshift(z, dim=(-2, -1))


class FourierOperator:
    def __init__(self, mask, sensitivity=None, sigma=0.0):
        if mask.ndim != 2 or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be a 2-D binary tensor")
        if sigma < 0:
            raise ValueError("sigma must be nonnegative")
        self.mask = mask
        self.sensitivity = torch.ones_like(mask, dtype=torch.complex64) if sensitivity is None else sensitivity
        if self.sensitivity.shape != mask.shape or not torch.isfinite(self.sensitivity).all():
            raise ValueError("Sensitivity must be finite and have the mask shape")
        self.sigma = float(sigma)

    def forward(self, x):
        return self.mask * fft2c(self.sensitivity * x)

    __call__ = forward

    def adjoint(self, y):
        return self.sensitivity.conj() * ifft2c(self.mask * y)

    def real_adjoint(self, y):
        return self.adjoint(y).real

    def normal(self, x):
        return self.adjoint(self(x))

    def real_normal(self, x):
        return self.real_adjoint(self(x))

    def measure(self, x, generator):
        shape = self(x).shape
        re = torch.randn(shape, device=x.device, dtype=x.dtype, generator=generator)
        im = torch.randn(shape, device=x.device, dtype=x.dtype, generator=generator)
        return self(x) + self.mask * self.sigma * torch.complex(re, im)

    def require_unit_sensitivity(self):
        if not torch.allclose(self.sensitivity, torch.ones_like(self.sensitivity)):
            raise ValueError("This closed form requires assumed S=1; use DPS/TV for nonconstant assumed S.")

    @property
    def real_spectrum(self):
        self.require_unit_sensitivity()
        return (self.mask + reflect_frequency(self.mask)) / 2

    def prox(self, x0, y, rho):
        """Exact real-valued minimizer of ||Az-y||² + rho ||z-x0||².

        Let mbar=(M(k)+M(-k))/2 and b=F(Re A^H y). Then
        Fz=(b+rho Fx0)/(mbar+rho). Equivalent to the brief's complex
        formula followed by Re only when mask is conjugate symmetric.
        rho=0 preserves x0 in the nullspace (minimum-change LS solution).
        """
        mbar = self.real_spectrum
        rho = torch.as_tensor(rho, device=x0.device, dtype=x0.dtype)
        if torch.any(rho < 0):
            raise ValueError("rho must be nonnegative")
        X = fft2c(x0)
        denom = mbar + rho
        numerator = fft2c(self.real_adjoint(y)) + rho * X
        return ifft2c(torch.where(denom > 0, numerator / denom.clamp_min(1e-20), X)).real

    def covariance_backproject(self, residual, r2):
        """A* (r² A A* + sigma² I)^-1 residual, without matrices.

        Real-space push-through identity: (r² A* A + sigma² I)^-1 A* r.
        A*A has Fourier multiplier mbar. This also accounts for conjugate
        redundancy, which the complex-extension AA^H=M shortcut neglects.
        """
        denom = r2 * self.real_spectrum + self.sigma**2
        B = fft2c(self.real_adjoint(residual))
        return ifft2c(torch.where(denom > 0, B / denom.clamp_min(1e-20), torch.zeros_like(B))).real


def make_mask(size, kind="cartesian_random", acceleration=4, acs_lines=8, seed=0):
    if acceleration < 1 or not 0 <= acs_lines <= size:
        raise ValueError("Require R>=1 and 0<=ACS<=size")
    count = round(size / acceleration)
    if count < acs_lines:
        raise ValueError(f"R={acceleration} requests {count} lines, fewer than ACS={acs_lines}")
    gen = torch.Generator().manual_seed(seed)
    center = torch.arange((size-acs_lines)//2, (size-acs_lines)//2+acs_lines)
    if kind in ("cartesian_random", "cartesian_equispaced"):
        line = torch.zeros(size)
        line[center] = 1
        candidates = torch.where(line == 0)[0]
        n = count - acs_lines
        if n:
            if kind == "cartesian_random":
                selected = candidates[torch.randperm(len(candidates), generator=gen)[:n]]
            else:
                # Equally spaced ranks outside ACS, exact requested count.
                selected = candidates[torch.floor((torch.arange(n)+.5)*len(candidates)/n).long()]
            line[selected] = 1
        return line[:, None].expand(size, size).clone()
    if kind == "radial":
        # Cartesian-grid rasterized radial spokes, not a non-Cartesian NUFFT.
        mask = torch.zeros(size, size)
        mask[center[:, None], center] = 1
        target = round(size * size / acceleration)
        if mask.sum() > target:
            raise ValueError("Radial ACS square exceeds target density")
        if acceleration == 1:
            return torch.ones_like(mask)
        coord = torch.linspace(-size, size, size*8)
        golden = math.pi * (3 - math.sqrt(5))
        for spoke in range(size*8):
            angle = spoke * golden
            row = (size//2 + coord * math.sin(angle)).round().long()
            col = (size//2 + coord * math.cos(angle)).round().long()
            good = (row >= 0) & (row < size) & (col >= 0) & (col < size)
            candidate = mask.clone()
            candidate[row[good], col[good]] = 1
            if abs(candidate.sum().item()-target) >= abs(mask.sum().item()-target) and mask.sum() > acs_lines**2:
                break
            mask = candidate
            if mask.sum() >= target:
                break
        return mask
    raise ValueError(f"Unknown mask type {kind}")


def mismatch_mask(truth, level, acs_lines, seed):
    if not torch.equal(truth, truth[:, :1].expand_as(truth)):
        raise ValueError("Line-disagreement mask axis requires a Cartesian mask; radial supports noise/sensitivity axes.")
    n = len(truth)
    swaps_float = float(level) * n / 2
    swaps = round(swaps_float)
    if level < 0 or abs(swaps-swaps_float) > 1e-7:
        raise ValueError(f"Mask level must be a nonnegative multiple of {2/n:g}")
    eligible = truth[:, 0].clone()
    eligible[(n-acs_lines)//2:(n-acs_lines)//2+acs_lines] = 0
    on, off = torch.where(eligible == 1)[0], torch.where(truth[:, 0] == 0)[0]
    if swaps > min(len(on), len(off)):
        raise ValueError("Requested disagreement is infeasible at this R while preserving ACS/density")
    g = torch.Generator().manual_seed(seed)
    out = truth.clone()
    out[on[torch.randperm(len(on), generator=g)[:swaps]]] = 0
    out[off[torch.randperm(len(off), generator=g)[:swaps]]] = 1
    return out


def sensitivity_map(size, level, device="cpu"):
    if level < 0:
        raise ValueError("Sensitivity level must be nonnegative")
    u = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    phase = level * (xx + .5*yy + .3*xx*yy)
    magnitude = torch.exp(level * .3 * (xx.square()+yy.square()-2/3))
    return magnitude * torch.exp(1j*phase)

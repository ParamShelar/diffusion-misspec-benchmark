"""Fixed-range image metrics and genuine pretrained perceptual features."""
import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(reference, estimate):
    mse = (reference.double()-estimate.double()).square().mean().item()
    return -10*np.log10(max(mse, 1e-16))


class Metrics:
    def __init__(self, device):
        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, reference, estimate, y, assumed, truth):
        if not torch.isfinite(estimate).all():
            raise FloatingPointError("Cannot report metrics for nonfinite reconstruction")
        ref = reference[0, 0].cpu().numpy()
        est = estimate[0, 0].cpu().numpy()
        den = y.norm().clamp_min(1e-12)
        ra = ((assumed(estimate)-y).norm()/den).item()
        rt = ((truth(estimate)-y).norm()/den).item()
        return dict(psnr=float(psnr(reference, estimate)),
                    ssim=float(structural_similarity(ref, est, data_range=1.)),
                    lpips=self.lpips((2*reference-1).expand(-1,3,-1,-1),
                                     (2*estimate.clamp(0,1)-1).expand(-1,3,-1,-1)).item(),
                    residual_assumed=ra, residual_true=rt, residual_gap=rt-ra)


class InceptionFeatures:
    def __init__(self, device):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images):
        rgb = (images.to(self.device).clamp(0, 1)*255).round().to(torch.uint8).expand(-1,3,-1,-1)
        return self.model(rgb)[0].cpu().double().numpy()


def fid(features_a, features_b):
    """Exact empirical FID via low-rank covariance factors, including n<2048.

    tr sqrt(Ca Cb) = nuclear_norm(A B^T), with centered/scaled features A,B.
    No fake reduced-dimensional 'FID' is substituted in smoke mode.
    """
    from scipy.linalg import svdvals
    a, b = np.asarray(features_a, dtype=np.float64), np.asarray(features_b, dtype=np.float64)
    if min(len(a), len(b)) < 2:
        raise ValueError("FID needs at least two samples per set")
    mean = ((a.mean(0)-b.mean(0))**2).sum()
    a = (a-a.mean(0))/np.sqrt(len(a)-1)
    b = (b-b.mean(0))/np.sqrt(len(b)-1)
    result = mean + (a*a).sum() + (b*b).sum() - 2*svdvals(a@b.T).sum()
    return float(max(result, 0.))

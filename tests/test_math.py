import pytest
import torch
from src.operators.fourier import FourierOperator, make_mask, sensitivity_map, mismatch_mask, fft2c, ifft2c
from src.solvers import SOLVERS
from src.solvers.common import tweedie
from src.solvers.pigdm import vjp
from src.solvers.baselines import gradient, gradient_adjoint


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(17)
    torch.set_num_threads(2)


@pytest.mark.parametrize("kind", ["cartesian_random", "cartesian_equispaced", "radial"])
@pytest.mark.parametrize("level", [0., .6])
def test_complex_and_real_adjoint(kind, level):
    op = FourierOperator(make_mask(16, kind, 2, 2), sensitivity_map(16, level).to(torch.complex128))
    x = torch.randn(2, 1, 16, 16, dtype=torch.complex128)
    y = torch.randn_like(x)
    lhs = (op(x).conj()*y).sum()
    rhs = (x.conj()*op.adjoint(y)).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)
    xr = x.real
    torch.testing.assert_close((op(xr).conj()*y).sum().real, (xr*op.real_adjoint(y)).sum(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["cartesian_random", "cartesian_equispaced"])
@pytest.mark.parametrize("r", [1, 2, 4, 8])
def test_cartesian_count_and_acs(kind, r):
    m = make_mask(64, kind, r, 8)
    assert m[:, 0].sum() == round(64/r)
    assert torch.all(m[28:36] == 1)


def test_radial_density_and_acs():
    for r in [2, 4, 8]:
        m = make_mask(64, "radial", r, 4)
        assert abs(m.sum()-4096/r) <= 2*64
        assert torch.all(m[30:34,30:34] == 1)


def test_mask_axis_monotonic_paired_and_feasible():
    base = make_mask(64, acceleration=4, acs_lines=8)
    old = torch.zeros_like(base, dtype=torch.bool)
    for level in [0., .125, .25]:
        m = mismatch_mask(base, level, 8, 42)
        changed = m != base
        assert float(changed.float().mean()) == level
        assert m.sum() == base.sum()
        assert torch.all(m[28:36] == 1)
        assert torch.all(changed | ~old)
        old = changed
    with pytest.raises(ValueError):
        mismatch_mask(base, .5, 8, 42)


def real_matrix(op, size):
    basis = torch.eye(size*size, dtype=torch.float64).reshape(size*size,1,size,size)
    cols = op(basis).reshape(size*size,-1).T
    return torch.cat((cols.real, cols.imag), dim=0)


@pytest.mark.parametrize("size", [4, 5])
def test_diffpir_real_closed_form_matches_dense_least_squares(size):
    mask = (torch.rand(size,size) > .4).double()
    op = FourierOperator(mask)
    x0 = torch.randn(1,1,size,size, dtype=torch.float64)
    y = torch.randn(1,1,size,size, dtype=torch.complex128)
    rho = .27
    A = real_matrix(op, size)
    yr = torch.cat((y.real.flatten(), y.imag.flatten()))
    dense = torch.linalg.solve(A.T@A+rho*torch.eye(size*size,dtype=A.dtype), A.T@yr+rho*x0.flatten())
    torch.testing.assert_close(op.prox(x0,y,rho).flatten(), dense, atol=1e-9, rtol=1e-9)


def test_complex_closed_form_for_symmetric_mask():
    from src.operators.fourier import reflect_frequency
    m = (torch.rand(8,8)>.5).double()
    m = torch.maximum(m, reflect_frequency(m))
    op = FourierOperator(m)
    x = torch.randn(1,1,8,8,dtype=torch.float64)
    y = torch.randn(1,1,8,8,dtype=torch.complex128)
    direct = ifft2c((m*y+.4*fft2c(x))/(m+.4)).real
    torch.testing.assert_close(op.prox(x,y,.4), direct)


def test_pigdm_covariance_matches_dense_real_inverse():
    n = 4
    op = FourierOperator((torch.rand(n,n)>.5).double(), sigma=.13)
    A = real_matrix(op,n)
    r = torch.randn(1,1,n,n,dtype=torch.complex128)
    rr = torch.cat((r.real.flatten(),r.imag.flatten()))
    dense = A.T @ torch.linalg.solve(.7*A@A.T+op.sigma**2*torch.eye(2*n*n), rr)
    torch.testing.assert_close(op.covariance_backproject(r,.7).flatten(), dense, atol=1e-8, rtol=1e-8)


def test_pigdm_vjp_finite_difference_through_network():
    model = torch.nn.Sequential(torch.nn.Conv2d(1,2,3,padding=1),torch.nn.Tanh(),torch.nn.Conv2d(2,1,1)).double()
    x = torch.randn(1,1,4,4,dtype=torch.float64,requires_grad=True)
    d, v = torch.randn_like(x), torch.randn_like(x)
    a = x.new_tensor(.6)
    f = lambda z: tweedie(z, model(z), a)
    analytic = (vjp(f(x),x,v)*d).sum()
    h = 1e-5
    finite = ((f(x+h*d)-f(x-h*d))*v).sum()/(2*h)
    torch.testing.assert_close(analytic,finite,atol=1e-7,rtol=1e-7)


def test_tv_gradient_adjoint():
    x = torch.randn(1,1,8,8)
    p = torch.randn(2,1,1,8,8)
    torch.testing.assert_close((gradient(x)*p).sum(),(x*gradient_adjoint(p)).sum())


class TinyPrior:
    alphas_cumprod = torch.linspace(.999, .05, 1000)
    def predict_eps(self,x,t):
        return .1*torch.tanh(x)


class HighNoisePrior:
    """A deliberately extreme first timestep for sampler-stability tests."""
    alphas_cumprod = torch.cat((torch.full((999,), .999), torch.tensor([1e-8])))

    def predict_eps(self, x, t):
        return torch.zeros_like(x)


@pytest.mark.parametrize("name", ["dps", "diffpir", "pigdm"])
def test_guided_solvers_bound_high_noise_clean_estimates(name):
    # A partially observed problem preserves null-space components.  Without
    # clipping the Tweedie x0 estimate at abar=1e-8, those components reach
    # O(1e4) and make all three guided samplers unusable.
    op = FourierOperator(make_mask(8, acceleration=2, acs_lines=2), sigma=.02)
    clean = torch.rand(1, 1, 8, 8)
    result = SOLVERS[name](op(clean), op, HighNoisePrior(), dict(steps=2, seed=2, zeta=.1))
    assert torch.isfinite(result).all()
    assert result.abs().max() < 5


@pytest.mark.parametrize("name", ["dps", "diffpir", "pigdm"])
def test_all_solvers_full_noiseless_with_explicit_projection(name):
    # Projection is tested honestly as a wrapper; not proof of a trained prior.
    x = torch.rand(1,1,8,8)
    op = FourierOperator(torch.ones(8,8), sigma=0.)
    result = SOLVERS[name](op(x),op,TinyPrior(),dict(steps=4,seed=2,zeta=.1,final_consistency=True))
    assert -10*torch.log10((result-x).square().mean()) > 100


@pytest.mark.parametrize("name", ["dps", "diffpir", "pigdm"])
def test_solvers_deterministic_and_use_measurements(name):
    op = FourierOperator(make_mask(8,acceleration=2,acs_lines=2),sigma=.02)
    y = op(torch.rand(1,1,8,8))
    c = dict(steps=4,seed=41,zeta=.1)
    a = SOLVERS[name](y,op,TinyPrior(),c)
    b = SOLVERS[name](y,op,TinyPrior(),c)
    other = SOLVERS[name](y*.7,op,TinyPrior(),c)
    assert torch.equal(a,b)
    assert not torch.equal(a,other)
    assert torch.isfinite(a).all()


def test_dps_retains_network_jacobian():
    from src.solvers.dps import update
    op = FourierOperator(torch.ones(4,4),sigma=.1)
    # Stay in the unsaturated x0 region: this test targets the epsilon-network
    # Jacobian, while clipping is covered by the high-noise stability test.
    x = .1*torch.randn(1,1,4,4)
    class Linear:
        def predict_eps(self,z,t):
            return .4*z
    a, ap = torch.tensor(.5),torch.tensor(.8)
    y = op(torch.rand_like(x))
    gen = torch.Generator().manual_seed(0)
    result = update(x,0,a,ap,y,op,Linear(),dict(zeta=.2),gen)
    derivative = (1-(1-a).sqrt()*.4)/a.sqrt()
    x0 = derivative*x
    residual = y-op(x0)
    norm = residual.norm()
    expected_grad = -derivative*op.real_adjoint(residual)/norm
    expected = ap.sqrt()*x0+(1-ap).sqrt()*.4*x-.2/norm*expected_grad
    torch.testing.assert_close(result,expected)


def test_sampling_honors_expired_budget():
    from src.solvers.common import sample
    with pytest.raises(TimeoutError,match="wall-clock"):
        sample(TinyPrior(),1,2,"cpu",deadline=0.)

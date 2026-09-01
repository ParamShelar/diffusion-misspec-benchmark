import copy
import random
import numpy as np
import pytest
import torch
from src.models.unet import MedicalUNet
from src.models.checkpoint import capture_rng,restore_rng,atomic_save,latest_checkpoint,prune_checkpoints
from src.models.train import learning_rate,require_drive


def test_architecture_parameter_budget_and_attention_resolution():
    from src.models.unet import Attention
    model = MedicalUNet(gradient_checkpointing=False).eval()
    assert 15_000_000 <= sum(p.numel() for p in model.parameters()) <= 25_000_000
    seen = []
    hooks = [m.register_forward_pre_hook(lambda m,x: seen.append(tuple(x[0].shape[-2:]))) for m in model.modules() if isinstance(m,Attention)]
    with torch.no_grad():
        out = model(torch.randn(1,1,64,64),torch.tensor([250])).sample
    assert out.shape == (1,1,64,64) and torch.isfinite(out).all()
    assert seen == [(16,16),(16,16)]
    for h in hooks:
        h.remove()


def test_checkpointed_model_backprop():
    m = MedicalUNet(base=8,expansion=1).train()
    result = m(torch.randn(1,1,64,64),torch.tensor([12])).sample
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_exact_optimizer_ema_and_rng_resume(tmp_path):
    torch.manual_seed(32)
    np.random.seed(32)
    random.seed(32)
    m = torch.nn.Linear(3,2)
    ema = copy.deepcopy(m)
    opt = torch.optim.AdamW(m.parameters(),lr=.01)
    def step(model,average,optimizer):
        x = torch.randn(5,3)
        loss = model(x).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for a,p in zip(average.parameters(),model.parameters()):
                a.lerp_(p,.1)
    step(m,ema,opt)
    payload = dict(model=m.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=1,
                   rng=capture_rng(),model_config={},alphas_cumprod=torch.ones(2))
    atomic_save(payload,tmp_path/"step-000000001.pt")
    step(m,ema,opt)
    expected_random = (random.random(),np.random.rand(),torch.rand(1))
    path,payload = latest_checkpoint(tmp_path)
    restored,avg = torch.nn.Linear(3,2),torch.nn.Linear(3,2)
    restored.load_state_dict(payload["model"])
    avg.load_state_dict(payload["ema"])
    opt2 = torch.optim.AdamW(restored.parameters(),lr=.01)
    opt2.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    step(restored,avg,opt2)
    actual_random = (random.random(),np.random.rand(),torch.rand(1))
    assert expected_random[0:2] == actual_random[0:2]
    assert torch.equal(expected_random[2],actual_random[2])
    for a,b in zip(m.parameters(),restored.parameters()):
        assert torch.equal(a,b)
    for a,b in zip(ema.parameters(),avg.parameters()):
        assert torch.equal(a,b)


def test_retention_keeps_three_and_best(tmp_path):
    for n in range(6):
        atomic_save({"step":n},tmp_path/f"step-{n:09d}.pt")
    atomic_save({"step":0},tmp_path/"best.pt")
    prune_checkpoints(tmp_path)
    assert len(list(tmp_path.glob("step-*.pt"))) == 3
    assert (tmp_path/"best.pt").exists()


def test_warmup_and_drive_enforcement(tmp_path):
    assert learning_rate(0) == pytest.approx(2e-4/500)
    assert learning_rate(499) == pytest.approx(2e-4)
    assert learning_rate(800) == pytest.approx(2e-4)
    with pytest.raises(RuntimeError,match="Mount Google Drive"):
        require_drive(tmp_path)


def test_trained_prior_loads_ema_instead_of_raw(tmp_path):
    from src.models.prior import load_prior
    m = MedicalUNet(base=8,expansion=1)
    ema = copy.deepcopy(m.state_dict())
    raw = {k:torch.zeros_like(v) for k,v in ema.items()}
    path = tmp_path/"checkpoint.pt"
    torch.save(dict(model=raw,ema=ema,model_config=m.config,
                    alphas_cumprod=torch.linspace(.99,.01,1000)),path)
    prior = load_prior({"prior":{"kind":"trained","checkpoint":str(path)}},"cpu")
    for key,value in prior.model.state_dict().items():
        assert torch.equal(value,ema[key])
    assert not prior.model.training
    assert not any(p.requires_grad for p in prior.model.parameters())

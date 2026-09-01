import json
import numpy as np
import pytest
import torch
from src.data.datasets import preprocess, get_dataset
from src.ledger import Ledger, FIELDS
from src.metrics.quality import fid
from src.runtime import load_config


def test_percentile_hot_pixel_and_zero_image():
    x = np.full((2,64,64),50,dtype=np.uint8)
    x[0,0,0] = 255
    x[1] = 0
    y = preprocess(x)
    assert y.dtype == torch.float32 and y.shape == (2,1,64,64)
    assert y[0].min() == 1 and y[0].max() == 1
    assert y[1].max() == 0


def test_npy_splits_cache_and_invalidation(tmp_path):
    for i,split in enumerate(["train","val","test"]):
        (tmp_path/split).mkdir()
        x = np.zeros((64,64),dtype=np.uint8)
        x[i+1:i+5] = 40
        np.save(tmp_path/split/"slice.npy",x)
    a = get_dataset("npy_dir","train",tmp_path)
    assert not torch.equal(a,get_dataset("npy_dir","val",tmp_path))
    assert len(list(tmp_path.glob("*.pt"))) == 1
    np.save(tmp_path/"train"/"slice.npy",np.ones((64,64),dtype=np.uint8))
    assert not torch.equal(a,get_dataset("npy_dir","train",tmp_path))
    assert len(list(tmp_path.glob("*.pt"))) == 2


def test_csv_determinism_resume_and_torn_tail(tmp_path):
    row = {k:"value" for k in FIELDS}
    row.update(level=.125,seed=4,image_id=2,psnr=22.123456789012345)
    paths = [tmp_path/"a.csv",tmp_path/"b.csv"]
    for p in paths:
        ledger = Ledger(p)
        assert ledger.append(row)
        assert not ledger.append(row)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    with paths[0].open("ab") as f:
        f.write(b"broken,torn,row")
    assert len(Ledger(paths[0]).rows()) == 1
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_fid_equal_sets_and_known_shift():
    a = np.random.default_rng(3).normal(size=(12,30))
    assert fid(a,a) < 1e-9
    assert abs(fid(a,a+1)-30) < 1e-8


def test_local_rejects_full_run_and_oversized_batch(tmp_path):
    with pytest.raises(ValueError,match="Local tier"):
        load_config("configs/local.yaml")
    c = load_config("configs/local.yaml",smoke=True)
    assert c["sampling"]["batch_size"] == 1


def test_gate_rejects_missing_and_smoke_for_full(tmp_path):
    from scripts.run_sweep import validate_gate
    class Prior:
        identity = "abc"
    c = {"output":str(tmp_path),"smoke":False,"sampling":{"steps":100}}
    with pytest.raises(RuntimeError,match="missing"):
        validate_gate(c,Prior(),"val")
    (tmp_path/"prior_gate.json").write_text(json.dumps({"status":"SMOKE_PASS"}))
    with pytest.raises(RuntimeError,match="failed"):
        validate_gate(c,Prior(),"val")

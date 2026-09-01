"""Colab-specific training engine used by notebooks/train_colab.ipynb.

The time budget covers this call, including setup, sampling, checkpointing and
quality evaluation. Reserve a configurable tail for the full gate; an incomplete
gate is a FAIL, never a research permit. Drive I/O and a running CUDA kernel are
cooperative boundaries, so a few seconds of finalization overrun are possible.
"""
import copy
import json
import random
from pathlib import Path
import time
import warnings
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from diffusers import DDPMScheduler
from src.data.datasets import from_config, tensor_digest
from src.models.unet import MedicalUNet, make_model
from src.models.prior import Prior, weights_digest
from src.models.checkpoint import capture_rng, restore_rng, atomic_save, prune_checkpoints, latest_checkpoint
from src.solvers.common import sample
from src.runtime import atomic_json, device_for, file_digest


def require_drive(path):
    p = Path(path).resolve()
    root = Path("/content/drive/MyDrive")
    if not p.is_relative_to(root) or not root.is_dir():
        raise RuntimeError("Mount Google Drive and use /content/drive/MyDrive/... for ALL checkpoints, logs and samples")
    p.mkdir(parents=True,exist_ok=True)
    return p


def learning_rate(step,lr=2e-4,warmup=500):
    return lr*min((step+1)/warmup,1.)


def train_colab(config, max_minutes=90, smoke=False, compile_model=False):
    started = time.monotonic()
    if max_minutes <= 0:
        raise ValueError("max_minutes must be positive")
    deadline = started + max_minutes*60
    if config["tier"] != "colab" or not torch.cuda.is_available():
        raise RuntimeError("Training requires an L4 Colab GPU runtime")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    total = torch.cuda.get_device_properties(device).total_memory/2**30
    print(f"GPU: {name}; total VRAM: {total:.2f} GiB",flush=True)
    if "T4" in name:
        warnings.warn("T4 DETECTED: bf16 is unsupported. Select an L4 runtime; training is stopped.")
    if "L4" not in name:
        raise RuntimeError(f"Expected an L4, got {name}. Select the L4 Colab runtime.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires native bf16 support")
    device = device_for(config,training=True)
    out = require_drive(config["output"])
    ckptdir = require_drive(config["training"]["checkpoint_dir"])
    log = out / "train.jsonl"
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    data = from_config(config,"train")
    data_hash = tensor_digest(data)
    gpu_data = (data*255).round().to(torch.uint8).to(device)
    del data
    print(f"Entire training split resident on GPU: {tuple(gpu_data.shape)}, {gpu_data.numel()/2**20:.1f} MiB uint8",flush=True)
    path, saved = latest_checkpoint(ckptdir)
    model = make_model(saved["model_config"]) if saved else MedicalUNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}",flush=True)
    if not 15_000_000 <= params <= 25_000_000:
        raise RuntimeError("Model parameter count must be between 15M and 25M")
    model = model.to(device,memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4)
    schedule = DDPMScheduler(num_train_timesteps=1000,beta_schedule="squaredcos_cap_v2",prediction_type="epsilon")
    abar = schedule.alphas_cumprod.to(device)
    step, best_loss, previous_seconds = 0, float("inf"), 0.
    if saved:
        if saved.get("data_hash") != data_hash:
            raise RuntimeError("Checkpoint dataset fingerprint differs; use a new Drive experiment directory")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        abar = saved["alphas_cumprod"].to(device)
        step, best_loss = saved["step"], saved.get("best_ema_loss",float("inf"))
        previous_seconds = saved.get("total_seconds",0.)
        restore_rng(saved["rng"])
        print(f"Resumed {path.name} at step {step}",flush=True)
    running = torch.compile(model) if compile_model else model
    model.train()
    batch = 1 if smoke else config["training"].get("batch_size",128)
    # Fixed GPU-only validation examples/noise/times for EMA checkpoint ranking.
    validation = from_config(config,"val")[:min(batch,16)].to(device)*2-1
    vg = torch.Generator(device=device).manual_seed(9183)
    vt = torch.randint(0,1000,(len(validation),),device=device,generator=vg)
    vn = torch.randn(validation.shape,device=device,generator=vg)
    va = abar[vt,None,None,None]
    vx = va.sqrt()*validation+(1-va).sqrt()*vn
    last_ckpt_time, initial_step = time.monotonic(), step
    throughput_start = time.monotonic()
    reserve = min(max_minutes*60*.5, config["training"].get("quality_reserve_minutes",45)*60)
    train_deadline = deadline-reserve
    if smoke:
        train_deadline = deadline-30
    prior = Prior(ema,abar,"in-memory-ema")

    def write_log(record):
        with log.open("a") as f:
            f.write(json.dumps(record,allow_nan=False)+"\n")
            f.flush()

    def save_checkpoint(reason):
        nonlocal best_loss,last_ckpt_time
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            ema_loss = F.mse_loss(ema(vx,vt).sample.float(),vn).item()
        improved = ema_loss < best_loss
        best_loss = min(best_loss,ema_loss)
        payload = dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                       step=step,rng=capture_rng(),model_config=model.config,alphas_cumprod=abar.cpu(),
                       best_ema_loss=best_loss,ema_loss=ema_loss,data_hash=data_hash,
                       total_seconds=previous_seconds+time.monotonic()-started,config=config)
        target = ckptdir/f"step-{step:09d}.pt"
        atomic_save(payload,target)
        if improved:
            atomic_save(payload,ckptdir/"best.pt")
        prune_checkpoints(ckptdir,keep=3)
        last_ckpt_time = time.monotonic()
        write_log(dict(event="checkpoint",step=step,ema_loss=ema_loss,reason=reason,path=str(target)))
        return target

    error = None
    try:
        while time.monotonic() < train_deadline:
            indices = torch.randint(len(gpu_data),(batch,),device=device)
            clean = (gpu_data[indices].float()/127.5-1).contiguous(memory_format=torch.channels_last)
            times = torch.randint(0,1000,(batch,),device=device)
            noise = torch.randn_like(clean)
            a = abar[times,None,None,None]
            noisy = a.sqrt()*clean+(1-a).sqrt()*noise
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss = F.mse_loss(running(noisy,times).sample.float(),noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for avg,current in zip(ema.parameters(),model.parameters()):
                    avg.lerp_(current,1-.9995)
            step += 1
            new_steps = step-initial_step
            if new_steps in (10,50,100):
                torch.cuda.synchronize()
                rate = new_steps/(time.monotonic()-throughput_start)
                projected = step+max(0,train_deadline-time.monotonic())*rate
                print(f"step={step} measured {rate:.2f} it/s; projected final step={projected:.0f} (quality reserve excluded)",flush=True)
                write_log(dict(event="throughput",step=step,iterations_per_second=rate,projected_step=int(projected)))
            if step%100==0 or smoke:
                write_log(dict(event="train",step=step,loss=loss.item(),lr=optimizer.param_groups[0]["lr"]))
            if step%1000==0 or time.monotonic()-last_ckpt_time>=600:
                save_checkpoint("interval")
            if step%2000==0:
                # Explicit generator keeps monitoring independent of training RNG.
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    try:
                        images = sample(prior,16,100,device,seed=4821,deadline=train_deadline)
                    except TimeoutError:
                        images = None
                if images is not None:
                    save_image(images,out/f"ema-samples-{step:09d}.png",nrow=4)
            if smoke and new_steps>=2:
                break
    except (KeyboardInterrupt,Exception) as caught:
        error = caught
    finally:
        checkpoint_path = save_checkpoint("final")
        write_log(dict(event="training_stopped",step=step,seconds=time.monotonic()-started,
                       error=str(error) if error else None))
    # Release optimizer/model memory before the quality gate.
    del running,optimizer,model
    torch.cuda.empty_cache()
    gate_config = copy.deepcopy(config)
    gate_config["prior"] = {"kind":"trained","checkpoint":str(checkpoint_path)}
    gate_config["smoke"] = smoke
    if smoke:
        gate_config["sampling"]["steps"] = 2
    import yaml
    config_path = out / "trained_sweep.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(gate_config,f,sort_keys=False)
    prior.identity = file_digest(checkpoint_path)
    from scripts.check_prior import run_gate
    # Keep a last-checkpoint/log margin; gate emits FAIL if incomplete.
    if error is None:
        try:
            result = run_gate(gate_config,prior=prior,deadline=deadline-5)
        except Exception as gate_error:
            result = {"status":"FAIL","reason":f"Quality evaluation failed: {gate_error}"}
            atomic_json(out/"prior_gate.json",result)
    else:
        result = {"status":"FAIL","reason":str(error)}
        atomic_json(out/"prior_gate.json",result)
    atomic_json(out/"training_summary.json",dict(step=step,checkpoint=str(checkpoint_path),
                elapsed_seconds=time.monotonic()-started,max_minutes=max_minutes,gate=result,
                sweep_config=str(config_path)))
    print(f"Training ended at step {step}; gate={result['status']}; elapsed={(time.monotonic()-started)/60:.1f} min",flush=True)
    if error is not None:
        raise error
    return result

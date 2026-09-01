"""64-channel DDPM U-Net, (1,2,2,2) levels, two residual blocks per level.

Residual hidden expansion=4 keeps level outputs at the specified widths while
meeting the 15–25M parameter target. Attention occurs only at 16x16, never at
the 8x8 bottleneck. Checkpointing is optional and affects training only.
"""
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Residual(nn.Module):
    def __init__(self, incoming, outgoing, time_channels=256, expansion=4):
        super().__init__()
        hidden = outgoing*expansion
        self.norm1 = nn.GroupNorm(8, incoming)
        self.conv1 = nn.Conv2d(incoming, hidden, 3, padding=1)
        self.time = nn.Linear(time_channels, hidden)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.conv2 = nn.Conv2d(hidden, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self,x,t):
        h = self.conv1(F.silu(self.norm1(x))) + self.time(F.silu(t))[:,:,None,None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels,3*channels,1)
        self.out = nn.Conv2d(channels,channels,1)

    def forward(self,x):
        b,c,h,w = x.shape
        q,k,v = self.qkv(self.norm(x)).reshape(b,3,4,c//4,h*w).unbind(1)
        attn = F.scaled_dot_product_attention(q.transpose(-1,-2),k.transpose(-1,-2),v.transpose(-1,-2))
        return x + self.out(attn.transpose(-1,-2).reshape(b,c,h,w))


class MedicalUNet(nn.Module):
    def __init__(self, base=64, multipliers=(1,2,2,2), expansion=4, gradient_checkpointing=True):
        super().__init__()
        self.config = dict(model_type="medical_unet",base=base,multipliers=list(multipliers),
                           expansion=expansion,gradient_checkpointing=gradient_checkpointing)
        self.gradient_checkpointing = gradient_checkpointing
        channels = [base*m for m in multipliers]
        tc = base*4
        self.time = nn.Sequential(nn.Linear(base,tc),nn.SiLU(),nn.Linear(tc,tc))
        self.input = nn.Conv2d(1,channels[0],3,padding=1)
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i,c in enumerate(channels):
            self.down.append(nn.ModuleList([Residual(c,c,tc,expansion),Residual(c,c,tc,expansion),
                                            Attention(c) if i==2 else nn.Identity()]))
            if i<3:
                self.downsample.append(nn.Conv2d(c,channels[i+1],3,stride=2,padding=1))
        self.middle = nn.ModuleList([Residual(channels[-1],channels[-1],tc,expansion) for _ in range(2)])
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(3,-1,-1):
            c = channels[i]
            self.up.append(nn.ModuleList([Residual(2*c,c,tc,expansion),Residual(c,c,tc,expansion),
                                          Attention(c) if i==2 else nn.Identity()]))
            if i>0:
                self.upsample.append(nn.Conv2d(c,channels[i-1],3,padding=1))
        self.out = nn.Sequential(nn.GroupNorm(8,base),nn.SiLU(),nn.Conv2d(base,1,3,padding=1))

    def residual(self,block,x,t):
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block,x,t,use_reentrant=False)
        return block(x,t)

    def forward(self,x,t):
        if tuple(x.shape[1:]) != (1,64,64):
            raise ValueError("MedicalUNet requires N x 1 x 64 x 64")
        t = torch.as_tensor(t,device=x.device).reshape(-1).expand(x.shape[0])
        half = self.config["base"]//2
        frequencies = torch.exp(-math.log(10000)*torch.arange(half,device=x.device)/(half-1))
        angles = t.float()[:,None]*frequencies[None]
        embedding = self.time(torch.cat((angles.cos(),angles.sin()),1))
        h, skips = self.input(x), []
        for i,blocks in enumerate(self.down):
            h = self.residual(blocks[0],h,embedding)
            h = blocks[2](self.residual(blocks[1],h,embedding))
            skips.append(h)
            if i<3:
                h = self.downsample[i](h)
        for block in self.middle:
            h = self.residual(block,h,embedding)
        for i,blocks in enumerate(self.up):
            h = torch.cat((h,skips.pop()),dim=1)
            h = self.residual(blocks[0],h,embedding)
            h = blocks[2](self.residual(blocks[1],h,embedding))
            if i<3:
                h = self.upsample[i](F.interpolate(h,scale_factor=2,mode="nearest"))
        return SimpleNamespace(sample=self.out(h))


def make_model(config):
    config = dict(config)
    kind = config.pop("model_type", None)
    if kind == "medical_unet":
        return MedicalUNet(**config)
    from diffusers import UNet2DModel
    return UNet2DModel.from_config(config)

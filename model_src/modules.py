"""Pyramid encoders, restoration updaters, fusion initialization and task readouts."""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .mambair import VSSBlock
from .nafnet import NAFBlock
from .restormer import Downsample, OverlapPatchEmbed, TransformerBlock
from .skff import SKFF


class RestormerEncoder3Scale(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.intro = OverlapPatchEmbed(cin, 32, bias=False)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(c, h, 2.66, False, "BiasFree")
                for c, h in zip((32, 64, 128), (1, 2, 4))
            ]
        )
        self.down = nn.ModuleList([Downsample(c) for c in (32, 64)])

    def forward(self, x):
        x = self.intro(x)
        out = []
        for k, block in enumerate(self.blocks):
            x = block(x)
            out.append(x)
            if k < 2:
                x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
                x = self.down[k](x)
        return out


def replace_encoders(model):
    model.ev = RestormerEncoder3Scale(3)
    model.ei = RestormerEncoder3Scale(1)
    assert not (
        {p.data_ptr() for p in model.ev.parameters()}
        & {p.data_ptr() for p in model.ei.parameters()}
    )


class VSSUpdate(nn.Module):
    """Original concatenated state/condition -> delta; caller adds state once."""

    def __init__(self, old, c):
        super().__init__()
        self.input_proj = copy.deepcopy(old.body[0])
        assert (
            self.input_proj.in_channels == 2 * c and self.input_proj.out_channels == c
        )
        self.core = VSSBlock(
            hidden_dim=c,
            d_state=16,
            expand=2.0,
            d_conv=3,
            dt_rank="auto",
            drop_path=0.0,
            attn_drop_rate=0.0,
            is_light_sr=False,
        )
        self.output_proj = nn.Conv2d(c, c, 1, bias=False)
        self.activation_checkpointing = True

    def forward(self, *xs):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                self._delta_forward, *xs, use_reentrant=False, preserve_rng_state=True
            )
        return self._delta_forward(*xs)

    def _delta_forward(self, *xs):
        h = self.input_proj(torch.cat(xs, 1))
        b, c, H, W = h.shape
        tokens = h.flatten(2).transpose(1, 2).contiguous()
        u = self.core(tokens, (H, W)).transpose(1, 2).contiguous().reshape(b, c, H, W)
        return self.output_proj(u - h)


def replace_updaters(model):
    for attr in ("rv", "ri"):
        old = getattr(model, attr)
        setattr(
            model,
            attr,
            nn.ModuleList([VSSUpdate(m, c) for m, c in zip(old, (32, 64, 128))]),
        )
    cores = [m.core for m in list(model.rv) + list(model.ri)]
    assert len({id(m) for m in cores}) == 6
    pointers = [p.data_ptr() for m in cores for p in m.parameters()]
    assert len(pointers) == len(set(pointers))


class SKFFInit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.core = SKFF(in_channels=channels, height=2, reduction=8, bias=False)

    def forward(self, vis_fused, ir):
        if (
            vis_fused.ndim != 4
            or vis_fused.shape != ir.shape
            or vis_fused.shape[1] != self.channels
            or vis_fused.dtype != ir.dtype
            or vis_fused.device != ir.device
        ):
            raise ValueError(
                "SKFF expects matching BCHW features of the configured width, dtype and device"
            )
        return self.core([vis_fused, ir])


class NAFRecoveryHead(nn.Module):
    """Decode final recovery states into an image-space correction."""

    def __init__(self, out_channels):
        super().__init__()
        self.middle_blks = nn.Sequential(NAFBlock(128))
        self.ups = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2))
                for c in (128, 64)
            ]
        )
        self.decoders = nn.ModuleList(
            [nn.Sequential(NAFBlock(c), NAFBlock(c)) for c in (64, 32)]
        )
        self.ending = nn.Conv2d(32, out_channels, 3, padding=1, bias=True)

    def forward(self, states):
        fine, mid, coarse = states
        x = self.middle_blks(coarse)
        for skip, up, decode in zip((mid, fine), self.ups, self.decoders):
            x = up(x)
            h, w = skip.shape[-2:]
            # RestormerEncoder3Scale pads odd spatial lengths at the bottom/right
            # before each downsample. Remove just that padding at the matching skip.
            if x.shape[-2] - h not in (0, 1) or x.shape[-1] - w not in (0, 1):
                raise ValueError(
                    f"Inconsistent decoder pyramid: {x.shape} vs {skip.shape}"
                )
            x = decode(x[..., :h, :w] + skip)
        return self.ending(x)


class CDDFuseDecoderCore(nn.Module):
    """Restormer_Decoder readout for inp_img=None; returns raw RGB logits."""

    def __init__(self, out_channels):
        super().__init__()
        self.reduce_channel = nn.Conv2d(128, 64, 1, bias=False)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(64, 8, 2, False, "WithBias") for _ in range(4)]
        )
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
        )
        self.sigmoid = nn.Identity()

    def forward(self, inp_img, base_feature, detail_feature):
        if inp_img is not None:
            raise ValueError("The RSSFuse fusion decoder reads only the shared states")
        combined = self.reduce_channel(torch.cat((base_feature, detail_feature), dim=1))
        x = combined
        for block in self.encoder_level2:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
            else:
                x = block(x)
        return self.sigmoid(self.output(x)), combined


class CDDFuseReadout(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(c, 64, 1, bias=False) for c in (32, 64, 128)]
        )
        self.core = CDDFuseDecoderCore(out_channels)

    def forward(self, states):
        fine, mid, coarse = (
            projection(x) for projection, x in zip(self.projections, states)
        )
        size = fine.shape[-2:]
        context = F.interpolate(
            mid, size=size, mode="bilinear", align_corners=False
        ) + F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        logits, _ = self.core(inp_img=None, base_feature=context, detail_feature=fine)
        return logits

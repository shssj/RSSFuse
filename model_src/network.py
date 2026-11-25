"""RSSFuse and the reference backbone used to generate training labels."""

import random

import numpy as np
import torch
from torch import nn

from .backbone import _MemoryBackbone
from .fusion import install_local_initializer, install_variant
from .modules import (
    CDDFuseReadout,
    NAFRecoveryHead,
    SKFFInit,
    replace_encoders,
    replace_updaters,
)
from .restoration import install_full_restoration


class ReferenceBackbone(_MemoryBackbone):
    """Reference trajectory for the frozen four-action teacher."""

    def __init__(self):
        widths = (32, 64, 128)
        super().__init__(widths=widths, rounds=2)
        replace_encoders(self)
        replace_updaters(self)
        self.init = nn.ModuleList(SKFFInit(width) for width in widths)
        self.hv = NAFRecoveryHead(3)
        self.hi = NAFRecoveryHead(1)
        self.df = CDDFuseReadout(3)

    def forward(self, visible, infrared, mode="C0", trace=False):
        """Process aligned BCHW visible (RGB) and infrared (one-channel) images."""
        if mode != "C0":
            raise ValueError(
                "RSSFuse updates each restoration branch from its own state"
            )
        if (
            visible.ndim != 4
            or infrared.ndim != 4
            or visible.shape[:2] != (infrared.shape[0], 3)
            or infrared.shape[1] != 1
            or visible.shape[2:] != infrared.shape[2:]
        ):
            raise ValueError(
                "Expected aligned [B,3,H,W] visible and [B,1,H,W] infrared tensors"
            )
        v, i, s = self.initial_states(visible, infrared)
        if trace:
            states = [(v, i, s)]
        for _ in range(self.rounds):
            v, i, s = self.step(v, i, s, mode)
            if trace:
                states.append((v, i, s))
        output = self.decode(v, i, s, visible, infrared)
        if trace:
            output["states"] = states
        return output


class RSSFuse(ReferenceBackbone):
    """Dual restoration and recurrent shared-state RGB fusion.

    All encoder calibrations, adaptive restoration steps, shallow selective
    memory, local-detail initialization and gain-guided writers are enabled.
    Fine/middle/coarse channels are 32/64/128. The same per-scale writers are
    used twice. The output dictionary contains vis, ir and fused.
    """

    def __init__(self):
        super().__init__()
        install_variant(self, delta_gain=True, selective_read=True)
        install_local_initializer(self)
        install_full_restoration(self)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_model(seed=42):
    """Set the experiment RNGs and construct the full RSSFuse student on CPU."""
    _seed(seed)
    return RSSFuse()


def build_teacher(seed=42):
    """Construct the reference architecture. Load its trained weights before calibration."""
    _seed(seed)
    return ReferenceBackbone()

"""Shallow selective prior adaptation, local-detail initialization and gain-guided writing."""

# LocalDetailSKFF modifies the MIRNetv2 SKFF. See its Academic Public License in LICENSES/.
import torch
from torch import nn
from torch.nn import functional as F


class LocalDetailSKFF(nn.Module):
    """Two-branch SKFF with one learned local VIS-versus-IR logit offset."""

    def __init__(self, core):
        super().__init__()
        if getattr(core, "height", None) != 2 or len(core.fcs) != 2:
            raise ValueError("the local comparison requires two-branch MIRNet-v2 SKFF")
        self.height = core.height
        self.avg_pool = core.avg_pool
        self.conv_du = core.conv_du
        self.fcs = core.fcs
        self.softmax = core.softmax
        self.detail_gain = nn.Parameter(core.fcs[0].weight.new_zeros(()))
        self.training = core.training

    @staticmethod
    def _local_detail(image):
        # Pool channel-averaged absolute finite differences on the feature grid.
        # The final column/row has no forward neighbour and contributes zero.
        dx = F.pad(
            (image[..., 1:] - image[..., :-1]).abs().mean(1, keepdim=True), (0, 1, 0, 0)
        )
        dy = F.pad(
            (image[..., 1:, :] - image[..., :-1, :]).abs().mean(1, keepdim=True),
            (0, 0, 0, 1),
        )
        return F.avg_pool2d(
            dx + dy, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )

    def branch_weights(self, features):
        """Return [B, 2, C, H, W] weights; channels keep SKFF's global prior."""
        visible, infrared = features
        summary = self.conv_du(self.avg_pool((torch.stack(features, dim=1)).sum(dim=1)))
        logits = torch.stack([head(summary) for head in self.fcs], dim=1)
        prior = self.softmax(logits)
        # Magnitude is an observational cue which may also respond to noise.
        # A single signed gain can learn which direction, if any, is helpful.
        calc_dtype = (
            torch.float32
            if prior.dtype in (torch.bfloat16, torch.float16)
            else prior.dtype
        )
        detail_v = self._local_detail(visible.to(calc_dtype))
        detail_i = self._local_detail(infrared.to(calc_dtype))
        cue = (detail_v - detail_i) / (detail_v + detail_i + 1e-6)
        offset = self.detail_gain.to(calc_dtype).tanh() * cue
        pv, pi = prior.to(calc_dtype).unbind(dim=1)
        # Equivalent to softmax((logit_v + offset, logit_i)). Writing the
        # normalizer as 1 + pv*expm1(offset) preserves the original softmax
        # bit-for-bit at gain zero while admitting a first-order gradient.
        denominator = 1 + pv * offset.expm1()
        return torch.stack(
            (pv * offset.exp() / denominator, pi / denominator), dim=1
        ).to(prior.dtype)

    def forward(self, features):
        weights = self.branch_weights(features)
        return torch.sum(torch.stack(features, dim=1) * weights, dim=1)


def install_local_initializer(model):
    """Change the finest actual SKFF path in-place before creating the optimizer."""
    if len(model.init) != 3 or [item.channels for item in model.init] != [32, 64, 128]:
        raise ValueError("expected RSSFuse three-scale SKFFInit modules")
    first = model.init[0]
    if isinstance(first.core, LocalDetailSKFF):
        raise ValueError("local SKFF control already installed")
    first.core = LocalDetailSKFF(first.core)
    return model


class SelectivePrototypeRead(nn.Module):
    """Wrap a loaded shallow PrototypeMemory. Return its scaled residual only.

    The scalar gate sees pooled features from one image. At initialization it
    preserves a normal read exactly. The selector learns from the main fusion objective.
    """

    def __init__(self, base_memory, channels=32):
        super().__init__()
        self.base_memory = base_memory
        self.selector = nn.Linear(2 * channels, 1)
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)

    # Expose prototype addressing for inspection without duplicating parameters.
    @property
    def mode(self):
        return self.base_memory.mode

    @mode.setter
    def mode(self, value):
        self.base_memory.mode = value

    @property
    def slots(self):
        return self.base_memory.slots

    @property
    def query(self):
        return self.base_memory.query

    def address(self, tokens):
        return self.base_memory.address(tokens)

    def gate_logits(self, feature, residual=None):
        if residual is None:
            residual = self.base_memory(feature)
        # Only the selector learns through this cue calculation. The main
        # fusion path still propagates gradients into feature and memory values.
        cues = torch.cat(
            [item.detach().abs().mean(dim=(2, 3)) for item in (feature, residual)],
            dim=1,
        )
        return self.selector(cues)

    def read_with_mode(self, feature, mode="learned"):
        if mode == "off":
            return torch.zeros_like(feature)
        if mode not in {"on", "learned"}:
            raise ValueError("memory read mode must be 'learned', 'on' or 'off'")
        residual = self.base_memory(feature)
        if mode == "on":
            return residual
        logits = self.gate_logits(feature, residual)
        strength = (2 * logits.sigmoid()).clamp(max=1).to(residual.dtype)
        return residual * strength[:, :, None, None]

    def forward(self, feature):
        return self.read_with_mode(feature)


class DeltaGainWriter(nn.Module):
    """Keep both updated states; weight only the two explicit increments in U."""

    MODES = ("learned", "pass", "zero_vis", "zero_ir", "zero_both")

    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.predict = nn.Linear(4, 2)
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)
        self.mode = "learned"

    def logits(self, vis, ir, dvis, dir_):
        # The auxiliary signal trains the gate head only. Gradients on the normal
        # output path through state/increment values keep their original routing.
        features = torch.stack(
            [x.detach().float().abs().mean((1, 2, 3)) for x in (vis, ir, dvis, dir_)],
            dim=1,
        )
        return self.predict(features.to(self.predict.weight.dtype))

    def write_weights(self, vis, ir, dvis, dir_):
        # Supervision compares omission with a normal write. Do not extrapolate
        # beyond the normal increment; at zero logits the multiplier is exactly 1.
        return (2 * self.logits(vis, ir, dvis, dir_).sigmoid()).clamp(max=1)

    def forward(self, shared, vis, ir, dvis, dir_):
        if self.mode not in self.MODES:
            raise ValueError(f"Unknown write mode {self.mode}")
        if self.mode == "learned":
            weights = self.write_weights(vis, ir, dvis, dir_)
            wv, wi = weights[:, 0, None, None, None], weights[:, 1, None, None, None]
            dvis, dir_ = dvis * wv.to(dvis), dir_ * wi.to(dir_)
        else:
            if self.mode in ("zero_vis", "zero_both"):
                dvis = torch.zeros_like(dvis)
            if self.mode in ("zero_ir", "zero_both"):
                dir_ = torch.zeros_like(dir_)
        return self.writer(shared, vis, ir, dvis, dir_)


def install_variant(model, *, delta_gain=False, selective_read=False):
    """Use only requested components, in place, before creating the optimizer."""
    if not (delta_gain or selective_read):
        return model
    if len(model.u) != 3 or model.rounds != 2:
        raise ValueError("RSSFuse expects three feature scales and two rounds")
    read = None
    if selective_read:
        if isinstance(model.assoc_memory, SelectivePrototypeRead):
            raise ValueError("SelectivePrototypeRead already installed")
        base = model.assoc_memory
        read = SelectivePrototypeRead(base, channels=base.query.in_channels)
        read.to(device=base.query.weight.device, dtype=base.query.weight.dtype)
        read.training = base.training
        read.selector.train(base.training)
    if delta_gain:
        if any(isinstance(u, DeltaGainWriter) for u in model.u):
            raise ValueError("DeltaGainWriter already installed")
        writers = []
        for u in model.u:
            p = next(u.parameters())
            item = DeltaGainWriter(u).to(device=p.device, dtype=p.dtype)
            item.training = u.training
            item.predict.train(u.training)
            writers.append(item)
        updated = nn.ModuleList(writers)
        updated.training = model.u.training
        model.u = updated
    if read is not None:
        model.assoc_memory = read
    return model

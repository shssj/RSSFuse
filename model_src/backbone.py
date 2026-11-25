"""Recurrent visible, infrared and fusion states used by RSSFuse.

The temporary modules in _StateBackbone preserve the initialization order and
seeded parameters used in training. The constructor in network.py replaces
the encoders, recovery blocks, initializers and decoders before any forward.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class LightBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.GroupNorm(math.gcd(8, c), c),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    def __init__(self, cin, widths):
        super().__init__()
        self.stages = nn.ModuleList()
        for k, c in enumerate(widths):
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(cin, c, 3, padding=1, stride=1 if k == 0 else 2),
                    nn.GELU(),
                    LightBlock(c),
                )
            )
            cin = c

    def forward(self, x):
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out


class Update(nn.Module):
    def __init__(self, cin, c, small=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, c, 1),
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
        )
        if small:
            nn.init.normal_(self.body[-1].weight, std=1e-3)
            nn.init.zeros_(self.body[-1].bias)

    def forward(self, *xs):
        return self.body(torch.cat(xs, 1))


class Decoder(nn.Module):
    def __init__(self, widths, cout, zero=False):
        super().__init__()
        self.top = LightBlock(widths[2])
        self.mid = nn.Sequential(
            nn.Conv2d(widths[2] + widths[1], widths[1], 1),
            nn.GELU(),
            LightBlock(widths[1]),
        )
        self.low = nn.Sequential(
            nn.Conv2d(widths[1] + widths[0], widths[0], 1),
            nn.GELU(),
            LightBlock(widths[0]),
        )
        self.head = nn.Conv2d(widths[0], cout, 3, padding=1)
        if zero:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, xs):
        x = self.top(xs[2])
        x = self.mid(
            torch.cat(
                [
                    F.interpolate(
                        x, size=xs[1].shape[-2:], mode="bilinear", align_corners=False
                    ),
                    xs[1],
                ],
                1,
            )
        )
        x = self.low(
            torch.cat(
                [
                    F.interpolate(
                        x, size=xs[0].shape[-2:], mode="bilinear", align_corners=False
                    ),
                    xs[0],
                ],
                1,
            )
        )
        return self.head(x)


class _StateBackbone(nn.Module):
    def __init__(self, widths=(32, 64, 128), rounds=2):
        super().__init__()
        self.rounds = rounds
        self.ev = Encoder(3, widths)
        self.ei = Encoder(1, widths)
        self.init = nn.ModuleList([Update(2 * c, c) for c in widths])
        self.av = nn.ModuleList([nn.Conv2d(c, c, 1) for c in widths])
        self.ai = nn.ModuleList([nn.Conv2d(c, c, 1) for c in widths])
        self.rv = nn.ModuleList([Update(2 * c, c, True) for c in widths])
        self.ri = nn.ModuleList([Update(2 * c, c, True) for c in widths])
        self.u = nn.ModuleList([Update(5 * c, c, True) for c in widths])
        self.hv = Decoder(widths, 3, True)
        self.hi = Decoder(widths, 1, True)
        self.df = Decoder(widths, 3)

    def decode(self, v, i, s, vl, il):
        # Training L1 sees unclamped restoration; evaluation/export clamps to [0,1].
        return dict(vis=vl + self.hv(v), ir=il + self.hi(i), fused=self.df(s).sigmoid())


class _VisibleAdapterBackbone(_StateBackbone):
    def __init__(self, widths=(32, 64, 128), rounds=2):
        super().__init__(widths, rounds)
        self.vf_adapter = nn.ModuleList([nn.Conv2d(c, c, 1) for c in widths])
        for layer in self.vf_adapter:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.adapter_enabled = True

    def correction(self, scale, x):
        return (
            self.vf_adapter[scale](x) if self.adapter_enabled else torch.zeros_like(x)
        )

    def initial_states(self, vl, il):
        v, i = self.ev(vl), self.ei(il)
        return (
            v,
            i,
            [
                layer(a + self.correction(k, a), b)
                for k, (layer, a, b) in enumerate(zip(self.init, v, i))
            ],
        )

    def step(self, v, i, s, mode):
        assert mode == "C0", "A4 is defined for C0 without forward feedback"
        vn, inn, sn = [], [], []
        for k, (vv, ii, ss) in enumerate(zip(v, i, s)):
            cv, ci = self.av[k](vv), self.ai[k](ii)
            dv, di = self.rv[k](vv, cv), self.ri[k](ii, ci)
            a, b = vv + dv, ii + di
            # Consistent state and increment adaptation; identity is bit-exact at zero init.
            ra, rv = self.correction(k, a), self.correction(k, vv)
            adapted_a = a + ra
            adapted_dv = dv + (ra - rv)
            vn.append(a)
            inn.append(b)
            sn.append(ss + self.u[k](ss, adapted_a, b, adapted_dv, di))
        return vn, inn, sn


class PrototypeMemory(nn.Module):
    def __init__(
        self, channels=32, key_dim=16, slots=16, temperature=0.2, chunk_size=4096
    ):
        super().__init__()
        assert (
            channels > 0
            and key_dim > 0
            and slots > 1
            and temperature > 0
            and chunk_size > 0
        )
        self.query = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.keys = nn.Parameter(torch.randn(slots, key_dim))
        self.values = nn.Parameter(torch.randn(slots, channels) * 0.02)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.slots = slots
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.mode = "normal"

    def address(self, tokens):
        query = F.normalize(tokens.float(), dim=-1, eps=1e-6)
        keys = F.normalize(self.keys.float(), dim=-1, eps=1e-6)
        return (query @ keys.t() / self.temperature).softmax(dim=-1)

    def forward(self, x):
        b, c, height, width = x.shape
        tokens = self.query(x).flatten(2).transpose(1, 2)
        if self.mode == "shuffle":
            tokens = tokens.roll(max(1, tokens.shape[1] // 2), dims=1)
        assert self.mode in ("normal", "uniform", "shuffle")
        pieces = []
        for part in tokens.split(self.chunk_size, dim=1):
            attention = self.address(part)
            if self.mode == "uniform":
                attention = torch.full_like(attention, 1 / self.slots)
            # Center the retrieval: exactly uniform addressing produces zero residual.
            retrieved = (attention - 1 / self.slots) @ self.values.float()
            pieces.append(retrieved.to(x.dtype))
        retrieved = (
            torch.cat(pieces, dim=1).transpose(1, 2).reshape(b, c, height, width)
        )
        return self.output(retrieved)


class _MemoryBackbone(_VisibleAdapterBackbone):
    def __init__(self, widths=(32, 64, 128), rounds=2):
        super().__init__(widths, rounds)
        self.assoc_memory = PrototypeMemory(widths[0])
        self.memory_enabled = True

    def correction(self, scale, x):
        adapter = super().correction(scale, x)
        if scale == 0 and self.memory_enabled:
            return adapter + self.assoc_memory(x + adapter)
        return adapter

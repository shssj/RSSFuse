"""Selective state scan backend for the four MambaIR scan directions."""

import torch
from torch.nn import functional as F


def _reference_scan(u, delta, A, B, C, D, delta_bias):
    """Grouped real-valued SSM recurrence used by RSSFuse; portable, slow on full images.

    B and C are per-sample, per-direction and per-position. Each direction
    shares its B/C vectors across its own channels. Parameters A and D are
    channel-specific. No native extension or kernel binary is required.
    """
    batch, channels, length = u.shape
    groups, states = B.shape[1:3]
    width = channels // groups
    dt = F.softplus(delta + delta_bias.reshape(1, -1, 1)).reshape(
        batch, groups, width, length
    )
    signal = u.reshape(batch, groups, width, length)
    rates = A.reshape(groups, width, states)
    carry = u.new_zeros(batch, groups, width, states)
    skip = D.reshape(groups, width)
    output = []
    for position in range(length):
        step = dt[..., position, None]
        value = signal[..., position]
        carry = (step * rates).exp() * carry + (step * value.unsqueeze(-1)) * B[
            :, :, None, :, position
        ]
        output.append((carry * C[:, :, None, :, position]).sum(-1) + value * skip)
    return torch.stack(output, dim=-1).reshape(batch, channels, length)


def selective_scan_fn(
    u, delta, A, B, C, D, *, z, delta_bias, delta_softplus, return_last_state
):
    """Use the experiment kernel on CUDA and the explicit recurrence on CPU."""
    if z is not None or delta_bias is None or not delta_softplus or return_last_state:
        raise ValueError("Unsupported selective scan configuration")
    if u.is_cuda:
        try:
            from mamba_ssm.ops.selective_scan_interface import (
                selective_scan_fn as native,
            )
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Install the mamba-ssm CUDA extension matching your PyTorch build"
            ) from error
        return native(
            u,
            delta,
            A,
            B,
            C,
            D,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        )
    return _reference_scan(u, delta, A, B, C, D, delta_bias)

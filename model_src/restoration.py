"""Modality-specific encoder calibration and recovery step sizes for RSSFuse."""

import math

import torch
from torch import nn


class ChannelCalibration(nn.Module):
    """Rescale each channel from its image-wide mean and one learned scale strength."""

    def __init__(self, channels):
        super().__init__()
        self.phi = nn.Conv2d(channels, channels, 1, groups=channels)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, feature):
        summary = feature.mean(dim=(-2, -1), keepdim=True)
        adjustment = self.alpha * self.phi(summary).tanh()
        return feature * (1 + adjustment.to(feature.dtype))


class CalibratedPyramidEncoder(nn.Module):
    """Keep the Restormer pyramid and calibrate each of its three outgoing features."""

    def __init__(self, encoder, widths):
        super().__init__()
        self.encoder = encoder
        self.calibrations = nn.ModuleList(ChannelCalibration(width) for width in widths)
        self.calibrations.to(next(encoder.parameters()))
        self.train(encoder.training)

    def forward(self, image):
        features = self.encoder(image)
        if len(features) != len(self.calibrations):
            raise ValueError("Encoder features and calibration scales must match")
        return [
            calibrate(feature)
            for calibrate, feature in zip(self.calibrations, features)
        ]


class RestorationStepGate(nn.Module):
    """A per-channel sigmoid gate conditioned on the current state and raw increment."""

    def __init__(self, channels):
        super().__init__()
        shape = (1, channels, 1, 1)
        self.state_weight = nn.Parameter(torch.zeros(shape))
        self.residual_weight = nn.Parameter(torch.zeros(shape))
        self.bias = nn.Parameter(
            torch.full(shape, math.log(19.0))
        )  # sigmoid(log(19)) = 0.95

    def forward(self, state, residual):
        current = state.mean(dim=(-2, -1), keepdim=True)
        change = residual.mean(dim=(-2, -1), keepdim=True)
        logits = self.bias + self.state_weight * current + self.residual_weight * change
        return logits.sigmoid().to(residual.dtype)


class AdaptiveRestorationUpdater(nn.Module):
    """Return a gated increment; the existing recurrent loop adds it to the state."""

    def __init__(self, updater, channels):
        super().__init__()
        self.updater = updater
        self.gate = RestorationStepGate(channels).to(next(updater.parameters()))
        self.train(updater.training)

    def forward(self, state, condition):
        raw_increment = self.updater(state, condition)
        return self.gate(state, raw_increment) * raw_increment


def install_full_restoration(model):
    """Install independent VIS/IR controls under the existing gradient-routing roots."""
    widths = tuple(item.channels for item in model.init)
    if widths != (32, 64, 128) or model.rounds != 2:
        raise ValueError("RSSFuse requires the 32/64/128-channel, two-round pyramid")
    if isinstance(model.ev, CalibratedPyramidEncoder):
        raise ValueError("Full restoration controls are already installed")
    if len(model.rv) != len(widths) or len(model.ri) != len(widths):
        raise ValueError("VIS and IR must have one restoration updater per scale")

    model.ev = CalibratedPyramidEncoder(model.ev, widths)
    model.ei = CalibratedPyramidEncoder(model.ei, widths)
    for scale, width in enumerate(widths):
        model.rv[scale] = AdaptiveRestorationUpdater(model.rv[scale], width)
        model.ri[scale] = AdaptiveRestorationUpdater(model.ri[scale], width)
    return model

"""Deterministic degradation and image I/O shared by training and evaluation."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# Part of the original random seed key. Changing this string changes the images.
VERSION = "DARC-CX-v1-20261110"
CONDITIONS = {
    "VIS_lowlight": (1, 0), "VIS_haze": (2, 0), "VIS_noise": (4, 0),
    "IR_lowcontrast": (0, 1), "IR_stripes": (0, 2), "IR_noise": (0, 4),
    "VIS_compound": (7, 0), "IR_compound": (0, 7),
    "BOTH_compound": (7, 7), "CLEAN": (0, 0),
}


def stable(*parts):
    seed_key = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(seed_key).digest()[:8], "little") % (2**63 - 1)


def degrade(x, key, modality, mask=None):
    """Apply frozen sensor-specific degradations to uint8 HWC. Return pixels and parameters."""
    if modality not in ("vi", "ir") or x.dtype != np.uint8 or x.ndim != 3:
        raise ValueError("Expected a uint8 HWC image with modality 'vi' or 'ir'.")
    seed = stable(VERSION, key, modality)
    rng = np.random.default_rng(seed)
    if mask is None:
        mask = 0 if rng.random() < 0.1 else int(rng.integers(1, 8))
    if mask not in range(8):
        raise ValueError(f"Invalid degradation bit mask {mask}")
    y = x.astype(np.float32) / 255.
    parameters = {}
    if modality == "vi":
        if mask & 1:
            gain = float(rng.uniform(.65, .9))
            gamma = float(rng.uniform(1.2, 1.8))
            y = np.clip(gain * y**gamma, 0, 1)
            parameters["lowlight"] = dict(gain=gain, gamma=gamma)
        if mask & 2:
            transmission = float(rng.uniform(.70, .90))
            airlight = float(rng.uniform(.8, 1.))
            y = np.clip(transmission * y + (1 - transmission) * airlight, 0, 1)
            parameters["haze"] = dict(transmission=transmission, airlight=airlight)
    else:
        if mask & 1:
            factor = float(rng.uniform(.55, .85))
            mean = y.mean((0, 1), keepdims=True)
            y = np.clip(mean + factor * (y - mean), 0, 1)
            parameters["lowcontrast"] = dict(factor=factor)
        if mask & 2:
            sigma = float(rng.uniform(4., 10.))
            stripe = rng.normal(0, sigma / 255., (1, y.shape[1], 1)).astype(np.float32)
            stripe -= stripe.mean()
            y = np.clip(y + stripe, 0, 1)
            parameters["stripes"] = dict(sigma_255=sigma)
    if mask & 4:
        sigma = float(rng.uniform(5., 15.))
        y = np.clip(y + rng.normal(0, sigma / 255., y.shape).astype(np.float32), 0, 1)
        parameters["noise"] = dict(sigma_255=sigma)
    return np.rint(y * 255).astype(np.uint8), dict(seed=seed, mask=mask, parameters=parameters)


def read_image(path, mode):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert(mode)).copy()
    return pixels[:, :, None] if pixels.ndim == 2 else pixels


def to_tensor(pixels):
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.


def uint8_image(tensor):
    pixels = (tensor.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

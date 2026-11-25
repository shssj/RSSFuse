"""Check the seed-42 reproduction files and run a small CPU inference."""
import hashlib
import json
import re
from pathlib import Path

import torch

from inference import load_model, predict_pair
from model_src import build_model, build_teacher, make_probe
from training.data import load_reference_trace
from training.losses import FusionLoss
from training.runner import load_initial_weights, restore_probe


ROOT = Path(__file__).resolve().parent
ASSETS = {
    "weights/RSSFuse.pth", "reproduction/initial.pth", "reproduction/teacher.pth",
    "reproduction/probe.pth", "reproduction/train_trace.csv",
    "reproduction/archive_hashes.json", "evaluation/reference/per_image.csv",
}


def verify():
    torch.set_num_threads(2)
    manifest = (ROOT / "reproduction/assets.sha256").read_text(encoding="utf-8")
    checked = set()
    for line in manifest.splitlines():
        digest, name = line.split("  ", 1)
        if name not in ASSETS or name in checked or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Unexpected checksum entry: {name}")
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch for {name}")
        checked.add(name)
    if checked != ASSETS:
        raise ValueError(f"Missing entries: {sorted(ASSETS - checked)}")
    artifact_dir = ROOT / "reproduction"
    source = json.loads((artifact_dir / "archive_hashes.json").read_text(encoding="utf-8"))
    initial = load_initial_weights(build_model(42), artifact_dir / "initial.pth", 42)
    if initial != source["initial_state_sha256"]:
        raise ValueError("Student initialization differs from the experiment record")
    teacher = build_teacher(42)
    snapshot = torch.load(artifact_dir / "teacher.pth", weights_only=True, map_location="cpu")
    if (snapshot["seed"], snapshot["step"], snapshot["scheduler"]["T_max"]) != (42, 5000, 20000):
        raise ValueError("Unexpected teacher training schedule")
    if snapshot["source_sha256"] != source["teacher_checkpoint_sha256"]:
        raise ValueError("Unexpected original teacher checkpoint")
    teacher.load_state_dict(snapshot["model"], strict=True)
    probe = make_probe(teacher, FusionLoss())
    restore_probe(probe, artifact_dir / "probe.pth", 42, source["teacher_checkpoint_sha256"])
    load_reference_trace(artifact_dir / "train_trace.csv", 5000, 42)

    model = load_model(ROOT / "weights/RSSFuse.pth", "cpu")
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != source["parameters"]:
        raise ValueError(f"Unexpected number of trainable parameters: {parameters}")
    # The two inputs are deterministic and test padding and output cropping.
    ramp = torch.linspace(0, 1, 17 * 19).reshape(1, 17, 19)
    visible = torch.cat((ramp, 1 - ramp, ramp.flip(-1)), dim=0)
    predictions = predict_pair(model, visible, ramp, "cpu")
    for key, channels in (("fused", 3), ("vis", 3), ("ir", 1)):
        value = predictions[key]
        if tuple(value.shape) != (channels, 17, 19) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {key} prediction")
    print(f"Verified {len(checked)} files, seed-42 initialization, teacher, probe and 5,000 batch records.")
    print(f"Loaded {parameters:,} trained parameters; three finite predictions at 17 x 19.")


if __name__ == "__main__":
    verify()

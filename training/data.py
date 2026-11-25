"""Locked, full-resolution training pairs with deterministic paired geometry."""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.protocol import VERSION, degrade, stable


BATCH_SIZE, CROP_SIZE = 8, 256
FIELDS = ("vh", "ih", "vl", "il")
PAIR_MANIFEST = Path(__file__).with_name("train_pairs.csv")


def batch_digest(batch):
    digest = hashlib.sha256()
    for name in FIELDS:
        digest.update(batch[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_reference_trace(path, steps, seed):
    """Read an original or release CSV and match it to the seed's archived stream."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        fingerprint_key = next((key for key in ("batch_tensor_sha256", "batch_sha256") if key in fields), None)
        names_key = next((key for key in ("batch_names", "sample_ids") if key in fields), None)
        if "step" not in fields or fingerprint_key is None:
            raise ValueError("Reference trace requires step and batch_tensor_sha256 or batch_sha256")
        trace = []
        for number, row in enumerate(reader, 1):
            fingerprint = row[fingerprint_key]
            if (row["step"] != str(number) or not fingerprint or len(fingerprint) != 64
                    or any(c not in "0123456789abcdef" for c in fingerprint)):
                raise ValueError(f"Invalid reference trace at step {number}")
            trace.append((fingerprint, row[names_key] if names_key else None))
    known = json.loads(Path(__file__).with_name("stream_checks.json").read_text()).get(str(seed))
    if not known or len(trace) != steps or [item[0] for item in trace[:len(known)]] != known:
        raise ValueError(f"Expected the complete {steps}-step reference stream for seed {seed}")
    return trace


class TrainingPairs:
    """Select exactly the archived pair IDs in their original manifest order.

    Raw images use root/DATASET/{vi,ir}/NUMERIC_ID.{png,jpg,jpeg}.
    A local lossless cache avoids synthesizing the same full image every epoch.
    """
    def __init__(self, root, cache, seed=42, reference_trace=None):
        self.root, self.cache, self.seed = Path(root), Path(cache), seed
        self.cache.mkdir(parents=True, exist_ok=True)
        with PAIR_MANIFEST.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.paths = {}
        for dataset in sorted({r["dataset"] for r in self.rows}):
            for modality in ("vi", "ir"):
                folder = self.root / dataset / modality
                if not folder.is_dir():
                    raise FileNotFoundError(f"Missing raw image folder: {folder}")
                index = {}
                for path in folder.iterdir():
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        key = path.stem.split("__")[-1]
                        if key in index:
                            raise ValueError(f"Ambiguous image ID {key} in {folder}")
                        index[key] = path
                for row in self.rows:
                    if row["dataset"] != dataset:
                        continue
                    file_id = Path(row["name"]).stem.split("__")[-1]
                    if file_id not in index:
                        raise FileNotFoundError(f"Missing {dataset}/{modality}/{file_id} in {folder}")
                    self.paths[row["name"], modality] = index[file_id]
        checks = json.loads(Path(__file__).with_name("stream_checks.json").read_text())
        self.checks = checks.get(str(seed), [])
        self.reference_trace = reference_trace
        self.verified = set()

    def images(self, row):
        images = {}
        for modality, mode, hq, lq in (("vi","RGB","vh","vl"), ("ir","L","ih","il")):
            identity = row["name"], modality
            path = self.paths[identity]
            with Image.open(path) as image:
                clean = np.asarray(image.convert(mode)).copy()
            fingerprint = row[modality + "_pixel_sha256"]
            if identity not in self.verified:
                if hashlib.sha256(clean.tobytes()).hexdigest() != fingerprint:
                    raise ValueError(f"Decoded training pixels differ from the locked manifest: {path}")
                self.verified.add(identity)
            if clean.ndim == 2:
                clean = clean[..., None]
            cache_file = self.cache / f"{row['name']}.{modality}.{fingerprint}.png"
            if cache_file.exists():
                with Image.open(cache_file) as image:
                    degraded = np.asarray(image.convert(mode)).copy()
                if degraded.ndim == 2:
                    degraded = degraded[..., None]
            else:
                degraded, _ = degrade(clean, "train|" + row["name"], modality, None)
                plane = degraded[..., 0] if mode == "L" else degraded
                # Same cache can safely be populated by two training processes.
                with tempfile.NamedTemporaryFile(dir=self.cache, delete=False) as handle:
                    temporary = handle.name
                try:
                    Image.fromarray(plane).save(temporary, format="PNG")
                    os.replace(temporary, cache_file)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            images[hq], images[lq] = clean, degraded
        if len({im.shape[:2] for im in images.values()}) != 1:
            raise ValueError(f"Unaligned image dimensions for {row['name']}")
        return images

    def batch(self, step):
        slices = {name: [] for name in FIELDS}
        names = []
        for pos in range(BATCH_SIZE):
            cycle, index = divmod((step - 1) * BATCH_SIZE + pos, len(self.rows))
            order = np.random.default_rng(stable(VERSION, self.seed, "order", cycle)).permutation(len(self.rows))
            row = self.rows[int(order[index])]
            images = self.images(row)
            height, width = images["vh"].shape[:2]
            rng = np.random.default_rng(stable(VERSION, self.seed, "geometry", step, pos, row["name"]))
            y, x = int(rng.integers(height - CROP_SIZE + 1)), int(rng.integers(width - CROP_SIZE + 1))
            turn, flip = int(rng.integers(4)), bool(rng.integers(2))
            for name, image in images.items():
                crop = np.rot90(image[y:y+CROP_SIZE, x:x+CROP_SIZE], turn)
                if flip:
                    crop = crop[:, ::-1]
                slices[name].append(torch.from_numpy(crop.copy()).permute(2,0,1).float() / 255.)
            names.append(row["name"])
        batch = {key: torch.stack(group) for key, group in slices.items()}
        fingerprint = batch_digest(batch)
        if 1 <= step <= len(self.checks) and fingerprint != self.checks[step-1]:
            raise ValueError(f"Training batch at step {step} differs from the locked stream; check inputs/cache")
        if self.reference_trace is not None:
            expected, names_at_step = self.reference_trace[step-1]
            if fingerprint != expected or (names_at_step is not None and names_at_step != ";".join(names)):
                raise ValueError(f"Training batch at step {step} differs from the archived reference trace")
        return batch, names, fingerprint

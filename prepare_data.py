"""Assemble and verify the training, development and locked test images from original releases."""
import argparse
import csv
import hashlib
import io
import os
import tempfile
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

from PIL import Image


ROOT = Path(__file__).resolve().parent
DATASETS = ("FMB", "LLVIP", "M3FD")
SUFFIXES = {".png", ".jpg", ".jpeg"}


def requests(split):
    items = []
    manifests = {"train": "training/train_pairs.csv", "dev": "evaluation/manifests/dev_images.csv",
                 "test": "evaluation/manifests/test_images.csv"}
    for group, manifest in manifests.items():
        if split not in ("all", group):
            continue
        with (ROOT / manifest).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["id"] = (row["pair_id"] if group == "test" else Path(row["name"]).stem.split("__")[-1])
                items.append((group, row))
    return items


def index_sources(paths, ids, stack):
    index = defaultdict(list)

    def add(name, container, member):
        file_id = name.stem.split("__")[-1]
        if name.suffix.lower() in SUFFIXES and file_id in ids:
            index[file_id].append((name, container, member))

    for path in paths:
        if path.is_dir():
            for directory, _, filenames in os.walk(path):
                for filename in filenames:
                    entry = Path(directory, filename)
                    add(entry, None, entry)
        elif path.is_file() and zipfile.is_zipfile(path):
            archive = stack.enter_context(zipfile.ZipFile(path))
            for info in archive.infolist():
                add(PurePosixPath(info.filename.replace("\\", "/")), archive, info)
        else:
            raise ValueError(f"Supply an extracted image directory or an original ZIP: {path}")
    return index


def verified_pixels(raw, row, modality):
    expected_file = row.get(modality + "_sha256")
    if expected_file and hashlib.sha256(raw).hexdigest() != expected_file:
        return None
    with Image.open(io.BytesIO(raw)) as source:
        pixels = source.convert("RGB" if modality == "vi" else "L")
    if "width" in row and pixels.size != (int(row["width"]), int(row["height"])):
        return None
    if hashlib.sha256(pixels.tobytes()).hexdigest() != row[modality + "_pixel_sha256"]:
        return None
    return pixels


def save_bytes(target, image_bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(image_bytes)
        temporary.replace(target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def to_png(image, **options):
    with io.BytesIO() as handle:
        image.save(handle, format="PNG", **options)
        return handle.getvalue()


def materialize(index, group, row, modality, args):
    file_id = row["id"]
    name = row[modality] if group == "test" else file_id + ".png"
    target = args.output / group / row["dataset"] / modality / name
    if not args.check_only and target.exists():
        if verified_pixels(target.read_bytes(), row, modality) is None:
            raise ValueError(f"Existing output differs from the manifest: {target}")
        return
    aliases = {"vi", "vis", "visible", "rgb"} if modality == "vi" else {"ir", "infrared"}
    candidates = sorted(index[file_id], key=lambda item: aliases.isdisjoint(p.lower() for p in item[0].parts[:-1]))
    for _, archive, entry in candidates:
        raw = archive.read(entry) if archive is not None else entry.read_bytes()
        pixels = verified_pixels(raw, row, modality)
        if pixels is None:
            continue
        if not args.check_only:
            save_bytes(target, raw if group == "test" else to_png(pixels, compress_level=1))
        return
    raise ValueError(f"No matching {group} {row['dataset']}/{modality}/{file_id} "
                     f"among {len(candidates)} candidate(s). Check the dataset release and extract any nested ZIPs.")


def prepare_dev_inputs(root):
    import numpy as np
    from evaluation.protocol import CONDITIONS, degrade

    with (ROOT / "evaluation/manifests/dev_cases.csv").open(encoding="utf-8", newline="") as handle:
        cases = list(csv.DictReader(handle))
    for row in cases:
        name, ds, condition = row["name"], row["dataset"], row["condition"]
        file_id = Path(name).stem.split("__")[-1]
        key = f"val_{condition}|{name}"
        for modality, mode, mask in zip(("vi", "ir"), ("RGB", "L"), CONDITIONS[condition]):
            target = root / "dev_inputs" / condition / ds / modality / (file_id + ".png")
            expected = row[modality + "_lq_sha256"]
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Existing development input differs from the manifest: {target}")
                continue
            with Image.open(root / "dev" / ds / modality / (file_id + ".png")) as image:
                pixels = np.asarray(image.convert(mode))
            if mode == "L":
                pixels = pixels[..., None]
            degraded, _ = degrade(pixels, key, modality, mask)
            raw = to_png(Image.fromarray(degraded if mode == "RGB" else degraded[..., 0]))
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Generated development PNG differs for {ds}/{file_id}/{condition}/{modality}. "
                                 "Check the pinned NumPy, Pillow and PNG/zlib runtime.")
            save_bytes(target, raw)
    print(f"Verified development inputs: {len(cases)} cases, {2 * len(cases)} PNGs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for ds in DATASETS:
        parser.add_argument("--" + ds.lower(), required=True, nargs="+", type=Path,
                            help=f"One or more original {ds} image directories or ZIP files")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="all")
    parser.add_argument("--check-only", action="store_true", help="Verify source files without creating any output")
    parser.add_argument("--with-dev-inputs", action="store_true",
                        help="Also generate and verify the 600 degraded/clean development inputs")
    args = parser.parse_args()
    if args.with_dev_inputs and (args.check_only or args.split not in ("all", "dev")):
        parser.error("--with-dev-inputs requires a materialized dev split (use --split dev or --split all)")
    items = requests(args.split)
    for ds in DATASETS:
        selected = [(group, row) for group, row in items if row["dataset"] == ds]
        counts = defaultdict(int)
        with ExitStack() as stack:
            index = index_sources(getattr(args, ds.lower()), {row["id"] for _, row in selected}, stack)
            for group, row in selected:
                for modality in ("vi", "ir"):
                    materialize(index, group, row, modality, args)
                counts[group] += 1
        print(ds, " ".join(f"{k}={v} pairs" for k, v in counts.items()), flush=True)
    if args.with_dev_inputs:
        prepare_dev_inputs(args.output)


if __name__ == "__main__":
    main()

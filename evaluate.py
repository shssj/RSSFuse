"""Evaluate the fixed 174-pair test split; verify input hashes before scoring."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import torch
from filelock import FileLock

from evaluation.metrics import read_gray, score
from evaluation.protocol import CONDITIONS, degrade, file_sha256, read_image, to_tensor
from inference import load_model, predict_pair


MANIFESTS = Path(__file__).resolve().parent / "evaluation" / "manifests"
DATASETS = {"FMB": 99, "LLVIP": 44, "M3FD": 31}
METRICS = ("SD", "MI", "EN", "CC", "SCD", "VIF", "vis_PSNR", "vis_SSIM", "ir_PSNR", "ir_SSIM")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def case_key(row):
    return row["dataset"], row["pair_id"], row["condition"]


def load_manifest():
    pairs = read_rows(MANIFESTS / "test_images.csv")
    images = {(r["dataset"], r["pair_id"]): r for r in pairs}
    cases = read_rows(MANIFESTS / "test_cases.csv")
    expected = {(ds, pid, c) for ds, pid in images for c in CONDITIONS}
    if (Counter(r["dataset"] for r in pairs) != DATASETS or len(pairs) != len(images)
            or set(map(case_key, cases)) != expected or len(cases) != len(expected)):
        raise ValueError("Invalid locked test split")
    return images, cases


@lru_cache(maxsize=4)
def load_source(path, mode, file_hash, pixel_hash, height, width):
    if file_sha256(path) != file_hash:
        raise ValueError(f"Source file differs from the locked image: {path}")
    pixels = read_image(path, mode)
    if pixels.shape[:2] != (height, width) or hashlib.sha256(pixels.tobytes()).hexdigest() != pixel_hash:
        raise ValueError(f"Decoded source differs from the locked image: {path}")
    return pixels, read_gray(path)


def make_case(root, row, case, locked_lq=None):
    batch, gray = {}, {}
    ds, pid, condition = case_key(case)
    for m, high, low, mode, mask in zip(("vi", "ir"), ("vh", "ih"), ("vl", "il"),
                                         ("RGB", "L"), CONDITIONS[condition]):
        path = root / ds / m / row[m]
        pixels, gray[m] = load_source(path, mode, row[m + "_sha256"], row[m + "_pixel_sha256"],
                                      int(row["height"]), int(row["width"]))
        lq, info = degrade(pixels, case["key"], m, mask)
        if info["seed"] != int(case[m + "_seed"]):
            raise ValueError(f"Wrong generation seed: {ds}/{pid}/{condition}/{m}")
        expected = case[m + "_lq_pixels_sha256"]
        if hashlib.sha256(lq.tobytes()).hexdigest() != expected:
            if locked_lq is not None:
                lq = read_image(locked_lq / condition / ds / f"{pid}_{m}.png", mode)
            if hashlib.sha256(lq.tobytes()).hexdigest() != expected:
                raise ValueError(f"Generated pixels differ from the lock: {ds}/{pid}/{condition}/{m}. "
                                 "Use an exact precomputed PNG via --locked-lq or match the generation environment.")
        batch[high], batch[low] = to_tensor(pixels), to_tensor(lq)
    digest = hashlib.sha256()
    for name in ("vh", "ih", "vl", "il"):
        digest.update(batch[name].contiguous().numpy().tobytes())
    if digest.hexdigest() != case["input_hash"]:
        raise ValueError(f"Input tensor differs from the archived run: {ds}/{pid}/{condition}")
    return batch, gray


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["condition"]].append(row)

    def mean(items):
        return {metric: statistics.mean(float(row[metric]) for row in items) for metric in METRICS}

    values = {}
    for ds, count in DATASETS.items():
        for cond in CONDITIONS:
            group = groups[ds, cond]
            if len(group) != count:
                raise ValueError(f"Incomplete group {ds}/{cond}")
            values[ds, cond] = mean(group)
        values[ds, "DEG9"] = mean([values[ds, c] for c in CONDITIONS if c != "CLEAN"])
        values[ds, "ALL10"] = mean([values[ds, c] for c in CONDITIONS])
    labels = ("DEG9", "ALL10", *CONDITIONS)
    for cond in labels:
        values["ALL", cond] = mean([values[ds, cond] for ds in DATASETS])
    return [dict(dataset=ds, condition=c, **values[ds, c]) for ds in ("ALL", *DATASETS) for c in labels]


def run(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    images, cases = load_manifest()
    if args.check_inputs or args.check_sample:
        checked = set()
        for c in cases:
            dataset, pid, condition = case_key(c)
            if args.check_sample and (dataset, condition) in checked:
                continue
            make_case(args.data_root, images[dataset, pid], c, args.locked_lq)
            checked.add((dataset, condition))
        print(f"Input checks passed for {len(checked) if args.check_sample else len(cases)} cases.")
        return
    if args.weights is None or args.output is None:
        raise ValueError("Evaluation requires --weights and --output.")
    args.output.mkdir(exist_ok=True, parents=True)
    with FileLock(str(args.output / ".evaluation.lock"), timeout=0):
        score_cases(args, images, cases)


def score_cases(args, images, cases):
    data_path = args.output / "per_image.csv"
    meta_path = args.output / "provenance.json"
    manifest_hash = hashlib.sha256(json.dumps([list(images.values()), cases], sort_keys=True).encode()).hexdigest()
    base = Path(__file__).resolve().parent
    code_files = [Path(__file__).resolve(), base / "evaluation" / "metrics.py",
                  base / "evaluation" / "protocol.py", base / "inference.py"]
    code_files.extend(sorted((base / "model_src").rglob("*.py")))
    meta = {"model": "RSSFuse", "weights_sha256": file_sha256(args.weights), "manifest_sha256": manifest_hash,
            "source_sha256": {p.relative_to(base).as_posix(): file_sha256(p) for p in code_files},
            "runtime": {"pytorch": str(torch.__version__), "device": str(args.device), "opencv": cv2.__version__}}
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != meta:
            raise ValueError("Output directory contains results from a different evaluation.")
    elif data_path.exists() or (args.output / "summary.csv").exists():
        raise ValueError("Output contains results with missing provenance; choose an empty output directory.")
    else:
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(meta_path)
    raw = read_rows(data_path) if data_path.exists() else []
    expected_hash = {case_key(c): c["input_hash"] for c in cases}
    done = set()
    for row in raw:
        key = case_key(row)
        if (key in done or row["input_hash"] != expected_hash.get(key)
                or not all(math.isfinite(float(row[m])) for m in METRICS)):
            raise ValueError(f"Invalid cached row {key}")
        done.add(key)
    model = load_model(args.weights, args.device, args.trusted_snapshot) if len(done) != len(cases) else None
    for case in cases:
        key = case_key(case)
        if key in done:
            continue
        b, gray = make_case(args.data_root, images[key[:2]], case, args.locked_lq)
        predicted = predict_pair(model, b["vl"], b["il"], args.device)
        values = score(predicted, b, gray["vi"], gray["ir"])
        raw.append(dict(dataset=key[0], pair_id=key[1], condition=key[2],
                        input_hash=case["input_hash"], **values))
        if len(raw) == 1 or len(raw) % 10 == 0 or len(raw) == len(cases):
            write_rows(data_path, raw)
            print(f"{len(raw)}/{len(cases)}", flush=True)
    write_rows(args.output / "summary.csv", summarize(raw))
    print(args.output / "summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Root containing FMB, LLVIP and M3FD")
    parser.add_argument("--weights", type=Path, help="RSSFuse weight file")
    parser.add_argument("--output", type=Path, help="Directory for per_image.csv, summary.csv and provenance.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--locked-lq", type=Path, help="Optional hash-verified exact PNGs for platform differences")
    parser.add_argument("--trusted-snapshot", action="store_true", help="Load a trusted full training snapshot")
    check = parser.add_mutually_exclusive_group()
    check.add_argument("--check-inputs", action="store_true", help="Check all 1,740 inputs, without a model")
    check.add_argument("--check-sample", action="store_true", help="Check one pair per dataset and condition")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Verify both per-image CSVs against the locked inputs, then compare all ten metrics.

The tolerance is absolute in each metric's own units. Finite values are required,
including in the reference. A NaN or either infinity always fails the check.
"""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path


MANIFEST = Path(__file__).resolve().parent / "manifests" / "test_cases.csv"
KEY = ("dataset", "pair_id", "condition")
METRICS = ("SD", "MI", "EN", "CC", "SCD", "VIF", "vis_PSNR", "vis_SSIM", "ir_PSNR", "ir_SSIM")
COUNTS = {"FMB": 99, "LLVIP": 44, "M3FD": 31}
CONDITIONS = {"VIS_lowlight", "VIS_haze", "VIS_noise", "IR_lowcontrast", "IR_stripes",
              "IR_noise", "VIS_compound", "IR_compound", "BOTH_compound", "CLEAN"}


def read_cases(path, metrics):
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        missing = set((*KEY, "input_hash", *metrics)) - set(header)
        if missing or len(header) != len(set(header)):
            raise ValueError(f"{path}: invalid CSV header (missing {sorted(missing)} or duplicate columns)")
        for row in reader:
            key = tuple(row[name] for name in KEY)
            if not all(key) or key in result or None in row:
                raise ValueError(f"{path}:{reader.line_num}: missing/duplicate key or malformed row: {key}")
            sha = row["input_hash"] or ""
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError(f"{path}: invalid input SHA-256 for {key}")
            values = {}
            for metric in metrics:
                try:
                    value = float(row[metric])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: invalid {metric} for {key}: {row[metric]!r}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path}: nonfinite {metric} for {key}: {row[metric]!r}")
                values[metric] = value
            result[key] = sha, values
    return result


def locked_cases():
    locked = read_cases(MANIFEST, ())
    ids = {(dataset, pair_id) for dataset, pair_id, _ in locked}
    expected = {(dataset, pair_id, condition) for dataset, pair_id in ids for condition in CONDITIONS}
    if set(locked) != expected or Counter(dataset for dataset, _ in ids) != COUNTS:
        raise ValueError(f"Incomplete locked case manifest: {MANIFEST}")
    return locked


def checked_scores(path, locked):
    scores = read_cases(path, METRICS)
    missing = locked.keys() - scores.keys()
    extra = scores.keys() - locked.keys()
    if missing or extra:
        raise ValueError(f"{path}: {len(missing)} missing and {len(extra)} unexpected cases; "
                         f"examples: missing={sorted(missing)[:3]}, unexpected={sorted(extra)[:3]}")
    for key, (sha, _) in scores.items():
        if sha != locked[key][0]:
            raise ValueError(f"{path}: input hash mismatch for {key}: {sha} (locked {locked[key][0]})")
    return scores


def compare(result_path, reference_path, atol):
    locked = locked_cases()
    actual = checked_scores(result_path, locked)
    expected = checked_scores(reference_path, locked)
    failures = 0
    maxima = dict.fromkeys(METRICS, 0.0)
    for key in sorted(locked):
        for metric in METRICS:
            value = actual[key][1][metric]
            target = expected[key][1][metric]
            delta = abs(value - target)
            maxima[metric] = max(maxima[metric], delta)
            if delta > atol:
                failures += 1
                if failures <= 12:
                    print(f"{'/'.join(key)} {metric}: got {value:.17g}, expected {target:.17g}, "
                          f"absolute difference {delta:.8g}")
    total = len(locked) * len(METRICS)
    if failures > 12:
        print(f"{failures - 12} additional differences exceeded tolerance.")
    print("Maximum absolute difference: " + ", ".join(f"{key}={value:.8g}" for key, value in maxima.items()))
    print(f"{len(locked)} matching case keys and input hashes; {total-failures}/{total} finite values "
          f"within absolute tolerance {atol:g}.")
    return failures == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_image", type=Path, help="per_image.csv produced by evaluate.py")
    parser.add_argument("--reference", required=True, type=Path, help="Archived per-image metric CSV")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="Absolute tolerance in each metric's own units (default: 0; no relative tolerance)")
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    try:
        return 0 if compare(args.per_image, args.reference, args.atol) else 1
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

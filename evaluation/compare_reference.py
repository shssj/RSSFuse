"""Compare seed-42, step-5000 locked results with the archived five-decimal report."""
import argparse
import csv
import math
from pathlib import Path


REFERENCE = Path(__file__).resolve().parent / "reference" / "seed42_step5000.csv"
METRICS = ("SD", "MI", "EN", "CC", "SCD", "VIF", "vis_PSNR", "vis_SSIM", "ir_PSNR", "ir_SSIM")


def read(path):
    scores = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["condition"])
            if key in scores:
                raise ValueError(f"Duplicate {key} in {path}")
            scores[key] = {m: float(row[m]) for m in METRICS}
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary.csv produced by evaluate.py")
    parser.add_argument("--atol", type=float, default=5.00001e-6,
                        help="Absolute tolerance in each metric's native units; default checks five printed decimals")
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    actual = read(args.summary)
    expected = read(REFERENCE)
    matched = total = 0
    for key, reference in expected.items():
        for metric, target in reference.items():
            total += 1
            score = actual.get(key, {}).get(metric, math.nan)
            if math.isfinite(score) and abs(score - target) <= args.atol:
                matched += 1
            else:
                print(f"{key[0]}/{key[1]} {metric}: got {score:.8g}, expected {target:.5f}, delta {score-target:+.8g}")
    print(f"{matched}/{total} archived values within absolute tolerance {args.atol:g}")
    return 0 if matched == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

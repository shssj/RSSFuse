"""Train RSSFuse, or recreate the teacher used to label its recovery increments."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher", "train"))
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset folders with vi/ and ir/ subfolders")
    parser.add_argument("--output", type=Path, required=True, help="A separate run directory for this stage")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--teacher", type=Path, help="Trusted teacher checkpoint, step 5000 of its 20000-step schedule")
    source.add_argument("--probe", type=Path, help="Archived, calibrated four-action probe for this seed")
    parser.add_argument("--initial-weights", type=Path, help="Archived step-zero student weights for this seed")
    parser.add_argument("--reference-trace", type=Path, help="CSV of all 5000 archived training batch fingerprints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, help="Shared generated training-image cache")
    parser.add_argument("--resume", action="store_true", help="Resume the newest checkpoint of this run")
    args = parser.parse_args()
    if (args.stage == "train") != (args.teacher is not None or args.probe is not None):
        parser.error("Supply either --teacher or --probe for the train stage only.")
    if args.initial_weights is not None and args.stage != "train":
        parser.error("--initial-weights is a student checkpoint for the train stage.")
    for path in (args.teacher, args.probe, args.initial_weights, args.reference_trace):
        if path is not None and not path.is_file():
            parser.error(f"File does not exist: {path}")
    from training.runner import run
    run(args)


if __name__ == "__main__":
    main()

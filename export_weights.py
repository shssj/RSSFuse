"""Convert an RSSFuse training checkpoint into a portable tensor-only state dictionary."""
import argparse
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


REFERENCE_CHECKPOINT = "6902412a30ad19e8460216a4ac81bc3ca83c03297ed4453daaf5780a9024d89b"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def export(source, destination, trusted=False):
    from model_src import RSSFuse

    if source.resolve() == destination.resolve():
        raise ValueError("Choose a separate path for the exported weights.")
    if destination.exists():
        raise FileExistsError(destination)
    # Original training snapshots include NumPy RNG state. Only unpickle the
    # known experiment file or a training snapshot the caller explicitly trusts.
    known = digest(source) == REFERENCE_CHECKPOINT
    payload = torch.load(source, map_location="cpu", weights_only=not (known or trusted))
    state = payload.get("model", payload)
    model = RSSFuse()
    model.load_state_dict(state, strict=True)
    result = model.state_dict()
    if any(not value.isfinite().all() for value in result.values()):
        raise ValueError("The input contains nonfinite weights.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as staged:
        temp_path = Path(staged.name)
    try:
        with temp_path.open("wb") as handle:
            torch.save(result, handle)
        reloaded = torch.load(temp_path, map_location="cpu", weights_only=True)
        if result.keys() != reloaded.keys() or any(not torch.equal(v, reloaded[k]) for k, v in result.items()):
            raise OSError("Export verification failed.")
        os.link(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"{digest(destination)}  {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trusted-snapshot", action="store_true",
                        help="Allow loading a training snapshot you created yourself.")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.trusted_snapshot)

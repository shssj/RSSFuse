"""Single-device FP32 training with the four-action probe and paired stripe loss."""
import csv
import hashlib
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from model_src import (
    SensorStripeConsistency, build_model, build_teacher, forward_with_gain,
    make_probe, restore_ir,
)
from .data import BATCH_SIZE, PAIR_MANIFEST, TrainingPairs, load_reference_trace
from .losses import FusionLoss, restoration_loss, routed_backward


STEPS = 5000
LOG_FIELDS = ["step", "recovery", "fusion", "gain_bce", "stripe_reference",
              "stripe_consistency", "lr", "preclip_norm", "sample_ids", "batch_sha256"]


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(obj, path):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def to_gpu(batch, start=0, count=BATCH_SIZE):
    return {key: value[start:start+count].cuda() for key, value in batch.items()}


def state_digest(state):
    """Match the historical runner's initialization checksum across archive formats."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_initial_weights(model, path, seed):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("step") != 0 or checkpoint.get("seed") != seed:
        raise ValueError("Initial student weights require step zero and the same seed as this run")
    model.load_state_dict(checkpoint["model"], strict=True)
    digest = state_digest(model.state_dict())
    if digest != checkpoint.get("state_sha256"):
        raise ValueError("Student initialization differs from the exported state_sha256")
    return digest


def restore_probe(probe, path, seed, teacher_digest=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    source = state.get("teacher_sha256", state.get("teacher_sha"))
    if (state.get("seed") != seed or not isinstance(source, str) or len(source) != 64
            or any(c not in "0123456789abcdef" for c in source)
            or (teacher_digest is not None and source != teacher_digest)):
        raise ValueError("Saved probe belongs to a different teacher or seed, or lacks its source checksum")
    weights = state["model"] if "model" in state else state["weights"]
    probe.load_state_dict(weights, strict=True)
    if probe.action_counts.tolist() != [32] * 4:
        raise ValueError("Expected 32 train-only samples for each probe action")


def frozen_probe(teacher_path, seed, data, loss, folder, teacher_digest, probe_path=None):
    teacher = build_teacher(seed)
    if probe_path is None:
        # The original, trusted full teacher checkpoint includes optimizer/RNG metadata.
        snap = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher_seed = snap.get("seed", snap.get("protocol", {}).get("common", {}).get("seed"))
        if snap.get("step") != 5000 or teacher_seed != seed or snap.get("scheduler", {}).get("T_max") != 20000:
            raise ValueError("Teacher requires the same seed and checkpoint 5000 of the 20000-step cosine schedule")
        teacher.load_state_dict(snap["model"], strict=True)
        del snap
    probe = make_probe(teacher, loss.cpu()).cuda()
    loss.cuda()
    del teacher
    cached = folder / "training_probe.pth"
    if probe_path is not None:
        # The archive contains the full frozen reference, calibrated heads and their action policy.
        restore_probe(probe, probe_path, seed)
    elif cached.is_file():
        restore_probe(probe, cached, seed, teacher_digest)
    else:
        batches = (to_gpu(data.batch(step)[0], 0, 2) for step in range(1, 17))
        probe.calibrate(batches, split="train", lr=1e-4, max_steps=16)
        if probe.action_counts.tolist() != [32] * 4:
            raise ValueError("Expected 32 train-only samples for each probe action")
        atomic_save(dict(seed=seed, teacher_sha256=teacher_digest, model=probe.state_dict()), cached)
    return probe


def train_stripes(model, batch, seed, step):
    augment = SensorStripeConsistency(sigma_255=(4, 10))
    slot = (step // 4) % len(batch["ih"])
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed * 1000000 + step)
    turn = 1 if (step // 4) % 2 == 0 else 3
    pair = augment.make_pair(batch["vh"][slot:slot+1].to(device), batch["ih"][slot:slot+1].to(device),
                             quarter_turns=turn, noise_mode="fixed", generator=generator)
    _, transported = pair.transported_reference()
    a, b = restore_ir(model, torch.cat((pair.ir_scene, transported))).chunk(2)
    consistency = augment.loss(b, a, pair, reference_is_transported=True)
    target = 0.5 * ((a - pair.target_scene).abs().mean() + (b - pair.target_scene).abs().mean())
    (0.1 * target + 0.05 * consistency).backward()
    return target.item(), consistency.item()


def prepare_log(path, step, reference_trace=None):
    """A process can stop after logging a step and before saving its checkpoint."""
    rows = []
    if step:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != LOG_FIELDS:
                raise ValueError("Training log header differs from this release")
            rows = list(itertools.islice(reader, step))
        if len(rows) != step or any(row["step"] != str(i) for i, row in enumerate(rows, 1)):
            raise ValueError("Training log is incomplete before the latest checkpoint")
        if reference_trace is not None:
            for row, (fingerprint, names) in zip(rows, reference_trace):
                if row["batch_sha256"] != fingerprint or (names is not None and row["sample_ids"] != names):
                    raise ValueError(f"Persisted training log differs from the archived stream at step {row['step']}")
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def train(args):
    student = args.stage == "train"
    teacher_digest = file_digest(args.teacher) if args.teacher is not None else None
    config = dict(model="RSSFuse", stage=args.stage, seed=args.seed, steps=STEPS,
                  cosine_steps=STEPS if student else 20000, effective_batch=BATCH_SIZE,
                  microbatch=4 if student else 8, crop=256, lr=0.0001, lr_min=0.000001,
                  betas=[0.9, 0.99], weight_decay=0, grad_clip=1, fp32=True,
                  lambda_rec=0.5, fusion_coefficients=[1, 10, 1], vis_fusion_gradient_scale=0.25,
                  teacher_sha256=teacher_digest, train_ids_sha256=file_digest(PAIR_MANIFEST))
    if student:
        config.update(gain_interval=8, gain_weight=0.01, sccl_interval=4, sccl_first_step=2,
                      stripe_target_weight=0.1, stripe_consistency_weight=0.05)
    for key in ("initial_weights", "probe", "reference_trace"):
        path = getattr(args, key)
        if path is not None:
            config[f"{key}_sha256"] = file_digest(path)
    trace = load_reference_trace(args.reference_trace, STEPS, args.seed) if args.reference_trace else None
    snapshots = sorted((p for p in args.output.glob("*.pth") if p.stem.isdecimal()),
                       key=lambda path: int(path.stem))
    config_file = args.output / "config.json"
    if config_file.exists():
        if not args.resume:
            raise FileExistsError(f"{args.output} contains a previous run. Choose a fresh folder or --resume.")
        if json.loads(config_file.read_text(encoding="utf-8")) != config:
            raise ValueError("The requested configuration differs from the existing run")
    elif args.resume or snapshots or (args.output / "train.csv").exists():
        raise ValueError("A matching run configuration is required to resume")
    else:
        temp = config_file.with_suffix(".tmp")
        temp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        temp.replace(config_file)
    data = TrainingPairs(args.data_root, args.cache_dir or args.output / "image_cache", args.seed, trace)
    fusion_loss = FusionLoss().cuda()
    if student:
        probe = frozen_probe(args.teacher, args.seed, data, fusion_loss, args.output, teacher_digest, args.probe)
        model = build_model(args.seed)
    else:
        probe = None
        model = build_teacher(args.seed)
    if args.initial_weights is None:
        initial_digest = state_digest(model.state_dict())
    else:
        initial_digest = load_initial_weights(model, args.initial_weights, args.seed)
    print(f"Initial model state SHA-256: {initial_digest}", flush=True)
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.99), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["cosine_steps"], eta_min=1e-6)
    step_done = 0
    if snapshots:
        state = torch.load(snapshots[-1], map_location="cpu", weights_only=False)
        step_done = state["step"]
        if (step_done != int(snapshots[-1].stem) or step_done != state["scheduler"]["last_epoch"]
                or state["config"] != config):
            raise ValueError("Checkpoint and recorded configuration disagree")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        del state
    log = args.output / "train.csv"
    prepare_log(log, step_done, trace)
    with log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for step in range(step_done+1, STEPS+1):
            cpu, names, fingerprint = data.batch(step)
            optimizer.zero_grad(set_to_none=True)
            rec_sum = fuse_sum = gain_sum = stripe_l1 = stripe_consistency = 0.
            micro = config["microbatch"]
            for pos in range(0, BATCH_SIZE, micro):
                batch = to_gpu(cpu, pos, micro)
                if student and step % 8 == 0:
                    output, gain, _ = forward_with_gain(model, batch, fusion_loss, probe)
                else:
                    output = model(batch["vl"], batch["il"], "C0")
                    gain = output["fused"].new_zeros(())
                rec = restoration_loss(output, batch)
                fuse, _ = fusion_loss(output["fused"], batch["vh"], batch["ih"])
                if not torch.isfinite(rec + fuse + gain):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                weight = micro / BATCH_SIZE
                routed_backward(model, rec * weight, (fuse + 0.01 * gain) * weight)
                rec_sum += weight * rec.item()
                fuse_sum += weight * fuse.item()
                gain_sum += weight * gain.item()
                del output, rec, fuse, gain, batch
            if student and step % 4 == 2:
                stripe_l1, stripe_consistency = train_stripes(model, cpu, args.seed, step)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            writer.writerow([step, rec_sum, fuse_sum, gain_sum, stripe_l1, stripe_consistency,
                             lr, norm, ";".join(names), fingerprint])
            if step <= 3 or step % 25 == 0:
                print(f"{step}/{STEPS}  recovery={rec_sum:.5f}  fusion={fuse_sum:.5f}", flush=True)
                handle.flush()
            if step == 3 or step % 500 == 0:
                handle.flush()
                os.fsync(handle.fileno())
                atomic_save(dict(step=step, seed=args.seed, model=model.state_dict(), config=config,
                                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                 rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                                 rng_numpy=np.random.get_state(), rng_python=random.getstate()),
                            args.output / f"{step}.pth")
    if student:
        atomic_save(dict(model=model.state_dict()), args.output / "RSSFuse.pth")
    print(f"Finished. Outputs: {args.output}", flush=True)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training needs a CUDA GPU and the selective-scan extension")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".training.lock"), timeout=0):
        train(args)

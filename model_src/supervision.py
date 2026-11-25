"""Training-only gain supervision and sensor-coordinate stripe consistency."""

import math
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .fusion import DeltaGainWriter

ACTIONS = ((0, 0), (1, 0), (0, 1), (1, 1))


@contextmanager
def capture_writes(writers):
    """Capture the actual arguments of the most recent invocation per scale."""
    records = {}
    handles = []

    def save(scale):
        def hook(_module, args):
            records[scale] = args

        return hook

    try:
        for k, module in enumerate(writers):
            handles.append(module.register_forward_pre_hook(save(k)))
        yield records
    finally:
        for h in handles:
            h.remove()


def writer_supervision(model, last_inputs, utilities, *, margin):
    """BCE on the sign of per-image single-modality gains from final-round Q.

    Three scales receive ONE image-level prediction loss on the mean logit.
    Labels with |gain| <= the preselected margin carry zero weight.
    """
    if margin < 0 or not all(isinstance(w, DeltaGainWriter) for w in model.u):
        raise ValueError("Use a nonnegative margin and enable delta_gain first")
    logits = torch.stack(
        [w.logits(*last_inputs[k][1:]) for k, w in enumerate(model.u)]
    ).mean(0)
    gain = torch.stack((utilities["vis"], utilities["ir"]), dim=-1).detach().to(logits)
    if gain.shape != logits.shape or not torch.isfinite(gain).all():
        raise ValueError("Expected finite per-image VIS/IR utilities")
    valid = (gain.abs() > margin).to(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits, (gain > 0).to(logits), reduction="none"
    )
    return (bce * valid).sum() / valid.sum().clamp_min(1)


def require_training_split(split):
    if split != "train":
        raise ValueError(
            "Clean-source proxy generation/calibration requires the training split"
        )


def gain_from_quality(quality):
    q00, q10, q01, q11 = (quality[action] for action in ACTIONS)
    return {"vis": q10 - q00, "ir": q01 - q00, "interaction": q11 - q10 - q01 + q00}


class FourActionProbe(nn.Module):
    """Frozen reference trajectory + independently calibrated final U/decoder.

    Source loss must be a fixed deterministic callable returning (loss, details),
    as in original_losses.FusionLoss. Provide a separate fixed reference for each
    training seed/arm if that is part of the chosen comparison protocol.
    """

    def __init__(self, reference_model, source_loss, *, policy="delta_only"):
        super().__init__()
        if reference_model.rounds != 2 or any(
            isinstance(x, DeltaGainWriter) for x in reference_model.u
        ):
            raise ValueError("Provide a frozen reference checkpoint with two rounds")
        if policy not in ("delta_only", "coherent_state"):
            raise ValueError(
                "Use delta_only for writer targets or coherent_state for a coherent-state diagnostic"
            )
        self._policy = policy
        self.reference = deepcopy(reference_model).eval().requires_grad_(False)
        self.read_u = deepcopy(reference_model.u).eval().requires_grad_(False)
        self.read_df = deepcopy(reference_model.df).eval().requires_grad_(False)
        self.source_loss = deepcopy(source_loss).eval().requires_grad_(False)
        self.register_buffer("action_counts", torch.zeros(4, dtype=torch.long))
        self.eval()

    @property
    def policy(self):
        return self._policy

    def get_extra_state(self):
        return {"policy": self.policy, "round": 2}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError(
                "Probe checkpoint action policy/round does not match this instance"
            )

    @torch.no_grad()
    def _context(self, batch):
        ref = self.reference
        if self.training or ref.training:
            raise RuntimeError("The reference trajectory must use eval mode")
        vis, ir, shared = ref.initial_states(batch["vl"], batch["il"])
        vis, ir, shared = ref.step(vis, ir, shared, "C0")
        with capture_writes(ref.u) as calls:
            ref.step(vis, ir, shared, "C0")
        # Preserve the native new-state and delta arguments of the all-new path.
        # In particular adapted VIS delta = dv + (correction_new-correction_old).
        return [
            (vis[k] + ref.correction(k, vis[k]), ir[k], *calls[k])
            for k in range(len(ref.u))
        ]

    def _read(self, context, action):
        v_new, i_new = action
        next_shared = []
        for u, (old_vis, old_ir, shared, vis, ir, dvis, dir_) in zip(
            self.read_u, context
        ):
            # For delta_only, the new states are held fixed, exactly like the
            # student's inference gate. coherent_state measures a different effect.
            fixed_new = self.policy == "delta_only"
            delta = u(
                shared,
                vis if fixed_new or v_new else old_vis,
                ir if fixed_new or i_new else old_ir,
                dvis if v_new else torch.zeros_like(dvis),
                dir_ if i_new else torch.zeros_like(dir_),
            )
            next_shared.append(shared + delta)
        return self.read_df(next_shared).sigmoid()

    def calibrate(self, batches, *, split, lr, max_steps):
        """Apply balanced updates to separate probe weights on train-only pairs.

        The reference, student and published checkpoint are never updated here.
        Repeat this same schedule/data for matched control arms.
        """
        require_training_split(split)
        if max_steps < 1 or lr <= 0 or not math.isfinite(lr):
            raise ValueError("Provide a positive calibration budget and learning rate")
        self.eval()
        modules = (self.read_u, self.read_df)
        for module in modules:
            module.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [p for m in modules for p in m.parameters()], lr=lr, weight_decay=0
        )
        steps = 0
        try:
            for batch in batches:
                context = self._context(batch)
                optimizer.zero_grad(set_to_none=True)
                for action in ACTIONS:
                    fused = self._read(context, action)
                    loss, _ = self.source_loss(fused, batch["vh"], batch["ih"])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite action calibration loss")
                    (loss / len(ACTIONS)).backward()
                optimizer.step()
                self.action_counts.add_(len(batch["vl"]))
                steps += 1
                if steps == max_steps:
                    break
        finally:
            for module in modules:
                module.zero_grad(set_to_none=True)
                module.requires_grad_(False)
        if steps != max_steps:
            raise ValueError(
                f"Requested {max_steps} calibration batches but received {steps}"
            )
        return steps

    @torch.no_grad()
    def score(self, batch, *, split):
        require_training_split(split)
        if not torch.all(self.action_counts > 0):
            raise RuntimeError(
                "All four actions require balanced train-only calibration"
            )
        context = self._context(batch)
        scores = {}
        for action in ACTIONS:
            fused = self._read(context, action)
            # Original FusionLoss averages over the batch; obtain one Q per image.
            scores[action] = -torch.stack(
                [
                    self.source_loss(
                        fused[j : j + 1], batch["vh"][j : j + 1], batch["ih"][j : j + 1]
                    )[0]
                    for j in range(len(fused))
                ]
            )
            if not torch.isfinite(scores[action]).all():
                raise ValueError("Nonfinite source proxy quality")
        return scores, gain_from_quality(scores)

    def training_utilities(self, batch, *, split):
        if self.policy != "delta_only":
            raise ValueError(
                "Coherent state+delta effects are diagnostics, not explicit-delta gate targets"
            )
        return self.score(batch, split=split)[1]


def _turn(image, quarter_turns):
    return torch.rot90(image, quarter_turns, dims=(-2, -1))


@dataclass
class StripePair:
    vis_ref: torch.Tensor
    vis_scene: torch.Tensor
    ir_ref: torch.Tensor
    ir_scene: torch.Tensor
    target_ref: torch.Tensor
    target_scene: torch.Tensor
    bias_ref: torch.Tensor
    bias_scene: torch.Tensor
    valid_ref: torch.Tensor
    valid_scene: torch.Tensor
    quarter_turns: int
    noise_mode: str

    def transported_reference(self):
        """Reference input rotated into the second scene's coordinates: T(x+b)."""
        return _turn(self.vis_ref, self.quarter_turns), _turn(
            self.ir_ref, self.quarter_turns
        )


def _masked_l1(difference, valid):
    weight = valid.expand_as(difference)
    return (difference.abs() * weight).sum() / weight.sum().clamp_min(1)


class SensorStripeConsistency:
    """Sensor-coordinate stripe pairs and consistency loss. Inputs use unit intensity.

    'fixed' reuses the sensor's column offsets for the rotated scene (square
    crops or compatible widths); 'independent' resamples those offsets in the
    sensor frame. 'transported' rotates the first observation and its noise
    together, serving as the ordinary augmentation/equivariance control.
    """

    def __init__(self, sigma_255=(4.0, 10.0)):
        if len(sigma_255) != 2 or not 0 <= sigma_255[0] <= sigma_255[1]:
            raise ValueError("sigma_255 must be a nonnegative ordered pair")
        self.sigma_255 = tuple(sigma_255)

    def make_pair(
        self,
        vis,
        ir_base,
        *,
        quarter_turns=1,
        noise_mode="fixed",
        generator=None,
        valid=None,
    ):
        if (
            ir_base.ndim != 4
            or ir_base.shape[1] != 1
            or not ir_base.is_floating_point()
        ):
            raise ValueError(
                "ir_base must contain clean floating IR crops [B, 1, H, W]"
            )
        if (
            vis.ndim != 4
            or vis.shape[0] != ir_base.shape[0]
            or vis.shape[2:] != ir_base.shape[2:]
        ):
            raise ValueError(
                "all VIS channels must be paired spatially with the clean IR"
            )
        if noise_mode not in {"fixed", "independent", "transported"}:
            raise ValueError("unknown stripe noise mode")
        if not isinstance(quarter_turns, int):
            raise ValueError("quarter_turns must be an integer")
        k = quarter_turns % 4
        target_scene = _turn(ir_base, k)
        if noise_mode == "fixed" and target_scene.shape[-1] != ir_base.shape[-1]:
            raise ValueError(
                "fixed sensor offsets require equal sensor widths; use square crops for 90 degrees"
            )
        if valid is None:
            valid = torch.ones_like(ir_base, dtype=torch.bool)
        elif valid.shape != ir_base.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean IR mask shaped [B, 1, H, W]")
        sampling = {
            "device": ir_base.device,
            "dtype": torch.float32,
            "generator": generator,
        }
        batch = ir_base.shape[0]
        levels = (
            torch.empty((batch, 1, 1, 1), device=ir_base.device).uniform_(
                *self.sigma_255, generator=generator
            )
            / 255
        )

        def columns(width):
            offset = torch.randn((batch, 1, 1, width), **sampling)
            return ((offset - offset.mean(-1, keepdim=True)) * levels).to(ir_base.dtype)

        bias_ref = columns(ir_base.shape[-1]).expand_as(ir_base)
        if noise_mode == "transported":
            bias_scene = _turn(bias_ref, k)
        elif noise_mode == "independent":
            bias_scene = columns(target_scene.shape[-1]).expand_as(target_scene)
        else:
            bias_scene = bias_ref[:, :, :1, :].expand_as(target_scene)
        raw_ref, raw_scene = ir_base + bias_ref, target_scene + bias_scene
        valid_ref = valid & (raw_ref >= 0) & (raw_ref <= 1)
        valid_scene = _turn(valid, k) & (raw_scene >= 0) & (raw_scene <= 1)
        return StripePair(
            vis,
            _turn(vis, k),
            raw_ref.clamp(0, 1),
            raw_scene.clamp(0, 1),
            ir_base,
            target_scene,
            bias_ref,
            bias_scene,
            valid_ref,
            valid_scene,
            k,
            noise_mode,
        )

    @staticmethod
    def loss(restored_ref, restored_scene, pair, *, reference_is_transported=False):
        """Paired content consistency only; retain ordinary supervised losses on both views.

        For the proposal's matched T(x+b) versus T(x)+b, feed both through the
        model in the rotated frame and set reference_is_transported=True.
        For x+b versus T(x)+b this method first undoes the scene rotation.
        """
        if reference_is_transported:
            restored_ref = _turn(restored_ref, -pair.quarter_turns)
        aligned = _turn(restored_scene, -pair.quarter_turns)
        if (
            aligned.shape != pair.target_ref.shape
            or restored_ref.shape != aligned.shape
        ):
            raise ValueError(
                "IR outputs must match their respective full-resolution inputs"
            )
        valid = pair.valid_ref & _turn(pair.valid_scene, -pair.quarter_turns)
        return _masked_l1(restored_ref - aligned, valid)


def make_probe(reference_model, fusion_loss):
    """Independent frozen teacher; calibrate its copied readout on train data."""
    probe = FourActionProbe(reference_model, fusion_loss, policy="delta_only")
    return probe.to(next(reference_model.parameters()).device)


def forward_with_gain(model, batch, fusion_loss, probe, *, margin=1e-7):
    """One student forward plus train-only final-round, four-action supervision.

    Signs of Q10 - Q00 and Q01 - Q00 label image-level single-modality gains.
    Q is negative per-image original FusionLoss. The interaction is recorded
    separately. Each scale learns two gates, with supervision on the mean of
    final-round logits across scales. Thus the labels do not claim to measure
    per-scale, per-pixel, or first-round gains.
    """
    if tuple(fusion_loss.weights) != tuple(probe.source_loss.weights):
        raise ValueError(
            "Student and teacher must use the same fusion-loss coefficients"
        )
    # Score without activation tapes before constructing the student's graph.
    quality, utilities = probe.score(batch, split="train")
    with capture_writes(model.u) as writes:
        out = model(batch["vl"], batch["il"], "C0")
    aux = writer_supervision(model, writes, utilities, margin=margin)
    with torch.no_grad():
        weights = torch.stack(
            [w.write_weights(*writes[k][1:]) for k, w in enumerate(model.u)]
        )
        gains = torch.stack((utilities["vis"], utilities["ir"]), dim=1)
        stats = {
            **{f"q{v}{i}": float(score.mean()) for (v, i), score in quality.items()},
            "gain_vis": float(gains[:, 0].mean()),
            "gain_ir": float(gains[:, 1].mean()),
            "interaction": float(utilities["interaction"].mean()),
            "valid_labels": int((gains.abs() > margin).sum()),
            "gate_vis": float(weights[:, :, 0].mean()),
            "gate_ir": float(weights[:, :, 1].mean()),
        }
    return out, aux, stats


def restore_ir(model, il):
    """C0 infrared path, shared in both SCCL views, with full image residual."""
    states = model.ei(il)
    for _ in range(model.rounds):
        states = [
            state + update(state, project(state))
            for state, project, update in zip(states, model.ai, model.ri)
        ]
    return il + model.hi(states)

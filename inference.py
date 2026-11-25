"""Run RSSFuse on a registered 8-bit visible/infrared image pair."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from evaluation.protocol import read_image, to_tensor, uint8_image


def load_model(weights, device, trusted_snapshot=False):
    from model_src import build_model

    checkpoint = torch.load(weights, map_location="cpu", weights_only=not trusted_snapshot)
    state = checkpoint.get("model", checkpoint)
    model = build_model(seed=42)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_pair(model, visible, infrared, device):
    """Full-frame FP32 prediction. Only non-multiple-of-four inputs get edge padding."""
    if visible.shape[0] != 3 or infrared.shape[0] != 1 or visible.shape[1:] != infrared.shape[1:]:
        raise ValueError("Expected aligned RGB visible and grayscale infrared images of the same size.")
    height, width = visible.shape[-2:]
    vis = visible.unsqueeze(0).to(device)
    ir = infrared.unsqueeze(0).to(device)
    padding = (0, -width % 4, 0, -height % 4)
    if any(padding):
        vis, ir = F.pad(vis, padding, mode="replicate"), F.pad(ir, padding, mode="replicate")
    outputs = model(vis, ir)
    return {name: outputs[name][0, :, :height, :width] for name in ("fused", "vis", "ir")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--visible", type=Path, required=True)
    parser.add_argument("--infrared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for three PNG images")
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda:0")
    parser.add_argument("--trusted-snapshot", action="store_true",
                        help="Allow a trusted full training snapshot containing optimizer/RNG objects")
    args = parser.parse_args()
    vis = to_tensor(read_image(args.visible, "RGB"))
    ir = to_tensor(read_image(args.infrared, "L"))
    model = load_model(args.weights, args.device, args.trusted_snapshot)
    outputs = predict_pair(model, vis, ir, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for key, filename in (("fused", "fused.png"), ("vis", "restored_visible.png"), ("ir", "restored_ir.png")):
        Image.fromarray(uint8_image(outputs[key])).save(args.output / filename)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

# RSSFuse

RSSFuse takes a registered RGB visible image and a grayscale infrared image of the same scene. It predicts a fused RGB image, a restored visible image and a restored infrared image. The fusion decoder reads a shared feature state that receives information from both restoration branches twice.

**2,386,148 trainable parameters · two updates · three feature scales (32 / 64 / 128 channels).**

[Reproduce the seed-42 experiment](REPRODUCE.md) · [中文模型与复现实验说明](docs/模型与复现.md) · [Evaluation and test data](evaluation/README.md) · [Component origins and licenses](THIRD_PARTY.md)

## Install

The reference training environment used Linux, Python 3.9.25, PyTorch 2.1.0+cu121, Mamba 1.0.1 and causal-conv1d 1.0.0. Create a fresh environment, then install the packages from the repository root:

```bash
conda create -n rssfuse python=3.9.25 -y
conda activate rssfuse
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For GPU training and inference, install the native extensions matching your operating system, Python, PyTorch, CUDA and C++ ABI. The original environment used a Linux x86-64 CPython 3.9 Mamba wheel built for PyTorch 2.1. A source build requires a matching CUDA toolkit. See [Mamba releases](https://github.com/state-spaces/mamba/releases/tag/v1.0.1) and [causal-conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases).

```bash
python -m pip install packaging ninja wheel transformers==4.35.2 tokenizers==0.15.0 huggingface-hub==0.19.4 safetensors==0.4.0
python -m pip install causal-conv1d==1.0.0 --no-build-isolation
python -m pip install mamba-ssm==1.0.1 --no-build-isolation
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn"
```

For Linux x86-64, CPython 3.9, PyTorch 2.1 and the old C++ ABI, the authors also publish matching CUDA 12 wheel builds. After installing the Python dependencies above, the following two wheel installs can replace the two source installs. Check the ABI with `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` (expected `False` for these wheels).

```bash
python -m pip install --no-deps "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.0.0/causal_conv1d-1.0.0%2Bcu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
python -m pip install --no-deps "https://github.com/state-spaces/mamba/releases/download/v1.0.1/mamba_ssm-1.0.1%2Bcu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn"
```

For a small CPU check, install the CPU build of torch 2.1.0 and the packages in `requirements.txt`. The CPU path implements the scan recurrence directly and is slow on full-resolution images. The following checks a fresh model and the three output shapes without requiring data or weights:

```bash
python -c "import torch; from model_src import RSSFuse; torch.set_num_threads(2); m=RSSFuse().eval(); print(sum(p.numel() for p in m.parameters())); print({k:tuple(v.shape) for k,v in m(torch.zeros(1,3,16,16), torch.zeros(1,1,16,16)).items()})"
```

## Fuse an image pair

To check a fresh checkout before supplying any data, run `python verify_release.py`. It verifies the archived weights and training files, then makes one CPU prediction.

Use the included [inference weights](weights/RSSFuse.pth) from the seed-42 run at step 5,000. Supply your own aligned image pair:

```bash
python inference.py --weights weights/RSSFuse.pth --visible visible.png --infrared infrared.png --output outputs/pair
```

The output folder contains `fused.png`, `restored_visible.png`, and `restored_ir.png`. Use aligned, eight-bit source images of equal height and width. For CPU inference on a small crop, append `--device cpu`.

| Seed-42 artifact | SHA-256 |
|---|---|
| Original full training snapshot | `6902412a30ad19e8460216a4ac81bc3ca83c03297ed4453daaf5780a9024d89b` |
| Reference tensor-only export, `RSSFuse.pth` (9,745,938 bytes) | `169c2ad386e22688a94885fe2af7dcd3804f3273fbf563370b3fd0495bad6351` |

An independent reserialization can change the file digest while preserving every tensor value.

## Train

Obtain the paired images from the dataset providers. The input root contains `FMB/{vi,ir}/`, `LLVIP/{vi,ir}/` and `M3FD/{vi,ir}/`. Training selects 150 pairs per dataset using the IDs and decoded-pixel hashes in [the training manifest](training/train_pairs.csv). For the locked test, preserve the exact filenames, extensions and encoded files recorded in [the test manifest](evaluation/manifests/test_images.csv). Dataset sources: [FMB](https://github.com/JinyuanLiu-CV/SegMiF), [LLVIP](https://github.com/bupt-ai-cz/LLVIP), [M³FD](https://github.com/JinyuanLiu-CV/TarDAL).

```bash
python train.py train --data-root data/train --output runs/student --seed 42 \
  --initial-weights reproduction/initial.pth --probe reproduction/probe.pth \
  --reference-trace reproduction/train_trace.csv
```

This command starts at the archived student initialization, loads the calibrated gain probe and checks each of the 5,000 training batches against its original fingerprint. The included training artifacts belong to seed 42. The student writes `runs/student/RSSFuse.pth`. See the [reproduction guide](REPRODUCE.md) for data preparation, reference metrics, and optional teacher recalibration or retraining.

| Student setting | Value |
|---|---|
| Input and batch | Paired 256 × 256 crops, synchronized rotation and flipping, FP32, effective batch 8 (two microbatches of 4) |
| Optimizer | AdamW, β = (0.9, 0.99), weight decay 0, gradient norm clipped at 1 |
| Learning rate | Cosine decay from 10⁻⁴ to 10⁻⁶ |
| Main objective | 0.5 × recovery L1 + fusion loss. Intensity, gradient and color have weights 1, 10 and 1. The fusion gradient entering the VIS restoration parameters is scaled by 0.25. |
| Gain supervision | Every eighth step, a frozen four-action probe supplies modality-level labels. BCE weight 0.01 on active steps. |
| Stripe consistency | Steps 2, 6, 10, …, one paired IR example, reference L1 weight 0.10 and consistency weight 0.05. |

## Evaluate

The [RSSFuse results](evaluation/reference/summary.csv) report six fusion and four restoration metrics for each dataset and condition. The [per-image table](evaluation/reference/per_image.csv) records all 1,740 predictions and the input fingerprint for each one.

```bash
python evaluate.py --data-root data/test --check-inputs
python evaluate.py --data-root data/test --weights weights/RSSFuse.pth --output results/locked_test
python evaluation/compare_reference.py results/locked_test/summary.csv
```

The locked test contains 174 pairs and ten conditions per pair. The evaluator writes per-image scores, per-dataset summaries and run provenance. The **DEG9** figures below are from the original seed-42, step-5,000 checkpoint. Images are averaged within each dataset and condition, then the nine degraded conditions and the three datasets receive equal weight. CLEAN is counted separately in ALL10. These are single-seed reference values.

| Dataset | SD ↑ | MI ↑ | EN ↑ | CC ↑ | SCD ↑ | VIF ↑ |
|---|---:|---:|---:|---:|---:|---:|
| FMB | 35.51714 | 3.21714 | 6.70139 | 0.58954 | 1.48749 | 0.66661 |
| LLVIP | 48.43830 | 2.78615 | 7.34587 | 0.68682 | 1.50147 | 0.67654 |
| M³FD | 35.53997 | 3.22277 | 6.94294 | 0.40888 | 1.52859 | 0.59070 |
| Three-dataset mean | 39.83180 | 3.07535 | 6.99673 | 0.56175 | 1.50585 | 0.64462 |

## Code map

| Path | Contents |
|---|---|
| [model_src/network.py](model_src/network.py) | Public model class, reference backbone and builders |
| [model_src/backbone.py](model_src/backbone.py) | Initialization, two-round update and three readouts |
| [model_src/restoration.py](model_src/restoration.py), [modules.py](model_src/modules.py) | Encoder calibration, adaptive recovery step, pyramid encoders, VSS recovery blocks and decoders |
| [model_src/fusion.py](model_src/fusion.py) | Shallow selective prototype read, detail-biased initialization and gated increment writing |
| [model_src/supervision.py](model_src/supervision.py) | Training-only four-action probe and paired stripe constraint |
| [training/](training/), [train.py](train.py) | Teacher and student training, augmentation, loss routing, logging and resume |
| [inference.py](inference.py), [evaluate.py](evaluate.py), [evaluation/](evaluation/) | Single-pair inference and locked evaluation |

Upstream source files and adaptations retain their respective notices, including the MIRNet-v2 academic noncommercial terms for the SKFF component. See [component provenance and notices](THIRD_PARTY.md).

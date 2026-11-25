# Reproducing RSSFuse at seed 42

Use the bundled weights to recompute the reported scores, or train the student from its archived initialization. Both routes use the same locked test inputs. Run commands from the repository root.

## 1. Check the artifacts and runtime

```bash
sha256sum -c reproduction/assets.sha256
```

The command should print `OK` for each listed file. The released inference weights, `weights/RSSFuse.pth`, are 9,745,938 bytes and have SHA-256 `169c2ad386e22688a94885fe2af7dcd3804f3273fbf563370b3fd0495bad6351`. Use this file for prediction. The reproduction directory contains a step-zero student, a calibrated four-action probe, the training teacher and the 5,000-step batch trace. `assets.sha256` identifies the packaged files. `archive_hashes.json` records the original archive fingerprints. Original training snapshots contained additional state and can have different file hashes.

Follow the GPU installation commands in the [README](README.md#install). The archived student used a single NVIDIA A100, FP32, Python 3.9.25, PyTorch 2.1.0+cu121, mamba-ssm 1.0.1 and causal-conv1d 1.0.0. The data and metric implementations used NumPy 1.23.5, Pillow 9.3.0 and OpenCV 4.6.0.66. Test the native scan import before starting a GPU run. The CPU recurrence supports small functional checks.

After installing the Python requirements, run `python verify_release.py` from the repository root on Linux or Windows. This verifies seven files, loads the student, teacher and calibrated probe, checks the training trace and runs a small CPU prediction with the released model. It requires no source photographs or GPU.

## 2. Prepare the source images

Obtain the original visible and infrared images from the [FMB](https://github.com/JinyuanLiu-CV/SegMiF), [LLVIP](https://github.com/bupt-ai-cz/LLVIP) and [M³FD](https://github.com/JinyuanLiu-CV/TarDAL) providers. The preparation command accepts one or more local folders or ZIP files per dataset. Replace the three source arguments below with the locations of your copies, including all images specified by the manifests.

```bash
python prepare_data.py --fmb /datasets/FMB --llvip /datasets/LLVIP \
  --m3fd /datasets/M3FD --output data --split all
python evaluate.py --data-root data/test --check-inputs
```

The three splits contain 450 training pairs, 60 development pairs and 174 locked test pairs. `--check-inputs` must validate all 1,740 test cases before comparing reported metrics. For evaluation alone, use `--split test`. To additionally generate the 600 development inputs, add `--with-dev-inputs` to the preparation command with `--split all` or `--split dev`. The [evaluation guide](evaluation/README.md) explains original file identity and input verification.

## 3. Recompute the locked test results

```bash
python evaluate.py --data-root data/test --weights weights/RSSFuse.pth --output results/released
python evaluation/compare_reference.py results/released/summary.csv
python -m evaluation.compare_per_image results/released/per_image.csv \
  --reference evaluation/reference/per_image.csv --atol 0
```

The evaluator writes `per_image.csv` (1,740 cases), `summary.csv` (48 dataset/condition rows) and `provenance.json`. The summary comparison checks the archived five-decimal values in `evaluation/reference/seed42_step5000.csv`. The per-image command compares cases by dataset, pair ID and condition, verifies the input hashes, and tests six fusion and four restoration scores. Use `--atol 0` for exact stored values. To measure agreement across hardware, record the largest differences and the absolute tolerance you used.

The main DEG9 row gives equal weight to nine degraded conditions, then to each dataset. CLEAN enters ALL10. The archived three-dataset DEG9 means are **39.83180 / 3.07535 / 6.99673 / 0.56175 / 1.50585 / 0.64462** for SD / MI / EN / CC / SCD / VIF.

The [full reference table](evaluation/reference/summary.csv) contains all 48 rows and ten metrics, recomputed from the [archived per-image outputs](evaluation/reference/per_image.csv).

We regenerated and checked all 1,740 locked evaluation inputs. As a cross-device check, we ran three full frames with the released weights on an RTX 3060 using the archived Mamba 1.0.1 FP32 forward kernel. The cases were FMB 01049 (visible low light), LLVIP 220355 (IR stripes) and M³FD 01690 (both sensors degraded). The largest absolute difference across the 30 metric values from the original A100 results was 0.00000435. This check covers three independent predictions. The remaining rows are preserved from the original evaluation.

## 4. Replay the archived student training setup

```bash
python train.py train --data-root data/train --output runs/replay --seed 42 \
  --initial-weights reproduction/initial.pth --probe reproduction/probe.pth \
  --reference-trace reproduction/train_trace.csv
python evaluate.py --data-root data/test --weights runs/replay/RSSFuse.pth --output results/replay
python evaluation/compare_reference.py results/replay/summary.csv
```

The student starts at step zero and trains for 5,000 optimizer updates. Its initial tensor hash is checked on load. Each batch is checked against the trace before the update. The calibrated probe supplies training labels every eighth step. It is independent of the student's starting weights. The final inference file is `runs/replay/RSSFuse.pth`; numbered snapshots in the same directory include optimizer and scheduler state. After an interruption, repeat the training command with the same output and add `--resume`.

The archived initial weights, probe, trace and measured outcomes are for **seed 42**. A different seed requires its own experimental artifacts and results. The stored data and checkpoint checksums verify the supplied inputs. A new training run still requires its own output comparison; GPU kernels and runtime differences can affect floating-point trajectories. Full retraining with this release has not been independently verified end to end.

### Optional teacher routes

To recalibrate a probe from the included teacher, run the student with `--teacher` in place of `--probe` and use a fresh output directory:

```bash
python train.py train --data-root data/train --output runs/recalibrated --seed 42 \
  --initial-weights reproduction/initial.pth --teacher reproduction/teacher.pth \
  --reference-trace reproduction/train_trace.csv
```

Calibration uses 16 training batches with two samples each and all four actions. The teacher is the reference architecture at step 5,000 of a **20,000-step cosine schedule**. To train a replacement teacher, run `python train.py teacher --seed 42 --data-root data/train --output runs/teacher`, then substitute `runs/teacher/5000.pth` in the command above. That teacher command runs 5,000 optimizer steps; RSSFuse has its own 5,000-step cosine schedule. The archived probe route in step 4 is the direct route for reproducing the student's original labels.

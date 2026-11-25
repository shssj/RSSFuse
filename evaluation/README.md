# Locked evaluation

The test list contains 174 registered visible/infrared pairs: FMB (99), LLVIP (44) and M³FD (31). Each pair has nine synthetic degradations and one clean condition. One RSSFuse forward pass produces three images for scoring: fused RGB, restored visible RGB and restored infrared grayscale.

## Prepare images

Obtain the source images from their respective dataset providers under their terms. Use the original visible and infrared files listed in `manifests/test_images.csv`. Preserve the original filenames, extensions and bytes, including LLVIP's JPEGs. Arrange the six source directories as follows:

```text
data/test/
    FMB/vi/01049.png
    FMB/ir/01049.png
    LLVIP/vi/220355.jpg
    LLVIP/ir/220355.jpg
    M3FD/vi/01690.png
    M3FD/ir/01690.png
```

The manifest contains both encoded-file and decoded-pixel SHA-256 hashes. The second manifest, `test_cases.csv`, provides the random key, both sensor seeds, both degraded pixel hashes and the four-tensor input hash for every case. The raw photographs are distributed by their dataset owners.

## Run

```bash
python evaluate.py --data-root data/test --check-sample
python evaluate.py --data-root data/test --check-inputs
python evaluate.py --data-root data/test --weights weights/RSSFuse.pth --output results/locked_test
```

`--check-sample` verifies 30 representative inputs (one per dataset and condition). `--check-inputs` verifies all 1,740 without loading a checkpoint. The metric run independently checks its inputs. Use a tensor-only state dictionary for `--weights`. An existing trusted training snapshot containing optimizer and NumPy RNG state can be loaded with `--trusted-snapshot`.

JPEG decoders, NumPy versions and CPU implementations can affect degraded pixels. Generation has been checked with NumPy 1.23.5 and Pillow 9.3.0 on the archived images. If the installed environment produces different LQ pixels, the evaluator reports the failing case. The optional `--locked-lq DIR` supplies precomputed eight-bit PNGs at `DIR/<condition>/<dataset>/<pair_id>_<vi|ir>.png`; each override must match its archived pixel hash.

The runner evaluates one full frame at a time in FP32. The two three-scale downsamplings may require multiples of four; all locked images meet this condition. For arbitrary image pairs, the single-pair inference command pads at the bottom/right if needed, then crops the three predictions to their original size. The infrared input is grayscale and the visible input is RGB. Only the two degraded tensors go into the network. Reference images enter the scoring stage.

## Read the outputs

- `per_image.csv` contains the metric values and verified input hash for each of the 1,740 cases.
- `summary.csv` contains 48 rows: each dataset and the three-dataset mean, each with 10 individual conditions, DEG9 and ALL10.
- `provenance.json` records the checkpoint hash, manifest hash, evaluation sources and runtime. Reusing the same output path resumes from the last fully written batch (up to ten images). Use a separate output path for each model or checkpoint.

Fusion scores compare each uint8 fused image with the two high-quality source images, converted to grayscale with OpenCV. The six measures are SD, summed mutual information (MI), entropy (EN), mean correlation (CC), sum of correlations of differences (SCD), and summed four-scale pixel-domain VIF. Restoration PSNR is measured on normalized float images after clamping predictions to [0, 1]; SSIM uses an 11×11 Gaussian window (σ=1.5), valid pixels and per-channel averaging.

Compute the image mean within each dataset and condition first. DEG9 gives equal weight to the nine degraded conditions. ALL10 gives equal weight to all ten, including CLEAN. The overall row is the unweighted mean of the three dataset summaries. All ten reported measures use the higher-is-better convention.

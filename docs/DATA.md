# Reconstruct the exact experiment inputs

The three dataset providers distribute the paired source images. Download these editions under their respective access terms. RSSFuse uses its own frozen selection from these releases.

| Dataset | Files needed | Provider |
| --- | --- | --- |
| FMB | Original visible and infrared images from both train and test. | [SegMiF project](https://github.com/JinyuanLiu-CV/SegMiF), [authors' FMB download](https://drive.google.com/drive/folders/1T_jVi80tjgyHTQDpn-TjfySyW4CK1LlF?usp=sharing) |
| LLVIP | Registered visible and infrared JPEG pairs at the original resolution from the train and test folders. | [LLVIP dataset and download instructions](https://github.com/bupt-ai-cz/LLVIP) |
| M³FD | The 300-pair fusion set supplies training and development images. The 4,200-pair detection release supplies the locked test images. Both include original VIS and IR images. | [TarDAL project](https://github.com/JinyuanLiu-CV/TarDAL), [authors' archive folder](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing) |

From the repository root, replace the example source paths with your own and run:

```bash
python prepare_data.py --fmb /datasets/FMB/train /datasets/FMB/test --llvip /datasets/LLVIP --m3fd /datasets/M3FD_Fusion.zip /datasets/M3FD_Detection.zip --output data
```

Each dataset option accepts one or more extracted directories or ZIP files. Pass an image ZIP directly if its contents have not been extracted. If a download contains additional archives, extract the outer archive first. Use a fresh output directory or rerun with the same destination to verify and reuse existing images. For a single split, add `--split train`, `--split dev` or `--split test`. Add `--check-only` to verify the supplied source files without writing images.

| Dataset | Training pairs | Development pairs | Locked test pairs |
| --- | ---: | ---: | ---: |
| FMB | 150 | 20 | 99 |
| LLVIP | 150 | 20 | 44 |
| M³FD | 150 | 20 | 31 |
| Total | 450 | 60 | 174 |

Each split is written to `data/<split>/<dataset>/{vi,ir}/`. The script selects images by dataset, original numeric ID and SHA-256, including the leading zeros in the ID. It checks decoded RGB and grayscale pixels against the experiment manifests. Training and development images are saved as lossless PNGs. Locked test images retain their exact original file bytes and extensions. The locked test IDs are separate from training and development. The historical train–development audit found zero identical same-modality decoded images, 63 perceptual-hash candidate pairs and 23 nearby-ID candidate pairs. These checks concern image identity and proximity; they do not establish scene-level independence of the development split.

Training reads `data/train`. The development list records the historical selection used during model development. The released `evaluate.py` computes the locked test results on `data/test`. Each locked pair is evaluated under the same nine degraded conditions and one clean condition, producing 1,740 model inputs. The pinned degradation code, seeds and per-input hashes are in `evaluation/`.

For development or ablations, add `--with-dev-inputs` to the preparation command. This reproduces the 600 development cases (60 pairs × 10 conditions) under `data/dev_inputs/<condition>/<dataset>/{vi,ir}/`. It uses the original name-based seeds and checks the compressed PNG bytes of all 1,200 sensor images against `evaluation/manifests/dev_cases.csv`. Training and locked evaluation can run without generating these optional development files.

```bash
python evaluate.py --data-root data/test --check-inputs
python evaluate.py --data-root data/test --weights weights/RSSFuse.pth --output results/locked_test
python evaluation/compare_reference.py results/locked_test/summary.csv
```

The comparison checks the archived seed-42, step-5,000 report: ten metrics on the nine-condition mean for each of the three datasets, their equal-weight mean, and the overall ten-condition mean including clean inputs. The reference CSV stores the five decimal places available in the report. By default the checker allows half a unit in the last printed decimal, and prints the actual, expected and signed difference for each mismatch. If comparing independently retrained models or a different numerical runtime, inspect those differences directly. `--atol` lets you declare an explicit comparison tolerance in each metric's units. A run on different seeds is a separate experiment.

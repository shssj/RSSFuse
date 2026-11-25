# Component origins and notices

| Source | Use in RSSFuse | Upstream notice |
|---|---|---|
| [Restormer](https://github.com/swz30/Restormer/tree/68dc6ac472db26f16361150cb7a96a1bc87da93f) | Transformer blocks in the two pyramid encoders and the shared-state decoder. RSSFuse adds modality-specific calibration and multi-scale projections. | [MIT](LICENSES/Restormer.txt) |
| [MambaIR](https://github.com/csguoh/MambaIR/tree/19d88bb6f8d74d42e278cf30e08bee7ba124887a) | VSS blocks in the six recovery updaters, with modality conditioning, a learned step size and reuse across rounds. The integration selects the installed CUDA scan or a small PyTorch reference implementation. | [Apache-2.0](LICENSES/MambaIR.txt) |
| [MIRNet-v2](https://github.com/swz30/MIRNetv2/tree/9014049f051ad714eed8b816f23a571e986a46c7) | Two-input SKFF within LDSI. The finest scale adds a local-detail bias. The upstream terms cover the adapted SKFF code in `model_src/skff.py` and `LocalDetailSKFF` in `model_src/fusion.py`. | [Academic Public License, noncommercial](LICENSES/MIRNet-v2.md) |
| [NAFNet](https://github.com/megvii-research/NAFNet/tree/2b4af71ebe098a92a75910c233a3965a3e93ede4) | NAF blocks in the two independent recovery heads with residual image readout. | [MIT and bundled notices](LICENSES/NAFNet.txt) |
| [CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse/tree/73681d28f6100461f1b214ecf77a93b495be227b) | The decoder layout informed PMFR. The distributed implementation uses Restormer blocks, three shared-state inputs and an RGB projection. | Implementation uses the [Restormer notice](LICENSES/Restormer.txt). |

GPU execution calls the separately installed [Mamba selective-scan library](https://github.com/state-spaces/mamba). Please cite the upstream research relevant to your use. See each dataset provider for terms covering its image files.

Additional copyright notice: [Copyright © 2026 Hao Zhang (MIT)](LICENSES/Hao-Zhang-MIT.txt).

The accompanying license texts apply to their respective implementations. The authors have not selected a separate public license for the original RSSFuse code in this distribution.

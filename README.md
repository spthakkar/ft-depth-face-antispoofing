# ft-depth-face-antispoofing
Fine-tuning depth analysis for lightweight face anti-spoofing
| OULU-NPU | Protocol A & B | PyTorch | Edge deployment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

---

## Manuscript

This repository contains the official implementation for:

> **"Optimal Fine-Tuning Depth Analysis for Lightweight
> Edge-Oriented Face Anti-Spoofing Systems"**
> Shital Thakkar, Vinay Thumar
> *Under review at The Visual Computer, Springer Nature*
> DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.20271183)

If you use this code in your research, please cite
this manuscript. BibTeX entry is provided below.

---

## Overview

This repository contains the complete experimental code for a
systematic study of fine-tuning depth across seven architectures
for lightweight face anti-spoofing, evaluated under intra-dataset
and cross-dataset protocols.

**Key finding:** Partial fine-tuning (L2) recovers 87.5% of total
achievable accuracy gain while training only 72.9% of parameters,
with full fine-tuning producing regression in 32% of cases.

---

## Architectures Evaluated

| Model | Params | GFLOPs | Family |
|---|---|---|---|
| MobileNetV4 | 2.50M | 0.185 | Depthwise Separable |
| MobileViT-XXS | 0.95M | 0.257 | Hybrid CNN-Transformer |
| ConvNeXt-Femto | 4.83M | 0.779 | Modern CNN |
| GhostNetV2 | 4.88M | 0.184 | Feature Reuse |
| EfficientNet-B0 | 4.01M | 0.385 | NAS |
| ShuffleNetV2 | 5.35M | 0.596 | Channel Split |
| ResNet50 | 25.6M | 4.132 | Baseline |

---

## Datasets

| Dataset | Access | Official Link |
|---|---|---|
| OULU-NPU | Request via EULA | [sites.google.com/site/oulunpudatabase](https://sites.google.com/site/oulunpudatabase/) |
| Replay-Attack | Free download | [idiap.ch/en/scientific-research/data/replayattack](https://www.idiap.ch/en/scientific-research/data/replayattack) |
| MSU-MFSD | Email request | [biometrics.cse.msu.edu](http://biometrics.cse.msu.edu/Publications/Databases/MSUMobileFaceSpoofing/index.htm) |
| CASIA-FASD | Request from authors | [bob.db.casia_fasd](https://github.com/183amir/bob.db.casia_fasd) |

> **Dataset Access:** All four datasets are publicly available
> for non-commercial research. OULU-NPU and MSU-MFSD require
> a formal signed agreement; Replay-Attack is freely downloadable;
> CASIA-FASD requires author request. We do not redistribute
> any dataset. See `docs/dataset_setup.md` for full instructions.

---

## Installation

```bash
git clone https://github.com/spthakkar/ft-depth-face-antispoofing
cd ft-depth-face-antispoofing
pip install -r requirements.txt
```

---

## Usage

```bash
# Phase 1: Dataset preprocessing
python main.py --phase 1

# Phase 3: Training (all models, all FT levels)
python main.py --phase 3

# Phase 3: Single model
python main.py --phase 3 --model shufflenet_v2_x2_0 --ft_level L2

# Phase 6: t-SNE and Grad-CAM visualisation
python main.py --phase 6
```

---

## Fine-Tuning Levels

| Level | Description | Trainable Params |
|---|---|---|
| L1 | Classifier only | < 0.10% |
| L2 | Upper 30% backbone + head | 56–81% |
| L3 | Full fine-tuning | 100% |

---

## Results Summary

**Protocol A — Intra-dataset EER (%) at L2:**

| Model | CASIA | MSU | OULU | Replay | Mean |
|---|---|---|---|---|---|
| ShuffleNetV2 | 1.35 | 3.47 | 3.93 | 0.39 | 2.28 |
| ResNet50 | 1.11 | 0.14 | 5.11 | 0.06 | 1.61 |
| EfficientNet-B0 | 0.84 | 0.83 | 8.86 | 0.79 | 2.83 |
| GhostNetV2 | 0.51 | 7.22 | 5.14 | 0.51 | 3.35 |
| ConvNeXt-Femto | 1.12 | 10.00 | 3.02 | 0.38 | 3.63 |
| MobileNetV4 | 2.50 | 5.97 | 8.48 | 0.61 | 4.39 |
| MobileViT-XXS | 0.79 | 10.14 | 9.66 | 0.56 | 5.29 |

**Protocol B — Cross-dataset HTER (%) at L3:**

| Model | C+I+M→O | O+C+I→M | O+C+M→I | O+I+M→C | Mean |
|---|---|---|---|---|---|
| ResNet50 | 11.19 | 10.14 | 23.62 | 14.89 | 14.96 |
| ConvNeXt-Femto | 10.11 | 12.36 | 22.71 | 16.88 | 15.52 |
| MobileViT-XXS | 6.40 | 21.25 | 15.60 | 20.64 | 15.97 |
| EfficientNet-B0 | 9.59 | 11.53 | 21.38 | 21.43 | 15.98 |
| ShuffleNetV2 | 7.79 | 19.31 | 25.41 | 16.14 | 17.16 |
| MobileNetV4 | 9.77 | 15.42 | 26.74 | 19.76 | 17.92 |
| GhostNetV2 | 10.18 | 25.00 | 23.02 | 19.90 | 19.53 |

---

## Repository Structure

```
├── config.py          # Training hyperparameters and paths
├── main.py            # Entry point with phase selection
├── phases/            # Training, evaluation, visualisation
├── models/            # Architecture definitions
├── utils/             # Logging, checkpointing, data loading
├── docs/              # Dataset setup instructions
└── requirements.txt   # Python dependencies
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{thakkar2026ftdepth,
  title   = {Optimal Fine-Tuning Depth Analysis for
             Lightweight Edge-Oriented Face
             Anti-Spoofing Systems},
  author  = {Thakkar, Shital and Thumar, Vinay},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Under review},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

## Contact

**Shital Thakkar** — shitalthakkar.ec@ddu.ac.in  
Dharmsinh Desai University, Nadiad, Gujarat, India

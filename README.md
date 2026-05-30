# ARMultKAN

Official implementation of:

> **Adaptive Grid-based Residual Multiplicative KANs for Physics-Informed Learning**

---

## Overview

ARMultKAN combines:

* **Residual Multiplicative KAN (RMultKAN)** for stable representation learning;
* **Adaptive Grid Update (AGU)** for dynamic spline-grid refinement;
* Physics-informed and PDE-constrained learning frameworks.

Compared with conventional MLPs and fixed-grid KANs, ARMultKAN improves approximation accuracy while maintaining computational efficiency.

---

## Repository Structure

```text
ARMultKAN
│
├── rkan/
│   ├── RKANLayer.py
│   ├── ...
│
├── LQR_HJB_train.py
│
├── requirements.txt
│
└── README.md
```

### Main Components

| File               | Description                               |
| ------------------ | ----------------------------------------- |
| `rkan/`            | ARMultKAN model implementation            |
| `LQR_HJB_train.py` | Training script for the LQR-HJB benchmark |
| `requirements.txt` | Python dependencies                       |

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Train ARMultKAN with AGU

```bash
python LQR_HJB_train.py --name LQR_RMultKAN

python LQR_HJB_train.py --name LQR_ARMultKAN --use_agu

python LQR_HJB_train.py --name LQR_ARMultKAN --use_agu --use_agu_eps
```
Tested with Python 3.10.

### Arguments

| Argument        | Description                           |
| --------------- | ------------------------------------- |
| `--name`        | Experiment name                       |
| `--use_agu`     | Enable Adaptive Grid Update (AGU)     |
| `--use_agu_eps` | Enable dynamic AGU mixing coefficient |
| `--seed`        | Random seed                           |

## Responsible Use

This repository is intended for academic research and educational purposes only. Users are responsible for validating the correctness and suitability of the code for their own applications.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{shen2026armultkan,
  title={Adaptive Grid-based Residual Multiplicative KANs for Physics-Informed Learning},
  author={Shen, Xiya and others},
  journal={},
  year={2026}
}
```

## License

This project is released under the MIT License.

See the LICENSE file for details.

# ARMultKAN

**Adaptive-Grid Residual Multiplicative Kolmogorov-Arnold Networks (ARMultKAN)**

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
python LQR_HJB_train.py \
    --name LQR_ARMultKAN \
    --use_agu \
    --use_agu_eps
```

### Arguments

| Argument        | Description                           |
| --------------- | ------------------------------------- |
| `--name`        | Experiment name                       |
| `--use_agu`     | Enable Adaptive Grid Update (AGU)     |
| `--use_agu_eps` | Enable dynamic AGU mixing coefficient |
| `--seed`        | Random seed                           |

## License

This project is released for academic research purposes.

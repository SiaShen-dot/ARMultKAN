# Adaptive Grid-based Residual Multiplicative Kolmogorov–Arnold Networks and Their Application to Physics-Informed Learning

## Abstract

Kolmogorov-Arnold Networks (KANs) and their multiplicative variants (MultKANs) have emerged as a promising alternative to traditional Multi-Layer Perceptrons, replacing fixed activation functions with learnable splines. However, this novel design introduces two inherent architectural challenges: 1) the unconstrained learning of spline functions can lead to pathological behaviors and training instability, and 2) their grid adaptation mechanism is purely data-driven, allocating resolution based on data density rather than model performance, which leads to a suboptimal trade-off between accuracy and computational cost. To overcome these fundamental limitations, we propose the Adaptive Grid-based Residual MultKAN (ARMultKAN). Our architecture introduces two synergistic improvements. First, we incorporate skip connections to create a stable Residual MultKAN (RMultKAN) backbone, theoretically guaranteeing stable gradient dynamics as proven by our Gradient Decomposition Theorem. Second, we develop a loss-driven Adaptive Grid Update (AGU) mechanism that intelligently allocates grid points to high-complexity regions, enhancing representational power without biasing the training objective. The efficacy of this approach is formally established by our Approximation Theorem. To demonstrate the practical benefits of our stabilized and efficient architecture, we apply it within a physics-informed framework to solve complex partial differential equations. This application, termed Physics-Informed ARMultKAN (PI-ARM), leverages the physics-informed loss to simultaneously optimize network parameters and guide the adaptive grid. Comprehensive experiments validate that the RMultKAN backbone surpasses MultKANs in stability and accuracy, and the full PI-ARM framework significantly outperforms state-of-the-art physics-informed models.


## Overview



This project demonstrates the application of novel ARMultKAN architectures to solve challenging scientific computing problems. The core components are:
- **PI-RM**: A Physics-Informed model using the RMultKAN backbone, which incorporates residual connections for enhanced training stability.

- **PI-ARM**: The full, adaptive framework that integrates a loss-driven Adaptive Grid Update (AGU) mechanism for superior accuracy and efficiency.

  

## Directory Structure



├── data/                  # Directory for storing the ground-truth numerical solutions of PDEs.

├── model/                 # Directory for saving trained model checkpoints.

├── rkan/                  # Source code for the RMultKAN library.

├── train_Diffusion_PIRM.ipynb   # Jupyter notebook to train the baseline PI-RM model.

├── train_Diffusion_PIARM.ipynb  # Jupyter notebook to train the full PI-ARM model with AGU.

└── Val_Diffusion_PIARM.ipynb    # Jupyter notebook for validating and analyzing the trained PI-ARM model.



## Requirements



The code is implemented in Python.

You can install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```



## Usage



To reproduce the results, please follow these steps:



### 1. Training the Baseline (PI-RM)



- Open and run the `train_Diffusion_PIRM.ipynb` notebook.
- This will train the non-adaptive RMultKAN model. Trained models will be saved in the `/model` directory.



### 2. Training the Adaptive Model (PI-ARM)



- Open and run the `train_Diffusion_PIARM.ipynb` notebook.
- This will train the full RMultKAN model with our proposed AGU mechanism.



### 3. Validation and Analysis



- Open and run the `Val_Diffusion_PIARM.ipynb` notebook.

- This notebook loads the trained models from the `/model` directory and performs a detailed validation and error analysis.

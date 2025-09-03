# Adaptive Grid-based Residual Multiplicative Kolmogorov–Arnold Networks and Their Application to Physics-Informed Learning



This repository contains the official implementation for training and validating the Physics-Informed ARMultKAN (PI-ARM) and its baseline PI-RM (without adaptive) models for solving partial differential equations (PDEs).



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
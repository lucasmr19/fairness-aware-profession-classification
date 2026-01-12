# Fairness-Aware Profession Classification

This repository presents a **data-centric analysis of fairness and performance trade-offs** in deep learning–based profession classification.
Using a fixed ResNet-50 architecture, the project investigates how **dataset design choices**, including **balancing and controlled imbalance of sensitive attributes**, affect predictive performance, fairness metrics, and learned representations.

The study combines **quantitative evaluation** with **Grad-CAM interpretability** to provide insight into both model behavior and internal attention mechanisms.

## Project Overview

The experiments are conducted on four dataset variants:

* **Experiment A** – Fully balanced (gender × race × profession)
* **Experiment B** – Race imbalance
* **Experiment C** – Gender imbalance
* **Experiment D** – Naturalistic target imbalance (real-world setting)

All experiments share:

* Identical architecture (pre-trained **ResNet-50**)
* Same optimizer and hyperparameters
* Same prediction task (4-class profession classification)

This design isolates the **impact of data composition** as the primary experimental variable.

## Key Contributions

* Controlled comparison of **fairness vs. performance trade-offs**
* Analysis of **data scarcity vs. fairness stability**
* Evaluation using **accuracy, balanced accuracy, macro-F1**
* Fairness assessment via **Demographic Parity (DP)** and **Equal Opportunity (EO)**
* **Grad-CAM–based interpretability** across experiments
* Discussion of implications under **EU AI Act (high-risk systems)**

## Repository Structure

```
fairness-aware-profession-classification/
│
├── fairness_dataset_design_experiments.ipynb
│
├── Descripción_proyecto_individual.pdf
├── datasets_comparison.png
├── gradcam_interpretability.png
├── tree.txt
│
├── checkpoints/
│   ├── A_resnet50_best.pth
│   ├── B_resnet50_best.pth
│   ├── C_resnet50_best.pth
│   └── D_resnet50_best.pth
│
├── training/
│   ├── config.py              # Training configuration
│   ├── constants.py           # Global constants
│   ├── dataset.py             # Dataset loading and preprocessing
│   ├── transforms.py          # Image transformations
│   ├── model_creation.py      # ResNet-50 model definition
│   ├── training.py            # Training and validation loops
│   ├── mixup.py               # Data augmentation (MixUp)
│   ├── utils.py               # Utility functions (metrics, logging)
│   └── __init__.py
│
└── README.md
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/<username>/fairness-aware-profession-classification.git
cd fairness-aware-profession-classification
```

2. Install dependencies (example):

```bash
pip install torch torchvision matplotlib numpy scikit-learn pytorch-grad-cam
```

3. Open the notebook:

```bash
jupyter notebook fairness_dataset_design_experiments.ipynb
```

Pre-trained checkpoints are provided for reproducibility.

## Interpretability

Grad-CAM is applied to the **last convolutional block of ResNet-50** to visualize model attention across experiments.
These visualizations reveal how dataset imbalance shapes **feature attribution patterns**, complementing numerical fairness metrics.

## Ethical and Regulatory Context

The task studied falls under **high-risk AI systems** according to the proposed **EU AI Act**, as it relates to employment and professional categorization.

The findings demonstrate that:

* Dataset balancing alone is insufficient for fairness guarantees
* Post hoc evaluation must be complemented with interpretability
* Fairness should be treated as a **design-time concern**

## Future Work

* Fairness-aware loss functions and constraints
* Multiple random seeds for variance estimation
* Alternative architectures and self-supervised pretraining
* Intersectional fairness analysis
* Deployment-oriented evaluation

## References

Key references include Hardt et al. (2016), Barocas et al. (2023), Selvaraju et al. (2017), and He et al. (2016).
See the report for the full bibliography.

# MIT License
This project is under MIT License. See [LICENSE](LICENSE) file for more information

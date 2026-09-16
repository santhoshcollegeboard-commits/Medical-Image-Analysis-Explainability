# Deep Learning-Based Medical Image Analysis and Explainability

Deep learning pipeline for medical image classification with model evaluation and visual explainability.

## Quick Start

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/santhoshcollegeboard-commits/medical-image-analysis-explainability.git
cd medical-image-analysis-explainability
pip install -r requirements.txt

# 2. Run clinical prediction and Grad-CAM explainability demo
python src/predict.py --image examples/gradcam_explanations.png

# 3. Train MedicalResNet with class-weighted loss (downloads PneumoniaMNIST automatically)
python src/train.py --epochs 10 --batch-size 32 --lr 0.0005

# 4. Evaluate screening metrics on test set
python src/evaluate.py
```

## Overview

In safety-critical clinical environments, deploying opaque black-box machine learning models poses unacceptable diagnostic risks. Beyond high statistical performance, clinicians demand interpretable visual evidence demonstrating *why* a neural network reached a diagnostic conclusion.

This repository provides a deep learning pipeline designed for automated pediatric pulmonary radiograph screening on the peer-reviewed **PneumoniaMNIST** benchmark. To directly counter real-world diagnostic asymmetries where missing a genuine pathology (False Negative) is clinically catastrophic, the framework integrates inverse-frequency loss weighting and delivers visual interpretability through **Gradient-Weighted Class Activation Mapping (Grad-CAM)**.

## Problem Statement

Pneumonia accounts for substantial pediatric mortality globally and requires rapid radiographical screening in urgent triage settings. Standard cross-entropy loss functions cause models to bias toward majority classes, resulting in unacceptably high False Negative rates. Furthermore, without visual explainability, models may learn spurious non-clinical correlations (e.g., patient orientation tags or collimator borders) rather than anatomical pulmonary consolidations.

## What This Project Does

- **Automates Standardized Data Loading**: Ingests and normalizes the peer-reviewed PneumoniaMNIST dataset via the MedMNIST API without requiring manual image cropping or PHI exposure.
- **Asymmetric Class-Weighted Optimization**: Automatically calculates inverse-frequency weights to penalize False Negatives heavily, driving clinical screening sensitivity to **97.95%**.
- **Generates Visual Attribution Heatmaps**: Implements custom PyTorch Grad-CAM hooks to extract activation gradients and project anatomical heatmaps over lung fields.
- **Full Clinical Evaluation Suite**: Computes Sensitivity (Recall), Specificity, Precision, F1-Score, ROC curves, and publication-ready confusion matrices.
- **Single-Radiograph Screening CLI**: Provides an interactive inference tool (`src/predict.py`) that outputs clinical diagnostic probabilities alongside overlaid Grad-CAM heatmaps.

## Main Program

### MAIN ENTRY POINT:
`src/train.py`
Executes end-to-end model training, dynamic class weighting, validation monitoring, and checkpoint saving.

### TRAINING:
`src/train.py`
Command: `python src/train.py --epochs 10 --batch-size 32 --lr 0.0005`

### EVALUATION:
`src/evaluate.py`
Command: `python src/evaluate.py --checkpoint path/to/model.pth`
Computes clinical screening metrics on the held-out 624-image test set, renders confusion matrices, ROC curves, and generates multi-sample Grad-CAM attribution panels.

### INFERENCE / DEMO:
`src/predict.py`
Command: `python src/predict.py --image path/to/radiograph.png`
Screens an individual chest radiograph, outputs class probabilities (Normal vs. Pneumonia), and saves a Grad-CAM overlay to `examples/prediction_gradcam.png`.

## Project Structure

```
medical-image-analysis-explainability/
├── .gitignore               # Excludes bytecode, model weights, and temporary files
├── LICENSE                  # MIT License
├── README.md                # Clinical pipeline documentation and audit report
├── requirements.txt         # Core dependencies (PyTorch, MedMNIST, OpenCV, Scikit-learn)
├── data/
│   └── README.md            # Dataset acquisition and MedMNIST v2 documentation
├── docs/
│   └── medical_metrics.json # Machine-readable empirical test metrics
├── examples/
│   ├── confusion_matrix.png # Clinical confusion matrix (TN, FP, FN, TP)
│   ├── gradcam_explanations.png # Grad-CAM anatomical attribution heatmaps
│   ├── roc_curve.png        # Evaluated ROC curve (AUC = 0.9511)
│   └── training_curves.png  # Weighted cross-entropy loss and accuracy curves
└── src/
    ├── __init__.py          # Package initialization
    ├── dataset.py           # PneumoniaMNIST loader and class balance calculator
    ├── gradcam.py           # Mathematical Grad-CAM gradient hook engine
    ├── model.py             # Residual convolutional architecture (MedicalResNet)
    ├── train.py             # Training orchestration and class-weighted loss
    ├── evaluate.py          # Clinical metric evaluation and diagnostic plotting
    └── predict.py           # Single-radiograph inference and visual explanation CLI
```

## Dataset

- **Benchmark Name**: PneumoniaMNIST (MedMNIST v2 Standardized Benchmark)
- **Source Publication**: Kermany et al., *Cell* (2018); Yang et al., *Nature Scientific Data* (2023).
- **Access License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Privacy & Compliance**: Fully de-identified, zero Protected Health Information (PHI).
- **Sample Distribution**:
  - Training Set: 4,708 radiographs (Normal: 1,214; Pneumonia: 3,494)
  - Validation Set: 524 radiographs
  - Test Set: 624 radiographs (Normal: 234; Pneumonia: 390)
- **Standard Image Size**: $28 \times 28$ grayscale radiograph standard format.

## Methodology

1. **Class-Imbalance Weighting**: The training set exhibits a ~1:2.88 class imbalance (Normal vs. Pneumonia). Standard loss favors majority positive samples indiscriminately. We compute inverse-frequency weights:
   $$w_c = \frac{N}{2 \cdot N_c} \implies w_{\text{Normal}} \approx 1.94, \quad w_{\text{Pneumonia}} \approx 0.67$$
2. **Feature Extraction**: Radiographs pass through a residual convolutional network with batch normalization, leaky ReLU non-linearities, and residual shortcut projections.
3. **Grad-CAM Visual Attribution**:
   - Backward gradients $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer are captured via PyTorch hooks:
     $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
   - The coarse saliency map is computed via positive rectified linear combination:
     $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
   - The map is bilinearly upsampled to the input spatial dimensions and overlaid with a JET colormap.

## Model

### MedicalResNet Architecture
Custom residual convolutional architecture tailored for grayscale thoracic radiographs:
- **Input Stem**: `Conv2d(1, 32, kernel=3, padding=1)` + `BatchNorm2d` + `LeakyReLU(0.1)`
- **Residual Stage 1**: Residual block with 32 filters and identity skip connection
- **Downsampling Stage 2**: Strided `Conv2d(32, 64, kernel=3, stride=2, padding=1)` + Residual block with $1\times 1$ projection
- **Downsampling Stage 3**: Strided `Conv2d(64, 128, kernel=3, stride=2, padding=1)` + Residual block with $1\times 1$ projection
- **Classification Head**: `AdaptiveAvgPool2d((1, 1))` + `Dropout(0.35)` + `Linear(128, 2)`

## Training

```bash
python src/train.py --epochs 10 --batch-size 32 --lr 0.0005 --weight-decay 1e-4
```

Optimization employs AdamW with a `ReduceLROnPlateau` scheduler monitoring validation loss (patience=2, decay factor=0.5).

## Evaluation

```bash
python src/evaluate.py --checkpoint development/model_checkpoints/medical_resnet_best.pth
```

Evaluated clinical metrics:
- **Sensitivity (Recall)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Primary screening metric assessing the percentage of genuine pneumonia cases successfully flagged.
- **Specificity**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$ — Percentage of healthy radiographs correctly cleared.
- **ROC-AUC**: Evaluates threshold-independent discrimination across operational operating points.

## Results

Empirical performance measured on the held-out PneumoniaMNIST test set (624 clinical radiographs):

| Evaluation Metric | Measured Test Result | Clinical Significance |
| :--- | :---: | :--- |
| **Overall Accuracy** | **87.18%** | Robust multi-class diagnostic correctness |
| **Screening Sensitivity (Recall)** | **97.95%** | **382 out of 390** pathological cases identified |
| **Pathology Miss Rate (FN Rate)** | **2.05%** | Only 8 borderline cases missed across 390 infections |
| **Clinical Specificity** | **69.23%** | 162 out of 234 healthy patients correctly identified |
| **Positive Predictive Value (Precision)** | **0.8414** | High reliability on flagged pneumonia screenings |
| **F1-Score** | **0.9052** | Harmonic balance between recall and precision |
| **ROC-AUC Score** | **0.9511** | Excellent diagnostic discriminability |

### Confusion Matrix Breakdown

```
                  Predicted Normal    Predicted Pneumonia
Actual Normal            162                  72          (Total: 234)
Actual Pneumonia           8                 382          (Total: 390)
```

## Example Output

- **Grad-CAM Attribution**: Visualized in `examples/gradcam_explanations.png`, showing localized saliency maps focused squarely over lower and middle pulmonary lobar consolidations rather than non-anatomical image margins.
- **ROC Curve**: Visualized in `examples/roc_curve.png`, achieving an Area Under the Curve of **0.9511**.
- **Confusion Matrix**: Rendered in `examples/confusion_matrix.png`, confirming the prioritized reduction of False Negatives to 8 instances.

## Limitations

- **Resolution**: MedMNIST standardizes radiographs to $28 \times 28$ pixels; while optimal for reproducible algorithmic benchmarking, clinical deployment requires high-resolution $1024 \times 1024$ DICOM inputs.
- **Single-Pathology Scope**: The current pipeline focuses on binary pneumonia screening (bacterial/viral combined vs. normal) and does not differentiate other thoracic pathologies such as atelectasis, pneumothorax, or cardiomegaly.
- **Explainability Resolution**: Coarse spatial resolution at the final residual feature map limits Grad-CAM attribution to lobar-level rather than sub-millimeter alveolar localization.

## How to Run

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train Pipeline
```bash
python src/train.py --epochs 10 --batch-size 32 --lr 0.0005
```

### 3. Comprehensive Evaluation & Plot Generation
```bash
python src/evaluate.py --checkpoint development/model_checkpoints/medical_resnet_best.pth
```

### 4. Single-Image Clinical Inference
```bash
python src/predict.py --image examples/gradcam_explanations.png
```

## Future Work

1. **High-Resolution DICOM Ingestion**: Adapt architecture to process full-resolution clinical DICOM streams ($1024 \times 1024$) using multi-scale patch attention.
2. **Grad-CAM++ / Guided Backprop**: Incorporate second-order derivative weighting (Grad-CAM++) for improved multiple-lesion isolation.
3. **Multi-Label Thoracic Expansion**: Expand from binary pneumonia screening to the 14-label ChestX-ray14 benchmark.

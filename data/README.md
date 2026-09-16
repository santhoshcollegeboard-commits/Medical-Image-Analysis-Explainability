# Dataset Guide: PneumoniaMNIST (MedMNIST v2)

## Benchmark Overview
- **Dataset**: PneumoniaMNIST (MedMNIST benchmark collection)
- **Clinical Origin**: Kermany et al., *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, **Cell** 172(5), 1122-1131, 2018.
- **Benchmark Publication**: Yang et al., *"MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification"*, **Scientific Data (Nature)**, 2023.
- **Hosting / DOI**: Zenodo [doi:10.5281/zenodo.4269852](https://doi.org/10.5281/zenodo.4269852)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Ethical Usage & Privacy Compliance
- **De-Identification**: All radiographs in this benchmark are fully de-identified and anonymized.
- **HIPAA / GDPR Compliance**: Contains zero Protected Health Information (PHI). Safe for public research, reproducibility, and educational portfolios.
- **Disclaimer**: This project is developed for educational and engineering benchmarking purposes. It is **not** an FDA/CE-approved medical device and must not be used for primary clinical diagnosis.

## Dataset Specifications
- **Modality**: Pediatric Chest Radiographs (Anterior-Posterior thoracic views).
- **Task**: Binary classification (Screening for pulmonary opacities / pneumonia).
- **Classes**:
  - Class 0: **Normal** (Clear lungs, no pulmonary consolidation)
  - Class 1: **Pneumonia** (Bacterial or viral pulmonary consolidation / infiltrate)
- **Resolution**: Standardized 28x28 grayscale images.
- **Partitioning**:
  - **Train Set**: 4,708 samples (Normal: 1,214 | Pneumonia: 3,494)
  - **Validation Set**: 524 samples (Normal: 135 | Pneumonia: 389)
  - **Test Set**: 624 samples (Normal: 234 | Pneumonia: 390)

## Automatic Ingestion
Dataset download and cache verification are handled natively by MedMNIST:
```python
import medmnist
from medmnist import INFO
info = INFO['pneumoniamnist']
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', download=True)
```
Ingested files are cached locally in `~/.medmnist/pneumoniamnist.npz`.

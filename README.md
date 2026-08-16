# MRI_images_Brain_Tumor_Classification
This model is trained using the Brain Tumor MRI Dataset by Masoud Nickparvar on Kaggle.
Msoud Nickparvar. (2021). Brain Tumor MRI Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2645886

# AWS deployment:
Deployed on AWS EC2 instance
[click to view interface](http://52.3.228.145:8501/)

# 🧠 Brain Tumor Classification using DenseNet121

A deep learning pipeline for automated brain tumor classification from MRI scans using transfer learning with DenseNet121 and PyTorch. Achieves **98.40% test accuracy** across four classes on a held-out test set of 1,311 images.

---

## Model

**Architecture:** DenseNet121 pretrained on ImageNet, with a custom classification head:

```
DenseNet121 Backbone
      ↓
Dropout (0.4)
      ↓
Linear (1024 → 256) + ReLU
      ↓
Dropout (0.3)
      ↓
Linear (256 → 4)
```

**Classes:** Glioma · Meningioma · No Tumor · Pituitary

---

## Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 99.66% | 96.33% | 97.97% |
| Meningioma | 96.45% | 97.71% | 97.08% |
| No Tumor | 98.06% | 99.75% | 98.90% |
| Pituitary | 99.67% | 99.33% | 99.50% |
| **Overall** | **98.42%** | **98.28%** | **98.36%** |

**Test Accuracy: 98.40% · Validation Accuracy: 98.37% · Stopped at epoch 35/60**

---

## Key Design Decisions

**1. Transfer Learning**
Training from scratch on ~7,000 images risks underfitting. ImageNet pretrained weights provide strong low-level feature representations that transfer well to MRI scans.

**2. Progressive Layer Unfreezing**
The first three DenseBlocks are frozen for epochs 1–15, letting the classifier head adapt first without disturbing pretrained features. All layers are unfrozen from epoch 16 for full fine-tuning.

**3. Mixed Precision Training (AMP)**
Reduces GPU memory usage by ~40%, enabling stable training on 4GB VRAM with a batch size of 24.

**4. OneCycleLR Scheduler**
Enables faster convergence compared to step-based schedulers, reaching high validation accuracy within 35 epochs.

**5. `num_workers=0`**
Required on Windows — avoids DataLoader multiprocessing crashes caused by lack of Unix-style forking.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 24 |
| Learning Rate | 0.001 |
| Optimizer | SGD (Nesterov momentum=0.9) |
| Scheduler | OneCycleLR |
| Loss | CrossEntropyLoss |
| Max Epochs | 60 (early stopping at 35) |
| Early Stopping Patience | 10 |
| Mixed Precision | true |

**Augmentation (training only):** Random horizontal/vertical flip, rotation (±15°), color jitter, affine transforms.

---

## Project Structure

```
├── pipeline_training_testing.ipynb     # Training script
├── app.py                              # Streamlit web interface
├── checkpoints/
│   └── best_model.pth                  # Saved model weights
├── Training/                           # Training images (by class folder)
├── Testing/                            # Test images (by class folder)
├── confusion_matrix.png
└── training_history.png
```

---

## Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Install Dependencies

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install streamlit plotly pillow scikit-learn matplotlib seaborn tqdm numpy
```

### 3. Prepare Dataset

Organize your dataset as follows:

```
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

### 4. Train the Model

```bash
pipeline_training_testing.ipynb
```

Best model is saved automatically to `checkpoints/best_model.pth`.

### 5. Run the Web Interface

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser to upload MRI scans and view predictions.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
plotly>=5.18.0
pillow>=10.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
numpy>=1.24.0
```

---

> **Disclaimer:** This project is for educational and research purposes only and is not intended for clinical use.

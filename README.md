# 🧠 NeuroScan AI — Brain Tumor Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning system for automated **Lower-Grade Glioma (LGG)** brain tumor segmentation from MRI scans using a standard **PyTorch U-Net** architecture with skip connections, optimized using a custom compound objective combining **Binary Cross-Entropy (BCE)** and **Dice Loss**, paired with an interactive **Streamlit** clinical evaluation web app.

---

## 📌 Project Overview

- **Dataset:** LGG Brain MRI dataset (`./data/kaggle_3m/`) with `.tif` fluid-attenuated inversion recovery (FLAIR) / T1/T2 multi-sequence slices paired with `_mask.tif` ground truth segmentation masks.
- **Data Preprocessing:** Automated slice-mask pairing, binarization, balanced stratified sampling, and Albumentations-based geometric transformations and normalizations at $256 \times 256$ resolution.
- **Model Architecture:** Standard 4-level encoder-decoder U-Net with skip connections connecting encoder feature representations directly to corresponding decoder stages.
- **Loss Function:** Compound BCE + Dice Loss mitigating class imbalance between microscopic tumor lesions and large brain/background tissue.
- **Deployment:** Interactive Streamlit web interface for single-slice upload or benchmark evaluation with real-time confidence thresholding, multi-modal visualizations, and quantitative pixel metrics.

---

## 🏗️ Architecture & Loss

### U-Net Architecture
- **Contracting Path (Encoder):** 4 downsampling blocks (`DoubleConv` with Conv $3\times3$, Batch Normalization, and ReLU activations, followed by $2\times2$ Max Pooling).
- **Bottleneck:** High-level contextual representation ($512$ feature channels).
- **Expansive Path (Decoder):** 4 upsampling stages with skip connections concatenating spatial encoder features with upsampled decoder features.
- **Output Layer:** $1\times1$ convolution mapping to binary segmentation mask logits.

### Compound Loss Function
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum (p_i \cdot y_i) + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$

---

## 📁 Repository Structure

```
brain-tumor-segmentation/
├── app.py                 # Streamlit interactive evaluation dashboard
├── dataset.py             # Dataset pairing, Albumentations transforms & PyTorch DataLoaders
├── loss.py                # Combined BCEDiceLoss and Dice/IoU evaluation metrics
├── model.py               # Standard PyTorch U-Net with skip connections
├── train.py               # Training loop with validation checkpoints
├── requirements.txt       # Project dependencies
├── training_history.json  # Saved metrics per training epoch
├── unet_brain_tumor.pth   # Saved model weights checkpoint
└── README.md              # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/kumkumgupta01/neuroscan-ai.git
cd neuroscan-ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training
Run training on GPU or CPU:
```bash
python train.py --epochs 5 --batch-size 8 --lr 1e-3 --output-weights unet_brain_tumor.pth
```

### 3. Launch the Interactive App
Start the Streamlit dashboard:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Training Results (5 Epochs)

- **Best Validation Dice Score:** **0.5300**
- **Best Validation IoU (Jaccard Index):** **0.5008**
- **Convergence:** Steady decrease in compound loss across epochs with consistent metric improvement.

---

## 🔬 Clinical Evaluation Features

1. **Test Slice Upload:** Upload custom `.tif`, `.png`, or `.jpg` brain MRI scans.
2. **Benchmark Browser:** Instant 1-click loading from dataset sample slices.
3. **Multi-View Inspection:**
   - Preprocessed scan ($256 \times 256$)
   - Predicted binary tumor mask
   - Translucent overlay with highlighted tumor contours
   - Continuous sigmoid probability heatmap (`magma` colormap)
   - Pixel probability distribution histogram
4. **Quantitative Metrics:**
   - Tumor Area in pixels and percentage of intracranial brain tissue
   - Peak probability & mean detection confidence
   - Validation against Ground Truth (Dice similarity and IoU) when masks are available
5. **Exporting:** 1-click export of predicted masks and composite images to PNG.

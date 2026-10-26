# 🧠 NeuroScan AI — Brain Tumor Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning system for automated **Lower-Grade Glioma (LGG)** brain tumor segmentation from MRI scans using a standard **PyTorch U-Net** architecture with skip connections, optimized using a custom compound objective combining **Binary Cross-Entropy (BCE)** and **Dice Loss**, paired with an interactive **Streamlit** clinical evaluation web app.

---

## 📌 Dataset Details

- **Source:** LGG Brain MRI Segmentation dataset (`./data/kaggle_3m/`) with fluid-attenuated inversion recovery (FLAIR), pre-contrast, and post-contrast multi-sequence `.tif` images.
- **Dataset Scale:** 110 unique patients, **3,929 total MRI slice-mask pairs**:
  - **1,373** tumor-positive slices
  - **2,556** normal / clear slices
- **Preprocessing Pipeline:**
  - Automated pair detection and mask binarization ($1.0$ for tumor pixels, $0.0$ for healthy tissue/background).
  - Spatial resizing to uniform $256 \times 256$ resolution.
  - ImageNet per-channel standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  - Geometric augmentations (random horizontal/vertical flips, 90° rotations, affine transforms) applied via Albumentations during training.
- **Benchmark Samples:** A curated subset of 5 benchmark slices (3 positive tumor scans, 2 normal clear scans) are stored in [`samples/`](file:///c:/Users/DELL/Desktop/brain-tumor-segmentation/samples/) for offline testing and quick evaluation without needing the full raw dataset archive.

---

## 🏗️ Model Architecture & Loss Summary

### Standard PyTorch U-Net Architecture
- **Contracting Path (Encoder):** 4 downsampling blocks consisting of consecutive $3\times3$ convolutions, Batch Normalization, and ReLU activations (`DoubleConv`), followed by $2\times2$ Max Pooling. Feature channels double at each stage ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$).
- **Bottleneck:** Deep latent representation ($512$ feature channels) capturing high-level anatomical context.
- **Expansive Path (Decoder):** 4 upsampling stages utilizing bilinear interpolation and **skip connections** that concatenate encoder spatial feature maps to preserve fine-grained tumor boundary details ($512 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 32$).
- **Output Layer:** $1\times1$ convolution mapping feature maps to single-channel binary segmentation logits.
- **Total Parameters:** **4,318,401** trainable parameters (~16.47 MB in-memory footprint).

### Compound Loss Function (BCE + Dice Loss)
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum (p_i \cdot y_i) + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$

- **Binary Cross-Entropy (BCE):** Optimizes pixel-wise classification probabilities independently across the entire field of view.
- **Dice Loss:** Direct surrogate for the Dice Similarity Coefficient (F1-score), counteracting severe class imbalance between small tumor lesions and vast intracranial tissue/background.

---

## 📊 Training Results & Metrics Table

Trained for 5 epochs using the AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$) with Cosine Annealing learning rate scheduling.

| Epoch | Learning Rate | Train Loss | Val Loss | Train Dice | Val Dice | Val IoU | Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1/5** | $0.001000$ | 1.5517 | 1.6815 | 0.1683 | 0.1647 | 0.1467 | 76.8s |
| **2/5** | $0.000905$ | 1.4161 | 1.5206 | 0.3124 | 0.3048 | 0.2862 | 75.3s |
| **3/5** | $0.000655$ | 1.3433 | 1.3639 | 0.4504 | 0.4917 | 0.4631 | 75.8s |
| **4/5** | $0.000346$ | 1.3074 | 1.3359 | 0.5404 | 0.5217 | 0.4934 | 76.4s |
| **5/5** | $0.000096$ | 1.2957 | 1.3298 | 0.5375 | **0.5300** | **0.5008** | 76.2s |

- **Best Validation Dice Score:** **0.5300**
- **Best Validation IoU (Jaccard Index):** **0.5008**
- **Weights File:** `unet_brain_tumor.pth` (saved checkpoint at highest validation Dice metric).

---

## 📁 Repository Structure

```
brain-tumor-segmentation/
├── app.py                 # Streamlit interactive clinical evaluation web dashboard
├── dataset.py             # Dataset pairing, Albumentations transforms & PyTorch DataLoaders
├── eval_ml.py             # Diagnostic ML script: parameter inspection, forward pass, & metrics
├── loss.py                # Combined BCEDiceLoss, Dice coefficient, and IoU evaluation metrics
├── model.py               # Standard PyTorch U-Net with skip connections
├── requirements.txt       # Project dependencies
├── samples/               # Benchmark MRI slices & masks (3 positive, 2 normal)
├── src/                   # Source package with U-Net architecture modules
│   ├── __init__.py
│   └── model.py
├── train.py               # 5-epoch training loop with validation checkpoints
├── training_history.json  # Saved metrics per training epoch
├── unet_brain_tumor.pth   # Saved model weights checkpoint (~51.9 MB)
└── README.md              # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and set up the virtual environment:
```bash
git clone https://github.com/kumkumgupta01/neuroscan-ai.git
cd neuroscan-ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Diagnostic Evaluation Script
Execute the standalone evaluation script to inspect model parameters, perform forward inference on benchmark slices, and compute tumor metrics:
```bash
python eval_ml.py
```

### 3. Launch the Interactive Streamlit Dashboard
Start the web application:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Training (Optional)
To retrain or run training experiments:
```bash
python train.py --epochs 5 --batch-size 8 --lr 1e-3 --output-weights unet_brain_tumor.pth
```

---

## 🔬 Clinical Evaluation Features

1. **Test Slice Upload:** Upload custom `.tif`, `.tiff`, `.png`, or `.jpg` brain MRI scans.
2. **Benchmark Browser:** Instant 1-click loading from bundled benchmark slices in `samples/`.
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

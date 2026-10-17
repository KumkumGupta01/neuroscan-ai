"""
NeuroScan AI — Brain Tumor Segmentation Interactive App
Powered by PyTorch U-Net & Streamlit

Allows evaluators to upload MRI slices (.tif, .png, .jpg) or select benchmark
slices from the LGG Brain MRI dataset, visualizes predicted tumor masks and overlays,
and computes detailed quantitative tumor metrics.
"""

import os
import io
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model import UNet


# --- Page Configuration ---
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Medical Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-positive {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-negative {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_unet_model(weights_path: str = "unet_brain_tumor.pth", init_features: int = 32):
    """
    Loads the trained U-Net model from disk with caching for performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, init_features=init_features, bilinear=True)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            val_dice = checkpoint.get("val_dice", None)
            epoch = checkpoint.get("epoch", None)
            info = f"Loaded checkpoint (Epoch: {epoch}, Val Dice: {val_dice:.4f})" if val_dice else "Loaded checkpoint"
        else:
            model.load_state_dict(checkpoint)
            info = "Loaded raw state_dict"
        model_loaded = True
    else:
        info = f"Weights file '{weights_path}' not found! Running with initialized weights."
        model_loaded = False

    model.to(device)
    model.eval()
    return model, device, model_loaded, info


def preprocess_image(image_rgb: np.ndarray, target_size: int = 256) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Prepares input RGB image:
    Resizes to target_size x target_size and normalizes with ImageNet statistics.
    Returns (resized_rgb, input_tensor).
    """
    resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Standard normalization matching Albumentations pipeline
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std

    # [H, W, C] -> [1, C, H, W]
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return resized, tensor


def compute_metrics(pred_mask: np.ndarray, probs: np.ndarray, resized_img: np.ndarray, gt_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculates detailed clinical and morphological tumor metrics.
    """
    tumor_pixels = int(np.sum(pred_mask > 0))
    total_pixels = int(pred_mask.size)

    # Approximate intracranial brain tissue (exclude black air background)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    brain_mask = (gray > 18).astype(np.uint8)
    brain_pixels = int(np.sum(brain_mask))

    tumor_pct_slice = (tumor_pixels / total_pixels) * 100.0
    tumor_pct_brain = (tumor_pixels / max(1, brain_pixels)) * 100.0
    has_tumor = tumor_pixels >= 15  # noise filter threshold

    mean_conf = float(np.mean(probs[pred_mask > 0])) if tumor_pixels > 0 else 0.0
    max_conf = float(np.max(probs))

    metrics = {
        "has_tumor": has_tumor,
        "tumor_pixels": tumor_pixels,
        "brain_pixels": brain_pixels,
        "total_pixels": total_pixels,
        "tumor_pct_slice": tumor_pct_slice,
        "tumor_pct_brain": tumor_pct_brain,
        "mean_confidence": mean_conf,
        "max_confidence": max_conf,
    }

    if gt_mask is not None:
        gt_bin = (gt_mask > 0).astype(np.uint8)
        gt_pixels = int(np.sum(gt_bin))
        intersection = int(np.sum((pred_mask > 0) & (gt_bin > 0)))
        union = int(np.sum((pred_mask > 0) | (gt_bin > 0)))

        dice = (2.0 * intersection) / (tumor_pixels + gt_pixels + 1e-6) if (tumor_pixels + gt_pixels) > 0 else 1.0
        iou = (intersection) / (union + 1e-6) if union > 0 else 1.0

        metrics.update({
            "gt_pixels": gt_pixels,
            "dice_score": dice,
            "iou_score": iou,
            "has_gt": True,
        })
    else:
        metrics["has_gt"] = False

    return metrics


def create_overlay(resized_rgb: np.ndarray, pred_mask: np.ndarray, color=(255, 40, 40), alpha: float = 0.45) -> np.ndarray:
    """
    Creates an overlaid composite image with translucent segmentation and yellow contour boundary.
    """
    overlay = resized_rgb.copy()
    mask_bool = pred_mask > 0

    if np.any(mask_bool):
        # Color wash over tumor region
        overlay[mask_bool] = (
            overlay[mask_bool] * (1.0 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)

        # Draw smooth boundary contour
        contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 235, 59), 2)

    return overlay


def get_benchmark_samples(samples_dir: str = "./samples", data_dir: str = "./data/kaggle_3m", cache_file: str = "./data/pairs_cache.csv") -> list:
    """
    Finds sample slices with confirmed tumor masks and healthy slices.
    Prioritizes the self-contained ./samples/ directory, falling back to data/kaggle_3m.
    """
    samples_path = Path(samples_dir)
    if samples_path.exists():
        candidates = []
        for p in sorted(samples_path.glob("*.tif")):
            if p.name.endswith("_mask.tif"):
                continue
            mask_p = p.parent / p.name.replace(".tif", "_mask.tif")
            mask_exists = mask_p.exists()
            has_pos = False
            if mask_exists:
                m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                if m is not None and np.sum(m > 0) > 0:
                    has_pos = True
            label_prefix = "Tumor Positive" if has_pos else "Normal / Clear"
            candidates.append({
                "label": f"{label_prefix} — {p.name}",
                "image_path": str(p),
                "mask_path": str(mask_p) if mask_exists else None,
                "type": "positive" if has_pos else "negative"
            })
        if candidates:
            # Sort so positive samples appear first
            candidates.sort(key=lambda x: (0 if x["type"] == "positive" else 1, x["label"]))
            return candidates

    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            pos_df = df[df["has_tumor"] == 1].head(10)
            neg_df = df[df["has_tumor"] == 0].head(5)
            samples = []
            for _, r in pos_df.iterrows():
                samples.append({
                    "label": f"Tumor Positive — {r['patient_id']} ({r['filename']})",
                    "image_path": r["image_path"],
                    "mask_path": r["mask_path"],
                    "type": "positive"
                })
            for _, r in neg_df.iterrows():
                samples.append({
                    "label": f"Normal/Clear — {r['patient_id']} ({r['filename']})",
                    "image_path": r["image_path"],
                    "mask_path": r["mask_path"],
                    "type": "negative"
                })
            return samples
        except Exception:
            pass

    samples = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return samples

    # Fallback to rglob if cache not available
    found_pos = 0
    found_neg = 0
    for p in data_path.rglob("*.tif"):
        if p.name.endswith("_mask.tif"):
            continue
        mask_p = p.parent / p.name.replace(".tif", "_mask.tif")
        if not mask_p.exists():
            continue

        m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        has_pos = np.sum(m > 0) > 100

        if has_pos and found_pos < 8:
            samples.append({
                "label": f"Tumor Positive — {p.parent.name} ({p.name})",
                "image_path": str(p),
                "mask_path": str(mask_p),
                "type": "positive"
            })
            found_pos += 1
        elif not has_pos and found_neg < 4:
            samples.append({
                "label": f"Normal/Clear — {p.parent.name} ({p.name})",
                "image_path": str(p),
                "mask_path": str(mask_p),
                "type": "negative"
            })
            found_neg += 1

        if found_pos >= 8 and found_neg >= 4:
            break

    return samples


def main():
    # --- Sidebar ---
    st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=70)
    st.sidebar.title("NeuroScan AI")
    st.sidebar.markdown("**LGG Brain MRI Tumor Segmentation**")
    st.sidebar.markdown("---")

    # Load Model
    weights_file = "unet_brain_tumor.pth"
    model, device, is_loaded, status_info = load_unet_model(weights_file)

    st.sidebar.subheader("System Status")
    if is_loaded:
        st.sidebar.success(f"Model: {weights_file} Loaded\nDevice: {device}")
    else:
        st.sidebar.warning(f"{status_info}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Inference Settings")
    threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Pixels with model sigmoid probability above this threshold are classified as tumor."
    )

    alpha = st.sidebar.slider(
        "Overlay Transparency (Alpha)",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
    )

    overlay_color_name = st.sidebar.selectbox(
        "Tumor Highlight Color",
        ["Vibrant Red", "Neon Green", "Electric Cyan", "Amber Yellow"],
        index=0
    )
    color_map = {
        "Vibrant Red": (255, 45, 45),
        "Neon Green": (34, 197, 94),
        "Electric Cyan": (6, 182, 212),
        "Amber Yellow": (245, 158, 11),
    }
    selected_color = color_map[overlay_color_name]

    # --- Main Header ---
    st.markdown('<div class="main-header">🧠 NeuroScan AI — Brain Tumor Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Automated Lower-Grade Glioma (LGG) segmentation pipeline utilizing a PyTorch U-Net with skip connections '
        'and joint Binary Cross-Entropy + Dice Loss optimization.'
        '</div>',
        unsafe_allow_html=True
    )

    # --- Input Mode Tabs ---
    tab1, tab2, tab3 = st.tabs(["🔍 Slice Evaluation", "📊 Model Architecture & Loss", "📈 Training History"])

    with tab1:
        st.markdown("#### Select Input MRI Slice")
        input_source = st.radio(
            "Input Mode:",
            ["Pick from Dataset Benchmark Slices", "Upload Custom MRI Slice (.tif / .png / .jpg)"],
            horizontal=True
        )

        selected_image_rgb = None
        selected_gt_mask = None
        slice_name = ""

        if input_source == "Pick from Dataset Benchmark Slices":
            samples = get_benchmark_samples()
            if samples:
                sample_options = {s["label"]: s for s in samples}
                chosen_label = st.selectbox("Choose a benchmark slice:", list(sample_options.keys()))
                chosen_sample = sample_options[chosen_label]
                slice_name = chosen_sample["label"]

                img_bgr = cv2.imread(chosen_sample["image_path"])
                if img_bgr is not None:
                    selected_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                if chosen_sample.get("mask_path") and os.path.exists(chosen_sample["mask_path"]):
                    mask_gray = cv2.imread(chosen_sample["mask_path"], cv2.IMREAD_GRAYSCALE)
                    if mask_gray is not None:
                        selected_gt_mask = cv2.resize(mask_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
                        selected_gt_mask = (selected_gt_mask > 0).astype(np.uint8)
            else:
                st.info("No samples found in `./samples` or `./data/kaggle_3m`. Please upload an MRI slice below.")

        else:
            uploaded_file = st.file_uploader(
                "Upload Brain MRI Scan (.tif, .png, .jpg, .jpeg)",
                type=["tif", "tiff", "png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    selected_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    selected_image_rgb = np.array(pil_img)
                slice_name = uploaded_file.name

        # --- Inference & Display ---
        if selected_image_rgb is not None:
            # Preprocess
            resized_rgb, input_tensor = preprocess_image(selected_image_rgb, target_size=256)

            # Predict
            with torch.no_grad():
                logits = model(input_tensor.to(device))
                probs_tensor = torch.sigmoid(logits).squeeze().cpu()
                probs = probs_tensor.numpy()
                pred_mask = (probs >= threshold).astype(np.uint8)

            # Compute Metrics
            metrics = compute_metrics(
                pred_mask=pred_mask,
                probs=probs,
                resized_img=resized_rgb,
                gt_mask=selected_gt_mask
            )

            # Create Composite Overlay
            overlay_img = create_overlay(
                resized_rgb=resized_rgb,
                pred_mask=pred_mask,
                color=selected_color,
                alpha=alpha
            )

            # Heatmap colormap
            heatmap_colored = (cm.magma(probs)[:, :, :3] * 255).astype(np.uint8)

            st.markdown("---")
            # --- Results KPI Dashboard ---
            st.markdown(f"### Segmentation Analysis: `{slice_name}`")

            # Status Badge
            col_stat, col_m1, col_m2, col_m3, col_m4 = st.columns([1.5, 1, 1, 1, 1])
            with col_stat:
                if metrics["has_tumor"]:
                    st.markdown(
                        '<div style="padding-top: 8px;">'
                        '<span class="badge-positive">⚠️ TUMOR DETECTED</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="padding-top: 8px;">'
                        '<span class="badge-negative">✅ NO TUMOR DETECTED</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

            with col_m1:
                st.metric("Tumor Area (px)", f"{metrics['tumor_pixels']:,} px")
            with col_m2:
                st.metric("Tumor / Brain %", f"{metrics['tumor_pct_brain']:.2f}%")
            with col_m3:
                st.metric("Peak Probability", f"{metrics['max_confidence']:.1%}")
            with col_m4:
                if metrics["has_gt"]:
                    st.metric("Dice Score (vs GT)", f"{metrics['dice_score']:.3f}")
                else:
                    st.metric("Mean Confidence", f"{metrics['mean_confidence']:.1%}")

            if metrics["has_gt"]:
                st.info(
                    f"**Ground Truth Validation:** Overlap Dice Score = **{metrics['dice_score']:.4f}** | "
                    f"IoU (Jaccard Index) = **{metrics['iou_score']:.4f}** | "
                    f"Ground Truth Tumor Area: **{metrics['gt_pixels']:,} px**"
                )

            # --- Synchronized Multi-View Columns ---
            st.markdown("#### Multi-Modal Visual Output")
            cols = st.columns(4 if metrics["has_gt"] else 3)

            with cols[0]:
                st.image(resized_rgb, caption="1. Original MRI Slice (256x256)", use_container_width=True)

            with cols[1]:
                st.image(pred_mask * 255, caption=f"2. Predicted Tumor Mask (Thresh={threshold})", use_container_width=True)

            with cols[2]:
                st.image(overlay_img, caption="3. Clinical Overlay with Tumor Boundary", use_container_width=True)

            if metrics["has_gt"]:
                with cols[3]:
                    st.image(selected_gt_mask * 255, caption="4. Ground Truth Mask (_mask.tif)", use_container_width=True)

            # Secondary inspection row: Probability Heatmap & Histogram
            st.markdown("#### Confidence Heatmap & Pixel Probability Distribution")
            heat_col, hist_col = st.columns(2)
            with heat_col:
                st.image(heatmap_colored, caption="Continuous Sigmoid Confidence Map (Magma)", use_container_width=True)
            with hist_col:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(probs.ravel(), bins=50, color="#3B82F6", alpha=0.8, edgecolor="black")
                ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
                ax.set_title("Pixel Probability Distribution", fontsize=11, fontweight="bold")
                ax.set_xlabel("Sigmoid Output Probability")
                ax.set_ylabel("Pixel Count")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            # Download Options
            st.markdown("#### Export Results")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                # Convert mask to PNG
                pil_mask = Image.fromarray((pred_mask * 255).astype(np.uint8))
                buf_mask = io.BytesIO()
                pil_mask.save(buf_mask, format="PNG")
                st.download_button(
                    label="💾 Download Predicted Mask (PNG)",
                    data=buf_mask.getvalue(),
                    file_name="predicted_tumor_mask.png",
                    mime="image/png"
                )
            with dcol2:
                pil_overlay = Image.fromarray(overlay_img)
                buf_overlay = io.BytesIO()
                pil_overlay.save(buf_overlay, format="PNG")
                st.download_button(
                    label="💾 Download Composite Overlay (PNG)",
                    data=buf_overlay.getvalue(),
                    file_name="tumor_overlay_composite.png",
                    mime="image/png"
                )

        else:
            st.info("👈 Please select a benchmark slice or upload an MRI slice to begin analysis.")

    # --- Tab 2: Architecture & Loss ---
    with tab2:
        st.markdown("### Deep Learning Architecture: U-Net")
        st.markdown("""
        The network employs a standard **encoder-decoder U-Net** featuring:
        - **Contracting Path (Encoder):** 4 downsampling blocks consisting of consecutive $3\\times 3$ Convolutions, Batch Normalization, and ReLU activations, followed by $2\\times 2$ Max Pooling.
        - **Bottleneck:** High-level contextual representation ($512$ feature channels).
        - **Expansive Path (Decoder):** Bilinear upsampling blocks with **skip connections** linking low-level spatial detail from the encoder to the decoder.
        - **Final Layer:** $1\\times 1$ convolution mapping feature maps to single-channel binary tumor logits.
        """)

        st.markdown("### Custom Compound Objective: BCE + Dice Loss")
        st.latex(r"\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}")
        st.latex(r"\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum (p_i \cdot y_i) + \epsilon}{\sum p_i + \sum y_i + \epsilon}")
        st.markdown("""
        - **Binary Cross-Entropy (BCE):** Optimizes pixel-wise classification probabilities independently.
        - **Dice Loss:** Direct surrogate for the Dice Similarity Coefficient (F1-score), resolving severe class imbalance between small tumor regions and vast brain/background areas.
        """)

    # --- Tab 3: Training History ---
    with tab3:
        st.markdown("### Model Training Logs & Convergence")
        history_path = Path("training_history.json")
        if history_path.exists():
            with open(history_path, "r") as f:
                history_data = json.load(f)

            epochs_range = list(range(1, len(history_data["train_loss"]) + 1))

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
                ax_loss.plot(epochs_range, history_data["train_loss"], marker="o", label="Train Loss", color="#EF4444")
                ax_loss.plot(epochs_range, history_data["val_loss"], marker="s", label="Val Loss", color="#3B82F6")
                ax_loss.set_title("Loss Convergence (BCE + Dice)", fontweight="bold")
                ax_loss.set_xlabel("Epoch")
                ax_loss.set_ylabel("Loss")
                ax_loss.legend()
                ax_loss.grid(alpha=0.3)
                st.pyplot(fig_loss)
                plt.close(fig_loss)

            with col_h2:
                fig_dice, ax_dice = plt.subplots(figsize=(6, 4))
                ax_dice.plot(epochs_range, history_data["train_dice"], marker="o", label="Train Dice", color="#10B981")
                ax_dice.plot(epochs_range, history_data["val_dice"], marker="s", label="Val Dice", color="#8B5CF6")
                ax_dice.set_title("Validation Dice Metric", fontweight="bold")
                ax_dice.set_xlabel("Epoch")
                ax_dice.set_ylabel("Dice Score")
                ax_dice.legend()
                ax_dice.grid(alpha=0.3)
                st.pyplot(fig_dice)
                plt.close(fig_dice)

            st.write("**Epoch Summary Table:**")
            import pandas as pd
            summary_df = pd.DataFrame({
                "Epoch": epochs_range,
                "Train Loss": [round(x, 4) for x in history_data["train_loss"]],
                "Val Loss": [round(x, 4) for x in history_data["val_loss"]],
                "Train Dice": [round(x, 4) for x in history_data["train_dice"]],
                "Val Dice": [round(x, 4) for x in history_data["val_dice"]],
                "Val IoU": [round(x, 4) for x in history_data.get("val_iou", [0]*len(epochs_range))],
                "Duration (s)": [round(x, 1) for x in history_data.get("epoch_times", [0]*len(epochs_range))],
            })
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No `training_history.json` detected yet. Run `train.py` to generate training history.")


if __name__ == "__main__":
    main()

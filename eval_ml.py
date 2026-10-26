"""
Diagnostic Model Evaluation Script for NeuroScan AI U-Net
Loads src/model.py, evaluates unet_brain_tumor.pth, inspects parameters,
runs forward inference on sample brain MRI slices, and computes quantitative metrics.
"""

import os
import sys
import time
from pathlib import Path
import importlib.util

# Ensure terminal stdout handles characters gracefully on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import cv2
import torch
import torch.nn.functional as F


# 1. Dynamically load UNet from src/model.py or model.py
def load_unet_class():
    src_path = Path("src/model.py")
    root_path = Path("model.py")

    target_path = src_path if src_path.exists() else root_path
    if not target_path.exists():
        raise FileNotFoundError("Could not locate model.py in src/ or project root.")

    spec = importlib.util.spec_from_file_location("model_module", target_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UNet, str(target_path)


UNet, MODEL_SOURCE_PATH = load_unet_class()


def get_model_parameters(model: torch.nn.Module):
    """Calculates total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return total_params, trainable_params, model_size_mb


def preprocess_slice(image_path: Path, img_size: int = 256):
    """Loads and normalizes an MRI slice to [1, 3, img_size, img_size]."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Failed to read image at: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Standard ImageNet normalization matching training pipeline
    norm = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (norm - mean) / std

    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return resized, tensor


def load_ground_truth_mask(mask_path: Path, img_size: int = 256):
    """Loads and binarizes ground truth mask if available."""
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 0).astype(np.uint8)


def evaluate():
    print("=" * 68)
    print("NEUROSCAN AI - DIAGNOSTIC ML EVALUATION PIPELINE")
    print("=" * 68)

    # --- 1. Model Source & Instantiation ---
    print(f"[*] Loading Model Definition from: {MODEL_SOURCE_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Compute Device: {device}")

    model = UNet(in_channels=3, out_channels=1, init_features=32, bilinear=True)
    total_params, trainable_params, size_mb = get_model_parameters(model)

    print(f"\n--- Model Parameter Analysis ---")
    print(f"  * Architecture:           Standard 4-level U-Net (with skip connections)")
    print(f"  * Base Features:          32 channels")
    print(f"  * Input Channels:         3 (RGB / Multi-sequence MRI)")
    print(f"  * Output Channels:        1 (Binary segmentation logits)")
    print(f"  * Total Parameters:       {total_params:,}")
    print(f"  * Trainable Parameters:   {trainable_params:,}")
    print(f"  * Model In-Memory Size:   {size_mb:.2f} MB")

    # --- 2. Load Trained Weights ---
    weights_path = Path("unet_brain_tumor.pth")
    print(f"\n--- Loading Checkpoint Weights ---")
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint file '{weights_path}' does not exist!")

    checkpoint = torch.load(str(weights_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "N/A")
        val_dice = checkpoint.get("val_dice", "N/A")
        val_loss = checkpoint.get("val_loss", "N/A")
        print(f"  * Weights File:           {weights_path.resolve()}")
        print(f"  * File Size:              {weights_path.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"  * Checkpoint Epoch:       {epoch}")
        if val_dice != "N/A":
            print(f"  * Checkpoint Val Dice:    {val_dice:.4f}")
        if val_loss != "N/A":
            print(f"  * Checkpoint Val Loss:    {val_loss:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  * Loaded raw state dictionary from {weights_path}")

    model.to(device)
    model.eval()

    # --- 3. Sample Acquisition ---
    print(f"\n--- Test Sample Acquisition ---")
    samples_dir = Path("samples")
    chosen_image = None
    chosen_mask = None

    if samples_dir.exists():
        # Look for a positive sample first
        pos_candidates = sorted(samples_dir.glob("*_11.tif")) + sorted(samples_dir.glob("*_10.tif")) + sorted(samples_dir.glob("*.tif"))
        for candidate in pos_candidates:
            if not candidate.name.endswith("_mask.tif"):
                chosen_image = candidate
                mask_candidate = candidate.parent / candidate.name.replace(".tif", "_mask.tif")
                if mask_candidate.exists():
                    chosen_mask = mask_candidate
                break

    if chosen_image is None or not chosen_image.exists():
        print("  [!] No sample files found in samples/. Generating synthetic brain MRI tensor...")
        input_tensor = torch.randn(1, 3, 256, 256)
        resized_img = np.zeros((256, 256, 3), dtype=np.uint8)
        gt_mask = None
        sample_name = "Synthetic Random Tensor"
    else:
        sample_name = chosen_image.name
        print(f"  * Selected Sample Slice:  {sample_name}")
        print(f"  * Path:                   {chosen_image.resolve()}")
        resized_img, input_tensor = preprocess_slice(chosen_image, img_size=256)
        gt_mask = load_ground_truth_mask(chosen_mask, img_size=256) if chosen_mask else None
        if gt_mask is not None:
            print(f"  * Paired Ground Truth:    {chosen_mask.name} ({np.sum(gt_mask):,} positive pixels)")

    # --- 4. PyTorch Forward Inference ---
    print(f"\n--- Executing Forward Inference ---")
    input_tensor = input_tensor.to(device)

    # Warmup pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Timed inference
    latencies = []
    with torch.no_grad():
        for _ in range(5):
            t0 = time.perf_counter()
            logits = model(input_tensor)
            probs_tensor = torch.sigmoid(logits)
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = float(np.mean(latencies))
    probs = probs_tensor.squeeze().cpu().numpy()
    pred_mask = (probs >= 0.50).astype(np.uint8)

    print(f"  * Inference Output Shape: {tuple(logits.shape)}")
    print(f"  * Average Latency:        {avg_latency:.2f} ms")
    print(f"  * Probability Range:      Min={probs.min():.4f}, Max={probs.max():.4f}, Mean={probs.mean():.4f}")

    # --- 5. Quantitative Tumor & Area Metrics ---
    tumor_pixels = int(np.sum(pred_mask))
    total_pixels = int(pred_mask.size)

    # Segment brain tissue area (non-background intracranial space)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY) if resized_img.ndim == 3 else resized_img
    brain_mask = (gray > 15).astype(np.uint8)
    brain_pixels = int(np.sum(brain_mask))

    tumor_pct_slice = (tumor_pixels / total_pixels) * 100.0
    tumor_pct_brain = (tumor_pixels / max(1, brain_pixels)) * 100.0
    has_tumor = tumor_pixels >= 15
    mean_tumor_conf = float(np.mean(probs[pred_mask > 0])) if tumor_pixels > 0 else 0.0

    print(f"\n--- Segmentation & Morphological Metrics ---")
    print(f"  * Total Slice Pixels:     {total_pixels:,} px (256x256)")
    print(f"  * Brain Intracranial px:  {brain_pixels:,} px")
    print(f"  * Tumor Pixel Count:      {tumor_pixels:,} px")
    print(f"  * Tumor Slice Fraction:   {tumor_pct_slice:.2f}%")
    print(f"  * Tumor / Brain Fraction: {tumor_pct_brain:.2f}%")
    print(f"  * Mean Tumor Confidence:  {mean_tumor_conf:.2%}")
    print(f"  * Peak Probability:       {float(np.max(probs)):.2%}")
    verdict = "[!] TUMOR DETECTED" if has_tumor else "[OK] NO TUMOR DETECTED"
    print(f"  * Clinical Verdict:       {verdict}")

    # --- 6. Ground Truth Validation (if applicable) ---
    if gt_mask is not None:
        gt_pixels = int(np.sum(gt_mask))
        intersection = int(np.sum((pred_mask > 0) & (gt_mask > 0)))
        union = int(np.sum((pred_mask > 0) | (gt_mask > 0)))
        dice = (2.0 * intersection) / (tumor_pixels + gt_pixels + 1e-6) if (tumor_pixels + gt_pixels) > 0 else 1.0
        iou = intersection / (union + 1e-6) if union > 0 else 1.0

        print(f"\n--- Ground Truth Validation Metrics ---")
        print(f"  * Ground Truth Tumor px:  {gt_pixels:,} px")
        print(f"  * Intersection (TP):      {intersection:,} px")
        print(f"  * Dice Score (F1):        {dice:.4f}")
        print(f"  * IoU (Jaccard Index):    {iou:.4f}")

    print("\n" + "=" * 68)
    print("SUCCESS: EVALUATION DIAGNOSTIC COMPLETED")
    print("=" * 68)


if __name__ == "__main__":
    evaluate()

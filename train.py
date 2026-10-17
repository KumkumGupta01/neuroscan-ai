"""
Training Pipeline for Brain Tumor Segmentation U-Net
Runs training and validation loops, computes BCEDiceLoss and Dice/IoU metrics,
and saves the trained model weights to unet_brain_tumor.pth.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import UNet
from loss import BCEDiceLoss, dice_coefficient, iou_score
from dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on LGG Brain MRI dataset")
    parser.add_argument("--data-dir", type=str, default="./data/kaggle_3m", help="Path to LGG dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=256, help="Target image height and width")
    parser.add_argument("--init-features", type=int, default=32, help="U-Net base feature multiplier")
    parser.add_argument("--output-weights", type=str, default="unet_brain_tumor.pth", help="Output path for weights")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data reserved for validation")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max dataset subset size for quick test")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_dice = dice_coefficient(outputs, masks)
        batch_iou = iou_score(outputs, masks)

        total_loss += loss.item()
        total_dice += batch_dice
        total_iou += batch_iou

        if (batch_idx + 1) % max(1, num_batches // 5) == 0 or (batch_idx + 1) == num_batches:
            print(
                f"  Batch [{batch_idx + 1}/{num_batches}] - "
                f"Loss: {loss.item():.4f} | Dice: {batch_dice:.4f} | IoU: {batch_iou:.4f}"
            )

    return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou_score(outputs, masks)

    return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("BRAIN TUMOR SEGMENTATION - U-NET TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset directory: {args.data_dir}")
    print(f"Image Resolution: {args.img_size}x{args.img_size}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print(f"Target Output Weights: {args.output_weights}")
    print("=" * 60)

    # 1. Load Data
    print("Loading and preparing dataset pairs...")
    train_loader, val_loader, train_df, val_df = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        img_size=args.img_size,
        random_state=args.seed,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    print(f"Found {len(train_df)} training slices across {train_df['patient_id'].nunique()} patients.")
    print(f"Found {len(val_df)} validation slices across {val_df['patient_id'].nunique()} patients.")

    # 2. Instantiate Model, Loss, Optimizer, Scheduler
    print("\nInitializing U-Net Architecture...")
    model = UNet(
        in_channels=3,
        out_channels=1,
        init_features=args.init_features,
        bilinear=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_dice = -1.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "epoch_times": []
    }

    start_total_time = time.time()
    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.6f}) ---")

        train_loss, train_dice, train_iou = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_dice, val_iou = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        epoch_duration = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["epoch_times"].append(epoch_duration)

        print(
            f"Epoch {epoch}/{args.epochs} Results [{epoch_duration:.1f}s]:\n"
            f"  Train -> Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}\n"
            f"  Val   -> Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}"
        )

        # Checkpoint: Save best or latest
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            print(f"  (*) New best validation Dice score: {best_val_dice:.4f}! Saving checkpoint...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_loss": val_loss,
                "init_features": args.init_features,
                "img_size": args.img_size,
            }, args.output_weights)

    total_training_time = time.time() - start_total_time
    print("\n" + "=" * 60)
    print(f"Training Complete in {total_training_time:.1f}s!")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Model saved to: {os.path.abspath(args.output_weights)}")
    print("=" * 60)

    # Save training history to json
    history_file = Path("training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_file.resolve()}")


if __name__ == "__main__":
    main()

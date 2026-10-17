"""
Custom Loss Functions and Evaluation Metrics for Brain Tumor Segmentation
Combines Binary Cross-Entropy (BCE) with Dice Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Computes the Dice Loss for binary segmentation.
    Dice = (2 * |X & Y| + smooth) / (|X| + |Y| + smooth)
    Loss = 1 - Dice
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs before sigmoid [B, 1, H, W] or [B, H, W]
            targets (torch.Tensor): Ground truth binary mask [B, 1, H, W] or [B, H, W]
        """
        probs = torch.sigmoid(logits)

        # Flatten tensors per batch item: [B, num_pixels]
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    Combines pixel-level distribution learning (BCE) with overlap optimization (Dice).

    Loss = (bce_weight * BCE) + (dice_weight * DiceLoss)
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Calculates Dice similarity coefficient between predicted binary mask and target mask.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        targets = targets.float()

        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        cardinality = preds.sum(dim=1) + targets.sum(dim=1)

        # For cases where both are completely empty (no tumor), score is 1.0
        both_empty = (cardinality == 0)
        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice = torch.where(both_empty, torch.tensor(1.0, device=logits.device), dice)

        return dice.mean().item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Calculates Intersection over Union (Jaccard Index) metric.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        targets = targets.float()

        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

        both_empty = (union == 0)
        iou = (intersection + smooth) / (union + smooth)
        iou = torch.where(both_empty, torch.tensor(1.0, device=logits.device), iou)

        return iou.mean().item()


if __name__ == "__main__":
    # Smoke test for losses and metrics
    pred = torch.randn(4, 1, 256, 256)
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()
    criterion = BCEDiceLoss()
    loss = criterion(pred, target)
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)
    print(f"Loss test passed! Combined Loss: {loss.item():.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}")

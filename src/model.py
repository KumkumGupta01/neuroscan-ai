"""
Standard U-Net Architecture Implementation with Skip Connections for Brain Tumor Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutional layers each followed by
    Batch Normalization and ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with 2x2 MaxPool followed by DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: upsamples the feature map, concatenates the skip connection
    from the corresponding contracting path layer, and applies DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 spatial dimensions (height and width)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        # Concatenate along channel dimension (skip connection)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    1x1 convolution mapping to desired number of classes / output channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Full U-Net model with 4 encoder levels, bottleneck, 4 decoder levels,
    and skip connections.

    Args:
        in_channels (int): Number of input channels (e.g. 3 for RGB MRI slices).
        out_channels (int): Number of output channels (1 for binary tumor mask).
        init_features (int): Base feature multiplier (default: 32).
        bilinear (bool): Whether to use bilinear upsampling or transpose convolution.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        f = init_features

        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, f)          # e.g. 3 -> 32
        self.down1 = Down(f, f * 2)                    # 32 -> 64
        self.down2 = Down(f * 2, f * 4)                # 64 -> 128
        self.down3 = Down(f * 4, f * 8)                # 128 -> 256
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, f * 16 // factor)     # 256 -> 256 (or 512)

        # Expansive Path (Decoder) with Skip Connections
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)
        self.outc = OutConv(f, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass with skip connection caching
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder forward pass with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    # Smoke test for U-Net
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print(f"U-Net smoke test passed! Input shape: {dummy_input.shape}, Output shape: {output.shape}")

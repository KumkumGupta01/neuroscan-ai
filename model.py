"""
Multi-Model Architectures for Brain Tumor Segmentation
Includes:
  1) Standard 4-level U-Net (Selected Production Model - loaded from unet_brain_tumor.pth)
  2) FCN-8s (Fully Convolutional Network baseline)
  3) ResUNet (Deep Residual U-Net with skip residual blocks)
  4) Attention U-Net (U-Net with Attention Gates on skip connections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Standard U-Net (Selected Production Architecture)
# ============================================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
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
    """Downscaling with 2x2 MaxPool followed by DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv with skip connection concatenation"""
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
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution mapping to desired number of classes / logits"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard 4-level U-Net model with skip connections.
    Primary trained production model matching unet_brain_tumor.pth checkpoint.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.name = "Standard 4-Level U-Net"
        f = init_features

        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, f)          # 3 -> 32
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ============================================================================
# 2. FCN-8s (Fully Convolutional Network Baseline)
# ============================================================================

class FCN8s(nn.Module):
    """
    Fully Convolutional Network (FCN-8s) baseline with multi-scale skip fusions
    from pool3 (8x), pool4 (16x), and pool5 (32x) features.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        self.name = "FCN-8s"
        f = init_features

        # Stage 1 (1x -> /2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, kernel_size=3, padding=1),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # Stage 2 (/2 -> /4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(f, f * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 2, f * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # Stage 3 (/4 -> /8, pool3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(f * 2, f * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 4, f * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # Stage 4 (/8 -> /16, pool4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(f * 4, f * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 8, f * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # Stage 5 (/16 -> /32, pool5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 8, f * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        # Fully Convolutional Bottleneck
        self.fc6 = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

        # 1x1 Scoring heads
        self.score_fr = nn.Conv2d(f * 8, out_channels, kernel_size=1)
        self.score_pool4 = nn.Conv2d(f * 8, out_channels, kernel_size=1)
        self.score_pool3 = nn.Conv2d(f * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        p3 = self.conv3(c2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        fc = self.fc7(self.fc6(p5))
        score_fr = self.score_fr(fc)

        # 2x upsample score_fr and fuse with score_pool4
        up_score_fr = F.interpolate(score_fr, size=p4.shape[2:], mode='bilinear', align_corners=True)
        score_p4 = self.score_pool4(p4)
        fuse_p4 = up_score_fr + score_p4

        # 2x upsample fuse_p4 and fuse with score_pool3
        up_fuse_p4 = F.interpolate(fuse_p4, size=p3.shape[2:], mode='bilinear', align_corners=True)
        score_p3 = self.score_pool3(p3)
        fuse_p3 = up_fuse_p4 + score_p3

        # 8x upsample back to input spatial resolution
        out = F.interpolate(fuse_p3, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
        return out


# ============================================================================
# 3. ResUNet (U-Net with Residual Skip Blocks)
# ============================================================================

class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with identity shortcut projection"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv_block(x)
        return self.relu(out + res)


class ResUNet(nn.Module):
    """
    Deep Residual U-Net replacing standard double convolutions with residual blocks.
    Enables deep gradient propagation through residual shortcuts.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        self.name = "ResUNet"
        f = init_features

        # Encoder
        self.enc1 = ResidualBlock(in_channels, f)       # 3 -> 32
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(f, f * 2)             # 32 -> 64
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(f * 2, f * 4)         # 64 -> 128
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(f * 4, f * 8)         # 128 -> 256
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(f * 8, f * 16)  # 256 -> 512

        # Decoder with residual blocks
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResidualBlock(f * 16 + f * 8, f * 8)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ResidualBlock(f * 8 + f * 4, f * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ResidualBlock(f * 4 + f * 2, f * 2)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ResidualBlock(f * 2 + f, f)

        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return self.outc(d4)


# ============================================================================
# 4. Attention U-Net (U-Net with Attention Gates on Skip Connections)
# ============================================================================

class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018).
    Suppresses irrelevant background features while highlighting salient tumor regions
    along the contracting-to-expanding skip pathways.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions if needed
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class AttentionUNet(nn.Module):
    """
    Attention U-Net featuring gating signals and spatial attention filters
    on all 4 skip connections.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        self.name = "Attention U-Net"
        f = init_features

        # Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)

        # Attention Gates
        self.ag1 = AttentionGate(F_g=f * 8, F_l=f * 8, F_int=f * 4)
        self.ag2 = AttentionGate(F_g=f * 4, F_l=f * 4, F_int=f * 2)
        self.ag3 = AttentionGate(F_g=f * 2, F_l=f * 2, F_int=f)
        self.ag4 = AttentionGate(F_g=f, F_l=f, F_int=f // 2)

        # Decoder Up-convolution & DoubleConv
        self.up_conv1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(f * 16, f * 8)

        self.up_conv2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(f * 8, f * 4)

        self.up_conv3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(f * 4, f * 2)

        self.up_conv4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(f * 2, f)

        self.outc = OutConv(f, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with Attention Gates
        d1 = self.up_conv1(x5)
        x4_att = self.ag1(g=d1, x=x4)
        d1 = self.conv1(torch.cat([x4_att, d1], dim=1))

        d2 = self.up_conv2(d1)
        x3_att = self.ag2(g=d2, x=x3)
        d2 = self.conv2(torch.cat([x3_att, d2], dim=1))

        d3 = self.up_conv3(d2)
        x2_att = self.ag3(g=d3, x=x2)
        d3 = self.conv3(torch.cat([x2_att, d3], dim=1))

        d4 = self.up_conv4(d3)
        x1_att = self.ag4(g=d4, x=x1)
        d4 = self.conv4(torch.cat([x1_att, d4], dim=1))

        logits = self.outc(d4)
        return logits


# ============================================================================
# Model Factory / Helper
# ============================================================================

def get_model(model_name: str, in_channels: int = 3, out_channels: int = 1, init_features: int = 32) -> nn.Module:
    """
    Factory to retrieve model architecture by identifier or alias.
    """
    clean_name = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if "fcn" in clean_name:
        return FCN8s(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    elif "res" in clean_name:
        return ResUNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    elif "att" in clean_name:
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    else:
        return UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features, bilinear=True)


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 256, 256)
    models = {
        "Standard U-Net": UNet(3, 1, 32),
        "FCN-8s": FCN8s(3, 1, 32),
        "ResUNet": ResUNet(3, 1, 32),
        "Attention U-Net": AttentionUNet(3, 1, 32)
    }
    print("Testing multi-model forward passes:")
    for name, m in models.items():
        out = m(dummy)
        params = sum(p.numel() for p in m.parameters())
        print(f" • {name:20s}: Output Shape={out.shape}, Parameters={params:,}")

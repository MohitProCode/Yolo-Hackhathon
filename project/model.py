from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetScratch(nn.Module):
    """
    UNet from scratch (no pretrained backbone) with auxiliary output for deep supervision.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        deep_supervision: bool = True,
    ):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.deep_supervision = bool(deep_supervision)

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = ConvBlock(c4, c4 * 2)

        self.up4 = nn.ConvTranspose2d(c4 * 2, c4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(c3, num_classes, kernel_size=1) if self.deep_supervision else None

    def forward(self, x):
        input_size = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        main_logits = self.head(d1)
        if main_logits.shape[2:] != input_size:
            main_logits = F.interpolate(main_logits, size=input_size, mode="bilinear", align_corners=False)

        aux_logits = None
        if self.deep_supervision and self.aux_head is not None:
            aux_logits = self.aux_head(d3)
            aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
        return main_logits, aux_logits


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    name = str(model_cfg.get("name", "unet_scratch")).lower()
    if name != "unet_scratch":
        raise ValueError(f"Unsupported scratch model '{name}'. Expected 'unet_scratch'.")
    return UNetScratch(
        in_channels=int(model_cfg["in_channels"]),
        num_classes=int(cfg["data"]["num_classes"]),
        base_channels=int(model_cfg.get("base_channels", 32)),
        deep_supervision=bool(model_cfg.get("deep_supervision", True)),
    )


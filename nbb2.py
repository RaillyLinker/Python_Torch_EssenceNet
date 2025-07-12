import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


# todo : 1x1 conv 깊이 변경
class ResidualDoubleConvBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, ks, stride, padding, drop_prob, block_size):
        super().__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.proj = (
            nn.Identity() if in_ch == out_ch and stride == 1 else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.act = nn.SiLU()

    def forward(self, x):
        skip = self.proj(x)
        x = self.act(x)
        out = self.conv_seq(x)
        return out + skip


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # todo conv 채널 변경
        self.feats_convs = nn.ModuleList([
            ResidualDoubleConvBlock(3, 32, 16, 3, 1, 1, 0.0, 1),  # 243x243 -> 243x243
            ResidualDoubleConvBlock(16, 64, 32, 3, 3, 0, 0.0, 1),  # 243x243 -> 81x81
            ResidualDoubleConvBlock(32, 128, 64, 3, 3, 0, 0.15, 5),  # 81x81 -> 27x27
            ResidualDoubleConvBlock(64, 256, 128, 3, 3, 0, 0.20, 5),  # 27x27 -> 9x9
            ResidualDoubleConvBlock(128, 512, 256, 3, 3, 0, 0.20, 3),  # 9x9 -> 3x3
            ResidualDoubleConvBlock(256, 1024, 512, 3, 1, 0, 0.0, 1)  # 3x3 -> 1x1
        ])

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."
        feats_list = []

        feat = x
        for feats_conv in self.feats_convs:
            res_out = feats_conv(feat)
            feats_list.append(res_out)
            feat = res_out
        return feats_list


# ----------------------------------------------------------------------------------------------------------------------
class EssenceNetSegmenter(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = EssenceNet()

        # 백본 특징맵 피라미드 채널 수
        self.feat_channels = [16, 32, 64, 128, 256, 512]
        self.encoder_input = sum(self.feat_channels)

        # 분류기 헤드
        # todo : 1x1 conv 깊이 변경
        hidden_ch = self.encoder_input // 2
        self.classifier_head = nn.Sequential(
            nn.Conv2d(self.encoder_input, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(hidden_ch, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feats = self.backbone(x)

        target_size = feats[0].shape[2:]

        concat_feats = torch.cat([F.interpolate(f, size=target_size, mode='nearest') for f in feats], dim=1)

        logits = self.classifier_head(concat_feats)

        return logits

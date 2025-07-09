import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


def _single_conv_block(in_ch, out_ch, ks, strd, pdd):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )


# todo : 1x1 conv 깊이 변경
def _double_conv_block(in_ch, mid_ch, out_ch, ks, strd, pdd, dp, bs):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(),

        DropBlock2D(drop_prob=dp, block_size=bs),

        # 픽셀별 의미 추출
        nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )


# todo : 처음엔 kirsch 를 적용 + 3배수로 처리해보기(243, 81, 27, 9, 3)
# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 흑백 변환 가중치 저장
        self.register_buffer("rgb2gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

        # todo conv 채널 변경
        self.feats_convs = nn.ModuleList([
            _single_conv_block(1, 8, 3, 2, 1),  # 320x320 -> 160x160
            _single_conv_block(8, 16, 3, 2, 1),  # 160x160 -> 80x80
            _single_conv_block(16, 32, 3, 2, 1),  # 80x80 -> 40x40
            _single_conv_block(32, 64, 3, 2, 1),  # 40x40 -> 20x20
            _double_conv_block(64, 256, 128, 3, 2, 1, 0.1, 3),  # 20x20 -> 10x10
            _double_conv_block(128, 512, 256, 3, 2, 1, 0.1, 2),  # 10x10 -> 5x5
            _double_conv_block(256, 1024, 512, 3, 2, 1, 0.1, 2),  # 5x5 -> 3x3
            _double_conv_block(512, 2048, 1024, 3, 1, 0, 0.0, 1),  # 3x3 -> 1x1
        ])

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."
        feats_list = []

        # 순수 하게 CNN 형태 분석을 위한 흑백 변환
        gray_feats = (x * self.rgb2gray.to(x.device, x.dtype)).sum(dim=1, keepdim=True)

        # 컬러 이미지(해상도 반토막)
        color_feats = torch.nn.functional.interpolate(x, size=(x.shape[2] // 2, x.shape[3] // 2), mode='area')
        feats_list.append(color_feats)

        feat = gray_feats
        for conv in self.feats_convs:
            feat = conv(feat)
            feats_list.append(feat)

        return feats_list


# ----------------------------------------------------------------------------------------------------------------------
class EssenceNetSegmenter(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = EssenceNet()

        # 백본 특징맵 피라미드 채널 수
        self.feat_channels = [3, 8, 16, 32, 64, 128, 256, 512, 1024]  # RGB + 8개 피쳐맵
        self.encoder_input = sum(self.feat_channels)

        # 분류기 헤드
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

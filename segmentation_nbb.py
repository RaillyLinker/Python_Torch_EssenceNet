import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


# todo : 다음 실험 :  conv 증량
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def _single_conv_block(in_ch, out_ch, ks, strd, pdd):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(out_ch),
        Swish()
    )


def _double_conv_block(in_ch, mid_ch, out_ch, ks, strd, pdd, dp, bs):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(mid_ch),
        Swish(),

        # 픽셀별 의미 Projection(희소한 특징 압축)
        nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        Swish(),

        DropBlock2D(drop_prob=dp, block_size=bs)
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_gray = nn.Sequential(
            # 평면당 형태를 파악
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

        self.feats_convs = nn.ModuleList([
            _single_conv_block(1, 16, 3, 2, 1),  # 320x320 -> 160x160
            _double_conv_block(16, 48, 24, 3, 2, 1, 0.05, 3),  # 160x160 -> 80x80
            _double_conv_block(24, 64, 32, 3, 2, 1, 0.10, 3),  # 80x80 -> 40x40
            _double_conv_block(32, 96, 48, 3, 2, 1, 0.15, 5),  # 40x40 -> 20x20
            _double_conv_block(48, 128, 64, 3, 2, 1, 0.20, 5),  # 20x20 -> 10x10
            _double_conv_block(64, 192, 96, 3, 2, 1, 0.20, 3),  # 10x10 -> 5x5
            _double_conv_block(96, 256, 128, 3, 2, 1, 0.15, 3),  # 5x5 -> 3x3
            _double_conv_block(128, 384, 192, 3, 1, 0, 0.0, 1)  # 3x3 -> 1x1
        ])

        # 특징맵 피라미드 채널 수
        def get_last_conv_out_channels(block: nn.Module) -> int:
            for layer in reversed(list(block.children())):
                if isinstance(layer, nn.Conv2d):
                    return layer.out_channels
            raise ValueError("No Conv2d layer found in block")

        encoder_input = sum([
            get_last_conv_out_channels(conv)
            for conv in self.feats_convs
        ])

        # 인코더 헤드
        self.encoder_output = 1280
        self.encoder_head = nn.Sequential(
            nn.Conv2d(encoder_input + 3, self.encoder_output, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.encoder_output),
            Swish(),

            nn.Dropout2d(0.2)
        )

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."
        feats_list = []

        # 순수한 형태 분석을 위한 흑백 변환
        gray_feats = self.to_gray(x)

        # conv 형태 분석
        feat = gray_feats
        for idx, conv in enumerate(self.feats_convs):
            # Conv 연산
            feat = conv(feat)

            # 특징 저장
            feats_list.append(feat)

        # 특징맵 피라미드 업샘플링
        shape_feats = [F.interpolate(f, size=feats_list[0].shape[2:], mode='nearest') for f in feats_list]

        # 색 특징 다운 샘플링
        color_feats = F.adaptive_avg_pool2d(x, output_size=(shape_feats[0].shape[2], shape_feats[0].shape[3]))

        # 색 특징 + 형태 특징 결합
        concat_feats = torch.cat([color_feats, *shape_feats], dim=1)

        # 이미지 특징 최종 Projection
        return self.encoder_head(concat_feats)


# ----------------------------------------------------------------------------------------------------------------------
# (Segment 모델)
class EssenceNetSegmenter(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # conv 백본 모델
        self.backbone = EssenceNet()

        # 백본 출력 채널
        backbone_output_ch = self.backbone.encoder_output

        # 분류기 헤드
        hidden = backbone_output_ch // 2
        self.classifier_head = nn.Sequential(
            nn.Conv2d(backbone_output_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            Swish(),

            nn.Dropout2d(0.2),

            nn.Conv2d(hidden, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 백본 특징맵 피라미드 추출
        feats = self.backbone(x)

        # 픽셀별 분류 헤드 적용
        logits = self.classifier_head(feats)

        return logits

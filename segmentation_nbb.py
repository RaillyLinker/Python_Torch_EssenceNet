import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


# SiLU 함수(ONNX 변환시 호환 되도록 직접 구현)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 일반 conv + 1x1 conv 블록(형태 특징 추출 후 채널 압축, 의미 projection)
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

        # 오버피팅 방지를 위한 DropBlock
        DropBlock2D(drop_prob=dp, block_size=bs)
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 모델 입력 이미지 사이즈
        self.input_img_dim = (3, 320, 320)

        # todo : 채널 수 변경해보기
        # 특징맵 레이어(중간 결과물들을 전부 사용하는 특징맵 피라미드 구조)
        self.feats_convs = nn.ModuleList([
            _double_conv_block(3, 96, 48, 3, 2, 1, 0.0, 1),  # 320x320 -> 160x160
            _double_conv_block(48, 128, 64, 3, 2, 1, 0.05, 3),  # 160x160 -> 80x80
            _double_conv_block(64, 192, 96, 3, 2, 1, 0.10, 3),  # 80x80 -> 40x40
            _double_conv_block(96, 256, 128, 3, 2, 1, 0.15, 5),  # 40x40 -> 20x20
            _double_conv_block(128, 384, 192, 3, 2, 1, 0.20, 5),  # 20x20 -> 10x10
            _double_conv_block(192, 512, 256, 3, 2, 1, 0.20, 3),  # 10x10 -> 5x5
            _double_conv_block(256, 768, 384, 3, 2, 1, 0.15, 3),  # 5x5 -> 3x3
            _double_conv_block(384, 1048, 512, 3, 1, 0, 0.0, 1)  # 3x3 -> 1x1
        ])

        # 외부에 공개할 피쳐 피라미드 shapes
        self.output_shapes = [
            (160, 160, 48),
            (80, 80, 64),
            (40, 40, 96),
            (20, 20, 128),
            (10, 10, 192),
            (5, 5, 256),
            (3, 3, 384),
            (1, 1, 512)
        ]

    def forward(self, x):
        # 특징맵 피라미드 리스트
        feats_list = []

        # conv 형태 분석 시작
        feat = x
        for conv in self.feats_convs:
            # Conv 연산
            feat = conv(feat)

            # 특징 저장
            feats_list.append(feat)

        # 이미지 특징맵 피라미드 반환
        return feats_list


# ----------------------------------------------------------------------------------------------------------------------
# (Segment 모델)
class EssenceNetSegmenter(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # conv 백본 모델
        self.backbone = EssenceNet()

        # 전체 출력 채널 합산
        backbone_output_ch = sum([shape[2] for shape in self.backbone.output_shapes])

        # 업샘플링 기준이 되는 특징 사이즈(= 특징맵 피라미드에서 가장 큰 해상도)
        self.anchor_feat_size = self.backbone.output_shapes[0][:2]

        # 분류기 헤드
        hidden = 1280
        self.classifier_head = nn.Sequential(
            # 픽셀 특징 Projection
            nn.Conv2d(backbone_output_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            Swish(),

            nn.Dropout2d(0.2),

            # 픽셀 분류
            nn.Conv2d(hidden, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 백본 특징맵 피라미드 추출
        feats_list = self.backbone(x)

        # 특징맵 피라미드 업샘플링 및 결합
        concat_feats = torch.cat(
            [F.interpolate(f, size=self.anchor_feat_size, mode='nearest') for f in feats_list], dim=1
        )

        # 픽셀 단위 분류 헤드 적용
        logits = self.classifier_head(concat_feats)

        # logits 을 입력 이미지 크기로 업샘플링
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


# SiLU 함수(ONNX 변환시 호환 되도록 직접 구현)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 일반 conv 한개의 블록(최초 데이터 형태 특징 추출)
def _single_conv_block(in_ch, out_ch, ks, strd, pdd):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(out_ch),
        Swish()
    )


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

        # 백본 필요 이미지 형태
        self.input_img_dim = (3, 320, 320)

        # todo : 증량해보기
        # 특징맵 레이어(중간 결과물들을 전부 사용하는 특징맵 피라미드 구조)
        self.feats_convs = nn.ModuleList([
            _single_conv_block(3, 48, 3, 2, 1),  # 320x320 -> 160x160
            _double_conv_block(48, 128, 64, 3, 2, 1, 0.05, 3),  # 160x160 -> 80x80
            _double_conv_block(64, 192, 96, 3, 2, 1, 0.10, 3),  # 80x80 -> 40x40
            _double_conv_block(96, 256, 128, 3, 2, 1, 0.15, 5),  # 40x40 -> 20x20
            _double_conv_block(128, 384, 192, 3, 2, 1, 0.20, 5),  # 20x20 -> 10x10
            _double_conv_block(192, 512, 256, 3, 2, 1, 0.20, 3),  # 10x10 -> 5x5
            _double_conv_block(256, 768, 384, 3, 2, 1, 0.15, 3),  # 5x5 -> 3x3
            _double_conv_block(384, 1024, 512, 3, 1, 0, 0.0, 1)  # 3x3 -> 1x1
        ])

    def forward(self, x):
        # 입력 이미지의 크기 및 채널 수 검증
        actual_shape = x.shape[1:]  # (C, H, W)
        assert actual_shape == self.input_img_dim, \
            f"Input tensor must have shape {self.input_img_dim}, but got {actual_shape}."

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

        # 모델 입력 이미지 사이즈
        self.input_img_dim = self.backbone.input_img_dim

        with torch.no_grad():
            # 백본 출력 형태 파악을 위한 더미 데이터 입력 및 출력값 저장
            dummy_input = torch.zeros(2, *self.input_img_dim)
            feats_list = self.backbone(dummy_input)

            # (H, W, C) 형태의 List 로 저장
            # ex :
            # [
            #     (160, 160, 32),   # 첫 번째 conv block 출력: 320x320 -> 160x160, 채널 32
            #     ...
            #     (3, 3, 384)       # 마지막 conv block 출력: 3x3 -> 1x1 stride=1 유지, 채널 384 (해상도 약간 다를 수도 있음)
            # ]
            self.backbone_feat_shapes = [(f.shape[2], f.shape[3], f.shape[1]) for f in feats_list]

            # 전체 출력 채널 합산
            backbone_output_ch = sum([shape[2] for shape in self.backbone_feat_shapes])

        # 분류기 헤드
        hidden = 1280
        self.classifier_head = nn.Sequential(
            nn.Conv2d(backbone_output_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            Swish(),

            nn.Dropout2d(0.2),

            nn.Conv2d(hidden, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 입력 이미지의 크기 및 채널 수 검증
        actual_shape = x.shape[1:]  # (C, H, W)
        assert actual_shape == self.input_img_dim, \
            f"Input tensor must have shape {self.input_img_dim}, but got {actual_shape}."

        # 백본 특징맵 피라미드 추출
        feats_list = self.backbone(x)

        # 특징맵 피라미드 업샘플링 및 결합
        concat_feats = torch.cat(
            [F.interpolate(f, size=self.backbone_feat_shapes[0][:2], mode='nearest') for f in feats_list],
            dim=1
        )

        # 픽셀별 분류 헤드 적용
        logits = self.classifier_head(concat_feats)

        # logits 을 입력 이미지 크기로 업샘플링
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        return logits

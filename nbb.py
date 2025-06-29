import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


def _single_conv_block(in_ch, mid_ch, out_ch, ks, strd, pdd, drop_prob, block_size):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(),

        DropBlock2D(drop_prob=drop_prob, block_size=block_size),

        # 채널간 패턴 분석
        nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("rgb2gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

        # 구역별 멀티 스케일 분석
        # todo : ch 크기 변경
        self.feats_conv1 = _single_conv_block(1, 256, 64, 3, 3, 0, 0.05, 5)  # 243x243 -> 81x81
        self.feats_conv2 = _single_conv_block(1, 512, 128, 9, 9, 0, 0.1, 5)  # 243x243 -> 27x27
        self.feats_conv3 = _single_conv_block(1, 1024, 256, 27, 27, 0, 0.15, 3)  # 243x243 -> 9x9
        self.feats_conv4 = _single_conv_block(1, 2048, 512, 81, 81, 0, 0.2, 3)  # 243x243 -> 3x3
        self.feats_conv5 = _single_conv_block(1, 4096, 1024, 243, 243, 0, 0.0, 1)  # 243x243 -> 1x1

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        # 입력값 사이즈
        _, _, h, w = x.shape
        # 입력 값 3 배수 사이즈로 제한 (ex : 3, 9, 27, 81, 243, 729, ...)
        assert h == w and h % 3 == 0, "Input size must be square and divisible by 3"

        # 컬러 이미지
        # (B, 3, h, w)
        color_feats = x

        # 순수 하게 CNN 형태 분석을 위한 흑백 변환
        # (B, 1, h, w)
        gray_feats = (color_feats * self.rgb2gray.to(x.device, x.dtype)).sum(dim=1, keepdim=True)

        # 특징 저장 리스트(사이즈 별 특징 피라미트, 지역 -> 전역)
        result_feats_list = []

        # 3x3 커널 특징 추출
        k3_feats = self.feats_conv1(gray_feats)
        _, _, gh, gw = k3_feats.shape
        result_feats_list.append(
            torch.cat([F.interpolate(color_feats, size=(gh, gw), mode='area'), k3_feats], dim=1)
        )

        # 9x9 커널 특징 추출
        k9_feats = self.feats_conv2(gray_feats)
        _, _, gh, gw = k9_feats.shape
        result_feats_list.append(
            torch.cat([F.interpolate(color_feats, size=(gh, gw), mode='area'), k9_feats], dim=1)
        )

        # 27x27 커널 특징 추출
        k27_feats = self.feats_conv3(gray_feats)
        _, _, gh, gw = k27_feats.shape
        result_feats_list.append(
            torch.cat([F.interpolate(color_feats, size=(gh, gw), mode='area'), k27_feats], dim=1)
        )

        # 81x81 커널 특징 추출
        k81_feats = self.feats_conv4(gray_feats)
        _, _, gh, gw = k81_feats.shape
        result_feats_list.append(
            torch.cat([F.interpolate(color_feats, size=(gh, gw), mode='area'), k81_feats], dim=1)
        )

        # 243x243 커널 특징 추출
        k243_feats = self.feats_conv5(gray_feats)
        _, _, gh, gw = k243_feats.shape
        result_feats_list.append(
            torch.cat([F.interpolate(color_feats, size=(gh, gw), mode='area'), k243_feats], dim=1)
        )

        # 가장 큰 해상도 (첫 피쳐맵 기준)
        target_h, target_w = result_feats_list[0].shape[2:]

        # 모든 피쳐맵을 가장 큰 해상도로 업샘플링 후 concat
        pyramid_concat = torch.cat([
            F.interpolate(feat, size=(target_h, target_w), mode='nearest')
            for feat in result_feats_list
        ], dim=1)

        return pyramid_concat


# ----------------------------------------------------------------------------------------------------------------------
class EssenceNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EssenceNet()

        # 임시 입력으로 채널 수와 해상도 계산
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 243, 243)
            concat_feats = self.backbone(dummy_input)  # 이제는 하나의 tensor
            self.feat_dim = concat_feats.shape[1]  # (B, C_total, H, W)

        # todo : 히든 벡터 사이즈 변경
        # 분류기 히든 벡터 사이즈
        classifier_hidden_vector_size = num_classes * 4
        classifier_hidden_vector_size1 = num_classes * 2

        # todo : 레이어 깊이 변경
        # 픽셀별 벡터 → 클래스 logits (MLP처럼 1x1 Conv)
        self.head = nn.Sequential(
            nn.Conv2d(self.feat_dim, classifier_hidden_vector_size, kernel_size=1),
            nn.BatchNorm2d(classifier_hidden_vector_size),
            nn.SiLU(),
            nn.Dropout(p=0.5),  # Dropout 확률 조절 가능

            nn.Conv2d(classifier_hidden_vector_size, classifier_hidden_vector_size1, kernel_size=1),
            nn.BatchNorm2d(classifier_hidden_vector_size1),
            nn.SiLU(),
            nn.Dropout(p=0.5),  # Dropout 확률 조절 가능

            nn.Conv2d(classifier_hidden_vector_size1, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # (B, C_total, H, W)
        concat_feats = self.backbone(x)

        # 1x1 로 다운샘플링 (area : 평균)
        down_feats = F.interpolate(concat_feats, size=(1, 1), mode='area')

        # 1x1 conv 로 픽셀별 logits
        logits = self.head(down_feats)  # (B, num_classes, H/2, W/2)

        return logits.view(logits.size(0), -1)  # (B, num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


def _single_conv_block(in_ch, mid_ch, out_ch, ks, strd, pdd, dp, bs):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(),

        DropBlock2D(drop_prob=dp, block_size=bs),

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

        # todo 채널 변경
        # 구역별 멀티 스케일 분석
        # 2D 데이터의 형태적 특징을 멀티 스케일로 추출 하기 위해 1채널 오리진 정보를 N번 입력
        # 전역적 정보를 지역적 정보로 투영하기 위해 커널이 큰 순서대로 배치(ex : 이 점은 책장 안의 책 안의 글자 안의 곡선 안에 속한 점이다.(즉, 지역은 전역에 속함))
        self.feats_convs = nn.ModuleList([
            _single_conv_block(1, 4096, 2048, 256, 256, 0, 0.0, 1),  # 256x256 -> 1x1
            _single_conv_block(1, 2048, 1024, 128, 128, 0, 0.3, 2),  # 256x256 -> 2x2
            _single_conv_block(1, 1024, 512, 64, 64, 0, 0.25, 3),  # 256x256 -> 4x4
            _single_conv_block(1, 512, 256, 32, 32, 0, 0.2, 5),  # 256x256 -> 8x8
            _single_conv_block(1, 256, 128, 16, 16, 0, 0.15, 5),  # 256x256 -> 16x16
            _single_conv_block(1, 128, 64, 8, 8, 0, 0.1, 5),  # 256x256 -> 32x32
            _single_conv_block(1, 64, 32, 4, 4, 0, 0.05, 5),  # 256x256 -> 64x64
            _single_conv_block(1, 32, 16, 3, 2, 1, 0.05, 5),  # 256x256 -> 128x128
        ])

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        # 입력값 사이즈
        _, _, h, w = x.shape
        assert h == w and h == 256, "Input must be 256x256"

        # 컬러 이미지
        # (B, 3, h, w)
        color_feats = x

        # 순수 하게 CNN 형태 분석을 위한 흑백 변환
        # (B, 1, h, w)
        gray_feats = (color_feats * self.rgb2gray.to(x.device, x.dtype)).sum(dim=1, keepdim=True)

        concat_pyramid_feats = None
        target_h = target_w = None

        for idx, conv in enumerate(self.feats_convs):
            k_feats = conv(gray_feats)
            _, _, gh, gw = k_feats.shape

            resized_color = F.interpolate(color_feats, size=(gh, gw), mode='area')
            feat = torch.cat([resized_color, k_feats], dim=1)

            if idx == 0:
                # 첫 번째 피처맵 해상도 저장
                target_h, target_w = feat.shape[2:]
                concat_pyramid_feats = feat  # 초기값
            else:
                feat = F.interpolate(feat, size=(target_h, target_w), mode='nearest')
                concat_pyramid_feats = torch.cat([concat_pyramid_feats, feat], dim=1)

        return concat_pyramid_feats


# ----------------------------------------------------------------------------------------------------------------------
class EssenceNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EssenceNet()

        # 임시 입력으로 채널 수와 해상도 계산
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 256, 256)
            concat_feats = self.backbone(dummy_input)  # 이제는 하나의 tensor
            self.feat_dim = concat_feats.shape[1]  # (B, C_total, H, W)

        # todo : 히든 벡터 사이즈 변경
        # 분류기 히든 벡터 사이즈
        classifier_hidden_vector_size = num_classes * 2

        # todo : 레이어 깊이 변경
        # 픽셀별 벡터 → 클래스 logits (MLP처럼 1x1 Conv)
        self.head = nn.Sequential(
            nn.Conv2d(self.feat_dim, classifier_hidden_vector_size, kernel_size=1),
            nn.BatchNorm2d(classifier_hidden_vector_size),
            nn.SiLU(),
            nn.Dropout(p=0.5),  # Dropout 확률 조절 가능

            nn.Conv2d(classifier_hidden_vector_size, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # (B, C_total, H, W)
        concat_feats = self.backbone(x)

        # 1x1 로 다운샘플링 (area : 평균)
        down_feats = F.interpolate(concat_feats, size=(1, 1), mode='area')

        # 1x1 conv 로 픽셀별 logits
        logits = self.head(down_feats)  # (B, num_classes, 1, 1)

        return logits.view(logits.size(0), -1)  # (B, num_classes)

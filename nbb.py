import torch
import torch.nn as nn
import torch.nn.functional as F


def _single_conv_block(in_ch, out_ch, ks, strd, pdd):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )


# (EssenceNet 백본)
# 기본 conv 로 흑백 1채널 이미지의 2차원 위치적 특징을 벡터 형태로 추출 하는 것에 집중
# 좁은 범위 고해상도 -> 넓은 범위 저해상도로 피쳐 피라미드 반환
# 글로벌 피쳐와 디테일 피쳐간 조합으로 모든 형태를 표현하는 개념(저해상도를 interpolate nearest 로 업스케일링하여 고해상도와 조립)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 레이어 하나당 커널수(2배수)
        self.exp_space = 64

        self.feat_blocks = [
            (_single_conv_block(1, self.exp_space, 3, 1, 1), 256),
            (_single_conv_block(1, self.exp_space, 3, 1, 1), 128),
            (_single_conv_block(1, self.exp_space, 3, 1, 1), 64),
            (_single_conv_block(1, self.exp_space, 3, 1, 1), 32),
            (_single_conv_block(1, self.exp_space, 3, 1, 1), 16),
            (_single_conv_block(1, self.exp_space, 3, 1, 1), 8),
            (_single_conv_block(1, self.exp_space, 3, 1, 1), 4),
            (_single_conv_block(1, self.exp_space, 3, 1, 1), 2),
        ]

        # 3x3 -> 1x1
        self.global_conv = _single_conv_block(1, self.exp_space, 3, 1, 0)

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        # 컬러 이미지
        # (B, 3, 256, 256)
        color_feats = x

        # 형태 분석을 위한 흑백 이미지
        # (B, 1, 256, 256)
        gray_feats = (
            (color_feats * torch.tensor(
                [0.299, 0.587, 0.114], device=color_feats.device).view(1, 3, 1, 1)
             ).sum(dim=1, keepdim=True)
        )

        # 특징 저장 리스트(사이즈 별 특징 피라미트, 지역 -> 전역)
        result_feats_list = []

        # 멀티 스케일 특징 추출
        # (B, self.exp_space, N, N)
        for conv_block, size in self.feat_blocks:
            # (B, self.exp_space, size, size)
            conv_feats = conv_block(F.interpolate(gray_feats, size=(size, size), mode='area'))
            # 채널 개수 : 3 + self.exp_space
            conv_feats = torch.cat([F.interpolate(color_feats, size=(size, size), mode='area'), conv_feats], dim=1)
            result_feats_list.append(conv_feats)

        # global 특징 추출
        # (B, self.exp_space, 1, 1)
        conv_feats = self.global_conv(F.interpolate(gray_feats, size=(3, 3), mode='area'))
        # 채널 개수 : 3 + self.exp_space
        conv_feats = torch.cat([F.interpolate(color_feats, size=(1, 1), mode='area'), conv_feats], dim=1)
        result_feats_list.append(conv_feats)

        return result_feats_list


# ----------------------------------------------------------------------------------------------------------------------
# todo : EssenceNet 에서 반환된 피쳐 피라미드를 어텐션으로 헤드 구성
class EssenceNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EssenceNet()

        dummy_input = torch.zeros(2, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy_input)

        out_channels_list = [f.shape[1] for f in feats]
        total_channels = sum(out_channels_list)

        hidden_dim = total_channels // 2

        self.fc = nn.Sequential(
            nn.Linear(total_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        pooled_feats = []
        # 전체 공간에서 모든 특징에 대해 평균 값을 구해서 MLP 로 파악합니다.
        # 만약 일정 범위라면 해당 범위에 해당하는 구역의 값을 평균내서 일자로 만들어 파악하면 되며,
        # 픽셀단위 이미지 세그먼트는 픽셀단위 일자로 분리해 판단
        for f in feats:
            if f.shape[2:] != (1, 1):
                p = F.adaptive_avg_pool2d(f, 1)
            else:
                p = f
            pooled_feats.append(p)

        concat_feat = torch.cat(pooled_feats, dim=1)
        flatten_feat = concat_feat.view(concat_feat.size(0), -1)
        logits = self.fc(flatten_feat)
        return logits


class EssenceNetSegClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EssenceNet()
        self.channels = [3, 16, 24, 32, 48, 64, 80, 96, 112]
        total_channels = sum(self.channels)
        self.num_classes = num_classes

        # 1x1 Convolution to produce (num_classes + 1) channel map
        self.pixel_classifier = nn.Conv2d(
            in_channels=total_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):  # x: (B, 3, 320, 320)
        feats = self.backbone(x)  # List of feature maps
        upsampled_feats = [F.interpolate(f, size=(320, 320), mode='nearest') for f in feats]
        concat_feats = torch.cat(upsampled_feats, dim=1)  # (B, C_total, 320, 320)

        class_logits = self.pixel_classifier(concat_feats)  # (B, num_classes, 320, 320)
        return class_logits

# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
#
#
# class UpsampleConcatClassifier(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#
#         # EfficientNet-B0 로드 (특징 추출기만 사용)
#         self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
#         self.backbone_features = self.backbone.features  # (B, 1280, 10, 10)
#
#         # dummy input으로 출력 크기 확인
#         dummy_input = torch.zeros(1, 3, 224, 224)
#         with torch.no_grad():
#             backbone_out = self.backbone_features(dummy_input)
#         _, ch, h, w = backbone_out.shape
#
#         self.classifier = nn.Sequential(
#             nn.Conv2d(ch, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.SiLU(),
#             nn.AdaptiveAvgPool2d(1),  # (B, 256, 1, 1)
#             nn.Flatten(),  # (B, 256)
#             nn.Linear(256, num_classes)
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.backbone_features(x)
#         x = self.classifier(x)
#         return x

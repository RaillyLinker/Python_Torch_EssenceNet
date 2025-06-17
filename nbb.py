import torch
import torch.nn as nn
import torch.nn.functional as F


class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 멀티 스케일 + 복합 형태 정보
        self.comp_feat_blocks = nn.ModuleList([
            # 3x3 픽셀 단위 특징 검출
            # 정보가 혼재된 노이즈 신호
            self._double_conv_block(1, 32, 16, 3, 2, 1),  # 320x320 -> 160x160
            # 7x7 픽셀 단위 특징 검출
            # 노이즈 + 점의 의미
            self._depthwise_separable_conv_block(16, 64, 3, 2, 1),  # 160x160 -> 80x80
            # 15x15 픽셀 단위 특징 검출
            # 점 + 작은 단위 선
            self._depthwise_separable_conv_block(64, 96, 3, 2, 1),  # 80x80 -> 40x40
            # 31x31 픽셀 단위 특징 검출
            # 점 + 자유로운 선 + 최소 단위 도형
            self._depthwise_separable_conv_block(96, 128, 3, 2, 1),  # 40x40 -> 20x20
            # 63x63 픽셀 단위 특징 검출
            # 점 + 자유로운 선 + 자유로운 도형
            self._depthwise_separable_conv_block(128, 160, 3, 2, 1),  # 20x20 -> 10x10
            # 127x127 픽셀 단위 특징 검출
            # 점 + 자유로운 선 + 자유로운 도형 + 질감
            # 실사 이미지 표현이 가능한 영역이며, 아래부터는 추상적 정보
            self._depthwise_separable_conv_block(160, 224, 3, 2, 1),  # 10x10 -> 5x5
            # 255x255 픽셀 단위 특징 검출
            self._depthwise_separable_conv_block(224, 256, 3, 2, 1),  # 5x5 -> 3x3
            # 320x320 픽셀 단위 특징 검출
            self._depthwise_separable_conv_block(256, 512, 3, 1, 0)  # 3x3 -> 1x1
        ])

    def _single_conv_block(self, in_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            # 평면당 형태를 파악
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def _double_conv_block(self, in_ch, mid_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            # 평면당 형태를 파악
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),

            # 채널간 패턴 분석
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    # Ghost-Net 방식으로 핵심 정보를 가진 첫째 레이어에서부터 시작해서 다음으로는 Depthwise-Pointwise 구조로 정보 추출
    # todo : 만약 이 방식의 정확도가 나오지 않는다면 전체 double_conv_block 적용(경량화 필요)
    def _depthwise_separable_conv_block(self, in_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_ch, in_ch, kernel_size=ks, stride=strd, padding=pdd, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),

            # Pointwise convolution (1x1)
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        result_feats_list = []
        target_h, target_w = x.shape[2] // 2, x.shape[3] // 2

        # 컬러 특징 저장(160x160)
        result_feats_list.append(F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False))

        # 분석을 위한 흑백 변환(320x320)
        # 컬러 이미지는 그 자체로 색이란 특징을 지닌 특징 맵이고, 형태 특징을 구하기 위한 입력 값은 흑백으로 충분
        # 1x1 conv 를 굳이 할 필요 없이 검증된 알고리즘을 사용
        gray_feats = (0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :])

        # 멀티 스케일 형태 정보 추출
        x = gray_feats
        comp_feats_list = []
        for i, comp_feat_block in enumerate(self.comp_feat_blocks):
            # 특징 추출
            x = comp_feat_block(x)

            # 멀티 스케일 형태 정보 저장
            comp_feats_list.append(x)

        # 저장된 인터폴레이션 결과 추가
        for feat in comp_feats_list:
            result_feats_list.append(F.interpolate(feat, size=(target_h, target_w), mode='nearest'))

        # 특징 정보들 torch concat
        essential_feats = torch.cat(result_feats_list, dim=1)

        return essential_feats


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = EssenceNet()

        dummy_input = torch.zeros(2, 3, 320, 320)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        hidden_dim = backbone_output_ch * 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(backbone_output_ch),
            nn.Dropout(0.3),

            nn.Linear(backbone_output_ch, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x

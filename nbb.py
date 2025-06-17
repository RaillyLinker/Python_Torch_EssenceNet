import torch
import torch.nn as nn
import torch.nn.functional as F


class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 선(직선 + 곡선), 선명도, 질감, 도형 검출 (채널 크기 증량, 감축 가능)
        # 멀티 스케일 + 복합 형태 정보
        self.comp_feat_blocks = nn.ModuleList([
            # 처음은 픽셀 단위 직선 검출 (확정 구조)
            # 연결성 패턴(가로+세로 10, 대각 위 10, 대각 아래 10, 전체 1)
            self._double_conv_block(1, 32, 1, 3, 2, 1),  # 320x320 -> 160x160
            # 두번째는 픽셀 단위 직선 3개의 조합(곡선)으로, 여전히 선형 탐지이며, 입력 채널이 1 이므로 32*1 를 해서 32 커널
            # 첫번째 레이어가 각도를 나타내기 위해 1 채널을 썼다면, 분절된 out_ch 개의 공간의 각도를 표현하는 개념. (구불구불한 선의 종류)
            self._double_conv_block(1, 32, 16, 3, 2, 1),  # 160x160 -> 80x80
            # 세번째 부터 도형과 면의 개념이 추출되기 시작하며 멀티 스케일 개념
            # 출력 채널의 개수는 전부 같고, conv 커널 개수도 입력 값에 따르기 때문에 같아짐
            self._double_conv_block(16, 128, 64, 3, 2, 1),  # 80x80 -> 40x40
            self._double_conv_block(64, 512, 64, 3, 2, 1),  # 40x40 -> 20x20
            self._double_conv_block(64, 512, 64, 3, 2, 1),  # 20x20 -> 10x10
            self._double_conv_block(64, 512, 64, 3, 2, 1),  # 10x10 -> 5x5
            self._double_conv_block(64, 512, 64, 3, 2, 1),  # 5x5 -> 3x3
            self._double_conv_block(64, 512, 64, 3, 1, 0),  # 3x3 -> 1x1
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

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        result_feats_list = []
        target_h, target_w = x.shape[2], x.shape[3]

        # 컬러 특징 저장(320x320)
        result_feats_list.append(x)

        # 분석을 위한 흑백 변환(320x320)
        # 컬러 이미지는 그 자체로 색이란 특징을 지닌 특징 맵이고, 형태 특징을 구하기 위한 입력 값은 흑백으로 충분
        # 1x1 conv 를 굳이 할 필요 없이 검증된 알고리즘을 사용
        gray_feats = (0.2989 * x[:, 0:1, :, :] +
                      0.5870 * x[:, 1:2, :, :] +
                      0.1140 * x[:, 2:3, :, :])

        # 이미지 전 범위 평균 밝기 계산 및 저장(320x320)
        result_feats_list.append(gray_feats.mean(dim=(2, 3), keepdim=True).expand(-1, -1, target_h, target_w))

        # 멀티 스케일 형태 정보 추출
        x = gray_feats
        for i, comp_feat_block in enumerate(self.comp_feat_blocks):
            # 특징 추출
            x = comp_feat_block(x)

            # 멀티 스케일 형태 정보 저장
            result_feats_list.append(F.interpolate(x, size=(target_h, target_w), mode='nearest'))

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

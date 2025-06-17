import torch
import torch.nn as nn
import torch.nn.functional as F


class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 특징맵을 저장할 comp_feat_blocks 의 레이어 인덱스 리스트
        # 초기 인덱스는 역치에 따라 저장하지 않을 수 있고, 일정 간격을 스킵하는 방식으로 저장할 수도 있음
        self.save_feat_idx_list = [1, 2, 3, 4, 5, 6, 7]

        # 추출할 특징 크기(클수록 표현력이 늘어나지만 self.save_feat_idx_list 의 개수와 곱하는 값 만큼 메모리 사용량이 늘어남)
        self.comp_feats_ch = 32

        # 선(직선 + 곡선), 선명도, 질감, 도형 검출 (채널 크기 증량, 감축 가능)
        # 멀티 스케일 + 복합 형태 정보
        self.comp_feat_blocks = nn.ModuleList([
            # 처음은 3x3 픽셀 단위 특징 검출
            self._double_conv_block(1, 512, self.comp_feats_ch, 3, 2, 1),  # 320x320 -> 160x160
            # 다음은 9x9 픽셀 단위 특징 검출(아직 질감 등은 추출 되지 않음)
            # 여기부턴 커널 개수를 늘려도 됩니다. 최대는 512 * self.comp_feats_ch
            self._double_conv_block(self.comp_feats_ch, 512, self.comp_feats_ch, 3, 2, 1),  # 160x160 -> 80x80
            # 세번째 부터 도형과 면의 개념이 제대로 추출 되기 시작 하며 멀티 스케일 개념
            # 출력 채널의 개수는 전부 같고, conv 커널 개수도 입력 값에 따르기 때문에 같아짐
            self._double_conv_block(self.comp_feats_ch, 512, self.comp_feats_ch, 3, 2, 1),  # 80x80 -> 40x40
            self._double_conv_block(self.comp_feats_ch, 512, self.comp_feats_ch, 3, 2, 1),  # 40x40 -> 20x20
            self._double_conv_block(self.comp_feats_ch, 512, self.comp_feats_ch, 3, 2, 1),  # 20x20 -> 10x10
            self._double_conv_block(self.comp_feats_ch, 512, self.comp_feats_ch, 3, 2, 1),  # 10x10 -> 5x5
            self._double_conv_block(self.comp_feats_ch, 512, self.comp_feats_ch, 3, 2, 1),  # 5x5 -> 3x3
            self._double_conv_block(self.comp_feats_ch, 512, self.comp_feats_ch, 3, 1, 0)  # 3x3 -> 1x1
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

        # 멀티 스케일 형태 정보 추출
        x = gray_feats
        comp_feats_list = []
        for i, comp_feat_block in enumerate(self.comp_feat_blocks):
            # 특징 추출
            x = comp_feat_block(x)

            # 멀티 스케일 형태 정보 저장
            if i in self.save_feat_idx_list:
                comp_feats_list.append(x)

        # 저장된 인터폴레이션 결과를 역순으로 추가
        for feat in reversed(comp_feats_list):
            result_feats_list.append(F.interpolate(feat, size=(target_h, target_w), mode='nearest'))

        # 특징 정보들 torch concat
        essential_feats = torch.cat(result_feats_list, dim=1)

        return essential_feats


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = EssenceNet()

        # 백본 모델 특징맵 앞의 N채널(컬러 3채널 + 글로벌 영역부터 32 단위의 멀티 스케일 레이어)만 분리
        self.backbone_output_ch = 3 + (32 * 7)

        hidden_dim = self.backbone_output_ch * 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(self.backbone_output_ch),
            nn.Dropout(0.3),

            nn.Linear(self.backbone_output_ch, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        x = x[:, :self.backbone_output_ch, :, :]

        x = self.classifier(x)
        return x

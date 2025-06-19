import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth


# todo residual 시 + 말고 concat -> 1x1 conv 로 해보기
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 멀티 스케일 + 복합 형태 정보
        self.comp_feat_blocks = nn.ModuleList([
            # 3x3 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지
            self._double_conv_block(1, 12, 8, 3, 2, 1),  # 320x320 -> 160x160
            # 7x7 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 직선 + 작은 곡선 = 픽셀 아이콘 영역
            self._double_conv_block(8, 24, 16, 3, 2, 1),  # 160x160 -> 80x80
            # 15x15 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 직선 + 곡선 + 패턴 = 픽셀 아이콘 영역
            self._double_conv_block(16, 36, 24, 3, 2, 1),  # 80x80 -> 40x40
            # 31x31 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 자유로운 선 + 패턴 + 제한된 도형 = 픽셀 아트 영역
            self._double_conv_block(24, 48, 32, 3, 2, 1),  # 40x40 -> 20x20
            # 63x63 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 자유로운 선 + 패턴 + 도형 = 픽셀 아트 영역
            self._double_conv_block(32, 64, 48, 3, 2, 1),  # 20x20 -> 10x10
            # 127x127 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 자유로운 선 + 패턴 + 자유로운 도형 + 질감 = 실사 이미지
            self._double_conv_block(48, 80, 64, 3, 2, 1),  # 10x10 -> 5x5
            # 255x255 픽셀 단위 특징 검출
            # 아래부터는 추상적 정보
            self._double_conv_block(64, 96, 80, 3, 2, 1),  # 5x5 -> 3x3
            # 320x320 픽셀 단위 특징 검출
            self._double_conv_block(80, 112, 96, 3, 1, 0)  # 3x3 -> 1x1
        ])

        # Residual 연결을 위한 1x1 conv 계층
        self.residual_convs = nn.ModuleList()
        for block in self.comp_feat_blocks:
            conv_layers = [l for l in block.children() if isinstance(l, nn.Conv2d)]
            in_ch = conv_layers[0].in_channels
            out_ch = conv_layers[-1].out_channels
            self.residual_convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))

        # Post-BN + Activation 블록
        self.post_bn_activations = nn.ModuleList()
        for block in self.comp_feat_blocks:
            conv_layers = [l for l in block.children() if isinstance(l, nn.Conv2d)]
            out_ch = conv_layers[-1].out_channels
            self.post_bn_activations.append(nn.Sequential(
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            ))

        num_blocks = len(self.comp_feat_blocks)
        drop_probs = [float(i) / num_blocks * 0.2 for i in range(num_blocks)]
        self.drop_paths = nn.ModuleList([
            StochasticDepth(p, mode='row') for p in drop_probs
        ])

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
        target_h, target_w = x.shape[2] // 2, x.shape[3] // 2

        # 컬러 특징 저장(160x160)
        result_feats_list.append(F.interpolate(x, scale_factor=0.5, mode='nearest'))

        # 분석을 위한 흑백 변환(320x320)
        # 컬러 이미지는 그 자체로 색이란 특징을 지닌 특징 맵이고, 형태 특징을 구하기 위한 입력 값은 흑백으로 충분
        # 1x1 conv 를 굳이 할 필요 없이 검증된 알고리즘을 사용
        gray_feats = (0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :])

        # 멀티 스케일 형태 정보 추출
        x_in = gray_feats
        comp_feats_list = []
        for i, (block, res_conv, drop_path, post_bn_act) in enumerate(
                zip(self.comp_feat_blocks, self.residual_convs, self.drop_paths, self.post_bn_activations)
        ):
            x_out = block(x_in)
            res = res_conv(x_in)
            if res.shape[2:] != x_out.shape[2:]:
                res = F.interpolate(res, size=x_out.shape[2:], mode='nearest')
            x_in = drop_path(x_out + res)
            x_in = post_bn_act(x_in)
            comp_feats_list.append(x_in)

        # 저장된 인터폴레이션 결과 추가
        # 픽셀 색상 데이터, 에지 데이터, 작은 범위 comp_feats 데이터, 조금 더 큰 범위 comp_feats 데이터...Global feats 데이터
        # 위와 같은 방식으로 뒤로 갈수록 더 큰 범주의 특징이 나옵니다.
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

        self.classifier = nn.Sequential(
            nn.Conv2d(backbone_output_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout(p=0.3),  # Dropout 추가
            nn.AdaptiveAvgPool2d(1),  # (B, 256, 1, 1)
            nn.Flatten(),  # (B, 256)
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x

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

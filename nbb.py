import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import torchvision.transforms.functional as TF


class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden_dim = channels // 4
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # (B, C)
        y = self.fc(y).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * y


class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 멀티 스케일 + 복합 형태 정보
        self.comp_feat_blocks = nn.ModuleList([
            # 3x3 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지
            self._double_conv_block(1, 32, 16, 3, 2, 1),  # 320x320 -> 160x160
            # 7x7 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 직선 + 작은 곡선 = 픽셀 아이콘 영역
            self._double_conv_block(16, 48, 24, 3, 2, 1),  # 160x160 -> 80x80
            # 15x15 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 직선 + 곡선 + 패턴 = 픽셀 아이콘 영역
            self._double_conv_block(24, 64, 32, 3, 2, 1),  # 80x80 -> 40x40
            # 31x31 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 자유로운 선 + 패턴 + 제한된 도형 = 픽셀 아트 영역
            self._double_conv_block(32, 80, 48, 3, 2, 1),  # 40x40 -> 20x20
            # 63x63 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 자유로운 선 + 패턴 + 도형 = 픽셀 아트 영역
            self._double_conv_block(48, 96, 64, 3, 2, 1),  # 20x20 -> 10x10
            # 127x127 픽셀 단위 특징 검출
            # 노이즈 + 점 + 에지 + 자유로운 선 + 패턴 + 자유로운 도형 + 질감 = 실사 이미지
            self._double_conv_block(64, 112, 80, 3, 2, 1),  # 10x10 -> 5x5
            # 255x255 픽셀 단위 특징 검출
            # 아래부터는 추상적 정보
            self._double_conv_block(80, 128, 96, 3, 2, 1),  # 5x5 -> 3x3
            # 320x320 픽셀 단위 특징 검출
            self._double_conv_block(96, 144, 112, 3, 1, 0)  # 3x3 -> 1x1
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

        self.se_blocks = nn.ModuleList()
        for block in self.comp_feat_blocks:
            conv_layers = [l for l in block.children() if isinstance(l, nn.Conv2d)]
            out_ch = conv_layers[-1].out_channels
            self.se_blocks.append(SEBlock(out_ch))

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

        # 특징 저장 리스트(320, 160, 80, 40, 20, 10, 5, 3, 1 해상도 피라미트)
        result_feats_list = [x]  # 컬러 특징 저장

        # 분석을 위한 흑백 변환(320x320)
        # 컬러 이미지는 그 자체로 색이란 특징을 지닌 특징 맵이고, 형태 특징을 구하기 위한 입력 값은 흑백으로 충분
        gray_feats = TF.rgb_to_grayscale(x, num_output_channels=1)

        # 첫 입력값은 흑백 이미지
        x_in = gray_feats
        for i, (
                block,
                res_conv,
                drop_path,
                post_bn_act,
                se_block
        ) in enumerate(
            zip(
                self.comp_feat_blocks,
                self.residual_convs,
                self.drop_paths,
                self.post_bn_activations,
                self.se_blocks
            )
        ):
            # 특징 추출
            x_out = block(x_in)

            # 채널 중 중요한 채널을 강조
            x_out = se_block(x_out)

            # 이전 결과값과 이번 결과값이 크게 나타나는 곳을 강조 및 역전파 지름길 만들기
            res = res_conv(x_in)
            if res.shape[2:] != x_out.shape[2:]:
                # 부드러운 다운스케일링(평균값 사용)
                res = F.interpolate(res, size=x_out.shape[2:], mode='area')
            x_in = drop_path(x_out + res)
            x_in = post_bn_act(x_in)

            # 레이어 반환 특징맵 저장
            result_feats_list.append(x_in)

        return result_feats_list


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = EssenceNet()

        # 더미 입력
        dummy_input = torch.zeros(2, 3, 320, 320)
        with torch.no_grad():
            feats = self.backbone(dummy_input)

        self.reducers = nn.ModuleList()
        for feat in feats:
            _, c, h, w = feat.shape
            if h == 1 and w == 1:
                self.reducers.append(nn.Identity())
            elif h == w:
                self.reducers.append(nn.AdaptiveAvgPool2d(1))
            else:
                raise ValueError(f"Unsupported feature map size: {h}x{w}")

        total_channels = sum([feat.shape[1] for feat in feats])

        self.classifier = nn.Sequential(
            nn.Linear(total_channels, total_channels // 2),
            nn.BatchNorm1d(total_channels // 2),
            nn.SiLU(),
            nn.Dropout(p=0.5),
            nn.Linear(total_channels // 2, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        reduced_feats = [reducer(feat) for feat, reducer in zip(feats, self.reducers)]
        x = torch.cat(reduced_feats, dim=1)  # [B, total_channels, 1, 1]
        x = x.flatten(1)  # [B, total_channels]
        return self.classifier(x)  # [B, num_classes]

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

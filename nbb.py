import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask


class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 멀티 스케일 + 복합 형태 정보
        self.comp_feat_blocks = nn.ModuleList([
            # 3x3 픽셀 단위 특징 검출
            # 정보가 혼재된 노이즈 신호
            self._double_conv_block_with_drop(1, 32, 16, 3, 2, 1, 7, 0.1),  # 320x320 -> 160x160
            # 7x7 픽셀 단위 특징 검출
            # 노이즈 + 점의 의미
            self._double_conv_block_with_drop(16, 64, 64, 3, 2, 1, 5, 0.09),  # 160x160 -> 80x80
            # 15x15 픽셀 단위 특징 검출
            # 점 + 작은 단위 선
            self._double_conv_block_with_drop(64, 96, 96, 3, 2, 1, 3, 0.08),  # 80x80 -> 40x40
            # 31x31 픽셀 단위 특징 검출
            # 점 + 자유로운 선 + 최소 단위 도형
            self._double_conv_block_with_drop(96, 128, 128, 3, 2, 1, 3, 0.07),  # 40x40 -> 20x20
            # 63x63 픽셀 단위 특징 검출
            # 점 + 자유로운 선 + 자유로운 도형
            self._double_conv_block(128, 160, 160, 3, 2, 1),  # 20x20 -> 10x10
            # 127x127 픽셀 단위 특징 검출
            # 점 + 자유로운 선 + 자유로운 도형 + 질감
            # 실사 이미지 표현이 가능한 영역이며, 아래부터는 추상적 정보
            self._double_conv_block(160, 224, 224, 3, 2, 1),  # 10x10 -> 5x5
            # 255x255 픽셀 단위 특징 검출
            self._double_conv_block(224, 256, 256, 3, 2, 1),  # 5x5 -> 3x3
            # 320x320 픽셀 단위 특징 검출
            self._double_conv_block(256, 512, 512, 3, 1, 0)  # 3x3 -> 1x1
        ])

        # Residual 연결을 위한 1x1 conv 계층
        self.residual_convs = nn.ModuleList()
        for block in self.comp_feat_blocks:
            first_conv = None
            second_conv = None
            for layer in block.children():
                if isinstance(layer, nn.Conv2d):
                    if first_conv is None:
                        first_conv = layer
                    else:
                        second_conv = layer
                        break
            in_ch = first_conv.in_channels
            out_ch = second_conv.out_channels
            self.residual_convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))

        self.post_bn_activations = nn.ModuleList()
        for block in self.comp_feat_blocks:
            second_conv = None
            for layer in block.children():
                if isinstance(layer, nn.Conv2d):
                    second_conv = layer  # 마지막 Conv2d
            out_ch = second_conv.out_channels
            self.post_bn_activations.append(nn.Sequential(
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            ))

        num_blocks = len(self.comp_feat_blocks)
        drop_probs = [float(i) / num_blocks * 0.2 for i in range(num_blocks)]
        self.drop_paths = nn.ModuleList([
            StochasticDepth(p) for p in drop_probs
        ])

        self._init_weights()

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

    def _double_conv_block_with_drop(self, in_ch, mid_ch, out_ch, ks, strd, pdd, dbs, dbp):
        return nn.Sequential(
            # 평면당 형태를 파악
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),
            DropBlock2D(block_size=dbs, drop_prob=dbp),

            # 채널간 패턴 분석
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x_in = gray_feats
        comp_feats_list = []
        for i, (block, res_conv, drop_path, post_bn_act) in enumerate(
                zip(self.comp_feat_blocks, self.residual_convs, self.drop_paths, self.post_bn_activations)):
            x_out = block(x_in)
            x_out = drop_path(x_out)
            res = res_conv(x_in)
            if res.shape[2:] != x_out.shape[2:]:
                res = F.interpolate(res, size=x_out.shape[2:], mode='nearest')
            x_in = x_out + res
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

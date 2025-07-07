import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


# todo : 1x1 conv 깊이 변경
# (2D 특징 추출 블록)
# 말 그대로 2차원 특징 추출만 수행
def _single_conv_block(in_ch, mid_ch, out_ch, ks, strd, pdd, dp, bs):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(),

        DropBlock2D(drop_prob=dp, block_size=bs),

        # 픽셀별 의미 추출
        nn.Conv2d(mid_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(),

        nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(),
        nn.Dropout2d(0.2)
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # todo conv 채널 변경
        self.feats_convs = nn.ModuleList([
            _single_conv_block(3, 32, 16, 3, 2, 1, 0.05, 3),  # 320x320 -> 160x160
            _single_conv_block(16, 64, 32, 3, 2, 1, 0.05, 3),  # 160x160 -> 80x80
            _single_conv_block(32, 128, 64, 3, 2, 1, 0.05, 3),  # 80x80 -> 40x40
            _single_conv_block(64, 256, 128, 3, 2, 1, 0.1, 3),  # 40x40 -> 20x20
            _single_conv_block(128, 512, 256, 3, 2, 1, 0.1, 3),  # 20x20 -> 10x10
            _single_conv_block(256, 1024, 512, 3, 2, 1, 0.1, 2),  # 10x10 -> 5x5
            _single_conv_block(512, 2048, 1024, 3, 2, 1, 0.1, 2),  # 5x5 -> 3x3
            _single_conv_block(1024, 4096, 2048, 3, 1, 0, 0.0, 1),  # 3x3 -> 1x1
        ])

        # 자동 채널 추출 기반 residual projection 생성
        self.res_projs = nn.ModuleList()

        for conv_block in self.feats_convs:
            layers = [m for m in conv_block.modules() if isinstance(m, nn.Conv2d)]
            self.res_projs.append(nn.Conv2d(layers[0].in_channels, layers[-1].out_channels, kernel_size=1, bias=False))

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        feats_list = []
        feat = x
        for conv, res_proj in zip(self.feats_convs, self.res_projs):
            res = feat
            feat = conv(feat)
            res_proj = F.interpolate(res_proj(res), size=feat.shape[2:], mode='area')
            feat = feat + res_proj
            feats_list.append(feat)

        return feats_list


# ----------------------------------------------------------------------------------------------------------------------
# (EssenceNet 이미지 분류기)
class EssenceNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 위치 특징 추출 백본 모델
        self.backbone = EssenceNet()

        # 백본이 출력하는 특징맵 채널 수
        feat_channels = [16, 32, 64, 128, 256, 512, 1024, 2048]
        encoder_input = sum(c for c in feat_channels)

        # 시각 정보 벡터의 사이즈
        vision_context_size = 2048

        # todo : 채널, 깊이 변경
        # 시각 컨텍스트 인코더
        hidden = encoder_input // 2
        self.vision_context_encoder = nn.Sequential(
            nn.Conv2d(encoder_input, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(hidden, vision_context_size, kernel_size=1)
        )

        # todo : 채널, 깊이 변경
        # 분류기
        classifier_head_hidden = vision_context_size // 2
        self.classifier_head = nn.Sequential(
            nn.Conv2d(vision_context_size, classifier_head_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(classifier_head_hidden),
            nn.SiLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(classifier_head_hidden, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 백본 특징맵 피라미드 추출
        feats = self.backbone(x)

        all_feats = torch.cat([F.interpolate(f, size=feats[0].shape[2:], mode='nearest') for f in feats], dim=1)

        vision_context = self.vision_context_encoder(all_feats)
        logits_map = self.classifier_head(vision_context)

        logits_sum = logits_map.sum(dim=(2, 3))

        return logits_sum

# 추론시 동작 :
# 위 백본에서 추출된 피쳐맵 피라미드에서 1x1 특징맵부터 128x128 특징맵을 f1, f2, f3, f4, f5, f6, f7, f8 이라고 하겠습니다.
# 이를 가지고 EssenceNetClassifier 를 구현합니다.
#
# 1. f1 을 f2 크기로 업샘플링 후 concat 해서, 이를 1x1 conv 로, num_classes 크기로 만듭니다.
# 그러면 2x2 크기의 num_classes 들이 준비됩니다.
# 각각의 num_classes 들에 softmax 를 취해서 2x2 크기의 num_classes 크기의 확률 벡터가 준비되고,
# 각 확률 벡터를 확인하여 최대 값이 0.8을 넘어서는 것이 있다면 그것을 그대로 return 으로 반환합니다.
#
# 2. 만약 위에서 return 되지 못했다면, 다음에는 1번에서 concat 한 2x2 특징맵을 f3 크기인 4x4 로 업샘플링하고, 이를 f3 과 concat 해서,
# 이를 1x1 conv 로, num_classes 크기로 만듭니다.
# 그러면 4x4 크기의 num_classes 들이 준비됩니다.
# 각각의 num_classes 들에 softmax 를 취해서 4x4 크기의 num_classes 크기의 확률 벡터가 준비되고,
# 각 확률 벡터를 확인하여 최대 값이 0.8을 넘어서는 것이 있다면 그것을 그대로 return 으로 반환합니다.
#
# 3. 위와 같은 방식을 반복하다가 만약 f8 까지 진행했을 때에도 결과가 나오지 않았다면,
# 결국 가장 max 값이 큰 벡터를 반환해줍니다.

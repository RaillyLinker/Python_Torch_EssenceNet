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
        nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 흑백 변환 가중치 저장
        self.register_buffer("rgb2gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

        # todo conv 채널 변경
        # 구역별 멀티 스케일 분석
        # 2D 데이터의 형태적 특징을 멀티 스케일로 추출 하기 위해 1채널 오리진 정보를 N번 입력
        # 전역적 정보를 지역적 정보로 투영하기 위해 커널이 큰 순서대로 배치
        # 작은 선만으로는 아무 정보가 없음. 큰 형태에서 작은 디테일로 정보가 보정 되는 것이 자연스러움
        self.feats_convs = nn.ModuleList([
            _single_conv_block(1, 2048, 1024, 256, 256, 0, 0.0, 1),  # 256x256 -> 1x1
            _single_conv_block(1, 1024, 512, 128, 128, 0, 0.1, 2),  # 256x256 -> 2x2
            _single_conv_block(1, 512, 256, 64, 64, 0, 0.1, 2),  # 256x256 -> 4x4
            _single_conv_block(1, 256, 128, 32, 32, 0, 0.1, 3),  # 256x256 -> 8x8
            _single_conv_block(1, 128, 64, 16, 16, 0, 0.1, 3),  # 256x256 -> 16x16
            _single_conv_block(1, 64, 32, 8, 8, 0, 0.05, 3),  # 256x256 -> 32x32
            _single_conv_block(1, 32, 16, 4, 4, 0, 0.05, 3)  # 256x256 -> 64x64
        ])

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        # 컬러 이미지
        # (B, 3, 256, 256)
        color_feats = x

        # 순수 하게 CNN 형태 분석을 위한 흑백 변환
        # (B, 1, 256, 256)
        gray_feats = (color_feats * self.rgb2gray.to(x.device, x.dtype)).sum(dim=1, keepdim=True)

        # conv 특징 추출
        k_feats_list = [conv(gray_feats) for conv in self.feats_convs]

        # color_feats 를 k_feats 크기에 맞게 리사이즈 (area = 다운 샘플링 평균값)
        k_sizes = [(f.shape[2], f.shape[3]) for f in k_feats_list]
        resized_colors = [F.interpolate(color_feats, size=size, mode='area') for size in k_sizes]

        # conv 특징들과 컬러 특징 결합
        feats_list = [torch.cat([rc, kf], dim=1) for rc, kf in zip(resized_colors, k_feats_list)]

        # 최종 특징(1, 2, 4, 8, 16, 32, 64, 128 사이즈 특징 피라미드)
        return feats_list


# ----------------------------------------------------------------------------------------------------------------------
# (EssenceNet 이미지 분류기)
class EssenceNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 위치 특징 추출 백본 모델
        self.backbone = EssenceNet()

        # 백본 추출 특징맵 피라미드 채널 개수
        feat_channels = [1027, 515, 259, 131, 67, 35, 19]

        # 식별 가능 특징맵 인덱스 선정
        self.total_levels = len(feat_channels)
        self.identify_idx = self.total_levels - 1

        # todo : 벡터 사이즈 변경
        # 시각 정보 벡터의 사이즈
        vision_context_size = 2048

        # todo : 레이어 깊이 변경
        # 이미지 인코더
        encoder_input = sum(feat_channels[:self.identify_idx + 1])
        hidden = encoder_input // 2
        self.vision_context_encoder = nn.Sequential(
            nn.Conv2d(encoder_input, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(hidden, vision_context_size, kernel_size=1)
        )

        # todo : 레이어 깊이 변경
        # 이미지 분류기 헤더
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

        # 각 레벨의 해상도 크기 추출
        feat_sizes = [f.shape[2:] for f in feats]

        # 이전까지의 특징맵을 현재 해상도(i)에 맞춰 업샘플링
        upsampled_feats = [
            [F.interpolate(f, size=feat_sizes[j], mode='nearest') if i != j else f
             for j in range(len(feats))]
            for i, f in enumerate(feats)
        ]

        # 특징맵 피라미드 순회 및 logit 벡터 수집
        logits_list = []
        for i in range(self.total_levels):
            feat_i = feats[i]
            prev_feats = [upsampled_feats[k][i] for k in range(i)]

            # 마지막 레벨만 encoder+head 적용
            if i == self.identify_idx:
                # 특징 concat
                prev_feat = torch.cat(prev_feats + [feat_i], dim=1) if i > 0 else feat_i
                # 비전 인코딩
                vision_vec = self.vision_context_encoder(prev_feat)
                # 클래스 분류
                logits_i = self.classifier_head(vision_vec)
                # logits 벡터 리스트 저장
                logits_list.append(logits_i)

        # classifier_head 로 추출한 모든 logits 를 전부 누적
        b = x.shape[0]
        flat_logits_all = [logits.view(b, logits.shape[1], -1) for logits in logits_list]
        all_logits_flat = torch.cat(flat_logits_all, dim=2)

        final_outputs = all_logits_flat.sum(dim=2)
        return final_outputs

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

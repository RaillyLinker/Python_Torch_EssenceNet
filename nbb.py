import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


# (2D 특징 추출 블록)
# 말 그대로 2차원 특징 추출만 수행
def _single_conv_block(in_ch, out_ch, ks, strd, pdd, dp, bs):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(),
        DropBlock2D(drop_prob=dp, block_size=bs)
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 흑백 변환 가중치 저장
        self.register_buffer("rgb2gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

        # todo 채널 변경
        # 구역별 멀티 스케일 분석
        # 2D 데이터의 형태적 특징을 멀티 스케일로 추출 하기 위해 1채널 오리진 정보를 N번 입력
        # 전역적 정보를 지역적 정보로 투영하기 위해 커널이 큰 순서대로 배치(ex : 이 점은 책장 안의 책 안의 글자 안의 곡선 안에 속한 점이다.(즉, 지역은 전역에 속함))
        self.feats_convs = nn.ModuleList([
            _single_conv_block(1, 1024, 256, 256, 0, 0.0, 1),  # 256x256 -> 1x1
            _single_conv_block(1, 512, 128, 128, 0, 0.3, 2),  # 256x256 -> 2x2
            _single_conv_block(1, 256, 64, 64, 0, 0.25, 3),  # 256x256 -> 4x4
            _single_conv_block(1, 128, 32, 32, 0, 0.2, 5),  # 256x256 -> 8x8
            _single_conv_block(1, 64, 16, 16, 0, 0.15, 5),  # 256x256 -> 16x16
            _single_conv_block(1, 32, 8, 8, 0, 0.1, 5),  # 256x256 -> 32x32
            _single_conv_block(1, 16, 4, 4, 0, 0.05, 5),  # 256x256 -> 64x64
            _single_conv_block(1, 8, 3, 2, 1, 0.05, 5),  # 256x256 -> 128x128
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

        # color_feats 를 k_feats 크기에 맞게 리사이즈
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
        # 특징맵 추출 백본 모델
        self.backbone = EssenceNet()

        # 백본 feature 피라미드 채널 계산
        feat_out_ch = [conv[0].out_channels for conv in self.backbone.feats_convs]
        # ex : f_ch = [1024+3, 512+3, 256+3, 128+3, 64+3, 32+3, 16+3, 8+3] = [1027, 515, 259, 131, 67, 35, 19, 11]
        f_ch = [c + 3 for c in feat_out_ch]

        self.total_feat_ch = sum(f_ch)  # concat 후 총 채널 수

        hidden = int(self.total_feat_ch / 2)

        # todo : 히든 변경 및 깊이 변경
        # MLP: (concat된 채널 수) → num_classes
        self.mlp = nn.Sequential(
            nn.Linear(self.total_feat_ch, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # 백본 피쳐맵 피라미드 추출
        feats_list = self.backbone(x)

        # 가장 큰 해상도: 128x128 (마지막 출력의 spatial size)
        target_size = feats_list[-1].shape[2:]

        # 모든 피처를 같은 해상도로 업샘플링 후 채널 방향으로 concat: (B, C_total, 128, 128)
        upsampled_feats = torch.cat(
            [F.interpolate(f, size=target_size, mode='nearest') for f in feats_list], dim=1
        )

        # (B, C, H, W) → (B, H*W, C)
        B, C, H, W = upsampled_feats.shape
        fused_feats = upsampled_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # MLP 적용 → (B, H*W, num_classes)
        logits = self.mlp(fused_feats)

        # 1) 픽셀별 가장 높은 클래스 인덱스
        pred_classes = logits.argmax(dim=-1)  # (B, H*W)

        final_logits = []

        for b in range(B):
            preds = pred_classes[b]  # (H*W,)
            logit_vectors = logits[b]  # (H*W, num_classes)

            # 2) 모드 클래스(가장 많이 나온 클래스)
            values, counts = preds.unique(return_counts=True)
            mode_class = values[counts.argmax()]

            # 3) 모드 클래스에 속하는 픽셀 인덱스
            mode_mask = (preds == mode_class)

            # 4) 해당 픽셀들의 로짓 벡터들
            mode_logits = logit_vectors[mode_mask]  # (N_mode, num_classes)

            # 5) 각 픽셀 로짓 벡터의 '크기' 계산 (예: L2 norm or sum of logits)
            # 여기서는 L2 norm 사용
            logits_norm = mode_logits.norm(dim=1)  # (N_mode,)

            # 6) 중간값(median) 인덱스 찾기
            median_value = logits_norm.median()
            median_idx = (logits_norm - median_value).abs().argmin()

            # 7) 중간값에 가장 가까운 로짓 벡터 선택
            selected_logit = mode_logits[median_idx]  # (num_classes,)

            final_logits.append(selected_logit)

        # (B, num_classes) 텐서로 변환
        final_logits = torch.stack(final_logits)

        return final_logits

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

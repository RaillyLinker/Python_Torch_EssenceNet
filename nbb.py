import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


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

        # todo 채널 변경
        # 구역별 멀티 스케일 분석
        # 2D 데이터의 형태적 특징을 멀티 스케일로 추출 하기 위해 1채널 오리진 정보를 N번 입력
        # 전역적 정보를 지역적 정보로 투영하기 위해 커널이 큰 순서대로 배치
        # 작은 선만으로는 아무 정보가 없음. 큰 형태에서 작은 디테일로 정보가 보정 되는 것이 자연스러움
        self.feats_convs = nn.ModuleList([
            _single_conv_block(1, 2048, 1024, 256, 256, 0, 0.0, 1),  # 256x256 -> 1x1
            _single_conv_block(1, 1024, 512, 128, 128, 0, 0.3, 2),  # 256x256 -> 2x2
            _single_conv_block(1, 512, 256, 64, 64, 0, 0.25, 3),  # 256x256 -> 4x4
            _single_conv_block(1, 256, 128, 32, 32, 0, 0.2, 5),  # 256x256 -> 8x8
            _single_conv_block(1, 128, 64, 16, 16, 0, 0.15, 5),  # 256x256 -> 16x16
            _single_conv_block(1, 64, 32, 8, 8, 0, 0.1, 5),  # 256x256 -> 32x32
            _single_conv_block(1, 32, 16, 4, 4, 0, 0.05, 5),  # 256x256 -> 64x64
            _single_conv_block(1, 16, 8, 3, 2, 1, 0.05, 5),  # 256x256 -> 128x128
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
# todo : vision_context 를 사용하는 방식 수정해보기 : 지금처럼 concat 하지 말고 residual 처럼 더하고 norm 하는 방식
class EssenceNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 위치 특징 추출 백본 모델
        self.backbone = EssenceNet()

        # 백본 추출 특징맵 피라미드 채널 개수
        feat_channels = [1027, 515, 259, 131, 67, 35, 19, 11]

        # todo : 벡터 사이즈 변경
        # 시각 정보 벡터의 사이즈
        vision_context_size = 1024

        # todo : 레이어 깊이 변경
        # 시각 정보 인코더 리스트
        self.vision_context_encoders = nn.ModuleList()
        for idx, ch in enumerate(feat_channels):
            if idx == 0:
                input_ch = ch
            else:
                input_ch = ch + vision_context_size
            hidden = input_ch // 2
            self.vision_context_encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_ch, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.SiLU(),
                    nn.Dropout2d(0.2),
                    nn.Conv2d(hidden, vision_context_size, kernel_size=1)
                )
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
        # 이미지 특징맵 피라미드 추출
        feats = self.backbone(x)

        # 첫번째 특징맵 분석
        vision_vector = self.vision_context_encoders[0](feats[0])
        logits = self.classifier_head(vision_vector)

        for i in range(1, len(feats)):
            feat_i = feats[i]
            B, C, H, W = feat_i.shape

            # vision_vector 크기 맞추기
            if vision_vector.shape[2:] != (H, W):
                vision_vector = F.interpolate(vision_vector, size=(H, W), mode='nearest')

            # 현재 특징맵과 이전 vision vector concat
            concat_input = torch.cat([feat_i, vision_vector], dim=1)

            # vision context encoding
            vision_vector = self.vision_context_encoders[i](concat_input)

            # 이번 로짓 벡터 추출
            logits_i = self.classifier_head(vision_vector)

            # 이전 logits 업샘플링
            prev_logits = F.interpolate(logits, size=(H, W), mode='nearest')

            # 정규화: (mean, std) 기준
            mean1, std1 = prev_logits.mean(dim=(1, 2, 3), keepdim=True), prev_logits.std(dim=(1, 2, 3), keepdim=True)
            mean2, std2 = logits_i.mean(dim=(1, 2, 3), keepdim=True), logits_i.std(dim=(1, 2, 3), keepdim=True)

            prev_logits_norm = (prev_logits - mean1) / (std1 + 1e-5)
            logits_i_norm = (logits_i - mean2) / (std2 + 1e-5)

            # 정규화된 두 로짓을 합산
            logits = prev_logits_norm + logits_i_norm

        # 최종 logits: (B, num_classes, 128, 128)
        B, C, H, W = logits.shape
        flat_logits = logits.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, num_classes)

        final_outputs = []
        for b in range(B):
            logits_b = flat_logits[b]  # (H*W, num_classes)
            max_val, _ = logits_b.max(dim=1)
            max_pos = max_val.argmax()
            class_id = logits_b[max_pos].argmax()

            class_mask = logits_b.argmax(dim=1) == class_id
            selected_logits = logits_b[class_mask]  # (M, num_classes)

            class_scores = selected_logits[:, class_id]
            median_score = class_scores.median()
            diff = (class_scores - median_score).abs()
            idx = diff.argmin()

            final_outputs.append(selected_logits[idx])  # (num_classes,)

        return torch.stack(final_outputs, dim=0)  # (B, num_classes)

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

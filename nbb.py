import torch
import torch.nn as nn
import torch.nn.functional as F


# todo : 1x1 conv 로 축소 처리
def _single_conv_block(in_ch, out_ch, ks, strd, pdd):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("rgb2gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

        # 동일 커널(같은 형태)로 다른 해상도의 멀티 스케일 분석
        # todo : out_ch 크기 변경
        self.feats_conv = _single_conv_block(1, 64, 3, 3, 0)

        # todo : 보다 큰 커널로 멀티 해상도 탐지

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        # 입력값 사이즈
        _, _, h, w = x.shape
        # 입력 값 3 배수 사이즈로 제한 (ex : 3, 9, 27, 81, 243, 729, ...)
        assert h == w and h % 3 == 0, "Input size must be square and divisible by 3"

        # 컬러 이미지
        # (B, 3, h, w)
        color_feats = x

        # 순수 하게 CNN 형태 분석을 위한 흑백 변환
        # (B, 1, h, w)
        gray_feats = (color_feats * self.rgb2gray.to(x.device, x.dtype)).sum(dim=1, keepdim=True)

        # 특징 저장 리스트(사이즈 별 특징 피라미트, 지역 -> 전역)
        result_feats_list = []

        # 3x3 커널 다중 해상도 멀티 스케일 특징 추출
        current_size = h
        # cur_size 가 3의 배수이고, 3 이상이면 반복
        while current_size % 3 == 0 and current_size >= 3:
            # 3x3 커널 특징 추출(입력값 사이즈의 1/3 출력)
            resized_gray = F.interpolate(gray_feats, size=(current_size, current_size), mode='area')
            k3_feats = self.feats_conv(resized_gray)

            # 컬러 특징 리사이징(gray_feats 의 사이즈와 동일)
            _, _, gh, gw = k3_feats.shape
            resized_color = F.interpolate(color_feats, size=(gh, gw), mode='area')

            # resized_color 와 gray_feats 를 합친 특징을 저장
            merged_feats = torch.cat([resized_color, k3_feats], dim=1)
            result_feats_list.append(merged_feats)

            # cur_size 갱신
            current_size = current_size // 3

        # 가장 큰 해상도 (첫 피쳐맵 기준)
        target_h, target_w = result_feats_list[0].shape[2:]

        # 모든 피쳐맵을 가장 큰 해상도로 업샘플링 후 concat
        pyramid_concat = torch.cat([
            F.interpolate(feat, size=(target_h, target_w), mode='nearest')
            for feat in result_feats_list
        ], dim=1)

        return pyramid_concat


# ----------------------------------------------------------------------------------------------------------------------
class EssenceNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EssenceNet()

        # 임시 입력으로 채널 수와 해상도 계산
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 243, 243)
            concat_feats = self.backbone(dummy_input)  # 이제는 하나의 tensor
            self.feat_dim = concat_feats.shape[1]  # (B, C_total, H, W)

        # todo : 히든 벡터 사이즈 변경
        # 분류기 히든 벡터 사이즈
        classifier_hidden_vector_size = num_classes * 2

        # 픽셀별 벡터 → 클래스 logits (MLP처럼 1x1 Conv)
        self.head = nn.Sequential(
            nn.Conv2d(self.feat_dim, classifier_hidden_vector_size, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout 확률 조절 가능
            nn.Conv2d(classifier_hidden_vector_size, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # (B, C_total, H, W)
        concat_feats = self.backbone(x)

        # 1x1 로 다운샘플링 (area : 평균)
        down_feats = F.interpolate(concat_feats, size=(1, 1), mode='area')

        # 1x1 conv 로 픽셀별 logits
        logits = self.head(down_feats)  # (B, num_classes, H/2, W/2)

        return logits.view(logits.size(0), -1)  # (B, num_classes)

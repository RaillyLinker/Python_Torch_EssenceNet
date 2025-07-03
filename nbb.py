import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


# (2D 특징 추출 블록)
# 말 그대로 2차원 특징 추출만 수행
def _single_conv_block(in_ch, out_ch, ks, strd, gn, pdd, dp, bs):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.GroupNorm(num_groups=gn, num_channels=out_ch),
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
            _single_conv_block(1, 1024, 256, 256, 32, 0, 0.0, 1),  # 256x256 -> 1x1
            _single_conv_block(1, 512, 128, 128, 32, 0, 0.3, 2),  # 256x256 -> 2x2
            _single_conv_block(1, 256, 64, 64, 16, 0, 0.25, 3),  # 256x256 -> 4x4
            _single_conv_block(1, 128, 32, 32, 8, 0, 0.2, 5),  # 256x256 -> 8x8
            _single_conv_block(1, 64, 16, 16, 4, 0, 0.15, 5),  # 256x256 -> 16x16
            _single_conv_block(1, 32, 8, 8, 4, 0, 0.1, 5),  # 256x256 -> 32x32
            _single_conv_block(1, 16, 4, 4, 4, 0, 0.05, 5),  # 256x256 -> 64x64
            _single_conv_block(1, 8, 3, 2, 4, 1, 0.05, 5),  # 256x256 -> 128x128
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

        # 결과 분류 임계값(이 값이 넘어가면 정답으로 확정)
        self.threshold = 0.8

        # 백본 feature 피라미드 채널 계산
        feat_out_ch = [conv[0].out_channels for conv in self.backbone.feats_convs]
        f_ch = [c + 3 for c in feat_out_ch]

        # todo : xor 해결 가능하게 깊게 구성해보기
        # 스테이지별 분류 헤드
        self.heads = nn.ModuleList([
            nn.Conv2d(f_ch[i] + f_ch[i + 1], num_classes, kernel_size=1)
            for i in range(len(f_ch) - 1)
        ])

        # concat 후 프로젝션(conv) to next stage 채널 수
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(f_ch[i] + f_ch[i + 1], f_ch[i + 1], kernel_size=1)
            for i in range(len(f_ch) - 1)
        ])

    # todo : 컨샙은 결정됨. 드롭아웃 방식과 근본 방식 고민해보기.
    #   다음 판단시 압축해서 concat 하기 방식 사용해보기
    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 3, 256, 256)
        feats = self.backbone(x)
        prev_feat = feats[0]

        for i, (head, proj) in enumerate(zip(self.heads, self.proj_convs), start=1):
            curr = feats[i]
            up = F.interpolate(prev_feat, size=curr.shape[2:], mode='nearest')
            concat = torch.cat([up, curr], dim=1)

            logits = head(concat)  # (1, C, H, W)
            probs = F.softmax(logits, dim=1)
            max_vals, _ = probs.max(dim=1)  # (1, H, W)

            mask = max_vals > self.threshold
            if mask.view(-1).any():
                flat_vals = max_vals.view(-1).clone()
                flat_vals[flat_vals <= self.threshold] = -1
                idx = flat_vals.argmax().item()
                H, W = max_vals.shape[1], max_vals.shape[2]
                h, w = divmod(idx, W)
                return probs[0, :, h, w]  # (num_classes,)

            prev_feat = proj(concat)

        # 마지막 스테이지 결과
        flat = max_vals.view(-1)
        idx = flat.argmax().item()
        H, W = max_vals.shape[1], max_vals.shape[2]
        h, w = divmod(idx, W)
        return probs[0, :, h, w]  # (num_classes,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # 배치 크기에 상관없이 동작
        outputs = []
        for i in range(B):
            single_out = self._forward_single(x[i:i + 1])
            outputs.append(single_out)
        return torch.stack(outputs, dim=0)  # (B, num_classes)

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

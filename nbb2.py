import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


def _double_conv_block(in_ch, mid_ch, out_ch, ks, strd, pdd, dp, bs):
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(),

        # 픽셀별 의미 추출(희소한 특징 압축)
        nn.Conv2d(mid_ch, (mid_ch + out_ch) // 2, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d((mid_ch + out_ch) // 2),
        nn.SiLU(),

        nn.Conv2d((mid_ch + out_ch) // 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(),

        DropBlock2D(drop_prob=dp, block_size=bs)
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # todo conv 채널 변경, 해상도를 2 배수(320)로 해보기(단계가 많아지므로 residual 을 없애고, 특징맵 해상도 반토막으로 시작하기)
        self.feats_convs = nn.ModuleList([
            _double_conv_block(3, 32, 16, 3, 1, 1, 0.0, 1),  # 243x243 -> 243x243
            _double_conv_block(16, 64, 32, 3, 3, 0, 0.0, 1),  # 243x243 -> 81x81
            _double_conv_block(32, 128, 64, 3, 3, 0, 0.15, 5),  # 81x81 -> 27x27
            _double_conv_block(64, 256, 128, 3, 3, 0, 0.20, 5),  # 27x27 -> 9x9
            _double_conv_block(128, 512, 256, 3, 3, 0, 0.20, 3),  # 9x9 -> 3x3
            _double_conv_block(256, 1024, 512, 3, 1, 0, 0.0, 1)  # 3x3 -> 1x1
        ])

        # feats_convs 를 기반으로 projection 레이어 자동 생성
        self.projections = nn.ModuleList()
        prev_out_ch = 3  # 입력 이미지 채널 (RGB)
        for conv_block in self.feats_convs:
            conv_layers = [layer for layer in conv_block.modules() if isinstance(layer, nn.Conv2d)]
            last_conv = conv_layers[-1]
            out_ch = last_conv.out_channels

            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(prev_out_ch, (prev_out_ch + out_ch) // 2, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d((prev_out_ch + out_ch) // 2),
                    nn.SiLU(),

                    nn.Conv2d((prev_out_ch + out_ch) // 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(),

                    nn.Dropout2d(0.2)
                )
            )

            prev_out_ch = out_ch

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."
        feats_list = []

        feat = x
        for idx, conv in enumerate(self.feats_convs):
            # Residual 용 특징 저장
            identity = feat

            # Conv 연산
            feat = conv(feat)

            # Residual 해상도 맞추기
            if not isinstance(self.projections[idx], nn.Identity):
                identity = F.interpolate(identity, size=feat.shape[2:], mode='area')

            # Residual 채널 맞추기
            projected = self.projections[idx](identity)

            # Residual 합치기
            feat = feat + projected
            feats_list.append(feat)

        return feats_list


# ----------------------------------------------------------------------------------------------------------------------
# (Segment 모델)
class EssenceNetSegmenter(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # conv 백본 모델
        self.backbone = EssenceNet()

        # 백본 특징맵 피라미드 채널 수
        self.feat_channels = [16, 32, 64, 128, 256, 512]
        self.encoder_input = sum(self.feat_channels)

        # 분류기 헤드
        self.classifier_head = nn.Sequential(
            nn.Conv2d(self.encoder_input, (self.encoder_input + num_classes) // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d((self.encoder_input + num_classes) // 2),
            nn.SiLU(),

            nn.Dropout2d(0.2),

            nn.Conv2d((self.encoder_input + num_classes) // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 백본 특징맵 피라미드 추출
        feats = self.backbone(x)

        # 특징맵 피라미드들을 최고 해상도 기준으로 합치기
        concat_feats = torch.cat([F.interpolate(f, size=feats[0].shape[2:], mode='nearest') for f in feats], dim=1)

        # 픽셀별 분류 헤드 적용
        logits = self.classifier_head(concat_feats)

        return logits

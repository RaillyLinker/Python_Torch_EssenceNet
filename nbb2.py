import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


def _double_conv_block(in_ch, mid_ch, out_ch, ks, strd, pdd, dp, bs):
    hidden_dim = (mid_ch + out_ch) // 2
    return nn.Sequential(
        # 평면당 형태를 파악
        nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(),

        # 픽셀별 의미 추출(희소한 특징 압축)
        nn.Conv2d(mid_ch, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.SiLU(),

        nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(),

        DropBlock2D(drop_prob=dp, block_size=bs)
    )


# (EssenceNet 백본)
class EssenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # todo conv 채널 변경
        self.feats_convs = nn.ModuleList([
            _double_conv_block(3, 64, 32, 3, 2, 1, 0.0, 1),  # 320x320 -> 160x160
            _double_conv_block(32, 128, 32, 3, 2, 1, 0.05, 3),  # 160x160 -> 80x80
            _double_conv_block(32, 128, 64, 3, 2, 1, 0.10, 3),  # 80x80 -> 40x40
            _double_conv_block(64, 256, 64, 3, 2, 1, 0.15, 5),  # 40x40 -> 20x20
            _double_conv_block(64, 256, 128, 3, 2, 1, 0.20, 5),  # 20x20 -> 10x10
            _double_conv_block(128, 512, 128, 3, 2, 1, 0.20, 3),  # 10x10 -> 5x5
            _double_conv_block(128, 512, 256, 3, 2, 1, 0.15, 3),  # 5x5 -> 3x3
            _double_conv_block(256, 1024, 256, 3, 1, 0, 0.0, 1)  # 3x3 -> 1x1
        ])

        # # feats_convs 를 기반으로 projection 레이어 자동 생성
        # self.projections = nn.ModuleList()
        # prev_out_ch = 3  # 입력 이미지 채널 (RGB)
        # for conv_block in self.feats_convs:
        #     conv_layers = [layer for layer in conv_block.modules() if isinstance(layer, nn.Conv2d)]
        #     last_conv = conv_layers[-1]
        #     out_ch = last_conv.out_channels
        #
        #     self.projections.append(
        #         nn.Sequential(
        #             nn.Conv2d(prev_out_ch, (prev_out_ch + out_ch) // 2, kernel_size=1, stride=1, bias=False),
        #             nn.BatchNorm2d((prev_out_ch + out_ch) // 2),
        #             nn.SiLU(),
        #
        #             nn.Conv2d((prev_out_ch + out_ch) // 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        #             nn.BatchNorm2d(out_ch),
        #             nn.SiLU(),
        #
        #             nn.Dropout2d(0.2)
        #         )
        #     )
        #
        #     prev_out_ch = out_ch

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."
        feats_list = []

        feat = x
        for idx, conv in enumerate(self.feats_convs):
            # # Residual 용 특징 저장
            # identity = feat

            # Conv 연산
            feat = conv(feat)

            # # Residual 해상도 맞추기
            # if not isinstance(self.projections[idx], nn.Identity):
            #     identity = F.interpolate(identity, size=feat.shape[2:], mode='area')
            #
            # # Residual 채널 맞추기
            # projected = self.projections[idx](identity)
            #
            # # Residual 합치기
            # feat = feat + projected

            # 특징 저장
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
        self.feat_channels = [32, 32, 64, 64, 128, 128, 256, 256]
        self.encoder_input = sum(self.feat_channels)

        # 분류기 헤드
        hidden_dim = (self.encoder_input + num_classes) // 2
        self.classifier_head = nn.Sequential(
            nn.Conv2d(self.encoder_input, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),

            nn.Dropout2d(0.2),

            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 백본 특징맵 피라미드 추출
        feats = self.backbone(x)

        # 특징맵 피라미드들을 최고 해상도 기준으로 합치기
        concat_feats = torch.cat([F.interpolate(f, size=feats[0].shape[2:], mode='nearest') for f in feats], dim=1)

        # 픽셀별 분류 헤드 적용
        logits = self.classifier_head(concat_feats)

        return logits

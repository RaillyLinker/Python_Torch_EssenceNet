import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
import numpy as np
import random


# ADE20K는 최대 150 클래스 + ignore_index(255)
# 시각화를 위해 무작위 색상 팔레트 생성 (0~149)
def get_color_map(N=150):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(N, 3), dtype=np.uint8)
    colors = np.vstack([[0, 0, 0], colors])  # 클래스 0은 검정색으로
    return colors


COLOR_MAP = get_color_map()


def decode_segmap(label: torch.Tensor, color_map=COLOR_MAP):
    label_np = label.cpu().numpy()
    h, w = label_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in np.unique(label_np):
        if class_id == 255:
            continue  # ignore index
        color = color_map[class_id + 1]  # class_id 0 → 색상 1번부터
        color_mask[label_np == class_id] = color

    return color_mask


def show_sample(dataset, idx=None):
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    sample = dataset[idx]
    image = sample["pixel_values"]  # torch.Tensor [3, H, W]
    label = sample["labels"]  # torch.Tensor [H, W]

    # 이미지 복원 (unnormalize)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # unnormalize
    image = torch.clamp(image, 0, 1)

    # to numpy
    image_np = TF.to_pil_image(image).convert("RGB")
    mask_np = decode_segmap(label)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title(f"Image [{idx}]")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np)
    plt.title(f"Segmentation Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 예시: 저장된 train 데이터셋 불러온 후 확인
from datasets import load_from_disk

train_ds = load_from_disk("C:/dataset/ade20k/train")
train_ds.set_format(type='torch', columns=['pixel_values', 'labels'])

show_sample(train_ds, idx=0)  # idx는 생략하면 랜덤

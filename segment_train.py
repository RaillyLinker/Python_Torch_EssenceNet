from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from nbb import EssenceNetSegClassifier
from tqdm import tqdm


class COCOSegmentationDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        root: 이미지가 저장된 폴더 경로 (예: ./coco/train2017)
        annFile: 어노테이션 json 파일 경로 (예: ./coco/annotations/instances_train2017.json)
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # 이미지 로드
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # segmentation mask 생성 (각 클래스별 픽셀값 할당)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            class_id = ann['category_id']  # COCO category id는 1~90까지 (0은 배경)
            # mask에 해당 category id로 segmentation 채우기
            rle = self.coco.annToRLE(ann)
            m = self.coco.decodeMask(rle)
            mask[m == 1] = class_id

        # 변환 적용
        if self.transform is not None:
            img = self.transform(img)
            mask = Image.fromarray(mask)
            mask = self.transform(mask)  # 마스크도 transform 해야하는데, tensor로 변환

            # mask는 tensor로 변환 후 long type (class index)
            mask = torch.squeeze(mask).long()

        return img, mask


# 데이터 변환 (이미지 크기 320x320 고정, 텐서 변환, 정규화)
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

train_dataset = COCOSegmentationDataset(
    root='F:/dataset/coco/train2017',
    annFile='F:/dataset/coco/annotations/instances_train2017.json',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

val_dataset = COCOSegmentationDataset(
    root='F:/dataset/coco/val2017',
    annFile='F:/dataset/coco/annotations/instances_val2017.json',
    transform=val_transform
)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EssenceNetSegClassifier(num_classes=91).to(device)  # COCO class 수 + background(0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = F.cross_entropy(outputs, masks, ignore_index=0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = F.cross_entropy(outputs, masks, ignore_index=0)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

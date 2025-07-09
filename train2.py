import os
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as F
from datasets import load_dataset, load_from_disk
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ColorJitter, InterpolationMode
from PIL import Image
from tqdm import tqdm

from nbb2 import EssenceNetSegmenter

# reproducibility seed
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# paths & constants
PRETRAINED_MODEL_PATH = None
NUM_CLASSES = 150  # ADE20K number of classes
TRAIN_DISK_PATH = "C:/dataset/ade20k/train"
VAL_DISK_PATH = "C:/dataset/ade20k/val"
LOG_DIR = "runs/ade_exp"
CHECKPOINT_DIR = "checkpoints"
INPUT_SIZE = 320


# joint 이미지·마스크 공간 변형 함수
def joint_transform(image: Image.Image, mask: Image.Image, size=INPUT_SIZE):
    # 1) RandomResizedCrop
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=(0.5, 1.0), ratio=(0.75, 1.33)
    )
    image = F.resized_crop(image, i, j, h, w, size=(size, size), interpolation=InterpolationMode.BILINEAR)
    mask = F.resized_crop(mask, i, j, h, w, size=(size, size), interpolation=InterpolationMode.NEAREST)
    # 2) RandomHorizontalFlip
    if random.random() < 0.5:
        image = F.hflip(image)
        mask = F.hflip(mask)
    # 3) RandomPerspective
    if random.random() < 0.3:
        start, end = transforms.RandomPerspective.get_params(image.height, image.width, distortion_scale=0.1)
        image = F.perspective(image, start, end, interpolation=InterpolationMode.BILINEAR)
        mask = F.perspective(mask, start, end, interpolation=InterpolationMode.NEAREST)
    # 4) ColorJitter & GaussianBlur on image only
    if random.random() < 0.8:
        image = ColorJitter(0.4, 0.4, 0.4, 0.1)(image)
    if random.random() < 0.3:
        k = random.choice((3, 5))
        image = transforms.GaussianBlur(k, sigma=(0.1, 1.5))(image)
    # ToTensor & Normalize
    image = F.to_tensor(image)
    image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, mask


def val_joint_transform(image: Image.Image, mask: Image.Image, size=INPUT_SIZE):
    # resize → center crop
    image = F.resize(image, 350, interpolation=InterpolationMode.BILINEAR)
    mask = F.resize(mask, 350, interpolation=InterpolationMode.NEAREST)
    image = F.center_crop(image, size)
    mask = F.center_crop(mask, size)
    # ToTensor & Normalize only for image
    image = F.to_tensor(image)
    image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, mask


# mask 처리 함수 (train과 val 동일하게 적용)
def process_mask(mask: Image.Image) -> torch.Tensor:
    arr = np.array(mask, dtype=np.int64)
    arr[arr == 0] = 255  # ignore index
    arr[arr != 255] -= 1  # 클래스 index 1~ → 0~ 로 변경
    return torch.from_numpy(arr)


# dataset 전처리 함수
def apply_ade_transform(example, mode: str):
    image = example['image'].convert('RGB')
    mask = example['instances'][0].convert('L')
    if mode == 'train':
        img_t, m_t = joint_transform(image, mask)
        labels = process_mask(m_t)
        return {'pixel_values': img_t, 'labels': labels}
    else:
        img_t, m_t = val_joint_transform(image, mask)
        labels = process_mask(m_t)
        return {'pixel_values': img_t, 'labels': labels}


# 한 epoch 학습
def train_epoch(model, loader, optimizer, device, scaler, criterion_fn, use_amp):
    model.train()
    running_loss = 0
    for batch in tqdm(loader, desc='Train', leave=False):
        imgs = batch['pixel_values'].to(device)
        masks = batch['labels'].to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(imgs)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                loss = criterion_fn(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
            loss = criterion_fn(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


# 한 epoch 검증
def eval_epoch(model, loader, device, criterion_fn):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val', leave=False):
            imgs = batch['pixel_values'].to(device)
            masks = batch['labels'].to(device)
            outputs = model(imgs)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
            loss = criterion_fn(outputs, masks)
            running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


# 메인 함수
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    # prepare dataset
    ds = load_dataset("1aurent/ADE20K")

    # train_raw = ds['train']
    # val_raw = ds.get('validation', ds.get('val'))

    train_raw = ds['train'][:100]
    val_raw = ds['validation'][:50]

    if not os.path.exists(TRAIN_DISK_PATH):
        train_ds = train_raw.map(lambda x: apply_ade_transform(x, 'train'), num_proc=1)
        train_ds.save_to_disk(TRAIN_DISK_PATH)
    else:
        train_ds = load_from_disk(TRAIN_DISK_PATH)

    if not os.path.exists(VAL_DISK_PATH):
        val_ds = val_raw.map(lambda x: apply_ade_transform(x, 'val'), num_proc=1)
        val_ds.save_to_disk(VAL_DISK_PATH)
    else:
        val_ds = load_from_disk(VAL_DISK_PATH)

    train_ds.set_format(type='torch', columns=['pixel_values', 'labels'])
    val_ds.set_format(type='torch', columns=['pixel_values', 'labels'])

    num_workers = 0 if os.name == 'nt' else 4
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem)

    # model, optimizer, scheduler 준비
    model = EssenceNetSegmenter(num_classes=NUM_CLASSES).to(device)
    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))

    criterion_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    scaler = GradScaler()

    best_loss = float('inf')
    best_path = None
    patience = 5
    no_improve = 0
    max_epochs = 30

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer,
                                 device, scaler, criterion_fn, use_amp)
        val_loss = eval_epoch(model, val_loader, device, criterion_fn)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step()

        # 체크포인트 저장
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'last.pth'))
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            best_path = os.path.join(
                CHECKPOINT_DIR, f"best_epoch{epoch:02d}_loss{val_loss:.4f}.pth"
            )
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {os.path.basename(best_path)}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()


if __name__ == '__main__':
    main()

import os
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as F
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ColorJitter, InterpolationMode
from PIL import Image
from tqdm import tqdm
import math
import shutil

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
TRAIN_DISK_PATH = "C:/dataset/ade20k_320/train"
VAL_DISK_PATH = "C:/dataset/ade20k_320/val"
LOG_DIR = "runs/ade_exp"
CHECKPOINT_DIR = "checkpoints"
INPUT_SIZE = 320


def apply_ade_transform_batch(batch, mode: str):
    images, labels = [], []
    for img, mask in zip(batch["image"], batch["annotation"]):
        img_t, m_t = (joint_transform if mode == 'train' else val_joint_transform)(img, mask)
        # PIL/Tensor → numpy int64
        m_arr = (np.array(m_t, dtype=np.int64)
                 if isinstance(m_t, Image.Image)
                 else (m_t.numpy().astype(np.int64) if torch.is_tensor(m_t)
                       else np.array(m_t, dtype=np.int64)))
        if m_arr.ndim == 3:
            m_arr = m_arr[0]
        # ==== 여기를 이렇게 바꿔주세요 ====
        # 1 ≤ original_label ≤ 150 → 0~149 로 매핑
        # 그 외 (0 혹은 >150) → 255 (ignore_index)
        m_arr = np.where((m_arr >= 1) & (m_arr <= NUM_CLASSES),
                         m_arr - 1,
                         255)
        labels.append(torch.tensor(m_arr, dtype=torch.long))
        images.append(img_t)

    images = torch.stack(images).numpy()
    labels = torch.stack(labels).numpy()
    return {'pixel_values': images, 'labels': labels}


def process_in_chunks(dataset, chunk_size, mode, save_dir, num_workers):
    os.makedirs(save_dir, exist_ok=True)
    total_len = len(dataset)
    num_chunks = math.ceil(total_len / chunk_size)
    print(f"Total samples: {total_len}, chunks: {num_chunks}")

    chunk_paths = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_len)
        chunk = dataset.select(range(start, end))

        print(f"Processing chunk {i + 1}/{num_chunks} ({start}~{end})...")

        chunk_path = os.path.join(save_dir, f"{mode}_chunk_{i}")
        if os.path.exists(chunk_path):
            print(f"Chunk {i + 1} already exists, skipping.")
            chunk_paths.append(chunk_path)
            continue

        processed = chunk.map(
            lambda x: apply_ade_transform_batch(x, mode),
            batched=True,
            batch_size=8,
            num_proc=num_workers
        )

        processed.save_to_disk(chunk_path)
        chunk_paths.append(chunk_path)

    # 모든 조각 불러와서 병합
    all_chunks = [load_from_disk(p) for p in chunk_paths]
    full_dataset = concatenate_datasets(all_chunks)

    full_dataset.save_to_disk(save_dir)

    return full_dataset


# joint 이미지·마스크 공간 변형 함수
def joint_transform(image: Image.Image, mask: Image.Image, size=INPUT_SIZE):
    if image.mode != "RGB":
        image = image.convert("RGB")
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
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize → center crop
    image = F.resize(image, 350, interpolation=InterpolationMode.BILINEAR)
    mask = F.resize(mask, 350, interpolation=InterpolationMode.NEAREST)
    image = F.center_crop(image, size)
    mask = F.center_crop(mask, size)
    # ToTensor & Normalize only for image
    image = F.to_tensor(image)
    image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, mask


# 한 epoch 학습
def train_epoch(model, loader, optimizer, device, scaler, criterion_fn, use_amp, writer, epoch):
    model.train()
    running_loss = 0
    correct_pixels = 0
    total_pixels = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        imgs = batch['pixel_values'].to(device)
        masks = batch['labels'].to(device)
        optimizer.zero_grad()

        # --- forward & loss ---
        if use_amp:
            with autocast("cuda"):
                outputs = model(imgs)
                # resize if needed
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

        # --- accuracy 계산 (ignore_index 제외) ---
        preds = torch.argmax(outputs, dim=1)
        valid_mask = masks != 255  # ignore_index == 255
        correct_pixels += (preds[valid_mask] == masks[valid_mask]).sum().item()
        total_pixels += valid_mask.sum().item()

    # --- epoch 단위 평균 loss & accuracy ---
    avg_loss = running_loss / len(loader.dataset)
    train_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    # --- 화면 출력 및 TensorBoard 기록 ---
    print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)

    return avg_loss, train_acc


# 한 epoch 검증
def eval_epoch(model, loader, device, criterion_fn):
    model.eval()
    running_loss = 0
    correct_pixels = 0
    total_pixels = 0
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

            # 🔍 정확도 계산
            preds = torch.argmax(outputs, dim=1)  # 예측 클래스
            mask_valid = masks != 255  # ignore index 마스크
            correct_pixels += (preds[mask_valid] == masks[mask_valid]).sum().item()
            total_pixels += mask_valid.sum().item()

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    return avg_loss, accuracy


# 메인 함수
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    # prepare dataset
    ds = load_dataset("scene_parse_150", trust_remote_code=True)

    # # 2. 전체 구조 및 column 이름 보기
    # print(ds)  # DatasetDict({'train': Dataset, 'validation': Dataset})
    # print(ds['train'].column_names)
    # print(ds['train'].features)
    #
    # # 3. 샘플 출력
    # print(ds['train'][0])  # 또는 random.sample(ds['train'], 3)

    train_raw = ds['train']
    val_raw = ds['validation']

    # train_raw = ds['train'].select(range(100))
    # val_raw = ds['validation'].select(range(50))
    num_workers = 4
    chunk_size = 1000

    if not os.path.exists(TRAIN_DISK_PATH):
        train_ds = process_in_chunks(train_raw, chunk_size=chunk_size, mode='train', save_dir=TRAIN_DISK_PATH,
                                     num_workers=num_workers)
    else:
        train_ds = load_from_disk(TRAIN_DISK_PATH)

    if not os.path.exists(VAL_DISK_PATH):
        val_ds = process_in_chunks(val_raw, chunk_size=chunk_size, mode='val', save_dir=VAL_DISK_PATH,
                                   num_workers=num_workers)
    else:
        val_ds = load_from_disk(VAL_DISK_PATH)

    train_ds.set_format(type='torch', columns=['pixel_values', 'labels'])
    val_ds.set_format(type='torch', columns=['pixel_values', 'labels'])

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
    scaler = GradScaler(device)

    best_loss = float('inf')
    best_path = None
    patience = 5
    no_improve = 0
    max_epochs = 30

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, scaler,
            criterion_fn, use_amp, writer, epoch
        )
        val_loss, val_acc = eval_epoch(model, val_loader, device, criterion_fn)

        # 기존 writer 기록은 train_epoch 내부에서 했으니 val만 추가
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc * 100:.2f}%")

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

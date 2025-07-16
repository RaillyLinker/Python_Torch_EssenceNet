import os
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ColorJitter, InterpolationMode
from PIL import Image
from tqdm import tqdm
import math
import shutil
import time
from torchmetrics import ConfusionMatrix
from nbb2 import EssenceNetSegmenter
from contextlib import nullcontext

PRETRAINED_MODEL_PATH = None
NUM_CLASSES = 150
TRAIN_DISK_PATH = "C:/dataset/ade20k_320/train"
VAL_DISK_PATH = "C:/dataset/ade20k_320/val"
LOG_DIR = "runs/ade_exp"
CHECKPOINT_DIR = "checkpoints"
INPUT_SIZE = 320
SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
MAX_EPOCHS = 30
PATIENCE = 5
DEBUG_VIS = True

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_miou_stats(intersection: np.ndarray, union: np.ndarray):
    iou = np.where(union > 0, intersection / (union + 1e-10), np.nan)
    miou = np.nanmean(iou) if np.any(~np.isnan(iou)) else 0.0
    return miou, iou


def apply_ade_transform_batch(batch, mode: str):
    images, labels = [], []
    for img, mask in zip(batch["image"], batch["annotation"]):
        img_t, m_t = (train_transform if mode == 'train' else val_transform)(img, mask)
        m_arr = np.array(m_t, dtype=np.int64)
        if m_arr.ndim == 3:
            m_arr = m_arr[0]
        m_arr = np.where((m_arr >= 1) & (m_arr <= NUM_CLASSES), m_arr - 1, 255)
        images.append(img_t)
        labels.append(torch.tensor(m_arr, dtype=torch.long))
    return {'pixel_values': torch.stack(images), 'labels': torch.stack(labels)}


def process_in_chunks(ds, chunk_size, mode, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    total = len(ds)
    chunks = math.ceil(total / chunk_size)
    paths = []
    for i in range(chunks):
        start, end = i * chunk_size, min((i + 1) * chunk_size, total)
        out_path = os.path.join(save_dir, f"{mode}_chunk_{i}")
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
        sub = ds.select(range(start, end))
        proc = sub.map(lambda x: apply_ade_transform_batch(x, mode), batched=True,
                       batch_size=BATCH_SIZE, num_proc=NUM_WORKERS)
        proc.save_to_disk(out_path)
        paths.append(out_path)
    full = concatenate_datasets([load_from_disk(p) for p in paths])
    full.save_to_disk(save_dir)
    return full


COLOR_JITTER = ColorJitter(0.4, 0.4, 0.4, 0.1)
GAUSSIAN_BLURS = {
    3: transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    5: transforms.GaussianBlur(5, sigma=(0.1, 1.5)),
}


def train_transform(image: Image.Image, mask: Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=(0.5, 1.0), ratio=(0.75, 1.33)
    )
    image = TF.resized_crop(
        image, i, j, h, w,
        size=(INPUT_SIZE, INPUT_SIZE),
        interpolation=InterpolationMode.BILINEAR
    )
    mask = TF.resized_crop(
        mask, i, j, h, w,
        size=(INPUT_SIZE, INPUT_SIZE),
        interpolation=InterpolationMode.NEAREST
    )

    if random.random() < 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)

    if random.random() < 0.3:
        start, end = transforms.RandomPerspective.get_params(
            image.height, image.width, distortion_scale=0.1
        )
        image = TF.perspective(
            image, start, end, interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.perspective(
            mask, start, end, interpolation=InterpolationMode.NEAREST
        )

    if random.random() < 0.8:
        image = COLOR_JITTER(image)

    if random.random() < 0.3:
        k = random.choice((3, 5))
        image = GAUSSIAN_BLURS[k](image)

    image = TF.to_tensor(image)
    image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return image, mask


def val_transform(image: Image.Image, mask: Image.Image):
    if image.mode != 'RGB': image = image.convert('RGB')
    image = TF.resize(image, 350, interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, 350, interpolation=InterpolationMode.NEAREST)
    image, mask = TF.center_crop(image, INPUT_SIZE), TF.center_crop(mask, INPUT_SIZE)
    image = F.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, mask


def train_epoch(model, loader, optimizer, device, scaler, writer, epoch, amp_ctx):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    start = time.time()
    inference_times = []
    for batch in tqdm(loader, desc='Train', leave=False):
        imgs, masks = batch['pixel_values'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        with amp_ctx:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.time()
            outputs = model(imgs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.time()
            inference_times.append(t1 - t0)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = F.cross_entropy(outputs, masks, ignore_index=255)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        mask = masks != 255
        correct += (preds[mask] == masks[mask]).sum().item()
        total += mask.sum().item()
    elapsed = time.time() - start
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total if total > 0 else 0
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)
    writer.add_scalar('Time/train_epoch', elapsed, epoch)
    avg_infer = np.mean(inference_times)
    writer.add_scalar('Time/train_infer_avg', avg_infer, epoch)
    print(f"Train {epoch}: Inference Time Avg = {avg_infer:.4f}s")
    print(f"Train {epoch}: Loss={avg_loss:.4f}, Acc={acc * 100:.2f}%, Time={elapsed:.2f}s")
    return avg_loss, acc


def eval_epoch(model, loader, device, writer, epoch, confmat):
    confmat.reset()
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0

    start = time.time()
    with torch.no_grad(), autocast(device_type='cuda') if device.type == 'cuda' else nullcontext():
        for batch in tqdm(loader, desc='Val', leave=False):
            imgs = batch['pixel_values'].to(device)
            masks = batch['labels'].to(device)

            outputs = model(imgs)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = F.cross_entropy(outputs, masks, ignore_index=255)
            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(1)
            valid = masks != 255

            correct += (preds[valid] == masks[valid]).sum().item()
            total += valid.sum().item()

            confmat.update(preds[valid].flatten(), masks[valid].flatten())

    elapsed = time.time() - start
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total if total > 0 else 0

    cm = confmat.compute()
    inter = torch.diagonal(cm).double()
    union = cm.sum(1).double() + cm.sum(0).double() - inter
    miou = torch.nanmean(inter / (union + 1e-10)).item()

    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', acc, epoch)
    writer.add_scalar('mIoU/val', miou, epoch)
    writer.add_scalar('Time/val_epoch', elapsed, epoch)

    print(f"Val   {epoch}: Loss={avg_loss:.4f}, Acc={acc * 100:.2f}%, mIoU={miou * 100:.2f}%, Time={elapsed:.2f}s")
    return avg_loss, acc, miou


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    writer = SummaryWriter(LOG_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    confmat = ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES).to(device)
    amp_ctx = autocast(device_type='cuda') if use_amp else nullcontext()

    ds = load_dataset('scene_parse_150', trust_remote_code=True)
    train_raw, val_raw = ds['train'], ds['validation']
    train_ds = load_from_disk(TRAIN_DISK_PATH) if os.path.exists(TRAIN_DISK_PATH) else process_in_chunks(train_raw,
                                                                                                         1000, 'train',
                                                                                                         TRAIN_DISK_PATH)
    val_ds = load_from_disk(VAL_DISK_PATH) if os.path.exists(VAL_DISK_PATH) else process_in_chunks(val_raw, 1000, 'val',
                                                                                                   VAL_DISK_PATH)
    train_ds.set_format(type='torch', columns=['pixel_values', 'labels'])
    val_ds.set_format(type='torch', columns=['pixel_values', 'labels'])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'), persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'), persistent_workers=True)

    model = EssenceNetSegmenter(num_classes=NUM_CLASSES).to(device)
    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = GradScaler() if use_amp else None

    best_loss = float('inf')
    no_imp = 0
    for ep in range(1, MAX_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, device, scaler, writer, ep, amp_ctx)
        val_loss, _, _ = eval_epoch(model, val_loader, device, writer, ep, confmat)
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'last.pth'))
        if val_loss < best_loss:
            best_loss, no_imp = val_loss, 0
            path = os.path.join(CHECKPOINT_DIR, f'best_ep{ep:02d}_{val_loss:.4f}.pth')
            torch.save(model.state_dict(), path)
            print(f"Saved best: {os.path.basename(path)}")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"Early stopping at ep{ep}")
                break
    writer.close()


if __name__ == '__main__':
    main()

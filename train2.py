import os
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
import torch.nn.functional as F
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
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from nbb2 import EssenceNetSegmenter

PRETRAINED_MODEL_PATH = None
NUM_CLASSES = 150  # ADE20K number of classes
TRAIN_DISK_PATH = "C:/dataset/ade20k_320/train"
VAL_DISK_PATH = "C:/dataset/ade20k_320/val"
LOG_DIR = "runs/ade_exp"
CHECKPOINT_DIR = "checkpoints"
INPUT_SIZE = 320
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_miou(y_pred, y_true, num_classes, ignore_index=255, return_iou=False):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    mask = y_true != ignore_index
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    conf_matrix = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy(), labels=list(range(num_classes)))
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    iou = np.where(union > 0, intersection / (union + 1e-10), np.nan)
    miou = np.nanmean(iou) if np.any(~np.isnan(iou)) else 0.0

    return (miou, iou) if return_iou else miou


def apply_ade_transform_batch(batch, mode: str):
    images, labels = [], []
    for img, mask in zip(batch["image"], batch["annotation"]):
        img_t, m_t = (joint_transform if mode == 'train' else val_joint_transform)(img, mask)
        m_arr = (np.array(m_t, dtype=np.int64)
                 if isinstance(m_t, Image.Image)
                 else (m_t.numpy().astype(np.int64) if torch.is_tensor(m_t)
                       else np.array(m_t, dtype=np.int64)))
        if m_arr.ndim == 3:
            m_arr = m_arr[0]
        m_arr = np.where((m_arr >= 1) & (m_arr <= NUM_CLASSES), m_arr - 1, 255)
        labels.append(torch.tensor(m_arr, dtype=torch.long))
        images.append(img_t)

    images = torch.stack(images)
    labels = torch.stack(labels)
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

    all_chunks = [load_from_disk(p) for p in chunk_paths]
    full_dataset = concatenate_datasets(all_chunks)
    full_dataset.save_to_disk(save_dir)

    return full_dataset


def joint_transform(image: Image.Image, mask: Image.Image, size=INPUT_SIZE):
    if image.mode != "RGB":
        image = image.convert("RGB")
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=(0.5, 1.0), ratio=(0.75, 1.33)
    )
    image = TF.resized_crop(image, i, j, h, w, size=(size, size), interpolation=InterpolationMode.BILINEAR)
    mask = TF.resized_crop(mask, i, j, h, w, size=(size, size), interpolation=InterpolationMode.NEAREST)
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if random.random() < 0.3:
        start, end = transforms.RandomPerspective.get_params(image.height, image.width, distortion_scale=0.1)
        image = TF.perspective(image, start, end, interpolation=InterpolationMode.BILINEAR)
        mask = TF.perspective(mask, start, end, interpolation=InterpolationMode.NEAREST)
    if random.random() < 0.8:
        image = ColorJitter(0.4, 0.4, 0.4, 0.1)(image)
    if random.random() < 0.3:
        k = random.choice((3, 5))
        image = transforms.GaussianBlur(k, sigma=(0.1, 1.5))(image)
    image = TF.to_tensor(image)
    image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, mask


def val_joint_transform(image: Image.Image, mask: Image.Image, size=INPUT_SIZE):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = TF.resize(image, 350, interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, 350, interpolation=InterpolationMode.NEAREST)
    image = TF.center_crop(image, size)
    mask = TF.center_crop(mask, size)
    image = TF.to_tensor(image)
    image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, mask


def train_epoch(model, loader, optimizer, device, scaler, criterion_fn, use_amp, writer, epoch):
    model.train()
    running_loss = 0
    correct_pixels = 0
    total_pixels = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        imgs = batch['pixel_values'].to(device)
        masks = batch['labels'].to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast("cuda"):
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

        preds = torch.argmax(outputs, dim=1)
        valid_mask = masks != 255  # ignore_index == 255
        correct_pixels += (preds[valid_mask] == masks[valid_mask]).sum().item()
        total_pixels += valid_mask.sum().item()

    end_time = time.time()
    elapsed = end_time - start_time
    samples_per_sec = len(loader.dataset) / elapsed

    avg_loss = running_loss / len(loader.dataset)
    train_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    tqdm.write(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
               f"Epoch Time: {elapsed:.2f}s | Samples/sec: {samples_per_sec:.2f}")

    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Speed/train_samples_per_sec', samples_per_sec, epoch)
    writer.add_scalar('Time/train_epoch_time', elapsed, epoch)

    return avg_loss, train_acc


def eval_epoch(model, loader, device, criterion_fn, writer=None, epoch=None):
    model.eval()
    running_loss = 0
    correct_pixels = 0
    total_pixels = 0
    all_preds = []
    all_targets = []

    start_time = time.time()
    first_batch_saved = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Val', leave=False)):
            imgs = batch['pixel_values'].to(device)
            masks = batch['labels'].to(device)
            outputs = model(imgs)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
            loss = criterion_fn(outputs, masks)
            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(outputs, dim=1)
            valid_mask = masks != 255
            correct_pixels += (preds[valid_mask] == masks[valid_mask]).sum().item()
            total_pixels += valid_mask.sum().item()

            all_preds.append(preds)
            all_targets.append(masks)

            if (not first_batch_saved) and (epoch is not None):
                rand_idx = random.randint(0, imgs.size(0) - 1)
                img_np = imgs[rand_idx].detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # De-normalize
                img_np = np.clip(img_np, 0, 1)

                pred_mask = preds[rand_idx].cpu().numpy()
                gt_mask = masks[rand_idx].cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img_np)
                axs[0].set_title("Input Image")
                axs[1].imshow(gt_mask, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES - 1)
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred_mask, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES - 1)
                axs[2].set_title("Prediction")

                for ax in axs:
                    ax.axis("off")

                vis_dir = os.path.join(LOG_DIR, "val_vis")
                os.makedirs(vis_dir, exist_ok=True)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"epoch_{epoch:02d}_sample.png"))
                plt.close()

                first_batch_saved = True

    end_time = time.time()
    elapsed = end_time - start_time
    samples_per_sec = len(loader.dataset) / elapsed

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    miou = calculate_miou(
        torch.cat(all_preds).reshape(-1),
        torch.cat(all_targets).reshape(-1),
        NUM_CLASSES
    )

    print(f"Val Epoch Time: {elapsed:.2f}s | Samples/sec: {samples_per_sec:.2f}")

    if writer is not None and epoch is not None:
        writer.add_scalar('Speed/val_samples_per_sec', samples_per_sec, epoch)
        writer.add_scalar('Time/val_epoch_time', elapsed, epoch)

    return avg_loss, accuracy, miou


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    ds = load_dataset("scene_parse_150", trust_remote_code=True)

    train_raw = ds['train']
    val_raw = ds['validation']

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

    total_start_time = time.time()  # 전체 학습 시간 시작

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, scaler,
            criterion_fn, use_amp, writer, epoch
        )
        val_loss, val_acc, val_miou = eval_epoch(model, val_loader, device, criterion_fn, writer, epoch)

        # 기존 writer 기록은 train_epoch 내부에서 했으니 val만 추가
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('mIoU/val', val_miou, epoch)

        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc * 100:.2f}% | mIoU: {val_miou * 100:.2f}%")

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

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    print(f"\nTotal Training Time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")

    writer.close()


if __name__ == '__main__':
    main()

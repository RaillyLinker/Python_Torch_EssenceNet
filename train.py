import os
import random
import time
import glob

import torch
import torch.optim as optim
from datasets import load_dataset, load_from_disk
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ColorJitter, RandomErasing
from tqdm import tqdm

from nbb import EssenceNetClassifier

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

PRETRAINED_MODEL_PATH = None


# -------------------------
# Mixup 함수
# -------------------------
def mixup_data(x, y, alpha=0.4):
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ----------------------------
# 전처리 정의
# ----------------------------
class RandomizedGaussianBlur:
    def __init__(self, kernel_sizes=(3, 5), sigma=(0.1, 1.5), p=0.3):
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            k = random.choice(self.kernel_sizes)
            return transforms.GaussianBlur(kernel_size=k, sigma=self.sigma)(img)
        return img


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(320, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    RandomizedGaussianBlur(p=0.3),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.1)], p=0.3),
    transforms.ToTensor(),
    RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.CenterCrop((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_transform(example, mode):
    try:
        image = example["image"].convert("RGB")
    except Exception:
        return None
    if mode == "train":
        example["pixel_values"] = train_transform(image)
    else:
        example["pixel_values"] = val_transform(image)
    return {"pixel_values": example["pixel_values"], "label": example["label"]}


if __name__ == "__main__":
    writer = SummaryWriter(log_dir="runs/exp1")

    # ----------------------------
    # 데이터셋 로드 또는 전처리
    # ----------------------------
    if os.path.exists("./processed/food101/train") and os.path.exists("./processed/food101/val"):
        print("📦 전처리된 데이터셋 로딩 중...")
        train_ds = load_from_disk("./processed/food101/train")
        val_ds = load_from_disk("./processed/food101/val")
    else:
        print("⚙️ 전처리 중 (최초 실행 시 1회)...")
        raw_train_ds = load_dataset(
            "food101",
            split="train",
            # split="train[:100]"
        )
        raw_val_ds = load_dataset(
            "food101",
            split="validation",
            # split="validation[:50]"
        )

        train_ds = raw_train_ds.map(lambda x: apply_transform(x, mode="train"), num_proc=8)
        val_ds = raw_val_ds.map(lambda x: apply_transform(x, mode="val"), num_proc=8)

        train_ds = train_ds.filter(lambda x: x is not None)
        val_ds = val_ds.filter(lambda x: x is not None)

        train_ds.save_to_disk("./processed/food101/train")
        val_ds.save_to_disk("./processed/food101/val")
        print("✅ 전처리 및 저장 완료.")

    train_ds.set_format(type='torch', columns=['pixel_values', 'label'])
    val_ds.set_format(type='torch', columns=['pixel_values', 'label'])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4,
                            persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EssenceNetClassifier(num_classes=101).to(device)

    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print(f"Loaded pretrained model from {PRETRAINED_MODEL_PATH}")

    criterion_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-6  # 초기 정규화 약하게
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    scaler = GradScaler(device='cuda')


    def train_epoch(model, dataloader, optimizer, device, scaler, use_mixup=True):
        model.train()
        total_loss = total_correct = total_samples = 0
        pbar = tqdm(dataloader, desc="Training", leave=False)

        for batch in pbar:
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                if use_mixup:
                    inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
                    outputs = model(inputs_mixed)
                    loss = mixup_criterion(criterion_fn, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(inputs)
                    loss = criterion_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            if use_mixup:
                total_correct += ((preds == targets_a) | (preds == targets_b)).sum().item()
            else:
                total_correct += (preds == labels).sum().item()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}", "Avg Acc": f"{total_correct / total_samples:.4f}"})

        return total_loss / total_samples, total_correct / total_samples


    def eval_epoch(model, dataloader, device):
        model.eval()
        total_loss = total_correct = total_samples = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for batch in pbar:
                inputs = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = model(inputs)
                loss = criterion_fn(outputs, labels)
                preds = outputs.argmax(dim=1)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)
                pbar.set_postfix(
                    {"Batch Loss": f"{loss.item():.4f}", "Avg Acc": f"{total_correct / total_samples:.4f}"})

        end_time = time.perf_counter()
        inference_time = end_time - start_time
        avg_inf_time = inference_time / total_samples
        return total_loss / total_samples, total_correct / total_samples, inference_time, avg_inf_time


    def interpolate_weight_decay(epoch, max_epoch, min_wd=1e-6, max_wd=1e-4):
        alpha = epoch / max_epoch
        return (1 - alpha) * min_wd + alpha * max_wd


    # ----------------------------
    # 학습 실행
    # ----------------------------
    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0
    no_improve_epochs = 0
    patience = 5
    max_epochs = 30

    for epoch in range(1, max_epochs + 1):
        new_wd = interpolate_weight_decay(epoch, max_epochs)
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = new_wd

        print(f"\n===== Epoch {epoch} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scaler, use_mixup=True)
        val_loss, val_acc, val_time, avg_inf_time = eval_epoch(model, val_loader, device)
        scheduler.step()

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Inference Time: {val_time:.2f}s | Avg Per Image: {avg_inf_time * 1000:.2f}ms")

        torch.save(model.state_dict(), "checkpoints/last.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            for f in glob.glob("checkpoints/best_*.pth"):
                os.remove(f)
            best_path = f"checkpoints/best_epoch{epoch:02d}_acc{val_acc:.4f}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"💾 Saved best model as {os.path.basename(best_path)}.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"⏱ Early stopping at epoch {epoch} after {patience} epochs without improvement.")
                break

    writer.close()

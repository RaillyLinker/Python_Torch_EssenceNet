import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
import glob
from nbb import UpsampleConcatClassifier  # 사용자 정의 모델 import

# -----------------------------------
# 사전 학습된 모델 경로 (없으면 None)
# -----------------------------------
# PRETRAINED_MODEL_PATH = "checkpoints/best_model_epoch6.pth"  # 또는 None
PRETRAINED_MODEL_PATH = None  # 또는 None

if __name__ == "__main__":
    # ----------------------------
    # 데이터셋 로드 및 전처리
    # ----------------------------
    train_ds = load_dataset("food101", split="train")
    val_ds = load_dataset("food101", split="validation")


    def transform_example(example):
        from PIL import Image
        try:
            image = example["image"].convert("RGB")
        except Exception:
            return None

        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        example["pixel_values"] = transform(image)
        return example


    train_ds = train_ds.map(transform_example, num_proc=1)
    val_ds = val_ds.map(transform_example, num_proc=1)
    train_ds.set_format(type='torch', columns=['pixel_values', 'label'])
    val_ds.set_format(type='torch', columns=['pixel_values', 'label'])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    # ----------------------------
    # 모델/옵티마이저/로스 설정
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UpsampleConcatClassifier(num_classes=101).to(device)

    # ✅ 사전 학습된 모델 불러오기
    if PRETRAINED_MODEL_PATH is not None and os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print(f"✅ Loaded pretrained model from {PRETRAINED_MODEL_PATH}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)


    # ----------------------------
    # 학습 및 평가 함수 정의
    # ----------------------------
    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

            pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Acc": f"{(total_correct / total_samples):.4f}"
            })

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy


    def eval_epoch(model, dataloader, criterion, device):
        model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        start_time = time.perf_counter()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for batch in pbar:
                inputs = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)

                pbar.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "Avg Acc": f"{(total_correct / total_samples):.4f}"
                })

        end_time = time.perf_counter()
        inference_time = end_time - start_time
        avg_inference_time = inference_time / total_samples

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy, inference_time, avg_inference_time


    # ----------------------------
    # 전체 학습 실행 + 모델 저장
    # ----------------------------
    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0
    epoch = 0
    prev_train_acc = 0.0
    prev_val_acc = 0.0

    while True:
        epoch += 1
        print(f"\n===== Epoch {epoch} =====")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_time, avg_inf_time = eval_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Inference Time: {val_time:.2f}s | Avg Per Image: {avg_inf_time * 1000:.2f}ms")

        # ✅ 에포크별 모델 저장
        if epoch % 2 == 0:
            new_path = (
                f"checkpoints/epoch{epoch}_"
                f"train{train_acc * 100:.2f}_val{val_acc * 100:.2f}.pth"
            )
            torch.save(model.state_dict(), new_path)

        # ✅ 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            # 기존 best 모델 삭제
            for f in glob.glob("checkpoints/best_*.pth"):
                os.remove(f)

            # 새 파일명 생성
            new_best_path = (
                f"checkpoints/best_epoch{epoch}_"
                f"train{train_acc * 100:.2f}_val{val_acc * 100:.2f}.pth"
            )

            torch.save(model.state_dict(), new_best_path)
            print(f"💾 Saved best model as {os.path.basename(new_best_path)}.")

        # 상태 갱신
        prev_train_acc = train_acc
        prev_val_acc = val_acc

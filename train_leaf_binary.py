import argparse, os, time, json, math, random
from pathlib import Path

import torch
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_dataloaders(data_dir, img_size=224, bs=32, num_workers=4):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf_train = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(img_size),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean, std),
    ])
    tf_eval = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean, std),
    ])
    train_ds = tv.datasets.ImageFolder(f"{data_dir}/train", tf_train)
    val_root = Path(f"{data_dir}/validation")
    if not val_root.exists():

        print("No 'validation' split found; creating 85/15 split from train...")
        full = tv.datasets.ImageFolder(f"{data_dir}/train", tf_eval)

        idx_by_class = {}
        for i, (_, y) in enumerate(full.samples):
            idx_by_class.setdefault(y, []).append(i)
        val_idx, train_idx = [], []
        for _, idxs in idx_by_class.items():
            cut = max(1, int(0.15 * len(idxs)))
            random.shuffle(idxs)
            val_idx += idxs[:cut]
            train_idx += idxs[cut:]
        train_ds = torch.utils.data.Subset(train_ds, train_idx)
        val_ds = torch.utils.data.Subset(full, val_idx)
    else:
        val_ds = tv.datasets.ImageFolder(f"{data_dir}/validation", tf_eval)

    test_root = Path(f"{data_dir}/test")
    test_ds = tv.datasets.ImageFolder(f"{data_dir}/test", tf_eval) if test_root.exists() else None

    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  bs, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None
    class_names = (train_ds.dataset.classes if isinstance(train_ds, torch.utils.data.Subset)
                   else train_ds.classes)
    return train_dl, val_dl, test_dl, class_names

def build_model(num_classes=2):
    model = tv.models.mobilenet_v3_small(weights=tv.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, dl, device, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
        n += y.size(0)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    avg_loss = total_loss / max(1, n)
    acc = (y_true == y_pred).mean()
    return avg_loss, acc, y_true, y_pred

def train(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.data
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, test_dl, class_names = get_dataloaders(data_dir, args.img, args.bs, args.workers)
    print("Classes:", class_names)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = 0.0
    best_path = out_dir / "leaf_binary.pt"
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss, tr_correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        scheduler.step()
        tr_loss /= max(1, n); tr_acc = tr_correct / max(1, n)

        val_loss, val_acc, y_true, y_pred = evaluate(model, val_dl, device, criterion)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} acc {tr_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(),
                        "class_names": class_names}, best_path)
            print(f"  ✓ Saved best to {best_path} (val_acc={best_val:.3f})")

        if len(history) > 6 and max(h["val_acc"] for h in history[-6:-1]) >= history[-1]["val_acc"]:
            print("Early stopping triggered.")
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    val_loss, val_acc, y_true, y_pred = evaluate(model, val_dl, device, criterion)
    print("\nValidation report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    if test_dl:
        te_loss, te_acc, y_true, y_pred = evaluate(model, test_dl, device, criterion)
        print("\nTest report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_val_acc": best_val, "history": history}, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./pp2021_binary", help="root with train/val[/test]")
    p.add_argument("--out",  default="./models", help="output folder")
    p.add_argument("--img",  type=int, default=224)
    p.add_argument("--bs",   type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr",   type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    train(args)

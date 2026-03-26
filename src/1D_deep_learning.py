import copy
import argparse
import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """设置随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    work_dir = "work_save/legacy_train"
    pca_dir = os.path.join(work_dir, "降维结果")
    pseudo_dir = os.path.join(pca_dir, "伪图像")
    model_dir = os.path.join(work_dir, "模型结果")
    os.makedirs(pca_dir, exist_ok=True)
    os.makedirs(pseudo_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return work_dir, pca_dir, pseudo_dir, model_dir


def load_spectral_data(excel_path="光谱数据.xlsx", sheet_name=0):
    """加载并清洗光谱数据。要求前两列分别为样本ID和有机质含量。"""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"未找到数据文件: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    if df.shape[1] < 3:
        raise ValueError("Excel列数不足，至少需要3列（ID、目标、光谱）。")

    sample_ids = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    targets = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    spectral_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

    valid_mask = sample_ids.notna() & targets.notna() & spectral_df.notna().all(axis=1)
    if valid_mask.sum() == 0:
        raise ValueError("清洗后无有效样本，请检查Excel数据格式。")

    sample_ids = sample_ids[valid_mask].astype(int).to_numpy()
    targets = targets[valid_mask].astype(np.float32).to_numpy()
    spectral = spectral_df[valid_mask].to_numpy(dtype=np.float32)

    return sample_ids, targets, spectral


def fit_scaler_pca(train_spectral, pca_components=3):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_spectral)

    max_comp = min(train_scaled.shape[0], train_scaled.shape[1])
    if pca_components > max_comp:
        pca_components = max_comp

    pca = PCA(n_components=pca_components, random_state=42)
    pca.fit(train_scaled)
    return scaler, pca


def transform_with_scaler_pca(spectral, scaler, pca):
    scaled = scaler.transform(spectral)
    return pca.transform(scaled)


def pc_to_rgb_images(pc_values, image_size=(64, 64), min_vals=None, max_vals=None, noise_std=0.0):
    """将前三主成分映射成RGB伪图像。"""
    if pc_values.shape[1] < 3:
        raise ValueError("PCA维度不足3，无法映射为RGB。")

    pcs3 = pc_values[:, :3]
    if min_vals is None:
        min_vals = pcs3.min(axis=0)
    if max_vals is None:
        max_vals = pcs3.max(axis=0)

    denom = np.where((max_vals - min_vals) < 1e-8, 1.0, (max_vals - min_vals))
    norm = np.clip((pcs3 - min_vals) / denom, 0.0, 1.0)
    rgb = (norm * 255).astype(np.uint8)

    h, w = image_size
    images = np.zeros((rgb.shape[0], h, w, 3), dtype=np.uint8)
    images[:, :, :, 0] = rgb[:, 0][:, None, None]
    images[:, :, :, 1] = rgb[:, 1][:, None, None]
    images[:, :, :, 2] = rgb[:, 2][:, None, None]

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, images.shape)
        images = np.clip(images.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return images, min_vals, max_vals


def save_pseudo_images(sample_ids, pseudo_images, pseudo_dir):
    for sid, arr in zip(sample_ids, pseudo_images):
        path = os.path.join(pseudo_dir, f"样本_{int(sid)}.png")
        Image.fromarray(arr).save(path)


class MultiModalDataset(Dataset):
    """伪图像 + 真实图像多模态数据集。"""

    def __init__(self, sample_ids, targets, pseudo_images, real_image_dir, pseudo_transform, real_transform):
        self.sample_ids = sample_ids
        self.targets = targets
        self.pseudo_images = pseudo_images
        self.real_image_dir = real_image_dir
        self.pseudo_transform = pseudo_transform
        self.real_transform = real_transform
        self.real_exists = np.array(
            [os.path.exists(os.path.join(real_image_dir, f"{int(sid)}.jpg")) for sid in sample_ids],
            dtype=bool,
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = int(self.sample_ids[idx])
        target = float(self.targets[idx])

        pseudo_pil = Image.fromarray(self.pseudo_images[idx])
        pseudo_tensor = self.pseudo_transform(pseudo_pil)

        real_path = os.path.join(self.real_image_dir, f"{sid}.jpg")
        if self.real_exists[idx]:
            try:
                real_pil = Image.open(real_path).convert("RGB")
            except Exception:
                real_pil = Image.new("RGB", (224, 224), color=(160, 160, 160))
        else:
            real_pil = Image.new("RGB", (224, 224), color=(200, 200, 200))

        real_tensor = self.real_transform(real_pil)
        mask = torch.tensor([1.0 if self.real_exists[idx] else 0.0], dtype=torch.float32)

        return pseudo_tensor, real_tensor, mask, torch.tensor([target], dtype=torch.float32), sid


class MultiModalSoilModel(nn.Module):
    def __init__(self, pseudo_dropout=0.3, real_dropout=0.4, use_pretrained=True):
        super().__init__()

        self.pseudo_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(pseudo_dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.real_backbone = self._build_resnet18(use_pretrained)
        in_features = self.real_backbone.fc.in_features
        self.real_backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(real_dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(real_dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 64 + 1, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    @staticmethod
    def _build_resnet18(use_pretrained=True):
        if not use_pretrained:
            return models.resnet18(weights=None)

        try:
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            # 离线环境下回退到非预训练，避免首次运行报错中断。
            return models.resnet18(weights=None)

    def forward(self, pseudo_x, real_x, mask):
        pseudo_features = self.pseudo_branch(pseudo_x)
        real_features = self.real_backbone(real_x)

        real_features = real_features * mask.expand(-1, real_features.size(1))
        fusion = torch.cat([pseudo_features, real_features, mask], dim=1)
        return self.fusion_net(fusion)


def make_transforms():
    train_tf = {
        "pseudo": transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
        "real": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    }

    val_tf = {
        "pseudo": transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
        "real": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    }
    return train_tf, val_tf


def run_epoch(model, loader, optimizer, criterion, device, scaler_amp=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds, gts, masks = [], [], []

    for pseudo, real, mask, target, _ in loader:
        pseudo = pseudo.to(device)
        real = real.to(device)
        mask = mask.to(device)
        target = target.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(scaler_amp is not None)):
                out = model(pseudo, real, mask)
                loss = criterion(out, target)

            if train:
                if scaler_amp is not None:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()
        preds.append(out.detach().cpu().numpy())
        gts.append(target.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    preds = np.concatenate(preds).reshape(-1)
    gts = np.concatenate(gts).reshape(-1)
    masks = np.concatenate(masks).reshape(-1)

    mse = mean_squared_error(gts, preds)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(gts, preds)) if len(np.unique(gts)) > 1 else np.nan

    return avg_loss, rmse, r2, preds, gts, masks


def train_model(model, train_loader, val_loader, device, epochs=80, patience=12, lr=1e-3, weight_decay=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    use_amp = device.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = None
    best_val = float("inf")
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_rmse, train_r2, _, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, scaler_amp=scaler_amp, train=True
        )
        val_loss, val_rmse, val_r2, _, _, _ = run_epoch(
            model, val_loader, optimizer, criterion, device, scaler_amp=None, train=False
        )

        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_r2": train_r2,
                "val_r2": val_r2,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if True:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_r2={train_r2:.4f} val_r2={val_r2:.4f}"
            )

        if wait >= patience:
            print(f"触发早停，停止在 epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val


def evaluate_model(model, loader, device):
    criterion = nn.MSELoss()
    _, rmse, r2, preds, gts, masks = run_epoch(
        model, loader, optimizer=None, criterion=criterion, device=device, scaler_amp=None, train=False
    )

    result = {
        "rmse": rmse,
        "r2": r2,
        "preds": preds,
        "gts": gts,
        "masks": masks,
    }
    return result


def build_fold_dataloaders(sample_ids, targets, spectral, train_idx, val_idx, real_image_dir, batch_size=8):
    """每折内拟合Scaler/PCA，避免验证集信息泄漏。"""
    train_spec = spectral[train_idx]
    scaler, pca = fit_scaler_pca(train_spec, pca_components=3)

    train_pc = transform_with_scaler_pca(spectral[train_idx], scaler, pca)
    val_pc = transform_with_scaler_pca(spectral[val_idx], scaler, pca)

    train_pseudo, min_vals, max_vals = pc_to_rgb_images(train_pc, image_size=(64, 64), noise_std=0.0)
    val_pseudo, _, _ = pc_to_rgb_images(val_pc, image_size=(64, 64), min_vals=min_vals, max_vals=max_vals, noise_std=0.0)

    train_tf, val_tf = make_transforms()

    train_ds = MultiModalDataset(
        sample_ids[train_idx],
        targets[train_idx],
        train_pseudo,
        real_image_dir,
        pseudo_transform=train_tf["pseudo"],
        real_transform=train_tf["real"],
    )
    val_ds = MultiModalDataset(
        sample_ids[val_idx],
        targets[val_idx],
        val_pseudo,
        real_image_dir,
        pseudo_transform=val_tf["pseudo"],
        real_transform=val_tf["real"],
    )

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)

    return train_loader, val_loader, scaler, pca, min_vals, max_vals


def cross_validate(sample_ids, targets, spectral, real_image_dir, model_dir, n_splits=5, epochs=80, batch_size=8):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = get_device()

    all_records = []
    fold_metrics = []
    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(sample_ids), start=1):
        print(f"\n===== 第 {fold} 折 / {n_splits} =====")
        train_loader, val_loader, _, _, _, _ = build_fold_dataloaders(
            sample_ids, targets, spectral, train_idx, val_idx, real_image_dir, batch_size
        )

        model = MultiModalSoilModel(use_pretrained=True).to(device)
        model, history, _ = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            patience=12,
            lr=1e-3,
            weight_decay=1e-4,
        )

        eval_result = evaluate_model(model, val_loader, device)
        rmse = eval_result["rmse"]
        r2 = eval_result["r2"]
        print(f"Fold {fold} 验证集: RMSE={rmse:.4f}, R2={r2:.4f}")

        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))

        history_df = pd.DataFrame(history)
        history_df.to_excel(os.path.join(fold_dir, "history.xlsx"), index=False)
        best_epochs.append(int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"]))

        fold_metrics.append({"fold": fold, "rmse": rmse, "r2": r2})

        val_ids = sample_ids[val_idx]
        val_actual = eval_result["gts"]
        val_pred = eval_result["preds"]
        val_mask = eval_result["masks"].astype(int)
        for sid, y_true, y_hat, has_real in zip(val_ids, val_actual, val_pred, val_mask):
            all_records.append(
                {
                    "fold": fold,
                    "sample_id": int(sid),
                    "actual": float(y_true),
                    "pred": float(y_hat),
                    "has_real_image": int(has_real),
                }
            )

    overall_df = pd.DataFrame(all_records)
    overall_rmse = np.sqrt(mean_squared_error(overall_df["actual"], overall_df["pred"]))
    overall_r2 = r2_score(overall_df["actual"], overall_df["pred"])

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.loc[len(metrics_df)] = {"fold": "overall", "rmse": overall_rmse, "r2": overall_r2}

    metrics_df.to_excel(os.path.join(model_dir, "cross_validation_results.xlsx"), index=False)
    overall_df.to_excel(os.path.join(model_dir, "cv_overall_predictions.xlsx"), index=False)

    plt.figure(figsize=(8, 7))
    colors = np.where(overall_df["has_real_image"].to_numpy() == 1, "red", "blue")
    plt.scatter(overall_df["actual"], overall_df["pred"], c=colors, alpha=0.7, s=45)
    vmin = min(overall_df["actual"].min(), overall_df["pred"].min())
    vmax = max(overall_df["actual"].max(), overall_df["pred"].max())
    plt.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5)
    plt.xlabel("Actual Organic Matter")
    plt.ylabel("Predicted Organic Matter")
    plt.title(f"5-Fold CV Overall\nR2={overall_r2:.4f}, RMSE={overall_rmse:.4f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "cv_overall_results.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return metrics_df, int(np.round(np.mean(best_epochs)))


def train_final_model(sample_ids, targets, spectral, real_image_dir, pca_dir, pseudo_dir, model_dir, epochs=50, batch_size=8):
    """用全量数据训练最终模型，并导出PCA和伪图像。"""
    device = get_device()

    scaler, pca = fit_scaler_pca(spectral, pca_components=3)
    all_pc = transform_with_scaler_pca(spectral, scaler, pca)
    all_pseudo, min_vals, max_vals = pc_to_rgb_images(all_pc, image_size=(64, 64), noise_std=0.0)

    pca_df = pd.DataFrame({"样本编号": sample_ids, "有机质含量": targets})
    for i in range(all_pc.shape[1]):
        pca_df[f"主成分_{i+1}"] = all_pc[:, i]
    pca_df.to_excel(os.path.join(pca_dir, "降维后的光谱数据.xlsx"), index=False)

    save_pseudo_images(sample_ids, all_pseudo, pseudo_dir)

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "rgb_min": min_vals.tolist(),
        "rgb_max": max_vals.tolist(),
    }
    with open(os.path.join(pca_dir, "pca_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    train_tf, _ = make_transforms()
    dataset = MultiModalDataset(
        sample_ids,
        targets,
        all_pseudo,
        real_image_dir,
        pseudo_transform=train_tf["pseudo"],
        real_transform=train_tf["real"],
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    model = MultiModalSoilModel(use_pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n开始全量训练，共 {epochs} 轮...")
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for pseudo, real, mask, target, _ in loader:
            pseudo = pseudo.to(device)
            real = real.to(device)
            mask = mask.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(pseudo, real, mask)
                loss = criterion(out, target)

            if use_amp:
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        if True:
            print(f"Epoch {epoch:03d}/{epochs} | train_loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_model_dir = os.path.join(model_dir, "final_model_all_data")
    os.makedirs(final_model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_model_dir, "best_model.pth"))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "best_train_loss": best_loss,
        },
        os.path.join(final_model_dir, "final_model_all_data.pth"),
    )

    print(f"全量训练完成，最佳训练损失: {best_loss:.4f}")
    print(f"最终模型保存目录: {final_model_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="土壤有机质多模态融合训练脚本")
    parser.add_argument("--excel-path", type=str, default="光谱数据.xlsx", help="光谱Excel路径")
    parser.add_argument("--sheet-name", type=str, default="0", help="Excel sheet 名称或索引")
    parser.add_argument("--real-image-dir", type=str, default="图像数据", help="真实图像目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-splits", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--cv-epochs", type=int, default=80, help="每折训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--final-epochs", type=int, default=0, help="全量训练轮数，0表示自动使用推荐值")
    parser.add_argument("--skip-cv", action="store_true", help="跳过交叉验证，直接全量训练")
    return parser.parse_args()


def _parse_sheet_name(value):
    # 允许传入字符串名称，也允许传入数字索引。
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def main(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    work_dir, pca_dir, pseudo_dir, model_dir = ensure_dirs()

    excel_path = args.excel_path
    sheet_name = _parse_sheet_name(args.sheet_name)
    real_image_dir = args.real_image_dir

    print("开始执行：降维 + 伪图像生成 + 多模态融合预测")
    print(f"工作目录: {os.path.abspath(work_dir)}")
    print(f"设备: {get_device()}")

    sample_ids, targets, spectral = load_spectral_data(excel_path=excel_path, sheet_name=sheet_name)
    print(f"有效样本数: {len(sample_ids)}")
    print(f"光谱维度: {spectral.shape[1]}")
    print(f"有机质范围: {targets.min():.3f} - {targets.max():.3f}")

    if not os.path.exists(real_image_dir):
        print(f"警告: 未找到真实图像目录 {real_image_dir}，模型将主要依赖伪图像分支。")

    recommended_epochs = 50
    if not args.skip_cv:
        print("\n步骤1: 执行5折交叉验证（每折内独立PCA，避免泄漏）")
        cv_metrics, recommended_epochs = cross_validate(
            sample_ids,
            targets,
            spectral,
            real_image_dir,
            model_dir,
            n_splits=args.n_splits,
            epochs=args.cv_epochs,
            batch_size=args.batch_size,
        )

        overall = cv_metrics.iloc[-1]
        print("\n交叉验证完成")
        print(f"总体R2: {overall['r2']:.4f}")
        print(f"总体RMSE: {overall['rmse']:.4f}")
        print(f"推荐全量训练轮数: {recommended_epochs}")
    else:
        print("\n步骤1: 已跳过交叉验证（--skip-cv）")

    print("\n步骤2: 全量训练并导出降维结果与伪图像")
    final_epochs = args.final_epochs if args.final_epochs > 0 else max(30, recommended_epochs)
    train_final_model(
        sample_ids,
        targets,
        spectral,
        real_image_dir,
        pca_dir,
        pseudo_dir,
        model_dir,
        epochs=final_epochs,
        batch_size=args.batch_size,
    )

    print("\n全部流程完成。")
    print(f"降维结果目录: {pca_dir}")
    print(f"模型结果目录: {model_dir}")


if __name__ == "__main__":
    main()
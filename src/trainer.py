"""Pipeline pelatihan, validasi, logging run, dan uji silang-domain untuk T3-A."""

import csv
import os
import time
from typing import Callable, Dict, List, Optional

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import make_torch_dataset
from src.metrics import aggregate_per_image_metrics, benchmark_efficiency, evaluate_single_image
from src.models import build_loss, build_segmentation_model, set_seed
from src.xai_and_errors import categorize_segmentation_error


def get_train_transforms(aug_mode: str = "geom") -> Optional[A.Compose]:
    """Bangun augmentasi untuk eksperimen/ablasi ('none', 'geom', atau 'color_geom')."""
    if aug_mode == "none":
        return None
    if aug_mode == "geom":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    if aug_mode == "color_geom":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.10, p=0.8),
        ])
    raise ValueError(f"Mode augmentasi tidak dikenal: {aug_mode}")


@torch.no_grad()
def evaluate_loader(model, loader: DataLoader, device: str = "cuda", capture_records: bool = False):
    """Evaluasi model pada satu DataLoader dan hitung metrik wajib per citra."""
    model.eval()
    per_image = []
    records = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for imgs, masks, img_ids in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        gts = masks.numpy()

        for idx in range(len(preds)):
            m = evaluate_single_image(preds[idx], gts[idx])
            per_image.append(m)
            if capture_records:
                rgb_vis = np.clip(imgs[idx].cpu().numpy().transpose(1, 2, 0) * std + mean, 0.0, 1.0)
                rgb_u8 = (rgb_vis * 255.0).astype(np.uint8)
                cat = categorize_segmentation_error(preds[idx], gts[idx], rgb_u8, m["dice_fg_mean"])
                records.append({
                    "id": img_ids[idx],
                    "img_tensor": imgs[idx].cpu(),
                    "image_rgb": rgb_u8,
                    "gt_mask": gts[idx],
                    "pred_mask": preds[idx],
                    "dice_fg_mean": m["dice_fg_mean"],
                    "error_category": cat,
                })

    summary = aggregate_per_image_metrics(per_image)
    return summary, records


def run_experiment(
    run_name: str,
    owner: str,
    domain_dirs: Dict[str, str],
    split_csvs: Dict[str, str],
    train_domain: str = "D1",
    target_domain: str = "D2",
    arch: str = "unet",
    encoder_name: str = "resnet34",
    freeze_encoder: bool = False,
    loss_name: str = "ce_dice",
    aug_mode: str = "geom",
    train_normalizer=None,
    eval_normalizer=None,
    input_size: int = 256,
    batch_size: int = 16,
    epochs: int = 40,
    lr: float = 3e-4,
    seed: int = 42,
    log_csv_path: str = "logs/experiment_runs.csv",
    save_weights_path: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, object]:
    """Jalankan satu eksperimen penuh: train -> val model selection -> evaluate in-domain & cross-domain."""
    t_start = time.perf_counter()
    set_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_ds = make_torch_dataset(
        domain_dirs[train_domain],
        split_csvs[train_domain],
        "train",
        input_size=input_size,
        normalizer=train_normalizer,
        transform=get_train_transforms(aug_mode),
    )
    val_ds = make_torch_dataset(
        domain_dirs[train_domain],
        split_csvs[train_domain],
        "val",
        input_size=input_size,
        normalizer=train_normalizer,
    )
    in_test_ds = make_torch_dataset(
        domain_dirs[train_domain],
        split_csvs[train_domain],
        "test",
        input_size=input_size,
        normalizer=train_normalizer,
    )
    cross_test_ds = make_torch_dataset(
        domain_dirs[target_domain],
        split_csvs[target_domain],
        "test",
        input_size=input_size,
        normalizer=eval_normalizer,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    in_test_loader = DataLoader(in_test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    cross_test_loader = DataLoader(cross_test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_segmentation_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        classes=3,
        freeze_encoder=freeze_encoder,
    ).to(device)
    criterion = build_loss(loss_name)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_dice = -1.0
    best_state = None
    history = {"train_loss": [], "val_dice_fg": []}

    status = "success"
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for imgs, masks, _ in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(imgs)
            scheduler.step()

            val_summary, _ = evaluate_loader(model, val_loader, device=device, capture_records=False)
            val_dice = val_summary["dice_fg_mean"]["mean"]
            history["train_loss"].append(running_loss / len(train_ds))
            history["val_dice_fg"].append(val_dice)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if save_weights_path:
            os.makedirs(os.path.dirname(save_weights_path), exist_ok=True)
            torch.save(best_state, save_weights_path)

        in_summary, in_records = evaluate_loader(model, in_test_loader, device=device, capture_records=True)
        cross_summary, cross_records = evaluate_loader(model, cross_test_loader, device=device, capture_records=True)
        eff = benchmark_efficiency(model, input_size=input_size, warmup=10, n_iters=50, device=device)
    except Exception as exc:
        status = f"failed: {exc}"
        in_summary = cross_summary = eff = {}
        in_records = cross_records = []

    elapsed_sec = round(time.perf_counter() - t_start, 2)
    row = {
        "run_name": run_name,
        "owner": owner,
        "train_domain": train_domain,
        "target_domain": target_domain,
        "arch": arch,
        "encoder": encoder_name,
        "freeze_encoder": freeze_encoder,
        "loss": loss_name,
        "aug_mode": aug_mode,
        "input_size": input_size,
        "seed": seed,
        "status": status,
        "val_dice_fg": round(best_val_dice, 4),
        "in_dice_nuc": round(in_summary.get("dice_nucleus", {}).get("mean", 0.0), 4),
        "in_dice_cyto": round(in_summary.get("dice_cytoplasm", {}).get("mean", 0.0), 4),
        "in_miou_fg": round(in_summary.get("miou_fg", {}).get("mean", 0.0), 4),
        "cross_dice_nuc": round(cross_summary.get("dice_nucleus", {}).get("mean", 0.0), 4),
        "cross_dice_cyto": round(cross_summary.get("dice_cytoplasm", {}).get("mean", 0.0), 4),
        "cross_miou_fg": round(cross_summary.get("miou_fg", {}).get("mean", 0.0), 4),
        "cross_hd95_nuc": round(cross_summary.get("hd95_nucleus", {}).get("mean", 0.0), 2),
        "cross_hd95_cyto": round(cross_summary.get("hd95_cytoplasm", {}).get("mean", 0.0), 2),
        "params_m": eff.get("total_params_m", 0.0),
        "latency_ms": eff.get("latency_mean_ms", 0.0),
        "gpu": eff.get("gpu_name", "CPU"),
        "duration_sec": elapsed_sec,
    }

    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    write_header = not os.path.exists(log_csv_path)
    with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return {
        "row": row,
        "in_summary": in_summary,
        "cross_summary": cross_summary,
        "in_records": in_records,
        "cross_records": cross_records,
        "history": history,
        "model": model,
    }

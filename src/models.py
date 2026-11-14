"""Definisi dua paradigma arsitektur segmentasi (U-Net vs DeepLabV3+), loss, dan seed locking."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Kunci RNG untuk reproduksibilitas penuh antar percobaan dan multi-seed."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_segmentation_model(
    arch: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    classes: int = 3,
    freeze_encoder: bool = False,
):
    """Bangun U-Net (Encoder-Decoder) atau DeepLabV3+ (ASPP) berbasis bobot pralatih ImageNet."""
    import segmentation_models_pytorch as smp

    arch_key = arch.lower()
    if arch_key == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
        )
    elif arch_key in ("deeplabv3plus", "deeplabv3+"):
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
        )
    else:
        raise ValueError(f"Arsitektur tidak dikenal: {arch}. Gunakan 'unet' atau 'deeplabv3plus'.")

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model


def build_loss(loss_name: str = "ce_dice"):
    """Bangun fungsi loss untuk ablasi loss: 'ce', 'dice', atau 'ce_dice'."""
    import torch.nn as nn
    import segmentation_models_pytorch as smp

    key = loss_name.lower()
    ce = nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    if key == "ce":
        return ce
    if key == "dice":
        return dice
    if key == "ce_dice":
        return lambda logits, target: 0.5 * ce(logits, target) + 0.5 * dice(logits, target)
    raise ValueError(f"Loss tidak dikenal: {loss_name}. Gunakan 'ce', 'dice', atau 'ce_dice'.")

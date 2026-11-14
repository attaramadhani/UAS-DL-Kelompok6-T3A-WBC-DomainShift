"""Audit dataset Zheng et al. (D1/D2), konversi masker 3-kelas, split tetap, dan cek kebocoran."""

import csv
import glob
import hashlib
import os
import shutil
import subprocess
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


DATASET_URLS = {
    "Dataset_1": "https://data.mendeley.com/public-files/datasets/w7cvnmn4c5/files/77a30a73-7409-4d61-b55a-aea667a11142/file_downloaded",
    "Dataset_2": "https://data.mendeley.com/public-files/datasets/w7cvnmn4c5/files/543b0d57-cd1b-4316-8fb5-69ad28b8e767/file_downloaded",
}


def download_and_extract_zheng(data_root: str) -> Dict[str, str]:
    """Unduh Dataset 1 & 2 resmi dari Mendeley Data (10.17632/w7cvnmn4c5.1) lalu ekstrak."""
    raw_dir = os.path.join(data_root, "raw")
    ext_dir = os.path.join(data_root, "extracted")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)

    for name, url in DATASET_URLS.items():
        rar_path = os.path.join(raw_dir, f"{name}.rar")
        target_sub = os.path.join(ext_dir, name.replace("_", " "))
        if not os.path.exists(rar_path):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(rar_path, "wb") as f:
                shutil.copyfileobj(resp, f)
        if not os.path.isdir(target_sub) or len(os.listdir(target_sub)) == 0:
            # Windows tar/bsdtar mendukung RAR; di Linux/Colab gunakan unrar atau 7z jika tersedia
            cmd = ["tar", "-xf", rar_path, "-C", ext_dir]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                if shutil.which("unrar"):
                    subprocess.run(["unrar", "x", "-y", rar_path, ext_dir], check=True)
                elif shutil.which("7z"):
                    subprocess.run(["7z", "x", f"-o{ext_dir}", "-y", rar_path], check=True)
                else:
                    raise RuntimeError(f"Gagal mengekstrak {rar_path}: {res.stderr}")

    return {
        "D1": os.path.join(ext_dir, "Dataset 1"),
        "D2": os.path.join(ext_dir, "Dataset 2"),
    }


def encode_mask_to_classes(mask_arr: np.ndarray) -> np.ndarray:
    """Petakan nilai masker grayscale [0, 128, 255] (+ artefak kompressi tepi) -> {0: bg, 1: sitoplasma, 2: nukleus}."""
    arr = np.asarray(mask_arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    out = np.zeros(arr.shape, dtype=np.int64)
    out[(arr >= 64) & (arr < 192)] = 1
    out[arr >= 192] = 2
    return out


def compute_dhash(image_path: str, hash_size: int = 8) -> str:
    """Hitung perceptual difference hash (dHash) 64-bit tanpa bergantung pustaka tambahan."""
    img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def compute_sha256(file_path: str) -> str:
    """Hitung hash SHA-256 berkas citra."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hamming_distance_hex(h1: str, h2: str) -> int:
    """Jarak Hamming antara dua string dHash heksadesimal."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def _union_find_groups(records: List[Tuple[str, str, str, str, str]], max_dhash_dist: int = 4) -> List[List[int]]:
    """Kelompokkan citra duplikat (SHA-256 sama atau dHash <= threshold) agar tidak terpisah saat split."""
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if records[i][4] == records[j][4] or hamming_distance_hex(records[i][3], records[j][3]) <= max_dhash_dist:
                union(i, j)

    groups_map: Dict[int, List[int]] = {}
    for idx in range(n):
        groups_map.setdefault(find(idx), []).append(idx)
    return list(groups_map.values())


def generate_fixed_splits(
    domain_dirs: Dict[str, str],
    splits_out_dir: str,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Dict[str, str]:
    """Buat split tetap anti-kebocoran (group-aware untuk duplikat/kemiripan) dengan seed=42."""
    os.makedirs(splits_out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    saved = {}

    for domain_key, ddir in domain_dirs.items():
        bmps = sorted(glob.glob(os.path.join(ddir, "*.bmp")))
        records = []
        for bmp_path in bmps:
            stem = os.path.splitext(os.path.basename(bmp_path))[0]
            png_path = os.path.join(ddir, f"{stem}.png")
            if not os.path.exists(png_path):
                raise FileNotFoundError(f"Masker pasangan tidak ditemukan untuk {bmp_path}")
            records.append((
                stem,
                os.path.basename(bmp_path),
                os.path.basename(png_path),
                compute_dhash(bmp_path),
                compute_sha256(bmp_path)[:16],
            ))

        groups = _union_find_groups(records, max_dhash_dist=4)
        perm = rng.permutation(len(groups))
        n = len(records)
        target_train = int(round(n * ratios[0]))
        target_val = int(round(n * ratios[1]))

        split_labels = [""] * n
        train_cnt, val_cnt = 0, 0
        # Prioritaskan grup duplikat (>1 citra) masuk train agar val dan test bersih 100% unik
        ordered_g = sorted(perm.tolist(), key=lambda g_idx: (0 if len(groups[g_idx]) > 1 else 1))
        for g_idx in ordered_g:
            idxs = groups[g_idx]
            if len(idxs) > 1 or train_cnt < target_train:
                sp = "train"
                train_cnt += len(idxs)
            elif val_cnt < target_val:
                sp = "val"
                val_cnt += len(idxs)
            else:
                sp = "test"
            for idx in idxs:
                split_labels[idx] = sp

        csv_path = os.path.join(splits_out_dir, f"zheng_{domain_key.lower()}_split.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["domain", "id", "image_file", "mask_file", "split", "dhash", "sha256"])
            for (stem, img_name, mask_name, dh, sh), sp in zip(records, split_labels):
                writer.writerow([domain_key, stem, img_name, mask_name, sp, dh, sh])
        saved[domain_key] = csv_path

    return saved


def check_leakage_across_splits(split_csv_paths: Dict[str, str], max_dhash_dist: int = 4) -> Dict[str, object]:
    """Periksa kebocoran antarsubset (train/val/test) dan antardomain berbasis SHA-256 & dHash."""
    rows: List[Dict[str, str]] = []
    for _, path in split_csv_paths.items():
        with open(path, encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))

    exact_within_train = []
    exact_cross_split = []
    near_cross_split = []
    near_cross_domain = []
    split_counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        split_counts.setdefault(r["domain"], {"train": 0, "val": 0, "test": 0})
        split_counts[r["domain"]][r["split"]] += 1

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r1, r2 = rows[i], rows[j]
            same_domain = r1["domain"] == r2["domain"]
            different_split = r1["split"] != r2["split"]
            pair_label = (f"{r1['domain']}/{r1['id']}({r1['split']})", f"{r2['domain']}/{r2['id']}({r2['split']})")

            if r1["sha256"] == r2["sha256"]:
                if not same_domain or different_split:
                    exact_cross_split.append(pair_label)
                else:
                    exact_within_train.append(pair_label)

            dist = hamming_distance_hex(r1["dhash"], r2["dhash"])
            if dist <= max_dhash_dist:
                pair_d = (*pair_label, dist)
                if not same_domain:
                    near_cross_domain.append(pair_d)
                elif different_split:
                    near_cross_split.append(pair_d)

    return {
        "total_images": len(rows),
        "split_counts": split_counts,
        "cross_split_leakage_exact": len(exact_cross_split),
        "cross_split_leakage_exact_pairs": exact_cross_split,
        "cross_split_leakage_dhash_le_4bits": len(near_cross_split),
        "cross_split_leakage_dhash_pairs": near_cross_split,
        "cross_domain_near_duplicates_le_4bits": len(near_cross_domain),
        "cross_domain_near_duplicate_pairs": near_cross_domain,
        "within_train_exact_duplicates_grouped": exact_within_train,
    }


try:
    from torch.utils.data import Dataset as _TorchDatasetBase
except ImportError:
    _TorchDatasetBase = object


class ZhengWBCDataset(_TorchDatasetBase):
    """Dataset segmentasi WBC Zheng et al. dengan in-memory cache untuk mempercepat multi-epoch Colab."""

    def __init__(
        self,
        domain_dir: str,
        split_csv: str,
        split_name: str,
        input_size: int = 256,
        normalizer=None,
        transform=None,
        cache_in_memory: bool = True,
    ):
        self.domain_dir = domain_dir
        self.input_size = input_size
        self.normalizer = normalizer
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.items = []
        with open(split_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if split_name == "all" or row["split"] == split_name:
                    self.items.append(row)

        self._cache = [self._load_pair(item) for item in self.items] if cache_in_memory else None

    def _load_pair(self, item: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, str]:
        img = Image.open(os.path.join(self.domain_dir, item["image_file"])).convert("RGB")
        mask = Image.open(os.path.join(self.domain_dir, item["mask_file"])).convert("L")

        img = img.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.Resampling.NEAREST)

        img_np = np.asarray(img, dtype=np.uint8)
        mask_np = encode_mask_to_classes(np.asarray(mask)).astype(np.uint8)
        if self.normalizer is not None:
            img_np = self.normalizer.transform(img_np)
        return img_np, mask_np, item["id"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch

        if self._cache is not None:
            img_np, mask_np, item_id = self._cache[idx]
            img_np = img_np.copy()
            mask_np = mask_np.copy()
        else:
            img_np, mask_np, item_id = self._load_pair(self.items[idx])

        if self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np, mask_np = augmented["image"], augmented["mask"]

        img_f = (img_np.astype(np.float32) / 255.0 - self.mean) / self.std
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_np.astype(np.int64)).long()
        return img_t, mask_t, item_id


def make_torch_dataset(
    domain_dir: str,
    split_csv: str,
    split_name: str,
    input_size: int = 256,
    normalizer=None,
    transform=None,
    cache_in_memory: bool = True,
):
    """Bangun PyTorch Dataset untuk pelatihan/evaluasi di Colab."""
    return ZhengWBCDataset(
        domain_dir=domain_dir,
        split_csv=split_csv,
        split_name=split_name,
        input_size=input_size,
        normalizer=normalizer,
        transform=transform,
        cache_in_memory=cache_in_memory,
    )

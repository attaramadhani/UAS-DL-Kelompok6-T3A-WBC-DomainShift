"""Normalisasi pewarnaan (Macenko & Reinhard) untuk mitigasi domain shift citra WBC."""

from typing import Tuple
import numpy as np
from PIL import Image


def _estimate_background_rgb(img: np.ndarray, bg_percentile: float = 95.0) -> np.ndarray:
    """Estimasi titik putih illuminant latar per kanal RGB untuk Beer-Lambert adaptif domain."""
    return np.clip(np.percentile(img.astype(np.float32), bg_percentile, axis=(0, 1)), 40.0, 255.0)


def _rgb_to_od(img: np.ndarray, io_intensity: float | np.ndarray = 255.0) -> np.ndarray:
    """Konversi citra RGB uint8 menjadi Optical Density (OD) terhadap illuminant referensi."""
    arr = np.maximum(img.astype(np.float32), 1.0)
    io = np.asarray(io_intensity, dtype=np.float32)
    if io.ndim == 1:
        io = io[None, None, :]
    return np.maximum(-np.log(arr / io), 0.0)


def _extract_macenko_stain_matrix(
    img: np.ndarray,
    io_intensity: float | np.ndarray = 255.0,
    beta: float = 0.12,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ekstraksi matriks vektor pewarna (3x2) dan konsentrasi maksimal persentil ke-99."""
    od = _rgb_to_od(img, io_intensity).reshape((-1, 3))
    # Pilih piksel ber-OD cukup tinggi (nukleus/sitoplasma/RBC) di atas latar yang sudah dinormalisasi
    od_hat = od[np.linalg.norm(od, axis=1) >= beta]

    if len(od_hat) < 16:
        he_default = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]], dtype=np.float32)
        he_default /= np.linalg.norm(he_default, axis=0, keepdims=True)
        conc = np.maximum(np.linalg.lstsq(he_default, od.T, rcond=None)[0], 0.0)
        max_c = np.maximum(np.percentile(conc, 99, axis=1), 1e-2)
        return he_default, max_c.astype(np.float32)

    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    basis = eigvecs[:, [1, 2]]
    proj = od_hat @ basis
    phi = np.arctan2(proj[:, 1], proj[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100.0 - alpha)

    v1 = basis @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = basis @ np.array([np.cos(max_phi), np.sin(max_phi)])

    # Pastikan kedua vektor mengarah ke kuadran positif OD (fisik hukum Beer-Lambert)
    if v1.sum() < 0:
        v1 = -v1
    if v2.sum() < 0:
        v2 = -v2
    v1 = np.maximum(v1, 1e-4)
    v2 = np.maximum(v2, 1e-4)

    # Urutkan agar komponen pertama dominan merah/hematoksilin dibanding komponen kedua
    if v1[0] > v2[0]:
        he = np.stack([v1, v2], axis=1)
    else:
        he = np.stack([v2, v1], axis=1)

    he = (he / np.linalg.norm(he, axis=0, keepdims=True)).astype(np.float32)
    conc_fg = np.maximum(np.linalg.lstsq(he, od_hat.T, rcond=None)[0], 0.0)
    max_c = np.maximum(np.percentile(conc_fg, 99, axis=1), 1e-2).astype(np.float32)
    return he, max_c


class MacenkoNormalizer:
    """Normalisasi pewarnaan Macenko white-balanced terhadap satu citra referensi."""

    def __init__(self, bg_percentile: float = 95.0, beta: float = 0.12, alpha: float = 1.0):
        self.bg_percentile = bg_percentile
        self.beta = beta
        self.alpha = alpha
        self.target_bg: np.ndarray | None = None
        self.target_he: np.ndarray | None = None
        self.target_max_c: np.ndarray | None = None

    def fit(self, target_rgb: np.ndarray) -> "MacenkoNormalizer":
        """Pelajari illuminant latar, vektor pewarna, dan konsentrasi referensi dari citra target."""
        self.target_bg = _estimate_background_rgb(target_rgb, self.bg_percentile)
        self.target_he, self.target_max_c = _extract_macenko_stain_matrix(
            target_rgb, self.target_bg, self.beta, self.alpha
        )
        return self

    def transform(self, source_rgb: np.ndarray) -> np.ndarray:
        """Transformasi distribusi pewarna dan illuminant latar citra sumber agar selaras dengan referensi."""
        if self.target_he is None or self.target_max_c is None or self.target_bg is None:
            raise RuntimeError("MacenkoNormalizer.fit() harus dipanggil sebelum transform().")

        h, w, _ = source_rgb.shape
        src_bg = _estimate_background_rgb(source_rgb, self.bg_percentile)
        od = _rgb_to_od(source_rgb, src_bg).reshape((-1, 3))
        src_he, src_max_c = _extract_macenko_stain_matrix(
            source_rgb, src_bg, self.beta, self.alpha
        )
        # Konsentrasi fisik tidak negatif (Beer-Lambert) & dibatasi agar tidak meledak di inti pekat
        conc = np.maximum(np.linalg.lstsq(src_he, od.T, rcond=None)[0], 0.0)
        scale = np.clip(self.target_max_c / src_max_c, 0.25, 4.0)
        conc = conc * scale[:, None]

        norm_od = np.maximum(self.target_he @ conc, 0.0)
        norm_rgb = self.target_bg[:, None] * np.exp(-norm_od)
        return np.clip(norm_rgb.T.reshape((h, w, 3)), 0, 255).astype(np.uint8)


class ReinhardNormalizer:
    """Transfer statistik warna ruang l-alpha-beta (Reinhard et al., 2001) murni NumPy."""

    _RGB_TO_LMS = np.array(
        [[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]],
        dtype=np.float32,
    )
    _LMS_TO_RGB = np.array(
        [[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]],
        dtype=np.float32,
    )
    _LMS_TO_LAB = np.array(
        [[1.0 / np.sqrt(3.0), 0.0, 0.0], [0.0, 1.0 / np.sqrt(6.0), 0.0], [0.0, 0.0, 1.0 / np.sqrt(2.0)]],
        dtype=np.float32,
    ) @ np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, -2.0], [1.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    _LAB_TO_LMS = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -2.0, 0.0]],
        dtype=np.float32,
    ) @ np.array(
        [[1.0 / np.sqrt(3.0), 0.0, 0.0], [0.0, 1.0 / np.sqrt(6.0), 0.0], [0.0, 0.0, 1.0 / np.sqrt(2.0)]],
        dtype=np.float32,
    )

    def __init__(self):
        self.target_mean: np.ndarray | None = None
        self.target_std: np.ndarray | None = None

    @classmethod
    def _rgb_to_lab(cls, rgb: np.ndarray) -> np.ndarray:
        """Konversi RGB uint8 ke ruang log-LMS l-alpha-beta."""
        arr = np.maximum(rgb.astype(np.float32) / 255.0, 1.0 / 255.0)
        lms = np.maximum(arr @ cls._RGB_TO_LMS.T, 1e-6)
        return np.log10(lms) @ cls._LMS_TO_LAB.T

    @classmethod
    def _lab_to_rgb(cls, lab: np.ndarray) -> np.ndarray:
        """Konversi l-alpha-beta kembali ke RGB uint8."""
        log_lms = lab @ cls._LAB_TO_LMS.T
        lms = np.power(10.0, np.clip(log_lms, -6.0, 2.0))
        rgb = lms @ cls._LMS_TO_RGB.T
        return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    def fit(self, target_rgb: np.ndarray) -> "ReinhardNormalizer":
        """Hitung rerata dan standar deviasi kanal l-alpha-beta citra referensi."""
        lab = self._rgb_to_lab(target_rgb)
        self.target_mean = lab.mean(axis=(0, 1))
        self.target_std = np.maximum(lab.std(axis=(0, 1)), 1e-5)
        return self

    def transform(self, source_rgb: np.ndarray) -> np.ndarray:
        """Cocokkan rerata dan standar deviasi l-alpha-beta citra sumber ke referensi."""
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError("ReinhardNormalizer.fit() harus dipanggil sebelum transform().")
        lab = self._rgb_to_lab(source_rgb)
        src_mean = lab.mean(axis=(0, 1))
        src_std = np.maximum(lab.std(axis=(0, 1)), 1e-5)
        norm_lab = ((lab - src_mean) / src_std) * self.target_std + self.target_mean
        return self._lab_to_rgb(norm_lab)


def select_representative_reference(image_paths: list[str]) -> np.ndarray:
    """Pilih satu citra dari set latih dengan kontras median sebagai referensi normalisasi."""
    stds = []
    for p in image_paths:
        arr = np.asarray(Image.open(p).convert("RGB"))
        stds.append((float(arr.std()), p))
    stds.sort(key=lambda x: x[0])
    median_path = stds[len(stds) // 2][1]
    return np.asarray(Image.open(median_path).convert("RGB"))

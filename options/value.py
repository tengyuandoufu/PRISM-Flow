from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Tuple, Union, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from scipy import ndimage as ndi

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a numpy/torch tensor to a numpy array and squeeze redundant single-channel dimensions if present."""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim >= 3 and x.shape[-3] == 1 and x.ndim in (3, 4):
        x = np.squeeze(x, axis=-3)
    return x


def _binarize(x: np.ndarray, threshold: Optional[float]) -> np.ndarray:
    """Convert input to a boolean mask.
    - If threshold is None, assume input is already {0,1} or bool (or treat non-zero as True).
    - Otherwise apply thresholding: x > threshold.
    """
    if x.dtype == np.bool_:
        return x
    if threshold is None:
        return x.astype(bool)
    return x > threshold


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    """Extract the surface voxels of a binary mask using erosion and difference."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Compute point-wise Euclidean distances from surface of A to surface of B, returning a 1D array."""
    a, b = a.astype(bool), b.astype(bool)
    sa, sb = _binary_surface(a), _binary_surface(b)

    if not sa.any():
        return np.array([], dtype=np.float64)

    if not b.any():
        dt = ndi.distance_transform_edt(~b, sampling=spacing)
        return dt[sa]

    dt = ndi.distance_transform_edt(~sb, sampling=spacing)
    return dt[sa]


def _gaussian_filter_nd(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Isotropic Gaussian filtering over N dimensions."""
    if x.size == 0:
        return x
    return ndi.gaussian_filter(x, sigma=sigma, mode="nearest")


def dice_coefficient(
    y_pred: ArrayLike,
    y_true: ArrayLike,
    threshold: Optional[float] = None,
    epsilon: float = 1e-7,
) -> float:
    """Dice Similarity Coefficient (DSC) in [0,1], higher is better."""
    yp = _binarize(_to_numpy(y_pred), threshold)
    yt = _binarize(_to_numpy(y_true), threshold)

    inter = np.logical_and(yp, yt).sum(dtype=np.float64)
    size_sum = yp.sum(dtype=np.float64) + yt.sum(dtype=np.float64)
    return float((2.0 * inter + epsilon) / (size_sum + epsilon))


def hd95(
    y_pred: ArrayLike,
    y_true: ArrayLike,
    spacing: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
) -> float:
    """95% Hausdorff distance (95th percentile of symmetric surface distance). Lower is better."""
    yp = _binarize(_to_numpy(y_pred), threshold)
    yt = _binarize(_to_numpy(y_true), threshold)

    if spacing is None:
        spacing = tuple([1.0] * yt.ndim)
    spacing = tuple(spacing)
    assert len(spacing) == yt.ndim, "spacing length must match mask dimensions"

    if not yt.any() and not yp.any():
        return 0.0
    if yt.any() ^ yp.any():
        return float("inf")

    d1 = _surface_distances(yp, yt, spacing)
    d2 = _surface_distances(yt, yp, spacing)

    if d1.size + d2.size == 0:
        return 0.0

    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95))


def ssim(
    y_pred: ArrayLike,
    y_true: ArrayLike,
    *,
    data_range: Optional[float] = None,
    gaussian_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    epsilon: float = 1e-12,
) -> float:
    """Structural Similarity Index (SSIM) computed over the entire 2D/3D image. Higher is better.
    Note: no binarization is applied; intended for intensity similarity (same modality or harmonized images).
    """
    x = _to_numpy(y_pred).astype(np.float64)
    y = _to_numpy(y_true).astype(np.float64)
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"

    if data_range is None:
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        data_range = max(x_max, y_max) - min(x_min, y_min)
        if data_range <= 0:
            data_range = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x = _gaussian_filter_nd(x, sigma=gaussian_sigma)
    mu_y = _gaussian_filter_nd(y, sigma=gaussian_sigma)

    x2 = x * x
    y2 = y * y
    xy = x * y

    sigma_x2 = _gaussian_filter_nd(x2, sigma=gaussian_sigma) - mu_x * mu_x
    sigma_y2 = _gaussian_filter_nd(y2, sigma=gaussian_sigma) - mu_y * mu_y
    sigma_xy = _gaussian_filter_nd(xy, sigma=gaussian_sigma) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2) + epsilon

    ssim_map = num / den
    return float(np.mean(ssim_map))


def ncc(
    y_pred: ArrayLike,
    y_true: ArrayLike,
    epsilon: float = 1e-12,
) -> float:
    """Global Normalized Cross-Correlation (NCC), typically in [-1, 1]. Higher is better."""
    x = _to_numpy(y_pred).astype(np.float64).ravel()
    y = _to_numpy(y_true).astype(np.float64).ravel()
    assert x.size == y.size, f"size mismatch: {x.size} vs {y.size}"

    x_mean = x.mean()
    y_mean = y.mean()
    x_c = x - x_mean
    y_c = y - y_mean

    num = np.dot(x_c, y_c)
    den = np.sqrt(np.dot(x_c, x_c) * np.dot(y_c, y_c)) + epsilon
    return float(num / den)


def evaluate_batch(
    preds: ArrayLike,
    gts: ArrayLike,
    *,
    threshold: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    ssim_data_range: Optional[float] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute DSC, HD95, SSIM, and NCC for a batch of samples.

    Supported shapes:
      - 2D: [B, H, W] or [B, 1, H, W]
      - 3D: [B, D, H, W] or [B, 1, D, H, W]
      - If there is no batch dimension, it is treated as a single sample.
    """
    P = _to_numpy(preds)
    G = _to_numpy(gts)

    if P.ndim == G.ndim and P.ndim in (2, 3):
        P = P[None, ...]
        G = G[None, ...]
    elif P.ndim == G.ndim and P.ndim in (4, 5) and P.shape[1] == 1:
        P = np.squeeze(P, axis=1)
        G = np.squeeze(G, axis=1)
    elif P.ndim == G.ndim and P.ndim in (4, 5):
        pass
    else:
        raise ValueError(f"Unsupported shapes: pred {P.shape}, gt {G.shape}")

    dsc_list, hd_list, ssim_list, ncc_list = [], [], [], []
    for i in range(P.shape[0]):
        dsc_val = dice_coefficient(P[i], G[i], threshold=threshold)
        hd_val  = hd95(P[i], G[i], spacing=spacing, threshold=threshold)
        ssim_val = ssim(P[i], G[i], data_range=ssim_data_range)
        ncc_val  = ncc(P[i], G[i])

        dsc_list.append(dsc_val)
        hd_list.append(hd_val)
        ssim_list.append(ssim_val)
        ncc_list.append(ncc_val)
    return dsc_list, hd_list, ssim_list, ncc_list


class MetricAverager:
    """Online accumulation and reporting of mean ± std for four metrics."""

    def __init__(self):
        self._vals: List[Tuple[float, float, float, float]] = []

    def update(
        self,
        dscs: Iterable[float],
        hd95s: Iterable[float],
        ssims: Iterable[float],
        nccs: Iterable[float],
    ) -> None:
        for d, h, s, n in zip(dscs, hd95s, ssims, nccs):
            self._vals.append((float(d), float(h), float(s), float(n)))

    @staticmethod
    def _mean_std(xs: List[float]) -> Tuple[float, float]:
        if len(xs) == 0:
            return 0.0, 0.0
        arr = np.asarray(xs, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    def summarize(self) -> dict:
        if not self._vals:
            return {
                "DSC_mean": 0.0, "DSC_std": 0.0,
                "HD95_mean": 0.0, "HD95_std": 0.0,
                "SSIM_mean": 0.0, "SSIM_std": 0.0,
                "NCC_mean": 0.0, "NCC_std": 0.0,
                "N": 0,
            }
        dscs = [v[0] for v in self._vals]
        hds  = [v[1] for v in self._vals]
        ss   = [v[2] for v in self._vals]
        nc   = [v[3] for v in self._vals]
        d_mean, d_std = self._mean_std(dscs)
        h_mean, h_std = self._mean_std(hds)
        s_mean, s_std = self._mean_std(ss)
        n_mean, n_std = self._mean_std(nc)
        return {
            "DSC_mean": d_mean, "DSC_std": d_std,
            "HD95_mean": h_mean, "HD95_std": h_std,
            "SSIM_mean": s_mean, "SSIM_std": s_std,
            "NCC_mean": n_mean, "NCC_std": n_std,
            "N": len(self._vals),
        }

    def pretty(self, unit: str = "") -> str:
        """Return a human-readable string of metric means and standard deviations."""
        s = self.summarize()
        unit_str = f" {unit}" if unit else ""
        return (f"DSC = {s['DSC_mean']:.3f} ({s['DSC_std']:.3f}); "
                f"HD95 = {s['HD95_mean']:.3f}{unit_str} ({s['HD95_std']:.3f}); "
                f"SSIM = {s['SSIM_mean']:.4f} ({s['SSIM_std']:.4f}); "
                f"NCC = {s['NCC_mean']:.4f} ({s['NCC_std']:.4f}) "
                f"[N={s['N']}]")

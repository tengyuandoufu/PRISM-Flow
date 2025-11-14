import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import cv2

INPUT_ROOT  = r"xxx"
OUTPUT_ROOT = r"xxx"

NUM_SLICES = 90
OUT_SIZE   = 256
MASK_PAT   = re.compile(r"^([A-Za-z0-9]+)_mask\.nii(\.gz)?$", re.IGNORECASE)


def _is_nii(name: str) -> bool:
    n = name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def find_mask_file_in_dir(d: Path) -> Optional[Path]:
    """
    In directory d, preferentially find files like *mask.nii.gz or *mask.nii.
    If there are multiple candidates, return the first one.
    """
    cands = sorted([p for p in d.iterdir() if p.is_file() and _is_nii(p.name)])
    mask_first = [p for p in cands if "mask" in p.name.lower()]
    return (mask_first[0] if mask_first else (cands[0] if cands else None))


def case_id_from_name(p: Path) -> str:
    """
    Extract CASE_ID from filenames like 1BA001_mask.nii(.gz); if it fails,
    fall back to the parent directory name.
    """
    m = MASK_PAT.match(p.name)
    if m:
        return m.group(1).upper()
    return p.parent.name


def as_canonical_zyx(img_path: Path) -> np.ndarray:
    """
    Read with nibabel, convert to RAS canonical orientation,
    and return a float32 array of shape (Z, Y, X).
    """
    nb = nib.load(str(img_path))
    nb_ras = nib.as_closest_canonical(nb)
    data = nb_ras.get_fdata(dtype=np.float32)
    zyx = np.transpose(data, (2, 1, 0))
    return zyx


def center_indices(Z: int, n: int) -> List[int]:
    n = min(max(n, 1), Z)
    start = max(0, (Z - n) // 2)
    return list(range(start, start + n))


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """
    Pad a 2D array to a square (black padding) without scaling, to preserve aspect ratio.
    """
    h, w = img.shape
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return np.pad(img, ((top, bottom), (left, right)), mode="constant")


def resize_mask(img2d: np.ndarray, out_size=OUT_SIZE) -> np.ndarray:
    """
    Pad to a square and then resize to out_size × out_size using nearest-neighbor
    interpolation to preserve label values.
    """
    sq = pad_to_square(img2d)
    return cv2.resize(sq, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def binarize_to_u8(slice2d: np.ndarray) -> np.ndarray:
    """
    Convert an arbitrary-valued mask slice to 0/255 uint8.
    """
    m = (slice2d > 0).astype(np.uint8) * 255
    return m


def export_mask_slices(input_root: str, output_root: str,
                       num_slices: int = NUM_SLICES, out_size: int = OUT_SIZE):
    in_root = Path(input_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    subdirs = [p for p in sorted(in_root.iterdir()) if p.is_dir()]
    print(f"[SCAN] Number of subfolders: {len(subdirs)}")

    for d in subdirs:
        mask_file = find_mask_file_in_dir(d)
        if mask_file is None:
            print(f"[SKIP] {d} no *.nii / *.nii.gz found")
            continue
        if "mask" not in mask_file.name.lower():
            precise = sorted(
                p for p in d.iterdir()
                if p.is_file() and "mask" in p.name.lower() and _is_nii(p.name)
            )
            if precise:
                mask_file = precise[0]

        case_id = case_id_from_name(mask_file)
        try:
            vol = as_canonical_zyx(mask_file)
        except Exception as e:
            print(f"[ERROR] Failed to read {mask_file}: {e}")
            continue

        Z = vol.shape[0]
        zs = center_indices(Z, num_slices)

        case_out_dir = out_root / case_id / "mask"
        case_out_dir.mkdir(parents=True, exist_ok=True)

        for i, z in enumerate(zs):
            m2d = binarize_to_u8(vol[z])
            m2d = resize_mask(m2d, out_size=out_size)
            save_path = case_out_dir / f"z{i:03d}.png"
            cv2.imwrite(str(save_path), m2d)

        print(f"[OK] {case_id}: {len(zs)} slices -> {case_out_dir}")

    print("[DONE] Export completed for all cases.")


if __name__ == "__main__":
    export_mask_slices(INPUT_ROOT, OUTPUT_ROOT)

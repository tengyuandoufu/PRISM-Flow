import time
from pathlib import Path
from typing import List, Optional, Tuple
import re

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2

# ======== Hardcoded paths and parameters ========
PARENT_DIR  = Path(r"XXX")     # Directory containing amos_XXXX.nii/ folders
OUTPUT_ROOT = Path(r"XXX")    # Output root: <case>/{CT,MR}/...
OUT_SIZE    = 720
NUM_SLICES  = 90
ROI_MARGIN  = 8
CT_WL, CT_WW = 40.0, 400.0  # CT windowing (soft tissue window)
# ============================================

def log(*a): print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)

# ---------- Basic tools ----------
def list_all_niis(root: Path) -> List[Path]:
    return sorted(list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz")))

def case_id_from_dirname(d: Path) -> str:
    name = d.name
    if name.lower().endswith(".nii"):
        name = name[:-4]
    m = re.search(r"(\d+)$", name)
    return m.group(1).zfill(4) if m else name

def center_indices(Z: int, n: int) -> List[int]:
    n = min(max(n,1), Z); s = max(0, (Z-n)//2); return list(range(s, s+n))

# ---------- Read as RAS (with error tolerance for bad orientations) ----------
def read_as_RAS_any(nii_path: Path) -> sitk.Image:
    try:
        img = sitk.ReadImage(str(nii_path))
        of = sitk.DICOMOrientImageFilter()
        of.SetDesiredCoordinateOrientation("RAS")
        return of.Execute(img)
    except Exception as e:
        log(f"[WARN] SimpleITK read failed, falling back to nibabel: {nii_path} -> {e}")
    nb = nib.load(str(nii_path))
    nb_ras = nib.as_closest_canonical(nb)
    data = nb_ras.get_fdata(dtype=np.float32)       # (X,Y,Z) or (X,Y,Z,T)
    if data.ndim > 3:
        data = data[...,0]                          # Take the first time/series
    zooms = nb_ras.header.get_zooms()[:3]
    arr_zyx = np.transpose(data, (2, 1, 0))         # (Z,Y,X)
    out = sitk.GetImageFromArray(arr_zyx)
    out.SetSpacing(tuple(float(z) for z in zooms) if len(zooms)>=3 else (1.0,1.0,1.0))
    out.SetOrigin((0.0, 0.0, 0.0))
    return out

def sitk_to_npy_zyx(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,H,W)

# ---------- Modality identification: keywords + robust histogram heuristic ----------
CT_KEYS = ("ct",)
MR_KEYS = ("mr", "t1", "t2", "flair", "dwi", "adc", "pd")

def keyword_hit(p: Path, keys: Tuple[str,...]) -> bool:
    name = p.name.lower()
    return any(k in name for k in keys)

def robust_percentiles_3d(npvol: np.ndarray) -> Tuple[float,float,float]:
    x = npvol.astype(np.float32)
    x = x[x!=0] if np.count_nonzero(npvol)==npvol.size else x
    if x.size == 0: x = npvol.astype(np.float32)
    return float(np.percentile(x,1)), float(np.percentile(x,50)), float(np.percentile(x,99))

def is_ct_by_hist(npvol: np.ndarray) -> bool:
    p1, _, p99 = robust_percentiles_3d(npvol)
    return (p1 < -200.0) and (p99 > 300.0)

def choose_best_for_modality(cands: List[Path], modality: str) -> Optional[Path]:
    if not cands: return None
    keys = CT_KEYS if modality=="CT" else MR_KEYS
    kw_pool = [p for p in cands if keyword_hit(p, keys)]
    pool = kw_pool if kw_pool else cands

    scored = []
    for p in pool:
        try:
            img = read_as_RAS_any(p)                       # Lightweight read (can be cached)
            vol = sitk_to_npy_zyx(img)
            if modality == "CT":
                score = 1.0 if is_ct_by_hist(vol) else 0.0
            else:
                score = 1.0 if not is_ct_by_hist(vol) else 0.0
            nvox = int(np.prod(vol.shape))
            scored.append((score, nvox, p))
        except Exception:
            scored.append((0.0, 0, p))
    scored.sort(reverse=True)
    best = scored[0][2] if scored else None
    return best

# ---------- Visualization enhancement/ROI ----------
def ct_window01(x: np.ndarray, wl=CT_WL, ww=CT_WW) -> np.ndarray:
    lo, hi = wl-ww/2.0, wl+ww/2.0
    return np.clip((x-lo)/max(1e-6, hi-lo), 0, 1)

def clahe_u8(u8: np.ndarray, clip=2.2, tile=8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile)).apply(u8)

def unsharp(u8: np.ndarray, sigma=0.8, amount=0.45) -> np.ndarray:
    blur = cv2.GaussianBlur(u8, (0,0), sigma)
    return cv2.addWeighted(u8, 1+amount, blur, -amount, 0)

def enhance_ct(x2d: np.ndarray) -> np.ndarray:
    u8 = (ct_window01(x2d)*255).astype(np.uint8)
    u8 = clahe_u8(u8, 2.0, 8)
    u8 = cv2.bilateralFilter(u8, d=0, sigmaColor=16, sigmaSpace=3)
    u8 = unsharp(u8, 0.8, 0.4)
    return u8

def enhance_mr(x2d: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(x2d, [1,99]); vmax = vmax if vmax>vmin else vmin+1
    y   = np.clip((x2d - vmin)/(vmax - vmin), 0, 1)
    u8  = (y*255).round().astype(np.uint8)
    u8  = clahe_u8(u8, 2.2, 8)
    u8  = cv2.bilateralFilter(u8, d=0, sigmaColor=18, sigmaSpace=3)
    u8  = unsharp(u8, 0.8, 0.45)
    return u8

def body_mask_ct(vol: np.ndarray) -> np.ndarray:
    Z,H,W = vol.shape
    m = np.zeros((Z,H,W), np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    for z in range(Z):
        u8 = (ct_window01(vol[z])*255).astype(np.uint8)
        _, th = cv2.threshold(u8, 5, 255, cv2.THRESH_BINARY)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        mm = np.zeros_like(th); cv2.drawContours(mm, [max(cnts,key=cv2.contourArea)], -1, 255, -1)
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, ker, iterations=2)
        m[z] = (mm>0).astype(np.uint8)
    return m

def body_mask_mr(vol: np.ndarray) -> np.ndarray:
    Z,H,W = vol.shape
    m = np.zeros((Z,H,W), np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    for z in range(Z):
        x = vol[z]
        vmin, vmax = np.percentile(x, [1,99]); vmax = vmax if vmax>vmin else vmin+1
        u01 = np.clip((x - vmin)/(vmax - vmin), 0, 1)
        u8  = (u01*255).astype(np.uint8)
        _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        mm = np.zeros_like(th); cv2.drawContours(mm, [max(cnts,key=cv2.contourArea)], -1, 255, -1)
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, ker, iterations=2)
        m[z] = (mm>0).astype(np.uint8)
    return m

def bbox_from_3d(mask: np.ndarray, margin=ROI_MARGIN):
    if mask is None or mask.max()==0: return None
    proj = mask.max(axis=0).astype(np.uint8)
    ys, xs = np.where(proj > 0)
    if ys.size == 0: return None
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    y0 = max(0, y0-margin); x0 = max(0, x0-margin)
    y1 = min(mask.shape[1], y1+margin); x1 = min(mask.shape[2], x1+margin)
    return (y0, y1, x0, x1)

def crop_resize(img2d: np.ndarray, bbox, size=OUT_SIZE, is_mask=False) -> np.ndarray:
    if bbox is None:
        h, w = img2d.shape; side = max(h, w)
        pt,pb = (side-h)//2, side-h-(side-h)//2
        pl,pr = (side-w)//2, side-w-(side-w)//2
        pad = np.pad(img2d, ((pt,pb),(pl,pr)), mode="constant")
        return cv2.resize(pad, (size,size),
                          interpolation=(cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR))
    y0,y1,x0,x1 = bbox
    roi = img2d[y0:y1, x0:x1]
    return cv2.resize(roi, (size,size),
                      interpolation=(cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR))

# ---------- Save NIfTI (only write 3 zoom values) ----------
def save_nifti_3d_ras(vol_zyx: np.ndarray, ref_img: sitk.Image, out_path: Path):
    data_xyz = np.transpose(vol_zyx, (2,1,0))   # (X,Y,Z)
    nii = nib.Nifti1Image(data_xyz, np.eye(4, dtype=np.float32))
    spacing = ref_img.GetSpacing()
    if len(spacing) >= 3:
        sx, sy, sz = spacing[:3]
    elif len(spacing) == 2:
        sx, sy = spacing; sz = 1.0
    else:
        sx = sy = sz = 1.0
    nii.header.set_zooms((float(sx), float(sy), float(sz)))
    nib.save(nii, str(out_path))

# ---------- Process a single folder: find CT/MR ----------
def process_one_folder(folder: Path):
    cands = list_all_niis(folder)
    if not cands:
        log(f"[SKIP] No .nii/.nii.gz found in {folder.name}"); return

    # Select CT / MR
    ct_src = choose_best_for_modality(cands, "CT")
    mr_src = choose_best_for_modality(cands, "MR")
    case_id = case_id_from_dirname(folder)

    if ct_src:
        ct_img = read_as_RAS_any(ct_src)
        ct_np  = sitk_to_npy_zyx(ct_img)
        out_ct = (OUTPUT_ROOT / case_id / "CT"); out_ct.mkdir(parents=True, exist_ok=True)
        save_nifti_3d_ras(ct_np, ct_img, out_ct / "ct.nii.gz")
        Z = ct_np.shape[0]; zs = center_indices(Z, NUM_SLICES)
        bbox = bbox_from_3d(body_mask_ct(ct_np), ROI_MARGIN)
        for z in zs:
            u8 = enhance_ct(crop_resize(ct_np[z], bbox, OUT_SIZE))
            cv2.imwrite(str(out_ct / f"z{z:03d}.png"), u8)
        log(f"[OK][CT] {folder.name} -> {out_ct}  | Selected file: {ct_src.name}")
    else:
        log(f"[INFO] {folder.name} No suitable CT found (maybe no CT in this directory)")

    if mr_src:
        mr_img = read_as_RAS_any(mr_src)
        mr_np  = sitk_to_npy_zyx(mr_img)
        out_mr = (OUTPUT_ROOT / case_id / "MR"); out_mr.mkdir(parents=True, exist_ok=True)
        save_nifti_3d_ras(mr_np, mr_img, out_mr / "mr.nii.gz")
        Z = mr_np.shape[0]; zs = center_indices(Z, NUM_SLICES)
        bbox = bbox_from_3d(body_mask_mr(mr_np), ROI_MARGIN)
        for z in zs:
            u8 = enhance_mr(crop_resize(mr_np[z], bbox, OUT_SIZE))
            cv2.imwrite(str(out_mr / f"z{z:03d}.png"), u8)
        log(f"[OK][MR] {folder.name} -> {out_mr}  | Selected file: {mr_src.name}")
    else:
        log(f"[INFO] {folder.name} No suitable MR found (maybe no MR in this directory)")

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    folders = [d for d in PARENT_DIR.iterdir() if d.is_dir() and d.name.lower().endswith(".nii")]
    log(f"[Scanning] Folders to process: {len(folders)}")
    for f in sorted(folders):
        try:
            process_one_folder(f)
        except Exception as e:
            log(f"[ERROR] {f.name}: {e}")
    log("[DONE] All tasks completed.")

if __name__ == "__main__":
    main()

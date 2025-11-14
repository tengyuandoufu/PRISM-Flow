import os
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from util.visualizer import save_images as save_images_to_web
from voxelmorph.layers import SpatialTransformer
from options import value

# If the deformation direction of the grid is opposite to the training, change it to -1
FLOW_SIGN = +1


# -------------------- Utility: DP Compatibility & Feature Capture (Option B) --------------------
def _unwrap_dp(m):
    return m.module if hasattr(m, "module") else m

@torch.no_grad()
def capture_multiscale_feats_from_netG(netG, x, target_sizes):
    """Capture multi-scale features from netG's intermediate layers; fallback to custom pyramid."""
    net = _unwrap_dp(netG).eval()
    device = next(net.parameters()).device
    feats, handles = {}, []

    def hook_fn(_m, _inp, out):
        if not torch.is_tensor(out) or out.dim() != 4:
            return
        _, _, h, w = out.shape
        if 8 <= h <= x.shape[-2] and 8 <= w <= x.shape[-1]:
            feats[(h, w)] = out

    interesting = ("down", "enc", "encoder", "stem", "block", "stage", "layer")
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d) and any(k in name.lower() for k in interesting):
            handles.append(m.register_forward_hook(hook_fn))

    _ = net(x.to(device))
    for h in handles:
        h.remove()

    def _fallback(size):
        H, W = size
        return F.adaptive_avg_pool2d(x, (H, W))

    outs = []
    for ts in target_sizes:
        if ts in feats:
            outs.append(feats[ts])
        else:
            cand, best = None, 10**9
            for (h, w), f in feats.items():
                d = abs(h - ts[0]) + abs(w - ts[1])
                if d < best:
                    best, cand = d, f
            outs.append(cand if cand is not None else _fallback(ts))
    return outs[0], outs[1]


# -------------------- Label Path Parsing --------------------
def resolve_label_root(dataroot: str, phase: str) -> Path | None:
    cands = [
        Path(dataroot) / f"{phase}A_label",
        Path(dataroot) / "trainA_label",
        Path(dataroot) / "testA_label",
    ]
    for p in cands:
        if p.exists():
            print(f"[INFO] Using label root directory: {p}")
            return p
    print("[INFO] Label root directory not found (phaseA_label/trainA_label/testA_label), skipping label warp.")
    return None

def find_label_for_slice(label_root: Path, case: str, zname: str) -> Path | None:
    case_dir = label_root / case
    if not case_dir.exists():
        return None
    p1 = case_dir / zname
    if p1.exists():
        return p1
    p2 = case_dir / f"{case}_{zname}"
    if p2.exists():
        return p2
    stem = Path(zname).stem.lower()
    for q in sorted(case_dir.iterdir()):
        if q.is_file() and stem == q.stem.lower().split("_")[-1]:
            return q
    return None


# -------------------- Visualization and Web Utilities --------------------
def to_m1_1(x01: torch.Tensor) -> torch.Tensor:
    """0~1 -> [-1,1] (For web save to avoid grayish tone)"""
    return (x01 * 2.0 - 1.0).clamp(-1, 1)

def ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in NCHW format. If CHW/HW, automatically add batch and/or channel dimensions."""
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unexpected tensor dim {x.dim()}, expect 2/3/4.")

def safe_visuals(model):
    keys_try = ["real_A", "real_B", "fake_B", "registered", "regA", "warped",
                "dvf", "flow", "y_flow"]
    vis = {}
    for k in keys_try:
        if hasattr(model, k):
            t = getattr(model, k)
            if torch.is_tensor(t):
                vis[k] = t
    if "dvf" not in vis and "flow" in vis:
        vis["dvf"] = vis["flow"]
    if "regA" not in vis and "registered" in vis:
        vis["regA"] = vis["registered"]
    return vis

def test(opt):
    # Fix configurations
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # Web directory
    web_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}")
    print("Creating web directory", web_dir)
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")

    # Output directory
    results_root = Path(opt.checkpoints_dir) / f"Results_{opt.name}"
    results_root.mkdir(parents=True, exist_ok=True)

    # Label root
    label_root = resolve_label_root(opt.dataroot, opt.phase)
    phaseA = f"{opt.phase}A"

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()

        # ---- Parse case & slice ----
        a_path = Path(data["A_paths"][0])
        parts = a_path.parts
        idx = parts.index(phaseA) if phaseA in parts else max(j for j, p in enumerate(parts) if phaseA in p)
        case  = parts[idx + 1]
        zname = a_path.name
        stem  = Path(zname).stem

        # ---- Get visual tensors ----
        device  = model.real_A.device
        H, W    = model.real_A.shape[-2], model.real_A.shape[-1]
        scale01 = lambda x: (x / 2.0 + 0.5).clamp(0, 1)

        vis = safe_visuals(model)
        moving = scale01(vis.get("real_A", model.real_A))
        fixed  = scale01(vis.get("real_B", model.real_B))
        TransA = scale01(vis.get("fake_B", model.real_A))
        ReconA = scale01(vis.get("registered", vis.get("regA", model.real_A)))
        warped = scale01(vis.get("regA", ReconA))
        flow   = vis.get("dvf", None)

        # ---- If no dvf, calculate it using option B ----
        if flow is None:
            target_sizes = [(H // 4, W // 4), (H // 8, W // 8)]
            enc, enc3 = capture_multiscale_feats_from_netG(model.netG, model.real_B, target_sizes)
            enc, enc3 = enc.to(device), enc3.to(device)
            try:
                y_pred2 = model.netR(model.real_A, model.real_B, registration=True, enc=enc, enc3=enc3)
            except TypeError:
                y_pred2 = model.netR(model.real_A, model.real_B, registration=True)
            ReconA  = scale01(y_pred2[0])
            flow    = y_pred2[1]
            warped  = ReconA

        pred_seg = None
        stn_nearest = SpatialTransformer([H, W], mode="nearest").to(device)
        if label_root is not None:
            lab_path = find_label_for_slice(label_root, case, zname)
            if lab_path is not None:
                lab_img    = Image.open(lab_path).convert("L").resize((W, H), resample=Image.NEAREST)
                lab_tensor = ToTensor()(lab_img).unsqueeze(0).to(device)  # [1,1,H,W]
                pred_seg   = stn_nearest(lab_tensor, flow)                # [1,1,H,W]
        if pred_seg is not None:
            dsc_list, hd_list = value.evaluate_batch(
                preds=pred_seg,
                gts=lab_tensor,
                threshold=0.5,
                spacing=(1.0, 1.0)  # Replace with NIfTI original spacing if available
            )
            meter.update(dsc_list, hd_list)
            print(f"[CASE {case} {stem}] DSC={dsc_list[0]:.3f}, HD95={hd_list[0]:.2f}")

        # Visualization
        grid_path = a_path.parent / "XXX/deform256.jpg"   # Same level as slice
        if not grid_path.exists():
            raise FileNotFoundError(f"Missing training grid: {grid_path}")
        grid_img    = Image.open(grid_path).convert("RGB").resize((W, H), resample=Image.NEAREST)
        grid_tensor = ToTensor()(grid_img).unsqueeze(0).to(device)     # [1,3,H,W], 0~1
        # Clean lines (optional)
        grid_tensor = (grid_tensor > 0.5).float()
        dvf_grid    = stn_nearest(grid_tensor, flow * FLOW_SIGN)       # [1,3,H,W], 0~1

        # ---- Save to result directory ----
        save_dir = results_root / case / stem
        save_dir.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(moving,    str(save_dir / "moving.png"))
        torchvision.utils.save_image(fixed,     str(save_dir / "fixed.png"))
        torchvision.utils.save_image(dvf_grid,  str(save_dir / "dvf_grid.jpg"))
        torchvision.utils.save_image(TransA,    str(save_dir / "TransA.png"))
        torchvision.utils.save_image(ReconA,    str(save_dir / "ReconA.png"))
        if pred_seg is not None:
            torchvision.utils.save_image(pred_seg.clamp(0,1).cpu(), str(save_dir / "seg_warped.png"))
        save_images_to_web(
            webpage,
            {
                "real_A":  to_m1_1(ensure_nchw(moving)),
                "fake_B":  to_m1_1(ensure_nchw(TransA)),
                "real_B":  to_m1_1(ensure_nchw(fixed)),
                "dvf":     to_m1_1(ensure_nchw(dvf_grid)),
                "warped":  to_m1_1(ensure_nchw(warped)),
                "ReconA":  to_m1_1(ensure_nchw(ReconA)),
            },
            [str(a_path)]
        )

    webpage.save()
    print("[SUMMARY]", meter.pretty(unit="voxels"))
    print(f"[DONE] Webpage: {web_dir}")
    print(f"[DONE] Per-case images: {results_root}")

if __name__ == "__main__":
    opt = TestOptions().parse()
    meter = value.MetricAverager()
    test(opt)

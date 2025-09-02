#!/usr/bin/env python3
import argparse
import os
import json
import time
import subprocess
from glob import glob
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F

# Matplotlib only if overlays requested
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Utilities
# ----------------------------
def find_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_subprocess(cmd: List[str], cwd: Optional[str] = None) -> None:
    rc = subprocess.run(cmd, cwd=cwd)
    if rc.returncode != 0:
        raise RuntimeError(f"Command failed with code {rc.returncode}: {' '.join(cmd)}")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def pad_z(z: int) -> str:
    return f"{z:03d}"

def parse_case_id_from_dir(case_dir: str) -> str:
    return os.path.basename(os.path.normpath(case_dir))

def find_modality_paths(case_dir: str) -> Dict[str, Optional[str]]:
    def pick_one(patterns: List[str]) -> Optional[str]:
        for p in patterns:
            hits = sorted(glob(os.path.join(case_dir, p)))
            if len(hits) > 0:
                return hits[0]
        return None
    flair = pick_one(["*flair.nii.gz", "*flair.nii", "*_flair.nii.gz", "*_flair.nii"])
    t1ce  = pick_one(["*t1ce.nii.gz", "*t1ce.nii", "*_t1ce.nii.gz", "*_t1ce.nii"])
    t2    = pick_one(["*t2.nii.gz", "*t2.nii", "*_t2.nii.gz", "*_t2.nii"])
    t1    = pick_one(["*t1.nii.gz", "*t1.nii", "*_t1.nii.gz", "*_t1.nii"])
    seg   = pick_one(["*seg.nii.gz", "*seg.nii", "*_seg.nii.gz", "*_seg.nii"])
    return {"flair": flair, "t1ce": t1ce, "t2": t2, "t1": t1, "seg": seg}

def pick_base_image_path(mods: Dict[str, Optional[str]]) -> str:
    for key in ["flair", "t1ce", "t2", "t1"]:
        if mods.get(key) is not None:
            return mods[key]
    raise RuntimeError("No modality NIfTI found in case folder.")

def select_slices(strategy: str, seg_path: Optional[str], z_dim: int, min_pos: int, margin: int, neg_ratio: float) -> List[int]:
    if strategy.lower() == "all" or seg_path is None:
        return list(range(z_dim))
    # A2-like selection using seg
    seg = nib.load(seg_path).get_fdata()
    pos_z = [z for z in range(z_dim) if np.any(seg[:, :, z] > 0)]
    # Filter by min_pos (at least this many positive pixels)
    if min_pos > 1:
        pos_z = [z for z in pos_z if int((seg[:, :, z] > 0).sum()) >= min_pos]
    pos_z = sorted(set(pos_z))
    if len(pos_z) == 0:
        # fallback to all if no positive slices
        return list(range(z_dim))
    zmin, zmax = min(pos_z), max(pos_z)
    neg_slices = []
    # Include a symmetric margin of negatives around the positive band, similar to earlier usage (neg ≈ 2*margin)
    if margin > 0 and neg_ratio > 0:
        lower = [z for z in range(max(0, zmin - margin), zmin)]
        upper = [z for z in range(zmax + 1, min(z_dim, zmax + 1 + margin))]
        neg_slices = lower + upper
    selected = sorted(set(pos_z + neg_slices))
    return selected

def normalize_for_overlay(img2d: np.ndarray) -> np.ndarray:
    img = img2d.astype(np.float32)
    p2, p98 = np.percentile(img[np.isfinite(img)], [2, 98]) if np.isfinite(img).any() else (img.min(), img.max())
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img = np.zeros_like(img)
    return img

# ----------------------------
# B1/B2 ensuring
# ----------------------------
def ensure_b1_b2_for_slice(
    repo_root: str,
    case_dir: str,
    case_id: str,
    z: int,
    out_dir: str,
    n_segments: int,
    compactness: float,
    pos_thresh: float,
    skip_existing: bool = True,
) -> Tuple[str, str]:
    ztag = pad_z(z)
    b1_npz = os.path.join(out_dir, f"b1_{case_id}_z{ztag}.npz")
    b2_npz = os.path.join(out_dir, f"b2_{case_id}_z{ztag}.npz")

    # b1
    if not (skip_existing and os.path.isfile(b1_npz)):
        cmd_b1 = [
            os.path.join(repo_root, "gnn-env/bin/python3"),
            os.path.join(repo_root, "src/b1_superpixels.py"),
            "--case", case_dir,
            "--z", str(z),
            "--n_segments", str(n_segments),
            "--compactness", str(compactness),
            "--pos_thresh", str(pos_thresh),
            "--out_dir", out_dir,
        ]
        run_subprocess(cmd_b1, cwd=repo_root)
    # b2
    if not (skip_existing and os.path.isfile(b2_npz)):
        cmd_b2 = [
            os.path.join(repo_root, "gnn-env/bin/python3"),
            os.path.join(repo_root, "src/b2_graph_build.py"),
            "--b1_npz", b1_npz,
            "--out_dir", out_dir,
        ]
        run_subprocess(cmd_b2, cwd=repo_root)
    if not os.path.isfile(b2_npz):
        raise FileNotFoundError(f"Expected b2 npz not found: {b2_npz}")
    return b1_npz, b2_npz

# ----------------------------
# Model (same as C3)
# ----------------------------
def undirected_deg_plus_self(edges_und: torch.Tensor, num_nodes: int, device: str) -> torch.Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    if edges_und.numel() > 0:
        u = edges_und[0]; v = edges_und[1]
        one_u = torch.ones_like(u, dtype=torch.float32)
        deg.index_add_(0, u, one_u)
        deg.index_add_(0, v, one_u)
    deg = deg + 1.0
    return deg

def make_directed_with_self(edges_und: torch.Tensor, num_nodes: int, device: str) -> torch.Tensor:
    if edges_und.numel() == 0:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        u = edges_und[0]; v = edges_und[1]
        ei = torch.cat([torch.stack([u, v], dim=0), torch.stack([v, u], dim=0)], dim=1)
    self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    ei_self = torch.stack([self_loops, self_loops], dim=0)
    return torch.cat([ei, ei_self], dim=1)

class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
    def forward(self, x: torch.Tensor, edge_index_und: torch.Tensor) -> torch.Tensor:
        N = x.size(0); device = x.device
        deg = undirected_deg_plus_self(edge_index_und, N, device=device)
        ei = make_directed_with_self(edge_index_und, N, device=device)
        src, dst = ei[0], ei[1]
        norm = (deg[src] * deg[dst]).clamp_min(1e-12).rsqrt()
        h = self.lin(x)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * norm.unsqueeze(-1))
        return out

class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden)
        self.gc2 = GraphConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x: torch.Tensor, edge_index_und: torch.Tensor) -> torch.Tensor:
        x = self.gc1(x, edge_index_und)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.gc2(x, edge_index_und)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        logits = self.out(x).squeeze(-1)
        return logits

# ----------------------------
# Load graph and checkpoint
# ----------------------------
def load_b2_graph(b2_path: str, device: str) -> Dict[str, Any]:
    d = np.load(b2_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    edges = d["edges"]
    if edges.ndim == 2:
        if edges.shape[0] == 2:
            ei = edges.astype(np.int64)
        elif edges.shape[1] == 2:
            ei = edges.T.astype(np.int64)
        else:
            raise ValueError(f"Unexpected edges shape {edges.shape}")
    else:
        raise ValueError(f"Unexpected edges shape {edges.shape}")
    # Make undirected unique pairs
    ei_t = ei.T
    und_unique = np.unique(np.sort(ei_t, axis=1), axis=0)
    und = und_unique.T
    out = {
        "X": torch.as_tensor(X, dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(und, dtype=torch.long, device=device),
        "K": int(X.shape[0]),
        "F": int(X.shape[1]),
        "labels_img": d["labels"],  # 2D label map
        "z": int(np.array(d["z"]).item()) if "z" in d else None,
        "shape": np.array(d["shape"]).tolist() if "shape" in d else None,
        "modalities": d["modalities"] if "modalities" in d else None,
    }
    return out

def load_checkpoint(ckpt_path: str, in_dim_hint: Optional[int] = None, device: str = "cpu") -> Tuple[GCN, Dict[str, Any]]:
    obj = torch.load(ckpt_path, map_location=device)
    meta = obj.get("meta", {})
    in_dim = in_dim_hint if in_dim_hint is not None else meta.get("in_dim", None)
    if in_dim is None:
        raise RuntimeError("Cannot determine model input dimension from checkpoint; provide in_dim_hint.")
    hidden = int(meta.get("hidden", 64))
    dropout = float(meta.get("dropout", 0.2))
    model = GCN(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, meta

# ----------------------------
# Inference helpers
# ----------------------------
def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))

def reconstruct_masks(labels_img: np.ndarray, probs_spx: np.ndarray, thr: float) -> Tuple[np.ndarray, np.ndarray]:
    # Robust mapping in case labels are not 0..K-1
    labels = labels_img.astype(np.int64)
    H, W = labels.shape
    prob_mask = np.zeros((H, W), dtype=np.float32)
    uniq = np.unique(labels)
    for sid in uniq:
        if sid < 0:
            continue
        p = probs_spx[sid] if sid < len(probs_spx) else 0.0
        prob_mask[labels == sid] = p
    pred_mask = (prob_mask >= thr).astype(np.uint8)
    return prob_mask, pred_mask

def save_overlay_png(bg2d: np.ndarray, pred_mask: np.ndarray, out_png: str, title: str = ""):
    plt.figure(figsize=(6, 6))
    plt.imshow(normalize_for_overlay(bg2d), cmap="gray")
    pred_alpha = np.ma.masked_where(pred_mask == 0, pred_mask)
    plt.imshow(pred_alpha, cmap="autumn", alpha=0.35, vmin=0, vmax=1)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="C4: Infer tumor superpixel probabilities over a case and assemble 3D output.")
    parser.add_argument("--case_dir", type=str, required=True, help="Path to a BraTS case folder (containing modalities).")
    parser.add_argument("--model_ckpt", type=str, default="outputs/checkpoints/c3_gcn_best.pt", help="Checkpoint from C3.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--select_strategy", type=str, default="a2", choices=["a2", "all"], help="Slice selection strategy.")
    parser.add_argument("--min_pos", type=int, default=1, help="Min positive pixels in a slice (A2 strategy).")
    parser.add_argument("--margin", type=int, default=2, help="Negative slices margin per side (A2 strategy).")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="Kept for compatibility; if >0 and margin>0, include margin negatives.")
    parser.add_argument("--n_segments", type=int, default=600)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--pos_thresh", type=float, default=0.3, help="Superpixel positive threshold used by B1 (for stats; not needed for inference).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary masks.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip_existing", action="store_true", help="Skip b1/b2 if files already exist.")
    parser.add_argument("--save_overlays", action="store_true", help="If set, save per-slice PNG overlays.")
    args = parser.parse_args()

    repo_root = find_repo_root()
    case_dir = os.path.abspath(args.case_dir)
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"Case dir not found: {case_dir}")
    case_id = parse_case_id_from_dir(case_dir)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo_root, args.out_dir)
    ensure_dir(out_dir)
    slice_npz_dir = os.path.join(out_dir, "preds_c4")
    ensure_dir(slice_npz_dir)
    overlay_dir = os.path.join(out_dir, "overlays_c4")
    ensure_dir(overlay_dir)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[C4] Using device: {device}")

    # Modalities and base image
    mods = find_modality_paths(case_dir)
    base_img_path = pick_base_image_path(mods)
    base_img = nib.load(base_img_path)
    base_vol = base_img.get_fdata()
    H, W, Z = base_vol.shape
    seg_path = mods.get("seg", None)
    print(f"[C4] Case={case_id}  shape={H}x{W}x{Z}  seg={'yes' if seg_path else 'no'}")

    # Select slices
    z_list = select_slices(args.select_strategy, seg_path, Z, args.min_pos, args.margin, args.neg_ratio)
    print(f"[C4] Selected {len(z_list)} slice(s) with strategy={args.select_strategy}: z in [{min(z_list)}..{max(z_list)}]")

    # Ensure b1/b2 and collect graph paths
    b2_paths = []
    t0 = time.time()
    for i, z in enumerate(z_list, 1):
        try:
            _, b2_npz = ensure_b1_b2_for_slice(
                repo_root=repo_root,
                case_dir=case_dir,
                case_id=case_id,
                z=z,
                out_dir=out_dir,
                n_segments=args.n_segments,
                compactness=args.compactness,
                pos_thresh=args.pos_thresh,
                skip_existing=args.skip_existing,
            )
            b2_paths.append(b2_npz)
        except Exception as e:
            print(f"[C4] Warning: failed to build b1/b2 for z={z}: {e}")
    if len(b2_paths) == 0:
        raise RuntimeError("No b2 graphs available for inference.")

    # Load first graph to infer in_dim, load model
    tmp_g = np.load(b2_paths[0], allow_pickle=True)
    in_dim = int(tmp_g["X"].shape[1])
    model, meta = load_checkpoint(
        ckpt_path=args.model_ckpt if os.path.isabs(args.model_ckpt) else os.path.join(repo_root, args.model_ckpt),
        in_dim_hint=in_dim,
        device=device
    )
    print(f"[C4] Model loaded. in_dim={in_dim}, hidden={meta.get('hidden', 'n/a')}, dropout={meta.get('dropout', 'n/a')}")

    # Inference and stacking
    prob_vol = np.zeros((H, W, Z), dtype=np.float32)
    pred_vol = np.zeros((H, W, Z), dtype=np.uint8)

    per_slice_index = []
    for b2_p in b2_paths:
        g = load_b2_graph(b2_p, device)
        with torch.no_grad():
            logits = model(g["X"], g["edge_index"]).detach().cpu().numpy()
        probs = probs_from_logits(logits)
        prob_mask, pred_mask = reconstruct_masks(g["labels_img"], probs, thr=args.threshold)

        # Place into volume
        z = g["z"]
        if z is None:
            # Fallback: parse from filename
            base = os.path.basename(b2_p)
            zstr = base.split("_z")[-1].split(".")[0]
            z = int(zstr)
        prob_vol[:, :, z] = prob_mask
        pred_vol[:, :, z] = pred_mask

        # Save per-slice NPZ
        ztag = pad_z(z)
        out_npz = os.path.join(slice_npz_dir, f"c4_{case_id}_z{ztag}_pred.npz")
        np.savez(out_npz,
                 prob_spx=probs.astype(np.float32),
                 pred_spx=(probs >= args.threshold).astype(np.uint8),
                 labels=g["labels_img"],
                 prob_mask=prob_mask.astype(np.float32),
                 pred_mask=pred_mask.astype(np.uint8),
                 z=z,
                 case_id=case_id,
                 b2_npz=b2_p)
        # Optional overlay
        if args.save_overlays:
            # use base modality for background
            bg2d = base_vol[:, :, z]
            overlay_png = os.path.join(overlay_dir, f"c4_{case_id}_z{ztag}_overlay.png")
            save_overlay_png(bg2d, pred_mask, overlay_png, title=f"{case_id} z={ztag}")

        per_slice_index.append({"case_id": case_id, "z": int(z), "b2_npz": b2_p, "pred_npz": out_npz})

    # Save NIfTI volumes
    prob_img = nib.Nifti1Image(prob_vol, affine=base_img.affine, header=base_img.header)
    pred_img = nib.Nifti1Image(pred_vol.astype(np.uint8), affine=base_img.affine, header=base_img.header)
    out_prob_nii = os.path.join(out_dir, f"c4_{case_id}_prob.nii.gz")
    out_pred_nii = os.path.join(out_dir, f"c4_{case_id}_pred.nii.gz")
    nib.save(prob_img, out_prob_nii)
    nib.save(pred_img, out_pred_nii)

    # Save an index JSON for this run
    meta_out = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "case_id": case_id,
        "case_dir": case_dir,
        "model_ckpt": args.model_ckpt if os.path.isabs(args.model_ckpt) else os.path.relpath(args.model_ckpt, repo_root),
        "strategy": args.select_strategy,
        "threshold": args.threshold,
        "n_segments": args.n_segments,
        "compactness": args.compactness,
        "pos_thresh": args.pos_thresh,
        "slices": sorted([int(s["z"]) for s in per_slice_index]),
        "num_slices": len(per_slice_index),
        "prob_nii": out_prob_nii,
        "pred_nii": out_pred_nii,
        "save_overlays": args.save_overlays,
        "overlays_dir": overlay_dir if args.save_overlays else None,
    }
    idx_json = os.path.join(out_dir, f"c4_{case_id}_index.json")
    with open(idx_json, "w") as f:
        json.dump({"meta": meta_out, "items": per_slice_index}, f, indent=2)

    dt = time.time() - t0
    print(f"[C4] Saved 3D outputs:- prob: {out_prob_nii}- pred: {out_pred_nii}")
    if args.save_overlays:
        print(f"[C4] Overlays in: {overlay_dir}")
    print(f"[C4] Index JSON: {idx_json}")
    print(f"[C4] Done in {dt:.1f}s.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Micro-task B1: superpixels + features on one positive slice, with debug PNG.

Usage example:
  python3 src/b1_superpixels.py --case "/home/khan/Desktop/Capstone/GNNBasedTumorSegmentation/GNN_TumorSeg/data/BraTS2021_00495" \
      --slice_strategy first_pos --n_segments 350 --compactness 0.1 --pos_thresh 0.2 --out_dir outputs
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import nibabel as nib

from skimage.segmentation import slic, mark_boundaries
from skimage.filters import sobel
import matplotlib.pyplot as plt

def as_canonical_fdata(p: Path) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(str(p)))
    return img.get_fdata()

def robust_minmax(vol: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    v = vol[np.isfinite(vol)]
    if v.size == 0:
        low, high = 0.0, 1.0
    else:
        low = np.percentile(v, p_low)
        high = np.percentile(v, p_high)
        if not np.isfinite(low): low = float(np.nanmin(v))
        if not np.isfinite(high): high = float(np.nanmax(v))
        if high <= low: high = low + 1.0
    x = (vol - low) / (high - low + eps)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def find_modality(case_dir: Path, key: str) -> Path | None:
    hits = list(case_dir.glob(f"*_{key}.nii*"))
    return hits[0] if hits else None

def find_case_files(case_dir: Path) -> dict[str, Path | None]:
    return {
        "flair": find_modality(case_dir, "flair"),
        "t1ce": find_modality(case_dir, "t1ce"),
        "t2":   find_modality(case_dir, "t2"),
        "t1":   find_modality(case_dir, "t1"),
        "seg":  find_modality(case_dir, "seg"),
    }

def whole_tumor_mask(seg_vol: np.ndarray) -> np.ndarray:
    # BraTS labels: 0=background, 1/2/4 tumor subregions; WT = seg > 0
    return (seg_vol > 0).astype(np.uint8)

def pick_slice(wt: np.ndarray, strategy: str = "first_pos") -> int:
    H, W, D = wt.shape
    pos = [z for z in range(D) if int(wt[:, :, z].sum()) > 0]
    if not pos:
        return D // 2  # fallback
    if strategy == "first_pos":
        return pos[0]
    if strategy in ("middle_pos", "central_pos"):
        return pos[len(pos) // 2]
    if strategy == "last_pos":
        return pos[-1]
    return pos[0]

def run_slic(img_hw_c: np.ndarray, n_segments: int, compactness: float, sigma: float) -> np.ndarray:
    # skimage API changed: prefer channel_axis=-1; fallback to multichannel=True if needed
    try:
        labels = slic(
            img_hw_c, n_segments=n_segments, compactness=compactness,
            sigma=sigma, start_label=0, convert2lab=False, channel_axis=-1
        )
    except TypeError:
        labels = slic(
            img_hw_c, n_segments=n_segments, compactness=compactness,
            sigma=sigma, start_label=0, convert2lab=False, multichannel=True
        )
    return labels.astype(np.int32)

def bincount_stats(labels_flat: np.ndarray, values_flat: np.ndarray, n_labels: int, eps: float = 1e-8):
    counts = np.bincount(labels_flat, minlength=n_labels).astype(np.float32)
    s1 = np.bincount(labels_flat, weights=values_flat, minlength=n_labels).astype(np.float32)
    s2 = np.bincount(labels_flat, weights=values_flat * values_flat, minlength=n_labels).astype(np.float32)
    mean = s1 / (counts + eps)
    var = s2 / (counts + eps) - mean * mean
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var + eps)
    return counts, mean, std

def compute_features_and_labels(img_hw_c: np.ndarray, wt_slice: np.ndarray, labels_hw: np.ndarray, pos_thresh: float = 0.2):
    H, W, C = img_hw_c.shape
    lab = labels_hw.astype(np.int32)
    n = int(lab.max()) + 1
    lab_flat = lab.ravel()

    # Geometry
    counts = np.bincount(lab_flat, minlength=n).astype(np.float32)
    area_norm = counts / float(H * W)

    yy = np.repeat(np.arange(H, dtype=np.float32), W)
    xx = np.tile(np.arange(W, dtype=np.float32), H)
    cy = np.bincount(lab_flat, weights=yy, minlength=n) / (counts + 1e-8)
    cx = np.bincount(lab_flat, weights=xx, minlength=n) / (counts + 1e-8)
    cy_norm = cy / max(1.0, (H - 1))
    cx_norm = cx / max(1.0, (W - 1))

    feats = []
    # Per-channel intensity mean/std and gradient mean
    for c in range(C):
        Ic = img_hw_c[:, :, c].astype(np.float32)
        grad = sobel(Ic)
        c_counts, c_mean, c_std = bincount_stats(lab_flat, Ic.ravel(), n)
        _, g_mean, _ = bincount_stats(lab_flat, grad.ravel(), n)
        feats.append(c_mean)
        feats.append(c_std)
        feats.append(g_mean)

    # Stack features: [ch0_mean, ch0_std, ch0_grad, ch1_mean, ch1_std, ch1_grad, ..., area_norm, cy_norm, cx_norm]
    X = np.stack(feats + [area_norm, cy_norm, cx_norm], axis=1).astype(np.float32)

    # Labels: positive if tumor fraction in the superpixel >= pos_thresh
    # f_i = (#tumor pixels in SP_i) / (#pixels in SP_i)
    wt_flat = wt_slice.astype(np.float32).ravel()
    tumor_pix = np.bincount(lab_flat, weights=wt_flat, minlength=n).astype(np.float32)
    frac = tumor_pix / (counts + 1e-8)
    y = (frac >= float(pos_thresh)).astype(np.int64)

    return X, y, frac, dict(area_norm=area_norm, cy_norm=cy_norm, cx_norm=cx_norm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, required=True, help="Path to a single case folder (contains *_flair.nii.gz etc.)")
    ap.add_argument("--z", type=int, default=None, help="Slice index to use (overrides strategy)")
    ap.add_argument("--slice_strategy", type=str, default="first_pos",
                    choices=["first_pos", "middle_pos", "last_pos", "central_pos"],
                    help="How to pick a slice if --z not given")
    ap.add_argument("--n_segments", type=int, default=300)
    ap.add_argument("--compactness", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--pos_thresh", type=float, default=0.2, help="Fraction threshold for positive superpixel")
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    case_dir = Path(args.case)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_case_files(case_dir)
    if files["seg"] is None:
        print(f"[ERROR] No *_seg.nii* in {case_dir}")
        sys.exit(1)

    seg_vol = as_canonical_fdata(files["seg"]).astype(np.int16)
    wt = whole_tumor_mask(seg_vol)  # H×W×D
    H, W, D = wt.shape

    # Choose modalities (priority: flair, t1ce, t2, t1)
    mods_order = ["flair", "t1ce", "t2", "t1"]
    chosen = [m for m in mods_order if files[m] is not None]
    if not chosen:
        print(f"[ERROR] No modalities found (expected *_flair.nii* etc.) in {case_dir}")
        sys.exit(1)

    vols = []
    for m in chosen:
        v = as_canonical_fdata(files[m])
        # Normalize per volume robustly to [0,1] for stable SLIC
        v = robust_minmax(v, 1.0, 99.0)
        vols.append(v.astype(np.float32))
    # Stack channels last
    # Each vol: H×W×D -> we will slice along D after choosing z
    H2, W2, D2 = vols[0].shape
    assert (H2, W2, D2) == (H, W, D), "Modalities and seg must have matching shapes"

    # Pick slice
    if args.z is not None:
        z = int(args.z)
        if not (0 <= z < D):
            print(f"[ERROR] z={z} out of bounds [0,{D-1}]")
            sys.exit(1)
    else:
        z = pick_slice(wt, strategy=args.slice_strategy)

    # Build multi-channel slice
    img_slice = np.stack([v[:, :, z] for v in vols], axis=-1)  # H×W×C
    wt_slice = wt[:, :, z]

    # Superpixels
    labels = run_slic(img_slice, n_segments=args.n_segments, compactness=args.compactness, sigma=args.sigma)
    K = int(labels.max()) + 1

    # Features + labels
    X, y, frac, geo = compute_features_and_labels(img_slice, wt_slice, labels, pos_thresh=args.pos_thresh)
    pos_n = int(y.sum())
    neg_n = int((K - pos_n))

    # Save artifacts
    case_name = case_dir.name
    out_png = out_dir / f"b1_{case_name}_z{z:03d}.png"
    out_npz = out_dir / f"b1_{case_name}_z{z:03d}.npz"
    np.savez_compressed(
        out_npz,
        X=X, y=y, frac=frac.astype(np.float32),
        labels=labels.astype(np.int32),
        z=np.int32(z), shape=np.int32([H, W]),
        modalities=np.array(chosen, dtype=object)
    )

    # Debug figure
    base = img_slice[:, :, 0]  # show first modality (likely FLAIR)
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    ax = axs[0, 0]
    ax.imshow(base, cmap="gray")
    ax.set_title(f"{case_name} z={z}  base={chosen[0]}")
    ax.axis("off")

    ax = axs[0, 1]
    ax.imshow(mark_boundaries(base, labels, color=(0, 1, 0), mode="thick"))
    ax.set_title(f"SLIC boundaries (K={K})")
    ax.axis("off")

    ax = axs[1, 0]
    ax.imshow(base, cmap="gray")
    ax.imshow(wt_slice, cmap="Reds", alpha=0.35)
    ax.set_title("Tumor mask overlay (WT)")
    ax.axis("off")

    ax = axs[1, 1]
    pos_ids = np.flatnonzero(y == 1)
    pos_map = np.isin(labels, pos_ids)
    ax.imshow(base, cmap="gray")
    ax.imshow(pos_map, cmap="autumn", alpha=0.35)
    ax.imshow(mark_boundaries(base, labels, color=(0, 1, 0), mode="thick"))
    ax.set_title(f"Positive SPs (y=1, thresh={args.pos_thresh:.2f})")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"[OK] {case_name} z={z} modalities={','.join(chosen)}  K={K}  pos_spx={pos_n}  neg_spx={neg_n}  pos_frac={pos_n/max(1,K):.3f}")
    print(f"     saved: {out_png}")
    print(f"     saved: {out_npz}")

if __name__ == "__main__":
    main()
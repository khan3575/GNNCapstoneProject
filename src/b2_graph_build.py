#!/usr/bin/env python3
"""
Micro-task B2: Build a graph from B1 outputs (superpixel labels/features).

Usage example:
  python3 src/b2_graph_build.py \
    --b1_npz outputs/b1_BraTS2021_00495_z101.npz \
    --case "/home/khan/Desktop/Capstone/GNNBasedTumorSegmentation/GNN_TumorSeg/data/BraTS2021_00495" \
    --out_dir outputs
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

import nibabel as nib

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

def as_canonical_fdata(p: Path) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(str(p)))
    return img.get_fdata()

def find_modality(case_dir: Path, key: str) -> Path | None:
    hits = list(case_dir.glob(f"*_{key}.nii*"))
    return hits[0] if hits else None

def _unique_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = (a != b)
    if not np.any(mask):
        return np.zeros((0, 2), dtype=np.int32)
    u = np.minimum(a[mask], b[mask])
    v = np.maximum(a[mask], b[mask])
    pairs = np.unique(np.stack([u, v], axis=1), axis=0)
    return pairs.astype(np.int32)

def adjacency_from_labels(labels: np.ndarray) -> np.ndarray:
    L = labels.astype(np.int32)
    # Horizontal neighbors
    pairs_h = _unique_pairs(L[:, :-1].ravel(), L[:, 1:].ravel())
    # Vertical neighbors
    pairs_v = _unique_pairs(L[:-1, :].ravel(), L[1:, :].ravel())
    if pairs_h.size == 0 and pairs_v.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    edges = np.unique(np.vstack([pairs_h, pairs_v]), axis=0)
    return edges.astype(np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1_npz", type=str, required=True, help="Path to B1 .npz (contains X,y,labels,shape,z,modalities)")
    ap.add_argument("--case", type=str, default=None, help="Case folder for loading a base image for the debug PNG")
    ap.add_argument("--base_modality", type=str, default="", help="Which modality to show (default: first available in the B1 npz)")
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.b1_npz, allow_pickle=True)
    X = data["X"]                              # [K, F]
    y = data["y"].astype(np.int64)            # [K]
    labels = data["labels"].astype(np.int32)  # [H, W]
    H, W = map(int, data["shape"])
    z = int(data["z"])
    # modalities saved as object array → make it a list of strings
    modalities = [str(m) for m in data["modalities"].tolist()]

    K = X.shape[0]
    K_from_labels = int(labels.max()) + 1
    if K_from_labels != K:
        print(f"[WARN] K from labels ({K_from_labels}) != K from X ({K}). Proceeding with K={K}.")
    C = int((X.shape[1] - 3) // 3)
    if C < 1:
        print("[ERROR] Feature layout unexpected (need at least 1 channel).")
        sys.exit(1)

    # Build adjacency (4-neighborhood touch)
    edges = adjacency_from_labels(labels)  # [E,2]
    E = edges.shape[0]
    avg_deg = (2.0 * E) / max(1, K)

    # Node centroids from normalized coords at the end of X
    cy_norm = X[:, -2]   # y normalized in [0,1]
    cx_norm = X[:, -1]   # x normalized in [0,1]
    cy = cy_norm * max(1, (H - 1))
    cx = cx_norm * max(1, (W - 1))

    # Edge features: |mean0 diff|, |grad0 diff|, centroid distance
    d_mean0 = np.abs(X[edges[:, 0], 0] - X[edges[:, 1], 0])
    d_grad0 = np.abs(X[edges[:, 0], 2] - X[edges[:, 1], 2])
    dist = np.sqrt((cx[edges[:, 0]] - cx[edges[:, 1]])**2 + (cy[edges[:, 0]] - cy[edges[:, 1]])**2)
    edge_attr = np.stack([d_mean0, d_grad0, dist], axis=1).astype(np.float32)

    # Names
    stem = Path(args.b1_npz).stem  # e.g., b1_BraTS2021_00495_z101
    case_tag = stem.replace("b1_", "")
    out_npz = out_dir / f"b2_{case_tag}.npz"
    np.savez_compressed(
        out_npz,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        edges=edges.astype(np.int32),       # [E,2]
        edge_attr=edge_attr,                # [E,3]
        cx=cx.astype(np.float32),
        cy=cy.astype(np.float32),
        labels=labels.astype(np.int32),
        shape=np.int32([H, W]),
        z=np.int32(z),
        modalities=np.array(modalities, dtype=object),
        avg_deg=np.float32(avg_deg),
    )

    # Optional debug PNG if case folder is provided
    out_png = out_dir / f"b2_{case_tag}.png"
    if args.case is not None:
        case_dir = Path(args.case)
        # Choose a base modality to display
        chosen_mod = args.base_modality.strip().lower()
        if not chosen_mod:
            chosen_mod = modalities[0] if len(modalities) > 0 else "flair"

        mpath = find_modality(case_dir, chosen_mod)
        if mpath is None:
            # fall back to any present
            for fallback in ["flair", "t1ce", "t2", "t1"]:
                mpath = find_modality(case_dir, fallback)
                if mpath is not None:
                    chosen_mod = fallback
                    break

        base = None
        if mpath is not None:
            vol = as_canonical_fdata(mpath)
            if not (0 <= z < vol.shape[2]):
                print(f"[WARN] z={z} out of bounds for {chosen_mod} volume; skipping PNG.")
            else:
                base = robust_minmax(vol, 1.0, 99.0)[:, :, z]

        if base is not None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(base, cmap="gray")
            ax.imshow(mark_boundaries(base, labels, color=(0, 1, 0), mode="thick"))
            # Draw edges
            for u, v in edges:
                ax.plot([cx[u], cx[v]], [cy[u], cy[v]], color="white", alpha=0.25, linewidth=0.7)
            # Draw nodes (color by label y)
            pos = (y == 1)
            ax.scatter(cx[~pos], cy[~pos], s=6, c="#23a6d5", label="neg", alpha=0.8)
            ax.scatter(cx[pos],  cy[pos],  s=10, c="#ff7f0e", label="pos", alpha=0.9)
            ax.set_title(f"{case_tag}  K={K}  E={E}  avg_deg={avg_deg:.2f}  base={chosen_mod}")
            ax.axis("off")
            ax.legend(loc="lower right", fontsize=8, frameon=True)
            fig.tight_layout()
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
        else:
            print(f"[WARN] Could not render base image; PNG not created.")

    print(f"[OK] Graph built from {args.b1_npz}")
    print(f"     nodes K={K}, edges E={E}, avg_deg={avg_deg:.2f}, pos_nodes={int(y.sum())}")
    print(f"     saved: {out_npz}")
    if out_png.exists():
        print(f"     saved: {out_png}")

if __name__ == "__main__":
    main()
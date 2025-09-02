#!/usr/bin/env python3
"""
Micro-task B3: Sanity-check GNN on a single superpixel graph (node classification).

Usage:
  python3 src/b3_gnn_sanity.py \
    --b2_npz outputs/b2_BraTS2021_00495_z101.npz \
    --case "/home/khan/Desktop/Capstone/GNNBasedTumorSegmentation/GNN_TumorSeg/data/BraTS2021_00495" \
    --epochs 200 --hidden 64 --lr 1e-3 --dropout 0.2 \
    --out_dir outputs
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import nibabel as nib
import copy
import warnings

# -------------------- Utils --------------------

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

def build_norm_adj(K: int, edges: np.ndarray) -> torch.Tensor:
    """
    Build symmetric normalized adjacency A_hat = D^{-1/2} (A + I) D^{-1/2} as a sparse COO tensor.
    """
    # Undirected edges + self-loops
    rows = np.concatenate([edges[:, 0], edges[:, 1], np.arange(K)])
    cols = np.concatenate([edges[:, 1], edges[:, 0], np.arange(K)])
    vals = np.ones_like(rows, dtype=np.float32)

    # Degree (sum over rows)
    deg = np.bincount(rows, weights=vals, minlength=K).astype(np.float32)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm_vals = vals * d_inv_sqrt[rows] * d_inv_sqrt[cols]

    indices = np.vstack([rows, cols])
    A_hat = torch.sparse_coo_tensor(
        indices= torch.tensor(indices, dtype=torch.long),
        values = torch.tensor(norm_vals, dtype=torch.float32),
        size   = (K, K),
    )
    A_hat = A_hat.coalesce()
    return A_hat

def zscore_features(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)

# -------------------- Model --------------------

class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # X: [K, F], A_hat: [K, K] sparse
        H = torch.relu(torch.sparse.mm(A_hat, self.lin1(X)))
        H = self.dropout(H)
        Z = torch.sparse.mm(A_hat, self.lin2(H)).squeeze(-1)  # [K]
        return Z

# -------------------- Training / Eval --------------------

def split_masks(y: np.ndarray, val_frac=0.2, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    # Stratified split
    pos = idx[y == 1]
    neg = idx[y == 0]

    def split_group(gidx):
        rng.shuffle(gidx)
        n = len(gidx)
        n_test = int(round(test_frac * n))
        n_val  = int(round(val_frac  * n))
        test = gidx[:n_test]
        val  = gidx[n_test:n_test+n_val]
        train = gidx[n_test+n_val:]
        return train, val, test

    tr_p, va_p, te_p = split_group(pos)
    tr_n, va_n, te_n = split_group(neg)

    train = np.concatenate([tr_p, tr_n])
    val   = np.concatenate([va_p, va_n])
    test  = np.concatenate([te_p, te_n])

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

    K = len(y)
    train_mask = np.zeros(K, dtype=bool); train_mask[train] = True
    val_mask   = np.zeros(K, dtype=bool); val_mask[val]     = True
    test_mask  = np.zeros(K, dtype=bool); test_mask[test]   = True
    return train_mask, val_mask, test_mask

def metrics_from_logits(logits: np.ndarray, y: np.ndarray, threshold: float = 0.5):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y, probs))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y, probs))
    except Exception:
        out["pr_auc"] = float("nan")
    try:
        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        out["precision"] = float(prec)
        out["recall"]    = float(rec)
        out["f1"]        = float(f1)
    except Exception:
        out["precision"] = out["recall"] = out["f1"] = float("nan")

    out["acc"] = float(accuracy_score(y, preds))
    out["threshold"] = float(threshold)
    return out

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2_npz", type=str, required=True)
    ap.add_argument("--case", type=str, default=None, help="Case folder for base image (for overlay PNG)")
    ap.add_argument("--base_modality", type=str, default="", help="Which modality to render (default: first available)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.b2_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)          # [K, F]
    y = data["y"].astype(np.int64)            # [K]
    edges = data["edges"].astype(np.int64)    # [E,2]
    labels = data["labels"].astype(np.int32)  # [H,W]
    cx = data["cx"].astype(np.float32)        # [K]
    cy = data["cy"].astype(np.float32)        # [K]
    H, W = map(int, data["shape"])
    z = int(data["z"])
    modalities = [str(m) for m in data["modalities"].tolist()]

    K, Fdim = X.shape
    P = int(y.sum())
    N = K - P
    pos_weight = (N / max(1, P))
    print(f"[INFO] K={K}, F={Fdim}, positives={P} ({P/K:.3%}), pos_weight={pos_weight:.2f}")

    # Feature scaling
    Xs = zscore_features(X)
    X_t = torch.tensor(Xs, dtype=torch.float32)

    # Graph
    A_hat = build_norm_adj(K, edges)

    # Splits
    train_mask, val_mask, test_mask = split_masks(y, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(device)
    y_t = y_t.to(device)
    A_hat = A_hat.to(device)

    model = GCN(in_dim=Fdim, hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

    # Training loop with early stopping (by val PR-AUC)
    best_state = None
    best_val_score = -np.inf
    patience = 30
    wait = 0

    tr_idx = torch.tensor(np.where(train_mask)[0], dtype=torch.long, device=device)
    va_idx = torch.tensor(np.where(val_mask)[0], dtype=torch.long, device=device)
    te_idx = torch.tensor(np.where(test_mask)[0], dtype=torch.long, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t, A_hat)

        loss = criterion(logits[tr_idx], y_t[tr_idx])
        loss.backward()
        optimizer.step()

        # Eval on val
        model.eval()
        with torch.no_grad():
            logits_full = model(X_t, A_hat).detach().cpu().numpy()
            val_metrics = metrics_from_logits(logits_full[va_idx.cpu().numpy()], y[va_idx.cpu().numpy()])
            tr_metrics  = metrics_from_logits(logits_full[tr_idx.cpu().numpy()], y[tr_idx.cpu().numpy()])
        val_key = val_metrics.get("pr_auc", float("nan"))
        print(f"[E{epoch:03d}] loss={loss.item():.4f} | "
              f"val PR-AUC={val_metrics['pr_auc']:.3f} ROC-AUC={val_metrics['roc_auc']:.3f} "
              f"| train PR-AUC={tr_metrics['pr_auc']:.3f}")

        improved = np.isfinite(val_key) and (val_key > best_val_score + 1e-5)
        if improved:
            best_val_score = val_key
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[EARLY STOP] No improvement in {patience} epochs. Best val PR-AUC={best_val_score:.3f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits_full = model(X_t, A_hat).detach().cpu().numpy()

    train_metrics = metrics_from_logits(logits_full[train_mask], y[train_mask])
    val_metrics   = metrics_from_logits(logits_full[val_mask],   y[val_mask])
    test_metrics  = metrics_from_logits(logits_full[test_mask],  y[test_mask])

    print("[FINAL] Train:", train_metrics)
    print("[FINAL] Val  :", val_metrics)
    print("[FINAL] Test :", test_metrics)

    # Save predictions + metrics
    stem = Path(args.b2_npz).stem.replace("b2_", "")
    pred_npz = out_dir / f"b3_{stem}_pred.npz"
    np.savez_compressed(
        pred_npz,
        logits=logits_full.astype(np.float32),
        probs=(1.0 / (1.0 + np.exp(-logits_full))).astype(np.float32),
        y=y.astype(np.int64),
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        metrics=dict(train=train_metrics, val=val_metrics, test=test_metrics),
        z=np.int32(z),
    )
    print(f"[OK] Saved predictions: {pred_npz}")

    # Optional overlay PNG
    out_png = out_dir / f"b3_{stem}_overlay.png"
    base = None
    if args.case is not None:
        case_dir = Path(args.case)
        chosen_mod = args.base_modality.strip().lower()
        if not chosen_mod:
            chosen_mod = modalities[0] if len(modalities) > 0 else "flair"
        mpath = find_modality(case_dir, chosen_mod)
        if mpath is None:
            for fallback in ["flair", "t1ce", "t2", "t1"]:
                mpath = find_modality(case_dir, fallback)
                if mpath is not None:
                    chosen_mod = fallback
                    break
        if mpath is not None:
            vol = as_canonical_fdata(mpath)
            if 0 <= z < vol.shape[2]:
                base = robust_minmax(vol, 1.0, 99.0)[:, :, z]
            else:
                warnings.warn(f"z={z} out of bounds for {chosen_mod} volume; skipping PNG.")

    if base is not None:
        probs = 1.0 / (1.0 + np.exp(-logits_full))
        thr = 0.5
        preds = (probs >= thr)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(base, cmap="gray")
        ax.imshow(mark_boundaries(base, labels, color=(0, 1, 0), mode="thick"))
        # draw predicted positives
        ax.scatter(cx[~preds], cy[~preds], s=6, c="#23a6d5", alpha=0.8, label="pred neg")
        ax.scatter(cx[preds],  cy[preds],  s=12, c="#ff7f0e", alpha=0.9, label="pred pos")
        ax.set_title(f"{stem}  thr={thr}  Test AUC={test_metrics['roc_auc']:.3f} PR-AUC={test_metrics['pr_auc']:.3f}")
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"[OK] Saved overlay: {out_png}")
    else:
        print("[WARN] Base image not available; overlay skipped.")

if __name__ == "__main__":
    main()
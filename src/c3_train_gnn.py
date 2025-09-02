#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional sklearn metrics; fallback if missing
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
    SKL_OK = True
except Exception:
    SKL_OK = False

def find_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_to_tensor(x, dtype=torch.float32, device="cpu"):
    return torch.as_tensor(x, dtype=dtype, device=device)

def load_b2_graph(b2_path: str, device: str = "cpu") -> Dict[str, Any]:
    data = np.load(b2_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32).reshape(-1)
    edges = data["edges"]
    if edges.ndim == 2:
        if edges.shape[0] == 2:
            edge_index = edges.astype(np.int64)
        elif edges.shape[1] == 2:
            edge_index = edges.T.astype(np.int64)
        else:
            raise ValueError(f"Unexpected edges shape {edges.shape}")
    else:
        raise ValueError(f"Unexpected edges shape {edges.shape}")

    # Metadata
    z = None
    if "z" in data:
        try:
            z = int(np.array(data["z"]).item())
        except Exception:
            try:
                z = int(np.array(data["z"]).tolist())
            except Exception:
                z = None
    shape = None
    if "shape" in data:
        shape = [int(u) for u in np.array(data["shape"]).tolist()]
    case_id = parse_case_id_from_b2(b2_path)
    out = {
        "X": safe_to_tensor(X, dtype=torch.float32, device=device),
        "y": safe_to_tensor(y, dtype=torch.float32, device=device),
        "edge_index": safe_to_tensor(edge_index, dtype=torch.long, device=device),
        "K": int(X.shape[0]),
        "F": int(X.shape[1]),
        "z": z,
        "shape": shape,
        "case_id": case_id,
        "b2_path": b2_path,
    }
    return out

def parse_case_id_from_b2(b2_path: str) -> Optional[str]:
    # Expect names like b2_BraTS2021_00495_z101.npz
    base = os.path.basename(b2_path)
    if not base.startswith("b2_"):
        return None
    core = base[len("b2_"):]
    # split at _z
    parts = core.split("_z")
    if len(parts) >= 2:
        return parts[0]
    return None

def undirected_deg_plus_self(edges_und: torch.Tensor, num_nodes: int, device: str) -> torch.Tensor:
    # edges_und shape [2, Eund], undirected pairs unique (u,v)
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    if edges_und.numel() > 0:
        u = edges_und[0]
        v = edges_und[1]
        deg.index_add_(0, u, torch.ones_like(u, dtype=torch.float32))
        deg.index_add_(0, v, torch.ones_like(v, dtype=torch.float32))
    deg = deg + 1.0  # + self-loop
    return deg

def make_directed_with_self(edges_und: torch.Tensor, num_nodes: int, device: str) -> torch.Tensor:
    if edges_und.numel() == 0:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        u = edges_und[0]
        v = edges_und[1]
        ei = torch.cat([torch.stack([u, v], dim=0), torch.stack([v, u], dim=0)], dim=1)
    self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    ei_self = torch.stack([self_loops, self_loops], dim=0)
    ei = torch.cat([ei, ei_self], dim=1)
    return ei

class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index_und: torch.Tensor) -> torch.Tensor:
        # edge_index_und: [2, Eund] undirected unique pairs
        N = x.size(0)
        device = x.device
        # degree using undirected + self
        deg = undirected_deg_plus_self(edge_index_und, N, device=device)
        # build directed with self for message passing
        ei = make_directed_with_self(edge_index_und, N, device=device)  # [2, Edir]
        src, dst = ei[0], ei[1]
        # normalize coeffs: 1/sqrt(deg[u]*deg[v])
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

def list_from_split_paths(splits_json: str, split_key: str, repo_root: str) -> List[str]:
    with open(splits_json, "r") as f:
        obj = json.load(f)
    paths_rel = obj["splits"].get(split_key, [])
    abs_paths = [os.path.join(repo_root, p) if not os.path.isabs(p) else p for p in paths_rel]
    # Filter existing only
    return [p for p in abs_paths if os.path.isfile(p)]

def compute_pos_weight(train_graphs: List[Dict[str, Any]], mode: str = "global") -> float:
    if mode == "per_graph":
        # Not used here; per-graph handled in loss construction loop if desired.
        mode = "global"
    total_nodes = 0
    total_pos = 0
    for g in train_graphs:
        y = g["y"]
        total_nodes += int(y.numel())
        total_pos += int((y > 0.5).sum().item())
    if total_pos == 0:
        return 1.0
    # Using N/P as requested
    return float(total_nodes) / float(total_pos)

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.int32).reshape(-1)
    probs = sigmoid_np(logits.astype(np.float64).reshape(-1))
    y_pred = (probs >= 0.5).astype(np.int32)
    out = {}
    # Accuracy
    out["acc"] = float((y_pred == y_true).mean()) if y_true.size > 0 else 0.0
    # Precision/Recall/F1
    if SKL_OK:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        out["precision"] = float(p)
        out["recall"] = float(r)
        out["f1"] = float(f1)
    else:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        out["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        out["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        out["f1"] = float(2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])) if (out["precision"] + out["recall"]) > 0 else 0.0
    # ROC-AUC / PR-AUC
    if SKL_OK:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, probs))
        except Exception:
            out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out

def evaluate_graphs(graphs: List[Dict[str, Any]], model: nn.Module, device: str) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    all_logits = []
    all_labels = []
    per_graph_metrics = []
    with torch.no_grad():
        for g in graphs:
            logits = model(g["X"], g["edge_index"])
            y = g["y"]
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            m = metrics_from_logits(all_labels[-1], all_logits[-1])
            per_graph_metrics.append({"case_id": g["case_id"], "z": g["z"], **m})
    if len(all_labels) == 0:
        return {"acc": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan")}, per_graph_metrics
    y_cat = np.concatenate(all_labels, axis=0)
    logit_cat = np.concatenate(all_logits, axis=0)
    agg = metrics_from_logits(y_cat, logit_cat)
    return agg, per_graph_metrics

def save_predictions(graphs: List[Dict[str, Any]], model: nn.Module, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for g in graphs:
            logits = model(g["X"], g["edge_index"]).detach().cpu().numpy()
            probs = sigmoid_np(logits)
            pred = (probs >= 0.5).astype(np.uint8)
            case = g["case_id"] or "unknowncase"
            z = g["z"]
            ztag = f"{z:03d}" if isinstance(z, int) else str(z)
            out_npz = os.path.join(out_dir, f"c3_{case}_z{ztag}_pred.npz")
            np.savez(out_npz,
                     y=g["y"].detach().cpu().numpy().astype(np.uint8),
                     logits=logits.astype(np.float32),
                     prob=probs.astype(np.float32),
                     pred=pred.astype(np.uint8),
                     case_id=case,
                     z=z,
                     b2_npz=g["b2_path"])
            # Optionally: could save overlays later in C4
    print(f"[C3] Saved per-graph predictions to {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="C3: Train a multi-graph GCN for superpixel tumor classification.")
    parser.add_argument("--splits_json", type=str, default="outputs/splits.json", help="Path to splits.json from C2.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to write checkpoints and predictions.")
    parser.add_argument("--save_prefix", type=str, default="c3_gcn", help="Prefix for checkpoint and outputs.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pos_weight", type=str, default="global", choices=["global", "per_graph"], help="Use global or per-graph pos_weight.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    repo_root = find_repo_root()
    splits_path = os.path.join(repo_root, args.splits_json) if not os.path.isabs(args.splits_json) else args.splits_json
    out_dir = os.path.join(repo_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    preds_dir = os.path.join(out_dir, "preds_c3")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[C3] Using device: {device}")

    # Load split paths
    train_paths = list_from_split_paths(splits_path, "train", repo_root)
    val_paths = list_from_split_paths(splits_path, "val", repo_root)
    test_paths = list_from_split_paths(splits_path, "test", repo_root)
    if len(train_paths) == 0:
        raise RuntimeError("Train split is empty. Check your splits.json or build dataset with C1.")

    # Load graphs
    def load_many(paths: List[str]) -> List[Dict[str, Any]]:
        graphs = []
        for p in paths:
            g = load_b2_graph(p, device=device)
            # Ensure edges are undirected unique pairs (B2 should provide undirected; we enforce format)
            ei = g["edge_index"]
            if ei.shape[0] != 2:
                raise ValueError(f"edge_index shape unexpected: {ei.shape} in {p}")
            # Try to reduce to undirected unique pairs by sorting each pair (min,max) and unique
            u = torch.minimum(ei[0], ei[1])
            v = torch.maximum(ei[0], ei[1])
            und = torch.stack([u, v], dim=0)
            # unique columns
            und_t = und.t()
            und_unique, _ = torch.unique(und_t, dim=0, return_inverse=True)
            g["edge_index"] = und_unique.t().contiguous()
            graphs.append(g)
        return graphs

    train_graphs = load_many(train_paths)
    val_graphs = load_many(val_paths) if len(val_paths) > 0 else []
    test_graphs = load_many(test_paths) if len(test_paths) > 0 else []

    in_dim = train_graphs[0]["F"]
    model = GCN(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)

    # Loss and optimizer
    pos_w = compute_pos_weight(train_graphs, mode=args.pos_weight)
    print(f"[C3] Using pos_weight={pos_w:.4f} (mode={args.pos_weight})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    history = {"epoch": [], "train": [], "val": []}
    best_val = -math.inf
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    have_val = len(val_graphs) > 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_graphs)
        total_loss = 0.0
        total_nodes = 0
        for g in train_graphs:
            optimizer.zero_grad(set_to_none=True)
            logits = model(g["X"], g["edge_index"])
            loss = criterion(logits, g["y"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * g["K"]
            total_nodes += g["K"]

        avg_loss = total_loss / max(1, total_nodes)

        # Evaluate
        train_agg, _ = evaluate_graphs(train_graphs, model, device)
        if have_val:
            val_agg, _ = evaluate_graphs(val_graphs, model, device)
            val_pr = val_agg.get("pr_auc", float("nan"))
        else:
            val_agg = {"acc": float("nan"), "precision": float("nan"), "recall": float("nan"),
                       "f1": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan")}
            val_pr = float("nan")

        history["epoch"].append(epoch)
        history["train"].append({"loss": avg_loss, **train_agg})
        history["val"].append(val_agg)

        msg = f"[C3][{epoch:03d}] loss={avg_loss:.5f}  train PR-AUC={train_agg.get('pr_auc', float('nan')):.3f}"
        if have_val:
            msg += f"  val PR-AUC={val_pr:.3f}"
        print(msg)

        # Early stopping on val PR-AUC
        improved = False
        if have_val and not (np.isnan(val_pr) or val_pr is None):
            if val_pr > best_val:
                improved = True
                best_val = val_pr
                best_state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            # No val: keep last as "best" at the end
            best_state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in model.state_dict().items()}
            best_epoch = epoch

        if have_val and epochs_no_improve >= args.patience:
            print(f"[C3] Early stopping at epoch {epoch} (best val PR-AUC={best_val:.3f} @ epoch {best_epoch})")
            break

    dt = time.time() - t0

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"{args.save_prefix}_best.pt")
    meta = {
        "in_dim": in_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "pos_weight": float(pos_w),
        "device": device,
        "best_epoch": best_epoch,
        "best_val_pr_auc": float(best_val) if best_val != -math.inf else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "duration_sec": dt,
        "splits_json": os.path.relpath(splits_path, repo_root),
        "args": vars(args),
    }
    torch.save({"state_dict": best_state, "meta": meta}, ckpt_path)
    print(f"[C3] Saved checkpoint: {ckpt_path}")

    # Eval and save predictions with best state
    model.load_state_dict(best_state)
    train_agg, train_list = evaluate_graphs(train_graphs, model, device)
    print(f"[C3] Final Train PR-AUC: {train_agg.get('pr_auc', float('nan')):.3f}")
    if len(val_graphs) > 0:
        val_agg, val_list = evaluate_graphs(val_graphs, model, device)
        print(f"[C3] Final Val PR-AUC: {val_agg.get('pr_auc', float('nan')):.3f}")
    else:
        val_agg, val_list = ({"pr_auc": float("nan")}, [])
        print("[C3] No validation split; skipped validation metrics.")

    if len(test_graphs) > 0:
        test_agg, test_list = evaluate_graphs(test_graphs, model, device)
        print(f"[C3] Test PR-AUC: {test_agg.get('pr_auc', float('nan')):.3f}")
    else:
        test_agg, test_list = ({"pr_auc": float("nan")}, [])
        print("[C3] No test split; skipped test metrics.")

    # Save per-graph predictions
    save_predictions(train_graphs, model, preds_dir)
    if len(val_graphs) > 0:
        save_predictions(val_graphs, model, preds_dir)
    if len(test_graphs) > 0:
        save_predictions(test_graphs, model, preds_dir)

    # Save training log
    with open(os.path.join(out_dir, f"{args.save_prefix}_training_log.json"), "w") as f:
        json.dump({"meta": meta, "history": history,
                   "final_metrics": {"train": train_agg, "val": val_agg, "test": test_agg}}, f, indent=2)
    print(f"[C3] Wrote training log to {os.path.join(out_dir, f'{args.save_prefix}_training_log.json')}")
    print(f"[C3] Done in {dt/60.0:.1f} min.")

if __name__ == "__main__":
    main()
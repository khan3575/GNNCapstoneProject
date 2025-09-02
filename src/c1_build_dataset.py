#!/usr/bin/env python3
import argparse
import json
import os
import sys
import glob
import time
import random
import subprocess
from typing import List, Tuple, Dict, Any

import numpy as np
import nibabel as nib


def find_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_case_dirs(data_root: str, case_glob: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(data_root, case_glob)))
    return [p for p in paths if os.path.isdir(p)]


def find_seg_path(case_dir: str) -> str:
    case_id = os.path.basename(os.path.normpath(case_dir))
    candidates = []
    # Typical BraTS naming: <case>_seg.nii.gz
    candidates.extend(glob.glob(os.path.join(case_dir, f"{case_id}_seg.nii.gz")))
    # Fallback: any *_seg.nii.gz under the case directory
    candidates.extend(glob.glob(os.path.join(case_dir, "*_seg.nii.gz")))
    if not candidates:
        raise FileNotFoundError(f"No *_seg.nii.gz found in {case_dir}")
    # Prefer exact case-id match
    for c in candidates:
        if case_id in os.path.basename(c):
            return c
    return candidates[0]


def z_tag(z: int) -> str:
    return f"{int(z):03d}"


def find_artifact(out_dir: str, prefix: str, case_id: str, z: int, ext: str = ".npz") -> str:
    """Return existing artifact path if found (prefer padded), else canonical padded path."""
    pad = z_tag(z)
    candidates = [
        os.path.join(out_dir, f"{prefix}_{case_id}_z{pad}{ext}"),
        os.path.join(out_dir, f"{prefix}_{case_id}_z{z}{ext}"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Default to padded canonical name (used by builders)
    return candidates[0]


def select_slices_from_seg(
    seg_nii_path: str,
    min_pos: int = 1,
    margin: int = 2,
    neg_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    seg = nib.load(seg_nii_path).get_fdata()
    seg_bin = (seg > 0).astype(np.uint8)

    if seg_bin.ndim != 3:
        raise ValueError(f"Expected 3D seg volume, got shape {seg_bin.shape}")

    H, W, Z = seg_bin.shape
    pos_z = [int(z) for z in range(Z) if int(seg_bin[..., z].sum()) >= int(min_pos)]
    if len(pos_z) == 0:
        # No tumor in this case: return empty lists
        return [], [], []

    z_min, z_max = min(pos_z), max(pos_z)
    allowed_start = max(0, z_min - margin)
    allowed_end = min(Z - 1, z_max + margin)
    allowed = set(range(allowed_start, allowed_end + 1))
    pos_set = set(pos_z)
    neg_candidates = sorted(allowed - pos_set)

    # Number of negatives to sample
    if neg_ratio is None or neg_ratio < 0:
        n_neg = len(neg_candidates)
    else:
        n_neg = int(np.floor(neg_ratio * len(pos_z)))
        n_neg = min(n_neg, len(neg_candidates))
    neg_z = sorted(rng.sample(neg_candidates, n_neg)) if n_neg > 0 else []

    selected = sorted(set(pos_z + neg_z))
    return pos_z, neg_z, selected


def run_subprocess(cmd: List[str], cwd: str) -> None:
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(cmd)}")


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
    # Canonical padded output paths
    zstr = z_tag(z)
    b1_npz_abs = os.path.join(out_dir, f"b1_{case_id}_z{zstr}.npz")
    b2_npz_abs = os.path.join(out_dir, f"b2_{case_id}_z{zstr}.npz")

    # Build B1 if needed
    if not (skip_existing and os.path.isfile(b1_npz_abs)):
        b1_script = os.path.join(repo_root, "src", "b1_superpixels.py")
        if not os.path.isfile(b1_script):
            raise FileNotFoundError(f"Missing script: {b1_script}")
        cmd_b1 = [
            sys.executable,
            b1_script,
            "--case", case_dir,
            "--z", str(z),
            "--n_segments", str(n_segments),
            "--compactness", str(compactness),
            "--pos_thresh", str(pos_thresh),
            "--out_dir", out_dir,
        ]
        run_subprocess(cmd_b1, cwd=repo_root)

    # Find the actual b1 path (accept padded or unpadded)
    b1_npz_actual = find_artifact(out_dir, "b1", case_id, z, ".npz")
    if not os.path.isfile(b1_npz_actual):
        raise FileNotFoundError(f"B1 npz not found after build: {b1_npz_actual}")

    # Build B2 if needed
    if not (skip_existing and os.path.isfile(b2_npz_abs)):
        b2_script = os.path.join(repo_root, "src", "b2_graph_build.py")
        if not os.path.isfile(b2_script):
            raise FileNotFoundError(f"Missing script: {b2_script}")
        cmd_b2 = [
            sys.executable,
            b2_script,
            "--b1_npz", b1_npz_actual,
            "--out_dir", out_dir,
        ]
        run_subprocess(cmd_b2, cwd=repo_root)

    # Find the actual b2 path (accept padded or unpadded)
    b2_npz_actual = find_artifact(out_dir, "b2", case_id, z, ".npz")
    if not os.path.isfile(b2_npz_actual):
        raise FileNotFoundError(f"Expected B2 output not found (tried padded/unpadded): {b2_npz_actual}")

    return b1_npz_actual, b2_npz_actual


def load_b2_stats(b2_npz_path: str) -> Dict[str, Any]:
    data = np.load(b2_npz_path, allow_pickle=True)
    y = np.array(data["y"]).reshape(-1)
    K = int(y.shape[0])
    P = int(y.sum())
    pos_frac = float(P) / float(K) if K > 0 else 0.0

    shape = None
    if "shape" in data:
        shape = [int(x) for x in np.array(data["shape"]).tolist()]
    elif "labels" in data:
        shape = [int(x) for x in np.array(data["labels"]).shape[:2]]

    E = None
    if "edges" in data:
        edges = np.array(data["edges"])
        if edges.ndim == 2:
            if edges.shape[0] == 2:
                E = int(edges.shape[1])
            else:
                E = int(edges.shape[0])

    z_meta = None
    if "z" in data:
        try:
            z_meta = int(np.array(data["z"]).item())
        except Exception:
            try:
                z_meta = int(np.array(data["z"]).tolist())
            except Exception:
                z_meta = None

    modalities = None
    if "modalities" in data:
        try:
            modalities = list(np.array(data["modalities"]).tolist())
        except Exception:
            modalities = None

    return {
        "K": K,
        "P": P,
        "pos_frac": pos_frac,
        "E": E,
        "shape": shape,
        "z_meta": z_meta,
        "modalities": modalities,
    }


def main():
    parser = argparse.ArgumentParser(description="C1: Build multi-slice dataset by reusing B1 and B2.")
    parser.add_argument("--data_root", type=str, default="data", help="Root folder containing BraTS cases.")
    parser.add_argument("--case_glob", type=str, default="BraTS*", help="Glob pattern to match case folders.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to write outputs and index.")
    parser.add_argument("--cases_limit", type=int, default=None, help="Limit number of cases to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative slice sampling.")

    # Slice selection (A2 strategy)
    parser.add_argument("--min_pos", type=int, default=1, help="Min positive voxels to consider a slice positive.")
    parser.add_argument("--margin", type=int, default=2, help="Margin around tumor z-range to consider negatives.")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="Negatives per positive (within margin). Use <0 for all).")

    # B1 superpixel args
    parser.add_argument("--n_segments", type=int, default=600, help="Number of SLIC superpixels.")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness.")
    parser.add_argument("--pos_thresh", type=float, default=0.3, help="Superpixel positive threshold (tumor fraction).")

    # Behavior
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip computing if outputs exist.")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--include_existing", action="store_true", default=True, help="Include skipped items in index if present.")
    parser.add_argument("--max_slices_per_case", type=int, default=None, help="Optional cap on slices per case after selection.")

    args = parser.parse_args()

    repo_root = find_repo_root()
    data_root = os.path.join(repo_root, args.data_root) if not os.path.isabs(args.data_root) else args.data_root
    out_dir = os.path.join(repo_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Locate required scripts
    b1_script = os.path.join(repo_root, "src", "b1_superpixels.py")
    b2_script = os.path.join(repo_root, "src", "b2_graph_build.py")
    if not os.path.isfile(b1_script) or not os.path.isfile(b2_script):
        print("Error: Missing required scripts.", file=sys.stderr)
        print(f" - {b1_script}", file=sys.stderr)
        print(f" - {b2_script}", file=sys.stderr)
        sys.exit(1)

    case_dirs = find_case_dirs(data_root, args.case_glob)
    if args.cases_limit is not None:
        case_dirs = case_dirs[: args.cases_limit]

    rng = random.Random(args.seed)
    all_items = []
    per_case_counts = {}

    start_time = time.time()
    print(f"[C1] Found {len(case_dirs)} case(s). Building dataset index...")
    for idx, case_dir in enumerate(case_dirs, 1):
        case_id = os.path.basename(os.path.normpath(case_dir))
        try:
            seg_path = find_seg_path(case_dir)
        except FileNotFoundError as e:
            print(f"[{idx}/{len(case_dirs)}] {case_id}: {e}. Skipping.")
            continue

        try:
            pos_z, neg_z, selected_z = select_slices_from_seg(
                seg_path,
                min_pos=args.min_pos,
                margin=args.margin,
                neg_ratio=args.neg_ratio,
                seed=args.seed + idx,  # vary seed per case
            )
        except Exception as e:
            print(f"[{idx}/{len(case_dirs)}] {case_id}: slice selection failed ({e}). Skipping.")
            continue

        if len(selected_z) == 0:
            print(f"[{idx}/{len(case_dirs)}] {case_id}: no selected slices (no tumor or selection empty). Skipping.")
            continue

        if args.max_slices_per_case is not None and len(selected_z) > args.max_slices_per_case:
            selected_z = sorted(rng.sample(selected_z, args.max_slices_per_case))

        print(f"[{idx}/{len(case_dirs)}] {case_id}: pos={len(pos_z)}, neg={len(neg_z)}, selected={len(selected_z)}")

        per_case_counts[case_id] = {"pos": len(pos_z), "neg": len(neg_z), "selected": len(selected_z)}
        for z in selected_z:
            # Determine whether to build (if B2 already exists, skip)
            b2_existing = find_artifact(out_dir, "b2", case_id, z, ".npz")
            need_to_build = not (args.skip_existing and os.path.isfile(b2_existing))

            if need_to_build:
                try:
                    b1_abs, b2_abs = ensure_b1_b2_for_slice(
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
                except Exception as e:
                    print(f"  - {case_id} z={z}: build failed ({e}). Skipping this slice.")
                    continue
            else:
                # Use already existing artifacts
                b1_abs = find_artifact(out_dir, "b1", case_id, z, ".npz")
                b2_abs = find_artifact(out_dir, "b2", case_id, z, ".npz")

            if not os.path.isfile(b2_abs):
                print(f"  - {case_id} z={z}: B2 npz missing at {b2_abs}. Skipping this slice.")
                continue

            try:
                stats = load_b2_stats(b2_abs)
            except Exception as e:
                print(f"  - {case_id} z={z}: failed reading stats ({e}). Skipping this slice.")
                continue

            item = {
                "case_id": case_id,
                "case_path": os.path.relpath(case_dir, repo_root),
                "z": int(z),
                "b1_npz": os.path.relpath(b1_abs, repo_root),
                "b2_npz": os.path.relpath(b2_abs, repo_root),
                "K": int(stats["K"]) if stats.get("K") is not None else None,
                "P": int(stats["P"]) if stats.get("P") is not None else None,
                "pos_frac": float(stats["pos_frac"]) if stats.get("pos_frac") is not None else None,
                "E": int(stats["E"]) if stats.get("E") is not None else None,
                "shape": stats.get("shape"),
                "modalities": stats.get("modalities"),
            }
            all_items.append(item)

    # Save index JSON
    index_path = os.path.join(out_dir, "dataset_index.json")
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "data_root": os.path.relpath(data_root, repo_root),
        "out_dir": os.path.relpath(out_dir, repo_root),
        "case_glob": args.case_glob,
        "cases_processed": len(per_case_counts),
        "items": len(all_items),
        "args": {
            "min_pos": args.min_pos,
            "margin": args.margin,
            "neg_ratio": args.neg_ratio,
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "pos_thresh": args.pos_thresh,
            "skip_existing": args.skip_existing,
            "include_existing": args.include_existing,
            "max_slices_per_case": args.max_slices_per_case,
            "seed": args.seed,
        },
        "per_case_counts": per_case_counts,
    }
    with open(index_path, "w") as f:
        json.dump({"meta": meta, "items": all_items}, f, indent=2)

    dt = time.time() - start_time
    print(f"[C1] Wrote index with {len(all_items)} items across {len(per_case_counts)} case(s) to {index_path}")
    print(f"[C1] Done in {dt:.1f}s.")


if __name__ == "__main__":
    main()
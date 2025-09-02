#!/usr/bin/env python3
"""
Micro-task A2: select positive slices with margin and sample negatives.

Usage example:
  python src/a2_slice_selection.py --data_root /path/to/BraTSRoot --limit_cases 2 --min_pos 1 --margin 2 --neg_ratio 1.0
"""
import argparse
from pathlib import Path
import random
import numpy as np
import nibabel as nib

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def find_seg(case_dir: Path) -> Path | None:
    hits = list(case_dir.glob("*_seg.nii*"))
    return hits[0] if hits else None

def list_cases(data_root: Path) -> list[Path]:
    return [p for p in sorted(data_root.glob("*")) if p.is_dir() and find_seg(p) is not None]

def load_wt_mask(seg_path: Path) -> np.ndarray:
    # Whole tumor mask: seg > 0
    img = nib.as_closest_canonical(nib.load(str(seg_path)))
    seg = img.get_fdata().astype(np.int16)
    wt = (seg > 0).astype(np.uint8)  # H×W×D
    return wt

def select_slices_with_margin(wt: np.ndarray, min_pos: int = 1, margin: int = 2) -> tuple[list[int], list[int]]:
    H, W, D = wt.shape
    pos = [z for z in range(D) if int((wt[:, :, z] > 0).sum()) >= min_pos]
    if not pos:
        return [], []
    used = set()
    for z in pos:
        z0 = max(0, z - margin)
        z1 = min(D - 1, z + margin)
        for zz in range(z0, z1 + 1):
            used.add(zz)
    posm = sorted(used)
    return sorted(pos), posm

def sample_negatives(all_z: list[int], used_z: list[int], ratio: float = 1.0, seed: int = 42) -> list[int]:
    rng = random.Random(seed)
    used = set(used_z)
    neg_candidates = [z for z in all_z if z not in used]
    target = int(round(ratio * len(used_z)))
    target = min(target, len(neg_candidates))
    if target <= 0:
        return []
    return sorted(rng.sample(neg_candidates, target))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Folder containing case subfolders with *_seg.nii*")
    ap.add_argument("--limit_cases", type=int, default=2, help="Only probe first N cases")
    ap.add_argument("--min_pos", type=int, default=1, help="Min tumor pixels per slice to count as positive")
    ap.add_argument("--margin", type=int, default=2, help="Margin slices added around positives")
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="Negatives sampled vs used_z (≈1:1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.data_root)
    cases = list_cases(root)
    if args.limit_cases > 0:
        cases = cases[:args.limit_cases]

    if not cases:
        print(f"[ERROR] No cases with *_seg.nii* found under {root}")
        return

    print(f"Found {len(cases)} cases (showing up to {args.limit_cases}). min_pos={args.min_pos}, margin={args.margin}, neg_ratio={args.neg_ratio}")
    for i, case_dir in enumerate(cases, 1):
        seg_path = find_seg(case_dir)
        try:
            wt = load_wt_mask(seg_path)
        except Exception as e:
            print(f"[WARN] {case_dir.name}: failed to load seg: {e}")
            continue

        H, W, D = wt.shape
        pos_z, posm_z = select_slices_with_margin(wt, min_pos=args.min_pos, margin=args.margin)
        all_z = list(range(D))
        neg_z = sample_negatives(all_z, posm_z, ratio=args.neg_ratio, seed=args.seed)

        print(f"[{i:02d}] {case_dir.name}: shape=({H},{W},{D})  pos={len(pos_z)}  pos+margin={len(posm_z)}  neg_sample={len(neg_z)}")
        if len(pos_z) > 0:
            print(f"     examples → pos: {pos_z[:5]}   pos+margin: {posm_z[:7]}   neg: {neg_z[:7]}")

if __name__ == "__main__":
    main()
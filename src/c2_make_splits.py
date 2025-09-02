#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from typing import Dict, List, Any

def find_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_index(index_json: str) -> Dict[str, Any]:
    with open(index_json, "r") as f:
        data = json.load(f)
    if "items" not in data:
        # If stored as {"meta":..., "items":[...]} handle both
        if "meta" in data and "items" in data["meta"]:
            items = data["meta"]["items"]
        else:
            raise ValueError("Malformed index JSON: missing 'items'")
    return data

def main():
    parser = argparse.ArgumentParser(description="C2: Make case-wise train/val/test splits.")
    parser.add_argument("--index_json", type=str, default="outputs/dataset_index.json", help="Path to dataset_index.json from C1.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for splits.json")
    parser.add_argument("--train", type=float, default=0.7, help="Train fraction by case.")
    parser.add_argument("--val", type=float, default=0.15, help="Val fraction by case.")
    parser.add_argument("--test", type=float, default=0.15, help="Test fraction by case.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling cases.")
    parser.add_argument("--allow_few_cases", action="store_true", help="If set, allows 1-2 cases with degenerate splits.")
    args = parser.parse_args()

    repo_root = find_repo_root()
    index_path = os.path.join(repo_root, args.index_json) if not os.path.isabs(args.index_json) else args.index_json
    out_dir = os.path.join(repo_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    data = load_index(index_path)
    items = data["items"] if "items" in data else data.get("items", [])
    # Filter items whose b2 exists
    existing_items = []
    for it in items:
        b2_rel = it["b2_npz"]
        b2_abs = os.path.join(repo_root, b2_rel) if not os.path.isabs(b2_rel) else b2_rel
        if os.path.isfile(b2_abs):
            existing_items.append(it)

    # Group by case_id
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for it in existing_items:
        cid = it["case_id"]
        by_case.setdefault(cid, []).append(it)

    cases = sorted(by_case.keys())
    n_cases = len(cases)
    if n_cases == 0:
        raise RuntimeError("No cases/items found. Did C1 produce any b2 files?")

    if not args.allow_few_cases and n_cases < 3:
        raise RuntimeError(f"Only {n_cases} case(s) found. Need >=3 for train/val/test case-wise split. "
                           f"Either build more cases with C1 or re-run with --allow_few_cases.")

    rng = random.Random(args.seed)
    rng.shuffle(cases)

    # Compute counts by case
    def clamp_nonneg(x: int) -> int:
        return max(0, x)

    t_ratio = max(0.0, min(1.0, args.train))
    v_ratio = max(0.0, min(1.0, args.val))
    s_ratio = max(0.0, min(1.0, args.test))
    if (t_ratio + v_ratio + s_ratio) <= 0:
        t_ratio, v_ratio, s_ratio = 0.7, 0.15, 0.15

    n_train = int(n_cases * t_ratio)
    n_val = int(n_cases * v_ratio)
    # Ensure total <= n_cases; put leftovers into test
    if n_train + n_val > n_cases:
        n_val = clamp_nonneg(n_cases - n_train)
    n_test = clamp_nonneg(n_cases - n_train - n_val)

    # Degenerate handling for 1-2 cases
    if args.allow_few_cases:
        if n_cases == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n_cases == 2:
            n_train, n_val, n_test = 1, 1, 0

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:n_train + n_val + n_test]

    # Map to b2 paths (relative to repo root)
    splits = {"train": [], "val": [], "test": []}
    for cid in train_cases:
        for it in by_case[cid]:
            splits["train"].append(it["b2_npz"])
    for cid in val_cases:
        for it in by_case[cid]:
            splits["val"].append(it["b2_npz"])
    for cid in test_cases:
        for it in by_case[cid]:
            splits["test"].append(it["b2_npz"])

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "source_index": os.path.relpath(index_path, repo_root),
        "seed": args.seed,
        "ratios": {"train": t_ratio, "val": v_ratio, "test": s_ratio},
        "n_cases": n_cases,
        "cases_per_split": {"train": len(train_cases), "val": len(val_cases), "test": len(test_cases)},
        "graphs_per_split": {k: len(v) for k, v in splits.items()},
        "case_lists": {"train": train_cases, "val": val_cases, "test": test_cases},
        "note": "Case-wise split. If allow_few_cases is set and n_cases<3, some splits may be empty."
    }

    out_path = os.path.join(out_dir, "splits.json")
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "splits": splits}, f, indent=2)
    print(f"[C2] Wrote {out_path}")
    print(f"[C2] Cases: total={n_cases} train={len(train_cases)} val={len(val_cases)} test={len(test_cases)}")
    print(f"[C2] Graphs: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

if __name__ == "__main__":
    main()
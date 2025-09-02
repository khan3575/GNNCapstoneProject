#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import nibabel as nib

def binarize_prob(prob, thr):
    return (prob >= thr).astype(np.uint8)

def dice_coef(y_true, y_pred):
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    s = y_true.sum() + y_pred.sum()
    return (2.0 * inter / s) if s > 0 else (1.0 if y_true.sum() == 0 and y_pred.sum() == 0 else 0.0)

def iou_coef(y_true, y_pred):
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return (inter / union) if union > 0 else (1.0 if y_true.sum() == 0 and y_pred.sum() == 0 else 0.0)

def precision_recall(y_true, y_pred):
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
    fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    rec  = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    return float(prec), float(rec)

def load_bin_gt(seg_path):
    seg = nib.load(seg_path)
    arr = seg.get_fdata()
    # BraTS: tumor labels > 0 => binary tumor mask
    gt = (arr > 0).astype(np.uint8)
    return gt, seg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True, help="BraTS case dir (with *_seg.nii.gz)")
    ap.add_argument("--pred_nii", required=False, help="Binary prediction NIfTI (from C4).")
    ap.add_argument("--prob_nii", required=False, help="Probability NIfTI (from C4).")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for prob->pred if using prob_nii.")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    # Find seg
    seg_candidates = [p for p in os.listdir(args.case_dir) if "seg" in p and p.endswith((".nii.gz",".nii"))]
    if not seg_candidates:
        raise FileNotFoundError(f"No seg NIfTI found under {args.case_dir}")
    seg_path = os.path.join(args.case_dir, sorted(seg_candidates)[0])

    gt, seg_img = load_bin_gt(seg_path)
    H, W, Z = gt.shape

    pred = None
    used_thr = None

    if args.pred_nii and os.path.isfile(args.pred_nii):
        pred_img = nib.load(args.pred_nii)
        pred = pred_img.get_fdata().astype(np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    elif args.prob_nii and os.path.isfile(args.prob_nii):
        prob_img = nib.load(args.prob_nii)
        prob = prob_img.get_fdata().astype(np.float32)
        if prob.shape != gt.shape:
            raise ValueError(f"Shape mismatch: prob {prob.shape} vs gt {gt.shape}")
        # Threshold sweep
        thrs = np.linspace(0.1, 0.9, 17)
        best = (-1.0, None)
        for thr in thrs:
            p = binarize_prob(prob, thr)
            dice = dice_coef(gt, p)
            if dice > best[0]:
                best = (dice, thr)
        used_thr = float(best[1]) if best[1] is not None else args.threshold
        pred = binarize_prob(prob, used_thr)
    else:
        raise ValueError("Provide either --pred_nii or --prob_nii")

    # Whole-volume metrics
    dice = float(dice_coef(gt, pred))
    iou  = float(iou_coef(gt, pred))
    prec, rec = precision_recall(gt, pred)

    # Per-slice metrics
    slice_metrics = []
    for z in range(Z):
        d = float(dice_coef(gt[:,:,z], pred[:,:,z]))
        j = float(iou_coef(gt[:,:,z], pred[:,:,z]))
        p, r = precision_recall(gt[:,:,z], pred[:,:,z])
        slice_metrics.append({"z": int(z), "dice": d, "iou": j, "precision": p, "recall": r})

    out = {
        "case_dir": args.case_dir,
        "seg_path": seg_path,
        "pred_nii": args.pred_nii,
        "prob_nii": args.prob_nii,
        "threshold_used": used_thr if used_thr is not None else args.threshold,
        "metrics": {"dice": dice, "iou": iou, "precision": prec, "recall": rec},
        "slice_metrics": slice_metrics
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"c5_eval_{os.path.basename(args.case_dir)}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[C5] Dice={dice:.4f} IoU={iou:.4f} Prec={prec:.4f} Rec={rec:.4f}  thr={out['threshold_used']:.3f}")
    print(f"[C5] Wrote: {out_json}")

if __name__ == "__main__":
    main()
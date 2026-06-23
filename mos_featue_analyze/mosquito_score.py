"""
mosquito_score.py — Threshold-based mosquito noise scoring.

Usage:
    python mosquito_score.py                          # 分析所有 CSV
    python mosquito_score.py --threshold 0.45         # 自定义阈值
    python mosquito_score.py -i input.csv             # 对单个 CSV 评分

对每个 3×3 块（9行数据），输出 [0,1] 的连续噪声程度评分。
评分 > threshold 判定为 mosquito 噪声。
"""

import csv, math, sys, argparse
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

def clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def norm(x, lo, hi):
    """Map x from [lo,hi] -> [0,1], clipped."""
    return clip((x - lo) / (hi - lo)) if hi > lo else 0.0


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ============================================================
# Core scoring function
# ============================================================

def mosquito_score(group, params=None):
    """
    Compute mosquito noise score for a 3x3 block (9 rows).

    Args:
        group: list of 9 dicts (CSV rows) or a list of 9 lists of values
        params: dict of tunable parameters (optional)

    Returns:
        dict with final score and all sub-scores
    """
    P = params or {}

    # -- thresholds for normalization --
    mv_lo = P.get("mv_lo", 50)       # mean_var lower bound
    mv_hi = P.get("mv_hi", 2000)     # mean_var upper bound
    maxv_lo = P.get("maxv_lo", 200)
    maxv_hi = P.get("maxv_hi", 5000)
    hv_lo = P.get("hv_lo", 0)
    hv_hi = P.get("hv_hi", 30)
    lap_lo = P.get("lap_lo", 10)
    lap_hi = P.get("lap_hi", 100)

    # -- weights --
    w_var = P.get("w_var", 0.20)
    w_maxvar = P.get("w_maxvar", 0.10)
    w_ring = P.get("w_ring", 0.30)
    w_lap = P.get("w_lap", 0.20)
    w_highvar = P.get("w_highvar", 0.10)
    w_grad = P.get("w_grad", 0.10)

    # -- halo detection (quiet center + active surround) --
    halo_quiet = P.get("halo_quiet", 100)     # center mean_var below this = quiet
    halo_active = P.get("halo_active", 1200)  # any neighbor mean_var above this = active
    halo_max = P.get("halo_max", 0.35)         # max halo boost

    # -- threshold for binary decision --
    decision_th = P.get("threshold", 0.41)

    # ----------------------------------------------------------
    center = group[4] if isinstance(group[4], dict) else group[4]
    if isinstance(center, dict):
        c = {k: safe_float(v) for k, v in center.items()}
        mean_var_all = [safe_float(r["mean_var"]) for r in group]
        max_var_all = [safe_float(r["max_var"]) for r in group]
        ringing_all = [safe_float(r["profile_ringing_max"]) for r in group]
        hvc_all = [safe_float(r["high_var_count"]) for r in group]
        lap_mean_all = [safe_float(r["lap_mean"]) for r in group]
    else:
        # support raw list of values if needed
        c = {"mean_var": group[4][0], "max_var": group[4][1],
             "profile_ringing_max": group[4][2], "lap_mean": group[4][3],
             "high_var_count": group[4][4], "grad_max": group[4][5]}
        mean_var_all = [r[0] for r in group]
        max_var_all = [r[1] for r in group]
        ringing_all = [r[2] for r in group]
        hvc_all = [r[3] for r in group]
        lap_mean_all = [r[4] for r in group]

    # ===================== Sub-scores =====================

    # 1. Activity (variance)
    s_var = norm(c["mean_var"], mv_lo, mv_hi)
    s_maxvar = norm(c["max_var"], maxv_lo, maxv_hi)
    s_highvar = norm(c["high_var_count"], hv_lo, hv_hi)

    # 2. Ringing (mosquito noise signature)
    s_ring = clip(c.get("profile_ringing_max", 0))

    # 3. High-frequency energy (Laplacian)
    s_lap = norm(c["lap_mean"], lap_lo, lap_hi)

    # 4. Gradient
    s_grad = norm(c.get("grad_max", 0), 20, 200)

    # ===================== Base score =====================
    base = (w_var * s_var + w_maxvar * s_maxvar + w_highvar * s_highvar
            + w_ring * s_ring + w_lap * s_lap + w_grad * s_grad)

    # ===================== Context boosts =====================

    # 5. Halo boost: center is quiet but neighbors are very active
    center_quiet = c["mean_var"] < halo_quiet
    surround_active = max(mean_var_all) > halo_active
    halo_boost = 0.0
    if center_quiet and surround_active:
        ratio = max(mean_var_all) / max(c["mean_var"], 1e-8)
        halo_boost = min(halo_max * norm(ratio, 5, 50), halo_max)

    # 6. Neighbor ringing boost: any neighbor has strong ringing
    ring_boost = 0.0
    if max(ringing_all) > 0.8 and s_ring < 0.5:
        ring_boost = 0.10 * norm(max(ringing_all), 0.5, 1.0)

    # 7. Density boost: many high-var blocks in 3x3
    total_hvc = sum(1 for v in hvc_all if v > 0)
    den_boost = 0.05 * norm(total_hvc, 5, 9) if total_hvc >= 5 else 0.0

    # ===================== Context penalty =====================
    # If 3x3 has lots of near-zero cells, the signal might be diluted.
    # We don't penalize, but the base score already reflects this.

    # ===================== Final score =====================
    score = clip(base + halo_boost + ring_boost + den_boost)
    is_mosquito = score >= decision_th

    return {
        "score": score,
        "is_mosquito": is_mosquito,
        # sub-scores for debugging
        "sub_var": s_var,
        "sub_maxvar": s_maxvar,
        "sub_highvar": s_highvar,
        "sub_ringing": s_ring,
        "sub_lap": s_lap,
        "sub_grad": s_grad,
        "base": base,
        "halo_boost": halo_boost,
        "ring_boost": ring_boost,
        "den_boost": den_boost,
        "total_hvc": total_hvc,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(truth_labels, scores, threshold):
    """Return accuracy, FN, FP, etc."""
    dm_tp = sum(1 for l, s in zip(truth_labels, scores)
                if l == 1 and s >= threshold)
    dm_fn = sum(1 for l, s in zip(truth_labels, scores)
                if l == 1 and s < threshold)
    nd_tn = sum(1 for l, s in zip(truth_labels, scores)
                if l == 0 and s < threshold)
    nd_fp = sum(1 for l, s in zip(truth_labels, scores)
                if l == 0 and s >= threshold)
    total = len(truth_labels)
    acc = (dm_tp + nd_tn) / total if total else 0
    prec = dm_tp / (dm_tp + nd_fp) if (dm_tp + nd_fp) else 0
    recall = dm_tp / (dm_tp + dm_fn) if (dm_tp + dm_fn) else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
    return {
        "th": threshold, "acc": acc, "f1": f1,
        "dm_recall": recall, "nd_spec": nd_tn / (nd_tn + nd_fp) if (nd_tn + nd_fp) else 0,
        "fn": dm_fn, "fp": nd_fp,
        "tp": dm_tp, "tn": nd_tn,
        "total_dm": dm_tp + dm_fn, "total_nd": nd_tn + nd_fp,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mosquito noise threshold scoring")
    parser.add_argument("-i", "--input", help="Single CSV file to score (optional)")
    parser.add_argument("--threshold", type=float, default=0.41,
                        help="Decision threshold (default: 0.41)")
    parser.add_argument("--show", action="store_true",
                        help="Show per-block results")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds to find optimal")
    args = parser.parse_args()

    data_dir = Path(__file__).parent
    print("data_dir: ",data_dir)

    # ---- Load ----
    if args.input:
        csv_files = [Path(args.input)]
    else:
        csv_files = sorted(data_dir.glob("*.csv"))
        print("data_dir: ",csv_files)


    all_samples = []
    for path in csv_files:
        name = path.stem.lower()
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for i in range(len(rows) // 9):
            group = rows[i * 9:(i + 1) * 9]
            center = group[4]
            label = 1 if "dm" in name and "not_dm" not in name else (
                0 if "not_dm" in name else -1
            )
            all_samples.append({
                "source": path.name,
                "row": center["row"],
                "col": center["col"],
                "group": group,
                "label": label,
            })

    # ---- Score ----
    for s in all_samples:
        result = mosquito_score(s["group"], {"threshold": args.threshold})
        s["score"] = result["score"]
        s["detected"] = result["is_mosquito"]
        s["_details"] = result

    # ---- Report ----
    labeled = [s for s in all_samples if s["label"] >= 0]
    dm_s = [s["score"] for s in labeled if s["label"] == 1]
    nd_s = [s["score"] for s in labeled if s["label"] == 0]

    print(f"\n=== MOSQUITO NOISE SCORING ===")
    print(f"Samples: {len(all_samples)} total, "
          f"{len(dm_s)} DM, {len(nd_s)} non-DM")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"\nScore distribution:")
    if dm_s:
        print(f"  DM     [{len(dm_s):>2}] mean={sum(dm_s)/len(dm_s):.3f}  "
              f"range=[{min(dm_s):.3f}, {max(dm_s):.3f}]")
    if nd_s:
        print(f"  non-DM [{len(nd_s):>2}] mean={sum(nd_s)/len(nd_s):.3f}  "
              f"range=[{min(nd_s):.3f}, {max(nd_s):.3f}]")

    # ---- Evaluation (if labels available) ----
    if len(labeled) > 0 and any(s["label"] == 1 for s in labeled):
        labels = [s["label"] for s in labeled]
        scores = [s["score"] for s in labeled]

        e = evaluate(labels, scores, args.threshold)
        print(f"\n--- Evaluation at th={args.threshold:.2f} ---")
        print(f"  DM recall:  {e['dm_recall']:.3f} ({e['tp']}/{e['total_dm']})")
        print(f"  non-DM spec:{e['nd_spec']:.3f} ({e['tn']}/{e['total_nd']})")
        print(f"  FN={e['fn']}  FP={e['fp']}  Acc={e['acc']:.3f}  F1={e['f1']:.3f}")

        # ---- Sweep ----
        if args.sweep:
            print(f"\n--- Threshold sweep ---")
            print(f"{'th':>6} {'FN':>4} {'FP':>4} {'DM_rec':>8} {'ND_spec':>8} {'Acc':>6}")
            for th_pct in range(5, 80, 5):
                th = th_pct / 100
                e2 = evaluate(labels, scores, th)
                print(f"{th:.2f}  {e2['fn']:>4} {e2['fp']:>4} "
                      f"{e2['dm_recall']:>8.3f} {e2['nd_spec']:>8.3f} {e2['acc']:>6.3f}")

            # Best F1
            best = max((evaluate(labels, scores, t / 100)
                        for t in range(1, 100)), key=lambda x: x["f1"])
            print(f"\n  Best F1: th={best['th']:.2f}, F1={best['f1']:.3f}, "
                  f"FN={best['fn']}, FP={best['fp']}")

    # ---- Per-block detail ----
    if args.show:
        print(f"\n--- Per-block details ---")
        print(f"{'Source':>20} {'Pos':>10} {'Label':>4} {'Score':>6} "
              f"{'Base':>6} {'Halo':>6} {'RingB':>6} {'Dens':>5} "
              f"{'Detect':>7}")
        print("-" * 80)
        for s in all_samples:
            d = s["_details"]
            lbl = "DM" if s["label"] == 1 else ("ND" if s["label"] == 0 else "?")
            print(f"{s['source']:>20} ({s['row']:>3},{s['col']:>3}) "
                  f"{lbl:>4} {s['score']:>6.3f} "
                  f"{d['base']:>6.3f} {d['halo_boost']:>6.3f} "
                  f"{d['ring_boost']:>6.3f} {d['den_boost']:>5.3f} "
                  f"{'DM!' if s['detected'] else 'ok':>7}")


if __name__ == "__main__":
    main()

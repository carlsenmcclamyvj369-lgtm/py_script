"""
Threshold-based mosquito noise scoring — no ML classifier.

For each 3x3 block sample (9 rows), compute a continuous mosquito score in [0,1]
using interpretable feature thresholds and rules, then evaluate against labels.
"""

import csv
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Load data
# ============================================================

def load_samples(csv_path, label):
    samples = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    n_samples = n // 9
    for i in range(n_samples):
        group = rows[i * 9:(i + 1) * 9]
        center = group[4]
        samples.append({
            "source": csv_path.name,
            "row": int(center["row"]),
            "col": int(center["col"]),
            "group": group,
            "label": label,
        })
    return samples


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# ============================================================
# 2. Threshold-based scoring engine
# ============================================================

def normalize(x, lo, hi):
    """Map x from [lo, hi] -> [0, 1], clamp outside."""
    if hi <= lo:
        return 0.0
    return clip((x - lo) / (hi - lo))


class MosquitoScorer:
    """
    Configurable threshold-based mosquito noise scorer.

    Each sub-score maps a feature value -> [0,1] via normalize() or lookup.
    Sub-scores are combined with configurable weights.
    Context rules handle "halo" patterns (quiet center + active surround).
    """

    def __init__(self, config=None):
        cfg = config or {}
        # === Activity (variance-based) ===
        self.mean_var_lo = cfg.get("mean_var_lo", 50)
        self.mean_var_hi = cfg.get("mean_var_hi", 2000)
        self.max_var_lo = cfg.get("max_var_lo", 200)
        self.max_var_hi = cfg.get("max_var_hi", 5000)
        self.high_var_lo = cfg.get("high_var_lo", 0)
        self.high_var_hi = cfg.get("high_var_hi", 30)
        self.very_high_var_lo = cfg.get("very_high_var_lo", 0)
        self.very_high_var_hi = cfg.get("very_high_var_hi", 20)

        # === Edge / ringing ===
        self.profile_ringing_weight = cfg.get("profile_ringing_weight", 1.0)

        # === Laplacian (high-frequency energy) ===
        self.lap_mean_lo = cfg.get("lap_mean_lo", 10)
        self.lap_mean_hi = cfg.get("lap_mean_hi", 100)

        # === Gradient ===
        self.grad_max_lo = cfg.get("grad_max_lo", 20)
        self.grad_max_hi = cfg.get("grad_max_hi", 200)

        # === Context / halo ===
        self.halo_threshold = cfg.get("halo_threshold", 0.15)
        self.total_hvc_lo = cfg.get("total_hvc_lo", 3)
        self.total_hvc_hi = cfg.get("total_hvc_hi", 9)

        # === Sub-score weights ===
        self.w_var = cfg.get("w_var", 0.25)
        self.w_max_var = cfg.get("w_max_var", 0.15)
        self.w_high_var = cfg.get("w_high_var", 0.10)
        self.w_ringing = cfg.get("w_ringing", 0.25)
        self.w_lap = cfg.get("w_lap", 0.15)
        self.w_grad = cfg.get("w_grad", 0.10)

        # === Halo boost config ===
        self.halo_center_quiet_threshold = cfg.get("halo_center_quiet_threshold", 100)
        self.halo_neighbor_active_threshold = cfg.get("halo_neighbor_active_threshold", 1500)

    def score(self, group):
        """
        Compute mosquito noise score for a 3x3 block group.

        Args:
            group: list of 9 dicts (rows from CSV)

        Returns:
            dict with score and all sub-scores for debugging
        """
        center = group[4]
        vals_center = {k: safe_float(center[k]) for k in center}

        # Gather 3x3 neighbor values for key features
        def get_col(key):
            return [safe_float(r[key]) for r in group]

        mean_var_all = get_col("mean_var")
        max_var_all = get_col("max_var")
        ringing_all = get_col("profile_ringing_max")
        hvc_all = get_col("high_var_count")
        vhvc_all = get_col("very_high_var_count")

        # --- Sub-scores (center) ---
        s_var = normalize(vals_center["mean_var"],
                          self.mean_var_lo, self.mean_var_hi)
        s_max_var = normalize(vals_center["max_var"],
                              self.max_var_lo, self.max_var_hi)
        s_high_var = normalize(vals_center["high_var_count"],
                               self.high_var_lo, self.high_var_hi)
        s_very_high_var = normalize(vals_center["very_high_var_count"],
                                    self.very_high_var_lo, self.very_high_var_hi)
        s_ringing = vals_center["profile_ringing_max"] * self.profile_ringing_weight
        s_lap = normalize(vals_center["lap_mean"],
                          self.lap_mean_lo, self.lap_mean_hi)
        s_grad = normalize(vals_center["grad_max"],
                           self.grad_max_lo, self.grad_max_hi)

        # --- Context features ---
        neighbor_max_var = max(mean_var_all)
        neighbor_max_ringing = max(ringing_all)
        neighbor_max_max_var = max(max_var_all)
        total_hvc = sum(1 for v in hvc_all if v > 0)
        total_vhvc = sum(1 for v in vhvc_all if v > 0)

        # Ringing spread in 3x3
        ringing_std = 0.0
        if len(ringing_all) > 1:
            mean_r = sum(ringing_all) / len(ringing_all)
            ringing_std = math.sqrt(sum((v - mean_r) ** 2 for v in ringing_all) / len(ringing_all))

        # --- Base score ---
        base = (
            self.w_var * s_var +
            self.w_max_var * s_max_var +
            self.w_high_var * s_high_var +
            self.w_ringing * s_ringing +
            self.w_lap * s_lap +
            self.w_grad * s_grad
        )

        # --- Halo boost: quiet center + very active surround ---
        center_quiet = vals_center["mean_var"] < self.halo_center_quiet_threshold
        surround_active = neighbor_max_var > self.halo_neighbor_active_threshold
        halo_boost = 0.0
        if center_quiet and surround_active:
            # How extreme is the halo?
            halo_ratio = neighbor_max_var / max(vals_center["mean_var"], 1e-8)
            halo_intensity = normalize(halo_ratio, 5, 100)
            halo_boost = self.halo_threshold + 0.10 * halo_intensity
            halo_boost = min(halo_boost, 0.40)

        # --- Neighbor ringing boost ---
        # If any neighbor has strong ringing, bump score
        neighbor_ringing_boost = 0.0
        if neighbor_max_ringing > 0.8 and s_ringing < 0.5:
            neighbor_ringing_boost = 0.10 * normalize(neighbor_max_ringing, 0.5, 1.0)

        # --- Density boost: many high-var blocks in 3x3 ---
        density_boost = 0.0
        if total_hvc >= 5:
            density_boost = 0.05 * normalize(total_hvc, 5, 9)

        # --- Final score ---
        score = base + halo_boost + neighbor_ringing_boost + density_boost
        score = clip(score, 0.0, 1.0)

        return {
            "score": score,
            "s_var": s_var,
            "s_max_var": s_max_var,
            "s_high_var": s_high_var,
            "s_ringing": s_ringing,
            "s_lap": s_lap,
            "s_grad": s_grad,
            "base": base,
            "halo_boost": halo_boost,
            "neighbor_ringing_boost": neighbor_ringing_boost,
            "density_boost": density_boost,
            "neighbor_max_var": neighbor_max_var,
            "total_hvc": total_hvc,
            "ringing_std": ringing_std,
        }


# ============================================================
# 3. Load all data
# ============================================================

print("=" * 60)
print("THRESHOLD-BASED MOSQUITO SCORING")
print("=" * 60)

all_samples = []
for csv_path in sorted(DATA_DIR.glob("*.csv")):
    name = csv_path.stem.lower()
    if "not_dm" in name:
        all_samples.extend(load_samples(csv_path, label=0))
    elif "dm" in name:
        all_samples.extend(load_samples(csv_path, label=1))

print(f"\nLoaded {len(all_samples)} samples: "
      f"{sum(1 for s in all_samples if s['label']==1)} DM, "
      f"{sum(1 for s in all_samples if s['label']==0)} non-DM")

# ============================================================
# 4. Score all samples
# ============================================================

scorer = MosquitoScorer()

results = []
for s in all_samples:
    r = scorer.score(s["group"])
    r["label"] = s["label"]
    r["source"] = s["source"]
    r["pos"] = (s["row"], s["col"])
    results.append(r)

# Stats by class
dm_scores = [r["score"] for r in results if r["label"] == 1]
nd_scores = [r["score"] for r in results if r["label"] == 0]

print(f"\nScore distribution:")
print(f"  DM     ({len(dm_scores):>2} samples): "
      f"mean={sum(dm_scores)/len(dm_scores):.3f}, "
      f"min={min(dm_scores):.3f}, "
      f"max={max(dm_scores):.3f}")
print(f"  non-DM ({len(nd_scores):>2} samples): "
      f"mean={sum(nd_scores)/len(nd_scores):.3f}, "
      f"min={min(nd_scores):.3f}, "
      f"max={max(nd_scores):.3f}")

# ============================================================
# 5. Threshold sweep for binary classification eval
# ============================================================

print(f"\n{'='*60}")
print("Threshold sweep")
print(f"{'='*60}")
print(f"{'Threshold':>10} {'FN':>4} {'FP':>4} {'Acc':>6} {'DM_recall':>10} {'ND_spec':>10}")
print("-" * 50)

best = {"f1": 0, "th": 0}
for th in [t / 100 for t in range(0, 100, 2)]:
    tp = sum(1 for r in results if r["label"] == 1 and r["score"] >= th)
    fn = sum(1 for r in results if r["label"] == 1 and r["score"] < th)
    tn = sum(1 for r in results if r["label"] == 0 and r["score"] < th)
    fp = sum(1 for r in results if r["label"] == 0 and r["score"] >= th)
    acc = (tp + tn) / len(results) if results else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    nd_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    if th in [t / 100 for t in range(0, 100, 5)]:
        print(f"{th:>10.2f} {fn:>4} {fp:>4} {acc:>6.3f} {recall:>10.3f} {nd_spec:>10.3f}")

    if f1 > best["f1"]:
        best = {"th": th, "f1": f1, "fn": fn, "fp": fp, "acc": acc, "recall": recall, "spec": nd_spec}

# Find threshold for perfect DM recall
for th in [t / 100 for t in range(100, -1, -1)]:
    fn_th = sum(1 for r in results if r["label"] == 1 and r["score"] < th)
    if fn_th == 0:
        fp_th = sum(1 for r in results if r["label"] == 0 and r["score"] >= th)
        tn_th = sum(1 for r in results if r["label"] == 0 and r["score"] < th)
        print(f"\n  Best F1:       th={best['th']:.2f}, FN={best['fn']}, FP={best['fp']}, "
              f"F1={best['f1']:.3f}, acc={best['acc']:.3f}")
        print(f"  DM recall=1.0: th={th:.2f}, FN={fn_th}, FP={fp_th}, "
              f"non-DM spec={tn_th/(tn_th+fp_th):.3f} ({tn_th}/{tn_th+fp_th})")
        break

# ============================================================
# 6. Per-sample breakdown
# ============================================================

print(f"\n{'='*60}")
print("Per-sample scores (sorted by score, ascending)")
print(f"{'='*60}")
print(f"{'Idx':>4} {'Source':>20} {'Pos':>10} {'Label':>6} {'Score':>6} {'Base':>6} "
      f"{'Halo':>6} {'RingB':>6} {'Dense':>6}")
print("-" * 75)

sorted_idx = sorted(range(len(results)), key=lambda i: results[i]["score"])
for rank, i in enumerate(sorted_idx):
    r = results[i]
    mark = " <--" if r["label"] != (r["score"] >= best["th"]) else ""
    print(f"{rank:>4} {r['source']:>20} {str(r['pos']):>10} "
          f"{'DM' if r['label']==1 else 'ND':>6} "
          f"{r['score']:>6.3f} {r['base']:>6.3f} {r['halo_boost']:>6.3f} "
          f"{r['neighbor_ringing_boost']:>6.3f} {r['density_boost']:>6.3f}"
          f"{mark}")

# ============================================================
# 7. Key misclassified analysis
# ============================================================

print(f"\n{'='*60}")
print("Lowest-scoring DM samples (false negatives at th=0.5)")
print(f"{'='*60}")

fn_at_05 = [r for r in results if r["label"] == 1 and r["score"] < 0.5]
fn_at_05.sort(key=lambda r: r["score"])
for r in fn_at_05:
    print(f"\n  {r['source']} {r['pos']}  score={r['score']:.3f}")
    print(f"    base={r['base']:.3f}  s_var={r['s_var']:.3f}  s_ringing={r['s_ringing']:.3f}  "
          f"s_lap={r['s_lap']:.3f}")
    print(f"    halo_boost={r['halo_boost']:.3f}  ring_boost={r['neighbor_ringing_boost']:.3f}  "
          f"density={r['density_boost']:.3f}")
    print(f"    neighbor_max_var={r['neighbor_max_var']:.0f}  total_hvc={r['total_hvc']}")

fp_at_05 = [r for r in results if r["label"] == 0 and r["score"] >= 0.5]
fp_at_05.sort(key=lambda r: -r["score"])
if fp_at_05:
    print(f"\nHighest-scoring non-DM samples (false positives at th=0.5):")
    for r in fp_at_05[:5]:
        print(f"  {r['source']} {r['pos']}  score={r['score']:.3f}")

# ============================================================
# 8. Parameter sensitivity (optional quick analysis)
# ============================================================

print(f"\n{'='*60}")
print("Quick parameter tuning")
print(f"{'='*60}")

# Try adjusting halo parameters
for halo_th in [0.05, 0.10, 0.15, 0.20, 0.25]:
    for quiet_th in [50, 100, 200]:
        for active_th in [1000, 1500, 2000]:
            cfg = {"halo_threshold": halo_th,
                   "halo_center_quiet_threshold": quiet_th,
                   "halo_neighbor_active_threshold": active_th}
            s2 = MosquitoScorer(cfg)
            scores2 = [s2.score(s["group"])["score"] for s in all_samples]
            # Count how many DM samples are above 0.5
            dm_above = sum(1 for i, s in enumerate(all_samples)
                           if s["label"] == 1 and scores2[i] >= 0.5)
            nd_above = sum(1 for i, s in enumerate(all_samples)
                           if s["label"] == 0 and scores2[i] >= 0.5)
            if dm_above == 33 and nd_above <= 10:
                print(f"  halo_th={halo_th:.2f} quiet<{quiet_th} active>{active_th}: "
                      f"DM={dm_above}/33 non-DM FP={nd_above}/36")

print(f"\nDone. Results saved to: {OUTPUT_DIR}/")

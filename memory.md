---
name: mosquito-noise-detector
description: Mosquito noise (Gibbs ringing) detection algorithm design and key parameters
metadata:
  type: project
---

# Mosquito Noise Detection — design notes

## Core approach
Per-patch (8×8) detection using edge normal profiles. Three feature families aggregated to patch level and thresholded.

## Key files
- `dct_show/gibs_hardware.py` — main single-image vectorized detector
- `batch_gibs_hardware.py` — batch version (sync all changes from gibs_hardware.py)
- `dct_show/inspect_features.py` — interactive click-to-inspect tool
- `dct_show/jibs.py` — original per-patch reference implementation

## Current parameters (gibs_hardware.py)
- `patch_size=5, stride=5, profile_radius=4`
- Canny thresholds: 100, 200
- Oscillation valid range: 0.5%~15% of edge step height
- Profile: integer direction quantization (4 directions), integer indexing, no interpolation

## Feature C: Gibbs score per edge point
`gibbs = residual_energy * oscillation_score * decay_from_prof`

- **residual_energy**: mean |profile residual| (profile_gray - smooth(profile_gray))
- **oscillation_score**: sign_alternation_score (fraction of adjacent non-zero signs that differ)
- **decay_from_prof**: `tanh(far_g / (near_g + 1e-6) / 3.0)` — far-to-near energy ratio

## Thresholds for final mask
- `loc_ratio > 1.5`
- `crossings > 1.2`
- `decay > 0.4`
- `gibbs_p90 > 3`

## History
- Switched from global MAD threshold to per-profile mean, then to 2%-20% Gibbs amplitude rule
- Reversed near/far ratio direction in decay score
- Added gradient magnitude suppression (later removed)
- Simplified oscillation from 3 variables to just sign_alternation
- Simplified decay from 3 variables to just near/far energy ratio

## Known issues
- High false positives on straight lines, buildings, grass
- Canny thresholds 100/200 may miss subtle ringing edges
- Profile radius 4 → 9-point profile, distance bands need adjustment

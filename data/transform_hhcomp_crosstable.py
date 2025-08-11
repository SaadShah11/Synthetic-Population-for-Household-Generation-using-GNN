import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def parse_col(col: str) -> Tuple[str, str, str]:
    """
    Parse a column name like "M 0_4 1PE" → (sex, age, hhcode).
    Returns ("", "", "") if it doesn't match the expected 3-part pattern.
    """
    parts = str(col).split(" ")
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def largest_remainder_rounding(values: np.ndarray, target_sum: int) -> np.ndarray:
    """
    Round a vector of non-negative floats to integers that sum exactly to target_sum.
    Uses largest-remainder method.
    """
    if values.sum() <= 0 or target_sum <= 0:
        return np.zeros_like(values, dtype=int)

    scaled = values * (target_sum / values.sum())
    base = np.floor(scaled).astype(int)
    remainder = scaled - base
    short = int(target_sum - base.sum())
    if short > 0:
        idx = np.argsort(-remainder)[:short]
        base[idx] += 1
    return base


def compute_sex_prior(row: pd.Series, sex: str, ages: List[str], codes: List[str]) -> np.ndarray:
    """
    Aggregate counts across ages for a given sex → prior over hhcomp codes.
    Returns a probability vector of length len(codes) (sums to 1, or uniform if no mass).
    """
    totals = np.zeros(len(codes), dtype=float)
    code_index = {c: i for i, c in enumerate(codes)}
    for age in ages:
        for code in codes:
            col = f"{sex} {age} {code}"
            if col in row:
                totals[code_index[code]] += float(row[col])
    s = totals.sum()
    if s <= 0:
        return np.full_like(totals, 1.0 / len(codes))
    return totals / s


def smooth_row_gpu(row: pd.Series,
                   sexes: List[str],
                   ages: List[str],
                   codes: List[str],
                   wide_cols: List[str],
                   lam: float,
                   alpha: float,
                   device: torch.device) -> pd.Series:
    """
    Vectorized smoothing using torch on GPU when available.
    Preserves each (sex, age) total via largest-remainder rounding.
    Operates over the [S, A, C] cube constructed from wide_cols.
    """
    out = row.copy()
    S, A, C = len(sexes), len(ages), len(codes)

    # Build tensor [S, A, C] for this row
    wide_vals = row[wide_cols].to_numpy(dtype=float)
    if wide_vals.size != S * A * C:
        # If some columns are missing, pad and fill selectively
        buffer = np.zeros(S * A * C, dtype=float)
        col_to_idx: Dict[str, int] = {col: i for i, col in enumerate(wide_cols)}
        # Already aligned, but keep guard
        buffer[:wide_vals.size] = wide_vals
        wide_vals = buffer
    counts = torch.as_tensor(wide_vals, dtype=torch.float32, device=device).view(S, A, C)

    # Compute per-sex prior over codes: sum across ages
    prior_raw = counts.sum(dim=1)  # [S, C]
    # Dirichlet-like smoothing of the prior
    prior = prior_raw + (alpha / C)
    prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-12)

    # For each (s, a): v_smooth = (1-lam)*v + lam * N_sa * prior[s]
    N_sa = counts.sum(dim=2, keepdim=True)  # [S, A, 1]
    v_prior = prior.unsqueeze(1).expand(S, A, C)  # [S, A, C]
    v_smooth = (1.0 - lam) * counts + lam * (N_sa * v_prior)

    # Largest-remainder rounding per (s,a)
    base = torch.floor(v_smooth)
    remainder = v_smooth - base
    base_sum = base.sum(dim=2, keepdim=True)  # [S, A, 1]
    short = (N_sa - base_sum).to(torch.int64).squeeze(-1)  # [S, A]

    # For each (s,a), add 1 to the top-k remainders
    # Flatten last dim; use topk along C
    topk_vals, topk_idx = torch.topk(remainder, k=min(C, int(C)), dim=2)  # [S,A,C]
    # Build an index mask of size [S,A,C] with ones for the first short[s,a] positions in topk order
    S_idx, A_idx = torch.meshgrid(torch.arange(S, device=device), torch.arange(A, device=device), indexing='ij')
    add_mask = torch.zeros_like(base, dtype=torch.int64)
    # Iterate over (s,a) on device (small loops only over S and A)
    for s in range(S):
        for a in range(A):
            k = int(short[s, a].item())
            if k <= 0:
                continue
            idxs = topk_idx[s, a, :k]
            add_mask[s, a, idxs] = 1

    v_int = base.to(torch.int64) + add_mask

    # Write back to row
    flat = v_int.view(-1).to('cpu').numpy().astype(int)
    out[wide_cols] = flat[:len(wide_cols)]
    return out


def detect_dimensions(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Infer the lists of (sexes, ages, hhcomp codes) from the wide crosstable columns.
    """
    sexes, ages, codes = set(), set(), set()
    for col in df.columns:
        s, a, c = parse_col(col)
        if s and a and c:
            sexes.add(s)
            ages.add(a)
            codes.add(c)
    # Keep stable ordering by sorting ages as they appear in file if possible; fallback to lexicographic
    # Try common age order from your pipeline
    common_age_order = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34',
                        '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    ages_sorted = [a for a in common_age_order if a in ages] + [a for a in sorted(ages) if a not in common_age_order]
    return (sorted(sexes), ages_sorted, sorted(codes))


def main():
    parser = argparse.ArgumentParser(
        description="Smooth HH composition by sex-by-age crosstable to reduce sparsity and improve distributional accuracy."
    )
    parser.add_argument(
        "--input",
        default=r"C:\\Development\\Thesis\\Synthetic Population for Household Generation using GNN\\data\\preprocessed-data\\crosstables\\HH_composition_by_age_by_sex_Main_Modified.csv",
        help="Path to the input HH composition crosstable (wide format)."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the smoothed crosstable. Defaults to <input_basename>_Smoothed.csv in the same folder."
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.35,
        help="Interpolation strength toward per-sex prior (0=no change, 1=prior only)."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Dirichlet-style smoothing parameter for the per-sex prior."
    )
    args = parser.parse_args()

    inp = args.input
    out = args.output
    lam = float(args.lam)
    alpha = float(args.alpha)

    if out is None:
        root, ext = os.path.splitext(inp)
        out = root + "_Smoothed.csv"

    print(f"Reading: {inp}")
    df = pd.read_csv(inp)

    # Torch device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Detect schema
    sexes, ages, codes = detect_dimensions(df)
    print(f"Detected sexes={sexes}, ages={len(ages)} groups, hhcomp_codes={len(codes)}")

    # Identify passthrough columns
    passthrough_cols = []
    for col in df.columns:
        s, a, c = parse_col(col)
        if not (s and a and c):
            passthrough_cols.append(col)

    # Build ordered wide columns once (existing only)
    wide_cols: List[str] = []
    for sex in sexes:
        for age in ages:
            for code in codes:
                col = f"{sex} {age} {code}"
                if col in df.columns:
                    wide_cols.append(col)

    # Smooth each row independently (per area)
    out_rows = []
    for idx, row in df.iterrows():
        smoothed = smooth_row_gpu(row, sexes, ages, codes, wide_cols, lam=lam, alpha=alpha, device=device)
        out_rows.append(smoothed)

    out_df = pd.DataFrame(out_rows)

    # Optional: re-order columns to keep passthrough first, then wide block in deterministic order
    # Reuse precomputed wide_cols for ordering
    ordered_cols = passthrough_cols + [c for c in wide_cols if c not in passthrough_cols]
    out_df = out_df[ordered_cols]

    # Verify per-(sex, age) totals preserved
    if len(df) > 0:
        original = df.iloc[0]
        smoothed = out_df.iloc[0]
        for sex in sexes:
            for age in ages:
                orig_total = 0
                new_total = 0
                for code in codes:
                    col = f"{sex} {age} {code}"
                    if col in df.columns:
                        orig_total += int(original.get(col, 0))
                    if col in out_df.columns:
                        new_total += int(smoothed.get(col, 0))
                if orig_total != new_total:
                    print(f"Warning: total changed for ({sex}, {age}) from {orig_total} → {new_total}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()




# db_gcn_patch.py
# Drop this file next to your main script, then add:
#   from db_gcn_patch import ensure_dirs, get_k_values, compute_theta, postprocess_to_excel
# Right after you parse args and define OutputDir, call:
#   DIRS = ensure_dirs(args.output_dir, args.seed, args.num_folds,
#                      getattr(args, "split_scope", "global"),
#                      getattr(args, "database", ""))
#   splits_dir = DIRS["splits"]   # <- fixes NameError in your code
#
# Optional: At the very end of your script (or after runs) you can call:
#   postprocess_to_excel(args.output_dir)
#
# This keeps folds identical for all runs with same (seed, num_folds, split_scope),
# and ensures per-fold gene lists are cached under OutputDir/genes/<tag>/...

from __future__ import annotations
import os
from typing import Dict, List

def _tag(seed: int, num_folds: int, split_scope: str = "global", database: str = "") -> str:
    split_scope = (split_scope or "global").lower().strip()
    if split_scope == "perdb" and database:
        return f"{database}_seed{int(seed)}_k{int(num_folds)}"
    return f"seed{int(seed)}_k{int(num_folds)}"

def ensure_dirs(output_dir: str, seed: int, num_folds: int,
                split_scope: str = "global", database: str = "") -> Dict[str, str]:
    """
    Creates and returns canonical folders used across *all* experiments so that
    folds & gene lists are reused consistently:
      Output/
        ├─ splits/single_<tag>.npz         (stratified folds; reused everywhere)
        ├─ genes/<tag>/G{N}_fold{i}.txt    (per-fold TRAIN-ONLY gene lists; reused)
        └─ runs/<timestamped>              (mirrors of artifacts from each run)
    """
    out = os.path.abspath(output_dir)
    os.makedirs(out, exist_ok=True)

    sd = os.path.join(out, "splits")
    gd = os.path.join(out, "genes", _tag(seed, num_folds, split_scope, database))
    rd = os.path.join(out, "runs")
    for d in (sd, gd, rd):
        os.makedirs(d, exist_ok=True)

    # canonical split filename used by your code
    split_tag = "" if (split_scope or "global").lower() == "global" else f"{database}_"
    split_file = os.path.join(sd, f"single_{split_tag}seed{int(seed)}_k{int(num_folds)}.npz")

    return {"root": out, "splits": sd, "genes": gd, "runs": rd, "split_file": split_file}

# --- Filters & K grid (you can adjust to taste) ---
def get_k_values(filter_type: str) -> List[int]:
    k_values_map = {
        "all":          [1, 2, 4, 6, 8, 10],
        "low":          [6, 8, 10, 12, 14],
        "high":         [6, 8, 10, 12, 14],
        "impulse_low":  [4, 6, 8, 10],
        "impulse_high": [4, 6, 8, 10],
        "band":         [8, 10, 12, 14, 16],      # (even is typical for symmetry)
        "band_reject":  [10, 12, 14, 16, 18],
        "comb":         [12, 14, 16, 18, 20],
    }
    ft = str(filter_type).strip().lower()
    if ft not in k_values_map:
        raise ValueError(f"Invalid filter type: {filter_type}")
    return k_values_map[ft]

def compute_theta(k: int, filter_type: str):
    if int(k) < 1:
        raise ValueError(f"K must be >= 1 (got {k})")
    ft = str(filter_type).strip().lower()
    if ft == "low":
        theta = [1 - i / k for i in range(k + 1)]
    elif ft == "high":
        theta = [i / k for i in range(k + 1)]
    elif ft == "band":
        theta = [0] * (k + 1); theta[k // 2] = 1
    elif ft == "band_reject":
        theta = [1 if i % 2 == 0 else 0 for i in range(k + 1)]
    elif ft == "impulse_low":
        theta = [1] + [0] * k
    elif ft == "impulse_high":
        theta = [0] * k + [1]
    elif ft == "all":
        theta = [1] * (k + 1)
    elif ft == "comb":
        theta = [(-1) ** i for i in range(k + 1)]
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    assert len(theta) == k + 1, f"compute_theta length mismatch for K={k}"
    return theta

# --- Excel post-processing (mean ± std for reviewers) ---
def _pm(mu, sd):
    if (mu is None) or (sd is None) or (not np.isfinite(mu)) or (not np.isfinite(sd)):
        return ""
    return f"{mu:.4f} ± {sd:.4f}"

def postprocess_to_excel(output_dir: str,
                         results_csv: str = "00finalcoexsingle2000.csv",
                         per_fold_csv: str = "per_fold_metrics.csv"):
    """
    Reads your CSVs and writes:
      - results.xlsx -> 'single_summary' and 'single_std' with mean ± std columns
      - folds.xlsx   -> 'single_folds_long' and (if possible) 'single_folds_wide'
    """
    out = os.path.abspath(output_dir)
    res_csv = os.path.join(out, results_csv)
    fold_csv = os.path.join(out, per_fold_csv)

    # Load safely
    df_summary = pd.read_csv(res_csv) if (os.path.exists(res_csv) and os.path.getsize(res_csv) > 0) else pd.DataFrame()
    df_folds   = pd.read_csv(fold_csv) if (os.path.exists(fold_csv) and os.path.getsize(fold_csv) > 0) else pd.DataFrame()

    # Build compact "std" table
    keep = [
        "database","cv_mode","stream_mode","num_omic","epochs","num_genes",
        "filter_type","k","batch_size","dropout","lr",
        "acc_mean","acc_std",
        "macro_precision_mean","macro_precision_std",
        "macro_recall_mean","macro_recall_std",
        "macro_f1_mean","macro_f1_std",
        "run_time_sec","peak_rss_mb","peak_vram_mb",
    ]
    for c in keep:
        if c not in df_summary.columns:
            df_summary[c] = np.nan
    std_tbl = df_summary[keep].copy()
    std_tbl["acc_mean±std"]    = std_tbl.apply(lambda r: _pm(r["acc_mean"], r["acc_std"]), axis=1)
    std_tbl["prec_mean±std"]   = std_tbl.apply(lambda r: _pm(r["macro_precision_mean"], r["macro_precision_std"]), axis=1)
    std_tbl["recall_mean±std"] = std_tbl.apply(lambda r: _pm(r["macro_recall_mean"], r["macro_recall_std"]), axis=1)
    std_tbl["f1_mean±std"]     = std_tbl.apply(lambda r: _pm(r["macro_f1_mean"], r["macro_f1_std"]), axis=1)

    # Write results.xlsx
    res_xlsx = os.path.join(out, "results.xlsx")
    with pd.ExcelWriter(res_xlsx, engine="openpyxl") as w:
        df_summary.to_excel(w, index=False, sheet_name="single_summary")
        std_tbl.to_excel(w, index=False, sheet_name="single_std")

    # Folds workbook
    folds_xlsx = os.path.join(out, "folds.xlsx")
    with pd.ExcelWriter(folds_xlsx, engine="openpyxl") as w:
        df_folds.to_excel(w, index=False, sheet_name="single_folds_long")
        try:
            # Optional wide view
            if not df_folds.empty and {"fold","accuracy"}.issubset(df_folds.columns):
                idx_cols = [c for c in ["database","stream_mode","num_omic","filter_type","k",
                                        "num_genes","batch_size","dropout","lr","cv_mode"] if c in df_folds.columns]
                wide = df_folds.pivot_table(index=idx_cols, columns="fold",
                                            values=["accuracy","macro_precision","macro_recall"],
                                            aggfunc="first").sort_index()
                wide.columns = [f"{met}_fold{fold}" for met, fold in wide.columns]
                wide = wide.reset_index()
                wide.to_excel(w, index=False, sheet_name="single_folds_wide")
        except Exception as e:
            # leave a note inside the workbook
            note = pd.DataFrame([{"info": f"wide view skipped: {e}"}])
            note.to_excel(w, index=False, sheet_name="notes")
    return {"results_xlsx": res_xlsx, "folds_xlsx": folds_xlsx}

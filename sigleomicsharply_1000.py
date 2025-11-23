#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Must be set before PyTorch/ cuBLAS is initialized
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
import argparse
import warnings
# ... (rest of stdlib imports)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sys, builtins
warnings.filterwarnings("ignore")
try:
    import seaborn as sns
except Exception:
    sns = None

# -------------------- import local libs --------------------
# -------------------- import local libs (robust) --------------------
import os, sys
from pathlib import Path

# Resolve directories relative to THIS file (best practice)
ROOT = Path(__file__).resolve().parent
LIB  = ROOT / "sharplylib"

# Prepend both ROOT and lib/ to sys.path (avoid duplicates)
for p in (str(ROOT), str(LIB)):
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"[Paths] added -> ROOT={ROOT}")
print(f"[Paths] added -> LIB ={LIB}")

from coarsening import graph_laplacian, rescale_to_unit as rescale_to_unit_interval
from layermodel import Graph_GCN
import utilsdata
from torch.utils.data import WeightedRandomSampler
from layermodel import Graph_GCN
import utilsdata

# =========================================================
#                     Core helpers
# =========================================================
# --- Make printing robust on Windows consoles that aren't UTF-8 ---
import sys, builtins
try:
    # Prefer UTF-8 for stdout/stderr if supported (Py3.7+)
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Fallback: if the console can't encode emojis, drop un-encodable chars instead of crashing
__orig_print = print
def print(*args, **kwargs):
    try:
        return __orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
        safe_args = tuple(
            a.encode(enc, errors="ignore").decode(enc) if isinstance(a, str) else a
            for a in args
        )
        return __orig_print(*safe_args, **kwargs)

def train_model(train_loader, net, optimizer, criterion, device, dropout_value, L, theta):
    net.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, _, logits, _ = net(batch_x, dropout_value, L, theta)  # logits
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

def test_model(loader, net, device, L, theta):
    net.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            _, _, logits, _ = net(batch_x, 0.0, L, theta)
            y_true.extend(batch_y.cpu().numpy())
            y_scores.extend(logits.softmax(dim=1).cpu().numpy())
    return np.array(y_true), np.array(y_scores)

def _normf(x, nd=8):
    try: return round(float(x), nd)
    except Exception: return x

# ===== Excel helpers: safe append to sheets without losing past runs =====
from pathlib import Path

from pathlib import Path

def _append_df_to_excel(file_path: str, sheet_name: str, df_new: pd.DataFrame):
    file_path = str(file_path)
    if not isinstance(df_new, pd.DataFrame) or df_new.empty:
        return
    if os.path.exists(file_path):
        try:
            df_old = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        except Exception:
            df_old = pd.DataFrame()
    else:
        df_old = pd.DataFrame()
    if not df_old.empty:
        all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new.copy()
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            df_out.to_excel(w, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl") as w:
            df_out.to_excel(w, index=False, sheet_name=sheet_name)

def export_runtime_memory_by_k(df: pd.DataFrame, out_dir: str, tag: str = ""):
    """
    Summarize inner/single fold runtime & memory vs K and emit small figures.
    Expects columns at least: database, stream_mode, num_omic, filter_type, k,
    and some of time_sec, rss_mb, vram_mb (will ignore missing ones).
    """
    if df is None or df.empty:
        print("⚠️ No runtime rows to summarize."); return
    os.makedirs(out_dir, exist_ok=True)

    # Keep only relevant columns if present
    keep = [c for c in ["database","stream_mode","num_omic","filter_type","k",
                        "time_sec","rss_mb","vram_mb"] if c in df.columns]
    d = df[keep].copy()

    # Ensure numerics
    for c in ["k","time_sec","rss_mb","vram_mb","num_omic"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Aggregate by (db, stream, omic, filter, K)
    grp_cols = [c for c in ["database","stream_mode","num_omic","filter_type","k"] if c in d.columns]
    if not grp_cols:
        print("⚠️ Missing grouping columns for runtime summary."); return

    agg = (d.groupby(grp_cols, dropna=False)
             .agg(time_sec_mean=("time_sec","mean") if "time_sec" in d.columns else ("k","size"),
                  time_sec_std =("time_sec","std")  if "time_sec" in d.columns else ("k","size"),
                  rss_mb_mean  =("rss_mb","mean")   if "rss_mb"   in d.columns else ("k","size"),
                  rss_mb_std   =("rss_mb","std")    if "rss_mb"   in d.columns else ("k","size"),
                  vram_mb_mean =("vram_mb","mean")  if "vram_mb"  in d.columns else ("k","size"),
                  vram_mb_std  =("vram_mb","std")   if "vram_mb"  in d.columns else ("k","size"),
                  n=("k","count"))
             .reset_index())

    csv_out = os.path.join(out_dir, f"runtime_by_k{('_'+tag) if tag else ''}.csv")
    agg.to_csv(csv_out, index=False); print(f"✅ Runtime-by-K CSV: {csv_out}")

    # One set of figures per (db, stream, omic)
    triples = [c for c in ["database","stream_mode","num_omic"] if c in agg.columns]
    if not triples:
        triples = [None]

    for key, sub in (agg.groupby(triples) if triples != [None] else [(None, agg)]):
        # Time vs K
        if "time_sec_mean" in sub.columns and sub["time_sec_mean"].notna().any():
            plt.figure(figsize=(8,6))
            for ft, g in sub.groupby("filter_type"):
                g = g.sort_values("k")
                plt.plot(g["k"], g["time_sec_mean"], marker="o", label=str(ft))
            plt.xlabel("K"); plt.ylabel("Inner fold time (s)")
            plt.grid(True, ls="--", alpha=0.6)
            plt.legend(title="Filter", loc="best")
            plt.tight_layout()
            key_tag = key if key is None else "_".join(str(x) for x in key)
            plt.savefig(os.path.join(out_dir, f"runtime_vs_k_{key_tag or 'all'}.pdf"),
                        dpi=600, bbox_inches="tight")
            plt.close()

        # Memory vs K
        if ("rss_mb_mean" in sub.columns and sub["rss_mb_mean"].notna().any()) or \
           ("vram_mb_mean" in sub.columns and sub["vram_mb_mean"].notna().any()):
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6), sharex=True)
            for ft, g in sub.groupby("filter_type"):
                g = g.sort_values("k")
                if "rss_mb_mean" in g.columns and g["rss_mb_mean"].notna().any():
                    ax1.plot(g["k"], g["rss_mb_mean"],  marker="o", label=str(ft))
                if "vram_mb_mean" in g.columns and g["vram_mb_mean"].notna().any():
                    ax2.plot(g["k"], g["vram_mb_mean"], marker="o", label=str(ft))
            ax1.set_title("Host RSS (MB)"); ax2.set_title("GPU VRAM (MB)")
            for ax in (ax1, ax2):
                ax.set_xlabel("K"); ax.set_ylabel("MB"); ax.grid(True, ls="--", alpha=0.6)
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, title="Filter", loc="lower center", ncol=4)
            fig.tight_layout(rect=[0,0.05,1,1])
            key_tag = key if key is None else "_".join(str(x) for x in key)
            fig.savefig(os.path.join(out_dir, f"memory_vs_k_{key_tag or 'all'}.pdf"),
                        dpi=600, bbox_inches="tight")
            plt.close(fig)

# --------- plotting (PDF) ----------
def ensure_pdf(save_path: str, fallback_name: str, OutputDir: str) -> str:
    if not save_path:
        save_path = os.path.join(OutputDir, fallback_name)
    base, _ = os.path.splitext(save_path)
    return base + ".pdf"
# Safe import so script can run without seaborn

def _append_csv(csv_path: str, df_new: pd.DataFrame):
    """Append df_new rows to a CSV file (create if missing) without duplicates."""
    if df_new is None or df_new.empty: 
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df_old = pd.read_csv(csv_path)
        except Exception:
            df_old = pd.DataFrame()
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new.copy()
    df_out.to_csv(csv_path, index=False)

def generate_confusion_matrix(y_true, y_pred, labels=None, OutputDir=".", save_path=None):
    if sns is None:
        print("⚠️ Skipping confusion matrix: seaborn not available.")
        return

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Auto-derive labels if not provided
    if labels is None:
        if y_true.size == 0 and y_pred.size == 0:
            print("⚠️ Skipping confusion matrix: no data.")
            return
        labels = sorted(np.unique(np.concatenate([y_true.ravel(), y_pred.ravel()])))

    if len(labels) == 0:
        print("⚠️ Skipping confusion matrix: no labels/classes found.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Row-normalize safely (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

    sns.set_context("paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[str(x) for x in labels],
        yticklabels=[str(x) for x in labels],
        annot_kws={"size": 9, "weight": "bold"},
        linewidths=0.8,
        linecolor="black",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = ensure_pdf(save_path, "confusion_matrix.pdf", OutputDir)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Confusion matrix saved to {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, num_classes, OutputDir, save_path=None):
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', linewidth=1.5)
    plt.plot([0, 1], [0, 1], 'k--', label="Random", linewidth=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', ncol=1)
    plt.grid(True, alpha=0.3)
    save_path = ensure_pdf(save_path, "roc_curve.pdf", OutputDir)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved to {save_path}")

def compute_theta(k, filter_type):
    if k < 1:
        raise ValueError(f"❌ K must be at least 1. Given: {k}")
    if filter_type == "low":
        theta = [1 - i / k for i in range(k + 1)]
    elif filter_type == "high":
        theta = [i / k for i in range(k + 1)]
    elif filter_type == "band":
        theta = [0] * (k + 1); theta[k // 2] = 1
    elif filter_type == "band_reject":
        # align to paper: alternating +/- emphasis
        theta = [(-1) ** i for i in range(k + 1)]
    elif filter_type == "impulse_low":
        theta = [1] + [0] * k
    elif filter_type == "impulse_high":
        theta = [0] * k + [1]
    elif filter_type == "all":
        theta = [1] * (k + 1)
    elif filter_type == "comb":
        # (optional) repurpose comb to 'even mask' if you still want it
        theta = [1 if i % 2 == 0 else 0 for i in range(k + 1)]
    else:
        raise ValueError(f"❌ Unknown filter type: {filter_type}")
    assert len(theta) == k + 1
    return theta


# =========================================================
#                         CLI
# =========================================================

parser = argparse.ArgumentParser()

# General
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_gene', type=int, default=2000)
parser.add_argument('--num_omic', type=int, default=2)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--num_folds', type=int, default=5)
parser.add_argument('--database', type=str, default='biogrid',
                    choices=['biogrid', 'string', 'coexpression'])

# Flags
parser.add_argument('--singleton', dest='singleton', action='store_true')
parser.add_argument('--no-singleton', dest='singleton', action='store_false'); parser.set_defaults(singleton=True)
parser.add_argument('--savemodel', dest='savemodel', action='store_true')
parser.add_argument('--no-savemodel', dest='savemodel', action='store_false'); parser.set_defaults(savemodel=False)
parser.add_argument('--loaddata', dest='loaddata', action='store_true')
parser.add_argument('--no-loaddata', dest='loaddata', action='store_false'); parser.set_defaults(loaddata=True)

# Optimizer / regularization
parser.add_argument(
    '--weight_decay', type=float, default=5e-4,
    help='L2 weight decay for AdamW (typical: 5e-4; try 1e-4 if underfitting).'
)
parser.add_argument('--num_selected_genes', type=int, default=10)

# Filters / model stream
parser.add_argument('--filter_type', type=str, default='impulse_low',
                    choices=['low','high','band','impulse_low','impulse_high','all','band_reject','comb'])

parser.add_argument('--stream_mode', type=str, default='fusion',
                    choices=['fusion','gcn_only','mlp_only'])

# Output / seed / data
parser.add_argument('--do_shap', action='store_true')
parser.add_argument('--output_dir', type=str, default="/media/rilgpu/RIL_Lab1/Raju_Folder/dd/gsfiltersharply/review2resluts")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_root', type=str, default="/media/rilgpu/RIL_Lab1/Raju_Folder/dd/gsfiltersharply/data1")

# Optional per-run tag (for artifact uniqueness)
parser.add_argument('--run_tag', type=str, default='',
                    help='Optional label to avoid overwrites (e.g., exp1 or gpuA).')

# ---------- NCV flags ----------
parser.add_argument('--ncv', action='store_true',
                    help='Use nested CV (outer=assessment, inner=selection).')
parser.add_argument('--outer_folds', type=int, default=5)
parser.add_argument('--inner_folds', type=int, default=3)
parser.add_argument('--score', type=str, default='macro_f1',
                    choices=['macro_f1','accuracy','macro_auc'],
                    help='Inner-CV selection metric.')
parser.add_argument('--ncv_log_runtime', action='store_true',
                    help='Log time/RAM/VRAM for inner folds (runtime/memory vs K figure).')
parser.add_argument('--ncv_log_perclass', action='store_true',
                    help='Log per-class metrics for each inner/outer fold (precision/recall/F1/F2).')

# ---- sweeps & overrides (NEW CLI flags) ----
parser.add_argument('--filters', type=str, default='',
    help='Comma-separated filter types to sweep, e.g. "all,band_reject,low"')
parser.add_argument('--batch_sizes', type=str, default='',
    help='Comma-separated batch sizes, e.g. "32,64,128"')
parser.add_argument('--dropouts', type=str, default='',
    help='Comma-separated dropouts, e.g. "0.1,0.2,0.3"')
parser.add_argument('--lrs', type=str, default='',
    help='Comma-separated learning rates, e.g. "0.001,0.005,0.01"')
parser.add_argument('--k_by_filter', type=str, default='',
    help='Per-filter K override, e.g. "all=6;band_reject=10;low=8,10"')

args = parser.parse_args()

# ------------------------------------------------------------------
# Run mode tag for clarity across files (single vs nested)
# ------------------------------------------------------------------
CV_MODE = "nested" if args.ncv else "single"

# -------------------- parsed sweeps & K overrides (NEW) --------------------
def _parse_list(s: str, cast):
    """Parse 'a,b,c' or 'a;b;c' into [cast(a), cast(b), cast(c)]."""
    if not s:
        return []
    parts = [p.strip() for p in s.replace(';', ',').split(',')]
    return [cast(p) for p in parts if p]

# ---------- Build sweep grids with correct precedence ----------
# Filters
if args.filters:
    filter_types = _parse_list(args.filters, str)
elif getattr(args, "filter_type", None):
    # Constrain sweep using single-value flag if list flag not provided
    filter_types = [str(args.filter_type)]
else:
    filter_types = ["all", "band_reject"]

# Batch sizes
if args.batch_sizes:
    batch_sizes = _parse_list(args.batch_sizes, int)
elif getattr(args, "batchsize", None):
    batch_sizes = [int(args.batchsize)]
else:
    batch_sizes = [128]

# Dropouts (no single-value fallback flag exists)
dropout_values = _parse_list(args.dropouts, float) if args.dropouts else [0.2]

# Learning rates
if args.lrs:
    lr_values = _parse_list(args.lrs, float)
elif getattr(args, "lr", None) is not None:
    lr_values = [float(args.lr)]
else:
    lr_values = [0.001]

# ---- K per filter: defaults + optional overrides from --k_by_filter ----
_default_k_map = {
   
        "all": [1, 2, 4, 6, 8, 10],  # Any K value works for all-pass
        "low": [6, 8, 10, 12, 14],  # Refining low-pass approximation
        "high": [6, 8, 10, 12, 14],  # High-pass behaves symmetrically
        "impulse_low": [4, 6, 8, 10],  # Emphasizes low frequencies
        "impulse_high": [4, 6, 8, 10],  # Enhances high-frequency retention
        "band": [8, 10, 12, 14, 16],  # Must be even for band-pass
        "band_reject": [10, 12, 14, 16, 18],  # Requires higher K for precision
        "comb": [12, 14, 16, 18, 20],  # Higher K needed for oscillatory behavior
      
    
}
_k_override = {}
if args.k_by_filter:
    # accepts forms like: "all=6;band_reject=10" or "low=8,10;band=12,16"
    for pair in args.k_by_filter.split(';'):
        pair = pair.strip()
        if not pair or '=' not in pair:
            continue
        name, vals = pair.split('=', 1)
        name = name.strip()
        ks = [v for v in vals.replace(' ', '').split(',') if v]
        try:
            ks = [int(v) for v in ks]
        except ValueError:
            raise ValueError(f"Invalid K list for filter '{name}': {vals}")
        if ks:
            _k_override[name] = ks

def get_k_values(filter_type: str):
    """Use CLI overrides if present; else fall back to defaults."""
    if filter_type in _k_override:
        return _k_override[filter_type]
    if filter_type in _default_k_map:
        return _default_k_map[filter_type]
    raise ValueError(f"Invalid filter type: {filter_type}")
## =========================================================
#                Setup / paths / device
# =========================================================
_stream_map = {'fusion': 'fusion', 'gcn_only': 'gcn', 'mlp_only': 'mlp'}
stream_mode = _stream_map[args.stream_mode]
print(f"[Ablation] stream_mode = {stream_mode}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- paths + data loading + outputs ----------
def _final_path(p: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    p = p.replace('"', '').replace("'", '')
    return os.path.normpath(p)

OutputDir = _final_path(args.output_dir)
DataRoot  = _final_path(args.data_root)

print(f"[Paths] OutputDir (final) = {OutputDir!r}")
print(f"[Paths] DataRoot  (final) = {DataRoot!r}")

# quick guard to catch WinError 123 style issues early
if OutputDir.startswith('"') or OutputDir.startswith("'"):
    raise ValueError(f"OutputDir still has a leading quote: {OutputDir!r}")
if DataRoot.startswith('"') or DataRoot.startswith("'"):
    raise ValueError(f"DataRoot still has a leading quote: {DataRoot!r}")

# --- robust output-dir creation (handles file collisions on Windows) ---
def _ensure_directory(p: str) -> str:
    p = os.path.normpath(p)
    if os.path.exists(p) and not os.path.isdir(p):
        # A file is blocking this path; use an alternate folder
        base, _ = os.path.splitext(p)
        alt = base + "_dir"
        print(f"⚠️ A file exists at output_dir path: {p}. Using alternate folder: {alt}")
        p = alt
    os.makedirs(p, exist_ok=True)
    try:
        os.chmod(p, 0o777)
    except Exception:
        pass
    return p

# Create/normalize once (no second makedirs)
OutputDir = _ensure_directory(OutputDir)


# ========================== Data paths & setup ==========================
def get_data_paths(database: str, data_root: str):
    database = str(database).lower().strip()
    expression_data_path      = os.path.join(data_root, "common_expression_data.tsv")
    cnv_data_path             = os.path.join(data_root, "common_cnv_data.tsv")
    expression_variance_file  = os.path.join(data_root, "expression_variance.tsv")
    shuffle_index_path        = os.path.join(data_root, "common_shuffle_index.tsv")

    if database == 'biogrid':
        adjacency_matrix_file = os.path.join(data_root, "adj_matrix_biogrid.npz")
        non_null_index_path   = os.path.join(data_root, "biogrid_non_null.csv")
    elif database == 'string':
        adjacency_matrix_file = os.path.join(data_root, "adj_matrix_string.npz")
        non_null_index_path   = os.path.join(data_root, "string_non_null.csv")
    elif database == 'coexpression':
        adjacency_matrix_file = os.path.join(data_root, "adj_matrix_coexpression.npz")
        non_null_index_path   = os.path.join(data_root, "coexpression_non_null.csv")
    else:
        raise ValueError(f"Invalid database selection: {database}")

    # Fail early with clear messages
    def _must_exist(p, desc):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required {desc} not found: {p}")

    _must_exist(expression_data_path, "expression_data")
    if args.num_omic != 1:
        _must_exist(cnv_data_path, "cnv_data")
    _must_exist(adjacency_matrix_file, "adjacency_matrix")
    _must_exist(non_null_index_path, "non_null_index")

    return (expression_data_path, cnv_data_path, expression_variance_file, shuffle_index_path,
            adjacency_matrix_file, non_null_index_path)

# Resolve data files (use cleaned DataRoot)
(expression_data_path, cnv_data_path, expression_variance_file, shuffle_index_path,
 adjacency_matrix_file, non_null_index_path) = get_data_paths(args.database, DataRoot)

print("Using adjacency matrix:", adjacency_matrix_file)
print("Using non-null index file:", non_null_index_path)

# ---------------------- Load raw data tables ----------------------
print("Loading raw data tables...")
if args.num_omic == 1:
    expr_all_df = utilsdata.load_singleomic_data(expression_data_path)
    cnv_all_df  = None
else:
    expr_all_df, cnv_all_df = utilsdata.load_multiomics_data(expression_data_path, cnv_data_path)

# Labels from expression table only
if "icluster_cluster_assignment" not in expr_all_df.columns:
    raise KeyError("Column 'icluster_cluster_assignment' not found in expression table.")
labels_all = (expr_all_df['icluster_cluster_assignment'].values - 1).astype(np.int64)
out_dim = int(np.unique(labels_all).size)
print("Classes:", out_dim)

# ---------------------- Normalization policy ----------------------
# Applied LATER inside each fold with train-only statistics:
# from utilsdata import normalize_by_train
NORM_METHOD = "zscore"   # options: "zscore", "minmax", "robust"
print(f"[Norm] method={NORM_METHOD} (per-fold, train-only stats)")

# =================== DB-GCN channel configuration ===================
F_0    = int(args.num_omic)   # channels per gene (1=Expr, 2=CNV+Expr)
D_g    = int(args.num_gene)   # number of genes used this run

# Graph stream (you can tune later)
CL1_F  = 96
FC1_F  = 384
FC2_F  = 0

# MLP stream
NN_FC1 = 384
NN_FC2 = 128

# -------------------------- Output dirs ---------------------------
hyperparam_dir = os.path.join(OutputDir, "hyperparameter_tuning")
bestmodel_dir  = os.path.join(OutputDir, "bestmodel")
for _d in (OutputDir, hyperparam_dir, bestmodel_dir):
    os.makedirs(_d, exist_ok=True)
    try:
        os.chmod(_d, 0o777)
    except Exception:
        pass

# ------------------------- Result files --------------------------
csv_file_path = os.path.join(OutputDir, "00finalcoexsingle2000.csv")
per_fold_csv  = os.path.join(OutputDir, "per_fold_metrics.csv")

# ---------------------- Runtime/memory helpers --------------------
try:
    import psutil
    _psutil_ok = True
    process = psutil.Process(os.getpid())
except Exception:
    _psutil_ok = False
    process = None
# =========================================================
#     One inner split eval (deterministic, leak-free)
# =========================================================
def _ncv_eval_one_split(expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
                        args, stream_mode, device, out_dim,
                        F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2,
                        train_idx, val_idx, k, filt, bs, dp, lr,
                        data_seed,  # deterministic loader seed
                        log_runtime=False, process=None):
    import json
    from torch.utils.data import WeightedRandomSampler
    t0 = time.perf_counter()

    # ---------- normalize/coerce hyperparams ----------
    filt = str(filt).strip().strip('"').strip("'")
    k = int(k); bs = int(bs); dp = float(dp); lr = float(lr)
    if k < 1:
        raise ValueError(f"❌ K must be at least 1. Given: {k}")

    if log_runtime and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- leak-free gene selection (TRAIN ONLY) ---
    gene_list, gene_idx = utilsdata.select_top_genes_from_train_fold(
        expr_all_df, non_null_index_path, train_idx, args.num_gene
    )

    # --- build data for this split ---
    if args.num_omic == 1:
        A_sel, X_all, y_all, _ = utilsdata.build_fold_data_singleomics(
            expr_all_df, adjacency_matrix_file, gene_list, gene_idx,
            singleton=args.singleton, non_null_index_path=non_null_index_path
        )
    else:
        A_sel, X_all, y_all, _ = utilsdata.build_fold_data_multiomics(
            expr_all_df, cnv_all_df, adjacency_matrix_file, gene_list, gene_idx,
            singleton=args.singleton, non_null_index_path=non_null_index_path
        )

    # ========= NEW: tiny sanity checks (shape/channels) BEFORE normalization =========
    # Expect X_all with shape (N, G, C). If (N, G), promote to (N, G, 1).
    assert X_all.ndim in (2, 3), f"Unexpected X_all shape: {X_all.shape}"
    if X_all.ndim == 2:
        X_all = X_all[:, :, None]
    assert X_all.shape[2] == int(args.num_omic), \
        f"Channel mismatch: X_all has C={X_all.shape[2]} but num_omic={args.num_omic}"
    # ================================================================================

    # ========= Train-only normalization (per gene, per channel) =========
    norm_method = globals().get("NORM_METHOD", "zscore")
    X_all, _norm_stats = utilsdata.normalize_by_train(
        X_all, train_index=np.asarray(train_idx, dtype=int), method=norm_method
    )

    # Optional: re-assert after normalization (defensive)
    assert X_all.ndim == 3, f"Post-norm X_all should be 3D, got {X_all.shape}"
    assert X_all.shape[2] == int(args.num_omic), \
        f"Post-norm channel mismatch: C={X_all.shape[2]} vs num_omic={args.num_omic}"
    # ===================================================================

    # --- Laplacian / theta (skip for MLP stream) ---
    if stream_mode == "mlp":
        L_list, theta = None, None
    else:
        if hasattr(A_sel, "data") and A_sel.data.size > 0:
            A_max = float(A_sel.data.max())
        else:
            A_max = 0.0
        A_norm = A_sel if (A_max == 0.0) else (A_sel / A_max)

        L_sp = graph_laplacian(A_norm, normalized=True)
        L_sp = rescale_to_unit_interval(L_sp)  # L/2 → spectrum in [0,1]
        L_torch = utilsdata.sparse_mx_to_torch_sparse_tensor(L_sp).to(device)
        L_list = [L_torch]
        theta = compute_theta(k, filt)

    # --- split tensors ---
    Xtr_np = X_all[train_idx].astype(np.float32, copy=False)
    Xva_np = X_all[val_idx].astype(np.float32, copy=False)

    X_tr = torch.tensor(Xtr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_all[train_idx], dtype=torch.long)
    X_va = torch.tensor(Xva_np, dtype=torch.float32)
    y_va = torch.tensor(y_all[val_idx], dtype=torch.long)

    pin = torch.cuda.is_available()
    g = torch.Generator().manual_seed(int(data_seed))

    # ----- class-balanced sampler for TRAIN ONLY -----
    y_train_np = y_tr.cpu().numpy()
    cls_counts = np.bincount(y_train_np, minlength=out_dim).astype(np.float32)
    cls_counts[cls_counts == 0] = 1.0
    cls_weights_for_sampling = 1.0 / cls_counts
    sample_weights = cls_weights_for_sampling[y_train_np]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    tr_loader = Data.DataLoader(
        Data.TensorDataset(X_tr, y_tr),
        batch_size=bs,
        sampler=sampler,
        shuffle=False,
        num_workers=0, pin_memory=pin, generator=g,
        worker_init_fn=torch_worker_init_fn if pin else None
    )
    va_loader = Data.DataLoader(
        Data.TensorDataset(X_va, y_va),
        batch_size=bs,
        shuffle=False,
        num_workers=0, pin_memory=pin
    )

    # --- model ---
    net_params = [F_0, len(gene_list), CL1_F, k, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
    net = Graph_GCN(net_params, stream_mode=stream_mode).to(device)

    def _weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(_weight_init)

    # ---- class-weighted loss (from y_tr) ----
    with torch.no_grad():
        counts = torch.bincount(y_tr, minlength=out_dim).clamp_min(1)
        inv = 1.0 / counts.float()
        class_weights = (inv / inv.sum() * out_dim).to(device=device, dtype=torch.float32)

    # ---- optimizer + scheduler + loss ----
    lr_this = 1e-3 if (stream_mode == "mlp") else float(lr)
    optimizer = optim.Adam(net.parameters(), lr=lr_this, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=1e-6
    )

    # --- train ---
    for _ in range(int(args.epochs)):
        train_model(tr_loader, net, optimizer, criterion, device, dp, L_list, theta)
        scheduler.step()

    # --- validate ---
    y_true, y_scores = test_model(va_loader, net, device, L_list, theta)
    y_pred = np.argmax(y_scores, axis=1)

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]
    acc = accuracy_score(y_true, y_pred)

    macro_auc = np.nan
    try:
        y_bin = label_binarize(y_true, classes=list(range(out_dim)))
        aucs = []
        for c in range(out_dim):
            if y_bin[:, c].sum() > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, c], y_scores[:, c])
                aucs.append(auc(fpr, tpr))
        if len(aucs) > 0:
            macro_auc = float(np.mean(aucs))
    except Exception:
        pass

    sel_score = {"macro_f1": macro_f1, "accuracy": acc, "macro_auc": macro_auc}[args.score]

    # --- deterministic save dir tag ---
    run_tag = (
        f"db-{args.database}_mode-{stream_mode}_omic-{args.num_omic}"
        f"_filt-{filt}_K-{k}_bs-{bs}_dp-{dp}_lr-{lr}_seed-{int(data_seed)}"
    )
    save_dir = os.path.join(args.output_dir, "ncv_cache", run_tag)
    os.makedirs(save_dir, exist_ok=True)

    # ---- persist split artifacts (CSV/NPY only; no Excel) ----
    try:
        np.save(os.path.join(save_dir, "train_idx.npy"), np.asarray(train_idx, dtype=np.int64))
        np.save(os.path.join(save_dir, "val_idx.npy"),   np.asarray(val_idx,   dtype=np.int64))
        with open(os.path.join(save_dir, "gene_list.txt"), "w", encoding="utf-8") as f:
            for g in gene_list:
                f.write(f"{g}\n")
        np.save(os.path.join(save_dir, "y_true.npy"),   np.asarray(y_true, dtype=np.int64))
        np.save(os.path.join(save_dir, "y_pred.npy"),   np.asarray(y_pred, dtype=np.int64))
        np.save(os.path.join(save_dir, "y_scores.npy"), np.asarray(y_scores, dtype=np.float32))
        np.save(os.path.join(save_dir, "class_weights.npy"),
                class_weights.detach().float().cpu().numpy())

        pd.DataFrame(rep).T.to_csv(os.path.join(save_dir, "classification_report.csv"), index=True)
        with open(os.path.join(save_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"accuracy={acc:.6f}\nmacro_f1={macro_f1:.6f}\nmacro_auc="
                f"{'nan' if np.isnan(macro_auc) else f'{macro_auc:.6f}'}\n"
            )
        print(f"🧾 Saved split artifacts to: {save_dir}")
    except Exception as e_save:
        print(f"⚠️ Skipped saving split artifacts: {e_save}")

    # ---- optional: save model weights ----
    try:
        if getattr(args, "savemodel", False):
            ckpt = {
                "model": "Graph_GCN",
                "stream_mode": stream_mode,
                "net_params": net_params,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "meta": {
                    "database": args.database,
                    "num_omic": args.num_omic,
                    "num_gene": args.num_gene,
                    "epochs": int(args.epochs),
                    "filter_type": filt,
                    "k": int(k),
                    "batch_size": int(bs),
                    "dropout": float(dp),
                    "lr": float(lr),
                    "seed": int(data_seed),
                    "selection_metric": args.score,
                    "selection_value": float(sel_score),
                    "acc": float(acc),
                    "macro_f1": float(macro_f1),
                    "macro_auc": (np.nan if np.isnan(macro_auc) else float(macro_auc)),
                }
            }
            ckpt_path = os.path.join(save_dir, "weights_final.pt")
            torch.save(ckpt, ckpt_path)
            print(f"💾 Weights saved → {ckpt_path}")
    except Exception as e_ckpt:
        print(f"⚠️ Could not save weights: {e_ckpt}")

    # ---- runtime / memory stats ----
    aux = {}
    if log_runtime:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        aux["time_sec"] = time.perf_counter() - t0
        aux["rss_mb"] = (process.memory_info().rss / (1024**2)) if (process is not None) else np.nan
        aux["vram_peak_mb"] = (
            torch.cuda.max_memory_allocated() / (1024**2)
        ) if torch.cuda.is_available() else 0.0

    # --- free big tensors ---
    del X_tr, y_tr, X_va, y_va, tr_loader, va_loader
    del X_all, y_all
    if stream_mode != "mlp":
        del L_sp, L_torch, L_list
    del net, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sel_score, rep, acc, macro_f1, macro_auc, aux


from sklearn.model_selection import StratifiedKFold
def nested_cv_run(expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
                  F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim,
                  filter_types, get_k_values_fn, batch_sizes, dropout_values, lr_values,
                  singleton, stream_mode, device, args, process=None):
    """
    Nested CV with train-only normalization (inner + outer).
    Writes:
      - per-fold rows -> per_fold_csv
      - one summary row (mean±std across outer folds) -> csv_file_path
    """
    from torch.utils.data import WeightedRandomSampler
    K_outer, K_inner = args.outer_folds, args.inner_folds
    kf_outer = StratifiedKFold(n_splits=K_outer, shuffle=True, random_state=int(args.seed))
    y_all_labels = (expr_all_df['icluster_cluster_assignment'].values - 1).astype(np.int64)

    outer_rows, search_rows, runtime_rows = [], [], []
    ncv_dir = os.path.join(args.output_dir, "nested_cv")
    os.makedirs(ncv_dir, exist_ok=True)

    # Helper: append CSV (header only if new)
    def _append_csv(df: pd.DataFrame, path: str):
        path_dir = os.path.dirname(path)
        if path_dir and not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        header = not os.path.exists(path)
        df.to_csv(path, mode="a", index=False, header=header)

    for outer_fold, (outer_tr_idx, outer_te_idx) in enumerate(kf_outer.split(expr_all_df, y_all_labels), start=1):
        # Save outer splits for reproducibility
        np.savetxt(os.path.join(ncv_dir, f"outer_fold_{outer_fold}_train_idx.txt"), outer_tr_idx, fmt="%d")
        np.savetxt(os.path.join(ncv_dir, f"outer_fold_{outer_fold}_test_idx.txt"),  outer_te_idx, fmt="%d")

        # ----- inner search on outer_tr_idx -----
        kf_inner = StratifiedKFold(
            n_splits=K_inner, shuffle=True, random_state=int(args.seed) + outer_fold
        )
        grid_records = []
        inner_dir = os.path.join(ncv_dir, f"outer_fold_{outer_fold}", "inner_splits")
        os.makedirs(inner_dir, exist_ok=True)

        # Precompute inner splits (indices against FULL dataset)
        inner_split_cache = []
        for inner_id, (inner_tr_local, inner_va_local) in enumerate(
            kf_inner.split(outer_tr_idx, y_all_labels[outer_tr_idx]), start=1
        ):
            tr_idx = outer_tr_idx[inner_tr_local]
            va_idx = outer_tr_idx[inner_va_local]
            np.savetxt(os.path.join(inner_dir, f"inner_{inner_id:02d}_train_idx.txt"), tr_idx, fmt="%d")
            np.savetxt(os.path.join(inner_dir, f"inner_{inner_id:02d}_val_idx.txt"),   va_idx, fmt="%d")
            inner_split_cache.append((tr_idx, va_idx))

        for filt_raw in filter_types:
            filt = str(filt_raw).strip().strip('"').strip("'")
            for k_val in get_k_values_fn(filt):
                k_val = int(k_val)
                for bs_val in batch_sizes:
                    bs_val = int(bs_val)
                    for dp_val in dropout_values:
                        dp_val = float(dp_val)
                        for lr_val in lr_values:
                            lr_val = float(lr_val)

                            inner_scores = []
                            for inner_id, (tr_idx, va_idx) in enumerate(inner_split_cache, start=1):
                                sc, _, _, _, _, aux = _ncv_eval_one_split(
                                    expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
                                    args, stream_mode, device, out_dim,
                                    F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2,
                                    tr_idx, va_idx, k_val, filt, bs_val, dp_val, lr_val,
                                    data_seed=int(args.seed) + 10_000*outer_fold + inner_id,
                                    log_runtime=bool(args.ncv_log_runtime),
                                    process=process if _psutil_ok else None
                                )
                                inner_scores.append(sc)

                                if args.ncv_log_runtime:
                                    runtime_rows.append({
                                        "cv_mode": "nested",
                                        "database": args.database,
                                        "stream_mode": stream_mode,
                                        "num_omic": args.num_omic,
                                        "num_gene": args.num_gene,
                                        "epochs": args.epochs,
                                        "outer_fold": outer_fold,
                                        "inner_fold": inner_id,
                                        "filter_type": filt,
                                        "k": k_val,
                                        "batch_size": bs_val,
                                        "dropout": dp_val,
                                        "lr": lr_val,
                                        "time_sec": aux.get("time_sec", np.nan),
                                        "rss_mb": aux.get("rss_mb", np.nan),
                                        "vram_peak_mb": aux.get("vram_peak_mb", np.nan),
                                    })

                            grid_records.append({
                                "cv_mode": "nested",
                                "database": args.database,
                                "stream_mode": stream_mode,
                                "num_omic": args.num_omic,
                                "num_gene": args.num_gene,
                                "epochs": args.epochs,
                                "outer_fold": outer_fold,
                                "filter_type": filt,
                                "k": k_val,
                                "batch_size": bs_val,
                                "dropout": dp_val,
                                "lr": lr_val,
                                f"{args.score}_mean": float(np.mean(inner_scores)) if inner_scores else np.nan,
                                f"{args.score}_std":  float(np.std(inner_scores, ddof=1)) if len(inner_scores) > 1 else 0.0,
                            })

        # ---- choose best config by inner mean of selection metric ----
        score_col = f"{args.score}_mean"
        grid_df = pd.DataFrame(grid_records)
        if grid_df.empty:
            raise RuntimeError("Inner search produced no configurations.")
        grid_df[score_col] = pd.to_numeric(grid_df[score_col], errors="coerce")
        grid_df = grid_df[np.isfinite(grid_df[score_col])]
        if grid_df.empty:
            raise RuntimeError(f"All inner-CV '{score_col}' values are NaN/inf.")

        # Save the inner grid for this outer fold (for audit)
        grid_csv_path = os.path.join(ncv_dir, f"outer_fold_{outer_fold}_inner_grid.csv")
        grid_df.sort_values(by=[score_col], ascending=False).to_csv(grid_csv_path, index=False)

        grid_df = grid_df.sort_values(by=[score_col], ascending=False).reset_index(drop=True)
        best = grid_df.iloc[0].to_dict()
        search_rows.extend(grid_df.to_dict("records"))

        # ---- retrain best on outer-train, test on outer-test ----
        best_filt = str(best["filter_type"]).strip().strip('"').strip("'")
        best_k    = int(best["k"])
        best_bs   = int(best["batch_size"])
        best_dp   = float(best["dropout"])
        best_lr   = float(best["lr"])

        # One evaluation pass (stats only; this already normalizes per-inner)
        _, rep, acc, macro_f1, macro_auc, _ = _ncv_eval_one_split(
            expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
            args, stream_mode, device, out_dim,
            F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2,
            outer_tr_idx, outer_te_idx, best_k, best_filt, best_bs, best_dp, best_lr,
            data_seed=int(args.seed) + 100_000*outer_fold,
            log_runtime=False, process=None
        )

        # ----- (Optional) artifacts + plots with explicit rebuild (kept minimal) -----
        # Rebuild to keep the saved artifacts consistent with chosen best
        try:
            gene_list, gene_idx = utilsdata.select_top_genes_from_train_fold(
                expr_all_df, non_null_index_path, outer_tr_idx, args.num_gene
            )
            if args.num_omic == 1:
                A_sel, X_all, y_all, _ = utilsdata.build_fold_data_singleomics(
                    expr_all_df, adjacency_matrix_file, gene_list, gene_idx,
                    singleton=args.singleton, non_null_index_path=non_null_index_path
                )
            else:
                A_sel, X_all, y_all, _ = utilsdata.build_fold_data_multiomics(
                    expr_all_df, cnv_all_df, adjacency_matrix_file, gene_list, gene_idx,
                    singleton=args.singleton, non_null_index_path=non_null_index_path
                )

            # ========= NEW: sanity checks BEFORE normalization =========
            assert X_all.ndim in (2, 3), f"Unexpected X_all shape: {X_all.shape}"
            if X_all.ndim == 2:
                X_all = X_all[:, :, None]
            assert X_all.shape[2] == int(args.num_omic), \
                f"Channel mismatch: X_all has C={X_all.shape[2]} but num_omic={args.num_omic}"
            # ==========================================================

            # ===== Train-only normalization for OUTER (method from global) =====
            norm_method = globals().get("NORM_METHOD", "zscore")  # set to "minmax" if needed
            X_all, _stats = utilsdata.normalize_by_train(
                X_all, train_index=np.asarray(outer_tr_idx, dtype=int), method=norm_method
            )

            # Optional: re-assert after normalization (defensive)
            assert X_all.ndim == 3, f"Post-norm X_all should be 3D, got {X_all.shape}"
            assert X_all.shape[2] == int(args.num_omic), \
                f"Post-norm channel mismatch: C={X_all.shape[2]} vs num_omic={args.num_omic}"
            # ===================================================================

            # Build Laplacian/theta unless mlp stream
            if stream_mode == "mlp":
                L_list, theta = None, None
            else:
                if hasattr(A_sel, "data") and A_sel.data.size > 0:
                    A_max = float(A_sel.data.max())
                else:
                    A_max = 0.0
                A_norm = A_sel if A_max == 0.0 else (A_sel / A_max)
                L_sp = graph_laplacian(A_norm, normalized=True)
                L_sp = rescale_to_unit_interval(L_sp)
                L_torch = utilsdata.sparse_mx_to_torch_sparse_tensor(L_sp).to(device)
                L_list = [L_torch]
                theta = compute_theta(best_k, best_filt)

            net_params = [F_0, len(gene_list), CL1_F, best_k, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
            net = Graph_GCN(net_params, stream_mode=stream_mode).to(device)
            def _wi(m):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            net.apply(_wi)

            # ---- class-weighted loss (from y_train) ----
            with torch.no_grad():
                y_train = torch.tensor(y_all[outer_tr_idx], dtype=torch.long)
                counts = torch.bincount(y_train, minlength=out_dim).clamp_min(1)
                inv = 1.0 / counts.float()
                class_weights = (inv / inv.sum() * out_dim).to(device=device, dtype=torch.float32)

            lr_this = 1e-3 if (stream_mode == "mlp") else float(best_lr)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr_this, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(args.epochs), eta_min=max(lr_this * 0.1, 1e-6)
            )

            g = torch.Generator().manual_seed(int(args.seed) + 500_000 + outer_fold)

            # Split tensors AFTER normalization
            X_train = torch.tensor(X_all[outer_tr_idx].astype(np.float32, copy=False))
            X_test  = torch.tensor(X_all[outer_te_idx].astype(np.float32, copy=False))
            y_test  = torch.tensor(y_all[outer_te_idx], dtype=torch.long)

            pin = torch.cuda.is_available()

            # ---- balanced sampler for outer-train ----
            y_train_np = y_train.cpu().numpy()
            cls_counts = np.bincount(y_train_np, minlength=out_dim).astype(np.float32)
            cls_counts[cls_counts == 0] = 1.0
            cls_w_samp = (1.0 / cls_counts) ** 0.5
            sample_w   = cls_w_samp[y_train_np]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_w, dtype=torch.double),
                num_samples=len(sample_w),
                replacement=True
            )

            tr_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=best_bs,
                sampler=sampler, shuffle=False,
                num_workers=0, pin_memory=pin, generator=g
            )
            te_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_test, y_test),
                batch_size=best_bs, shuffle=False,
                num_workers=0, pin_memory=pin
            )

            for _ in range(int(args.epochs)):
                train_model(tr_loader, net, optimizer, criterion, device, best_dp, L_list, theta)
                scheduler.step()

            y_true, y_scores = test_model(te_loader, net, device, L_list, theta)
            y_pred = np.argmax(y_scores, axis=1)

            # Save minimal best artifacts per outer fold (no Excel)
            best_dir = os.path.join(ncv_dir, f"outer_fold_{outer_fold}", "best_outer_artifacts")
            os.makedirs(best_dir, exist_ok=True)
            np.save(os.path.join(best_dir, "outer_train_idx.npy"), np.asarray(outer_tr_idx, dtype=np.int64))
            np.save(os.path.join(best_dir, "outer_test_idx.npy"),  np.asarray(outer_te_idx, dtype=np.int64))
            with open(os.path.join(best_dir, "gene_list.txt"), "w", encoding="utf-8") as f:
                for gname in gene_list: f.write(f"{gname}\n")
            np.save(os.path.join(best_dir, "y_true.npy"),   np.asarray(y_true, dtype=np.int64))
            np.save(os.path.join(best_dir, "y_pred.npy"),   np.asarray(y_pred, dtype=np.int64))
            np.save(os.path.join(best_dir, "y_scores.npy"), np.asarray(y_scores, dtype=np.float32))
            np.save(os.path.join(best_dir, "class_weights.npy"), class_weights.detach().float().cpu().numpy())

            if getattr(args, "savemodel", False):
                ckpt = {
                    "model": "Graph_GCN",
                    "stream_mode": stream_mode,
                    "net_params": net_params,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "meta": {
                        "phase": "outer_best_retrain",
                        "database": args.database,
                        "num_omic": args.num_omic,
                        "num_gene": args.num_gene,
                        "epochs": int(args.epochs),
                        "filter_type": best_filt,
                        "k": int(best_k),
                        "batch_size": int(best_bs),
                        "dropout": float(best_dp),
                        "lr": float(best_lr),
                        "outer_fold": int(outer_fold),
                        "acc": float(accuracy_score(y_true, y_pred)),
                        "macro_f1": float(classification_report(y_true, y_pred, output_dict=True)["macro avg"]["f1-score"]),
                    }
                }
                ckpt_path = os.path.join(best_dir, "weights_final.pt")
                torch.save(ckpt, ckpt_path)

        except Exception as _e:
            print(f"⚠️ Skipped outer-fold artifacts (OK for speed): {_e}")

        # ----- per-outer-fold CSV row -----
        per_class = {int(k): v for k, v in rep.items() if str(k).isdigit()}
        row = {
            "cv_mode": "nested",
            "database": args.database,
            "stream_mode": stream_mode,
            "num_omic": args.num_omic,
            "num_gene": args.num_gene,
            "epochs": args.epochs,
            "outer_fold": outer_fold,
            "filter_type": best_filt,
            "k": best_k,
            "batch_size": best_bs,
            "dropout": best_dp,
            "lr": best_lr,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "macro_auc": float(macro_auc) if not np.isnan(macro_auc) else np.nan,
        }
        for cid, m in per_class.items():
            row[f"prec_c{cid}"] = m.get("precision", np.nan)
            row[f"rec_c{cid}"]  = m.get("recall", np.nan)
            row[f"f1_c{cid}"]   = m.get("f1-score", np.nan)
        outer_rows.append(row)

        print(
            "[NCV] Outer {}: best=(filter={}, K={}, BS={}, dp={}, lr={}) "
            "macro-F1={:.4f} acc={:.4f} macro-AUC={}".format(
                outer_fold, best_filt, best_k, best_bs, best_dp, best_lr,
                macro_f1, acc, "NA" if np.isnan(macro_auc) else f"{macro_auc:.4f}"
            )
        )

    # ---------- aggregate & write results ----------
    outer_df   = pd.DataFrame(outer_rows)
    search_df  = pd.DataFrame(search_rows)
    runtime_df = pd.DataFrame(runtime_rows) if args.ncv_log_runtime else None

    # Append per-outer-fold rows to your single per_fold CSV
    _append_csv(outer_df, per_fold_csv)

    # One summary row (mean ± std) to your single “final” CSV
    def _mean_std(s):
        return float(np.mean(s)), float(np.std(s, ddof=1)) if len(s) > 1 else 0.0

    acc_mu, acc_sd   = _mean_std(outer_df["accuracy"])
    f1_mu,  f1_sd    = _mean_std(outer_df["macro_f1"])
    auc_mu, auc_sd   = _mean_std(outer_df["macro_auc"]) if "macro_auc" in outer_df else (np.nan, np.nan)

    summary_row = pd.DataFrame([{
        "cv_mode": "nested",
        "database": args.database,
        "stream_mode": stream_mode,
        "num_omic": args.num_omic,
        "num_gene": args.num_gene,
        "epochs": args.epochs,
        "outer_folds": K_outer,
        "inner_folds": K_inner,
        "norm_method": globals().get("NORM_METHOD", "zscore"),
        "acc_mean": acc_mu,        "acc_std": acc_sd,
        "macro_f1_mean": f1_mu,    "macro_f1_std": f1_sd,
        "macro_auc_mean": auc_mu,  "macro_auc_std": auc_sd,
    }])
    _append_csv(summary_row, csv_file_path)

    # Optional runtime log (separate small CSV in the same OutputDir)
    if runtime_df is not None and not runtime_df.empty:
        _append_csv(runtime_df, os.path.join(OutputDir, "ncv_runtime.csv"))

    return outer_df, search_df, runtime_df

def _df_to_latex_simple(df: pd.DataFrame) -> str:
    """
    Produce a minimal LaTeX tabular for the summary table.
    Expects columns: Metric, Mean, Std, n, cv_mode, database, stream_mode, num_omic, num_gene
    Extra cols are ignored; missing cols are shown as blanks.
    """
    cols_needed = ["Metric", "Mean", "Std", "n", "cv_mode", "database", "stream_mode", "num_omic", "num_gene"]
    df2 = df.copy()
    for c in cols_needed:
        if c not in df2.columns:
            df2[c] = ""

    lines = [
        r"\begin{tabular}{lccrllllr}",
        r"\toprule",
        r"Metric & Mean & Std & n & cv\_mode & database & stream\_mode & num\_omic & num\_gene \\",
        r"\midrule",
    ]
    for _, r in df2[cols_needed].iterrows():
        mean = _fmt_cell(r["Mean"])
        std  = _fmt_cell(r["Std"])
        # render n as int when possible
        n_val = r["n"]
        try:
            n_val = str(int(float(n_val)))
        except Exception:
            n_val = str(n_val)

        row_items = [
            _latex_escape(r["Metric"]),
            _latex_escape(mean),
            _latex_escape(std),
            _latex_escape(n_val),
            _latex_escape(r["cv_mode"]),
            _latex_escape(r["database"]),
            _latex_escape(r["stream_mode"]),
            _latex_escape(r["num_omic"]),
            _latex_escape(r["num_gene"]),
        ]
        lines.append(" & ".join(row_items) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def export_main_text_tables(outer_df: pd.DataFrame, out_dir: str,
                            top_k_classes: int = 3, split_by_mode: bool = True,
                            include_artifact_notes: bool = True):
    """
    Summarize outer-fold results into CSV + LaTeX tables.
    - Writes per-mode tables if split_by_mode=True, plus a combined table.
    - Also averages a few top per-class metrics (precision/recall/F1) for the most performant classes.
    - If include_artifact_notes=True and nested-CV folders exist in out_dir,
      append short rows indicating where split artifacts are stored.

    NOTE: For nested-CV artifact notes to appear, call this with out_dir pointing
    to the NCV folder that contains `outer_fold_*` (e.g., `ncv_dir`).
    """
    import glob
    os.makedirs(out_dir, exist_ok=True)

    # small hint for LaTeX users
    print("ℹ️  LaTeX: include \\usepackage{booktabs} in your preamble for top/mid/bottom rules.")

    if outer_df is None or len(outer_df) == 0:
        empty = pd.DataFrame(columns=["Metric","Mean","Std","n","cv_mode","database","stream_mode","num_omic","num_gene","epochs"])
        empty_csv = os.path.join(out_dir, "main_text_macro_and_class_empty.csv")
        empty_tex = os.path.join(out_dir, "main_text_macro_and_class_empty.tex")
        empty.to_csv(empty_csv, index=False)
        with open(empty_tex, "w", encoding="utf-8") as f:
            f.write(_df_to_latex_simple(empty))
        print(f"⚠️ export_main_text_tables: received empty DataFrame. Wrote placeholders to {out_dir}.")
        return

    # guard
    try:
        top_k_classes = max(0, int(top_k_classes))
    except Exception:
        top_k_classes = 0

    df = outer_df.copy()

    # numeric coercions
    for col in ("accuracy", "macro_f1", "macro_auc", "num_omic", "num_gene", "epochs"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cls_f1_cols  = [c for c in df.columns if c.startswith("f1_c")]
    cls_prec_cols = [c for c in df.columns if c.startswith("prec_c")]
    cls_rec_cols  = [c for c in df.columns if c.startswith("rec_c")]

    def _append_artifact_notes(rows, meta):
        if not include_artifact_notes:
            return rows
        outer_folders = sorted(glob.glob(os.path.join(out_dir, "outer_fold_*")))
        if not outer_folders:
            return rows
        for fold_path in outer_folders:
            fold_name = os.path.basename(fold_path)
            inner_dir = os.path.join(fold_path, "inner_splits")
            inner_tr = sorted(glob.glob(os.path.join(inner_dir, "inner_*_train_idx.txt")))
            inner_va = sorted(glob.glob(os.path.join(inner_dir, "inner_*_val_idx.txt")))
            n_inner = min(len(inner_tr), len(inner_va)) if os.path.isdir(inner_dir) else 0

            rows.append([
                f"{fold_name} inner_splits_saved",
                str(n_inner), "-", "1",
                meta.get("cv_mode",""), meta.get("database",""), meta.get("stream_mode",""),
                str(meta.get("num_omic","")), str(meta.get("num_gene","")), str(meta.get("epochs",""))
            ])
            rows.append([
                f"{fold_name} artifacts_dir",
                _latex_escape(os.path.relpath(fold_path, out_dir)), "-", "1",
                meta.get("cv_mode",""), meta.get("database",""), meta.get("stream_mode",""),
                str(meta.get("num_omic","")), str(meta.get("num_gene","")), str(meta.get("epochs",""))
            ])
        return rows

    def _summary_for(sub: pd.DataFrame) -> pd.DataFrame:
        if sub is None or len(sub) == 0:
            return pd.DataFrame(columns=["Metric","Mean","Std","n","cv_mode","database","stream_mode","num_omic","num_gene","epochs"])

        meta = {
            "cv_mode":     sub.get("cv_mode",     pd.Series([""])).iloc[0] if len(sub) else "",
            "database":    sub.get("database",    pd.Series([""])).iloc[0] if len(sub) else "",
            "stream_mode": sub.get("stream_mode", pd.Series([""])).iloc[0] if len(sub) else "",
            "num_omic":    sub.get("num_omic",    pd.Series([""])).iloc[0] if len(sub) else "",
            "num_gene":    sub.get("num_gene",    pd.Series([""])).iloc[0] if len(sub) else "",
            "epochs":      sub.get("epochs",      pd.Series([""])).iloc[0] if len(sub) else "",
        }

        n = int(len(sub))
        rows = [["Metric","Mean","Std","n","cv_mode","database","stream_mode","num_omic","num_gene","epochs"]]

        for m in [c for c in ("accuracy","macro_f1","macro_auc") if c in sub.columns]:
            vals = pd.to_numeric(sub[m], errors="coerce")
            mu = float(np.nanmean(vals)) if n else np.nan
            sd = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
            rows.append([m,
                         f"{mu:.4f}" if np.isfinite(mu) else "--",
                         f"{sd:.4f}" if np.isfinite(sd) else "--",
                         str(n),
                         meta["cv_mode"], meta["database"], meta["stream_mode"],
                         str(meta["num_omic"]), str(meta["num_gene"]), str(meta["epochs"])])

        # Per-class: pick top_k_classes by mean F1 (if present)
        if cls_f1_cols and top_k_classes > 0:
            means = []
            for c in cls_f1_cols:
                try:
                    cid = int(c.split("f1_c", 1)[1])
                except Exception:
                    continue
                vals = pd.to_numeric(sub.get(c, pd.Series(dtype=float)), errors="coerce")
                m = float(np.nanmean(vals)) if len(vals) else np.nan
                if np.isfinite(m):
                    means.append((cid, m))
            means.sort(key=lambda t: -t[1])
            selected = [cid for cid, _ in means[:top_k_classes]]

            for cid in selected:
                pr = float(np.nanmean(pd.to_numeric(sub.get(f"prec_c{cid}", pd.Series(dtype=float)), errors="coerce")))
                rc = float(np.nanmean(pd.to_numeric(sub.get(f"rec_c{cid}",  pd.Series(dtype=float)), errors="coerce")))
                f1 = float(np.nanmean(pd.to_numeric(sub.get(f"f1_c{cid}",   pd.Series(dtype=float)), errors="coerce")))
                rows.append([f"class {cid} precision", f"{pr:.4f}" if np.isfinite(pr) else "--", "-", str(n),
                             meta["cv_mode"], meta["database"], meta["stream_mode"],
                             str(meta["num_omic"]), str(meta["num_gene"]), str(meta["epochs"])])
                rows.append([f"class {cid} recall",    f"{rc:.4f}" if np.isfinite(rc) else "--", "-", str(n),
                             meta["cv_mode"], meta["database"], meta["stream_mode"],
                             str(meta["num_omic"]), str(meta["num_gene"]), str(meta["epochs"])])
                rows.append([f"class {cid} F1",        f"{f1:.4f}" if np.isfinite(f1) else "--", "-", str(n),
                             meta["cv_mode"], meta["database"], meta["stream_mode"],
                             str(meta["num_omic"]), str(meta["num_gene"]), str(meta["epochs"])])

        # Artifact notes (only meaningful for nested CV directories passed as out_dir)
        if str(meta["cv_mode"]).lower() == "nested":
            rows = _append_artifact_notes(rows, meta)

        return pd.DataFrame(rows[1:], columns=rows[0])

    # -------- write per-mode and combined tables --------
    if split_by_mode and "cv_mode" in df.columns:
        frames = []
        for mode, sub in df.groupby("cv_mode", dropna=False):
            mode_tag = (str(mode).strip() if pd.notna(mode) else "unknown") or "unknown"
            tbl = _summary_for(sub)
            frames.append(tbl)

            csv_path = os.path.join(out_dir, f"main_text_{mode_tag}_macro_and_class.csv")
            tex_path = os.path.join(out_dir, f"main_text_{mode_tag}_macro_and_class.tex")
            tbl.to_csv(csv_path, index=False)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(_df_to_latex_simple(tbl))

        combined = pd.concat(frames, ignore_index=True) if frames else _summary_for(df)
        combined_csv = os.path.join(out_dir, "main_text_macro_and_class_by_mode.csv")
        combined_tex = os.path.join(out_dir, "main_text_macro_and_class_by_mode.tex")
        combined.to_csv(combined_csv, index=False)
        with open(combined_tex, "w", encoding="utf-8") as f:
            f.write(_df_to_latex_simple(combined))
        print(f"✅ Main-text tables saved in {out_dir} (per-mode + combined).")
    else:
        tbl = _summary_for(df)
        csv_path = os.path.join(out_dir, "main_text_macro_and_class.csv")
        tex_path = os.path.join(out_dir, "main_text_macro_and_class.tex")
        tbl.to_csv(csv_path, index=False)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(_df_to_latex_simple(tbl))
        print(f"✅ Main-text table saved in {out_dir}.")

## =========================================================
#                MAIN: NCV or single-CV
# =========================================================

# Use parsed grids from CLI (fallback to defaults if user didn't pass them)
if not filter_types:
    filter_types = ["impulse_high", "impulse_low", "low", "high", "all", "band", "band_reject", "comb"]

# Default sweep: batch sizes 32/64/128 and dropouts 0.1/0.2/0.3
if not batch_sizes:
    batch_sizes = [32, 64, 128]
if not dropout_values:
    dropout_values = [0.1, 0.2, 0.3]

# LR: keep single default unless a sweep was provided earlier
if not lr_values:
    lr_values = [float(args.lr)]

print("[Grid] filters=", filter_types,
      "| batch_sizes=", batch_sizes,
      "| dropouts=", dropout_values,
      "| lrs=", lr_values)

# -------------------- NESTED CV --------------------
if args.ncv:
    print(">>> Running Nested Cross-Validation (outer=assessment, inner=selection)…")

    def _kvals(ft: str):
        return get_k_values(str(ft).strip().strip('"').strip("'"))

    outer_df, search_df, runtime_df = nested_cv_run(
        expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
        F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim,
        filter_types, _kvals, batch_sizes, dropout_values, lr_values,
        args.singleton, stream_mode, device, args, process=process if _psutil_ok else None
    )

    # Tag for context (tables downstream can use these)
    outer_df["cv_mode"]     = "nested"
    outer_df["database"]    = args.database
    outer_df["stream_mode"] = stream_mode
    outer_df["num_omic"]    = args.num_omic
    outer_df["num_gene"]    = args.num_gene
    outer_df["epochs"]      = args.epochs

    # (You asked to keep outputs to two CSVs overall for single-CV mode;
    # nested-CV artifacts are handled elsewhere in your pipeline, so we exit.)
    sys.exit(0)

# -------------------- SINGLE CV (only two CSV outputs) --------------------
print(">>> Running single CV (original) …")

expected_cols = [
    "database","stream_mode","num_omic",
    "filter_type","k","num_genes","batch_size","dropout","lr",
    "accuracy","precision","recall","macro_f1","macro_auc",
    "acc_mean","acc_std","macro_f1_mean","macro_f1_std",
    "macro_precision_mean","macro_precision_std","macro_recall_mean","macro_recall_std",
    "run_time_sec","peak_rss_mb","peak_vram_mb",
    "cv_mode","epochs","num_gene"
]

if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    df_existing = pd.read_csv(csv_file_path)
else:
    df_existing = pd.DataFrame(columns=expected_cols)

# add any missing expected cols
for c in expected_cols:
    if c not in df_existing.columns:
        df_existing[c] = np.nan

# numeric coercions for stability
for col in ["k","num_genes","batch_size","dropout","lr","accuracy",
            "acc_mean","acc_std","macro_f1_mean","macro_f1_std",
            "macro_precision_mean","macro_precision_std","macro_recall_mean","macro_recall_std",
            "run_time_sec","peak_rss_mb","peak_vram_mb","epochs","num_gene","num_omic"]:
    if col in df_existing.columns:
        df_existing[col] = pd.to_numeric(df_existing[col], errors="coerce")

if "cv_mode" in df_existing.columns and df_existing["cv_mode"].isna().any():
    df_existing.loc[df_existing["cv_mode"].isna(), "cv_mode"] = "single"

# default values for context cols
for col, default in [("database", args.database), ("stream_mode", stream_mode), ("num_omic", args.num_omic)]:
    if col not in df_existing.columns:
        df_existing[col] = default

_key_cols = ["database","stream_mode","num_omic","filter_type","k","num_genes","batch_size","dropout","lr"]

# compose the set of existing combinations (floats rounded via _normf)
if not df_existing.empty:
    existing_combinations = set(
        (
            row["database"], row["stream_mode"], int(row["num_omic"]),
            row["filter_type"], int(row["k"]), int(row["num_genes"]), int(row["batch_size"]),
            _normf(row["dropout"]), _normf(row["lr"]),
        )
        for _, row in df_existing[_key_cols].dropna().iterrows()
    )
else:
    existing_combinations = set()

# keep track of best accuracy so far (only among single CV rows)
best_result, best_accuracy = None, -1.0
if not df_existing.empty and "accuracy" in df_existing.columns and df_existing["accuracy"].notna().any():
    scope = df_existing[
        (df_existing["database"] == args.database) &
        (df_existing["stream_mode"] == stream_mode) &
        (df_existing["num_omic"] == args.num_omic) &
        (df_existing["cv_mode"] == "single")
    ]
    if not scope.empty:
        idx = scope["accuracy"].idxmax()
        best_result  = df_existing.loc[idx].to_dict()
        best_accuracy = float(df_existing.loc[idx, "accuracy"])

# ----------------- precompute single-CV splits ONCE (deterministic) and SAVE them -----------------
kf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=int(args.seed))
y_all_labels = (expr_all_df['icluster_cluster_assignment'].values - 1).astype(np.int64)

splits = []
singlecv_dir = os.path.join(
    OutputDir,
    f"single_cv_splits_db-{args.database}_stream-{stream_mode}_G-{int(args.num_gene)}_seed-{int(args.seed)}"
)
os.makedirs(singlecv_dir, exist_ok=True)

for fold_id, (train_index, val_index) in enumerate(kf.split(expr_all_df, y_all_labels), start=1):
    splits.append((train_index, val_index))
    # persist for reproducibility
    np.save(os.path.join(singlecv_dir, f"fold_{fold_id:02d}_train_idx.npy"),
            np.asarray(train_index, dtype=np.int64))
    np.save(os.path.join(singlecv_dir, f"fold_{fold_id:02d}_val_idx.npy"),
            np.asarray(val_index, dtype=np.int64))

with open(os.path.join(singlecv_dir, "MANIFEST.txt"), "w", encoding="utf-8") as f:
    f.write(
        "cv_mode=single\n"
        f"num_folds={int(args.num_folds)}\n"
        f"seed={int(args.seed)}\n"
        f"database={args.database}\n"
        f"stream_mode={stream_mode}\n"
        f"num_gene={int(args.num_gene)}\n"
    )
# -----------------------------------------------------------------------------------------------

from torch.utils.data import WeightedRandomSampler

first_fold_info_printed = False

# directory to store PER-FOLD gene lists & fold meta (once per fold)
gene_dir_root = os.path.join(
    OutputDir, "gene_lists",
    f"db-{args.database}_stream-{stream_mode}_G-{int(args.num_gene)}_omics-{int(args.num_omic)}_seed-{int(args.seed)}"
)
os.makedirs(gene_dir_root, exist_ok=True)
fold_meta_dir = os.path.join(gene_dir_root, "fold_meta")
os.makedirs(fold_meta_dir, exist_ok=True)

for raw_filter_type in filter_types:
    # Normalize to guard against stray quotes/whitespace
    filter_type = str(raw_filter_type).strip().strip('"').strip("'")
    k_values = get_k_values(filter_type)

    for batch_size in batch_sizes:
        for dropout_value in dropout_values:
            for lr in lr_values:
                for k in k_values:
                    combination_key = (
                        args.database, stream_mode, int(args.num_omic),
                        filter_type, int(k), int(args.num_gene), int(batch_size),
                        _normf(dropout_value), _normf(lr)
                    )

                    if combination_key in existing_combinations:
                        print(f"✅ Skipping already processed combo: {combination_key}")
                        continue

                    print(f"\n🔹 Processing: db={args.database} | stream={stream_mode} | "
                          f"filter={filter_type} | K={k} | G={args.num_gene} | "
                          f"BS={batch_size} | dp={dropout_value} | lr={lr}")

                    theta = None if stream_mode == "mlp" else compute_theta(int(k), filter_type)

                    all_y_true, all_y_pred, all_y_scores = [], [], []
                    fold_accs, fold_mF1s, fold_mPrecs, fold_mRecs = [], [], [], []
                    fold_rows = []

                    comb_start = time.perf_counter()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

                    # Use precomputed 'splits' for stability across all combos
                    for fold, (train_index, val_index) in enumerate(splits, start=1):
                        fold_t0 = time.perf_counter()
                        print(f"  Fold {fold}…")

                        # ---------------------- leak-free gene selection (TRAIN ONLY) ----------------------
                        gene_list, gene_idx = utilsdata.select_top_genes_from_train_fold(
                            expr_all_df, non_null_index_path, train_index, args.num_gene
                        )
                        G_fold = len(gene_list)

                        # -------- SAVE PER-FOLD GENE LIST + FOLD META (only once per fold) --------
                        csv_path = os.path.join(gene_dir_root, f"fold_{fold:02d}_genes.csv")
                        xlsx_path = os.path.join(gene_dir_root, f"fold_{fold:02d}_genes.xlsx")
                        meta_train_path = os.path.join(fold_meta_dir, f"fold_{fold:02d}_train_idx.npy")
                        meta_val_path   = os.path.join(fold_meta_dir, f"fold_{fold:02d}_val_idx.npy")
                        manifest_txt    = os.path.join(gene_dir_root, f"fold_{fold:02d}_manifest.txt")

                        # Write once per fold: if CSV already exists, we assume this fold is saved.
                        if not os.path.exists(csv_path):
                            # CSV (robust)
                            pd.Series(gene_list, name="gene").to_csv(csv_path, index=False)

                            # Optional XLSX (silently skip if engine missing)
                            try:
                                with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                                    pd.DataFrame({"gene": gene_list}).to_excel(writer, index=False, sheet_name="genes")
                                    pd.DataFrame([{
                                        "cv_mode":"single",
                                        "database": args.database,
                                        "stream_mode": stream_mode,
                                        "num_omic": int(args.num_omic),
                                        "num_gene": int(args.num_gene),
                                        "seed": int(args.seed),
                                        "fold": int(fold)
                                    }]).to_excel(writer, index=False, sheet_name="manifest")
                            except Exception:
                                pass

                            # Train/val indices (needed for co-expression recompute without leakage)
                            np.save(meta_train_path, np.asarray(train_index, dtype=np.int64))
                            np.save(meta_val_path,   np.asarray(val_index,   dtype=np.int64))

                            # Tiny plain-text manifest (handy if you browse the folder)
                            try:
                                with open(manifest_txt, "w", encoding="utf-8") as mf:
                                    mf.write(
                                        f"fold={int(fold)}\ncv_mode=single\n"
                                        f"database={args.database}\nstream_mode={stream_mode}\n"
                                        f"num_omic={int(args.num_omic)}\nnum_gene={int(args.num_gene)}\n"
                                        f"seed={int(args.seed)}\n"
                                    )
                            except Exception:
                                pass
                            print(f"   🧾 Saved gene list + meta → {gene_dir_root}")
                        # -------------------------------------------------------------------------------

                        # -------------------------- build data for this fold --------------------------
                        if args.num_omic == 1:
                            A_sel, X_all, y_all, _ = utilsdata.build_fold_data_singleomics(
                                expr_all_df, adjacency_matrix_file, gene_list, gene_idx,
                                singleton=args.singleton, non_null_index_path=non_null_index_path
                            )
                        else:
                            A_sel, X_all, y_all, _ = utilsdata.build_fold_data_multiomics(
                                expr_all_df, cnv_all_df, adjacency_matrix_file, gene_list, gene_idx,
                                singleton=args.singleton, non_null_index_path=non_null_index_path
                            )

                        # ------- NEW: shape/channel assertions (pre-normalization) -------
                        assert X_all.ndim in (2, 3), f"Unexpected X_all shape: {X_all.shape}"
                        if X_all.ndim == 2:
                            X_all = X_all[:, :, None]
                        assert X_all.shape[2] == int(args.num_omic), \
                            f"Channel mismatch: X_all has C={X_all.shape[2]} but num_omic={args.num_omic}"
                        if not first_fold_info_printed:
                            print(f"ℹ️  First-fold tensor shape: X_all={X_all.shape} (N,G,C), F_0={F_0}")
                            first_fold_info_printed = True
                        # ------------------------------------------------------------------

                        # Laplacian for this fold (sparse, [0,1]-scaled)
                        if stream_mode == "mlp":
                            L_list = None
                        else:
                            if hasattr(A_sel, "data") and A_sel.data.size > 0:
                                A_max = float(A_sel.data.max())
                            else:
                                A_max = 0.0
                            A_norm = A_sel if A_max == 0.0 else (A_sel / A_max)
                            L_sp = graph_laplacian(A_norm, normalized=True)
                            L_sp = rescale_to_unit_interval(L_sp)
                            L_torch = utilsdata.sparse_mx_to_torch_sparse_tensor(L_sp).to(device)
                            L_list = [L_torch]

                        # ===== TRAIN-ONLY NORMALIZATION per global NORM_METHOD =====
                        X_all_norm, _norm_stats = utilsdata.normalize_by_train(
                            X_all, train_index=np.asarray(train_index, dtype=int), method=NORM_METHOD
                        )
                        # Re-assert after normalization
                        assert X_all_norm.ndim == 3, f"Post-norm X_all should be 3D, got {X_all_norm.shape}"
                        assert X_all_norm.shape[2] == int(args.num_omic), \
                            f"Post-norm channel mismatch: C={X_all_norm.shape[2]} vs num_omic={args.num_omic}"

                        Xtr_np = X_all_norm[train_index].astype(np.float32, copy=False)
                        Xva_np = X_all_norm[val_index].astype(np.float32, copy=False)
                        # ===========================================================

                        X_train = torch.tensor(Xtr_np, dtype=torch.float32)
                        y_train = torch.tensor(y_all[train_index], dtype=torch.long)
                        X_val   = torch.tensor(Xva_np, dtype=torch.float32)
                        y_val   = torch.tensor(y_all[val_index], dtype=torch.long)

                        _gen = torch.Generator().manual_seed(int(args.seed) + fold)
                        _pin = torch.cuda.is_available()

                        # ---------- class-balanced sampler (TRAIN ONLY, softened) ----------
                        y_train_np = y_train.cpu().numpy()
                        cls_counts = np.bincount(y_train_np, minlength=out_dim).astype(np.float32)
                        cls_counts[cls_counts == 0] = 1.0
                        alpha = 0.5
                        cls_w_samp = (1.0 / cls_counts) ** alpha
                        sample_w   = cls_w_samp[y_train_np]

                        sampler = WeightedRandomSampler(
                            weights=torch.tensor(sample_w, dtype=torch.double),
                            num_samples=len(sample_w),
                            replacement=True
                        )

                        train_loader = Data.DataLoader(
                            Data.TensorDataset(X_train, y_train),
                            batch_size=int(batch_size),
                            sampler=sampler,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=_pin,
                            generator=_gen
                        )
                        val_loader = Data.DataLoader(
                            Data.TensorDataset(X_val, y_val),
                            batch_size=int(batch_size), shuffle=False,
                            num_workers=0, pin_memory=_pin
                        )

                        # ---------- model (per fold) ----------
                        net_params_fold = [F_0, G_fold, CL1_F, int(k), FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
                        net = Graph_GCN(net_params_fold, stream_mode=stream_mode).to(device)

                        def _weight_init(m):
                            if isinstance(m, (nn.Conv2d, nn.Linear)):
                                nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None:
                                    nn.init.zeros_(m.bias)
                        net.apply(_weight_init)

                        # ---------- class-weighted loss (from y_train) ----------
                        with torch.no_grad():
                            counts = torch.bincount(y_train, minlength=out_dim).clamp_min(1)
                            inv = 1.0 / counts.float()
                            class_weights = (inv / inv.sum() * out_dim).to(device)

                        # LR tweak for MLP-only
                        lr_this = 1e-3 if (stream_mode == "mlp") else float(lr)

                        # ---------- optimizer, loss (label smoothing), warmup→cosine LR ----------
                        optimizer = optim.AdamW(net.parameters(), lr=lr_this, weight_decay=3e-5)

                        try:
                            criterion = nn.CrossEntropyLoss(
                                weight=class_weights.to(device=device, dtype=torch.float32),
                                label_smoothing=0.02
                            )
                        except TypeError:
                            criterion = nn.CrossEntropyLoss(
                                weight=class_weights.to(device=device, dtype=torch.float32)
                            )

                        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
                        warmup_epochs = max(3, int(0.05 * int(args.epochs)))
                        cosine_epochs = max(1, int(args.epochs) - warmup_epochs)
                        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
                        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
                        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

                        # ---------- train ----------
                        for _epoch in range(int(args.epochs)):
                            train_model(train_loader, net, optimizer, criterion,
                                        device, float(dropout_value), L_list, theta)
                            scheduler.step()

                        # ---------- final fold evaluation ----------
                        y_true, y_scores = test_model(val_loader, net, device, L_list, theta)
                        y_pred = np.argmax(y_scores, axis=1)

                        all_y_true.extend(y_true)
                        all_y_pred.extend(y_pred)
                        all_y_scores.extend(y_scores)

                        fold_acc = accuracy_score(y_true, y_pred)
                        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                        fold_mF1   = rep['macro avg']['f1-score']
                        fold_mPrec = rep['macro avg']['precision']
                        fold_mRec  = rep['macro avg']['recall']
                        print(f"    Fold {fold} Acc: {fold_acc:.4f}")

                        fold_accs.append(fold_acc)
                        fold_mF1s.append(fold_mF1)
                        fold_mPrecs.append(fold_mPrec)
                        fold_mRecs.append(fold_mRec)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        fold_time_sec = time.perf_counter() - fold_t0
                        rss_now_mb = (process.memory_info().rss / (1024**2)) if _psutil_ok and process is not None else np.nan
                        vram_now_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

                        fold_rows.append({
                            "database": args.database, "stream_mode": stream_mode, "num_omic": args.num_omic,
                            "filter_type": filter_type, "k": int(k), "num_genes": int(args.num_gene),
                            "batch_size": int(batch_size), "dropout": float(dropout_value), "lr": float(lr),
                            "fold": int(fold), "accuracy": float(fold_acc), "macro_f1": float(fold_mF1),
                            "macro_precision": float(fold_mPrec), "macro_recall": float(fold_mRec),
                            "fold_time_sec": float(fold_time_sec), "rss_mb": float(rss_now_mb), "vram_peak_mb_sofar": float(vram_now_mb),
                            "cv_mode": "single", "epochs": int(args.epochs), "num_gene": int(args.num_gene)
                        })

                        # cleanup
                        del X_train, y_train, X_val, y_val, train_loader, val_loader
                        del X_all, y_all, A_sel
                        if stream_mode != "mlp":
                            del L_sp, L_torch, L_list
                        del net, optimizer, criterion, sampler
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # ----- aggregate folds -----
                    acc_mean   = float(np.mean(fold_accs))
                    acc_std    = float(np.std(fold_accs,   ddof=1)) if len(fold_accs) > 1 else 0.0
                    mF1_mean   = float(np.mean(fold_mF1s))
                    mF1_std    = float(np.std(fold_mF1s,   ddof=1)) if len(fold_mF1s) > 1 else 0.0
                    mPrec_mean = float(np.mean(fold_mPrecs))
                    mPrec_std  = float(np.std(fold_mPrecs, ddof=1)) if len(fold_mPrecs) > 1 else 0.0
                    mRec_mean  = float(np.mean(fold_mRecs))
                    mRec_std   = float(np.std(fold_mRecs,  ddof=1)) if len(fold_mRecs) > 1 else 0.0
                    print(f"   Fold-wise: Acc {acc_mean:.4f} ± {acc_std:.4f} | mF1 {mF1_mean:.4f} ± {mF1_std:.4f}")

                    # write/update per-fold CSV (append + de-dupe by identifying columns)
                    per_fold_df = pd.DataFrame(fold_rows)
                    key_cols = ["database","stream_mode","num_omic","filter_type","k","num_genes",
                                "batch_size","dropout","lr","fold","cv_mode","epochs","num_gene"]

                    if os.path.exists(per_fold_csv) and os.path.getsize(per_fold_csv) > 0:
                        _old = pd.read_csv(per_fold_csv)
                        per_fold_df = pd.concat([_old, per_fold_df], ignore_index=True)

                    per_fold_df = per_fold_df.sort_index().drop_duplicates(subset=key_cols, keep="last")
                    per_fold_df.to_csv(per_fold_csv, index=False)

                    # pooled predictions over all folds (final single-CV numbers)
                    all_y_true   = np.array(all_y_true)
                    all_y_pred   = np.array(all_y_pred)
                    all_y_scores = np.array(all_y_scores)

                    avg_accuracy = accuracy_score(all_y_true, all_y_pred)
                    report_dict  = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
                    precision    = report_dict["weighted avg"]["precision"]
                    recall       = report_dict["weighted avg"]["recall"]
                    macro_f1     = report_dict["macro avg"]["f1-score"]

                    # pooled macro-AUC (safe if some classes absent)
                    macro_auc_pooled = np.nan
                    try:
                        yb = label_binarize(all_y_true, classes=list(range(out_dim)))
                        aucs = []
                        for c in range(out_dim):
                            if yb[:, c].sum() > 0:
                                fpr_c, tpr_c, _ = roc_curve(yb[:, c], all_y_scores[:, c])
                                aucs.append(auc(fpr_c, tpr_c))
                        if len(aucs) > 0:
                            macro_auc_pooled = float(np.mean(aucs))
                    except Exception:
                        pass

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    comb_time_sec = time.perf_counter() - comb_start
                    peak_vram_mb  = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

                    # --- summary row (ONLY write to csv_file_path) ---
                    new_result = {
                        "database": args.database, "stream_mode": stream_mode, "num_omic": int(args.num_omic),
                        "filter_type": filter_type, "k": int(k), "num_genes": int(args.num_gene),
                        "batch_size": int(batch_size), "dropout": float(dropout_value), "lr": float(lr),
                        "accuracy": float(avg_accuracy), "precision": float(precision), "recall": float(recall), "macro_f1": float(macro_f1),
                        "macro_auc": (np.nan if np.isnan(macro_auc_pooled) else float(macro_auc_pooled)),
                        "acc_mean": acc_mean, "acc_std": acc_std,
                        "macro_f1_mean": mF1_mean, "macro_f1_std": mF1_std,
                        "macro_precision_mean": mPrec_mean, "macro_precision_std": mPrec_std,
                        "macro_recall_mean": mRec_mean, "macro_recall_std": mRec_std,
                        "run_time_sec": float(comb_time_sec),
                        "peak_rss_mb": (process.memory_info().rss / (1024**2)) if _psutil_ok and process is not None else np.nan,
                        "peak_vram_mb": float(peak_vram_mb),
                        "cv_mode": "single",
                        "epochs": int(args.epochs),
                        "num_gene": int(args.num_gene)
                    }

                    # Append then de-dupe (float-stable)
                    df_existing = pd.concat([df_existing, pd.DataFrame([new_result])], ignore_index=True)

                    # round float keys
                    for _col in ("dropout", "lr"):
                        if _col in df_existing.columns:
                            df_existing[_col] = pd.to_numeric(df_existing[_col], errors="coerce").round(8)

                    # int keys -> Int64
                    for _col in ("k", "num_genes", "batch_size", "epochs", "num_gene", "num_omic"):
                        if _col in df_existing.columns:
                            df_existing[_col] = pd.to_numeric(df_existing[_col], errors="coerce").astype("Int64")

                    # str keys -> strip quotes/space
                    for _col in ("database", "stream_mode", "filter_type", "cv_mode"):
                        if _col in df_existing.columns:
                            df_existing[_col] = df_existing[_col].astype(str).str.strip().str.strip('"').str.strip("'")

                    sum_keys = [
                        "database","stream_mode","num_omic","filter_type","k","num_genes",
                        "batch_size","dropout","lr","cv_mode","epochs","num_gene"
                    ]
                    df_existing = df_existing.sort_index().drop_duplicates(subset=sum_keys, keep="last")
                    df_existing.to_csv(csv_file_path, index=False)

                    # Keep existing_combinations in sync (normalize floats)
                    existing_combinations.add((
                        args.database, stream_mode, int(args.num_omic),
                        filter_type, int(k), int(args.num_gene), int(batch_size),
                        round(float(dropout_value), 8), round(float(lr), 8)
                    ))

                    print(f"✅ CSVs updated →\n   per-fold: {per_fold_csv}\n   pooled:   {csv_file_path}\n"
                          f"   Pooled Acc={avg_accuracy:.4f} | Macro-F1={macro_f1:.4f} | "
                          f"mean Acc={acc_mean:.4f}±{acc_std:.4f}")


# =========================================================
#     EXPORT PER-FOLD + SUMMARY TO A SINGLE EXCEL
#     + SELECT BEST ROW FOR CURRENT CONTEXT
#     + WEIGHTS MANIFEST (+ copy best weights if found)
# =========================================================

# --- extra imports this block needs ---
import shutil
from glob import glob
import matplotlib.pyplot as plt

def _ensure_pdf_from_noext(path_no_ext: str) -> str:
    """Ensure a .pdf extension on a given base path (no extension)."""
    if path_no_ext.lower().endswith(".pdf"):
        return path_no_ext
    return f"{path_no_ext}.pdf"

# =========================================================
#   SUMMARY EXCEL + ABLATION + MAIN-TEXT TABLE + MANIFEST
# =========================================================

# Defensive default for df_summary referenced after try/except
df_summary = df_existing.copy() if 'df_existing' in globals() and isinstance(df_existing, pd.DataFrame) else pd.DataFrame()

try:
    # ---- Guard columns for summary table ----
    if 'df_existing' not in globals() or not isinstance(df_existing, pd.DataFrame):
        df_existing = pd.DataFrame()
    if "cv_mode" not in df_existing.columns:
        df_existing["cv_mode"] = "single"

    # ---- Load per-fold CSV (if present), else build an empty shell ----
    if os.path.exists(per_fold_csv) and os.path.getsize(per_fold_csv) > 0:
        df_folds_all = pd.read_csv(per_fold_csv)
        df_folds_all.columns = df_folds_all.columns.str.strip()
    else:
        df_folds_all = pd.DataFrame(columns=[
            "database","stream_mode","num_omic","filter_type","k","num_genes",
            "batch_size","dropout","lr","fold","accuracy","macro_precision",
            "macro_recall","macro_f1","fold_time_sec","rss_mb","vram_peak_mb_sofar",
            "cv_mode","epochs","num_gene"
        ])

    if "cv_mode" not in df_folds_all.columns:
        df_folds_all["cv_mode"] = "single"

    # ---- Restrict to current context (db/stream/omics) ----
    df_folds = df_folds_all[
        (df_folds_all.get("database") == args.database) &
        (df_folds_all.get("stream_mode") == stream_mode) &
        (pd.to_numeric(df_folds_all.get("num_omic"), errors="coerce") == int(args.num_omic))
    ].copy()

    # ---- Numeric coercions for fold metrics ----
    num_cols = ["k","num_genes","batch_size","dropout","lr","fold",
                "accuracy","macro_precision","macro_recall","macro_f1",
                "fold_time_sec","rss_mb","vram_peak_mb_sofar","epochs","num_gene","num_omic"]
    for c in num_cols:
        if c in df_folds.columns:
            df_folds[c] = pd.to_numeric(df_folds[c], errors="coerce")

    # ---- Long view for folds (stable sort by id columns) ----
    id_cols = ["database","stream_mode","num_omic","filter_type","k",
               "num_genes","batch_size","dropout","lr","fold","cv_mode","epochs","num_gene"]
    metric_cols = ["accuracy","macro_precision","macro_recall","macro_f1",
                   "fold_time_sec","rss_mb","vram_peak_mb_sofar"]
    cols_long = [c for c in id_cols if c in df_folds.columns] + \
                [c for c in metric_cols if c in df_folds.columns]
    if not df_folds.empty:
        df_folds_long = df_folds[cols_long].sort_values(
            [c for c in id_cols if c in df_folds.columns], kind="mergesort"
        )
    else:
        df_folds_long = df_folds

    # ---- Wide view for folds (accuracy/prec/rec per fold) ----
    folds_wide = None
    if not df_folds_long.empty and "fold" in df_folds_long.columns:
        try:
            idx_cols = ["database","stream_mode","num_omic","filter_type","k",
                        "num_genes","batch_size","dropout","lr","cv_mode","epochs","num_gene"]
            idx_cols = [c for c in idx_cols if c in df_folds_long.columns]
            folds_wide = (
                df_folds_long
                .pivot_table(
                    index=idx_cols,
                    columns="fold",
                    values=[c for c in ["accuracy","macro_precision","macro_recall"] if c in df_folds_long.columns],
                    aggfunc="first"
                )
                .sort_index()
            )
            folds_wide.columns = [f"{met}_fold{fold}" for met, fold in folds_wide.columns]
            folds_wide = folds_wide.reset_index()
        except Exception:
            folds_wide = None

    # ---- Summary sheet = df_existing with pretty mean±std columns ----
    df_summary = df_existing.copy()
    if "cv_mode" not in df_summary.columns:
        df_summary["cv_mode"] = "single"

    # numeric coercions for summary
    for c in ["accuracy","macro_f1","macro_auc",
              "acc_mean","acc_std","macro_f1_mean","macro_f1_std",
              "macro_precision_mean","macro_precision_std","macro_recall_mean","macro_recall_std",
              "run_time_sec","peak_rss_mb","peak_vram_mb",
              "k","num_genes","batch_size","dropout","lr","epochs","num_gene","num_omic"]:
        if c in df_summary.columns:
            df_summary[c] = pd.to_numeric(df_summary[c], errors="coerce")

    def _pm(df, mean_col, std_col, out_col):
        if mean_col in df.columns and std_col in df.columns:
            df[out_col] = df.apply(
                lambda r: (f"{float(r[mean_col]):.4f} ± {float(r[std_col]):.4f}")
                if pd.notnull(r[mean_col]) and pd.notnull(r[std_col]) else "",
                axis=1
            )

    _pm(df_summary, "acc_mean", "acc_std", "acc_mean_std")
    _pm(df_summary, "macro_precision_mean", "macro_precision_std", "precision_mean_std")
    _pm(df_summary, "macro_recall_mean", "macro_recall_std", "recall_mean_std")
    _pm(df_summary, "macro_f1_mean", "macro_f1_std", "macro_f1_mean_std")

    # ---- Choose selection metric (macro_f1 preferred, else accuracy) ----
    select_metric = args.score if hasattr(args, "score") and (args.score in df_summary.columns) else (
        "macro_f1" if "macro_f1" in df_summary.columns else "accuracy"
    )

    df_view = df_summary[
        (df_summary.get("database") == args.database) &
        (df_summary.get("stream_mode") == stream_mode) &
        (pd.to_numeric(df_summary.get("num_omic"), errors="coerce") == int(args.num_omic)) &
        (df_summary.get("cv_mode") == "single")
    ].copy()

    best_result = None
    df_best_sheet = pd.DataFrame()
    if not df_view.empty and select_metric in df_view.columns:
        sort_cols, ascending = [select_metric], [False]
        if select_metric != "accuracy" and "accuracy" in df_view.columns:
            sort_cols.append("accuracy"); ascending.append(False)
        df_ranked = df_view.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
        df_ranked["rank"] = np.arange(1, len(df_ranked) + 1)
        df_ranked["is_best"] = df_ranked["rank"].eq(1)
        keep_cols = [
            "database","stream_mode","num_omic","cv_mode",
            "filter_type","k","num_genes","batch_size","dropout","lr",
            "accuracy","macro_f1","macro_auc",
            "acc_mean","acc_std","macro_f1_mean","macro_f1_std",
            "macro_precision_mean","macro_precision_std",
            "macro_recall_mean","macro_recall_std",
            "run_time_sec","peak_rss_mb","peak_vram_mb",
            "epochs","num_gene","rank","is_best"
        ]
        keep_cols = [c for c in keep_cols if c in df_ranked.columns]
        df_best_sheet = df_ranked[keep_cols].copy()
        best_result = df_ranked.iloc[0].to_dict()

    # ---- Write Excel with multiple sheets ----
    xlsx_path = os.path.join(OutputDir, "results_with_folds.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df_summary.to_excel(writer, index=False, sheet_name="summary")
            if not df_folds_long.empty:
                df_folds_long.to_excel(writer, index=False, sheet_name="folds_long")
            if folds_wide is not None:
                folds_wide.to_excel(writer, index=False, sheet_name="folds_wide")
            if not df_best_sheet.empty:
                df_best_sheet.to_excel(writer, index=False, sheet_name="best_single")
    except Exception:
        with pd.ExcelWriter(xlsx_path) as writer:
            df_summary.to_excel(writer, index=False, sheet_name="summary")
            if not df_folds_long.empty:
                df_folds_long.to_excel(writer, index=False, sheet_name="folds_long")
            if folds_wide is not None:
                folds_wide.to_excel(writer, index=False, sheet_name="folds_wide")
            if not df_best_sheet.empty:
                df_best_sheet.to_excel(writer, index=False, sheet_name="best_single")

    print(f"📊 Excel written: {xlsx_path}")

    # ===== PATCH 2: Dual-stream ablation summary + concise main-text table =====
    try:
        if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
            df_all = pd.read_csv(csv_file_path)
            df_all.columns = df_all.columns.str.strip()

            # Filter to current db/omics and single-CV rows
            mask = (
                (df_all.get("database") == args.database) &
                (pd.to_numeric(df_all.get("num_omic"), errors="coerce") == int(args.num_omic)) &
                (df_all.get("cv_mode", "single") == "single")
            )
            df_ctx = df_all[mask].copy()

            # Make sure we can sort numerically
            for c in ["accuracy","macro_f1","macro_auc"]:
                if c in df_ctx.columns:
                    df_ctx[c] = pd.to_numeric(df_ctx[c], errors="coerce")

            # --- (A) Dual-stream ablation: pick best per stream_mode ---
            ablation_rows = []
            for sm in ["fusion", "gcn", "mlp"]:
                sub = df_ctx[df_ctx.get("stream_mode") == sm].copy()
                if sub.empty:
                    continue
                sort_cols = []
                if "macro_f1" in sub.columns: sort_cols.append("macro_f1")
                if "accuracy" in sub.columns: sort_cols.append("accuracy")
                if not sort_cols:
                    continue
                sub = sub.sort_values(sort_cols, ascending=[False]*len(sort_cols), kind="mergesort")
                best = sub.iloc[0].to_dict()
                ablation_rows.append({
                    "stream_mode": sm,
                    "filter_type": best.get("filter_type"),
                    "k": best.get("k"),
                    "batch_size": best.get("batch_size"),
                    "dropout": best.get("dropout"),
                    "lr": best.get("lr"),
                    "macro_f1": best.get("macro_f1"),
                    "accuracy": best.get("accuracy"),
                    "macro_auc": best.get("macro_auc") if "macro_auc" in sub.columns else np.nan,
                })

            ablation_df = pd.DataFrame(ablation_rows)
            ablation_csv = os.path.join(OutputDir, "ablation_streams_summary.csv")
            ablation_df.to_csv(ablation_csv, index=False)
            print(f"✅ Dual-stream ablation table written: {ablation_csv}")

            # --- (B) Concise main-text table for single-CV ---
            if not df_ctx.empty:
                sort_cols = []
                if "macro_f1" in df_ctx.columns: sort_cols.append("macro_f1")
                if "accuracy" in df_ctx.columns: sort_cols.append("accuracy")
                df_best = df_ctx.sort_values(sort_cols, ascending=[False]*len(sort_cols), kind="mergesort").iloc[0]

                rows = [
                    ["Metric","Value"],
                    ["macro_F1", f"{float(df_best.get('macro_f1')):.4f}" if pd.notnull(df_best.get('macro_f1')) else "NA"],
                    ["accuracy", f"{float(df_best.get('accuracy')):.4f}" if pd.notnull(df_best.get('accuracy')) else "NA"],
                    ["macro_AUC", f"{float(df_best.get('macro_auc')):.4f}" if pd.notnull(df_best.get('macro_auc')) else "NA"],
                    ["stream_mode", str(df_best.get("stream_mode"))],
                    ["filter_type", str(df_best.get("filter_type"))],
                    ["K", str(int(df_best.get("k"))) if pd.notnull(df_best.get("k")) else ""],
                ]

                f1_cols = [c for c in df_best.index if isinstance(c, str) and c.startswith("f1_c")]
                if f1_cols:
                    pairs = []
                    for c in f1_cols:
                        try:
                            cid = int(c.split("f1_c",1)[1])
                            val = float(df_best[c])
                            if pd.notnull(val):
                                pairs.append((cid, val))
                        except Exception:
                            pass
                    pairs.sort(key=lambda t: -t[1])
                    for cid, val in pairs[:3]:
                        pr = df_best.get(f"prec_c{cid}")
                        rc = df_best.get(f"rec_c{cid}")
                        rows.append([f"class {cid} F1", f"{float(val):.4f}"])
                        rows.append([f"class {cid} precision", f"{float(pr):.4f}" if pd.notnull(pr) else "NA"])
                        rows.append([f"class {cid} recall",    f"{float(rc):.4f}" if pd.notnull(rc) else "NA"])

                main_small = pd.DataFrame(rows[1:], columns=rows[0])
                main_csv = os.path.join(OutputDir, "main_text_single_cv.csv")
                main_small.to_csv(main_csv, index=False)
                print(f"✅ Concise single-CV main-text table written: {main_csv}")

    except Exception as e:
        print(f"⚠️ Patch 2 (ablation/main-text) skipped due to: {e}")
    # ===== END PATCH 2 =====

    # ---- Persist best row as a quick text artifact ----
    best_model_txt = os.path.join(OutputDir, "bestmodel.txt")
    if best_result is not None:
        with open(best_model_txt, "w", encoding="utf-8") as f:
            f.write(f"selection_metric: {select_metric}\n")
            for k, v in best_result.items():
                if isinstance(v, (str, int, float, np.integer, np.floating)):
                    f.write(f"{k}: {v}\n")
        print(f"✅ Best model (single-CV context) saved to {best_model_txt}")

except Exception as e:
    print(f"⚠️ Excel export failed: {e}")

# =========================================================
#     BEST RESULT SUMMARY + SAFE ARTIFACT COPY + WEIGHTS
# =========================================================

if 'best_result' not in globals() or best_result is None:
    try:
        selm = args.score if ('args' in globals() and hasattr(args, 'score')) else 'macro_f1'
        if not ('df_existing' in globals() and isinstance(df_existing, pd.DataFrame) and not df_existing.empty):
            best_result = None
        else:
            df_ctx = df_existing[
                (df_existing.get("database") == args.database) &
                (df_existing.get("stream_mode") == stream_mode) &
                (pd.to_numeric(df_existing.get("num_omic"), errors="coerce") == int(args.num_omic)) &
                (df_existing.get("cv_mode") == "single")
            ].copy()
            if not df_ctx.empty:
                metric = selm if selm in df_ctx.columns else ("macro_f1" if "macro_f1" in df_ctx.columns else "accuracy")
                by_cols = [metric] + (["accuracy"] if metric != "accuracy" and "accuracy" in df_ctx.columns else [])
                df_ctx = df_ctx.sort_values(by_cols, ascending=False, kind="mergesort")
                best_result = df_ctx.iloc[0].to_dict()
            else:
                best_result = None
    except Exception:
        best_result = None

best_model_txt = os.path.join(OutputDir, "bestmodel.txt")
if best_result:
    with open(best_model_txt, "w", encoding="utf-8") as f:
        sel_metric = (args.score if hasattr(args, "score") else
                      ("macro_f1" if "macro_f1" in best_result else "accuracy"))
        f.write(f"selection_metric: {sel_metric}\n")
        for key, value in best_result.items():
            if isinstance(value, (str, int, float, np.integer, np.floating)):
                f.write(f"{key}: {value}\n")
    print(f"✅ Best model configuration saved to {best_model_txt}")

    print("\n✅ Best Overall Configuration (current context):")
    keys_pretty = ["database","stream_mode","num_omic","cv_mode",
                   "filter_type","k","num_genes","batch_size","dropout","lr",
                   "accuracy","macro_f1","macro_auc"]
    shown = set()
    for k in keys_pretty:
        if k in best_result:
            print(f"{k}: {best_result[k]}")
            shown.add(k)
    for k, v in best_result.items():
        if k not in shown and isinstance(v, (str, int, float, np.integer, np.floating)):
            print(f"{k}: {v}")

    os.makedirs(bestmodel_dir, exist_ok=True)
    existing_best_pdfs = glob(os.path.join(bestmodel_dir, "best_*.pdf"))
    if existing_best_pdfs:
        print(f"📂 {len(existing_best_pdfs)} best_* PDFs already present in {bestmodel_dir}.")
else:
    print("❌ No best result found.")

# --------------------- WEIGHTS MANIFEST ---------------------
try:
    manifest_rows = []

    # Single-CV weights: OutputDir/single_cv_cache/<run_tag>/fold_X/weights_final.pt
    scv_root = os.path.join(OutputDir, "single_cv_cache")
    if os.path.isdir(scv_root):
        for run_dir in glob(os.path.join(scv_root, "*")):
            run_tag = os.path.basename(run_dir)
            for w in glob(os.path.join(run_dir, "fold_*", "weights_final.pt")):
                manifest_rows.append({
                    "phase": "single_cv_fold",
                    "path": w,
                    "run_tag": run_tag
                })

    # NEW: NCV inner-split weights: OutputDir/ncv_cache/<run_tag>/weights_final.pt
    ncv_cache_root = os.path.join(OutputDir, "ncv_cache")
    if os.path.isdir(ncv_cache_root):
        for w in glob(os.path.join(ncv_cache_root, "*", "weights_final.pt")):
            run_tag = os.path.basename(os.path.dirname(w))
            manifest_rows.append({
                "phase": "ncv_inner",
                "path": w,
                "run_tag": run_tag
            })

    # NCV outer-best weights: OutputDir/nested_cv/outer_fold_*/best_outer_artifacts/weights_final.pt
    ncv_root = os.path.join(OutputDir, "nested_cv")
    if os.path.isdir(ncv_root):
        for w in glob(os.path.join(ncv_root, "outer_fold_*", "best_outer_artifacts", "weights_final.pt")):
            run_tag = os.path.basename(os.path.dirname(os.path.dirname(w)))  # outer_fold_X
            manifest_rows.append({
                "phase": "ncv_outer_best",
                "path": w,
                "run_tag": run_tag
            })

    weights_manifest = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(OutputDir, "weights_manifest.csv")
    weights_manifest.to_csv(manifest_csv, index=False)
    print(f"💾 Weights manifest written: {manifest_csv}")

    # Try to copy the matching best weights into bestmodel/
    def _match_best_single_cv_weight(row_dict):
        """Build the run_tag we used during single-CV saving and try to match it."""
        try:
            rt = (
                f"db-{row_dict.get('database')}_mode-{row_dict.get('stream_mode')}_omic-{int(row_dict.get('num_omic'))}"
                f"_filt-{row_dict.get('filter_type')}_K-{int(row_dict.get('k'))}"
                f"_bs-{int(row_dict.get('batch_size'))}_dp-{float(row_dict.get('dropout'))}"
                f"_lr-{float(row_dict.get('lr'))}"
            )
            return rt
        except Exception:
            return None

    best_weight_src = None
    if best_result:
        tag = _match_best_single_cv_weight(best_result)
        if tag:
            candidate = glob(os.path.join(OutputDir, "single_cv_cache", tag, "fold_*", "weights_final.pt"))
            if candidate:
                best_weight_src = sorted(candidate)[0]

        # Fallback: outer-best NCV
        if not best_weight_src and os.path.isdir(ncv_root):
            candidates = glob(os.path.join(ncv_root, "outer_fold_*", "best_outer_artifacts", "weights_final.pt"))
            if candidates:
                best_weight_src = sorted(candidates)[0]

        # Optional extra fallback: any inner NCV checkpoint
        if not best_weight_src and os.path.isdir(ncv_cache_root):
            candidates = glob(os.path.join(ncv_cache_root, "*", "weights_final.pt"))
            if candidates:
                best_weight_src = sorted(candidates)[0]

    if best_weight_src:
        dst = os.path.join(bestmodel_dir, "best_weights.pt")
        try:
            shutil.copy2(best_weight_src, dst)
            print(f"✅ Best weights copied to: {dst}")
        except Exception as e:
            print(f"⚠️ Could not copy best weights: {e}")
    else:
        print("⚠️ No matching weights file found for the best configuration (this is OK if --savemodel was not used).")

except Exception as e:
    print(f"⚠️ Weights manifest/copy step skipped: {e}")

# =========================================================
#                   Visualization from CSV
# =========================================================
results_csv = csv_file_path if os.path.exists(csv_file_path) else os.path.join(OutputDir, "00finalcoexsingle2000.csv")

if os.path.exists(results_csv) and os.path.getsize(results_csv) > 0:
    df_results = pd.read_csv(results_csv)
    df_results.columns = df_results.columns.str.strip()
else:
    print(f"⚠️ No results CSV found at {results_csv}. Creating an empty DataFrame.")
    df_results = pd.DataFrame(columns=[
        "database","stream_mode","num_omic",
        "filter_type","k","num_genes","batch_size","dropout","lr",
        "accuracy","precision","recall","macro_f1","macro_auc",
    ])

needed = {"database","stream_mode","num_omic","filter_type","k","num_genes","batch_size","dropout","lr","accuracy"}
missing = needed - set(df_results.columns)
if missing:
    print(f"⚠️ Missing columns in results CSV: {sorted(list(missing))}. Plots may be limited.")

if df_results.empty or not {"database","stream_mode","num_omic","filter_type"}.issubset(df_results.columns):
    print("❌ No data available for plotting. Skipping visualization.")
else:
    df_view = df_results[
        (df_results.get("database") == args.database) &
        (df_results.get("stream_mode") == stream_mode) &
        (pd.to_numeric(df_results.get("num_omic"), errors="coerce") == int(args.num_omic))
    ].copy()

    if df_view.empty:
        print(f"⚠️ No rows for db={args.database}, stream={stream_mode}, num_omic={args.num_omic}. Skipping plots.")
    else:
        for col in ["k","batch_size","dropout","lr","accuracy"]:
            if col in df_view.columns:
                df_view[col] = pd.to_numeric(df_view[col], errors="coerce")

        best_filter = (
            best_result.get("filter_type")
            if (best_result and
                best_result.get("database", args.database) == args.database and
                best_result.get("stream_mode", stream_mode) == stream_mode and
                int(best_result.get("num_omic", args.num_omic)) == int(args.num_omic))
            else None
        )

        def save_pdf_and_copy(fig_path_no_ext, is_best):
            pdf_path = _ensure_pdf_from_noext(fig_path_no_ext)
            plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
            plt.close()
            print(f"✅ Plot saved: {pdf_path}")
            if is_best:
                best_pdf = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
                os.makedirs(bestmodel_dir, exist_ok=True)
                shutil.copy2(pdf_path, best_pdf)
                print(f"✅ Best plot copied to bestmodel: {best_pdf}")

        # ----- 2D: Accuracy vs K (per filter) -----
        for ft in df_view["filter_type"].dropna().unique():
            df_ft = df_view[df_view["filter_type"] == ft]
            if not {"k","accuracy"}.issubset(df_ft.columns) or df_ft.empty:
                continue
            avg_acc_per_k = df_ft.groupby("k", as_index=True)["accuracy"].mean().sort_index()

            plt.figure(figsize=(8, 6))
            plt.plot(avg_acc_per_k.index, avg_acc_per_k.values, marker="o", linewidth=2)
            plt.xlabel("K (Bernstein Polynomial Order)", fontsize=14)
            plt.ylabel("Validation Accuracy", fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.xticks(fontsize=12); plt.yticks(fontsize=12)
            save_pdf_and_copy(
                os.path.join(hyperparam_dir, f"{args.database}_{stream_mode}_{ft}_k_vs_accuracy"),
                is_best=(ft == best_filter)
            )

        # ----- 2D: Combined Accuracy vs K (all filters) -----
        plt.figure(figsize=(10, 8))
        plotted_any = False
        y_min, y_max = np.inf, -np.inf

        for ft in df_view["filter_type"].dropna().unique():
            df_ft = df_view[df_view["filter_type"] == ft]
            if not {"k","accuracy"}.issubset(df_ft.columns) or df_ft.empty:
                continue
            avg_acc_per_k = df_ft.groupby("k")["accuracy"].mean().sort_index()
            if avg_acc_per_k.empty:
                continue
            plt.plot(avg_acc_per_k.index, avg_acc_per_k.values, marker="o", label=str(ft))
            y_min = min(y_min, float(np.nanmin(avg_acc_per_k.values)))
            y_max = max(y_max, float(np.nanmax(avg_acc_per_k.values)))
            plotted_any = True

        plt.xlabel("K (Bernstein Polynomial Order)", fontsize=14)
        plt.ylabel("Validation Accuracy", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        if plotted_any and np.isfinite(y_min) and np.isfinite(y_max):
            plt.ylim(max(0.0, y_min - 0.02), min(1.0, y_max + 0.02))
        if plotted_any:
            plt.legend(title="Filter Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        save_pdf_and_copy(
            os.path.join(hyperparam_dir, f"{args.database}_{stream_mode}_combined_k_vs_accuracy"),
            is_best=False
        )

        # ----- Helper to construct 3D surface arrays -----
        def _3d_surface(df, idx_col, col_col, val_col="accuracy"):
            pv = (df.groupby([idx_col, col_col])[val_col]
                    .mean()
                    .reset_index()
                    .pivot(index=idx_col, columns=col_col, values=val_col))
            pv = pv.sort_index().sort_index(axis=1)
            Xv = pv.columns.astype(float).values
            Yv = pv.index.astype(float).values
            X, Y = np.meshgrid(Xv, Yv)
            Z = pv.values.astype(float)
            return X, Y, Z

        # ----- 3D: Accuracy vs K & Batch Size -----
        for ft in df_view["filter_type"].dropna().unique():
            df_ft = df_view[df_view["filter_type"] == ft]
            if not {"k","batch_size","accuracy"}.issubset(df_ft.columns) or df_ft.empty:
                continue
            try:
                X, Y, Z = _3d_surface(df_ft, idx_col="k", col_col="batch_size", val_col="accuracy")
            except Exception as e:
                print(f"⚠️ Skipping 3D (K×Batch) for {ft}: {e}")
                continue

            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
            ax.set_xlabel("Batch Size", fontsize=14, labelpad=12, fontweight="bold")
            ax.set_ylabel("K", fontsize=14, labelpad=12, fontweight="bold")
            ax.set_zlabel("Accuracy", fontsize=14, labelpad=12, fontweight="bold")
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.12)
            ax.view_init(elev=25, azim=-60)
            fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)

            out_no_ext = os.path.join(hyperparam_dir, f"{args.database}_{stream_mode}_{ft}_accuracy_vs_k_bs_3d")
            pdf_path = _ensure_pdf_from_noext(out_no_ext)
            plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
            plt.close()
            print(f"✅ 3D plot (K vs Batch Size) saved for filter {ft}: {pdf_path}")
            if ft == best_filter:
                dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
                shutil.copy2(pdf_path, dst)
                print(f"✅ Best 3D plot (K vs Batch Size) copied to bestmodel: {dst}")

        # ----- 3D: Accuracy vs K & Dropout -----
        for ft in df_view["filter_type"].dropna().unique():
            df_ft = df_view[df_view["filter_type"] == ft]
            if not {"k","dropout","accuracy"}.issubset(df_ft.columns) or df_ft.empty:
                continue
            try:
                X, Y, Z = _3d_surface(df_ft, idx_col="k", col_col="dropout", val_col="accuracy")
            except Exception as e:
                print(f"⚠️ Skipping 3D (K×Dropout) for {ft}: {e}")
                continue

            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor="none")
            ax.set_xlabel("Dropout", fontsize=14, labelpad=12, fontweight="bold")
            ax.set_ylabel("K", fontsize=14, labelpad=12, fontweight="bold")
            ax.set_zlabel("Accuracy", fontsize=14, labelpad=12, fontweight="bold")
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.12)
            ax.view_init(elev=25, azim=-60)
            fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)

            out_no_ext = os.path.join(hyperparam_dir, f"{args.database}_{stream_mode}_{ft}_accuracy_vs_k_dropout_3d")
            pdf_path = _ensure_pdf_from_noext(out_no_ext)
            plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
            plt.close()
            print(f"✅ 3D plot (K vs Dropout) saved for filter {ft}: {pdf_path}")
            if ft == best_filter:
                dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
                shutil.copy2(pdf_path, dst)
                print(f"✅ Best 3D plot (K vs Dropout) copied to bestmodel: {dst}")

        # ----- 3D: Accuracy vs K & Learning Rate -----
        for ft in df_view["filter_type"].dropna().unique():
            df_ft = df_view[df_view["filter_type"] == ft]
            if not {"k","lr","accuracy"}.issubset(df_ft.columns) or df_ft.empty:
                continue
            try:
                X, Y, Z = _3d_surface(df_ft, idx_col="k", col_col="lr", val_col="accuracy")
            except Exception as e:
                print(f"⚠️ Skipping 3D (K×LR) for {ft}: {e}")
                continue

            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="none")
            ax.set_xlabel("Learning Rate", fontsize=14, labelpad=12, fontweight="bold")
            ax.set_ylabel("K", fontsize=14, labelpad=12, fontweight="bold")
            ax.set_zlabel("Accuracy", fontsize=14, labelpad=12, fontweight="bold")
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.12)
            ax.view_init(elev=25, azim=-60)
            fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)

            out_no_ext = os.path.join(hyperparam_dir, f"{args.database}_{stream_mode}_{ft}_accuracy_vs_k_lr_3d")
            pdf_path = _ensure_pdf_from_noext(out_no_ext)
            plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
            plt.close()
            print(f"✅ 3D plot (K vs Learning Rate) saved for filter {ft}: {pdf_path}")
            if ft == best_filter:
                dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
                shutil.copy2(pdf_path, dst)
                print(f"✅ Best 3D plot (K vs Learning Rate) copied to bestmodel: {dst}")

# ------------- quick VRAM/RSS recap from summary -------------
try:
    ctx = df_summary[
        (df_summary.get("database") == args.database) &
        (df_summary.get("stream_mode") == stream_mode) &
        (pd.to_numeric(df_summary.get("num_omic"), errors="coerce") == int(args.num_omic)) &
        (df_summary.get("cv_mode") == "single")
    ]
    if not ctx.empty and {"peak_rss_mb","peak_vram_mb"}.issubset(ctx.columns):
        print("📈 Peak memory (single-CV context): "
              f"RSS≈{float(ctx['peak_rss_mb'].max()):.1f} MB | "
              f"VRAM≈{float(ctx['peak_vram_mb'].max()):.1f} MB")
except Exception:
    pass
## ============================== SHAP PIPELINE (single-CV) ==============================
# This block builds train_data_all / labels / gene_names / L / best_net and runs SHAP.

# ===================== DROP-IN: SHAP PREP + EXPLAIN (GPU, 3000 samples, ≥20/class) =====================
import os, glob, warnings, copy
import numpy as np
import pandas as pd
from collections import defaultdict
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans as KMeans  # <<< faster at n>=3000
from coarsening import graph_laplacian, rescale_to_unit_interval

warnings.filterwarnings("ignore", category=UserWarning)
try:
    import shap
    # new-style maskers (safer memory use)
    _ = shap.maskers
except Exception as e:
    raise RuntimeError("shap is required. Install in your env:  pip install -U shap") from e

# ------------------ CONFIG ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reviewer-7 settings
NUM_SHAP_SAMPLES = 3000     # target total explained samples (cap by n)
BACKGROUND_K     = 50       # k for background subset (masker)
SAFE_NSAMPLES    = 128      # SHAP internal sampling per instance (keep modest)

# ------------------ OUTPUT DIR ------------------
# allow driver to override via env (for resamples)
OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    r"/media/rilgpu/RIL_Lab1/Raju_Folder/dd/gsfiltersharply/review2resluts"
)
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "sharp")  # keep your "sharp" folder name
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
print(f"✅ SHAP outputs will be saved to: {SHAP_OUTPUT_DIR}")

# ---------- 1) Choose the best (or first) fold for SHAP ----------
per_fold_df = pd.read_csv(per_fold_csv) if os.path.exists(per_fold_csv) else pd.DataFrame()
if not per_fold_df.empty:
    _best = (globals().get("best_result", {}) or {})
    mask = (
        (per_fold_df["database"] == args.database) &
        (per_fold_df["stream_mode"] == stream_mode) &
        (per_fold_df["num_omic"].astype(int) == int(args.num_omic)) &
        (per_fold_df["filter_type"].astype(str).str.strip() == str(_best.get("filter_type", "low")).strip()) &
        (per_fold_df["k"].astype(int) == int(_best.get("k", 14))) &
        (per_fold_df["num_genes"].astype(int) == int(args.num_gene))
    )
    best_fold = int(per_fold_df[mask].sort_values("accuracy", ascending=False)["fold"].iloc[0]) if not per_fold_df[mask].empty else 1
else:
    best_fold = 1
print(f"ℹ️ Using fold {best_fold:02d} for SHAP preparation.")

# ---------- 2) Load gene list + fold indices ----------
gene_csv     = os.path.join(gene_dir_root, f"fold_{best_fold:02d}_genes.csv")
train_idx_np = os.path.join(fold_meta_dir,   f"fold_{best_fold:02d}_train_idx.npy")
if not os.path.exists(gene_csv):     raise FileNotFoundError(f"Missing gene list: {gene_csv}")
if not os.path.exists(train_idx_np): raise FileNotFoundError(f"Missing train indices: {train_idx_np}")

gene_list   = pd.read_csv(gene_csv)["gene"].astype(str).tolist()
train_index = np.load(train_idx_np).astype(int)

# ---------- 3) Build fold data (single vs multi-omics) ----------
if int(args.num_omic) == 1:
    A_sel, X_all, y_all, gene_names = utilsdata.build_fold_data_singleomics(
        expr_all_df, adjacency_matrix_file, gene_list, None,
        singleton=args.singleton, non_null_index_path=non_null_index_path
    )
else:
    A_sel, X_all, y_all, gene_names = utilsdata.build_fold_data_multiomics(
        expr_all_df, cnv_all_df, adjacency_matrix_file, gene_list, None,
        singleton=args.singleton, non_null_index_path=non_null_index_path
    )

# Ensure (N, G, C)
if X_all.ndim == 2:
    X_all = X_all[:, :, None]
assert X_all.shape[2] == int(args.num_omic), f"Channel mismatch: C={X_all.shape[2]} vs {args.num_omic}"

# ---------- 3b) Remap train_index to current X_all row space (supports .npy/.csv) ----------
N = int(X_all.shape[0])
tr_idx = np.asarray(train_index, dtype=int)

def _safe_unique_int(a):
    a = np.asarray(a, dtype=int)
    return np.unique(a[(a >= 0) & (a < N)])

need_remap = tr_idx.max(initial=-1) >= N
if need_remap or (non_null_index_path and os.path.exists(non_null_index_path)):
    try:
        if str(non_null_index_path).lower().endswith(".npy"):
            kept_orig_ids = np.load(non_null_index_path, allow_pickle=True).astype(int)
        elif str(non_null_index_path).lower().endswith(".csv"):
            df_ids = pd.read_csv(non_null_index_path)
            cand_cols = [c for c in df_ids.columns if pd.api.types.is_integer_dtype(df_ids[c])]
            if "index" in df_ids.columns:       col = "index"
            elif "sample_id" in df_ids.columns: col = "sample_id"
            elif cand_cols:                      col = cand_cols[0]
            else:                                col = df_ids.columns[0]
            kept_orig_ids = df_ids[col].astype(int).to_numpy()
        else:
            raise ValueError(f"Unsupported non_null_index_path format: {non_null_index_path}")
        pos = {orig: i for i, orig in enumerate(kept_orig_ids)}
        mapped, missing = [], 0
        for t in tr_idx:
            mi = pos.get(int(t), None)
            if mi is None: missing += 1
            else:          mapped.append(mi)
        tr_idx = _safe_unique_int(mapped)
        if missing > 0:
            print(f"🟨 Remap: {missing} train indices not present in current fold; dropped.")
        print(f"✅ Remapped train_index → {len(tr_idx)} samples (X_all rows: {N})")
    except Exception as e:
        print(f"⚠️ Remap via non_null_index_path failed: {e}. Falling back to clipping.")
        tr_idx = _safe_unique_int(tr_idx)
else:
    tr_idx = _safe_unique_int(tr_idx)

if tr_idx.size == 0:
    raise RuntimeError("After remapping/clipping, train_index is empty. Check fold artifacts alignment.")
train_index = tr_idx

# ---------- 4) Leak-free normalization (TRAIN only) ----------
X_all_norm, _norm_stats = utilsdata.normalize_by_train(X_all, train_index=train_index, method=NORM_METHOD)
train_data_all = X_all_norm[train_index].astype(np.float32, copy=False)
labels         = y_all[train_index].astype(np.int64,  copy=False)

# ---------- 5) Laplacian (normalized + rescaled to [0,1]) ----------
A_max = float(A_sel.data.max()) if hasattr(A_sel, "data") and A_sel.data.size > 0 else 0.0
A_norm = A_sel if A_max == 0.0 else (A_sel / A_max)
L_sp   = graph_laplacian(A_norm, normalized=True)
L_sp   = rescale_to_unit_interval(L_sp)     # [0,1]
L      = [utilsdata.sparse_mx_to_torch_sparse_tensor(L_sp).to(device)]

# ---------- 6) Instantiate predictor; load weights if exist; else quick warmup ----------
_best = (globals().get("best_result", {}) or {})
best_filter  = str(_best.get("filter_type", "low")).strip()
best_k       = int(_best.get("k", 14))
best_dropout = float(_best.get("dropout", 0.3))
theta_best   = compute_theta(best_k, best_filter)

F_0_eff    = int(F_0)            # channels per node
D_g_eff    = int(len(gene_list)) # selected nodes
net_params = [F_0_eff, D_g_eff, CL1_F, best_k, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
best_net   = Graph_GCN(net_params, stream_mode=stream_mode).to(device)
best_net.eval()

loaded_ckpt = False
weights_dir = os.path.join(OUTPUT_ROOT, "weights")
if os.path.isdir(weights_dir):
    pattern = f"*db-{args.database}*stream-{stream_mode}*G-{int(args.num_gene)}*fold-{best_fold:02d}*.pt"
    for ck in glob.glob(os.path.join(weights_dir, pattern)):
        try:
            best_net.load_state_dict(torch.load(ck, map_location=device), strict=False)
            loaded_ckpt = True
            print(f"✅ Loaded weights: {ck}")
            break
        except Exception as e:
            print(f"⚠️ Could not load {ck}: {e}")

if not loaded_ckpt:
    print("🟨 No matching weights found; quick 5-epoch fit on this fold’s train split (for SHAP only).")
    Xtr = torch.tensor(train_data_all, dtype=torch.float32)   # CPU
    ytr = torch.tensor(labels,        dtype=torch.long)       # CPU
    dl  = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=min(128, max(16, len(Xtr)//8)),
        shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    optimizer = optim.AdamW(best_net.parameters(), lr=float(args.lr), weight_decay=3e-5)
    class_counts  = torch.bincount(ytr).clamp_min(1)
    inv           = 1.0 / class_counts.float()
    class_weights = (inv / inv.sum() * out_dim).to(device)
    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_net.train()
    for _ in range(5):
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = best_net(xb, best_dropout, L, theta_best)[2]
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    best_net.eval()

# expose dims for predict wrapper
D_g = int(D_g_eff)
F_0 = int(F_0_eff)
print(f"ℹ️ SHAP base data: train_data_all={train_data_all.shape}, labels={labels.shape}, "
      f"D_g={D_g}, F_0={F_0}, K={best_k}, filter={best_filter}, dropout={best_dropout}")

# ---------- 7) Prep SHAP inputs ----------
train_data_np   = np.asarray(train_data_all, dtype=np.float32)
train_labels_np = np.asarray(labels,        dtype=np.int64)

# Flatten (B, V, C) -> (B, V*C) for KernelExplainer stability
if train_data_np.ndim == 3:
    train_data_flat = train_data_np.reshape(train_data_np.shape[0], -1)
else:
    train_data_flat = train_data_np.copy()

# PCA + MiniBatchKMeans (capped to dataset size/rank)
print("🔍 Running PCA + MiniBatchKMeans for SHAP sample selection...")
n_samples  = train_data_flat.shape[0]
n_features = train_data_flat.shape[1]
n_clusters = min(NUM_SHAP_SAMPLES, max(1, n_samples))
pca_comps  = min(50, max(2, min(n_features - 1, n_samples - 1)))
pca        = PCA(n_components=pca_comps, random_state=42)
X_pca      = pca.fit_transform(train_data_flat)
kmeans     = KMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=10)
km_labels  = kmeans.fit_predict(X_pca)

initial_indices = []
for i in range(n_clusters):
    idxs = np.where(km_labels == i)[0]
    if idxs.size > 0:
        initial_indices.append(int(idxs[0]))
initial_indices = np.array(initial_indices, dtype=int)
print(f"✅ PCA+MiniBatchKMeans selected {len(initial_indices)} diverse centroids.")

# === STRATIFIED MIN-QUOTA (≥20/class where available), then fill to NUM_SHAP_SAMPLES ===
required_per_class = 20
rng = np.random.default_rng(42)

final_indices = set()
classes = np.unique(train_labels_np)

# 1) take up to required_per_class from each class (random within class)
under_quota = []  # classes that can't reach 20 in the TRAIN split
for c in classes:
    idxs = np.where(train_labels_np == c)[0]
    need = min(required_per_class, len(idxs))
    if len(idxs) < required_per_class:
        under_quota.append((int(c), int(len(idxs))))
    if need > 0:
        if len(idxs) <= need:
            sel = idxs
        else:
            sel = rng.choice(idxs, size=need, replace=False)
        final_indices.update(int(i) for i in sel)

# (optional) note the classes inherently < 20 in TRAIN
if len(under_quota) > 0:
    print("🟨 Classes with <20 available in TRAIN split (data scarcity):")
    for c, cnt in under_quota:
        print(f"   Class {c:2d}: only {cnt} available")

# 2) fill the remainder with diverse KMeans-picked indices
for idx in initial_indices:
    if len(final_indices) >= min(NUM_SHAP_SAMPLES, n_samples):
        break
    final_indices.add(int(idx))

# finalize
final_indices = np.array(sorted(final_indices), dtype=int)
final_indices = final_indices[:min(NUM_SHAP_SAMPLES, n_samples)]  # cap by data size too

# save + report
np.save(os.path.join(SHAP_OUTPUT_DIR, "shap_indices.npy"), final_indices)
print(f"✅ Final SHAP sample count: {len(final_indices)} "
      f"(requested {NUM_SHAP_SAMPLES}; classes with ≥1 sample: {len(set(train_labels_np[final_indices]))})")

print("\n📊 Per-class SHAP sample counts:")
for c in np.unique(train_labels_np):
    cnt = int(np.sum(train_labels_np[final_indices] == c))
    print(f"Class {c:2d}: {cnt:2d} samples")


# -------------------- 8) SHAP with GPU-batched predictor --------------------
# (A) Build small background for masker
BACKGROUND_K = max(1, int(BACKGROUND_K))
rng = np.random.default_rng(42)
bg_indices    = rng.choice(n_samples, min(BACKGROUND_K, n_samples), replace=False)
background_np = train_data_flat[bg_indices]

# (B) Keep model on GPU; SHAP calls into this predict function.
best_net_gpu = copy.deepcopy(best_net).to(device).eval()
L_gpu        = [L[0].to(device)]
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# (C) GPU predict wrapper with micro-batching
def gcn_predict_gpu(X_numpy, batch=64):
    with torch.no_grad():
        X = torch.tensor(X_numpy, dtype=torch.float32, device=device)
        if X.ndim == 2:
            X = X.view(-1, D_g, F_0)
        outs = []
        for i in range(0, X.shape[0], batch):
            xb = X[i:i+batch]
            logits = best_net_gpu(xb, best_dropout, L_gpu, theta_best)[2]
            probs  = torch.softmax(logits, dim=1)
            outs.append(probs.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

# (D) Build the explainer using a feature-independent masker (saves memory)
explainer = shap.KernelExplainer(
    model=gcn_predict_gpu,
    data=background_np,          # pass the NumPy background directly
    l1_reg="num_features(10)"
)

# clean stale files
for f in glob.glob(os.path.join(SHAP_OUTPUT_DIR, "shap_*.npy")):
    try: os.remove(f)
    except Exception: pass
open(os.path.join(SHAP_OUTPUT_DIR, "shap_labels.txt"), "w").close()

print("🔁 Computing SHAP values for selected samples (GPU predictions, Kernel SHAP)...")
from tqdm import tqdm
for sample_idx in tqdm(final_indices, desc="🔬 SHAP progress"):
    shap_path = os.path.join(SHAP_OUTPUT_DIR, f"shap_{sample_idx:04d}.npy")
    if os.path.exists(shap_path):
        continue

    sample = train_data_flat[sample_idx:sample_idx+1]
    try:
        shap_values = explainer.shap_values(sample, nsamples=int(SAFE_NSAMPLES))

        # normalize to a 1D vector
        if isinstance(shap_values, list):
            # multi-class: list of arrays; average |shap| across classes
            shap_sample = np.abs(np.mean([s[0] for s in shap_values], axis=0))
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3 and shap_values.shape[0] == 1:
            shap_sample = np.abs(shap_values[0].mean(axis=1))
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 2:
            shap_sample = np.abs(shap_values.mean(axis=1))
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 1:
            shap_sample = np.abs(shap_values)
        else:
            raise ValueError(f"Unexpected SHAP format: {type(shap_values)}")

        np.save(shap_path, shap_sample.astype(np.float32))
        with open(os.path.join(SHAP_OUTPUT_DIR, "shap_labels.txt"), "a") as f:
            f.write(f"{sample_idx},{int(train_labels_np[sample_idx])}\n")
    except Exception as e:
        print(f"❌ SHAP failed for sample {sample_idx}: {e}")

# ---------- 9) Load SHAP vectors ----------
shap_files  = sorted(glob.glob(os.path.join(SHAP_OUTPUT_DIR, "shap_*.npy")))
shap_matrix = []
for f in shap_files:
    try:
        val = np.load(f)
        if val.ndim == 2 and val.shape[0] == 1: val = val[0]
        if val.ndim != 1: continue
        shap_matrix.append(val)
    except Exception:
        pass
if len(shap_matrix) == 0:
    raise RuntimeError("No valid SHAP vectors saved. Check the explainer and inputs.")
shap_matrix = np.vstack(shap_matrix).astype(np.float32)
np.save(os.path.join(SHAP_OUTPUT_DIR, "shap_matrix.npy"), shap_matrix)
print(f"✅ Loaded SHAP matrix: {shap_matrix.shape}")

# ---------- 10) Aggregate to genes (for multi-omics) ----------
if int(args.num_omic) == 2:
    base_gene_names = sorted(set(n.split('_')[0] for n in gene_names))
    gene_to_indices = defaultdict(list)
    for i, name in enumerate(gene_names):
        gene_to_indices[name.split('_')[0]].append(i)

    shap_matrix_agg = np.zeros((shap_matrix.shape[0], len(base_gene_names)), dtype=np.float32)
    final_gene_names = []
    for g in base_gene_names:
        idxs = gene_to_indices[g]
        if len(idxs) == 1:
            shap_matrix_agg[:, len(final_gene_names)] = shap_matrix[:, idxs[0]]
            final_gene_names.append(g)
        elif len(idxs) == 2:
            shap_matrix_agg[:, len(final_gene_names)] = shap_matrix[:, idxs].mean(axis=1)
            final_gene_names.append(g)
        # silently ignore >2
else:
    shap_matrix_agg = shap_matrix
    final_gene_names = list(gene_names)

# overwrite with aggregated matrix
np.save(os.path.join(SHAP_OUTPUT_DIR, "shap_matrix.npy"), shap_matrix_agg)

# ---------- 10b) Save manifest for stability driver ----------
# labels from the text file (truncate to matrix rows)
label_file_path = os.path.join(SHAP_OUTPUT_DIR, "shap_labels.txt")
shap_labels = []
with open(label_file_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            shap_labels.append(int(parts[1]))
shap_labels = np.asarray(shap_labels[:shap_matrix_agg.shape[0]], dtype=np.int32)
np.save(os.path.join(SHAP_OUTPUT_DIR, "shap_labels.npy"), shap_labels)
np.save(os.path.join(SHAP_OUTPUT_DIR, "gene_names.npy"),  np.asarray(final_gene_names, dtype=object))

# ---------- 11) Save Top-5 genes per class to Excel (you can bump to 10/30 for SI) ----------
NUM_CLASSES = int(np.max(shap_labels)) + 1 if "NUM_CLASSES" not in globals() else NUM_CLASSES
TOPK = 5 if "NUM_TOP_GENES_PER_CLASS" not in globals() else int(NUM_TOP_GENES_PER_CLASS)

print("📊 Extracting top genes per class...")
shap_df = pd.DataFrame(shap_matrix_agg, columns=final_gene_names)
shap_df["label"] = shap_labels

excel_path = os.path.join(SHAP_OUTPUT_DIR, "top_genes_per_class.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    for cls in range(NUM_CLASSES):
        df_cls = shap_df[shap_df["label"] == cls].drop(columns="label")
        if df_cls.shape[0] == 0:
            print(f"⚠️ Skipping Class {cls} — no valid SHAP samples.")
            continue
        df_cls = df_cls.dropna(axis=1, how='all')
        mean_shap = df_cls.mean(axis=0).fillna(0.0)
        top_genes = mean_shap.sort_values(ascending=False).head(TOPK)
        pd.DataFrame({"Top Genes": top_genes.index, "Mean SHAP": top_genes.values}).to_excel(
            writer, sheet_name=f"Class_{cls}", index=False
        )
print(f"✅ Top genes per class saved to {excel_path}")
# ===================== END DROP-IN =====================

# ===================== END DROP-IN =====================

# =========================================================================================

import pandas as pd
import numpy as np
from collections import Counter

# Load all class sheets
all_sheets = pd.read_excel(excel_path, sheet_name=None)

# Build class frequency map
gene_class_map = {}
for sheet_name, df in all_sheets.items():
    class_label = sheet_name.replace("Class_", "")
    if "Top Genes" in df.columns:
        for gene in df["Top Genes"].dropna():
            gene_class_map.setdefault(gene, set()).add(class_label)

# Create frequency DataFrame
genes = list(gene_class_map.keys())
class_freq = [len(gene_class_map[g]) for g in genes]
freq_df = pd.DataFrame({"Gene": genes, "Class Frequency": class_freq})

# Mean SHAP per gene (across all classes) from your shap_df
mean_shap_all = shap_df.drop(columns="label").mean(axis=0).reset_index()
mean_shap_all.columns = ["Gene", "Mean SHAP"]

# Merge SHAP and frequency data
merged_df = pd.merge(freq_df, mean_shap_all, on="Gene", how="inner")

# Normalize both scores
merged_df["Norm SHAP"] = (merged_df["Mean SHAP"] - merged_df["Mean SHAP"].min()) / (merged_df["Mean SHAP"].max() - merged_df["Mean SHAP"].min())
merged_df["Norm Freq"] = (merged_df["Class Frequency"] - merged_df["Class Frequency"].min()) / (merged_df["Class Frequency"].max() - merged_df["Class Frequency"].min())

# Hybrid Score
merged_df["Hybrid Score"] = 0.5 * merged_df["Norm SHAP"] + 0.5 * merged_df["Norm Freq"]

# Sort and select top 20 by each criterion
top_20_by_shap = merged_df.sort_values("Mean SHAP", ascending=False).head(20)
top_20_by_freq = merged_df.sort_values("Class Frequency", ascending=False).head(20)
top_20_by_hybrid = merged_df.sort_values("Hybrid Score", ascending=False).head(20)

# Save all rankings to a new Excel
hybrid_excel_path = os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx")
with pd.ExcelWriter(hybrid_excel_path) as writer:
    top_20_by_shap.to_excel(writer, sheet_name="Top_20_by_SHAP", index=False)
    top_20_by_freq.to_excel(writer, sheet_name="Top_20_by_Frequency", index=False)
    top_20_by_hybrid.to_excel(writer, sheet_name="Top_20_by_Hybrid", index=False)

print(f"✅ Gene ranking comparison saved to {hybrid_excel_path}")

# ---------- NEW: build X_sample_flattened to match `final_gene_names` ----------
# We want the exact input rows that SHAP explained:
X_flat_selected = train_data_flat[final_indices]   # shape: (num_samples, V*C)

# If you aggregated SHAP to per-gene (multi-omics), aggregate inputs the same way
# so the number of columns matches `final_gene_names`.
if int(args.num_omic) == 2 and 'base_gene_names' in locals() and 'gene_to_indices' in locals():
    X_agg = np.zeros((X_flat_selected.shape[0], len(base_gene_names)), dtype=np.float32)
    for j, g in enumerate(base_gene_names):
        idxs = gene_to_indices[g]
        if len(idxs) == 1:
            X_agg[:, j] = X_flat_selected[:, idxs[0]]
        else:
            # average expr/cnv (or any modalities) for this gene
            X_agg[:, j] = X_flat_selected[:, idxs].mean(axis=1)
    X_sample_flattened = X_agg
else:
    # single-omics (or you didn’t aggregate) — use the flat features as-is
    X_sample_flattened = X_flat_selected

# Safety check: columns must match the names you plan to use
if X_sample_flattened.shape[1] != len(final_gene_names):
    raise ValueError(
        f"Column mismatch: X_sample_flattened has {X_sample_flattened.shape[1]} "
        f"but final_gene_names has {len(final_gene_names)}. "
        "If you aggregated SHAP to per-gene, aggregate inputs the same way (see code above)."
    )
# ---------- END NEW PATCH ----------

import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

def plot_shap_summary_for_rankings(excel_path, shap_matrix, X_sample, OutputDir):
    rankings = ["Top_20_by_SHAP", "Top_20_by_Frequency", "Top_20_by_Hybrid"]
    
    print("📊 Generating SHAP summary plots for all top 20 rankings...")

    for sheet in rankings:
        print(f"📂 Processing: {sheet}")
        df_top = pd.read_excel(excel_path, sheet_name=sheet)
        top_gene_names = df_top["Gene"].tolist()

        # Get feature indices
        full_gene_names = list(shap_matrix.columns)
        top_gene_indices = [full_gene_names.index(g) for g in top_gene_names if g in full_gene_names]

        # Subset data
        top_shap_values = shap_matrix.iloc[:, top_gene_indices].values
        top_X_sample = X_sample.iloc[:, top_gene_indices].values

        # Validate shapes
        if top_shap_values.shape[0] != top_X_sample.shape[0]:
            raise ValueError(f"❌ Shape mismatch for {sheet}: {top_shap_values.shape} vs {top_X_sample.shape}")

        # Create output folder
        plot_dir = os.path.join(OutputDir, sheet)
        os.makedirs(plot_dir, exist_ok=True)

        # Plot
        shap.summary_plot(
            top_shap_values,
            top_X_sample,
            feature_names=top_gene_names,
            show=False
        )
        plt.title(f"SHAP Summary Plot ({sheet})", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, "shap_summary.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {plot_path}")

plot_shap_summary_for_rankings(
    excel_path=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    shap_matrix=shap_df.drop(columns="label"),  # same as final aggregated SHAP
    X_sample=pd.DataFrame(X_sample_flattened, columns=final_gene_names),  # input features used in SHAP
    OutputDir=SHAP_OUTPUT_DIR
)


def plot_shap_dependence_for_rankings(excel_path, shap_matrix, X_sample, OutputDir):
    rankings = ["Top_20_by_SHAP", "Top_20_by_Frequency", "Top_20_by_Hybrid"]
    
    print("📊 Generating SHAP dependence plots for top 5 genes in each ranking...")

    for sheet in rankings:
        print(f"📂 Processing: {sheet}")
        df_top = pd.read_excel(excel_path, sheet_name=sheet)
        top_gene_names = df_top["Gene"].tolist()[:5]  # only top 5

        # Get indices of top genes in the original feature list
        full_gene_names = list(shap_matrix.columns)
        top_gene_indices = [full_gene_names.index(g) for g in top_gene_names if g in full_gene_names]

        # Subset SHAP and X_sample
        top_shap_values = shap_matrix.iloc[:, top_gene_indices].values
        top_X_sample = X_sample.iloc[:, top_gene_indices].values

        # Folder
        plot_dir = os.path.join(OutputDir, sheet)
        os.makedirs(plot_dir, exist_ok=True)

        # Plot for each of the top 5 genes
        for i, gene in enumerate(top_gene_names):
            shap.dependence_plot(
                i,
                top_shap_values,
                top_X_sample,
                feature_names=top_gene_names,
                show=False
            )
            plt.title(f"SHAP Dependence Plot for {gene}", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"shap_dependence_{gene}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved dependence plot for {gene} in {sheet}")

plot_shap_dependence_for_rankings(
    excel_path=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    shap_matrix=shap_df.drop(columns="label"),
    X_sample=pd.DataFrame(X_sample_flattened, columns=final_gene_names),
    OutputDir=SHAP_OUTPUT_DIR
)

import seaborn as sns
import matplotlib.pyplot as plt

def plot_shap_heatmaps_for_rankings(excel_path, shap_matrix, OutputDir, X_sample=None, num_samples=10):
    rankings = ["Top_20_by_SHAP", "Top_20_by_Frequency", "Top_20_by_Hybrid"]
    
    print("📊 Generating SHAP heatmaps for top 20 genes...")

    for sheet in rankings:
        print(f"📂 Processing: {sheet}")
        df_top = pd.read_excel(excel_path, sheet_name=sheet)
        top_gene_names = df_top["Gene"].tolist()

        # Get feature indices
        full_gene_names = list(shap_matrix.columns)
        top_gene_indices = [full_gene_names.index(g) for g in top_gene_names if g in full_gene_names]

        # Extract SHAP values
        top_shap_values = shap_matrix.iloc[:, top_gene_indices].values

        # Limit samples to first N
        num_samples_to_show = min(num_samples, top_shap_values.shape[0])
        heatmap_data = top_shap_values[:num_samples_to_show, :]

        # Create output folder
        plot_dir = os.path.join(OutputDir, sheet)
        os.makedirs(plot_dir, exist_ok=True)

        # Plot heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            heatmap_data,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            xticklabels=top_gene_names,
            yticklabels=[f"Sample {i}" for i in range(num_samples_to_show)]
        )
        plt.title(f"SHAP Heatmap ({sheet})", fontsize=16, fontweight="bold")
        plt.xlabel("Gene", fontsize=14)
        plt.ylabel("Sample", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "shap_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Heatmap saved for: {sheet}")
plot_shap_heatmaps_for_rankings(
    excel_path=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    shap_matrix=shap_df.drop(columns="label"),
    OutputDir=SHAP_OUTPUT_DIR,
    num_samples=10  # Can change to 30 or 50 if needed
)
def plot_shap_force_from_excel(
    shap_matrix_path,
    gene_ranking_excel,
    X_sample,
    gene_names,
    OutputDir,
    sheet="Top_20_by_Hybrid",
    sample_index=0
):
    """Plot SHAP Force Plot from saved matrix and Excel top gene list."""
    print(f"📌 Generating SHAP Force Plot using top genes from: {sheet}")

    # Load SHAP matrix
    shap_matrix = np.load(shap_matrix_path)
    print(f"✅ SHAP matrix shape: {shap_matrix.shape}")

    # Load top genes from Excel
    df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
    top_gene_names = df_top["Gene"].tolist()

    # Map top gene names to indices
    top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

    # Extract SHAP values and input for selected genes
    top_shap_values = shap_matrix[sample_index, top_gene_indices]
    top_X_sample = X_sample[sample_index, top_gene_indices]

    # Dummy base value (you can replace with model output mean if known)
    base_value = np.mean(shap_matrix)  # or explainer.expected_value if known

    # Plot settings
    dynamic_width = max(30, len(top_gene_names) * 2)
    dynamic_fontsize = max(8, min(14, 20 - (len(top_gene_names) // 10)))

    # Init and Plot
    shap.initjs()
    plt.figure(figsize=(dynamic_width, 8))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25)

    shap.force_plot(
        base_value,
        top_shap_values,
        top_gene_names,
        matplotlib=True
    )

    plt.text(
        0.5, -0.2,
        f"SHAP Force Plot ({sheet} - Sample {sample_index})",
        ha='center',
        va='top',
        transform=plt.gca().transAxes,
        fontsize=dynamic_fontsize,
        fontweight='bold'
    )

    # Save and show
    plot_dir = os.path.join(OutputDir, sheet)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"shap_force_sample_{sample_index}.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Force plot saved for {sheet} → sample {sample_index}")
plot_shap_force_from_excel(
    shap_matrix_path=os.path.join(SHAP_OUTPUT_DIR, "shap_matrix.npy"),
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    X_sample=X_sample_flattened,
    gene_names=final_gene_names,
    OutputDir=SHAP_OUTPUT_DIR,
    sheet="Top_20_by_Hybrid",  # You can also do "Top_20_by_SHAP" or "Top_20_by_Frequency"
    sample_index=0  # Sample to visualize
)
def plot_shap_importance_heatmap_from_excel(
    shap_matrix_path,
    gene_ranking_excel,
    gene_names,
    OutputDir,
    sheet="Top_20_by_Hybrid",
    num_samples=50  # Number of samples to show
):
    """
    Plots a SHAP importance heatmap from top genes listed in Excel.
    Supports both single-omics and multi-omics SHAP matrix formats.
    """
    print(f"📌 Generating SHAP Importance Heatmap from sheet: {sheet}")

    # Load SHAP matrix
    shap_matrix = np.load(shap_matrix_path)
    print(f"✅ SHAP matrix loaded with shape: {shap_matrix.shape}")

    # Load top gene names
    df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
    top_gene_names = df_top["Gene"].tolist()

    # Map top genes to indices
    top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

    # Multi-omics: average across last axis
    if shap_matrix.ndim == 3:
        print("📌 Multi-omics SHAP detected: averaging last axis.")
        top_shap_values = shap_matrix[:, top_gene_indices, :].mean(axis=-1)
    else:
        print("📌 Single-omics SHAP detected.")
        top_shap_values = shap_matrix[:, top_gene_indices]

    # Limit number of samples
    top_shap_values = top_shap_values[:num_samples, :]

    # Create heatmap DataFrame
    df_shap = pd.DataFrame(top_shap_values.T, index=[top_gene_names[i] for i in range(len(top_gene_indices))])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        df_shap, cmap="coolwarm", center=0, annot=False, linewidths=0.5,
        cbar_kws={"shrink": 0.5}, xticklabels=[f"S{i}" for i in range(df_shap.shape[1])], ax=ax
    )
    ax.set_title(f"SHAP Importance Heatmap ({sheet})", fontsize=16, fontweight="bold")
    ax.set_ylabel("Top Genes", fontsize=12)
    ax.set_xlabel("Samples", fontsize=12)

    # Save
    plot_dir = os.path.join(OutputDir, sheet)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "shap_importance_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ SHAP Importance Heatmap saved to: {os.path.join(plot_dir, 'shap_importance_heatmap.png')}")

plot_shap_importance_heatmap_from_excel(
    shap_matrix_path=os.path.join(SHAP_OUTPUT_DIR, "shap_matrix.npy"),
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    OutputDir=SHAP_OUTPUT_DIR,
    sheet="Top_20_by_Hybrid"
)

def plot_correlation_heatmap_from_excel(
    X_sample,
    gene_ranking_excel,
    gene_names,
    OutputDir,
    sheet="Top_20_by_Hybrid"
):
    """
    Plots the correlation heatmap of the selected top genes from Excel.
    Works for both single-omics and multi-omics (2D or 3D input).
    """
    print(f"📌 Generating Correlation Heatmap from: {sheet}")

    # Load top genes
    df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
    top_gene_names = df_top["Gene"].tolist()

    # Map genes to indices
    top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

    # Extract features
    top_gene_features = X_sample[:, top_gene_indices]

    # Handle multi-omics: average across last dimension
    if top_gene_features.ndim == 3:
        print(f"📌 Multi-omics data detected: Shape {top_gene_features.shape}, averaging omics dimension.")
        top_gene_features = top_gene_features.mean(axis=-1)

    # Compute correlation matrix
    corr_df = pd.DataFrame(top_gene_features, columns=top_gene_names).corr()

    # Plot size based on gene count
    num_genes = len(top_gene_names)
    fig_width = min(2 + 0.6 * num_genes, 20)
    fig_height = min(2 + 0.6 * num_genes, 15)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        corr_df,
        annot=num_genes <= 20,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8} if num_genes <= 20 else None
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(f"Correlation Heatmap of Top {num_genes} Genes ({sheet})", fontsize=14, fontweight="bold")

    # Save
    plot_dir = os.path.join(OutputDir, sheet)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "top_genes_correlation_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Correlation heatmap saved to: {plot_path}")

plot_correlation_heatmap_from_excel(
    X_sample=X_sample_flattened,
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    OutputDir=SHAP_OUTPUT_DIR,
    sheet="Top_20_by_Hybrid"  # Can also try "Top_20_by_SHAP" or "Top_20_by_Frequency"
)
def plot_gene_network_adjacency_from_excel(
    gene_ranking_excel,
    gene_names,
    gene_importance,
    adj_final,
    OutputDir,
    sheet="Top_20_by_Hybrid"
):
    """
    Plots an adjacency-based gene interaction network using top genes from Excel.
    Only SHAP importance colorbar is shown (no edge weight colorbar).
    """
    print(f"📌 Generating Adjacency-Based Gene Network from: {sheet}")

    # Load top genes
    df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
    top_gene_names = df_top["Gene"].tolist()

    # Map genes to indices
    selected_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

    # Submatrix from adjacency
    adj_dense = adj_final.toarray() if hasattr(adj_final, "toarray") else adj_final
    adj_submatrix = adj_dense[np.ix_(selected_indices, selected_indices)]

    # Threshold edges (top 20%)
    edge_values = adj_submatrix[np.triu_indices_from(adj_submatrix, k=1)]
    edge_threshold = np.percentile(edge_values, 80)

    # Construct graph
    G_adj = nx.Graph()
    shap_values_selected = {gene: gene_importance[gene_names.index(gene)] for gene in top_gene_names}
    min_shap, max_shap = min(shap_values_selected.values()), max(shap_values_selected.values())

    for gene in top_gene_names:
        G_adj.add_node(gene, shap_value=shap_values_selected[gene])

    edge_count = 0
    for i in range(len(top_gene_names)):
        for j in range(i + 1, len(top_gene_names)):
            weight = adj_submatrix[i, j]
            if weight >= edge_threshold:
                G_adj.add_edge(top_gene_names[i], top_gene_names[j], weight=weight)
                edge_count += 1

    if edge_count == 0:
        print("⚠️ No edges above threshold — consider lowering the percentile.")
        return

    # SHAP color map
    cmap_shap = cm.coolwarm
    norm_shap = mcolors.Normalize(vmin=min_shap, vmax=max_shap)
    node_colors = [cmap_shap(norm_shap(shap_values_selected[node])) for node in G_adj.nodes()]

    # Scale node sizes by degree
    degrees = dict(G_adj.degree())
    node_sizes = np.interp(list(degrees.values()), (min(degrees.values()), max(degrees.values())), (500, 3000))

    # Edge width scaling
    edge_weights = [G_adj[u][v]["weight"] for u, v in G_adj.edges()]
    edge_widths = np.interp(edge_weights, (min(edge_weights), max(edge_weights)), (0.5, 4))

    # Layout and plot
    pos = nx.spring_layout(G_adj, seed=42, k=2.5)
    fig, ax = plt.subplots(figsize=(16, 14))

    nx.draw_networkx_nodes(G_adj, pos, node_size=node_sizes, node_color=node_colors, edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G_adj, pos, width=edge_widths, edge_color="gray", alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G_adj, pos, font_size=12, font_color="black", ax=ax)

    plt.title(f"Gene Network ({sheet})", fontsize=16, fontweight="bold")

    # Only SHAP importance colorbar
    sm = cm.ScalarMappable(cmap=cmap_shap, norm=norm_shap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="SHAP Importance")

    # Save
    plot_dir = os.path.join(OutputDir, sheet)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "gene_network_adjacency.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Gene network saved to {os.path.join(plot_dir, 'gene_network_adjacency.png')}")

plot_gene_network_adjacency_from_excel(
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    gene_importance=np.mean(shap_matrix_agg, axis=0),  # Mean SHAP as importance
    adj_final=adj_final,
    OutputDir=SHAP_OUTPUT_DIR,
    sheet="Top_20_by_Hybrid"
)

def plot_gene_network_shap_from_excel(
    X_sample,
    gene_ranking_excel,
    gene_names,
    gene_importance,
    OutputDir,
    sheet="Top_20_by_Hybrid"
):
    """
    Generates a SHAP-based gene correlation network for top genes from Excel.
    - Nodes: colored by SHAP importance (fixed size).
    - Edges: correlation-based.
    - Only SHAP colorbar shown.
    """
    print(f"📌 Generating SHAP-Based Gene Network from: {sheet}")

    # Load top genes
    df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
    top_gene_names = df_top["Gene"].tolist()
    top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

    # Extract features
    top_gene_features = X_sample[:, top_gene_indices]
    if top_gene_features.ndim == 3:  # Multi-omics case
        print("📌 Multi-omics detected — averaging omics dimension.")
        top_gene_features = top_gene_features.mean(axis=-1)

    # SHAP importance
    shap_values_selected = {gene: gene_importance[gene_names.index(gene)] for gene in top_gene_names}
    min_shap, max_shap = min(shap_values_selected.values()), max(shap_values_selected.values())
    if max_shap - min_shap == 0:
        max_shap += 1e-6

    # Build correlation matrix
    correlation_matrix = pd.DataFrame(top_gene_features, columns=top_gene_names).corr()
    edge_values = correlation_matrix.abs().values[np.triu_indices_from(correlation_matrix, k=1)]
    edge_threshold = np.percentile(edge_values, 50)

    G_shap = nx.Graph()
    for gene in top_gene_names:
        G_shap.add_node(gene, shap_value=shap_values_selected[gene])

    edge_count = 0
    for i in range(len(top_gene_names)):
        for j in range(i + 1, len(top_gene_names)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) >= edge_threshold:
                G_shap.add_edge(top_gene_names[i], top_gene_names[j], weight=abs(corr))
                edge_count += 1

    if edge_count == 0:
        print("⚠️ No significant correlations found — lower the edge threshold.")
        return

    # Layout & visuals
    cmap_shap = cm.coolwarm
    norm_shap = mcolors.Normalize(vmin=min_shap, vmax=max_shap)
    node_colors = [cmap_shap(norm_shap(shap_values_selected[node])) for node in G_shap.nodes()]

    fixed_node_size = 2500
    edge_weights = [G_shap[u][v]['weight'] for u, v in G_shap.edges()]
    edge_widths = np.interp(edge_weights, (min(edge_weights), max(edge_weights)), (1, 5)) if edge_weights else []

    pos = nx.spring_layout(G_shap, seed=42, k=4, iterations=100, scale=2)
    for node in G_shap.nodes():
        if G_shap.degree(node) == 0:
            pos[node] = np.random.rand(2) * 2 - 1

    fig, ax = plt.subplots(figsize=(20, 18))
    nx.draw_networkx_nodes(G_shap, pos, node_size=fixed_node_size, node_color=node_colors, edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G_shap, pos, width=edge_widths, edge_color="gray", alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G_shap, pos, font_size=10, font_color="black", font_weight="bold", ax=ax)

    plt.title(f"SHAP-Based Gene Network ({sheet})", fontsize=16, fontweight="bold")
    sm = cm.ScalarMappable(cmap=cmap_shap, norm=norm_shap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="SHAP Importance")

    plot_dir = os.path.join(OutputDir, sheet)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "gene_network_shap.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ SHAP-based gene network saved to: {os.path.join(plot_dir, 'gene_network_shap.png')}")

plot_gene_network_shap_from_excel(
    X_sample=X_sample_flattened,
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    gene_importance=np.mean(shap_matrix_agg, axis=0),
    OutputDir=SHAP_OUTPUT_DIR,
    sheet="Top_20_by_Hybrid"
)

def plot_force_directed_gene_network_from_excel(
    gene_ranking_excel,
    gene_names,
    gene_importance,
    adj_final,
    OutputDir,
    sheet="Top_20_by_Hybrid"
):
    """
    Generates a force-directed gene network using adjacency matrix and SHAP importance.
    - Fixed node size
    - Node color = SHAP importance
    - Edge width = connection strength
    - ONLY SHAP colorbar shown
    """
    print(f"📌 Generating Force-Directed Gene Network from: {sheet}")

    # Load top gene names
    df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
    top_gene_names = df_top["Gene"].tolist()
    selected_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

    # Convert sparse to dense if needed
    adj_dense = adj_final.toarray() if hasattr(adj_final, "toarray") else np.array(adj_final)
    adj_submatrix = adj_dense[np.ix_(selected_indices, selected_indices)]

    # Edge thresholding
    edge_values = adj_submatrix[np.triu_indices_from(adj_submatrix, k=1)].flatten()
    if edge_values.size == 0:
        print("⚠️ No valid edge values found.")
        return
    edge_threshold = np.nanpercentile(edge_values, 50)

    # Construct graph
    G = nx.Graph()
    shap_values_selected = {gene: gene_importance[gene_names.index(gene)] for gene in top_gene_names}
    min_shap, max_shap = min(shap_values_selected.values()), max(shap_values_selected.values())
    if max_shap - min_shap == 0:
        max_shap += 1e-6

    for gene in top_gene_names:
        G.add_node(gene, shap_value=shap_values_selected[gene])

    edge_count = 0
    for i in range(len(top_gene_names)):
        for j in range(i + 1, len(top_gene_names)):
            weight = adj_submatrix[i, j]
            if weight >= edge_threshold:
                G.add_edge(top_gene_names[i], top_gene_names[j], weight=weight)
                edge_count += 1

    if edge_count == 0:
        print("⚠️ No edges added — consider lowering the threshold.")
        return

    # Visual settings
    fixed_node_size = 2500
    cmap_shap = cm.coolwarm
    norm_shap = mcolors.Normalize(vmin=min_shap, vmax=max_shap)
    node_colors = [cmap_shap(norm_shap(shap_values_selected[n])) for n in G.nodes()]

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = np.interp(edge_weights, (min(edge_weights), max(edge_weights)), (1, 5)) if edge_weights else []

    # Layout and label adjustment
    pos = nx.spring_layout(G, seed=42, k=3, iterations=300)
    pos_adjusted = {node: (x, y + 0.05) for node, (x, y) in pos.items()}

    # Plot
    fig, ax = plt.subplots(figsize=(20, 18))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=fixed_node_size,
                           edgecolors="black", cmap=cmap_shap, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos_adjusted, font_size=12, font_color="black", font_weight="bold", ax=ax)

    plt.title(f"Force-Directed Gene Network ({sheet})", fontsize=16, fontweight="bold")

    # SHAP Colorbar only
    sm = cm.ScalarMappable(cmap=cmap_shap, norm=norm_shap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="SHAP Importance")

    # Save
    plot_dir = os.path.join(OutputDir, sheet)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "force_directed_gene_network.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Force-directed gene network saved to {os.path.join(plot_dir, 'force_directed_gene_network.png')}")
plot_force_directed_gene_network_from_excel(
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    gene_importance=np.mean(shap_matrix_agg, axis=0),
    adj_final=adj_final,
    OutputDir=SHAP_OUTPUT_DIR,
    sheet="Top_20_by_Hybrid"
)
def plot_umap_tsne_from_excel(
    X_sample,
    gene_ranking_excel,
    gene_names,
    OutputDir,
    shap_matrix=None,  # Optional: Can also color by SHAP
    sheet="Top_20_by_Hybrid"
):
    """
    Generates UMAP and t-SNE projections for selected top genes from Excel.
    Colors by SHAP importance if shap_matrix provided.
    """
    print(f"📌 Generating UMAP and t-SNE projections from: {sheet}")

    # Load top genes
    df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
    top_gene_names = df_top["Gene"].tolist()
    top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

    # Subset features
    top_gene_features = X_sample[:, top_gene_indices]

    # If SHAP matrix given → Color by SHAP Importance
    if shap_matrix is not None:
        if shap_matrix.ndim == 3:  # Multi-omics
            print("📌 Multi-omics detected for SHAP matrix — averaging.")
            shap_importance = np.abs(shap_matrix).mean(axis=(1, 2))
        else:
            shap_importance = np.abs(shap_matrix).mean(axis=1)

        shap_importance = shap_importance[:X_sample.shape[0]]  # Trim if necessary
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=shap_importance.min(), vmax=shap_importance.max())
        colors = cmap(norm(shap_importance))
    else:
        # Default color if no SHAP
        colors = "blue"

    # Perform UMAP and t-SNE
    reducer_umap = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    reducer_tsne = TSNE(n_components=2, perplexity=15, random_state=42)

    umap_proj = reducer_umap.fit_transform(top_gene_features)
    tsne_proj = reducer_tsne.fit_transform(top_gene_features)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].scatter(umap_proj[:, 0], umap_proj[:, 1], c=colors, edgecolor="black", alpha=0.85, s=80)
    axes[0].set_title("UMAP Projection (Gene Embeddings)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("UMAP 1", fontsize=12)
    axes[0].set_ylabel("UMAP 2", fontsize=12)

    axes[1].scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors, edgecolor="black", alpha=0.85, s=80)
    axes[1].set_title("t-SNE Projection (Gene Embeddings)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("t-SNE 1", fontsize=12)
    axes[1].set_ylabel("t-SNE 2", fontsize=12)

    # Add colorbar if SHAP was used
    if shap_matrix is not None:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.75, aspect=30, label="SHAP Importance")

    # Save
    plot_dir = os.path.join(OutputDir, sheet)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "umap_tsne_projections.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ UMAP and t-SNE projections saved to {os.path.join(plot_dir, 'umap_tsne_projections.png')}")
plot_umap_tsne_from_excel(
    X_sample=X_sample_flattened,
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    OutputDir=SHAP_OUTPUT_DIR,
    shap_matrix=shap_matrix_agg,  # pass aggregated SHAP matrix
    sheet="Top_20_by_Hybrid"
)
def plot_gene_expression_clustering_from_excel(
    X_sample,
    gene_ranking_excel,
    gene_names,
    OutputDir,
    sheet="Top_20_by_Hybrid"
):
    """
    Generates a hierarchical clustering heatmap of gene expression levels
    using top genes from the specified Excel sheet.
    """
    print(f"📌 Generating Gene Expression Clustering Heatmap from: {sheet}")

    try:
        # ✅ Load top genes from Excel
        df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
        top_gene_names = df_top["Gene"].tolist()
        top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

        # ✅ Subset expression data
        top_gene_features = X_sample[:, top_gene_indices]
        if top_gene_features.ndim == 3:
            print("📌 Multi-omics detected: Averaging across last axis.")
            top_gene_features = top_gene_features.mean(axis=-1)

        if top_gene_features.shape[1] == 0:
            raise ValueError("❌ No genes available for clustering!")

        # ✅ Compute hierarchical clustering
        row_dist = pdist(top_gene_features.T, metric='euclidean')
        row_linkage = sch.linkage(row_dist, method="ward")

        df_expr = pd.DataFrame(top_gene_features, columns=top_gene_names)

        num_genes = len(top_gene_names)
        label_fontsize = max(8, min(14, 18 - (num_genes // 10)))

        g = sns.clustermap(
            df_expr,
            row_linkage=row_linkage, col_cluster=False, row_cluster=True,
            cmap="coolwarm", annot=False,
            figsize=(22, 14), linewidths=0.5, linecolor="white",
            dendrogram_ratio=(0.20, 0.02),
            cbar_kws={"shrink": 0.4, "aspect": 30},
            robust=True
        )

        g.ax_heatmap.set_position([0.35, 0.15, 0.6, 0.7])
        g.fig.suptitle(f"Gene Expression Clustering Heatmap ({sheet})", fontsize=18, fontweight="bold", y=1.05)

        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=label_fontsize, rotation=45, ha="right")
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=label_fontsize)

        g.ax_heatmap.set_xlabel("Samples (Patients)", fontsize=16, fontweight="bold", labelpad=12)
        g.ax_heatmap.set_ylabel("Clustered Genes", fontsize=16, fontweight="bold", labelpad=12)

        g.cax.set_position([0.98, 0.25, 0.02, 0.5])

        # Optional: Separate dendrogram plot
        fig, ax = plt.subplots(figsize=(6, 12))
        sch.dendrogram(row_linkage, labels=top_gene_names, orientation='left', ax=ax)
        ax.set_position([0.05, 0.15, 0.3, 0.7])
        ax.set_xlabel("Distance", fontsize=12, fontweight="bold")
        ax.set_ylabel("Genes", fontsize=12, fontweight="bold")
        plt.title("Hierarchical Clustering Dendrogram", fontsize=14, fontweight="bold")

        # ✅ Save both plots
        plot_dir = os.path.join(OutputDir, sheet)
        os.makedirs(plot_dir, exist_ok=True)
        g.savefig(os.path.join(plot_dir, "gene_expression_clustering.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(plot_dir, "gene_dendrogram_only.png"), dpi=300, bbox_inches="tight")
        plt.show()

        print(f"✅ Expression clustering saved to: {plot_dir}")

    except Exception as e:
        print(f"❌ Error in `plot_gene_expression_clustering_from_excel()`: {e}")
plot_gene_expression_clustering_from_excel(
    X_sample=X_sample_flattened,
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    OutputDir=SHAP_OUTPUT_DIR,
    sheet="Top_20_by_Hybrid"
)
def plot_gene_expression_distribution_from_excel(
    X_sample,
    gene_ranking_excel,
    gene_names,
    OutputDir,
    labels,
    sheet="Top_20_by_Hybrid"
):
    """
    Generates histogram + boxplot + swarm overlays for gene expression distributions.
    Plots top genes from an Excel sheet. Handles single/multi-omics automatically.
    """
    print(f"📌 Generating Gene Expression Distribution from: {sheet}")

    # Load top gene names
    df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
    top_gene_names = df_top["Gene"].tolist()
    top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

    # Subset X_sample
    top_gene_features = X_sample[:, top_gene_indices]
    if top_gene_features.ndim == 3:
        print("📌 Multi-omics detected — averaging last axis.")
        top_gene_features = top_gene_features.mean(axis=-1)

    expected_length = top_gene_features.shape[0]
    valid_genes, valid_features = [], []

    for gene, values in zip(top_gene_names, top_gene_features.T):
        if len(values) == expected_length:
            valid_genes.append(gene)
            valid_features.append(values)
        else:
            print(f"⚠️ Skipping '{gene}' (length mismatch).")

    if not valid_genes:
        raise ValueError("❌ No valid genes found after filtering!")

    print(f"✅ Plotting {len(valid_genes)} genes...")

    valid_features = np.array(valid_features).T  # (samples, genes)

    num_genes = len(valid_genes)
    num_cols = 4
    num_rows = int(np.ceil(num_genes / num_cols))

    # ----------------------- Histogram + KDE -----------------------
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 3), constrained_layout=True)
    axes = axes.flatten()

    for i, gene in enumerate(valid_genes):
        sns.histplot(valid_features[:, i], bins=30, kde=True, color="royalblue", edgecolor="black", ax=axes[i])
        axes[i].set_title(f"Expression of {gene}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Expression Level", fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plot_dir = os.path.join(OutputDir, sheet)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "gene_expression_distribution_histogram.png"), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Histogram + KDE saved to: {plot_dir}")

    # ---------------------- Boxplot + Swarmplot ----------------------
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4), constrained_layout=True)
    axes = axes.flatten()

    labels_subset = labels[:valid_features.shape[0]]
    for i, gene in enumerate(valid_genes):
        y_values = valid_features[:, i]
        if len(y_values) != len(labels_subset):
            print(f"⚠️ Skipping {gene} (label mismatch)")
            continue

        df = pd.DataFrame({'Expression': y_values, 'Class': labels_subset})
        sns.boxplot(x='Class', y='Expression', data=df, palette="coolwarm", width=0.6, ax=axes[i])
        sns.stripplot(x='Class', y='Expression', data=df, color="black", alpha=0.5, size=3, jitter=True, ax=axes[i])
        axes[i].set_title(f"Expression of {gene}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Class", fontsize=12)
        axes[i].set_ylabel("Expression Level", fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.savefig(os.path.join(plot_dir, "gene_expression_distribution_boxplot.png"), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Boxplot + Swarm saved to: {plot_dir}")
plot_gene_expression_distribution_from_excel(
    X_sample=X_sample_flattened,
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    OutputDir=SHAP_OUTPUT_DIR,
    labels=labels,
    sheet="Top_20_by_Hybrid"
)
def plot_top_20_shap_importance_from_excel(
    shap_matrix,
    gene_ranking_excel,
    gene_names,
    OutputDir,
    sheet="Top_20_by_Hybrid"
):
    """
    Generates a horizontal barplot for top 20 genes based on SHAP importance from Excel.
    Works for both single- and multi-omics SHAP matrices.
    """
    print(f"📌 Generating Top 20 SHAP Importance Plot from: {sheet}")

    try:
        # ✅ Load top genes from Excel
        df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
        top_gene_names = df_top["Gene"].tolist()

        # ✅ Calculate SHAP-based importance (mean abs SHAP)
        if shap_matrix.ndim == 3:
            print("📌 Multi-omics SHAP matrix detected — averaging across last axis.")
            gene_importance = np.abs(shap_matrix).mean(axis=(0, 2))
        else:
            print("📌 Single-omics SHAP matrix detected — averaging directly.")
            gene_importance = np.abs(shap_matrix).mean(axis=0)

        # ✅ Get importance for top genes
        gene_importance_map = dict(zip(gene_names, gene_importance))
        valid_genes = [g for g in top_gene_names if g in gene_importance_map]
        top_importance_values = [gene_importance_map[g] for g in valid_genes]

        if not valid_genes:
            raise ValueError("❌ None of the top genes match the SHAP matrix feature names.")

        # ✅ Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(valid_genes[::-1], top_importance_values[::-1], color="royalblue")
        ax.set_xlabel("SHAP Importance", fontsize=14, fontweight="bold")
        ax.set_ylabel("Genes", fontsize=14, fontweight="bold")
        ax.set_title(f"Top 20 Genes by SHAP Importance ({sheet})", fontsize=16, fontweight="bold")
        plt.gca().invert_yaxis()

        # ✅ Save
        plot_dir = os.path.join(OutputDir, sheet)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "top_20_shap_importance.png"), dpi=300, bbox_inches="tight")
        plt.show()

        print(f"✅ Top 20 SHAP Importance Plot saved to {plot_dir}")

    except Exception as e:
        print(f"❌ Error in `plot_top_20_shap_importance_from_excel()`: {e}")
plot_top_20_shap_importance_from_excel(
    shap_matrix=shap_matrix_agg,
    gene_ranking_excel=os.path.join(SHAP_OUTPUT_DIR, "shap_gene_ranking_comparison.xlsx"),
    gene_names=final_gene_names,
    OutputDir=SHAP_OUTPUT_DIR,
    sheet="Top_20_by_Hybrid"
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_pca_heatmap_from_excel(
    X_sample, gene_ranking_excel, gene_names, OutputDir, sheet="Top_20_by_Hybrid"
):
    print(f"📌 Generating PCA Heatmap from: {sheet}")
    try:
        df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
        top_gene_names = df_top["Gene"].tolist()
        top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

        top_gene_features = X_sample[:, top_gene_indices]
        if top_gene_features.ndim == 3:
            print("📌 Multi-omics detected: Averaging across last axis.")
            top_gene_features = top_gene_features.mean(axis=-1)

        scaler = StandardScaler()
        top_gene_features_scaled = scaler.fit_transform(top_gene_features.T)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(top_gene_features_scaled)
        df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=top_gene_names)

        plt.figure(figsize=(12, 8))
        sns.heatmap(df_pca, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.8, linecolor="white")
        plt.xlabel("Principal Components", fontsize=14, fontweight="bold")
        plt.ylabel("Genes", fontsize=14, fontweight="bold")
        plt.title("PCA-Based Heatmap of Gene Expression", fontsize=16, fontweight="bold")

        plot_dir = os.path.join(OutputDir, sheet)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "pca_gene_expression_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✅ PCA Heatmap saved to {plot_dir}")
    except Exception as e:
        print(f"❌ Error in PCA heatmap: {e}")
import umap

def plot_umap_gene_embedding_from_excel(
    X_sample, gene_ranking_excel, gene_names, OutputDir, sheet="Top_20_by_Hybrid"
):
    print(f"📌 Generating UMAP Gene Expression Embedding from: {sheet}")
    try:
        df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
        top_gene_names = df_top["Gene"].tolist()
        top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

        top_gene_features = X_sample[:, top_gene_indices]
        if top_gene_features.ndim == 3:
            print("📌 Multi-omics detected: Averaging across last axis.")
            top_gene_features = top_gene_features.mean(axis=-1)

        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
        umap_result = reducer.fit_transform(top_gene_features.T)

        df_umap = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"], index=top_gene_names)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x="UMAP1", y="UMAP2", data=df_umap, hue=df_umap.index, palette="tab10", legend=False)
        plt.xlabel("UMAP Dimension 1", fontsize=14, fontweight="bold")
        plt.ylabel("UMAP Dimension 2", fontsize=14, fontweight="bold")
        plt.title("UMAP-Based Gene Expression Embedding", fontsize=16, fontweight="bold")

        plot_dir = os.path.join(OutputDir, sheet)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "umap_gene_expression_embedding.png"), dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✅ UMAP Embedding saved to {plot_dir}")
    except Exception as e:
        print(f"❌ Error in UMAP embedding: {e}")
def plot_feature_importance_from_excel(
    shap_matrix, gene_ranking_excel, gene_names, OutputDir, sheet="Top_20_by_Hybrid"
):
    print(f"📌 Generating SHAP Feature Importance Barplot from: {sheet}")
    try:
        df_top = pd.read_excel(gene_ranking_excel, sheet_name=sheet)
        top_gene_names = df_top["Gene"].tolist()
        top_gene_indices = [gene_names.index(g) for g in top_gene_names if g in gene_names]

        if shap_matrix.ndim == 3:
            gene_importance = np.abs(shap_matrix).mean(axis=(0, 2))
        else:
            gene_importance = np.abs(shap_matrix).mean(axis=0)

        norm = mcolors.Normalize(vmin=np.min(gene_importance), vmax=np.max(gene_importance))
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])

        colors = [plt.cm.coolwarm(norm(gene_importance[i])) for i in top_gene_indices]
        plt.figure(figsize=(14, 7))
        plt.barh([gene_names[i] for i in top_gene_indices][::-1], gene_importance[top_gene_indices][::-1],
                 color=colors[::-1], edgecolor="black", linewidth=1)

        plt.xlabel("Mean SHAP Importance", fontsize=12, fontweight="bold", fontname="Times New Roman")
        plt.ylabel("Genes", fontsize=12, fontweight="bold", fontname="Times New Roman")
        plt.title("Top Genes by SHAP Importance", fontsize=14, fontweight="bold", fontname="Times New Roman")
        plt.gca().invert_yaxis()

        cbar = plt.colorbar(sm)
        cbar.set_label("SHAP Importance Level", fontsize=12, fontweight="bold", fontname="Times New Roman")
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        plot_dir = os.path.join(OutputDir, sheet)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "top_selected_genes_importance.png"), dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✅ SHAP Importance Barplot saved to {plot_dir}")
    except Exception as e:
        print(f"❌ Error in SHAP feature importance plot: {e}")
sheet_name = "Top_20_by_Hybrid"
plot_pca_heatmap_from_excel(X_sample_flattened, excel_path, final_gene_names, SHAP_OUTPUT_DIR, sheet=sheet_name)
plot_umap_gene_embedding_from_excel(X_sample_flattened, excel_path, final_gene_names, SHAP_OUTPUT_DIR, sheet=sheet_name)
plot_feature_importance_from_excel(shap_matrix_agg, excel_path, final_gene_names, SHAP_OUTPUT_DIR, sheet=sheet_name)
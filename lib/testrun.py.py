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
lib_path = os.path.abspath(os.path.join(os.getcwd(), "lib"))
sys.path.insert(0, lib_path)
print(f"Library path added: {lib_path}")

from coarsening import graph_laplacian
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


# --------- filters ----------
    """
def get_k_values(filter_type):
    
    Dynamic K grid per filter type (your 'best K' choice by default).
    Edit these to sweep wider if you want.

    k_values_map = {
        "all":          [6],
        "low":          [10],
        "high":         [8],
        "impulse_low":  [6],
        "impulse_high": [8],
        "band":         [16],
        "band_reject":  [10],
        "comb":         [14],
    }
    if filter_type not in k_values_map:
        raise ValueError(f"❌ Invalid filter type: {filter_type}")
    return k_values_map[filter_type]
"""
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
        theta = [1 if i % 2 == 0 else 0 for i in range(k + 1)]
    elif filter_type == "impulse_low":
        theta = [1] + [0] * k
    elif filter_type == "impulse_high":
        theta = [0] * k + [1]
    elif filter_type == "all":
        theta = [1] * (k + 1)
    elif filter_type == "comb":
        theta = [(-1) ** i for i in range(k + 1)]
    else:
        raise ValueError(f"❌ Unknown filter type: {filter_type}")
    assert len(theta) == k + 1, f"❌ compute_theta() length mismatch for K={k}"
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

# flags
parser.add_argument('--singleton', dest='singleton', action='store_true')
parser.add_argument('--no-singleton', dest='singleton', action='store_false'); parser.set_defaults(singleton=True)
parser.add_argument('--savemodel', dest='savemodel', action='store_true')
parser.add_argument('--no-savemodel', dest='savemodel', action='store_false'); parser.set_defaults(savemodel=False)
parser.add_argument('--loaddata', dest='loaddata', action='store_true')
parser.add_argument('--no-loaddata', dest='loaddata', action='store_false'); parser.set_defaults(loaddata=True)

parser.add_argument('--num_selected_genes', type=int, default=10)

parser.add_argument('--filter_type', type=str, default='impulse_low',
                    choices=['low','high','band','impulse_low','impulse_high','all','band_reject','comb'])

parser.add_argument('--stream_mode', type=str, default='fusion',
                    choices=['fusion','gcn_only','mlp_only'])

# output/seed/data
parser.add_argument('--do_shap', action='store_true')
parser.add_argument('--output_dir', type=str, default=r"D:\GS\gsfiltersharply\review2resluts")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_root', type=str, default=r"D:\GS\gsfiltersharply\data1")

# NEW: optional per-run tag to avoid overwrites and help organization
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
    "all":          [6],
    "low":          [10],
    "high":         [8],
    "impulse_low":  [6],
    "impulse_high": [8],
    "band":         [16],
    "band_reject":  [10],
    "comb":         [14],
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

# =========================================================
#                Setup / paths / seeding
# =========================================================
import random  # for full reproducibility

_stream_map = {'fusion': 'fusion', 'gcn_only': 'gcn', 'mlp_only': 'mlp'}
stream_mode = _stream_map[args.stream_mode]
print(f"[Ablation] stream_mode = {stream_mode}")

# Root output
OutputDir = args.output_dir
os.makedirs(OutputDir, exist_ok=True)
try:
    os.chmod(OutputDir, 0o777)
except Exception:
    pass

# ---- Per-run tag (never overwrite) ----
RUN_TAG = f"{args.database}_{stream_mode}_nomic{args.num_omic}_seed{args.seed}_{'ncv' if args.ncv else 'single'}"
if getattr(args, "run_tag", ""):
    RUN_TAG += f"_{args.run_tag}"
RUN_TAG += "_" + time.strftime("%Y%m%d-%H%M%S")

run_dir = os.path.join(OutputDir, "runs", RUN_TAG)
os.makedirs(run_dir, exist_ok=True)
print(f"[Run] artifacts will also be mirrored to: {run_dir}")

# Keep a common place for fold indices (single & nested)
splits_dir = os.path.join(OutputDir, "splits")
os.makedirs(splits_dir, exist_ok=True)

# ---- Deterministic seeding (Python/NumPy/PyTorch) ----
os.environ["PYTHONHASHSEED"] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Make PyTorch more deterministic where possible
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

try:
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Env] seed={args.seed} | device={device} | CUDA={torch.cuda.is_available()}")

# ---------- paths + data loading + outputs ----------
# Tag this run so later tables/plots can separate single vs nested CV
CV_MODE = "nested" if getattr(args, "ncv", False) else "single"

def get_data_paths(database: str, data_root: str):
    database = str(database).lower().strip()
    expression_data_path      = os.path.join(data_root, "normalized_common_expression_data.tsv")
    cnv_data_path             = os.path.join(data_root, "normalized_common_cnv_data.tsv")
    expression_variance_file  = os.path.join(data_root, "normalized_expression_variance.tsv")
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

    # Optional but useful sanity checks (fail early with clear messages)
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

# Resolve data files
(expression_data_path, cnv_data_path, expression_variance_file, shuffle_index_path,
 adjacency_matrix_file, non_null_index_path) = get_data_paths(args.database, args.data_root)

print("Using adjacency matrix:", adjacency_matrix_file)
print("Using non-null index file:", non_null_index_path)

# Load data (leak-free tables)
print("Loading raw data tables...")
if args.num_omic == 1:
    expr_all_df = utilsdata.load_singleomic_data(expression_data_path)
    cnv_all_df  = None
else:
    expr_all_df, cnv_all_df = utilsdata.load_multiomics_data(expression_data_path, cnv_data_path)

# Basic label checks
if "icluster_cluster_assignment" not in expr_all_df.columns:
    raise KeyError("Column 'icluster_cluster_assignment' not found in expression table.")
labels_all = (expr_all_df['icluster_cluster_assignment'].values - 1).astype(np.int64)
out_dim = int(np.unique(labels_all).size)
print("Classes:", out_dim)

F_0    = args.num_omic
D_g    = args.num_gene
CL1_F  = 5
FC1_F  = 32
FC2_F  = 0
NN_FC1 = 256
NN_FC2 = 32

# Output dirs
hyperparam_dir = os.path.join(OutputDir, "hyperparameter_tuning")
bestmodel_dir  = os.path.join(OutputDir, "bestmodel")
for _d in (OutputDir, hyperparam_dir, bestmodel_dir):
    os.makedirs(_d, exist_ok=True)
    try:
        os.chmod(_d, 0o777)
    except Exception:
        pass

# Results files (keep names stable; we add a cv_mode column later when writing)
csv_file_path = os.path.join(OutputDir, "00finalcoexsingle2000.csv")
per_fold_csv  = os.path.join(OutputDir, "per_fold_metrics.csv")

# Runtime/memory helpers
try:
    import psutil
    _psutil_ok = True
    process = psutil.Process(os.getpid())
except Exception:
    _psutil_ok = False
    process = None


def _ncv_eval_one_split(expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
                        args, stream_mode, device, out_dim,
                        F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2,
                        train_idx, val_idx, k, filt, bs, dp, lr,
                        log_runtime=False, process=None):
    t0 = time.perf_counter()

    # ---------- deterministic per-split seed (no time-based jitter) ----------
    def _stable_seed_from_indices(base_seed, tr, va, k_, filt_, bs_, dp_, lr_):
        # build a small 32-bit hash from indices + hyperparams
        h = (int(np.sum(tr) % 10000019) * 1315423911) ^ ((int(np.sum(va) % 10000079) << 1) & 0xFFFFFFFF)
        h ^= (int(k_) * 2654435761) & 0xFFFFFFFF
        h ^= (hash(str(filt_)) & 0xFFFFFFFF)
        h ^= (int(bs_) * 2246822519) & 0xFFFFFFFF
        h ^= (int(float(dp_) * 1e6)) & 0xFFFFFFFF
        h ^= (int(float(lr_) * 1e9)) & 0xFFFFFFFF
        return (int(base_seed) ^ (h & 0x7FFFFFFF)) % (2**31 - 1)

    local_seed = _stable_seed_from_indices(args.seed, train_idx, val_idx, k, filt, bs, dp, lr)
    g = torch.Generator(device='cpu').manual_seed(local_seed)

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

    # --- Laplacian / theta (skip for MLP stream) ---
    if stream_mode == "mlp":
        L_list, theta = None, None
    else:
        A_max = A_sel.max()
        if hasattr(A_max, "A"):  # numpy.matrix -> scalar
            A_max = A_max.A.item()
        A_norm = A_sel if A_max == 0 else (A_sel / A_max)
        L_sp = graph_laplacian(A_norm, normalized=True)  # CSR
        L_torch = torch.tensor(L_sp.toarray(), dtype=torch.float32, device=device)
        L_list = [L_torch]
        theta = compute_theta(int(k), str(filt))

    # --- tensors/loaders ---
    X_tr = torch.tensor(X_all[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_all[train_idx], dtype=torch.long)
    X_va = torch.tensor(X_all[val_idx],   dtype=torch.float32)
    y_va = torch.tensor(y_all[val_idx],   dtype=torch.long)

    pin = torch.cuda.is_available()
    tr_loader = Data.DataLoader(
        Data.TensorDataset(X_tr, y_tr),
        batch_size=int(bs), shuffle=True, generator=g,
        num_workers=0, pin_memory=pin
    )
    va_loader = Data.DataLoader(
        Data.TensorDataset(X_va, y_va),
        batch_size=int(bs), shuffle=False,
        num_workers=0, pin_memory=pin
    )

    # --- model ---
    net_params = [F_0, len(gene_list), CL1_F, int(k), FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
    net = Graph_GCN(net_params, stream_mode=stream_mode).to(device)

    def _weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(_weight_init)

    optimizer = optim.Adam(net.parameters(), lr=float(lr))
    criterion = nn.CrossEntropyLoss()

    # ensure VRAM peak is measured per-split
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- train ---
    for _ in range(int(args.epochs)):
        train_model(tr_loader, net, optimizer, criterion, device, float(dp), L_list, theta)

    # --- validate ---
    y_true, y_scores = test_model(va_loader, net, device, L_list, theta)
    y_pred = np.argmax(y_scores, axis=1)

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]
    acc = accuracy_score(y_true, y_pred)

    # macro AUC (safe if some classes absent)
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

    aux = {}
    if log_runtime:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        aux["time_sec"] = time.perf_counter() - t0
        aux["rss_mb"]   = (process.memory_info().rss / (1024**2)) if (process is not None) else np.nan
        aux["vram_mb"]  = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
        aux["seed"]     = int(local_seed)
        aux["train_n"]  = int(len(train_idx))
        aux["val_n"]    = int(len(val_idx))

    # --- free big tensors ---
    del X_tr, y_tr, X_va, y_va, tr_loader, va_loader, X_all, y_all, A_sel
    if stream_mode != "mlp":
        del L_sp, L_torch, L_list
    del net, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sel_score, rep, acc, macro_f1, macro_auc, aux

def nested_cv_run(expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
                  F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim,
                  filter_types, get_k_values_fn, batch_sizes, dropout_values, lr_values,
                  singleton, stream_mode, device, args, process=None):
    """
    Nested CV:
      - inner loop: hyperparam search (scores are averaged per config)
      - outer loop: retrain best config on outer-train, evaluate on outer-test
    Returns:
      outer_df  : per-outer-fold test metrics of the chosen best config
      search_df : inner-loop grid (mean/std of selection metric per config)
      runtime_df: optional inner-loop runtime/memory rows if args.ncv_log_runtime
    """
    # -------- labels for stratification --------
    if "icluster_cluster_assignment" not in expr_all_df.columns:
        raise KeyError("Column 'icluster_cluster_assignment' not found in expression table.")
    labels_all = (expr_all_df['icluster_cluster_assignment'].values - 1).astype(np.int64)

    K_outer, K_inner = args.outer_folds, args.inner_folds
    kf_outer = StratifiedKFold(n_splits=K_outer, shuffle=True, random_state=args.seed)

    outer_rows, search_rows, runtime_rows = [], [], []
    ncv_dir = os.path.join(args.output_dir, "nested_cv")
    os.makedirs(ncv_dir, exist_ok=True)

    # ---------- OUTER LOOP (assessment) ----------
    for outer_fold, (outer_tr_idx, outer_te_idx) in enumerate(
            kf_outer.split(np.zeros(len(labels_all)), labels_all), start=1):

        # Save outer splits for reproducibility
        np.savetxt(os.path.join(ncv_dir, f"outer_fold_{outer_fold}_train_idx.txt"),
                   outer_tr_idx, fmt="%d")
        np.savetxt(os.path.join(ncv_dir, f"outer_fold_{outer_fold}_test_idx.txt"),
                   outer_te_idx,  fmt="%d")

        # ---------- INNER LOOP (model selection) ----------
        kf_inner = StratifiedKFold(n_splits=K_inner, shuffle=True,
                                   random_state=args.seed + outer_fold)

        # Pre-compute & SAVE inner splits ONCE per outer fold; reuse for all configs
        inner_splits_dir = os.path.join(ncv_dir, f"outer_fold_{outer_fold}_inner_splits")
        os.makedirs(inner_splits_dir, exist_ok=True)
        inner_splits = list(
            kf_inner.split(np.zeros(len(outer_tr_idx)), labels_all[outer_tr_idx])
        )
        # Persist mapped-to-global indices for transparency
        for inner_id, (tr_loc, va_loc) in enumerate(inner_splits, start=1):
            tr_idx = outer_tr_idx[tr_loc]; va_idx = outer_tr_idx[va_loc]
            np.savetxt(os.path.join(inner_splits_dir, f"inner_{inner_id}_train_idx.txt"),
                       tr_idx, fmt="%d")
            np.savetxt(os.path.join(inner_splits_dir, f"inner_{inner_id}_val_idx.txt"),
                       va_idx, fmt="%d")

        grid_records = []

        for filt in filter_types:
            for k in get_k_values_fn(filt):
                for bs in batch_sizes:
                    for dp in dropout_values:
                        for lr in lr_values:
                            inner_scores = []
                            # Reuse the same inner splits for this outer fold
                            for inner_id, (tr_loc, va_loc) in enumerate(inner_splits, start=1):
                                tr_idx = outer_tr_idx[tr_loc]
                                va_idx = outer_tr_idx[va_loc]

                                sc, _, _, _, _, aux = _ncv_eval_one_split(
                                    expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
                                    args, stream_mode, device, out_dim,
                                    F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2,
                                    tr_idx, va_idx, int(k), str(filt), int(bs), float(dp), float(lr),
                                    log_runtime=args.ncv_log_runtime,
                                    process=process
                                )
                                inner_scores.append(sc)

                                if args.ncv_log_runtime and isinstance(aux, dict):
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
                                        "k": int(k),
                                        "batch_size": int(bs),
                                        "dropout": float(dp),
                                        "lr": float(lr),
                                        "time_sec": aux.get("time_sec", np.nan),
                                        "rss_mb": aux.get("rss_mb", np.nan),
                                        "vram_mb": aux.get("vram_mb", np.nan),
                                        "seed": aux.get("seed", np.nan),
                                        "train_n": aux.get("train_n", np.nan),
                                        "val_n": aux.get("val_n", np.nan),
                                    })

                            # record mean/std of the selection metric for this config
                            grid_records.append({
                                "cv_mode": "nested",
                                "database": args.database,
                                "stream_mode": stream_mode,
                                "num_omic": args.num_omic,
                                "num_gene": args.num_gene,
                                "epochs": args.epochs,
                                "outer_fold": outer_fold,
                                "filter_type": filt,
                                "k": int(k),
                                "batch_size": int(bs),
                                "dropout": float(dp),
                                "lr": float(lr),
                                f"{args.score}_mean": float(np.mean(inner_scores)) if len(inner_scores) else np.nan,
                                f"{args.score}_std":  float(np.std(inner_scores, ddof=1)) if len(inner_scores) > 1 else 0.0,
                            })

        # ---- Choose best config by inner mean of selection metric ----
        score_col = f"{args.score}_mean"
        grid_df = pd.DataFrame(grid_records)

        # 1) sanity checks
        if grid_df.empty:
            raise RuntimeError(
                "Inner search produced no configurations. "
                "Check --filters / --k_by_filter / --batch_sizes / --dropouts / --lrs."
            )
        if score_col not in grid_df.columns:
            raise KeyError(f"Expected column '{score_col}' not found in inner-search grid.")

        # 2) keep only finite scores (drop NaN/inf rows)
        grid_df[score_col] = pd.to_numeric(grid_df[score_col], errors="coerce")
        grid_df = grid_df[np.isfinite(grid_df[score_col])]
        if grid_df.empty:
            raise RuntimeError(
                f"All inner-CV '{score_col}' values are NaN/inf. Likely no valid folds produced a score."
            )

        # 3) sort and pick the best (desc)
        grid_df = grid_df.sort_values(by=[score_col], ascending=False).reset_index(drop=True)
        best = grid_df.iloc[0].to_dict()

        # keep the full cleaned grid in the search log
        search_rows.extend(grid_df.to_dict("records"))

        # 4) Retrain best on outer-train, test on outer-test
        best_filt = str(best["filter_type"])
        best_k    = int(best["k"])
        best_bs   = int(best["batch_size"])
        best_dp   = float(best["dropout"])
        best_lr   = float(best["lr"])

        _, rep, acc, macro_f1, macro_auc, _ = _ncv_eval_one_split(
            expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
            args, stream_mode, device, out_dim,
            F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2,
            outer_tr_idx, outer_te_idx, best_k, best_filt, best_bs, best_dp, best_lr,
            log_runtime=False, process=process
        )

        # Outer test row (add per-class if present)
        per_class = {int(k_): v for k_, v in rep.items() if str(k_).isdigit()}
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

    outer_df   = pd.DataFrame(outer_rows)
    search_df  = pd.DataFrame(search_rows)
    runtime_df = pd.DataFrame(runtime_rows) if (args.ncv_log_runtime and len(runtime_rows)) else None
    return outer_df, search_df, runtime_df


def _df_to_latex_simple(df: pd.DataFrame) -> str:
    """
    Minimal ASCII-safe LaTeX tabular for the summary table.
    Expects columns: Metric, Mean, Std, n, cv_mode, database, stream_mode, num_omic, num_gene, epochs
    """
    # 10 columns => alignment string must have 10 specifiers
    lines = [
        "\\begin{tabular}{lccrllllrr}",
        "\\toprule",
        "Metric & Mean & Std & n & cv\\_mode & database & stream & num\\_omic & num\\_gene & epochs \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['Metric']} & {r['Mean']} & {r['Std']} & {r['n']} & "
            f"{r['cv_mode']} & {r['database']} & {r['stream_mode']} & "
            f"{r['num_omic']} & {r['num_gene']} & {r['epochs']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def export_main_text_tables(outer_df, out_dir, top_k_classes=3, split_by_mode=True):
    """
    Build paper-ready macro & per-class summaries.
    - If cv_mode column exists, writes per-mode files and a combined-by-mode file.
    - Adds context cols: cv_mode, database, stream_mode, num_omic, num_gene, epochs.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = outer_df.copy()

    # Safe numerics
    for col in ("accuracy", "macro_f1", "macro_auc", "num_omic", "num_gene", "epochs"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cls_f1_cols = [c for c in df.columns if str(c).startswith("f1_c")]

    def _summary_for(sub: pd.DataFrame) -> pd.DataFrame:
        # meta (first row if present)
        meta = {
            "cv_mode":     (sub["cv_mode"].iloc[0]     if "cv_mode" in sub.columns and len(sub) else ""),
            "database":    (sub["database"].iloc[0]    if "database" in sub.columns and len(sub) else ""),
            "stream_mode": (sub["stream_mode"].iloc[0] if "stream_mode" in sub.columns and len(sub) else ""),
            "num_omic":    (sub["num_omic"].iloc[0]    if "num_omic" in sub.columns and len(sub) else ""),
            "num_gene":    (sub["num_gene"].iloc[0]    if "num_gene" in sub.columns and len(sub) else ""),
            "epochs":      (sub["epochs"].iloc[0]      if "epochs" in sub.columns and len(sub) else ""),
        }
        n = int(len(sub))

        rows = [["Metric","Mean","Std","n","cv_mode","database","stream_mode","num_omic","num_gene","epochs"]]
        for m in [c for c in ("accuracy","macro_f1","macro_auc") if c in sub.columns]:
            mu = float(np.nanmean(sub[m])) if n else np.nan
            sd = float(np.nanstd(sub[m], ddof=1)) if n > 1 else 0.0
            rows.append([m, f"{mu:.4f}", f"{sd:.4f}", str(n),
                         meta["cv_mode"], meta["database"], meta["stream_mode"],
                         meta["num_omic"], meta["num_gene"], meta["epochs"]])

        # Top-K classes by mean F1 (if present)
        if cls_f1_cols:
            means = []
            for c in cls_f1_cols:
                try:
                    cid = int(str(c).split("f1_c")[1])
                except Exception:
                    continue
                means.append((cid, float(np.nanmean(pd.to_numeric(sub.get(c, pd.Series(dtype=float)), errors="coerce")))))
            means.sort(key=lambda t: (-(t[1] if np.isfinite(t[1]) else -1e9)))
            selected = [cid for cid, _ in means[:max(0, int(top_k_classes))]]

            for cid in selected:
                pr = float(np.nanmean(pd.to_numeric(sub.get(f"prec_c{cid}", pd.Series(dtype=float)), errors="coerce")))
                rc = float(np.nanmean(pd.to_numeric(sub.get(f"rec_c{cid}",  pd.Series(dtype=float)), errors="coerce")))
                f1 = float(np.nanmean(pd.to_numeric(sub.get(f"f1_c{cid}",   pd.Series(dtype=float)), errors="coerce")))
                rows.append([f"class {cid} precision", f"{pr:.4f}", "-", str(n),
                             meta["cv_mode"], meta["database"], meta["stream_mode"],
                             meta["num_omic"], meta["num_gene"], meta["epochs"]])
                rows.append([f"class {cid} recall",    f"{rc:.4f}", "-", str(n),
                             meta["cv_mode"], meta["database"], meta["stream_mode"],
                             meta["num_omic"], meta["num_gene"], meta["epochs"]])
                rows.append([f"class {cid} F1",        f"{f1:.4f}", "-", str(n),
                             meta["cv_mode"], meta["database"], meta["stream_mode"],
                             meta["num_omic"], meta["num_gene"], meta["epochs"]])

        return pd.DataFrame(rows[1:], columns=rows[0])

    if split_by_mode and "cv_mode" in df.columns:
        # Per-mode exports
        frames = []
        for mode, sub in df.groupby("cv_mode", dropna=False):
            mode_tag = str(mode) if pd.notna(mode) else "unknown"
            tbl = _summary_for(sub)
            frames.append(tbl)
            csv_path = os.path.join(out_dir, f"main_text_{mode_tag}_macro_and_class.csv")
            tex_path = os.path.join(out_dir, f"main_text_{mode_tag}_macro_and_class.tex")
            tbl.to_csv(csv_path, index=False)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(_df_to_latex_simple(tbl))

        # Combined by-mode export
        combined = pd.concat(frames, ignore_index=True) if frames else _summary_for(df)
        combined_csv = os.path.join(out_dir, "main_text_macro_and_class_by_mode.csv")
        combined_tex = os.path.join(out_dir, "main_text_macro_and_class_by_mode.tex")
        combined.to_csv(combined_csv, index=False)
        with open(combined_tex, "w", encoding="utf-8") as f:
            f.write(_df_to_latex_simple(combined))

        print(f"✅ Main-text tables saved in {out_dir} (per-mode + combined).")
    else:
        # Single table (no cv_mode available)
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

# Use parsed grids from CLI (fallback to your defaults if user didn't pass them)
if not filter_types:    filter_types    = ["all", "band_reject"]
if not batch_sizes:     batch_sizes     = [128]
if not dropout_values:  dropout_values  = [0.2]
if not lr_values:       lr_values       = [0.001]

print("[Grid] filters=", filter_types,
      "| batch_sizes=", batch_sizes,
      "| dropouts=", dropout_values,
      "| lrs=", lr_values)

# -------------------- NESTED CV --------------------
if args.ncv:
    print(">>> Running Nested Cross-Validation (outer=assessment, inner=selection)…")

    def _kvals(ft): return get_k_values(ft)

    outer_df, search_df, runtime_df = nested_cv_run(
        expr_all_df, cnv_all_df, adjacency_matrix_file, non_null_index_path,
        F_0, CL1_F, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim,
        filter_types, _kvals, batch_sizes, dropout_values, lr_values,
        args.singleton, stream_mode, device, args, process=process if _psutil_ok else None
    )

    # Tag with context so tables/plots are self-descriptive
    outer_df["cv_mode"]     = "nested"
    outer_df["database"]    = args.database
    outer_df["stream_mode"] = stream_mode
    outer_df["num_omic"]    = args.num_omic
    outer_df["num_gene"]    = args.num_gene
    outer_df["epochs"]      = args.epochs

    ncv_dir = os.path.join(OutputDir, "nested_cv")
    os.makedirs(ncv_dir, exist_ok=True)

    outer_csv  = os.path.join(ncv_dir, "outer_test_metrics.csv")
    search_csv = os.path.join(ncv_dir, "inner_search_grid.csv")
    outer_df.to_csv(outer_csv, index=False)
    search_df.to_csv(search_csv, index=False)

    if runtime_df is not None and not runtime_df.empty:
        runtime_csv = os.path.join(ncv_dir, "inner_runtime_memory.csv")
        runtime_df.to_csv(runtime_csv, index=False)
        print(f"✅ NCV runtime saved: {runtime_csv}")
        # Optional small summary figs/tables (if helper is present)
        try:
            export_runtime_memory_by_k(runtime_df, ncv_dir,
                tag=f"{args.database}_{stream_mode}_nomic{args.num_omic}")
        except NameError:
            pass

    # Main-text macro + class highlights (per reviewer #4)
    export_main_text_tables(outer_df, ncv_dir, top_k_classes=3)

    # Mirror key artifacts to the per-run directory to avoid future overwrites
    import shutil
    for p in [outer_csv, search_csv] + (
        [os.path.join(ncv_dir, "inner_runtime_memory.csv")] if runtime_df is not None and not runtime_df.empty else []
    ):
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(run_dir, os.path.basename(p)))

    print(f"✅ NCV artifacts:\n  {outer_csv}\n  {search_csv}")
    sys.exit(0)

# -------------------- SINGLE CV (original) --------------------
print(">>> Running single CV (original) …")

# Load or create results CSV (include cv_mode + context columns)
expected_cols = [
    "database","stream_mode","num_omic",
    "filter_type","k","num_genes","batch_size","dropout","lr",
    "accuracy","precision","recall","macro_f1",
    "acc_mean","acc_std","macro_f1_mean","macro_f1_std",
    "macro_precision_mean","macro_precision_std","macro_recall_mean","macro_recall_std",
    "run_time_sec","peak_rss_mb","peak_vram_mb",
    "cv_mode","epochs","num_gene"   # <- extra context (epochs,num_gene) for clarity
]

if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    df_existing = pd.read_csv(csv_file_path)
else:
    df_existing = pd.DataFrame(columns=expected_cols)

# Ensure all expected columns exist
for c in expected_cols:
    if c not in df_existing.columns:
        df_existing[c] = np.nan

# Coerce numerics
for col in ["k","num_genes","batch_size","dropout","lr","accuracy",
            "acc_mean","acc_std","macro_f1_mean","macro_f1_std",
            "macro_precision_mean","macro_precision_std","macro_recall_mean","macro_recall_std",
            "run_time_sec","peak_rss_mb","peak_vram_mb","epochs","num_gene","num_omic"]:
    if col in df_existing.columns:
        df_existing[col] = pd.to_numeric(df_existing[col], errors="coerce")

# Fill context defaults for existing rows (back-compat)
if "cv_mode" not in df_existing.columns or df_existing["cv_mode"].isna().any():
    df_existing.loc[df_existing.get("cv_mode").isna() if "cv_mode" in df_existing.columns else slice(None), "cv_mode"] = "single"
for col, default in [
    ("database", args.database),
    ("stream_mode", stream_mode),
    ("num_omic", args.num_omic),
]:
    if col not in df_existing.columns:
        df_existing[col] = default


## ---------- STRATIFIED & SAVED FOLDS (single-CV) ----------
# Ensure everyone (all configs/streams) uses the identical splits
splits_dir = os.path.join(OutputDir, "splits")
os.makedirs(splits_dir, exist_ok=True)
splits_npz = os.path.join(splits_dir, f"single_{args.database}_seed{args.seed}_k{args.num_folds}.npz")

if os.path.exists(splits_npz):
    data = np.load(splits_npz, allow_pickle=True)
    fold_splits = list(zip(data["train_idx"], data["val_idx"]))
else:
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_splits = list(skf.split(np.zeros(len(labels_all)), labels_all))
    np.savez_compressed(
        splits_npz,
        train_idx=np.array([tr for tr, _ in fold_splits], dtype=object),
        val_idx=np.array([va for _,  va in fold_splits], dtype=object),
    )  # load later with allow_pickle=True

# Human-readable copies for the supplement (works for both branches)
single_splits_dir = os.path.join(OutputDir, "single_cv_splits")
os.makedirs(single_splits_dir, exist_ok=True)
for i, (tr, va) in enumerate(fold_splits, 1):
    np.savetxt(os.path.join(single_splits_dir, f"fold_{i}_train_idx.txt"), tr, fmt="%d")
    np.savetxt(os.path.join(single_splits_dir, f"fold_{i}_val_idx.txt"),   va, fmt="%d")

# Always mirror the NPZ file; and mirror txt splits if that folder exists
try:
    if os.path.exists(splits_npz):
        shutil.copy2(splits_npz, os.path.join(run_dir, os.path.basename(splits_npz)))
    if os.path.isdir(single_splits_dir):
        dst = os.path.join(run_dir, "single_cv_splits")
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(single_splits_dir):
            srcp = os.path.join(single_splits_dir, name)
            if os.path.isfile(srcp):
                shutil.copy2(srcp, os.path.join(dst, name))
except Exception as _e:
    print(f"⚠️ Could not mirror NPZ/split txts to run_dir: {_e}")


# Dedup key (do NOT include cv_mode so prior single runs still dedup)
_key_cols = ["database","stream_mode","num_omic","filter_type","k","num_genes","batch_size","dropout","lr"]
if not df_existing.empty:
    existing_combinations = set(
        (
            row["database"],
            row["stream_mode"],
            row["num_omic"],
            row["filter_type"],
            int(row["k"]),
            int(row["num_genes"]),
            int(row["batch_size"]),
            _normf(row["dropout"]),
            _normf(row["lr"]),
        )
        for _, row in df_existing[_key_cols].dropna().iterrows()
    )
else:
    existing_combinations = set()

# Best-so-far (within current context)
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

# ---------- SINGLE-CV TRAINING (uses saved, stratified splits) ----------
for filter_type in filter_types:
    k_values = get_k_values(filter_type)
    for batch_size in batch_sizes:
        for dropout_value in dropout_values:
            for lr in lr_values:
                for k in k_values:
                    # ✅ single, normalized dedup key
                    combination_key = (
                        args.database, stream_mode, args.num_omic,
                        filter_type, int(k), int(args.num_gene), int(batch_size),
                        _normf(dropout_value), _normf(lr)
                    )

                    if combination_key in existing_combinations:
                        print(f"✅ Skipping already processed combo: {combination_key}")
                        continue

                    print(f"\n🔹 Processing: db={args.database} | stream={stream_mode} | "
                          f"filter={filter_type} | K={k} | G={args.num_gene} | "
                          f"BS={batch_size} | dp={dropout_value} | lr={lr}")

                    theta = None if stream_mode == "mlp" else compute_theta(k, filter_type)

                    all_y_true, all_y_pred, all_y_scores = [], [], []
                    fold_accs, fold_mF1s, fold_mPrecs, fold_mRecs = [], [], [], []
                    fold_rows = []

                    comb_start = time.perf_counter()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()

                    candidate_gene_list = None

                    # >>> use the stratified, saved splits <<<
                    for fold, (train_index, val_index) in enumerate(fold_splits, start=1):
                        fold_t0 = time.perf_counter()
                        print(f"  Fold {fold}…")

                        # --- leak-free gene selection (TRAIN ONLY) ---
                        gene_list, gene_idx = utilsdata.select_top_genes_from_train_fold(
                            expr_all_df, non_null_index_path, train_index, args.num_gene
                        )
                        if candidate_gene_list is None:
                            candidate_gene_list = gene_list[:]
                        G_fold = len(gene_list)

                        # --- build fold data ---
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

                        # --- Laplacian (skip for mlp) ---
                        if stream_mode == "mlp":
                            L_list = None
                        else:
                            A_max = A_sel.max()
                            if hasattr(A_max, "A"):
                                A_max = A_max.A.item()
                            A_norm = A_sel if A_max == 0 else (A_sel / A_max)
                            L_sp = graph_laplacian(A_norm, normalized=True)
                            L_torch = torch.tensor(L_sp.toarray(), dtype=torch.float32, device=device)
                            L_list = [L_torch]

                        # --- tensors/loaders ---
                        X_train = torch.tensor(X_all[train_index], dtype=torch.float32)
                        y_train = torch.tensor(y_all[train_index], dtype=torch.long)
                        X_val   = torch.tensor(X_all[val_index],   dtype=torch.float32)
                        y_val   = torch.tensor(y_all[val_index],   dtype=torch.long)

                        _pin = torch.cuda.is_available()
                        _gen = torch.Generator().manual_seed(args.seed + fold)  # reproducible shuffling
                        train_loader = Data.DataLoader(
                            Data.TensorDataset(X_train, y_train),
                            batch_size=int(batch_size), shuffle=True, generator=_gen,
                            num_workers=0, pin_memory=_pin
                        )
                        val_loader = Data.DataLoader(
                            Data.TensorDataset(X_val, y_val),
                            batch_size=int(batch_size), shuffle=False,
                            num_workers=0, pin_memory=_pin
                        )

                        # --- model ---
                        net_params_fold = [F_0, G_fold, CL1_F, k, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
                        net = Graph_GCN(net_params_fold, stream_mode=stream_mode).to(device)

                        def _weight_init(m):
                            if isinstance(m, (nn.Conv2d, nn.Linear)):
                                nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None:
                                    nn.init.zeros_(m.bias)
                        net.apply(_weight_init)

                        optimizer = optim.Adam(net.parameters(), lr=float(lr))
                        criterion = nn.CrossEntropyLoss()

                        for _ in range(int(args.epochs)):
                            train_model(train_loader, net, optimizer, criterion,
                                        device, float(dropout_value), L_list, theta)

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
                        rss_now_mb = (process.memory_info().rss / (1024**2)) if _psutil_ok else np.nan
                        vram_now_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

                        fold_rows.append({
                            "database": args.database, "stream_mode": stream_mode, "num_omic": args.num_omic,
                            "filter_type": filter_type, "k": k, "num_genes": args.num_gene,
                            "batch_size": batch_size, "dropout": dropout_value, "lr": lr,
                            "fold": fold, "accuracy": fold_acc, "macro_f1": fold_mF1,
                            "macro_precision": fold_mPrec, "macro_recall": fold_mRec,
                            "fold_time_sec": fold_time_sec, "rss_mb": rss_now_mb, "vram_peak_mb_sofar": vram_now_mb,
                            "cv_mode": "single", "epochs": args.epochs, "num_gene": args.num_gene
                        })

                        # cleanup
                        del X_train, y_train, X_val, y_val, train_loader, val_loader
                        del X_all, y_all, A_sel
                        if stream_mode != "mlp":
                            del L_sp, L_torch, L_list
                        del net, optimizer, criterion
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # ----- aggregate folds -----
                    acc_mean   = float(np.mean(fold_accs))
                    acc_std    = float(np.std(fold_accs,   ddof=1))
                    mF1_mean   = float(np.mean(fold_mF1s))
                    mF1_std    = float(np.std(fold_mF1s,   ddof=1))
                    mPrec_mean = float(np.mean(fold_mPrecs))
                    mPrec_std  = float(np.std(fold_mPrecs, ddof=1))
                    mRec_mean  = float(np.mean(fold_mRecs))
                    mRec_std   = float(np.std(fold_mRecs,  ddof=1))
                    print(f"   Fold-wise: Acc {acc_mean:.4f} ± {acc_std:.4f} | mF1 {mF1_mean:.4f} ± {mF1_std:.4f}")

                    # persist per-fold metrics
                    per_fold_df = pd.DataFrame(fold_rows)
                    if os.path.exists(per_fold_csv) and os.path.getsize(per_fold_csv) > 0:
                        _old = pd.read_csv(per_fold_csv)
                        per_fold_df = pd.concat([_old, per_fold_df], ignore_index=True)
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

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    comb_time_sec = time.perf_counter() - comb_start
                    peak_vram_mb  = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

                    new_result = {
                        "database": args.database, "stream_mode": stream_mode, "num_omic": args.num_omic,
                        "filter_type": filter_type, "k": k, "num_genes": args.num_gene,
                        "batch_size": batch_size, "dropout": dropout_value, "lr": lr,
                        "accuracy": avg_accuracy, "precision": precision, "recall": recall, "macro_f1": macro_f1,
                        "acc_mean": acc_mean, "acc_std": acc_std,
                        "macro_f1_mean": mF1_mean, "macro_f1_std": mF1_std,
                        "macro_precision_mean": mPrec_mean, "macro_precision_std": mPrec_std,
                        "macro_recall_mean": mRec_mean, "macro_recall_std": mRec_std,
                        "run_time_sec": comb_time_sec,
                        "peak_rss_mb": (process.memory_info().rss / (1024**2)) if _psutil_ok else np.nan,
                        "peak_vram_mb": peak_vram_mb,
                        "cv_mode": "single",
                        "epochs": args.epochs,
                        "num_gene": args.num_gene
                    }

                    df_existing = pd.concat([df_existing, pd.DataFrame([new_result])], ignore_index=True)
                    df_existing.to_csv(csv_file_path, index=False)

                    # keep the normalized combo in the set
                    existing_combinations.add(combination_key)

                    print(f"✅ CSV updated: {csv_file_path}")
                    print(f"   Pooled Acc={avg_accuracy:.4f} | Macro-F1={macro_f1:.4f} | "
                          f"mean Acc={acc_mean:.4f}±{acc_std:.4f} | time={comb_time_sec:.1f}s | "
                          f"RSS≈{new_result['peak_rss_mb']:.1f}MB | VRAM≈{peak_vram_mb:.1f}MB")

print("✅ Done.")

# Mirror single-CV artifacts to per-run folder (no overwrites between runs)
try:
    import shutil
    if os.path.exists(per_fold_csv):
        shutil.copy2(per_fold_csv, os.path.join(run_dir, "per_fold_metrics.csv"))
    if os.path.exists(csv_file_path):
        shutil.copy2(csv_file_path, os.path.join(run_dir, "summary.csv"))
except Exception as _e:
    print(f"⚠️ Could not mirror single-CV artifacts to run_dir: {_e}")


# =========================================================
#     EXPORT PER-FOLD + SUMMARY TO A SINGLE EXCEL
#     + SELECT BEST ROW FOR CURRENT CONTEXT
# =========================================================
try:
    # 0) Ensure cv_mode exists for single-CV rows
    if "cv_mode" not in df_existing.columns:
        df_existing["cv_mode"] = "single"  # this block runs only in single-CV branch

    # 1) Load the per-fold table we’ve been appending
    if os.path.exists(per_fold_csv) and os.path.getsize(per_fold_csv) > 0:
        df_folds_all = pd.read_csv(per_fold_csv)
    else:
        df_folds_all = pd.DataFrame(columns=[
            "database","stream_mode","num_omic","filter_type","k","num_genes",
            "batch_size","dropout","lr","fold","accuracy","macro_precision",
            "macro_recall","macro_f1","fold_time_sec","rss_mb","vram_peak_mb_sofar",
            "cv_mode"
        ])

    # Back-fill cv_mode in folds table (these are all single-CV rows here)
    if "cv_mode" not in df_folds_all.columns:
        df_folds_all["cv_mode"] = "single"

    # Keep only the current run context (avoid mixing past runs across db/stream/omics)
    df_folds = df_folds_all[
        (df_folds_all["database"] == args.database) &
        (df_folds_all["stream_mode"] == stream_mode) &
        (df_folds_all["num_omic"] == args.num_omic)
    ].copy()

    # Ensure numeric types
    num_cols = ["k","num_genes","batch_size","dropout","lr","fold",
                "accuracy","macro_precision","macro_recall","macro_f1",
                "fold_time_sec","rss_mb","vram_peak_mb_sofar"]
    for c in num_cols:
        if c in df_folds.columns:
            df_folds[c] = pd.to_numeric(df_folds[c], errors="coerce")

    # Order columns nicely for the long format
    id_cols = ["database","stream_mode","num_omic","filter_type","k",
               "num_genes","batch_size","dropout","lr","fold","cv_mode"]
    metric_cols = ["accuracy","macro_precision","macro_recall","macro_f1",
                   "fold_time_sec","rss_mb","vram_peak_mb_sofar"]
    cols_long = [c for c in id_cols if c in df_folds.columns] + \
                [c for c in metric_cols if c in df_folds.columns]
    df_folds_long = df_folds[cols_long].sort_values(
        [c for c in id_cols if c in df_folds.columns],
        kind="mergesort"
    ) if not df_folds.empty else df_folds

    # Optional: wide view with fold metrics side-by-side
    folds_wide = None
    if not df_folds_long.empty:
        try:
            idx_cols = ["database","stream_mode","num_omic","filter_type","k",
                        "num_genes","batch_size","dropout","lr","cv_mode"]
            idx_cols = [c for c in idx_cols if c in df_folds_long.columns]
            folds_wide = df_folds_long.pivot_table(
                index=idx_cols,
                columns="fold",
                values=["accuracy","macro_precision","macro_recall"],
                aggfunc="first"
            ).sort_index()
            # flatten MultiIndex columns like ('accuracy', 1) -> 'accuracy_fold1'
            folds_wide.columns = [f"{met}_fold{fold}" for met, fold in folds_wide.columns]
            folds_wide = folds_wide.reset_index()
        except Exception:
            folds_wide = None

    # 2) Build a summary sheet from df_existing (one row per combo)
    df_summary = df_existing.copy()

    # Back-fill cv_mode in summary as well
    if "cv_mode" not in df_summary.columns:
        df_summary["cv_mode"] = "single"

    # Pretty "mean ± std" columns
    def _pm(df, mean_col, std_col, out_col):
        if mean_col in df.columns and std_col in df.columns:
            df[out_col] = df.apply(
                lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}"
                if pd.notnull(r[mean_col]) and pd.notnull(r[std_col]) else "",
                axis=1
            )
    _pm(df_summary, "acc_mean", "acc_std", "acc_mean_std")
    _pm(df_summary, "macro_precision_mean", "macro_precision_std", "precision_mean_std")
    _pm(df_summary, "macro_recall_mean", "macro_recall_std", "recall_mean_std")
    _pm(df_summary, "macro_f1_mean", "macro_f1_std", "macro_f1_mean_std")

    # 3) Choose best row for CURRENT CONTEXT (db + stream + num_omic) from summary
    #    Primary metric = args.score (macro_f1/accuracy/macro_auc), with safe fallback.
    select_metric = args.score if args.score in df_summary.columns else (
        "macro_f1" if "macro_f1" in df_summary.columns else "accuracy"
    )
    # Filter to current context and single-CV only (this file is the single-CV ledger)
    df_view = df_summary[
        (df_summary["database"] == args.database) &
        (df_summary["stream_mode"] == stream_mode) &
        (df_summary["num_omic"] == args.num_omic) &
        (df_summary["cv_mode"] == "single")
    ].copy()

    best_result = None  # <-- will be used by the later "BEST RESULT SUMMARY" block
    df_best_sheet = pd.DataFrame()
    if not df_view.empty and select_metric in df_view.columns:
        # rank by selected metric desc, then accuracy desc as tie-breaker
        sort_cols = [select_metric]
        ascending = [False]
        if select_metric != "accuracy" and "accuracy" in df_view.columns:
            sort_cols.append("accuracy")
            ascending.append(False)
        df_ranked = df_view.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
        df_ranked["rank"] = np.arange(1, len(df_ranked) + 1)
        df_ranked["is_best"] = df_ranked["rank"].eq(1)

        # expose a compact "best" sheet
        keep_cols = [
            "database","stream_mode","num_omic","cv_mode",
            "filter_type","k","num_genes","batch_size","dropout","lr",
            "accuracy","macro_f1","macro_auc",
            "acc_mean","acc_std","macro_f1_mean","macro_f1_std",
            "macro_precision_mean","macro_precision_std",
            "macro_recall_mean","macro_recall_std",
            "run_time_sec","peak_rss_mb","peak_vram_mb",
            "rank","is_best"
        ]
        keep_cols = [c for c in keep_cols if c in df_ranked.columns]
        df_best_sheet = df_ranked[keep_cols].copy()

        # set best_result for downstream section
        best_result = df_ranked.iloc[0].to_dict()

    # 4) Write to a single Excel file
    xlsx_path = os.path.join(OutputDir, "results_with_folds.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df_summary.to_excel(writer, index=False, sheet_name="summary")
            df_folds_long.to_excel(writer, index=False, sheet_name="folds_long")
            if folds_wide is not None:
                folds_wide.to_excel(writer, index=False, sheet_name="folds_wide")
            if not df_best_sheet.empty:
                df_best_sheet.to_excel(writer, index=False, sheet_name="best_single")
    except Exception:
        # Fallback without specifying engine
        with pd.ExcelWriter(xlsx_path) as writer:
            df_summary.to_excel(writer, index=False, sheet_name="summary")
            df_folds_long.to_excel(writer, index=False, sheet_name="folds_long")
            if folds_wide is not None:
                folds_wide.to_excel(writer, index=False, sheet_name="folds_wide")
            if not df_best_sheet.empty:
                df_best_sheet.to_excel(writer, index=False, sheet_name="best_single")

    print(f"📊 Excel written: {xlsx_path}")

    # 5) Persist best_result to a lightweight text file too (optional)
    best_model_txt = os.path.join(OutputDir, "bestmodel.txt")
    if best_result is not None:
        with open(best_model_txt, "w", encoding="utf-8") as f:
            f.write(f"selection_metric: {select_metric}\n")
            for k, v in best_result.items():
                # only dump simple scalars
                if isinstance(v, (str, int, float, np.integer, np.floating)):
                    f.write(f"{k}: {v}\n")
        print(f"✅ Best model (single-CV context) saved to {best_model_txt}")

    # Mirror summary artifacts into the per-run folder (no overwrite collisions)
    try:
        if os.path.exists(xlsx_path):
            shutil.copy2(xlsx_path, os.path.join(run_dir, os.path.basename(xlsx_path)))
        if os.path.exists(best_model_txt):
            shutil.copy2(best_model_txt, os.path.join(run_dir, os.path.basename(best_model_txt)))
    except Exception as _e:
        print(f"⚠️ Could not mirror Excel/bestmodel to run_dir: {_e}")

except Exception as e:
    print(f"⚠️ Excel export failed: {e}")

# =========================================================
#     BEST RESULT SUMMARY (no ROC/CM copies) + SAFE ARTIFACT COPY
# =========================================================
import shutil
from glob import glob

# If best_result wasn't set by the Excel export block (should be), derive it here as a fallback.
if 'best_result' not in globals() or best_result is None:
    try:
        # Prefer the chosen selection metric; fallback to macro_f1, then accuracy.
        select_metric = args.score if 'args' in globals() and hasattr(args, 'score') else 'macro_f1'
        if not ('df_existing' in globals() and isinstance(df_existing, pd.DataFrame) and not df_existing.empty):
            best_result = None
        else:
            df_ctx = df_existing[
                (df_existing.get("database") == args.database) &
                (df_existing.get("stream_mode") == stream_mode) &
                (df_existing.get("num_omic") == args.num_omic)
            ].copy()
            if not df_ctx.empty:
                metric = select_metric if select_metric in df_ctx.columns else (
                    "macro_f1" if "macro_f1" in df_ctx.columns else "accuracy"
                )
                df_ctx = df_ctx.sort_values([metric, "accuracy"] if metric != "accuracy" and "accuracy" in df_ctx.columns else [metric],
                                            ascending=False, kind="mergesort")
                best_result = df_ctx.iloc[0].to_dict()
            else:
                best_result = None
    except Exception:
        best_result = None

best_model_txt = os.path.join(OutputDir, "bestmodel.txt")

if best_result:
    # Write a clean summary (UTF-8)
    with open(best_model_txt, "w", encoding="utf-8") as f:
        # include the selection metric we used (if known)
        sel_metric = (args.score if hasattr(args, "score") else
                      ("macro_f1" if "macro_f1" in best_result else "accuracy"))
        f.write(f"selection_metric: {sel_metric}\n")
        for key, value in best_result.items():
            if isinstance(value, (str, int, float, np.integer, np.floating)):
                f.write(f"{key}: {value}\n")
    print(f"✅ Best model configuration saved to {best_model_txt}")

    # Pretty console print
    print("\n✅ Best Overall Configuration (current context):")
    # Show a stable, readable subset first
    keys_pretty = ["database","stream_mode","num_omic","cv_mode",
                   "filter_type","k","num_genes","batch_size","dropout","lr",
                   "accuracy","macro_f1","macro_auc"]
    shown = set()
    for k in keys_pretty:
        if k in best_result:
            print(f"{k}: {best_result[k]}")
            shown.add(k)
    # Then any remaining scalar fields
    for k, v in best_result.items():
        if k not in shown and isinstance(v, (str, int, float, np.integer, np.floating)):
            print(f"{k}: {v}")

    # Optional: copy any already-generated *best_* plots (if they exist from a previous run)
    os.makedirs(bestmodel_dir, exist_ok=True)
    existing_best_pdfs = glob(os.path.join(bestmodel_dir, "best_*.pdf"))
    if existing_best_pdfs:
        print(f"📂 {len(existing_best_pdfs)} best_* PDFs already present in {bestmodel_dir}.")

    # Mirror bestmodel.txt to run_dir (again, in case it was regenerated)
    try:
        if os.path.exists(best_model_txt):
            shutil.copy2(best_model_txt, os.path.join(run_dir, os.path.basename(best_model_txt)))
    except Exception as _e:
        print(f"⚠️ Could not mirror bestmodel to run_dir: {_e}")
else:
    print("❌ No best result found.")

# =========================================================
#   (Optional) Summarize runtime/memory vs K for SINGLE-CV
# =========================================================
try:
    if os.path.exists(per_fold_csv) and os.path.getsize(per_fold_csv) > 0 and 'export_runtime_memory_by_k' in globals():
        df_r = pd.read_csv(per_fold_csv)
        # rename to expected keys
        if "fold_time_sec" in df_r.columns:
            df_r = df_r.rename(columns={"fold_time_sec":"time_sec"})
        if "vram_peak_mb_sofar" in df_r.columns:
            df_r = df_r.rename(columns={"vram_peak_mb_sofar":"vram_mb"})
        df_r = df_r[
            (df_r["database"]==args.database) &
            (df_r["stream_mode"]==stream_mode) &
            (df_r["num_omic"]==args.num_omic)
        ]
        if not df_r.empty:
            single_dir = os.path.join(OutputDir, "single_cv")
            os.makedirs(single_dir, exist_ok=True)
            tag = f"{args.database}_{stream_mode}_nomic{args.num_omic}"
            export_runtime_memory_by_k(df_r, single_dir, tag=tag)

            # mirror those single_cv artifacts to run_dir
            try:
                for p in glob(os.path.join(single_dir, "*")):
                    shutil.copy2(p, os.path.join(run_dir, os.path.basename(p)))
            except Exception as _e:
                print(f"⚠️ Could not mirror single_cv runtime artifacts to run_dir: {_e}")
except Exception as _e:
    print(f"⚠️ Single-CV runtime summary failed: {_e}")

# =========================================================
#                   Visualization from CSV (unchanged)
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
        "accuracy","precision","recall","macro_f1"
    ])

needed = {
    "database","stream_mode","num_omic",
    "filter_type","k","num_genes","batch_size","dropout","lr","accuracy"
}
missing = needed - set(df_results.columns)
if missing:
    print(f"⚠️ Missing columns in results CSV: {sorted(list(missing))}. Plots may be limited.")

# Focus plots on the current database + stream + num_omic to avoid mixing runs
if df_results.empty or not {"database","stream_mode","num_omic","filter_type"}.issubset(df_results.columns):
    print("❌ No data available for plotting. Skipping visualization.")
else:
    df_view = df_results[
        (df_results["database"] == args.database) &
        (df_results["stream_mode"] == stream_mode) &
        (df_results["num_omic"] == args.num_omic)
    ].copy()

    if df_view.empty:
        print(f"⚠️ No rows for db={args.database}, stream={stream_mode}, num_omic={args.num_omic}. Skipping plots.")
    else:
        for col in ["k","batch_size","dropout","lr","accuracy"]:
            if col in df_view.columns:
                df_view[col] = pd.to_numeric(df_view[col], errors="coerce")

        best_filter = (best_result["filter_type"]
                       if best_result
                       and best_result.get("database", args.database) == args.database
                       and best_result.get("stream_mode", stream_mode) == stream_mode
                       and best_result.get("num_omic", args.num_omic) == args.num_omic
                       else None)

        def _ensure_pdf_from_noext(path_no_ext: str) -> str:
            base, _ = os.path.splitext(path_no_ext)
            return base + ".pdf"

        def save_pdf_and_copy(fig_path_no_ext, is_best):
            pdf_path = _ensure_pdf_from_noext(fig_path_no_ext)
            plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
            plt.close()
            print(f"✅ Plot saved: {pdf_path}")
            # Mirror to bestmodel (best-only) and run_dir (all)
            try:
                if is_best:
                    best_pdf = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
                    os.makedirs(bestmodel_dir, exist_ok=True)
                    shutil.copy2(pdf_path, best_pdf)
                    print(f"✅ Best plot copied to bestmodel: {best_pdf}")
                if os.path.exists(run_dir):
                    shutil.copy2(pdf_path, os.path.join(run_dir, os.path.basename(pdf_path)))
            except Exception as _e:
                print(f"⚠️ Could not mirror plot(s): {_e}")

        # ----- 2D: Accuracy vs K (per filter) -----
        for ft in df_view["filter_type"].dropna().unique():
            df_ft = df_view[df_view["filter_type"] == ft]
            if "k" not in df_ft.columns or "accuracy" not in df_ft.columns or df_ft.empty:
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
        y_min, y_max = 1.0, 0.0
        for ft in df_view["filter_type"].dropna().unique():
            df_ft = df_view[df_view["filter_type"] == ft]
            if "k" not in df_ft.columns or "accuracy" not in df_ft.columns or df_ft.empty:
                continue
            avg_acc_per_k = df_ft.groupby("k")["accuracy"].mean().sort_index()
            if avg_acc_per_k.empty:
                continue
            plt.plot(avg_acc_per_k.index, avg_acc_per_k.values, marker="o", label=str(ft))
            y_min = min(y_min, float(np.nanmin(avg_acc_per_k.values)))
            y_max = max(y_max, float(np.nanmax(avg_acc_per_k.values)))
        plt.xlabel("K (Bernstein Polynomial Order)", fontsize=14)
        plt.ylabel("Validation Accuracy", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        if np.isfinite(y_min) and np.isfinite(y_max):
            plt.ylim(max(0.0, y_min - 0.02), min(1.0, y_max + 0.02))
        plt.legend(title="Filter Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        save_pdf_and_copy(
            os.path.join(hyperparam_dir, f"{args.database}_{stream_mode}_combined_k_vs_accuracy"),
            is_best=False
        )

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
            # mirror to bestmodel (if best) and always to run_dir
            try:
                if ft == best_filter:
                    dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
                    os.makedirs(bestmodel_dir, exist_ok=True)
                    shutil.copy2(pdf_path, dst)
                    print(f"✅ Best 3D plot (K vs Batch Size) copied to bestmodel: {dst}")
                if os.path.exists(run_dir):
                    shutil.copy2(pdf_path, os.path.join(run_dir, os.path.basename(pdf_path)))
            except Exception as _e:
                print(f"⚠️ Could not mirror 3D (K×Batch) plot(s): {_e}")

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
            try:
                if ft == best_filter:
                    dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
                    os.makedirs(bestmodel_dir, exist_ok=True)
                    shutil.copy2(pdf_path, dst)
                    print(f"✅ Best 3D plot (K vs Dropout) copied to bestmodel: {dst}")
                if os.path.exists(run_dir):
                    shutil.copy2(pdf_path, os.path.join(run_dir, os.path.basename(pdf_path)))
            except Exception as _e:
                print(f"⚠️ Could not mirror 3D (K×Dropout) plot(s): {_e}")

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
            try:
                if ft == best_filter:
                    dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
                    os.makedirs(bestmodel_dir, exist_ok=True)
                    shutil.copy2(pdf_path, dst)
                    print(f"✅ Best 3D plot (K vs Learning Rate) copied to bestmodel: {dst}")
                if os.path.exists(run_dir):
                    shutil.copy2(pdf_path, os.path.join(run_dir, os.path.basename(pdf_path)))
            except Exception as _e:
                
                print(f"⚠️ Could not mirror 3D (K×LR) plot(s): {_e}") 


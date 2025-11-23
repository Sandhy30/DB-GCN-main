##!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# -------------------- quiet noisy deps --------------------
warnings.filterwarnings("ignore")

# -------------------- make sure we can import from ./lib --------------------
lib_path = os.path.abspath(os.path.join(os.getcwd(), "lib"))
sys.path.insert(0, lib_path)
print(f"Library path added: {lib_path}")

# -------------------- project modules --------------------
from coarsening import graph_laplacian
from layermodel import Graph_GCN
import utilsdata

# =========================================================
#                     Train / Evaluate
# =========================================================
def train_model(train_loader, net, optimizer, criterion, device, dropout_value, L, theta):
    """
    Train for one epoch with CrossEntropyLoss on logits.
    """
    net.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, _, logits, _ = net(batch_x, dropout_value, L, theta)  # logits (NOT log-softmax)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()


def test_model(loader, net, device, L, theta):
    """
    Evaluate and return (y_true, y_scores as probabilities).
    """
    net.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            _, _, logits, _ = net(batch_x, 0.0, L, theta)  # no dropout in eval
            y_true.extend(batch_y.cpu().numpy())
            y_scores.extend(logits.softmax(dim=1).cpu().numpy())
    return np.array(y_true), np.array(y_scores)

# =========================================================
#                  PDF / Plotting helpers
# =========================================================
# NOTE: OutputDir is set *after* parsing args. This function reads it at call-time.
def ensure_pdf(save_path: str, fallback_name: str) -> str:
    """Force .pdf extension and provide a fallback filename if None/empty."""
    if not save_path:
        save_path = os.path.join(OutputDir, fallback_name)
    base, _ = os.path.splitext(save_path)
    return base + ".pdf"

def generate_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """
    Row-normalized confusion matrix saved as PDF (dpi=1200). No title.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.where(row_sums == 0, 1, row_sums), dtype=float)

    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(30, 24))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 16, "weight": "bold"},
        linewidths=1.2, linecolor="black",
        square=True, ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted Class", fontsize=24, fontweight="bold", labelpad=20)
    ax.set_ylabel("True Class", fontsize=24, fontweight="bold", labelpad=20)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(left=0.25, right=0.95, top=0.90, bottom=0.10)

    save_path = ensure_pdf(save_path, "confusion_matrix.pdf")
    plt.savefig(save_path, dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    save_path = ensure_pdf(save_path, "confusion_matrix.pdf")
    generate_confusion_matrix(y_true, y_pred, labels, save_path=save_path)

def plot_roc_curve(fpr, tpr, roc_auc, num_classes, save_path=None):
    """
    Per-class ROC curves from precomputed fpr/tpr dicts. PDF (dpi=1200), no title.
    """
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', linewidth=1.5)
    plt.plot([0, 1], [0, 1], 'k--', label="Random", linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', ncol=2)
    plt.grid(True, alpha=0.3)

    save_path = ensure_pdf(save_path, "roc_curve.pdf")
    plt.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved to {save_path}")

# =========================================================
#            Bernstein filter hyperparams / theta
# =========================================================
def get_k_values(filter_type):
    """
    Dynamic K grid per filter type.
    """
    k_values_map = {
        "all":          [1, 2, 4, 6, 8, 10],
        "low":          [6, 8, 10, 12, 14],
        "high":         [6, 8, 10, 12, 14],
        "impulse_low":  [4, 6, 8, 10],
        "impulse_high": [4, 6, 8, 10],
        "band":         [8, 10, 12, 14, 16],
        "band_reject":  [10, 12, 14, 16, 18],
        "comb":         [12, 14, 16, 18, 20],
    }
    if filter_type not in k_values_map:
        raise ValueError(f"❌ Invalid filter type: {filter_type}")
    return k_values_map[filter_type]

def compute_theta(k, filter_type):
    """
    Build theta (length k+1) based on filter type.
    """
    if k < 1:
        raise ValueError(f"❌ K must be at least 1. Given: {k}")

    if filter_type == "low":
        theta = [1 - i / k for i in range(k + 1)]
    elif filter_type == "high":
        theta = [i / k for i in range(k + 1)]
    elif filter_type == "band":
        theta = [0] * (k + 1)
        theta[k // 2] = 1
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
    print(f"✔️ Theta computed correctly for K={k}, filter_type={filter_type}: {theta}")
    return theta

# =========================================================
#                         CLI
# =========================================================
parser = argparse.ArgumentParser()

# General
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate.")
parser.add_argument('--num_gene', type=int, default=2000, help="Number of genes.")
parser.add_argument('--num_omic', type=int, default=1, help="Number of omics (1=single, 2=multi).")
parser.add_argument('--epochs', type=int, default=60, help="Number of epochs.")
parser.add_argument('--batchsize', type=int, default=64, help="Batch size (used if not sweeping).")
parser.add_argument('--num_folds', type=int, default=5, help="KFold splits.")

parser.add_argument('--database', type=str, default='coexpression',
                    choices=['biogrid', 'string', 'coexpression'],
                    help="Network database.")

# Proper boolean flags
parser.add_argument('--singleton', dest='singleton', action='store_true', help="Include singleton nodes.")
parser.add_argument('--no-singleton', dest='singleton', action='store_false')
parser.set_defaults(singleton=True)

parser.add_argument('--savemodel', dest='savemodel', action='store_true', help="Save best model.")
parser.add_argument('--no-savemodel', dest='savemodel', action='store_false')
parser.set_defaults(savemodel=False)

parser.add_argument('--loaddata', dest='loaddata', action='store_true', help="Load original data.")
parser.add_argument('--no-loaddata', dest='loaddata', action='store_false')
parser.set_defaults(loaddata=True)

parser.add_argument('--num_selected_genes', type=int, default=10, help="Genes per class for SHAP summaries.")

# Single-run filter (ignored when sweeping)
parser.add_argument('--filter_type', type=str, default='impulse_low',
                    choices=['low', 'high', 'band', 'impulse_low', 'impulse_high', 'all', 'band_reject', 'comb'],
                    help="If not sweeping, use this filter type.")

parser.add_argument('--stream_mode', type=str, default='fusion',
                    choices=['fusion', 'gcn_only', 'mlp_only'],
                    help='Ablations: fusion / gcn_only / mlp_only')

# Output / seed / data root
parser.add_argument('--do_shap', action='store_true',
                    help='Run SHAP after CV (requires saved best model + gene list).')
parser.add_argument('--output_dir', type=str, default=r"D:\GS\gsfiltersharply\review2resluts",
                    help='Where to write results/plots.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--data_root', type=str, default=r"D:\GS\gsfiltersharply\data1",
                    help='Root folder containing all input data files.')

args = parser.parse_args()

# =========================================================
#                  Setup / paths / seeding
# =========================================================
_stream_map = {'fusion': 'fusion', 'gcn_only': 'gcn', 'mlp_only': 'mlp'}
stream_mode = _stream_map[args.stream_mode]
print(f"[Ablation] stream_mode = {stream_mode}")

t_start = time.perf_counter()
print(f"Selected Database: {args.database}")

OutputDir = args.output_dir
os.makedirs(OutputDir, exist_ok=True)
try:
    os.chmod(OutputDir, 0o777)
except Exception:
    pass

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- central data paths --------------------
def get_data_paths(database: str, data_root: str):
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

    return (expression_data_path, cnv_data_path, expression_variance_file, shuffle_index_path,
            adjacency_matrix_file, non_null_index_path)

(expression_data_path,
 cnv_data_path,
 expression_variance_file,
 shuffle_index_path,
 adjacency_matrix_file,
 non_null_index_path) = get_data_paths(args.database, args.data_root)

print(f"Using adjacency matrix: {adjacency_matrix_file}")
print(f"Using non-null index file: {non_null_index_path}")
print(f"Data root: {args.data_root}")

# -------------------- leak-free data load --------------------
print("Loading raw data tables (leak-free)…")
if args.num_omic == 1:
    expr_all_df = utilsdata.load_singleomic_data(expression_data_path)
    cnv_all_df  = None
elif args.num_omic == 2:
    expr_all_df, cnv_all_df = utilsdata.load_multiomics_data(expression_data_path, cnv_data_path)
else:
    raise ValueError("--num_omic must be 1 or 2")

labels_all = (expr_all_df['icluster_cluster_assignment'].values - 1).astype(np.int64)
out_dim = int(np.unique(labels_all).size)
print(f"Classes: {out_dim}")

# -------------------- model dimensions --------------------
F_0    = args.num_omic
D_g    = args.num_gene
CL1_F  = 5
FC1_F  = 32
FC2_F  = 0
NN_FC1 = 256
NN_FC2 = 32

# -------------------- output subdirs / CSV --------------------
hyperparam_dir = os.path.join(OutputDir, "hyperparameter_tuning")
bestmodel_dir  = os.path.join(OutputDir, "bestmodel")
for d in (OutputDir, hyperparam_dir, bestmodel_dir):
    os.makedirs(d, exist_ok=True)
try:
    os.chmod(OutputDir, 0o777)
except Exception:
    pass

csv_file_path = os.path.join(OutputDir, "00finalcoexsingle2000.csv")
expected_cols = [
    "filter_type","k","num_genes","batch_size","dropout","lr",
    "accuracy","precision","recall","macro_f1",
    "acc_mean","acc_std","macro_f1_mean","macro_f1_std",
    "macro_precision_mean","macro_precision_std","macro_recall_mean","macro_recall_std",
    "run_time_sec","peak_rss_mb","peak_vram_mb","stream_mode"
]
if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    df_existing = pd.read_csv(csv_file_path)
    for col in ["k","num_genes","batch_size","dropout","lr","accuracy"]:
        if col in df_existing.columns:
            df_existing[col] = pd.to_numeric(df_existing[col], errors="coerce")
else:
    df_existing = pd.DataFrame(columns=expected_cols)

existing_combinations = set(
    tuple(row)
    for row in df_existing[["filter_type","k","num_genes","batch_size","dropout","lr"]].dropna().values
) if not df_existing.empty else set()

# -------------------- hyperparam grids --------------------
filter_types   = ["all","band_reject"]
batch_sizes    = [128]
dropout_values = [0.2]
lr_values      = [0.001]
#filter_types   = ["all","low","high","band","band_reject","impulse_low","impulse_high","comb"]
#batch_sizes    = [32, 64, 128]
#dropout_values = [0.1, 0.2, 0.3]
#lr_values      = [0.001, 0.005, 0.01]

# -------------------- warm best (if any) --------------------
best_result, best_accuracy = None, -1.0
if not df_existing.empty and "accuracy" in df_existing.columns and df_existing["accuracy"].notna().any():
    idx = df_existing["accuracy"].idxmax()
    best_result = df_existing.loc[idx].to_dict()
    best_accuracy = float(best_result["accuracy"])

if best_result:
    best_filter   = best_result["filter_type"]
    best_k        = int(best_result["k"])
    best_batch_sz = int(best_result["batch_size"])
    best_dropout  = float(best_result["dropout"])
    best_lr       = float(best_result["lr"])
    theta_best    = compute_theta(best_k, best_filter)

# -------------------- runtime/mem helpers --------------------
try:
    import psutil
    _psutil_ok = True
    process = psutil.Process(os.getpid())
except Exception:
    _psutil_ok = False
    process = None

per_fold_csv = os.path.join(OutputDir, "per_fold_metrics.csv")

# =========================================================
#                     Training loops
# =========================================================
kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)

for filter_type in filter_types:
    k_values = get_k_values(filter_type)
    for batch_size in batch_sizes:
        for dropout_value in dropout_values:
            for lr in lr_values:
                for k in k_values:
                    combination = (filter_type, k, args.num_gene, batch_size, dropout_value, lr)
                    if combination in existing_combinations:
                        print(f"✅ Skipping already processed combination: {combination}")
                        continue

                    print(f"\n🔹 Processing combination: {combination}")
                    theta = compute_theta(k, filter_type)
                    base_filename = f"{filter_type}_K{k}_Genes{args.num_gene}_BS{batch_size}_DP{dropout_value}_LR{lr}"

                    all_y_true, all_y_pred, all_y_scores = [], [], []
                    fold_accs, fold_mF1s, fold_mPrecs, fold_mRecs = [], [], [], []
                    fold_rows = []

                    comb_start = time.perf_counter()
                    peak_rss_mb = 0.0
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()

                    candidate_gene_list = None

                    # ----------------- Cross-validation -----------------
                    for fold, (train_index, val_index) in enumerate(kf.split(expr_all_df), start=1):
                        fold_t0 = time.perf_counter()
                        print(f"  Fold {fold}...")

                        # --- Leak-free top-N gene selection on TRAIN ONLY ---
                        gene_list, gene_idx = utilsdata.select_top_genes_from_train_fold(
                            expr_all_df, non_null_index_path, train_index, args.num_gene
                        )
                        if candidate_gene_list is None:
                            candidate_gene_list = gene_list[:]
                        G_fold = len(gene_list)

                        # --- Build fold data ---
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

                        # --- Per-fold Laplacian (normalized) ---
                        if stream_mode == "mlp":
                            L_list = [torch.empty(0, device=device)]
                        else:
                            A_max = A_sel.max()
                            if hasattr(A_max, "A"):  # numpy matrix -> scalar
                                A_max = A_max.A.item()
                            A_norm = A_sel if A_max == 0 else (A_sel / A_max)
                            L_sp = graph_laplacian(A_norm, normalized=True)  # CSR
                            L_torch = torch.tensor(L_sp.toarray(), dtype=torch.float32, device=device)
                            L_list = [L_torch]

                        # --- Build loaders ---
                        X_train = torch.tensor(X_all[train_index], dtype=torch.float32)
                        y_train = torch.tensor(y_all[train_index], dtype=torch.long)
                        X_val   = torch.tensor(X_all[val_index], dtype=torch.float32)
                        y_val   = torch.tensor(y_all[val_index], dtype=torch.long)

                        _gen = torch.Generator().manual_seed(1 + fold)
                        _pin = torch.cuda.is_available()
                        train_loader = Data.DataLoader(
                            Data.TensorDataset(X_train, y_train),
                            batch_size=batch_size, shuffle=True, generator=_gen,
                            num_workers=0, pin_memory=_pin
                        )
                        val_loader = Data.DataLoader(
                            Data.TensorDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=_pin
                        )

                        # --- Model ---
                        net_params_fold = [F_0, G_fold, CL1_F, k, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
                        net = Graph_GCN(net_params_fold, stream_mode=stream_mode).to(device)

                        def _weight_init(m):
                            if isinstance(m, (nn.Conv2d, nn.Linear)):
                                nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None:
                                    nn.init.zeros_(m.bias)
                        net.apply(_weight_init)

                        optimizer = optim.Adam(net.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()

                        # --- Train ---
                        for epoch in range(args.epochs):
                            train_model(train_loader, net, optimizer, criterion, device, dropout_value, L_list, theta)

                        # --- Validate ---
                        y_true, y_scores = test_model(val_loader, net, device, L_list, theta)
                        y_pred = np.argmax(y_scores, axis=1)

                        # pooled
                        all_y_true.extend(y_true)
                        all_y_pred.extend(y_pred)
                        all_y_scores.extend(y_scores)

                        # metrics
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

                        # runtime/mem
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        fold_time_sec = time.perf_counter() - fold_t0
                        rss_now_mb = (process.memory_info().rss / (1024**2)) if _psutil_ok else np.nan
                        peak_rss_mb = max(peak_rss_mb, rss_now_mb if np.isfinite(rss_now_mb) else 0.0)
                        vram_now_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

                        fold_rows.append({
                            "filter_type": filter_type,
                            "k": k,
                            "num_genes": args.num_gene,
                            "batch_size": batch_size,
                            "dropout": dropout_value,
                            "lr": lr,
                            "stream_mode": stream_mode,
                            "fold": fold,
                            "accuracy": fold_acc,
                            "macro_f1": fold_mF1,
                            "macro_precision": fold_mPrec,
                            "macro_recall": fold_mRec,
                            "fold_time_sec": fold_time_sec,
                            "rss_mb": rss_now_mb,
                            "vram_peak_mb_sofar": vram_now_mb,
                        })

                        # free
                        del X_train, y_train, X_val, y_val, train_loader, val_loader
                        del X_all, y_all, A_sel
                        if stream_mode != "mlp":
                            del L_sp, L_torch, L_list
                        del net, optimizer, criterion
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # ------ mean±std across folds ------
                    acc_mean   = float(np.mean(fold_accs));   acc_std   = float(np.std(fold_accs,   ddof=1))
                    mF1_mean   = float(np.mean(fold_mF1s));   mF1_std   = float(np.std(fold_mF1s,   ddof=1))
                    mPrec_mean = float(np.mean(fold_mPrecs)); mPrec_std = float(np.std(fold_mPrecs, ddof=1))
                    mRec_mean  = float(np.mean(fold_mRecs));  mRec_std  = float(np.std(fold_mRecs,  ddof=1))
                    print(f"   Fold-wise: Acc {acc_mean:.4f} ± {acc_std:.4f} | mF1 {mF1_mean:.4f} ± {mF1_std:.4f}")

                    # persist per-fold table (append)
                    per_fold_df = pd.DataFrame(fold_rows)
                    if os.path.exists(per_fold_csv) and os.path.getsize(per_fold_csv) > 0:
                        _old = pd.read_csv(per_fold_csv)
                        per_fold_df = pd.concat([_old, per_fold_df], ignore_index=True)
                    per_fold_df.to_csv(per_fold_csv, index=False)

                    # ------ pooled metrics & plots ------
                    all_y_true   = np.array(all_y_true)
                    all_y_pred   = np.array(all_y_pred)
                    all_y_scores = np.array(all_y_scores)

                    avg_accuracy = accuracy_score(all_y_true, all_y_pred)
                    report_dict  = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
                    precision    = report_dict["weighted avg"]["precision"]
                    recall       = report_dict["weighted avg"]["recall"]
                    macro_f1     = report_dict["macro avg"]["f1-score"]

                    # selected classes for ROC
                    class_f1_scores = {int(cls): m['f1-score'] for cls, m in report_dict.items() if cls.isdigit()}
                    sorted_classes = sorted(class_f1_scores, key=class_f1_scores.get, reverse=True)
                    top2, mid3, bot3 = sorted_classes[:2], sorted_classes[2:5], sorted_classes[-3:]
                    _sel = top2 + mid3 + bot3
                    seen = set()
                    selected_classes = [c for c in _sel if not (c in seen or seen.add(c))]

                    base_noext = os.path.join(OutputDir, base_filename)

                    # Selected-classes ROC
                    y_bin = label_binarize(all_y_true, classes=list(range(out_dim)))
                    fpr_sel, tpr_sel, roc_auc_sel = {}, {}, {}
                    for i in selected_classes:
                        if i < y_bin.shape[1] and y_bin[:, i].sum() > 0:
                            fpr_sel[i], tpr_sel[i], _ = roc_curve(y_bin[:, i], all_y_scores[:, i])
                            roc_auc_sel[i] = auc(fpr_sel[i], tpr_sel[i])
                    plot_roc_curve(fpr_sel, tpr_sel, roc_auc_sel, num_classes=out_dim,
                                   save_path=base_noext + "_roc_selected.pdf")

                    # Confusion matrix
                    plot_confusion_matrix(all_y_true, all_y_pred,
                                          labels=list(np.unique(all_y_true)),
                                          save_path=base_noext + "_confusion.pdf")

                    # All-classes ROC
                    fpr_all, tpr_all, roc_auc_all = {}, {}, {}
                    for i in range(out_dim):
                        if i < y_bin.shape[1] and y_bin[:, i].sum() > 0:
                            fpr_all[i], tpr_all[i], _ = roc_curve(y_bin[:, i], all_y_scores[:, i])
                            roc_auc_all[i] = auc(fpr_all[i], tpr_all[i])
                    plot_roc_curve(fpr_all, tpr_all, roc_auc_all, num_classes=out_dim,
                                   save_path=base_noext + "_roc.pdf")

                    # report text
                    report_file_path = base_noext + "_report.txt"
                    with open(report_file_path, "w") as f:
                        f.write(classification_report(all_y_true, all_y_pred, zero_division=0))
                    print(f"📝 Report saved: {report_file_path}")

                    # combo runtime + memory
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    comb_time_sec = time.perf_counter() - comb_start
                    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

                    new_result = {
                        "filter_type": filter_type,
                        "k": k,
                        "num_genes": args.num_gene,
                        "batch_size": batch_size,
                        "dropout": dropout_value,
                        "lr": lr,
                        "stream_mode": stream_mode,
                        "accuracy": avg_accuracy,
                        "precision": precision,
                        "recall": recall,
                        "macro_f1": macro_f1,
                        "acc_mean": acc_mean, "acc_std": acc_std,
                        "macro_f1_mean": mF1_mean, "macro_f1_std": mF1_std,
                        "macro_precision_mean": mPrec_mean, "macro_precision_std": mPrec_std,
                        "macro_recall_mean": mRec_mean, "macro_recall_std": mRec_std,
                        "run_time_sec": comb_time_sec,
                        "peak_rss_mb": peak_rss_mb if _psutil_ok else np.nan,
                        "peak_vram_mb": peak_vram_mb,
                    }

                    df_existing = pd.concat([df_existing, pd.DataFrame([new_result])], ignore_index=True)
                    df_existing.to_csv(csv_file_path, index=False)
                    existing_combinations.add(combination)
                    print(f"✅ CSV updated: {csv_file_path}")
                    print(f"   Pooled Acc={avg_accuracy:.4f} | Macro-F1={macro_f1:.4f} | "
                          f"mean Acc={acc_mean:.4f}±{acc_std:.4f} | time={comb_time_sec:.1f}s | "
                          f"RSS≈{peak_rss_mb:.1f}MB | VRAM≈{peak_vram_mb:.1f}MB")

                    # best selection uses mean accuracy
                    if acc_mean > best_accuracy:
                        best_accuracy = acc_mean
                        best_result = new_result
                        # persist representative gene list
                        try:
                            np.save(os.path.join(OutputDir, "best_gene_list.npy"),
                                    np.array(candidate_gene_list, dtype=object))
                        except Exception as _e:
                            print(f"⚠️ Could not save best_gene_list.npy: {_e}")

# =========================================================
#     BEST RESULT SUMMARY & ARTIFACT COPY (PDFs only)
# =========================================================
import shutil

best_model_txt = os.path.join(OutputDir, "bestmodel.txt")

if best_result:
    with open(best_model_txt, "w") as f:
        for key, value in best_result.items():
            f.write(f"{key}: {value}\n")
    print(f"✅ Best model configuration saved to {best_model_txt}")
else:
    print("❌ No best result found.")

if best_result:
    print("\n✅ Best Overall Configuration:")
    for key, value in best_result.items():
        print(f"{key}: {value}")
else:
    print("\n❌ No best configuration found.")

if best_result:
    
    base_filename = (
    f"{args.database}_{stream_mode}_{filter_type}"
    f"_K{k}_Genes{args.num_gene}_BS{batch_size}_DP{dropout_value}_LR{lr}"
    )

    for suffix in ["_roc_selected.pdf", "_roc.pdf", "_confusion.pdf", "_report.txt"]:
        src = os.path.join(OutputDir, base_filename + suffix)
        dst = os.path.join(bestmodel_dir, base_filename + suffix)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"📂 Copied {os.path.basename(src)} to bestmodel folder.")

    for optional_name in ["best_model.pt", "best_gene_list.npy", "bestmodel.txt"]:
        src = os.path.join(bestmodel_dir if optional_name == "bestmodel.txt" else OutputDir, optional_name)
        if os.path.exists(src):
            dst = os.path.join(bestmodel_dir, optional_name)
            if src != dst:
                shutil.copy2(src, dst)
            print(f"📂 Copied {optional_name} to bestmodel folder.")

# =========================================================
#                   Visualization from CSV
# =========================================================
results_csv = csv_file_path if os.path.exists(csv_file_path) else os.path.join(OutputDir, "00finalcoexsingle2000.csv")

if os.path.exists(results_csv) and os.path.getsize(results_csv) > 0:
    df_results = pd.read_csv(results_csv)
    df_results.columns = df_results.columns.str.strip()
else:
    print(f"⚠️ No results CSV found at {results_csv}. Creating an empty DataFrame.")
    df_results = pd.DataFrame(columns=["filter_type","k","num_genes","batch_size","dropout","lr","accuracy","precision","recall","macro_f1"])

needed = {"filter_type","k","num_genes","batch_size","dropout","lr","accuracy"}
missing = needed - set(df_results.columns)
if missing:
    print(f"⚠️ Missing columns in results CSV: {sorted(list(missing))}. Plots may be limited.")

if df_results.empty or "filter_type" not in df_results.columns:
    print("❌ No data available for plotting. Skipping visualization.")
else:
    for col in ["k","batch_size","dropout","lr","accuracy"]:
        if col in df_results.columns:
            df_results[col] = pd.to_numeric(df_results[col], errors="coerce")

    best_filter = best_result["filter_type"] if best_result and "filter_type" in best_result else None

    def _ensure_pdf_from_noext(path_no_ext: str) -> str:
        base, _ = os.path.splitext(path_no_ext)
        return base + ".pdf"

    def save_pdf_and_copy(fig_path_no_ext, is_best):
        pdf_path = _ensure_pdf_from_noext(fig_path_no_ext)
        plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
        plt.close()
        print(f"✅ Plot saved: {pdf_path}")
        if is_best:
            best_pdf = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
            shutil.copy2(pdf_path, best_pdf)
            print(f"✅ Best plot copied to bestmodel: {best_pdf}")

    # ----- 2D: Accuracy vs K (per filter) -----
    for ft in df_results["filter_type"].dropna().unique():
        df_ft = df_results[df_results["filter_type"] == ft]
        if "k" not in df_ft.columns or "accuracy" not in df_ft.columns or df_ft.empty:
            continue
        avg_acc_per_k = df_ft.groupby("k", as_index=True)["accuracy"].mean().sort_index()

        plt.figure(figsize=(8, 6))
        plt.plot(avg_acc_per_k.index, avg_acc_per_k.values, marker="o", linewidth=2)
        plt.xlabel("K (Bernstein Polynomial Order)", fontsize=14)
        plt.ylabel("Validation Accuracy", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        save_pdf_and_copy(os.path.join(hyperparam_dir, f"{ft}_k_vs_accuracy"), is_best=(ft == best_filter))

    # ----- 2D: Combined Accuracy vs K (all filters) -----
    plt.figure(figsize=(10, 8))
    y_min, y_max = 1.0, 0.0
    for ft in df_results["filter_type"].dropna().unique():
        df_ft = df_results[df_results["filter_type"] == ft]
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
    save_pdf_and_copy(os.path.join(hyperparam_dir, "combined_k_vs_accuracy"), is_best=False)

    # ----- 3D helpers -----
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
    for ft in df_results["filter_type"].dropna().unique():
        df_ft = df_results[df_results["filter_type"] == ft]
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

        out_no_ext = os.path.join(hyperparam_dir, f"{ft}_accuracy_vs_k_bs_3d")
        pdf_path = _ensure_pdf_from_noext(out_no_ext)
        plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
        plt.close()
        print(f"✅ 3D plot (K vs Batch Size) saved for filter {ft}: {pdf_path}")
        if ft == best_filter:
            dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
            shutil.copy2(pdf_path, dst)
            print(f"✅ Best 3D plot (K vs Batch Size) copied to bestmodel: {dst}")

    # ----- 3D: Accuracy vs K & Dropout -----
    for ft in df_results["filter_type"].dropna().unique():
        df_ft = df_results[df_results["filter_type"] == ft]
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

        out_no_ext = os.path.join(hyperparam_dir, f"{ft}_accuracy_vs_k_dropout_3d")
        pdf_path = _ensure_pdf_from_noext(out_no_ext)
        plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
        plt.close()
        print(f"✅ 3D plot (K vs Dropout) saved for filter {ft}: {pdf_path}")
        if ft == best_filter:
            dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
            shutil.copy2(pdf_path, dst)
            print(f"✅ Best 3D plot (K vs Dropout) copied to bestmodel: {dst}")

    # ----- 3D: Accuracy vs K & Learning Rate -----
    for ft in df_results["filter_type"].dropna().unique():
        df_ft = df_results[df_results["filter_type"] == ft]
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

        out_no_ext = os.path.join(hyperparam_dir, f"{ft}_accuracy_vs_k_lr_3d")
        pdf_path = _ensure_pdf_from_noext(out_no_ext)
        plt.savefig(pdf_path, dpi=1200, bbox_inches="tight")
        plt.close()
        print(f"✅ 3D plot (K vs Learning Rate) saved for filter {ft}: {pdf_path}")
        if ft == best_filter:
            dst = os.path.join(bestmodel_dir, "best_" + os.path.basename(pdf_path))
            shutil.copy2(pdf_path, dst)
            print(f"✅ Best 3D plot (K vs Learning Rate) copied to bestmodel: {dst}")

import torch.nn as nn
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch.nn as nn
import os
import numpy as np
import torch
import shap
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
# ------------------------------
# Extract Best Hyperparameters
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#✅ Ensure adjacency matrix (`adj_final`) is correctly stored and updated
if "adj_final" not in globals():
    print("⚠️ Warning: `adj_final` is not defined. Using the latest adjacency matrix.")
    adj_final = adj.copy()  # Capture the latest version

# ✅ Extract best parameters after reading best_result
best_filter    = best_result["filter_type"]
best_k         = best_result["k"]
best_dropout   = best_result["dropout"]
theta_best     = compute_theta(best_k, best_filter)
print(f"\n✅ Loaded best model: {best_filter}, K={best_k}, Accuracy={best_result['accuracy']:.2f}")
import os
import numpy as np
import pandas as pd
import torch
import shap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict

# ---------------------- CONFIG ----------------------
NUM_CLASSES = 28
NUM_TOP_GENES_PER_CLASS = 10
NUM_SHAP_SAMPLES = 1000
#NUM_SHAP_SAMPLES = 50
NUM_BACKGROUND_SAMPLES = 30
import os

# Step 1: Define base output directory
OutputDir = r"D:\combined results\finalcoexsingle2000"

# Step 2: Create subfolder "shapresults" inside OutputDir
SHAP_OUTPUT_DIR = os.path.join(OutputDir, "shapresults")
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# Step 3: Print to confirm
print(f"✅ SHAP outputs will be saved to: {SHAP_OUTPUT_DIR}")

# ---------------------- LOAD SHAP INPUT ----------------------
train_data_np = np.asarray(train_data_all).astype(np.float32)
train_labels_np = np.asarray(labels).astype(np.int64)

# Flatten if needed (B, V, F) → (B, V*F)
if train_data_np.ndim == 3:
    train_data_flat = train_data_np.reshape(train_data_np.shape[0], -1)
else:
    train_data_flat = train_data_np.copy()

# ---------------------- SELECT 1000 SAMPLES ----------------------
print("🔍 Running PCA + KMeans for SHAP sample selection...")
pca = PCA(n_components=50)
X_pca = pca.fit_transform(train_data_flat)

# Step 1: Select 1000 diverse samples using PCA + KMeans
kmeans = KMeans(n_clusters=1000, random_state=42).fit(X_pca)
initial_indices = np.array([np.where(kmeans.labels_ == i)[0][0] for i in range(1000)])
print(f"✅ PCA+KMeans selected {len(initial_indices)} diverse samples.")

# Step 2: Ensure ≥10 samples per class if possible
required_per_class = 10
final_indices = []
label_counts = defaultdict(int)
used_indices = set()

for idx in initial_indices:
    cls = train_labels_np[idx]
    if label_counts[cls] < required_per_class:
        final_indices.append(idx)
        label_counts[cls] += 1
        used_indices.add(idx)

# Step 3: Fill remaining to reach 1000
for idx in initial_indices:
    if len(final_indices) >= 1000:
        break
    if idx not in used_indices:
        final_indices.append(idx)
        used_indices.add(idx)

final_indices = np.array(sorted(set(final_indices)))
np.save(os.path.join(SHAP_OUTPUT_DIR, "shap_indices.npy"), final_indices)

print(f"✅ Final SHAP sample count: {len(final_indices)} (covering {len(label_counts)} classes)")

# ✅ Optional: Print per-class sample summary
print("\n📊 Per-class SHAP sample counts:")
for c in range(NUM_CLASSES):
    count = sum(train_labels_np[i] == c for i in final_indices)
    print(f"Class {c:2d}: {count:2d} samples")

# ---------------------- STATIC WRAPPER ----------------------
# ✅ Replace DeepExplainer with KernelExplainer for full model interpretability

import shap
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- BACKGROUND ----------------------
bg_indices = np.random.choice(train_data_flat.shape[0], NUM_BACKGROUND_SAMPLES, replace=False)
background_np = train_data_flat[bg_indices]

# ---------------------- PREDICT FUNCTION ----------------------
def gcn_predict(X_numpy):
    with torch.no_grad():
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).view(-1, D_g, F_0).to(device)
        logits = best_net(X_tensor, best_dropout, L, theta_best)[2]
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

# ---------------------- EXPLAINER ----------------------
explainer = shap.KernelExplainer(
    model=gcn_predict,
    data=background_np,
    l1_reg="num_features(10)"  # ✅ safe alternative to LassoLarsIC
)
# ✅ Clear label file before SHAP loop (important!)
import glob, os
for f in glob.glob(os.path.join(SHAP_OUTPUT_DIR, "shap_*.npy")):
    os.remove(f)
from tqdm import tqdm  # ✅ Make sure this is imported at the top
# 🧹 Clear labels file at the beginning of SHAP computation
open(os.path.join(SHAP_OUTPUT_DIR, "shap_labels.txt"), "w").close()
print("🔁 Computing SHAP values for selected samples...")
shap_matrix = []
shap_labels = []
failed_samples = []

for idx, sample_idx in enumerate(tqdm(final_indices[:NUM_SHAP_SAMPLES], desc="🔬 SHAP progress")):
    shap_path = os.path.join(SHAP_OUTPUT_DIR, f"shap_{sample_idx:04d}.npy")

    if os.path.exists(shap_path):
        print(f"⏩ SHAP already exists: {shap_path} — skipping.")
        continue

    sample = train_data_flat[sample_idx:sample_idx+1]
    print(f"🔹 Sample index {sample_idx} — shape: {sample.shape}")

    try:
      
        shap_values = explainer.shap_values(sample, nsamples="auto")

        # ✅ Robust SHAP output handler
        if isinstance(shap_values, list):
            shap_sample = np.abs(np.mean([s[0] for s in shap_values], axis=0))
        elif shap_values.ndim == 3 and shap_values.shape[0] == 1:
            shap_sample = np.abs(shap_values[0].mean(axis=1))
        elif shap_values.ndim == 2:
            shap_sample = np.abs(shap_values.mean(axis=1))
        elif shap_values.ndim == 1:
            shap_sample = np.abs(shap_values)
        else:
            raise ValueError(f"❌ Unexpected SHAP format: {shap_values.shape}")

        print(f"✅ Final SHAP shape (should be 1D): {shap_sample.shape}")
        np.save(shap_path, shap_sample)
        print(f"💾 SHAP saved to {shap_path}")

        with open(os.path.join(SHAP_OUTPUT_DIR, "shap_labels.txt"), "a") as f:
            f.write(f"{sample_idx},{train_labels_np[sample_idx]}\n")

    except Exception as e:
        print(f"❌ SHAP failed for sample {sample_idx}: {e}")
        failed_samples.append(sample_idx)
        continue

# ---------------------- LOAD & VALIDATE SHAP FILES ----------------------
import glob

print("📥 Loading SHAP .npy files from:", SHAP_OUTPUT_DIR)
shap_files = sorted(glob.glob(os.path.join(SHAP_OUTPUT_DIR, "shap_*.npy")))
shap_matrix = []
valid_files = []
failed_files = []

for f in shap_files:
    try:
        val = np.load(f)
        # Accept both (F,) and (1, F)
        if val.ndim == 2 and val.shape[0] == 1:
            val = val[0]
        if val.ndim != 1:
            raise ValueError(f"❌ Invalid SHAP shape: {val.shape}")
        shap_matrix.append(val)
        valid_files.append(f)
    except Exception as e:
        print(f"❌ Skipping {f}: {e}")
        failed_files.append(f)

shap_matrix = np.vstack(shap_matrix)
print(f"✅ Loaded SHAP matrix with shape: {shap_matrix.shape}")
print(f"🧹 Skipped {len(failed_files)} malformed files.")


# ---------------------- Step 3: Multi-Omics SHAP Aggregation (Robust - Accepts 1 or 2 Features) ----------------------
if args.num_omic == 2:
    base_gene_names = sorted(set(name.split("_")[0] for name in gene_names))
    gene_to_indices = defaultdict(list)
    for i, name in enumerate(gene_names):
        gene_to_indices[name.split("_")[0]].append(i)

    valid_genes = []
    shap_matrix_agg = np.zeros((shap_matrix.shape[0], len(base_gene_names)))

    for i, gene in enumerate(base_gene_names):
        idxs = gene_to_indices[gene]

        if len(idxs) == 1:
            # Only one modality (expression or CNV)
            shap_matrix_agg[:, len(valid_genes)] = shap_matrix[:, idxs[0]]
        elif len(idxs) == 2:
            # Two modalities — average their SHAP
            shap_matrix_agg[:, len(valid_genes)] = shap_matrix[:, idxs].mean(axis=1)
        else:
            print(f"⚠️ Skipping {gene} — has {len(idxs)} features (expected 1 or 2).")
            continue

        valid_genes.append(gene)

    # Trim extra columns
    shap_matrix_agg = shap_matrix_agg[:, :len(valid_genes)]
    final_gene_names = valid_genes

else:
    # Single-omics case
    shap_matrix_agg = shap_matrix
    final_gene_names = gene_names

# ✅ Save aggregated SHAP matrix
np.save(os.path.join(SHAP_OUTPUT_DIR, "shap_matrix.npy"), shap_matrix_agg)

# ---------------------- Step 4: Save to Excel (Top 5 Genes Per Class) ----------------------
# ---------------------- Step 4: Save to Excel (Top 5 Genes Per Class) ----------------------
print("📊 Extracting top 5 genes per class...")

# ✅ Load labels
label_file_path = os.path.join(SHAP_OUTPUT_DIR, "shap_labels.txt")
shap_labels = []
with open(label_file_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            shap_labels.append(int(parts[1]))

# ✅ Truncate to match SHAP matrix rows (fixes error!)
shap_labels = shap_labels[:shap_matrix_agg.shape[0]]

# ✅ Now safe to use shap_labels
shap_df = pd.DataFrame(shap_matrix_agg, columns=final_gene_names)
shap_df["label"] = shap_labels


top_genes_per_class = {}

excel_path = os.path.join(SHAP_OUTPUT_DIR, "top_genes_per_class.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    for cls in range(NUM_CLASSES):
        df_cls = shap_df[shap_df["label"] == cls].drop(columns="label")

        if df_cls.shape[0] == 0:
            print(f"⚠️ Skipping Class {cls} — no valid SHAP samples.")
            continue

        df_cls = df_cls.dropna(axis=1, how='all')  # Drop genes with all NaNs
        mean_shap = df_cls.mean(axis=0).fillna(0)  # Fill any remaining NaNs with 0

        top_genes = mean_shap.sort_values(ascending=False).head(NUM_TOP_GENES_PER_CLASS)
        top_genes_per_class[cls] = list(top_genes.index)

        top_genes_df = pd.DataFrame({
            "Top Genes": top_genes.index,
            "Mean SHAP": top_genes.values
        })

        top_genes_df.to_excel(writer, sheet_name=f"Class_{cls}", index=False)

print(f"✅ Top genes per class saved to {excel_path}")

# 🔽 CONTINUATION: Add this after saving `top_genes_per_class.xlsx`

print("📊 Aggregating global top genes...")

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
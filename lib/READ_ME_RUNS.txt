How to use run_db_gcn_all.bat

1) Put this BAT file next to your Python script
   (default expected name: sigleomicsfiltersharply_results_coex_sing_1000.py).
   If your filename/path is different, open the BAT and change SCRIPT=...

2) Edit DATA_ROOT and OUTPUT_DIR at the top of the BAT to match your machine.
   - DATA_ROOT must contain the TSV/NPZ files your script expects.
   - OUTPUT_DIR is where results and logs will be saved.

3) Double-click the BAT to run, or execute from a Command Prompt.

The BAT will run:
  - Single-CV for all streams (fusion, gcn_only, mlp_only) and filters (all, band_reject)
  - Nested-CV for the same grid with --use_single_splits_in_ncv

Folds are stable across runs: --seed 47 --num_folds 5 --split_scope global.
Gene selection remains TRAIN-FOLD only (as in your script).

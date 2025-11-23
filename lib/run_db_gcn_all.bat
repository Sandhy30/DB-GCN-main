@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =================== USER SETTINGS ===================
REM Path to Python interpreter (leave as "python" if it's on PATH)
set "PY=python"

REM Your script filename (sits next to this .bat or give full path)
set "SCRIPT=sigleomicsfiltersharply_results_coex_sing_1000.py"

REM Data and output roots
set "DATA_ROOT=D:\GS\gsfiltersharply\data1"
set "OUTPUT_DIR=D:\GS\gsfiltersharply\review2resluts"

REM Fixed experiment settings
set "DATABASE=biogrid"
set "NUM_GENE=2000"
set "NUM_OMIC=2"
set "BATCH=128"
set "LR=0.01"
set "DROPOUT=0.2"
set "SEED=1"
set "KFOLDS=5"

REM Ablations to run
set "STREAMS=fusion gcn_only mlp_only"
set "FILTERS=all band_reject"

REM Nested-CV settings
set "OUTER=5"
set "INNER=3"
set "SCORE=macro_f1"

REM ================== SANITY CHECKS =====================
if not exist "%SCRIPT%" (
  echo [ERROR] Script not found: "%SCRIPT%"
  echo Edit SCRIPT=... at the top of this BAT or place it next to the BAT.
  exit /b 1
)
if not exist "%DATA_ROOT%" (
  echo [ERROR] DATA_ROOT not found: "%DATA_ROOT%"
  echo Edit DATA_ROOT=... at the top of this BAT.
  exit /b 1
)

REM Create logs dir
set "LOG_DIR=%OUTPUT_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul

REM Common flags (no sweeps; fixed hyperparams)
set "COMMON=--database %DATABASE% --num_omic %NUM_OMIC% --num_gene %NUM_GENE% --batchsize %BATCH% --lr %LR% --dropouts %DROPOUT% --seed %SEED% --num_folds %KFOLDS% --split_scope global --ncv_log_runtime --data_root %DATA_ROOT% --output_dir %OUTPUT_DIR%"

echo.
echo ===================== RUN START =====================
echo Output: %OUTPUT_DIR%
echo Logs  : %LOG_DIR%
echo =====================================================
echo.

REM -------------------- SINGLE-CV -----------------------
for %%S in (%STREAMS%) do (
  for %%F in (%FILTERS%) do (
    echo [SINGLE] stream=%%S filter=%%F
    call "%PY%" "%SCRIPT%" --stream_mode %%S --filter_type %%F %COMMON% 1>>"%LOG_DIR%\single_%%S_%%F.log" 2>&1
    if errorlevel 1 (
      echo   [FAIL] single: stream=%%S filter=%%F  ^(see log: %LOG_DIR%\single_%%S_%%F.log^)
    ) else (
      echo   [OK]   single: stream=%%S filter=%%F
    )
    echo.
  )
)

REM -------------------- NESTED-CV -----------------------
for %%S in (%STREAMS%) do (
  for %%F in (%FILTERS%) do (
    echo [NESTED] stream=%%S filter=%%F
    call "%PY%" "%SCRIPT%" --stream_mode %%S --filter_type %%F %COMMON% --ncv --outer_folds %OUTER% --inner_folds %INNER% --score %SCORE% --use_single_splits_in_ncv 1>>"%LOG_DIR%\nested_%%S_%%F.log" 2>&1
    if errorlevel 1 (
      echo   [FAIL] nested: stream=%%S filter=%%F  ^(see log: %LOG_DIR%\nested_%%S_%%F.log^)
    ) else (
      echo   [OK]   nested: stream=%%S filter=%%F
    )
    echo.
  )
)

echo.
echo ====================== DONE =========================
echo Look for results.xlsx / folds.xlsx in %OUTPUT_DIR%
echo (Each run also mirrored into %OUTPUT_DIR%\runs\*)
echo =====================================================
exit /b 0

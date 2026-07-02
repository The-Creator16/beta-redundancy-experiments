#!/usr/bin/env bash
# =============================================================================
# One-command reproduction of the full paper pipeline (no `make` required).
#
#   ./reproduce.sh          run all 7 experiments (paper protocol) + all figures
#   ./reproduce.sh smoke    fast pipeline check on tiny trials/widths
#
# Fixed seeds are baked into each experiment script, so the CSVs are
# reproduced exactly. Figures are written as PNG + PDF into
# results/experiment_XX/paper_figures/.
# =============================================================================
set -euo pipefail

PY="${PY:-python3}"
EXP="experiments"
RES="results"
WIDTHS="8 12 16 20 24"
TRIALS="${TRIALS:-10000}"
MODE="${1:-full}"

if [[ "$MODE" == "smoke" ]]; then
    echo "[smoke] fast pipeline check (not the paper protocol)"
    OUT="$RES/_smoke"
    $PY $EXP/experiment_01_clean_representation.py        --trials 500 --widths 8 12 --output-dir "$OUT/experiment_01"
    $PY $EXP/experiment_02_arithmetic_canonicalization.py --trials 500 --widths 8 12 --output-dir "$OUT/experiment_02"
    $PY $EXP/experiment_03_single_digit_corruption.py     --trials 500 --widths 8 12 --output-dir "$OUT/experiment_03"
    $PY $EXP/experiment_04_local_burst_corruption.py      --trials 500 --widths 8 12 --output-dir "$OUT/experiment_04"
    $PY $EXP/experiment_05_exhaustive.py --widths 8 --burst-widths 2 3 --output-dir "$OUT/experiment_05"
    $PY $EXP/experiment_06_algebraic_burst.py --trials 500 --widths 8 12 --output-dir "$OUT/experiment_06"
    $PY $EXP/experiment_07_guard_digits.py --trials 500 --widths 16 --guards 0 2 4 --output-dir "$OUT/experiment_07"
    echo "[smoke] OK -> $OUT"
    exit 0
fi

echo "[1/7] Experiment 1: clean representation"
$PY $EXP/experiment_01_clean_representation.py        --trials "$TRIALS" --widths $WIDTHS --output-dir "$RES/experiment_01"

echo "[2/7] Experiment 2: arithmetic canonicalization"
$PY $EXP/experiment_02_arithmetic_canonicalization.py --trials "$TRIALS" --widths $WIDTHS --output-dir "$RES/experiment_02"

echo "[3/7] Experiment 3: single-digit corruption"
$PY $EXP/experiment_03_single_digit_corruption.py     --trials "$TRIALS" --widths $WIDTHS --output-dir "$RES/experiment_03"

echo "[4/7] Experiment 4: local burst corruption"
$PY $EXP/experiment_04_local_burst_corruption.py      --trials "$TRIALS" --widths $WIDTHS --burst-widths 2 3 4 5 --output-dir "$RES/experiment_04"

echo "[5/7] Experiment 5: exhaustive verification (long-running)"
$PY $EXP/experiment_05_exhaustive.py --widths 4 6 8 10 12 14 16 --burst-widths 2 3 4 5 --output-dir "$RES/experiment_05"

echo "[6/7] Experiment 6: algebraic burst injection"
$PY $EXP/experiment_06_algebraic_burst.py --trials "$TRIALS" --widths $WIDTHS --output-dir "$RES/experiment_06"

echo "[7/7] Experiment 7: guard-digit truncation (single process -> matches Table 7)"
$PY $EXP/experiment_07_guard_digits.py --trials "$TRIALS" --widths $WIDTHS --guards 0 2 4 8 12 16 --output-dir "$RES/experiment_07"

echo "[figures] generating paper figures (PNG + PDF)"
$PY $EXP/plot_experiment_01.py
$PY $EXP/plot_experiment_02.py
$PY $EXP/plot_experiment_03.py
$PY $EXP/plot_experiment_04.py
$PY $EXP/plot_experiment_07.py

echo "[done] full pipeline complete -> $RES/"

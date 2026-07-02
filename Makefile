# =============================================================================
# Reproduction pipeline for the beta-expansion redundancy benchmarks.
#
#   make install     install runtime + plotting + test dependencies
#   make test        run the unit test suite
#   make smoke       fast end-to-end check (tiny trials/widths) -> results/_smoke
#   make experiments run all 7 data experiments with the paper protocol
#   make figures     regenerate every paper figure from the committed CSVs
#   make all         experiments + figures (full paper reproduction)
#   make clean       remove smoke output and Python caches
#
# Approximate wall-clock on a modern laptop (single core):
#   exp 01  ~10s     exp 05  ~30-60 min  (exhaustive, W up to 16)
#   exp 02  ~45s     exp 06  ~20s
#   exp 03  ~45s     exp 07  ~5 min      (full width sweep, single process)
#   exp 04  ~55s
# =============================================================================

PY      ?= python3
EXP      = experiments
RES      = results
WIDTHS   = 8 12 16 20 24
TRIALS   = 10000

.PHONY: install test smoke experiments figures all clean \
        exp01 exp02 exp03 exp04 exp05 exp06 exp07 \
        fig01 fig02 fig03 fig04 fig07

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

# ---------------------------------------------------------------------------
# Fast end-to-end smoke test (NOT the paper protocol; just checks the pipeline)
# ---------------------------------------------------------------------------
smoke:
	$(PY) $(EXP)/experiment_01_clean_representation.py       --trials 500 --widths 8 12 --output-dir $(RES)/_smoke/experiment_01
	$(PY) $(EXP)/experiment_02_arithmetic_canonicalization.py --trials 500 --widths 8 12 --output-dir $(RES)/_smoke/experiment_02
	$(PY) $(EXP)/experiment_03_single_digit_corruption.py     --trials 500 --widths 8 12 --output-dir $(RES)/_smoke/experiment_03
	$(PY) $(EXP)/experiment_04_local_burst_corruption.py      --trials 500 --widths 8 12 --output-dir $(RES)/_smoke/experiment_04
	$(PY) $(EXP)/experiment_05_exhaustive.py --widths 8 --burst-widths 2 3 --output-dir $(RES)/_smoke/experiment_05
	$(PY) $(EXP)/experiment_06_algebraic_burst.py --trials 500 --widths 8 12 --output-dir $(RES)/_smoke/experiment_06
	$(PY) $(EXP)/experiment_07_guard_digits.py --trials 500 --widths 16 --guards 0 2 4 --output-dir $(RES)/_smoke/experiment_07
	@echo "[smoke] pipeline OK -> $(RES)/_smoke"

# ---------------------------------------------------------------------------
# Full experiments (paper protocol, fixed seeds baked into each script)
# ---------------------------------------------------------------------------
experiments: exp01 exp02 exp03 exp04 exp05 exp06 exp07

exp01:
	$(PY) $(EXP)/experiment_01_clean_representation.py       --trials $(TRIALS) --widths $(WIDTHS) --output-dir $(RES)/experiment_01

exp02:
	$(PY) $(EXP)/experiment_02_arithmetic_canonicalization.py --trials $(TRIALS) --widths $(WIDTHS) --output-dir $(RES)/experiment_02

exp03:
	$(PY) $(EXP)/experiment_03_single_digit_corruption.py     --trials $(TRIALS) --widths $(WIDTHS) --output-dir $(RES)/experiment_03

exp04:
	$(PY) $(EXP)/experiment_04_local_burst_corruption.py      --trials $(TRIALS) --widths $(WIDTHS) --burst-widths 2 3 4 5 --output-dir $(RES)/experiment_04

# Exhaustive enumeration (no RNG): all even W <= 16. Long-running.
exp05:
	$(PY) $(EXP)/experiment_05_exhaustive.py --widths 4 6 8 10 12 14 16 --burst-widths 2 3 4 5 --output-dir $(RES)/experiment_05

exp06:
	$(PY) $(EXP)/experiment_06_algebraic_burst.py --trials $(TRIALS) --widths $(WIDTHS) --output-dir $(RES)/experiment_06

# Full width sweep in a single process (shared RNG stream) -> matches Table 7.
exp07:
	$(PY) $(EXP)/experiment_07_guard_digits.py --trials $(TRIALS) --widths $(WIDTHS) --guards 0 2 4 8 12 16 --output-dir $(RES)/experiment_07

# ---------------------------------------------------------------------------
# Figures (read the committed CSVs, write PNG + PDF into paper_figures/)
# ---------------------------------------------------------------------------
figures: fig01 fig02 fig03 fig04 fig07

fig01:
	$(PY) $(EXP)/plot_experiment_01.py
fig02:
	$(PY) $(EXP)/plot_experiment_02.py
fig03:
	$(PY) $(EXP)/plot_experiment_03.py
fig04:
	$(PY) $(EXP)/plot_experiment_04.py
fig07:
	$(PY) $(EXP)/plot_experiment_07.py

all: experiments figures

clean:
	rm -rf $(RES)/_smoke
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +

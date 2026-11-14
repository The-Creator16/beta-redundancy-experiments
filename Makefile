# All public rendering targets use archived CSVs and never rerun simulations.
.DEFAULT_GOAL := help
PY ?= python3
EXP = experiments
RES = results/data
WIDTHS = 8 12 16 20 24
TRIALS = 10000

.PHONY: help install all journal supplementary figures journal-figures supplement test experiments exp01 exp02 exp03 exp04 exp05 exp06 exp07 clean

help:
	@echo "make all            -> results/all (all figures + both document table sets)"
	@echo "make journal        -> results/journal (journal figures + tables 1-5)"
	@echo "make supplementary  -> results/supplementary (figures S1-S9 + tables S1-S9)"
	@echo "make test           -> unit + release mapping tests"
	@echo "make experiments    -> rerun all simulations; expensive; overwrites results/data/"
	@echo "make clean          -> delete generated result bundles only"

install:
	$(PY) -m pip install -r requirements.txt

all:
	$(PY) experiments/build_release.py all

journal:
	$(PY) experiments/build_release.py journal

supplementary:
	$(PY) experiments/build_release.py supplementary

figures: all
journal-figures: journal
supplement: supplementary

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf results/all results/journal results/supplementary

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


.PHONY: audits
audits:
	$(PY) experiments/audit/normalized_errors.py --repo .
	$(PY) experiments/audit/validate_audits.py --repo .

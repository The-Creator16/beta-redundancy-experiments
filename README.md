# Intrinsic Redundancy and Local Robustness in Finite β-Expansion Systems

Reproducible experiment code, data, and figures for the paper
**“Intrinsic Redundancy and Local Robustness in Finite β-Expansion Systems”**
by Adilbek Taizhanov and Miras Seilkhan.

The study separates *ambient semantic redundancy* from *canonical codebook
admissibility* in non-integer base (β-expansion) numeration systems, and
measures — in a strict finite-window (bounded register) model — what that
intrinsic redundancy actually provides for digital fault handling: structural
fault **detection**, value-preserving **canonicalization**, and the narrow
conditions under which a fault is **semantically survivable**.

Five representation systems are compared on equal-width registers:

| Key | System | Base | Alphabet | Canonical constraint |
|-----|--------|------|----------|----------------------|
| `binary`      | Standard binary            | 2                       | `{0,1}`    | none (every word admissible) |
| `signed2_naf` | Signed-digit radix-2 (NAF) | 2                       | `{-1,0,1}` | Non-Adjacent Form |
| `phi`         | Golden ratio (Φ)           | φ = (1+√5)/2            | `{0,1}`    | forbid `11` |
| `tribonacci`  | Tribonacci (T₃)            | root of x³=x²+x+1       | `{0,1}`    | forbid `111` |
| `tetranacci`  | Tetranacci (T₄)           | root of x⁴=x³+x²+x+1    | `{0,1}`    | forbid `1111` |

---

## Table of contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The seven experiments](#the-seven-experiments)
- [Reproducing the paper](#reproducing-the-paper)
- [Figures](#figures)
- [Package API](#package-api)
- [Reproducibility notes](#reproducibility-notes)
- [Testing](#testing)
- [Citing](#citing)
- [License](#license)

---

## Repository layout

```
.
├── README.md               This file.
├── FIGURES.md              Figure → paper-element mapping.
├── LICENSE                 MIT.
├── CITATION.cff            Machine-readable citation metadata.
├── Makefile                make smoke / experiments / figures / all.
├── reproduce.sh            Same pipeline without `make`.
├── pyproject.toml          Package metadata + optional extras.
├── requirements.txt        Loose runtime + plotting + test deps.
├── requirements-lock.txt   Exact pinned toolchain for the committed outputs.
│
├── src/redundancy_exp/     The reusable engine (see "Package API").
│   ├── window.py           Finite digit window I_{L,R}, overflow/truncation.
│   ├── systems.py          The five representation systems + value maps.
│   ├── sampling.py         Exact uniform sampling of constrained codebooks.
│   ├── normalization.py    Local rewrite rules (011→100, etc.).
│   ├── arithmetic.py       Digit-wise addition + bounded-window canonicalization.
│   ├── corruption.py       Single-digit fault model.
│   ├── burst.py            Local burst fault model.
│   ├── algebraic.py        Kernel-aligned (value-preserving) burst injection.
│   └── metrics.py          Benchmark drivers and metric aggregation.
│
├── experiments/
│   ├── experiment_01_clean_representation.py        Exp 1 (Table 1, 8)
│   ├── experiment_02_arithmetic_canonicalization.py Exp 2 (Table 2; Fig 6, 7)
│   ├── experiment_03_single_digit_corruption.py     Exp 3 (Table 3; Fig 8, 9)
│   ├── experiment_04_local_burst_corruption.py      Exp 4 (Table 4; Fig 10)
│   ├── experiment_05_exhaustive.py                  Exp 5 (Table 5, 6)
│   ├── experiment_06_algebraic_burst.py             Exp 6 (kernel survival)
│   ├── experiment_07_guard_digits.py               Exp 7 (Table 7; Fig 11)
│   └── plot_experiment_{01,02,03,04,07}.py         Figure generators.
│
├── tests/                  pytest suite for the engine.
│
└── results/                Committed outputs (CSV + Markdown + figures).
    ├── experiment_01/ ... experiment_07/
    │   ├── summary.csv           Canonical per-(system, W) metrics.
    │   ├── summary.md            Human-readable table (where applicable).
    │   └── paper_figures/        PNG + PDF figures (where applicable).
    └── ...
```

**Provenance.** The engine in `src/` and the experiment scripts are the
computational source of truth. The figure scripts (`plot_experiment_*.py`) apply
a single, consistent, colorblind-safe paper style (Okabe–Ito palette, one marker
per system, editable-text PDF + high-DPI PNG, shared legend strips).

---

## Installation

Requires **Python ≥ 3.10**.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Everything (engine + plotting + tests):
pip install -r requirements.txt

# For byte-for-byte reproduction of the committed figures:
# pip install -r requirements-lock.txt
```

The scripts add `src/` to `sys.path` automatically, so **no install step is
required** — you can run them directly from the repository root. Installing the
package (`pip install -e .`) is optional.

---

## Quick start

Verify the whole pipeline in under a minute (tiny trials, not the paper
protocol):

```bash
make smoke          # or:  ./reproduce.sh smoke
```

Run the test suite:

```bash
make test           # or:  python -m pytest tests/ -q
```

Regenerate every figure from the committed CSVs (fast, ~30s):

```bash
make figures
```

---

## The seven experiments

Every experiment writes a canonical `summary.csv` (one row per `system × W`, or
`× guard_digits`) plus, where relevant, auxiliary CSVs and a Markdown table.
Fixed random seeds are baked into each script, so runs are reproducible.

### Experiment 1 — Clean representation benchmark
Static geometry of each codebook before any fault or arithmetic: codebook size,
ambient size, sparsity, information capacity (bits/digit), normalized deficit,
nonzero density, and clean round-trip exactness.

```bash
python experiments/experiment_01_clean_representation.py --trials 10000 --widths 8 12 16 20 24
```
→ `results/experiment_01/{summary.csv,summary.md}` · **paper: Table 1, Table 8.**

### Experiment 2 — Arithmetic canonicalization benchmark
*Not* a fault experiment. Two admissible words are added digit-wise, producing a
raw state (with temporary digit `2`), which is canonicalized in the bounded
window. Measures arithmetic exactness, canonicalization success, rewrite cost
`κ`, propagation depth `π`, overflow/truncation, and cascade-length tails.

```bash
python experiments/experiment_02_arithmetic_canonicalization.py --trials 10000 --widths 8 12 16 20 24
```
→ `results/experiment_02/{summary.csv,summary.md,cascade_histogram.csv,cascade_histogram.md}` · **paper: Table 2, Fig 6, Fig 7.**

### Experiment 3 — Single-digit corruption benchmark
One uniformly random digit is corrupted, then the bounded-window repair is
applied. Measures structural detectability, structural exactness, semantic
survival (theoretically 0 — the No-Free-Lunch result), error magnitude, repair
cost, and position-wise detectability.

```bash
python experiments/experiment_03_single_digit_corruption.py --trials 10000 --widths 8 12 16 20 24
```
→ `results/experiment_03/{summary.csv,summary.md,position_summary.csv,position_summary.md}` · **paper: Table 3, Fig 8, Fig 9.**

### Experiment 4 — Local burst corruption benchmark
A contiguous burst of width `b ∈ {2,3,4,5}` with a random nonzero mask is
injected. Bursts can be semantically survivable when the error vector lands in
the algebraic kernel of the base (e.g. `100 ↔ 011`).

```bash
python experiments/experiment_04_local_burst_corruption.py --trials 10000 --widths 8 12 16 20 24 --burst-widths 2 3 4 5
```
→ `results/experiment_04/{summary.csv,summary.md,burst_width_summary.csv,burst_width_summary.md}` · **paper: Table 4, Fig 10.**

### Experiment 5 — Exhaustive small-window verification
No sampling: enumerates **every** admissible word and **every** valid
single-digit and burst corruption for small even widths `W ≤ 16`. Confirms the
Monte-Carlo results are not artifacts. Deterministic (no RNG).

```bash
python experiments/experiment_05_exhaustive.py --widths 4 6 8 10 12 14 16 --burst-widths 2 3 4 5
```
→ `results/experiment_05/{summary.csv,single_digit_summary.csv,burst_by_width.csv}` · **paper: Table 5, Table 6.**
*Long-running (tens of minutes at W = 16). Deterministic, so the committed CSVs are exact.*

### Experiment 6 — Algebraic burst injection
Replaces random noise with kernel-aligned substitutions (`100→011`, `1000→0111`,
`10000→01111`). Interior injections give semantic survival = 1.000, directly
confirming the Algebraic Kernel Criterion.

```bash
python experiments/experiment_06_algebraic_burst.py --trials 10000 --widths 8 12 16 20 24
```
→ `results/experiment_06/summary.csv` · **paper: Experiment 6 (kernel survival).**

### Experiment 7 — Guard-digit truncation analysis
Adds `g ∈ {0,2,4,8,12,16}` least-significant guard digits and measures how
lower-boundary truncation, arithmetic exactness, error, and cost respond
(validating the guard-digit error bound, Lemma 2).

```bash
python experiments/experiment_07_guard_digits.py --trials 10000 --widths 8 12 16 20 24 --guards 0 2 4 8 12 16
```
→ `results/experiment_07/summary.csv` · **paper: Table 7, Fig 11.**

All scripts accept `--help`, `--systems`, `--seed`, and `--output-dir`.

---

## Reproducing the paper

Full pipeline (all experiments with the paper protocol, then all figures):

```bash
make all            # or:  ./reproduce.sh
```

Individual stages:

```bash
make experiments    # all 7 data experiments
make figures        # all figures from the committed CSVs
make exp03 fig03    # just one experiment + its figures
```

Expected single-core wall-clock: exp 1 ≈ 10 s, exp 2–4 ≈ 45–55 s each,
exp 6 ≈ 20 s, exp 7 ≈ 5 min, exp 5 ≈ 30–60 min (exhaustive).

---

## Figures

Figure scripts read a committed `summary.csv` and write **both** `.png`
(240 DPI) and `.pdf` (editable text) into `results/experiment_XX/paper_figures/`.
Each experiment folder also gets a standalone legend strip for compact
multi-panel layouts.

A complete mapping from every generated figure filename to its paper element —
and which figures are **paper** figures versus **supplementary** extras — is in
[**FIGURES.md**](FIGURES.md).

---

## Package API

The engine lives under `src/redundancy_exp/`. Minimal public surface:

```python
from redundancy_exp import DigitWindow, make_default_systems

systems = make_default_systems()          # dict: name -> RepresentationSystem
phi     = systems["phi"]
window  = DigitWindow(L=12, R=11)         # register I_{L,R}, width W = 24

word    = phi.sample_admissible(window, rng)   # exact-uniform admissible word
value   = phi.value_key(word, window)          # exact value in the base's ring
```

Fault, arithmetic, and metric helpers live in `corruption.py`, `burst.py`,
`algebraic.py`, `arithmetic.py`, `normalization.py`, and `metrics.py`. The
experiment scripts are thin drivers over these modules; read any
`experiment_0*.py` for a worked example.

---

## Reproducibility notes

- **Determinism.** All Monte-Carlo experiments use `numpy.random.default_rng`
  seeded from `--seed` (defaults baked in: 20260620–20260623 for exp 1–4,
  `42` for exp 6, `100` for exp 7). Re-running with defaults reproduces the
  committed CSVs exactly. Experiment 5 is exhaustive and has no RNG.

- **Experiment 7 uses one RNG stream across all widths.** Run it as a single
  process over the full width sweep (as `make exp07` / `reproduce.sh` do) to
  reproduce Table 7 exactly. Running widths in separate processes re-seeds each
  one and yields a different (equally valid) realization.

- **Filename convention.** Every experiment writes `summary.csv`; every figure
  script reads `summary.csv`. (Earlier drafts had inconsistent `summary5.csv` /
  `summary7.csv` names — these have been standardized.)

- **Boundary accounting.** Writes past the most-significant edge are counted as
  *overflow*; writes past the least-significant edge as *truncation*. Both are
  discarded, so arithmetic exactness can legitimately fail even when the final
  in-window word is admissible.

---

## Testing

```bash
python -m pytest tests/ -q
```

The suite covers the window model, value maps, sampling, normalization, the
arithmetic canonicalizer, and the corruption/burst models.

---

## Citing

If you use this code, data, or figures, please cite the paper and the software
archive. See [`CITATION.cff`](CITATION.cff). The archived release carries a
permanent DOI (`10.5281/zenodo.XXXXXXX` — update once minted), and the code is
maintained at `https://github.com/USERNAME/REPOSITORY`.

---

## License

MIT — see [`LICENSE`](LICENSE).

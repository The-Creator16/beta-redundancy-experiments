# Intrinsic Redundancy and Local Robustness in Finite β-Expansion Systems

Reproducible experiment code, numerical data, figures and tables for **Intrinsic
Redundancy and Local Robustness in Finite Beta-Expansion Systems**, by Adilbek
Taizhanov and Miras Seilkhan.

The study distinguishes ambient semantic redundancy from canonical codebook
admissibility. It measures structural fault detection, value preservation after
bounded repair, semantic survival, arithmetic canonicalization cost, and boundary
loss in finite registers.

Five representation systems are compared at equal register width:

| Key | System | Base | Alphabet | Canonical constraint |
| --- | --- | --- | --- | --- |
| `binary` | Standard binary | 2 | {0,1} | Every binary word is admissible |
| `signed2_naf` | Signed Non-Adjacent Form | 2 | {-1,0,1} | No adjacent nonzero digits |
| `phi` | Golden-ratio system | φ=(1+√5)/2 | {0,1} | Forbid `11` |
| `tribonacci` | Tribonacci | Positive root of x³=x²+x+1 | {0,1} | Forbid `111` |
| `tetranacci` | Tetranacci | Positive root of x⁴=x³+x²+x+1 | {0,1} | Forbid `1111` |

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The seven experiments](#the-seven-experiments)
- [Reproducing the numerical results](#reproducing-the-numerical-results)
- [Figures and tables](#figures-and-tables)
- [Package API](#package-api)
- [Reproducibility notes](#reproducibility-notes)
- [Testing and audits](#testing-and-audits)
- [Troubleshooting](#troubleshooting)
- [Citation and license](#citation-and-license)

## Repository layout

| Path | Purpose |
| --- | --- |
| `README.md` | Installation, usage and experiment overview |
| `FIGURES.md` | Publication figure numbers and data sources |
| `REPRODUCIBILITY.md` | Protocols, table selections and audit scope |
| `Makefile` | Export, simulation, test and audit commands |
| `requirements.txt` | Python dependencies |
| `requirements-export-lock.txt` | Python plotting versions used for the export environment |
| `pyproject.toml` | Optional installable Python package and dependency groups |
| `CITATION.cff` | Authors and citation metadata |
| `LICENSE` | MIT license |
| `src/redundancy_exp/` | Reusable scientific engine |
| `experiments/experiment_*.py` | Seven numerical experiment runners |
| `experiments/plot_experiment_*.py` | Complete benchmark plot collection |
| `experiments/plot_journal.py` | Journal and supplementary numerical panels |
| `experiments/build_release.py` | Select and assemble publication outputs |
| `experiments/diagrams/` | Reproducible TikZ diagram sources |
| `experiments/audit/` | Audit replay and aggregate-validation scripts |
| `tests/` | Engine, table-selection and symbolic-identity checks |
| `results/data/experiment_01/` … `experiment_07/` | Archived numerical CSV inputs |
| `results/audit/` | Saved audit aggregates |
| `results/input_sha256.json` | Archived experiment CSV checksums |
| `results/journal/`, `results/supplementary/`, `results/all/` | Generated publication outputs |

The engine modules are `window.py` (finite registers), `systems.py` (bases and
value maps), `sampling.py` (constrained-word sampling), `normalization.py`
(rewrite rules), `arithmetic.py`, `corruption.py`, `burst.py`, `algebraic.py`,
and `metrics.py`.

## Installation

**Requirements:** Python **3.10+**; NumPy, pandas and Matplotlib; a working
**pdfLaTeX** installation with **standalone, TikZ/PGF, amsmath and amssymb**.
pytest is included for tests. Make is convenient on Linux/macOS but optional.
LaTeX is a system dependency: `pip install` does not install it.

Download and extract the repository, then open a terminal in its root directory
(the directory containing `README.md` and `Makefile`). Choose the instructions
for your operating system.

### Linux: Ubuntu / Debian

Install Python, Make and LaTeX:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip make texlive-latex-extra texlive-pictures texlive-fonts-recommended
```

Create the Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pdflatex --version
make journal
```

The generated files are in `results/journal/`.

### Linux: Fedora

Install Python, Make and LaTeX:

```bash
sudo dnf install python3 python3-pip make texlive-latex texlive-standalone texlive-pgf texlive-amsmath texlive-amsfonts
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pdflatex --version
make journal
```

### macOS

With [Homebrew](https://brew.sh/) installed:

```bash
brew install python
brew install --cask mactex-no-gui
```

MacTeX is a full TeX distribution and includes the required diagram packages.
After installation, reopen Terminal or add its executable directory to the
current session:

```bash
export PATH="/Library/TeX/texbin:$PATH"
```

In the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pdflatex --version
python experiments/build_release.py journal
```

If Make is available, `make journal` is equivalent to the final command.
Alternatively, install MacTeX using the installer from [tug.org/mactex](https://tug.org/mactex/).

### Windows: PowerShell

Install Python 3.10+ from [python.org](https://www.python.org/downloads/windows/)
with the Python launcher enabled. Install [MiKTeX](https://miktex.org/download)
or use WinGet:

```powershell
winget install --id MiKTeX.MiKTeX --exact
```

Complete MiKTeX setup, open MiKTeX Console and check for updates. Install the
`standalone`, `pgf`, `amsmath` and `amsfonts` packages from its Packages page.
The same packages can be installed from the command line after setup:

```powershell
mpm --install=standalone
mpm --install=pgf
mpm --install=amsmath
mpm --install=amsfonts
```

Use the user or administrator mode that matches the installation. Close and
reopen PowerShell after installing Python/MiKTeX so their commands are on PATH.

In the repository root, the following commands work without activation or Make:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
pdflatex --version
.\.venv\Scripts\python.exe experiments/build_release.py journal
```

For supplementary or all outputs:

```powershell
.\.venv\Scripts\python.exe experiments/build_release.py supplementary
.\.venv\Scripts\python.exe experiments/build_release.py all
```

Using the virtual environment's executable directly avoids PowerShell activation
policy issues. Run `py -3 --version` to confirm Python is at least 3.10.

### Optional package installation

The command-line runners add `src/` to their import path automatically. To use
the engine from a Python shell or another project, install it in editable mode
from the repository root:

```bash
python -m pip install -e .
```

To use the recorded plotting dependency versions, install
`requirements-export-lock.txt` in a compatible environment. These versions do
not guarantee identical PDF bytes: metadata and font/toolchain versions can vary.

## Quick start

After installation, choose the output you need:

| Command | Output | Contents |
| --- | --- | --- |
| `make journal` | `results/journal/` | Article Figures 1–4 and Tables 1–5 |
| `make supplementary` | `results/supplementary/` | Figures S1–S9 and Tables S1–S9 |
| `make all` | `results/all/` | Both publication sets and the complete benchmark plot collection |
| `make smoke` | `results/smoke/` | Small runs of all seven experiment scripts |
| `make test` | Terminal report | Unit and publication-selection tests |
| `make audits` | Audit validation report | Normalized errors and saved aggregate checks |
| `make clean` | Removes generated bundles and smoke outputs | Preserves archived CSVs and code |
| `make` | Terminal help | Available targets |

**`make all` regenerates figures and tables from saved data. It does not rerun
simulations.** To generate new numerical results, use the experiment commands
below.

The three export commands also work without Make on every operating system:

```bash
python experiments/build_release.py journal
python experiments/build_release.py supplementary
python experiments/build_release.py all
```

On Windows, replace `python` with `.\.venv\Scripts\python.exe` unless the
virtual environment is activated. Each mode replaces only its own output
directory after a successful build.

## The seven experiments

Numerical runners write their outputs under `results/data/experiment_XX/` by
default. `summary.csv` contains one row per configuration; some experiments also
produce position summaries, burst-width summaries or cost histograms.

The examples below write to `results/custom/` to keep publication inputs intact.
They illustrate individual runner usage; full publication reproduction requires
the default widths, seeds, trial counts and sweep ordering.

### Experiment 1: clean representation

Measures codebook and ambient cardinality, sparsity, information capacity,
nonzero density and clean round-trip exactness.

```bash
python experiments/experiment_01_clean_representation.py --trials 10000 --widths 8 12 16 20 24 --output-dir results/custom/experiment_01
```

Used by article Table 1 and supplementary Table S1.

### Experiment 2: arithmetic canonicalization

Adds two admissible operands digit by digit and canonicalizes the raw sum in the
bounded workspace. Measures arithmetic exactness, native operation cost, cost
tails and discarded writes at both boundaries.

```bash
python experiments/experiment_02_arithmetic_canonicalization.py --trials 10000 --widths 8 12 16 20 24 --output-dir results/custom/experiment_02
```

Used by article Figure 3 and Table 4, and supplementary Figure S3 and Table S2.

### Experiment 3: single-digit corruption

Changes one digit, then applies bounded repair. Measures detectability,
structural exactness, semantic survival, residual error and repair activity.
The position-conditioned profile retains the location of the selected digit.

```bash
python experiments/experiment_03_single_digit_corruption.py --trials 10000 --widths 8 12 16 20 24 --output-dir results/custom/experiment_03
```

Used by article Figure 2a–b and Table 2, supplementary Figures S4–S5 and S9,
and Table S3.

### Experiment 4: local burst corruption

Samples a burst width, exposed interval and nonzero substitution mask. Measures
repair outcomes and propagation, including semantic survival from value-neutral
perturbations or cancellation with boundary loss.

```bash
python experiments/experiment_04_local_burst_corruption.py --trials 10000 --widths 8 12 16 20 24 --burst-widths 2 3 4 5 --output-dir results/custom/experiment_04
```

Used by article Figure 2c–d and Tables 2–3, supplementary Figures S6–S7 and
Table S4. Article Table 3 additionally uses the burst mechanism audit.

### Experiment 5: exhaustive small-window enumeration

Enumerates admissible words and injection choices, with no random sampling.
Its normalization interface differs from the Monte Carlo repair routine:
Signed NAF rejects inadmissible inputs, and multinacci repair stops at an upper
carry. These results therefore do not directly validate the full Monte Carlo
repair interface. Full enumeration can be expensive.

```bash
python experiments/experiment_05_exhaustive.py --widths 4 6 8 10 12 14 16 --burst-widths 2 3 4 5 --output-dir results/custom/experiment_05
```

Used by supplementary Tables S5–S6.

### Experiment 6: targeted algebraic burst injection

Injects value-neutral substitutions `100 → 011`, `1000 → 0111` and
`10000 → 01111`. Only eligible sampled words receive an injection. Exact-value
and original-word recovery checks are supplied by the targeted audit.

```bash
python experiments/experiment_06_algebraic_burst.py --trials 10000 --widths 8 12 16 20 24 --output-dir results/custom/experiment_06
```

Used by supplementary Table S7 together with `targeted_audit.csv`.

### Experiment 7: guard-digit arithmetic

Adds least-significant guard digits, canonicalizes in the extended workspace,
and projects back to the stored register. Measures workspace truncation,
post-projection error, numerical equality and native operation cost.

```bash
python experiments/experiment_07_guard_digits.py --trials 10000 --widths 8 12 16 20 24 --guards 0 2 4 8 12 16 --output-dir results/custom/experiment_07
```

Used by article Figure 4 and Table 5, supplementary Figure S8 and Tables S8–S9.
Audit tables use `guard_audit.csv` and `tail_scales.csv`. The tail scale is a
reference quantity, not an established bound on accumulated cascade loss.

Use `--help` on any runner for supported options. Options differ between scripts;
for example, the exhaustive runner has no random seed.

## Reproducing the numerical results

To rerun all seven experiments with the publication protocol:

```bash
make experiments
```

This writes to **`results/data/`**, replacing the archived experiment CSVs.
To write a separate numerical run instead:

```bash
make experiments RES=results/rerun
```

Individual Make targets are `exp01` through `exp07`. For example:

```bash
make exp03 RES=results/rerun
```

On Windows without Make, use the seven Python commands in the experiment
sections, retaining the complete default configuration.

Publication exports use the archived data in `results/data/` and saved audit
aggregates in `results/audit/`. They check `results/input_sha256.json` before
combining them. A custom numerical run is not automatically substituted into
publication tables. Compare the complete results and relevant audit replays
before preparing a new numerical release; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Figures and tables

Each publication bundle contains:

| Path | Contents |
| --- | --- |
| `figures/` | Numbered vector PDF panels, symbolic diagrams and legends |
| `tables/` | CSV, Markdown and LaTeX representations of each selected table |
| `manifest.json` | SHA-256 hashes of input data, source files and generated outputs |

For example, `results/journal/figures/figure_3a.pdf` is article Figure 3a.
Supplementary outputs use `figure_S1.pdf` through `figure_S9.pdf`. Multi-panel
figures remain separate resources for composition in LaTeX.

The `all` bundle groups plots into `figures/journal/`,
`figures/supplementary/`, `figures/complete/` and `figures/additional/`.
The complete collection includes PNG and PDF exports with experiment-based
names, including plots not selected for either publication.

CSV tables retain numerical precision. Markdown and LaTeX apply display rounding,
so a CSV mean of `11.5291` can appear as `11.53` in the readable table. LaTeX
exports are `tabular` fragments to include inside a table environment.

Numerical plots are regenerated from CSVs with Matplotlib. Symbolic diagrams
are compiled from `.tex` sources with pdfLaTeX. Neither process reads manuscript
PDFs or screenshots. Manuscript PDFs may be placed in the repository root as
reference documents without affecting the build.

See [FIGURES.md](FIGURES.md) for the full figure map and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for table selections.

## Package API

After `python -m pip install -e .`, the reusable engine can be imported:

```python
import numpy as np
from redundancy_exp import DigitWindow, make_default_systems

systems = make_default_systems()
phi = systems["phi"]
window = DigitWindow(L=12, R=11)
rng = np.random.default_rng(42)

word = phi.sample_word(window, rng)
value = phi.value_key(word, window)
assert phi.is_admissible(word)
assert phi.normalize(word, window).digits == word
```

Fault, arithmetic and metric helpers live in `corruption.py`, `burst.py`,
`algebraic.py`, `arithmetic.py`, `normalization.py` and `metrics.py`.
The experiment runners provide complete examples of their use.

## Reproducibility notes

- **Random streams:** Experiments 1–4 use seeds 20260620–20260623; Experiments
  6 and 7 use 42 and 100. A script advances its stream across the whole sweep.
- **Guard comparisons:** Experiment 7 must consume preceding widths to reproduce
  a later slice. Different guard counts use fresh operands, not paired samples.
- **Boundary policy:** Overflow and truncation record discarded writes above
  and below the workspace. Both flags can occur in the same trial.
- **Cost interpretation:** Native operation counts differ by system and do not
  represent equal execution times or measured circuit latency.
- **Audit scope:** Exact-value replays and saved aggregate validation have
  different scopes. Passing aggregate checks does not rerun every trial.
- **Exact versus floating residuals:** Article Table 5 uses exact workspace
  zeros at g=16; supplementary Table S9 retains floating evaluation residuals.
- **Environment:** `requirements-export-lock.txt` records plotting dependency
  versions. Matching versions helps comparison but does not guarantee identical
  PDF metadata, font output or CSV serialization across environments.

## Testing and audits

Run the tests:

```bash
python -m pytest tests/ -q
```

Validate saved audit aggregates and recompute normalized errors:

```bash
python experiments/audit/normalized_errors.py --repo .
python experiments/audit/validate_audits.py --repo .
```

With Make, these commands are `make test` and `make audits`. `make smoke`
checks all numerical runners on small configurations; it does not require LaTeX
and does not replace the publication data.

Tests cover the scientific engine, publication table selections, selected
published values and exact symbolic identities. Full Monte Carlo and exhaustive
replays are separate computations. Replay commands and limitations are described
in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Troubleshooting

| Message | Action |
| --- | --- |
| `pdflatex is required` | Install the OS-specific LaTeX distribution above; confirm `pdflatex --version` works in the same terminal |
| `LaTeX package missing: standalone.cls` | Install `standalone` using the TeX distribution package manager |
| `LaTeX package missing: tikz.sty` | Install PGF/TikZ (`texlive-pgf` on Fedora, `texlive-pictures` on Ubuntu, `pgf` in MiKTeX) |
| macOS cannot find an installed MacTeX | Run `export PATH="/Library/TeX/texbin:$PATH"` and retry |
| `ModuleNotFoundError` for NumPy/pandas/Matplotlib | Install `requirements.txt` using the same Python executable used to run the script |
| `make` is unavailable | Use `python experiments/build_release.py <mode>`; on Windows use the virtual environment executable |
| `Archived input changed` | Compare the data with the archived ensemble and audit records; do not bypass the checksum guard for a custom run |

Dependencies are checked before publication rendering. A failed export leaves
the previous completed output bundle intact. Do not run two builds of the same
mode simultaneously.

## Citation and license

If using this code or numerical results, cite the research and software archive
identified in [CITATION.cff](CITATION.cff). The DOI
[10.5281/zenodo.21140212](https://doi.org/10.5281/zenodo.21140212) identifies the
original archived research release; it does not by itself identify a later
journal software release.

The code is distributed under the **MIT License** — see [LICENSE](LICENSE).

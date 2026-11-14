# Data, audits and table definitions

The scientific engine and archived CSVs originate from upstream commit
d4c9d7e82cd8a6edfdff2218af9d6e3444a807c9. The journal audit scripts/data were
supplied separately. Duplicated snapshots and manuscript-specific integrity
records were removed; results/input_sha256.json records the retained data.

## Table selections
| Table | Inputs and selection |
| --- | --- |
| Journal 1 | Experiment 1, W=24 |
| Journal 2 | Experiments 3/4, W=24, normalized_errors; Single before Burst |
| Journal 3 | burst_mechanisms, W=24 |
| Journal 4 | Experiment 2, W=24; success omitted because it is 1 |
| Journal 5 | guard_audit + tail_scales, W=16, Phi/T3/T4, g=0/16 |
| S1 | Experiment 1 full width sweep; deficit normalized by alphabet size |
| S2 | Experiment 2, W=24; includes success |
| S3 | Experiment 3, W=24, normalized errors |
| S4 | Experiment 4, W=24, normalized errors |
| S5 | Experiment 5, W=16, single injections |
| S6 | Experiment 5, W=16, burst injections |
| S7 | targeted_audit, all 15 configurations |
| S8 | Experiment 7, W=16, Phi/T3/T4, endpoints g=0/16 |
| S9 | guard_audit + tail_scales, floating workspace residuals |

Journal 5 uses exact workspace zeros at g=16, asserted against exact-value-key
audit counts; S9 intentionally preserves floating residuals of order 1e-14–1e-13.
Both condition on no upper overflow. The tail scale is not a proved cascade bound.
S8 equality uses the original numerical tolerance; its endpoint columns are not
paired samples. Exhaustive S5/S6 use a simpler repair interface than Monte Carlo.
Native operation costs are not equal execution-time measurements.

CSV files use full precision and descriptive column names. Markdown and LaTeX
use rounded display values; LaTeX files are tabular fragments, not standalone
documents. The current builder is the authoritative column/selection definition.

## Audits
All commands run from the repository root.
Saved aggregate validation (no Monte Carlo replay):
```sh
make audits
```
This recomputes normalized errors and validates saved audit aggregates.
Optional, potentially expensive replays:
```sh
python experiments/audit/audit_burst_mechanisms.py --repo . --out results/audit/burst_mechanisms.csv
python experiments/audit/audit_targeted_guard.py --repo . --out results/audit
python experiments/audit/audit_signed_naf_counts.py --output results/audit/signed_naf_exact_counts.csv
make audits
```
Targeted/guard replay covers W=16 guards with preceding RNG draws consumed;
it does not replay every width. Aggregate checks do not rerun all Monte Carlo
trials or prove unrestricted termination.

## Validation and environment
requirements.txt contains Python dependencies; requirements-export-lock.txt
records the plotting versions used for export. Install TeX Live separately.
PDF timestamps can vary; compare plotted data or rendered content, not PDF bytes.
The three Make exports compile TikZ directly with pdflatex and use Matplotlib
for numerical graphs. Temporary files are discarded after export.

Before release, run make test, make audits and all three export modes.
Check manifests, table formatting and figure layout. Changes in document scope
must be reflected in build_release.py, FIGURES.md and table tests together.

The supplied PDF manuscripts are not embedded in this source archive. No
unspecified editorial corrections to those documents were made.

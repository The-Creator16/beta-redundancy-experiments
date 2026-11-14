# Publication figure map
Numbers below follow the supplied journal and supplementary PDFs.

| Output PDF | Content / data |
| --- | --- |
| journal/figure_1 | Single-digit symbolic fault and repair; TikZ |
| journal/figure_2a | Experiment 3: detectability vs W |
| journal/figure_2b | Experiment 3: exactness gap vs W |
| journal/figure_2c | Experiment 4: detectability vs burst width, W=24 |
| journal/figure_2d | Experiment 4: exactness gap vs burst width, W=24 |
| journal/figure_3a | Experiment 2: mean native cost |
| journal/figure_3b | Experiment 2: empirical cost tail, W=24 |
| journal/figure_3c | Experiment 2: arithmetic exactness |
| journal/figure_3d | Experiment 2: truncation |
| journal/figure_4 | Experiment 7: truncation vs guards, W=16 |
| supplementary/figure_S1 | Ambient phi semantic flow; supplied TikZ |
| supplementary/figure_S2 | Burst mask and symbolic repair; reconstructed TikZ |
| supplementary/figure_S3 | Arithmetic p99 native cost |
| supplementary/figure_S4 | Single-fault MAE |
| supplementary/figure_S5 | Single-fault mean native cost |
| supplementary/figure_S6 | Burst propagation span |
| supplementary/figure_S7 | Burst p99 cost |
| supplementary/figure_S8 | Guard post-projection MAE |
| supplementary/figure_S9 | Single-fault position-conditioned detectability |

Paths above mean results/<mode>/figures/<name>.pdf. Legend resources have the
same figure-number prefix. All mode retains these names in figures/journal/
and figures/supplementary/. Its complete/ subfolder keeps experiment-based names
because those plots include content with no journal/supplementary figure number.

The numeric panel recipe is experiments/plot_journal.py. Its internal legacy
stems identify experiment data; build_release.py maps them to publication names.
Complete original plots use plot_experiment_*.py and journal_style.py.
No pre-rendered inputs are required.

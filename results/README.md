# Committed results

These are the exact outputs of the paper protocol (fixed seeds; see the root
`README.md` → *Reproducibility notes*). Regenerate any of them with
`make experiments` / `make figures`, or per-experiment targets such as
`make exp03 fig03`.

Each `experiment_XX/` folder contains:

- **`summary.csv`** — canonical metrics, one row per `system × W`
  (or `× guard_digits` for Experiment 7).
- **`summary.md`** — the same table rendered for humans (Experiments 1–4).
- **auxiliary CSVs** — `cascade_histogram.csv` (Exp 2),
  `position_summary.csv` (Exp 3), `burst_width_summary.csv` (Exp 4),
  `single_digit_summary.csv` + `burst_by_width.csv` (Exp 5).
- **`paper_figures/`** — PNG (240 DPI) + PDF (editable text) figures, plus a
  legend strip (Experiments 1, 2, 3, 4, 7).

See [`../FIGURES.md`](../FIGURES.md) for the figure → paper-element mapping.

Experiment 5 is exhaustive (no RNG); its CSVs are exact by construction.
Experiment 6 has no figures; its result is the interior kernel-survival rate in
`experiment_06/summary.csv`.

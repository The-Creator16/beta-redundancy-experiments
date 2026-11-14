"""Check replay aggregates and independent counts against the released CSVs.
Run from any directory: python validate_audits.py --repo PATH
This validates saved audit outputs; it does not rerun the Monte Carlo trials.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2] / "results/audit"
sys.path.insert(0, str(a.repo / 'src'))
from redundancy_exp import make_default_systems
from redundancy_exp.corruption import system_base_float

def load(rel):
    return pd.read_csv(root / rel)

checks = {}
d = load('signed_naf_exact_counts.csv').set_index('W')
e = load('../data/experiment_05/summary.csv')
e = e[e.system == 'signed2_naf'].set_index('W')
for prefix in ('single', 'burst'):
    assert np.array_equal(d[f'{prefix}_total'], e[f'{prefix}_total_corruptions'])
    assert np.array_equal(d[f'{prefix}_admissible'], e[f'{prefix}_structural_exactness_count'])
    assert np.array_equal(e[f'{prefix}_semantic_survival_count'], np.zeros(len(e)))
checks['signed_naf_all_seven_widths_integer_counts'] = True

b = load('burst_mechanisms.csv').set_index(['system', 'W'])
e = load('../data/experiment_04/summary.csv').set_index(['system', 'W'])
for x, y in [('semantic', 'semantic_survival_rate'),
             ('structural_exact', 'structural_exactness_rate')]:
    assert np.array_equal(b[x], np.rint(e[y] * e.trials).astype(int))
assert np.array_equal(b.semantic, b.semantic_and_exact + b.semantic_without_exact)
assert np.array_equal(b.semantic_without_exact, b.semantic_with_boundary_flag)
assert (e.truncation_rate == 0).all()
checks['burst_decomposition_all_25_configurations'] = True

t = load('targeted_audit.csv')
for col in ['detected', 'admissible', 'exact_structural', 'exact_semantic', 'same_word', 'success']:
    assert (t[col] == t.injections).all()
assert (t.overflow == 0).all() and (t.truncation == 0).all()
assert (t.max_steps == 1).all() and (t.max_error == 0).all()
assert (t.interior + t.boundary == t.injections).all()
e = load('../data/experiment_06/summary.csv').set_index(['system', 'W'])
tt = t.set_index(['system', 'W'])
assert np.array_equal(tt.injections, e.injections_made)
assert np.allclose(tt.detected / tt.injections, e.structural_detectability_rate, rtol=0, atol=1e-12)
assert np.allclose(tt.exact_semantic / tt.injections, e.semantic_survival_rate, rtol=0, atol=1e-12)
checks['targeted_all_15_configurations_exact_word_recovery'] = True

g = load('guard_audit.csv').set_index(['system', 'W', 'guard_digits'])
e = load('../data/experiment_07/summary.csv')
e = e[e.W == 16].set_index(['system', 'W', 'guard_digits'])
for count, rate in [('tolerance_exact','arithmetic_exactness_rate'), ('success','canonicalization_success_rate'),
                    ('overflow','overflow_rate'), ('truncation','truncation_rate')]:
    assert np.array_equal(g[count], np.rint(e[rate] * e.trials).astype(int))
for c in ['mean_abs_error', 'worst_abs_error', 'mean_normalization_steps',
          'max_normalization_steps', 'p99_normalization_steps']:
    assert np.allclose(g[c], e[c], atol=1e-12, rtol=0)
assert g.tolerance_false_positive.sum() == g.tolerance_false_negative.sum() == 0
checks['guard_all_30_W16_configurations'] = True
checks['guard_tolerance_false_positive'] = 0
checks['guard_tolerance_false_negative'] = 0

tail = []
for name, system in make_default_systems().items():
    if name in ('binary', 'signed2_naf'):
        continue
    beta = system_base_float(system)
    for guard in (0, 16):
        r = g.loc[(name, 16, guard)]
        scale = 2 * beta ** (-(7 + guard + 1)) / (1 - 1 / beta)
        tail.append(dict(system=name, guard_digits=guard, tail_scale=scale,
                         ratio=r.max_extended_error_no_overflow / scale))
pd.DataFrame(tail).to_csv(root / 'tail_scales.csv', index=False)
(root / 'validation_summary.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))

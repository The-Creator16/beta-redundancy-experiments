from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redundancy_exp import DigitWindow, make_default_systems
from redundancy_exp.corruption import value_key_to_float


@dataclass
class AlgebraicBurstSummary:
    system: str
    W: int
    trials: int
    injections_made: int
    structural_detectability_rate: float
    semantic_survival_rate: float
    mean_abs_error: float


def run_algebraic_burst(system, window: DigitWindow, trials: int, rng: np.random.Generator) -> dict:
    # Определение алгебраических ядер (что на что меняем)
    # Ищем каноничный паттерн и ломаем его в неканоничный, но алгебраически равный
    if system.name == "phi":
        source_pattern, target_pattern = (1, 0, 0), (0, 1, 1)
    elif system.name == "tribonacci":
        source_pattern, target_pattern = (1, 0, 0, 0), (0, 1, 1, 1)
    elif system.name == "tetranacci":
        source_pattern, target_pattern = (1, 0, 0, 0, 0), (0, 1, 1, 1, 1)
    else:
        # Для Binary и Signed NAF нет таких тривиальных локальных алгебраических ядер
        return None

    injections = 0
    survivals = 0
    detectable = 0
    errors = []
    pat_len = len(source_pattern)

    for _ in range(trials):
        word = system.sample_word(window, rng)

        # Ищем все вхождения source_pattern
        occurrences = [i for i in range(len(word) - pat_len + 1) if word[i:i + pat_len] == source_pattern]

        if not occurrences:
            continue  # Пропускаем, если паттерна нет

        # Выбираем случайное вхождение и инжектируем burst
        idx = rng.choice(occurrences)
        corrupted = list(word)
        corrupted[idx:idx + pat_len] = target_pattern
        corrupted_tuple = tuple(corrupted)

        injections += 1

        if not system.is_admissible(corrupted_tuple):
            detectable += 1

        norm_result = system.normalize(corrupted_tuple, window)

        # Считаем семантическое выживание и ошибку
        orig_val = value_key_to_float(system, system.value_key(word, window))
        rep_val = value_key_to_float(system, system.value_key(norm_result.digits, window))

        err = abs(rep_val - orig_val)
        errors.append(err)

        if err < 1e-9:
            survivals += 1

    if injections == 0:
        return None

    return AlgebraicBurstSummary(
        system=system.name,
        W=window.width,
        trials=trials,
        injections_made=injections,
        structural_detectability_rate=detectable / injections,
        semantic_survival_rate=survivals / injections,
        mean_abs_error=float(np.mean(errors))
    ).__dict__


def main() -> None:
    parser = argparse.ArgumentParser(description="Run algebraic burst benchmark (Experiment 6).")
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "experiment_06")
    args = parser.parse_args()

    systems = make_default_systems()
    rng = np.random.default_rng(args.seed)
    rows = []

    for W in args.widths:
        window = DigitWindow.balanced(W)
        for sys_name in ["phi", "tribonacci", "tetranacci"]:
            print(f"Running algebraic burst eval for {sys_name} at W={W}...")
            result = run_algebraic_burst(systems[sys_name], window, args.trials, rng)
            if result:
                rows.append(result)
                print(
                    f"  -> Injections: {result['injections_made']}, Semantic Survival: {result['semantic_survival_rate']:.4f}, MAE: {result['mean_abs_error']:.4g}")

    if rows:
        df = pd.DataFrame(rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_dir / "summary.csv", index=False)
        print(f"[done] Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
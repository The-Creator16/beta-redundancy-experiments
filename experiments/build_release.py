"""Build isolated publication bundles from archived data, without resampling."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["binary", "signed2_naf", "phi", "tribonacci", "tetranacci"]
MULTI = SYSTEMS[2:]
JOURNAL = [
 "fig_exp3_structural_detectability_vs_w", "fig_exp3_structural_exactness_gap_vs_w",
 "fig_exp4_detectability_by_burst_width_w_24", "fig_exp4_exactness_gap_by_burst_width_w_24",
 "fig_exp2_mean_rewrite_cost_clean", "fig_exp2_cascade_tail_Wmax_clean",
 "fig_exp2_arithmetic_exactness_rate_vs_w", "fig_exp2_truncation_rate_vs_w",
 "fig_exp7_truncation_vs_guard", "fig_exp2_legend_strip", "fig_exp3_legend_strip",
 "fig_guard_legend"]
SUPPLEMENT = [
 "fig_exp2_p99_rewrite_cost_clean", "fig_exp3_mean_abs_error_vs_w",
 "fig_exp3_mean_repair_steps_vs_w", "fig_exp4_propagation_depth_by_burst_width_w_24",
 "fig_exp4_p99_repair_steps_vs_w", "fig_exp7_mae_vs_guard",
 "fig_exp3_structural_detectability_by_position_w_24",
 "fig_exp2_legend_strip", "fig_exp3_legend_strip", "fig_exp7_legend_strip"]

def ordered(df, keys=("system",)):
    df = df.copy()
    df["system"] = pd.Categorical(df.system, SYSTEMS, ordered=True)
    if "channel" in df:
        df["channel"] = pd.Categorical(df.channel, ["Single", "Burst"], ordered=True)
    return df.sort_values(list(keys)).reset_index(drop=True)

LABELS = dict(zip(SYSTEMS, ["Binary", "Signed NAF", "Phi", "T3", "T4"]))

def display_cell(column, value, row, stem):
    if column == "system": return LABELS[str(value)]
    if column == "mean_normalization_steps":
        if stem == "journal_table_2":
            return f"{value:.2f}" if row["channel"] == "Single" else f"{value:.3f}"
        if stem in ("journal_table_4", "supplementary_table_S2"):
            return f"{value:.2f}"
    if column in ("NMAE", "NDelta_max"): return f"{value:.5f}"
    if column.endswith("_rate"): return f"{value:.4f}"
    if column in ("p99_normalization_steps", "max_normalization_steps"): return str(int(value))
    if isinstance(value, float): return f"{value:.8g}"
    return str(value)

def export_table(df, destination, stem, description):
    folder = destination / "tables"
    df.to_csv(folder / f"{stem}.csv", index=False)
    headers = list(df.columns)
    rows = [[display_cell(c, row[c], row, stem) for c in headers]
            for _, row in df.iterrows()]
    lines = ["# " + stem, "", description,
             "", "CSV retains full precision; this view applies display rounding.", "",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    (folder / f"{stem}.md").write_text("\n".join(lines) + "\n")
    def escape(value):
        return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    tex = ["% " + description, r"\begin{tabular}{" + "l" * len(headers) + "}",
           r"\hline", " & ".join(map(escape, headers)) + r" \\", r"\hline"]
    tex += [" & ".join(map(escape, row)) + r" \\" for row in rows]
    tex += [r"\hline", r"\end{tabular}"]
    (folder / f"{stem}.tex").write_text("\n".join(tex) + "\n")

def table_data(data, audit):
    def e(n):
        return pd.read_csv(data/f"experiment_{n:02}"/"summary.csv")
    def a(name):
        return pd.read_csv(audit/f"{name}.csv")
    def w(df, width=24):
        return ordered(df[df.W == width])
    clean=e(1)
    # D_X = 1 - log_|alphabet|(|C|)/W; unlike the raw redundancy field,
    # Signed NAF uses alphabet size 3 in the denominator.
    import numpy as np
    clean["normalized_deficit"] = 1-clean.capacity_bits_per_digit/np.where(clean.system=="signed2_naf",np.log2(3),1)
    norm=a("normalized_errors")
    faults=[]
    for n,channel in [(3,"Single"),(4,"Burst")]:
        df=w(e(n)).merge(norm[norm.experiment==n][["system","W","NMAE","NDelta_max"]],on=["system","W"],validate="one_to_one")
        df["channel"]=channel; faults.append(df)
    single,burst=faults
    corruption=ordered(pd.concat(faults),("system","channel"))
    arithmetic=w(e(2))
    mechanisms=w(a("burst_mechanisms"))
    guard=a("guard_audit")
    guard=guard[(guard.W==16)&guard.system.isin(MULTI)&guard.guard_digits.isin([0,16])]
    guard=ordered(guard.merge(a("tail_scales"),on=["system","guard_digits"],validate="one_to_one"),("system","guard_digits"))
    guard["workspace_error"]=guard.max_extended_error_no_overflow
    endpoints=guard.guard_digits==16
    if not (guard.loc[endpoints,"exact_extended"] == guard.loc[endpoints,"no_overflow_trials"]).all():
        raise ValueError("Exact-zero workspace assertion no longer supported by audit")
    guard.loc[endpoints,"workspace_error"]=0.0
    guard["workspace_error_basis"]=guard.guard_digits.map({0:"floating evaluation",16:"exact value-key audit"})
    exhaustive=w(e(5),16)
    targeted=ordered(a("targeted_audit"),("W","system"))
    targeted["rewrite"]=targeted.system.map(dict(zip(MULTI,["100 -> 011","1000 -> 0111","10000 -> 01111"])))
    targeted["structural_exactness_rate"]=targeted.exact_structural/targeted.injections
    targeted["semantic_survival_rate"]=targeted.exact_semantic/targeted.injections
    g=e(7);g=g[(g.W==16)&g.system.isin(MULTI)&g.guard_digits.isin([0,16])]
    metrics=["truncation_rate","arithmetic_exactness_rate","mean_abs_error","p99_normalization_steps"]
    endpoint_table=g.pivot(index="system",columns="guard_digits",values=metrics)
    endpoint_table.columns=[f"{m}_g{n}" for m,n in endpoint_table.columns]
    endpoint_table=ordered(endpoint_table.reset_index())
    j={
     "table_1":(w(clean)[["system","codebook_size","codebook_sparsity","capacity_bits_per_digit","mean_nonzero_density"]],"Clean representation, W=24"),
     "table_2":(corruption[["system","channel","structural_detectability_rate","structural_exactness_rate","semantic_survival_rate","NMAE","mean_normalization_steps"]],"Corruption, W=24; 10000 trials per system/channel"),
     "table_3":(mechanisms[["system","semantic","semantic_and_exact","semantic_without_exact"]],"Burst survival mechanisms, W=24"),
     "table_4":(arithmetic[["system","arithmetic_exactness_rate","overflow_rate","truncation_rate","mean_normalization_steps","p99_normalization_steps","max_normalization_steps"]],"Arithmetic, W=24; native costs are not equal-time operations"),
     "table_5":(guard[["system","guard_digits","no_overflow_trials","workspace_error","workspace_error_basis","tail_scale","max_projected_error_no_overflow"]],"W=16; conditional on no overflow. Exact zeros at g=16; tail scale is not a cascade bound"),
    }
    s={
     "table_S1":(ordered(clean,("W","system"))[["system","W","L","R","codebook_size","ambient_size","codebook_sparsity","capacity_bits_per_digit","normalized_deficit","mean_nonzero_density"]],"Full clean width sweep"),
     "table_S2":(arithmetic[["system","arithmetic_exactness_rate","canonicalization_success_rate","overflow_rate","truncation_rate","mean_normalization_steps","p99_normalization_steps","max_normalization_steps"]],"Full arithmetic summary, W=24"),
     "table_S3":(single[["system","structural_detectability_rate","structural_exactness_rate","semantic_survival_rate","mean_abs_error","NMAE","worst_abs_error","NDelta_max","mean_normalization_steps","p99_normalization_steps"]],"Single faults, W=24"),
     "table_S4":(burst[["system","structural_detectability_rate","structural_exactness_rate","semantic_survival_rate","mean_abs_error","NMAE","mean_normalization_steps","mean_propagation_depth"]],"Burst faults, W=24"),
     "table_S5":(exhaustive[["system","single_total_corruptions","single_structural_detectability_rate","single_structural_exactness_rate","single_semantic_survival_count","single_semantic_survival_rate"]],"Released exhaustive interface, W=16; differs from Monte Carlo repair"),
     "table_S6":(exhaustive[["system","burst_total_corruptions","burst_structural_detectability_rate","burst_structural_exactness_rate","burst_semantic_survival_rate"]],"Released exhaustive interface, W=16; injection-weighted bursts"),
     "table_S7":(targeted[["system","rewrite","W","placements","injections","interior","boundary","structural_exactness_rate","semantic_survival_rate","overflow","truncation"]],"Targeted algebraic identities, all widths"),
     "table_S8":(endpoint_table,"Guard endpoints at W=16, g=0 and16; equality uses the released numerical tolerance"),
     "table_S9":(guard[["system","guard_digits","no_overflow_trials","max_extended_error_no_overflow","tail_scale","ratio","max_projected_error_no_overflow"]],"Floating residuals retained; do not replace them with the exact zeros in journal table 5"),
    }
    return j,s

def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/"experiments"/script),*map(str,args)],
                   check=True,cwd=ROOT,env={**os.environ,"MPLBACKEND":"Agg"})

JOURNAL_NAMES = [
 "figure_2a", "figure_2b", "figure_2c", "figure_2d",
 "figure_3a", "figure_3b", "figure_3c", "figure_3d", "figure_4",
 "figure_3_legend", "figure_2_legend", "figure_4_legend"]
SUPPLEMENT_NAMES = [
 "figure_S3", "figure_S4", "figure_S5", "figure_S6", "figure_S7",
 "figure_S8", "figure_S9", "figure_S3_legend", "figure_S4-S7-S9_legend", "figure_S8_legend"]

def compile_diagram(stem, folder):
    if shutil.which("pdflatex") is None:
        raise RuntimeError("Install TeX Live with standalone, TikZ, amsmath; pdflatex is required")
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / "experiments/diagrams" / f"{stem}.tex"
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                           "-output-directory", str(folder), str(source)],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        raise RuntimeError(f"TikZ build failed: {stem}\n" + proc.stdout[-4000:])
    return folder / f"{stem}.pdf"

def build(mode):
    data=ROOT/"results/data"; audit=ROOT/"results/audit"
    for name, expected in json.loads((ROOT/"results/input_sha256.json").read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived input changed: {name}. Update audits before exporting a new protocol.")
    output=ROOT/"results"/mode
    with tempfile.TemporaryDirectory(dir=ROOT/"results",prefix=".build-") as tmp:
        staging=Path(tmp)/"bundle"
        for sub in ["tables","figures"]: (staging/sub).mkdir(parents=True)
        recipe=Path(tmp)/"panels"
        run("plot_journal.py","--data",data,"--output",recipe)
        j,s=table_data(data,audit)
        groups=["journal","supplementary"] if mode=="all" else [mode]
        for group in groups:
            panels=JOURNAL if group=="journal" else SUPPLEMENT
            names=JOURNAL_NAMES if group=="journal" else SUPPLEMENT_NAMES
            figdir=staging/"figures"/group if mode=="all" else staging/"figures"
            figdir.mkdir(exist_ok=True)
            for original, name in zip(panels,names):
                shutil.copy2(recipe/f"{original}.pdf",figdir/f"{name}.pdf")
            diagrams=["figure_1"] if group=="journal" else ["figure_S1","figure_S2"]
            for name in diagrams:
                pdf=compile_diagram(name,Path(tmp)/"latex")
                shutil.copy2(pdf,figdir/pdf.name)
            for stem,(frame,note) in (j if group=="journal" else s).items():
                export_table(frame,staging,group+"_"+stem,note)
        if mode=="all":
            for n in [1,2,3,4,7]:
                run(f"plot_experiment_{n:02}.py","--output-dir",staging/"figures"/"complete"/f"experiment_{n:02}")
            unused=staging/"figures"/"additional";unused.mkdir()
            for p in recipe.glob("*.pdf"):
                if p.stem not in set(JOURNAL+SUPPLEMENT): shutil.copy2(p,unused)
        entries=[{"path":str(p.relative_to(staging)), "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(staging.rglob("*")) if p.is_file()]
        (staging/"manifest.json").write_text(json.dumps({"mode":mode,"resampled":False,"files":entries},indent=2)+"\n")
        if output.exists(): shutil.rmtree(output)
        shutil.move(staging,output)
    print(f"Built {output.relative_to(ROOT)}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["all","journal","supplementary"])
    build(parser.parse_args().mode)

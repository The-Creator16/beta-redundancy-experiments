"""Recompute Experiment 4 and split survival by structural exactness."""
from pathlib import Path
import argparse,sys,csv
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo/"src"))
from redundancy_exp import DigitWindow,make_default_systems
from redundancy_exp.burst import evaluate_burst_trial
s=make_default_systems();rng=np.random.default_rng(20260623);rows=[]
for W in [8,12,16,20,24]:
 for name,x in s.items():
  win=DigitWindow.balanced(W)
  r=dict(system=name,W=W,trials=10000,semantic=0,semantic_and_exact=0,
         semantic_without_exact=0,semantic_with_boundary_flag=0,structural_exact=0)
  for _ in range(10000):
   t=evaluate_burst_trial(system=x,window=win,burst_widths=(2,3,4,5),rng=rng,max_steps=10*W)
   r["structural_exact"]+=int(t.structural_exact)
   r["semantic"]+=int(t.semantic_survival)
   r["semantic_and_exact"]+=int(t.semantic_survival and t.structural_exact)
   r["semantic_without_exact"]+=int(t.semantic_survival and not t.structural_exact)
   r["semantic_with_boundary_flag"]+=int(t.semantic_survival and (t.overflow or t.truncation))
  rows.append(r);print(r,flush=True)
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


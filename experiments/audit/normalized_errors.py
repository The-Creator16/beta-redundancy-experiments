"""Recompute Vmax by finite-state maximization, then normalize archived errors.
Usage: python normalized_errors.py --repo PATH
The maximization uses the stated alphabet and run constraint, not the paper's
printed Vmax numbers. Error magnitudes and maxima are floating-point quantities.
"""
from pathlib import Path
import argparse, sys
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--repo', type=Path, required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo/'src'))
from redundancy_exp import make_default_systems
from redundancy_exp.corruption import system_base_float
root=Path(__file__).resolve().parents[2] / "results/audit"
rows=[]
for name,system in make_default_systems().items():
    beta=system_base_float(system)
    cap=24 if name=='binary' else 1 if name=='signed2_naf' else system.degree-1
    alphabet=(-1,0,1) if name=='signed2_naf' else (0,1)
    states={0:0.0}
    for i in range(12,-12,-1):
        nxt={}
        for run,val in states.items():
            for digit in alphabet:
                if digit and run==cap: continue
                nr=run+1 if digit else 0
                nv=val+digit*beta**i
                nxt[nr]=max(nxt.get(nr,float('-inf')),nv)
        states=nxt
    vmax=max(states.values())
    for exp in (3,4):
        df=pd.read_csv(root/f'../data/experiment_{exp:02}/summary.csv')
        r=df[(df.system==name)&(df.W==24)].iloc[0]
        rows.append(dict(experiment=exp,system=name,W=24,Vmax=vmax,
                         NMAE=r.mean_abs_error/vmax,NDelta_max=r.worst_abs_error/vmax))
pd.DataFrame(rows).to_csv(root/'normalized_errors.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))

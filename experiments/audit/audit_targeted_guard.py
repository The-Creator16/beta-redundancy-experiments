"""Reproduce targeted-burst and W=16 guard audits without changing algorithms.
Usage: python audit_targeted_guard.py --repo PATH --out PATH
The guard audit preserves the released random-stream order for W=8,12,16.
Earlier widths consume the same samples but skip unused canonicalization.
"""
from pathlib import Path
import argparse, sys, csv, json, importlib.util
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--repo",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.repo/"src"))
from redundancy_exp import DigitWindow, make_default_systems
from redundancy_exp.corruption import value_key_to_float
from redundancy_exp.arithmetic import canonicalize_raw_sum
systems=make_default_systems()
def save(name,rows):
    with (args.out/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

rng=np.random.default_rng(42); rows=[]
for W in [8,12,16,20,24]:
    win=DigitWindow.balanced(W)
    for name in ["phi","tribonacci","tetranacci"]:
        s=systems[name];m=s.degree;src=(1,)+(0,)*m;tgt=(0,)+(1,)*m
        r=dict(system=name,W=W,trials=10000,placements=0,injections=0,
               interior=0,boundary=0,detected=0,admissible=0,
               exact_structural=0,exact_semantic=0,same_word=0,
               success=0,overflow=0,truncation=0,max_steps=0,max_error=0.)
        for _ in range(10000):
            w=s.sample_word(win,rng)
            occ=[i for i in range(W-m) if w[i:i+m+1]==src]
            r["placements"]+=len(occ)
            if not occ: continue
            i=int(rng.choice(occ));c=w[:i]+tgt+w[i+m+1:]
            q=s.normalize(c,win); out=tuple(q.digits)
            r["injections"]+=1;r["boundary"]+=int(i==0 or i+m+1==W)
            r["interior"]+=int(i!=0 and i+m+1!=W)
            r["detected"]+=int(not s.is_admissible(c))
            r["admissible"]+=int(s.is_admissible(out))
            r["exact_structural"]+=int(s.value_key(out,win)==s.value_key(c,win))
            r["exact_semantic"]+=int(s.value_key(out,win)==s.value_key(w,win))
            r["same_word"]+=int(out==w)
            r["success"]+=int(q.success);r["overflow"]+=int(q.overflow)
            r["truncation"]+=int(q.truncation)
            r["max_steps"]=max(r["max_steps"],q.steps)
            r["max_error"]=max(r["max_error"],abs(value_key_to_float(s,s.value_key(out,win))-value_key_to_float(s,s.value_key(w,win))))
        rows.append(r);print("targeted",name,W,r["injections"],flush=True)
save("targeted_audit.csv",rows)

rng=np.random.default_rng(100); rows=[]
for W in [8,12,16]:
    win=DigitWindow.balanced(W)
    for name in ["binary","signed2_naf","phi","tribonacci","tetranacci"]:
        s=systems[name]
        for g in [0,2,4,8,12,16]:
            ext=DigitWindow(L=win.L,R=win.R+g)
            errs=[];steps=[];depths=[];N=0;maxpre=0.;maxpost=0.
            cnt=dict(success=0,overflow=0,truncation=0,tolerance_exact=0,
                     exact_post_projection=0,exact_extended=0,
                     tolerance_false_positive=0,tolerance_false_negative=0)
            for _ in range(10000):
                a=s.sample_word(win,rng);b=s.sample_word(win,rng)
                if W!=16:continue
                raw=tuple(x+y for x,y in zip(a,b))
                rawext=raw+(0,)*g
                q=canonicalize_raw_sum(s,rawext,ext)
                out_ext=tuple(q.digits);out=out_ext[:W]
                v0=value_key_to_float(s,s.value_key(a,win))+value_key_to_float(s,s.value_key(b,win))
                vo=value_key_to_float(s,s.value_key(out,win))
                ve=value_key_to_float(s,s.value_key(out_ext,ext))
                err=abs(vo-v0);errpre=abs(ve-v0)
                exact=s.value_key(raw,win)==s.value_key(out,win)
                ext_exact=s.value_key(rawext,ext)==s.value_key(out_ext,ext)
                tol=err<1e-9
                errs.append(err);steps.append(q.steps);depths.append(q.propagation_depth)
                cnt["success"]+=int(q.success);cnt["overflow"]+=int(q.overflow)
                cnt["truncation"]+=int(q.truncation);cnt["tolerance_exact"]+=int(tol)
                cnt["exact_post_projection"]+=int(exact);cnt["exact_extended"]+=int(ext_exact)
                cnt["tolerance_false_positive"]+=int(tol and not exact)
                cnt["tolerance_false_negative"]+=int(exact and not tol)
                if not q.overflow:
                    N+=1;maxpre=max(maxpre,errpre);maxpost=max(maxpost,err)
            if W==16:
                r=dict(system=name,W=W,L=win.L,R=win.R,guard_digits=g,trials=10000,**cnt,
                       mean_abs_error=float(np.mean(errs)),worst_abs_error=max(errs),
                       mean_normalization_steps=float(np.mean(steps)),max_normalization_steps=max(steps),
                       p99_normalization_steps=float(np.quantile(steps,.99)),
                       no_overflow_trials=N,max_extended_error_no_overflow=maxpre,
                       max_projected_error_no_overflow=maxpost)
                rows.append(r);print("guard",name,g,cnt["tolerance_exact"],N,maxpre,maxpost,flush=True)
save("guard_audit.csv",rows)


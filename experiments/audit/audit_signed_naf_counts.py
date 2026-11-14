"""Independent finite-state counts for Signed NAF admissible-to-admissible faults."""
from collections import defaultdict
from pathlib import Path
import argparse,csv
p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
def paired_count(W,start,b):
    states={(False,False,False):1}
    for i in range(W):
        nxt=defaultdict(int)
        for (prev_d,prev_z,changed),count in states.items():
            for d in (-1,0,1):
                if prev_d and d:continue
                zs=(-1,0,1) if start<=i<start+b else (d,)
                for z in zs:
                    if prev_z and z:continue
                    nxt[(bool(d),bool(z),changed or d!=z)]+=count
        states=nxt
    return sum(v for (_,_,c),v in states.items() if c)
rows=[]
for W in [4,6,8,10,12,14,16]:
    code=(2**(W+2)-(-1)**W)//3
    single=sum(paired_count(W,j,1) for j in range(W))
    total_s=code*W*2
    burst=sum(paired_count(W,j,b) for b in [2,3,4,5] if b<=W for j in range(W-b+1))
    total_b=code*sum((W-b+1)*(3**b-1) for b in [2,3,4,5] if b<=W)
    rows.append(dict(W=W,codebook=code,single_total=total_s,
        single_admissible=single,single_exact_rate=single/total_s,
        burst_total=total_b,burst_admissible=burst,burst_exact_rate=burst/total_b))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(rows[-1])


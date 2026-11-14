"""Rebuild Section 4 figures from the archived CSV files; no resampling.
Usage: python plot_section4.py --data DATA_DIR --output FIGURES_DIR
DATA_DIR contains experiment_02, experiment_03, experiment_04, experiment_07.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=Path(__file__).resolve().parents[1]/"results"/"data")
p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"results"/"recipe_panels");args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,
"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
"axes.linewidth":.65,"pdf.fonttype":42,"ps.fonttype":42,
"grid.alpha":.3,"grid.linewidth":.45})
styles={
"binary":("Binary","B","#333333","s","-"),
"signed2_naf":("Signed NAF","S","#0072B2","D","--"),
"phi":(r"$\Phi$",r"$\Phi$","#009E73","o","-."),
"tribonacci":(r"$T_3$",r"$T_3$","#D08C00","^",":"),
"tetranacci":(r"$T_4$",r"$T_4$","#A85C92","v",(0,(5,1,1,1)))}
def read(n,name="summary"):return pd.read_csv(args.data/f"experiment_{n:02}"/f"{name}.csv")
def axis(wide=False):
 f,a=plt.subplots(figsize=(6.9,2.55) if wide else (3.35,2.15),layout="constrained")
 a.spines[["top","right"]].set_visible(False);a.grid(axis="y");a.set_axisbelow(True)
 return f,a
def save(f,name):
 f.savefig(args.output/(name+".pdf"),metadata={"Creator":"Section 4 reproducible figure script"})
 plt.close(f)
def line(df,x,y,name,xlabel,ylabel,scale=None,wide=False,ylim=None):
 f,a=axis(wide)
 for sys,st in styles.items():
  z=df[df.system==sys].sort_values(x)
  if z.empty:continue
  yy=1-z["structural_exactness_rate"] if y=="gap" else z[y]
  a.plot(z[x],yy,color=st[2],marker=st[3],linestyle=st[4],lw=1.2,ms=4,
         markeredgewidth=.4,markeredgecolor="white",label=st[0])
 if scale=="symlog":
  a.set_yscale("symlog",linthresh=.25,linscale=.9)
  a.text(.02,.97,"symlog scale",transform=a.transAxes,ha="left",va="top",fontsize=7.5)
 elif scale=="log1p":
  a.set_yscale("function",functions=(np.log1p,np.expm1))
  a.set_yticks([0,1,2,3,5,10,20,25])
  a.text(.02,.97,"log(1 + cost) scale",transform=a.transAxes,ha="left",va="top",fontsize=7.5)
 elif scale:
  a.set_yscale(scale)
 a.set_xlabel(xlabel);a.set_ylabel(ylabel)
 xx=sorted(df[x].unique())
 a.set_xticks([-11,-8,-4,0,4,8,12] if wide else xx)
 if ylim:a.set_ylim(*ylim)
 elif not scale:a.set_ylim(bottom=-.025*max(1e-6,a.get_ylim()[1]))
 save(f,name)
e2=read(2);e3=read(3);e4=read(4);e7=read(7)
line(e2,"W","mean_normalization_steps","fig_exp2_mean_rewrite_cost_clean",
     r"Window width $W$",r"Mean native cost $\bar\kappa$")
line(e2,"W","p99_normalization_steps","fig_exp2_p99_rewrite_cost_clean",
     r"Window width $W$",r"Native cost $p_{99}(\kappa)$")
line(e2,"W","arithmetic_exactness_rate","fig_exp2_arithmetic_exactness_rate_vs_w",
     r"Window width $W$",r"Arithmetic exactness $\rho_{\rm arith}$",ylim=(-.02,1.02))
line(e2,"W","truncation_rate","fig_exp2_truncation_rate_vs_w",
     r"Window width $W$",r"Truncation rate",ylim=(-.02,1.02))
h=read(2,"cascade_histogram");h=h[h.W==24]
f,a=axis()
for sys,st in styles.items():
 z=h[h.system==sys].sort_values("steps")
 support=np.arange(0,int(z.steps.max())+1)
 tail=np.array([z.loc[z.steps>=k,"probability"].sum() for k in support])
 # P[kappa >= s] changes at integer thresholds. "pre" puts each drop at s.
 a.step(support,tail,where="pre",color=st[2],ls=st[4],lw=1.2)
 a.plot(support[::6],tail[::6],ls="none",color=st[2],marker=st[3],ms=3)
a.set_yscale("log");a.set_ylim(5e-5,1.3);a.set_xlabel(r"Native operation threshold $s$")
a.set_ylabel(r"Empirical $\Pr[\kappa\geq s]$");save(f,"fig_exp2_cascade_tail_Wmax_clean")
f,a=axis()
for sys,st in styles.items():
 z=e2[(e2.W==24)&(e2.system==sys)].iloc[0]
 x=z.mean_normalization_steps;y=z.arithmetic_exactness_rate
 a.scatter(x,y,color=st[2],marker=st[3],s=36,zorder=3)
 a.annotate(st[1],(x,y),xytext=(4,5),textcoords="offset points",fontsize=8)
a.set_ylim(-.03,.85);a.set_xlim(8,28)
a.set_xlabel(r"Mean native cost $\bar\kappa$");a.set_ylabel(r"Arithmetic exactness $\rho_{\rm arith}$")
save(f,"fig_exp2_exactness_cost_tradeoff_Wmax_clean")
line(e3,"W","structural_detectability_rate","fig_exp3_structural_detectability_vs_w",
     r"Window width $W$",r"Detectability $\delta_{\rm det}$",ylim=(-.015,.65))
line(e3,"W","gap","fig_exp3_structural_exactness_gap_vs_w",
     r"Window width $W$",r"Exactness gap $1-\rho_{\rm exact}$")
line(e3,"W","mean_abs_error","fig_exp3_mean_abs_error_vs_w",
     r"Window width $W$","Mean absolute error")
line(e3,"W","mean_normalization_steps","fig_exp3_mean_repair_steps_vs_w",
     r"Window width $W$",r"Mean native cost $\bar\kappa$","symlog")
pos=read(3,"position_summary");pos=pos[pos.W==24]
line(pos,"position_value","structural_detectability_rate","fig_exp3_structural_detectability_by_position_w_24",
     r"Digit position $i$",r"Detectability before repair $\delta_{\rm det}$",wide=True,ylim=(-.02,.75))
b=read(4,"burst_width_summary");b=b[b.W==24]
line(b,"burst_width","structural_detectability_rate","fig_exp4_detectability_by_burst_width_w_24",
     r"Burst width $b$",r"Detectability $\delta_{\rm det}$",ylim=(-.02,1.02))
line(b,"burst_width","gap","fig_exp4_exactness_gap_by_burst_width_w_24",
     r"Burst width $b$",r"Exactness gap $1-\rho_{\rm exact}$")
line(b,"burst_width","mean_propagation_depth","fig_exp4_propagation_depth_by_burst_width_w_24",
     r"Burst width $b$",r"Mean propagation span $\bar\pi$")
line(e4,"W","p99_normalization_steps","fig_exp4_p99_repair_steps_vs_w",
     r"Window width $W$",r"Native cost $p_{99}(\kappa)$","log1p")
g=e7[e7.W==16]
line(g,"guard_digits","truncation_rate","fig_exp7_truncation_vs_guard",
     r"Guard digits $g$","Workspace truncation rate",ylim=(-.02,1.02))
line(g,"guard_digits","mean_abs_error","fig_exp7_mae_vs_guard",
     r"Guard digits $g$","Post-projection MAE","log")
for n in [2,3,7]:
 f=plt.figure(figsize=(6.9,.32))
 handles=[Line2D([],[],color=v[2],marker=v[3],ls=v[4],lw=1.2,ms=4,label=v[0]) for v in styles.values()]
 f.legend(handles=handles,loc="center",ncol=5,frameon=False,fontsize=8.5,
          handlelength=2.3,columnspacing=1.4)
 save(f,f"fig_exp{n}_legend_strip")
f=plt.figure(figsize=(3.35,.55))
handles=[Line2D([],[],color=v[2],marker=v[3],ls=v[4],lw=1.2,ms=4,label=v[0]) for v in styles.values()]
f.legend(handles=handles,loc="center",ncol=3,frameon=False,fontsize=7.5,
         handlelength=1.8,columnspacing=1.0,handletextpad=.6)
f.savefig(args.output/"fig_guard_legend.pdf",bbox_inches="tight",pad_inches=.02,
          metadata={"Creator":"Section 4 reproducible figure script"})
plt.close(f)
print("Wrote",len(list(args.output.glob("*.pdf"))),"PDF files")


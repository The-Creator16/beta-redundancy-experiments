"""Shared visual settings recovered from the supplied journal plotting script."""
import matplotlib.pyplot as plt
RC = {"font.family":"DejaVu Sans", "font.size":8.5, "axes.labelsize":9,
      "xtick.labelsize":8, "ytick.labelsize":8, "axes.linewidth":.65,
      "pdf.fonttype":42, "ps.fonttype":42, "grid.alpha":.3, "grid.linewidth":.45}
LINESTYLES = {"#333333":"-", "#0072b2":"--", "#009e73":"-.",
              "#d08c00":":", "#a85c92":(0,(5,1,1,1))}
def configure():
    plt.rcParams.update(RC)
def finish(fig):
    for ax in fig.axes:
        ax.set_title("")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(False)
        ax.grid(axis="y", alpha=.3, linewidth=.45)
        ax.set_axisbelow(True)
        for line in ax.lines:
            color=str(line.get_color()).lower()
            if color in LINESTYLES:
                if line.get_linestyle() not in ("None", "none", ""):
                    line.set_linestyle(LINESTYLES[color])
                line.set_linewidth(1.2)
                line.set_markersize(4)
                line.set_markeredgewidth(.4)

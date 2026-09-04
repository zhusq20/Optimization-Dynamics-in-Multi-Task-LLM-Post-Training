"""Draw the editable GPAS overview. Values are schematic, not training data."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#D28B12", "#397FBA", "#258D7E", "#C85F48"]
NAMES = ["Math", "Code", "IF", "Science"]
NAVY, GRAY, BORDER = "#193D56", "#667789", "#D8E2EB"

def box(ax, x, y, w, h, fill="#F7FAFD"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.005,rounding_size=0.012",
                               fc=fill,ec=BORDER,lw=0.7,zorder=0))

def text(ax, x, y, s, size=7, color=NAVY, **kw):
    return ax.text(x,y,s,fontsize=size,color=color,ha="center",va="center",**kw)

def main():
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":7,
                         "mathtext.fontset":"dejavusans","pdf.fonttype":42,
                         "svg.fonttype":"none"})
    fig,ax=plt.subplots(figsize=(5.5,2.48))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis("off")
    text(ax,.5,.953,"Spend more samples on less precise task updates",9,weight="bold")
    for left in (.008,.344,.680):
        box(ax,left,.17,.312,.705)
    text(ax,.164,.824,"1  Measure variability",7.7,weight="bold")
    text(ax,.5,.824,"2  Allocate micro-batches",7.7,weight="bold")
    text(ax,.836,.824,"3  Keep task weights fixed",7.7,weight="bold")

    text(ax,.164,.74,"Ordinary dense-loss gradients",6.6)
    text(ax,.164,.676,r"Apply optimizer scaling $D$",6.6,color=GRAY)
    # Noise ratios 1:9:4:4 yield standard deviations 1:3:2:2.
    for i,(label,e,color) in enumerate(zip(NAMES,[1,9,4,4],COLORS)):
        y=.574-i*.081
        text(ax,.09,y,label,6.4,color=color)
        ax.add_patch(Rectangle((.146,y-.014),.13*e/9,.028,fc=color,ec="none"))
    text(ax,.164,.222,r"Measured noise $e_i$",6.6)

    text(ax,.5,.729,r"$m_i \propto w_i\sqrt{e_i}$",10)
    text(ax,.5,.653,"Example: 16 micro-batches",6.5,color=GRAY)
    for i,(label,count,color) in enumerate(zip(NAMES,[2,6,4,4],COLORS)):
        y=.574-i*.081
        text(ax,.4,y,label,6.4,color=color)
        for j in range(count):
            ax.add_patch(Rectangle((.458+j*.021,y-.016),.017,.032,fc=color,ec="none"))
        text(ax,.622,y,str(count),6.4,color=color)
    text(ax,.5,.222,"More noise, more samples",6.6)

    text(ax,.836,.722,"Average within each task",6.7)
    text(ax,.836,.629,r"$A=\frac{1}{4}\bar g_{\rm Math}+\frac{1}{4}\bar g_{\rm Code}$",8.5)
    text(ax,.836,.527,r"$+\frac{1}{4}\bar g_{\rm IF}+\frac{1}{4}\bar g_{\rm Science}$",8.5)
    text(ax,.836,.392,"Different sample counts",6.6,color=GRAY)
    text(ax,.836,.333,"Same intended task balance",6.6,color=GRAY)
    text(ax,.836,.222,"Then take the AdamW step",6.6)

    for left in (.326,.662):
        ax.annotate("",xy=(left+.015,.5),xytext=(left-.004,.5),
                    arrowprops={"arrowstyle":"-|>","color":NAVY,"lw":.8,
                                "mutation_scale":7})
    text(ax,.5,.113,"Reuse this step's statistics to allocate the next step",7,color="#397FBA")
    text(ax,.5,.049,"Cost-aware mode also uses measured time per task",6.8,color=GRAY)
    out=ROOT/"figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"gpas_main_figure.{ext}",dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()

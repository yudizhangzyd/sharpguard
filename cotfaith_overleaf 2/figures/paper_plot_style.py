"""Shared style config for CoT-Faith figures (ICLR / NeurIPS D&B ready)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SIZE = 10
DPI = 300
FMT = "pdf"

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,   # embed fonts (no rasterized text)
    "ps.fonttype":  42,
})

# Colorblind-safe palette (Tol qualitative)
C_COT_TRAINED    = "#4477AA"   # blue (CoT-trained models)
C_NO_COT         = "#EE6677"   # red (no-CoT ablation)
C_BASELINE       = "#CCBB44"   # yellow (non-CoT OpenVLA)
C_ECOT_BRIDGE    = "#228833"   # green (Bridge-trained CoT-VLA)
C_CTRL           = "#BBBBBB"   # gray (controls)

# 4-bucket palette for attention
BUCKETS = ["visual", "instr", "cot", "action_prev"]
BUCKET_COLORS = {
    "visual":      "#4477AA",
    "instr":       "#EE6677",
    "cot":         "#228833",
    "action_prev": "#AA3377",
}

FIG_DIR = "/Users/yudizhang/Documents/sharpguard/figures"

def save(fig, name, fmt=FMT):
    p = f"{FIG_DIR}/{name}.{fmt}"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {p}")
    plt.close(fig)

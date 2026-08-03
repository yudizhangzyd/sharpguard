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


def ours_bracket(ax, n_ours, label="our fine-tunes", y=-0.20, pad=0.32,
                 fontsize=None):
    """Draw a bracket under the first ``n_ours`` x-positions.

    The model axis mixes this paper's fine-tunes with a released checkpoint.
    Tagging every one of ours with an "Ours" prefix made the tick labels wider
    than their slots -- `data-50A` and `data-50B` overlapped outright. A single
    bracket says the same thing once, in the place readers expect a grouping.

    ``y`` is in axes coordinates, below the tick labels. ``fontsize`` is for
    figures authored at their include width, where the default is too large
    relative to the panel rather than too small on the page.
    """
    x0, x1 = -pad, n_ours - 1 + pad
    tr = ax.get_xaxis_transform()          # x in data, y in axes fraction
    ax.plot([x0, x0, x1, x1], [y + 0.022, y, y, y + 0.022],
            transform=tr, color="0.35", lw=0.7, clip_on=False)
    ax.text((x0 + x1) / 2, y - 0.012, label, transform=tr, ha="center",
            va="top", fontsize=(FONT_SIZE - 3 if fontsize is None else fontsize),
            color="0.35", style="italic", clip_on=False)


def save(fig, name, fmt=FMT):
    p = f"{FIG_DIR}/{name}.{fmt}"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {p}")
    plt.close(fig)

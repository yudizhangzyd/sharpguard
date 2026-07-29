"""Fig 7 — per-scene-domain faithful rate: multi-task view.

For 8 CoT-VLA variants × 3 scene domains × direction_flip, show that
faithful rate varies across scene domains (kitchen/living_room/study).
Highlights per-task variance — the "multi-task" story.
"""
import json, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("/Users/yudizhang/Documents/sharpguard/figures/per_scene_data.json"))

MODELS = ["Ours r=8", "Ours r=16", "Ours r=32", "Ours r=64",
           "Ours no-CoT", "Ours data-50A", "Ours data-50B", "ECoT-bridge"]
SCENES = ["kitchen", "living_room", "study"]
SCENE_LABELS = ["kitchen (30 tasks)", "living-room (17)", "study (14)"]
FAMS = ["direction_flip", "subject_swap", "gripper_flip"]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4),
                          gridspec_kw={"wspace": 0.30})

for ai, fam in enumerate(FAMS):
    ax = axes[ai]
    x = np.arange(len(MODELS))
    w = 0.25
    for si, (scene, slabel) in enumerate(zip(SCENES, SCENE_LABELS)):
        vs = []
        for m in MODELS:
            e = d.get(m, {}).get(fam, {}).get(scene, {})
            vs.append(e.get("mean", 0.0) if e.get("n", 0) > 0 else 0)
        color = {"kitchen": "#EE6677", "living_room": "#4477AA", "study": "#228833"}[scene]
        ax.bar(x + (si - 1) * w, vs, w, label=slabel,
                color=color, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("Ours ", "").replace("data-50", "d50")
                         .replace("no-CoT", "no-CoT")
                         .replace("ECoT-bridge", "bridge")
                         for m in MODELS],
                        rotation=45, ha="right", fontsize=FONT_SIZE-2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("faithful rate" if ai == 0 else "")
    ax.set_title(fam.replace("_", " "), fontsize=FONT_SIZE, style="italic",
                   loc="left")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

axes[0].legend(loc="upper left", frameon=False, fontsize=FONT_SIZE-2)

save(fig, "fig7_per_scene")

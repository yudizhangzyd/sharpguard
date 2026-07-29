# CoT-Faith (ICLR 2026 submission) — Overleaf-ready

## Compile on Overleaf (30 seconds, no local install needed)
1. https://www.overleaf.com/project → New Project → Upload Project
2. Drag `cotfaith_overleaf.zip` in
3. Compiler = pdfLaTeX (default)
4. Recompile

## Files
- `main.tex`  — paper source (was cot_faith_iclr.tex)
- `figures/`  — 6 PDF figures (all vector) + generation scripts

## Figures
1. `fig1_workflow_matplotlib.pdf` — hero workflow (NEW; replaced overlapping TikZ)
2. `fig2_attention_distribution.pdf` — 11-model attention grouped bar
3. `fig3_edit_heatmap.pdf` — 8×10 causal-edit leaderboard heatmap
4. `fig4_dissociation.pdf` — attention-vs-causation 2-panel
5. `fig5_prompt_ablation.pdf` — prompt-truncation bar
6. `fig6_bridge_vs_libero.pdf` — 3-family Bridge-vs-LIBERO grouped bar

## What changed vs v1
- Restructured as **benchmark-artifact-first**, not findings-first
- Added §3 "The CoT-Faith Benchmark" (2.5 pp) with 4 design principles,
  3-tier edit taxonomy (T0 control / T1 word / T2 object-causal),
  formal Faithfulness Score equation, formal attention equation
- Added §5 as an explicit **CoT-Faith Leaderboard** table
- Added Table 3 comparison vs LIBERO / Colosseum / VLABench / LIBERO-Safety
  / Double-Edged Sword on 5 axes
- New Fig 1 (matplotlib, no TikZ overlap)
- Findings moved to §6 Analysis (Q1-Q4), no longer the frame
- Contribution bullets: artifact → protocol → coverage → release → findings-last

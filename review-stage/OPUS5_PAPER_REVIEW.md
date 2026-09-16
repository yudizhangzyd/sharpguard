# Opus 5 paper review — full submission (body + appendix), 2026-08-03

Four independent Opus 5 reviewers, each with a different lens, read
`cot_faith_arr.tex` (8pp body), `arr_appendix.tex` (12pp appendix) and
`build/cot_faith_arr.pdf` (20pp). No GPT/Codex reviewer was used (no API).

Reviewer lenses: (R1) senior ARR area chair on the body alone; (R2) structural
promote/demote audit across the two halves; (R3) exhibit audit over all 21
floats, reading the rendered pages; (R4) adversarial body-vs-appendix
consistency hunt.

Everything below marked **VERIFIED** was re-checked by hand against the source
after the reviewers reported it. Two reviewers independently found items 3, 5
and 6, which is why they are ranked where they are.

---

## Tier 1 — verified defects that damage credibility, all fixable by editing text

### 1. `A.5 "Discussion and limitations"` is an empty printed heading. VERIFIED
`arr_appendix.tex:344-346` emits `\subsection{Discussion and limitations}` with
zero body text, and three appendix references point into it, two of them citing
numbered items that exist nowhere in the PDF (`Section~\ref{sec:limitations}\,(viii)`
in the `tab:leaderboard` caption; `item~(v)` in the `fig:paths` caption). The
body compounds it: "The full list is Appendix~\ref{sec:appendix}".

### 2. Twenty appendix cross-references resolve to the list of deferred sections. VERIFIED
The builder re-homes each deferred section's label onto the deferred list so
LaTeX will not warn. Counted by label: `sec:paraphrase_null` 8,
`sec:f2_calib` 4, `sec:judge_edits` 3, `sec:limitations` 3, `sec:cross_family` 1,
`sec:deltapath` 1 (from the body). All print the same section number, so the
appendix says "see Section A.6" twenty times, where A.6 is titled "Sections
deferred to the full-length manuscript". No dangling `??`, but the reader is
sent in a circle.

### 3. The admission rule's stated ground is stale, and it is the claim the paper says will outlast the rest. VERIFIED
`cot_faith_arr.tex:355`: "It rejects seven of our own eight submissions ... That
rule ... is the part of this work we expect to outlast the rest."
`arr_appendix.tex:159` gives the ground: "Seven of the eight rows below do not
carry [a paraphrase floor]". They all carry one now: `tab:leaderboard` has a
populated `para.` column on all 8 rows and `tab:floors` prints a paraphrase floor
for all 12 configurations. The honest statement is stronger and true: the rule
admits **none** of the eight, seven because they score below their own
paraphrase floor and the eighth because its floor sits at its ceiling.
Worse, `scripts/verify_paper_numbers.py:2316` only checks that the *sentence is
present in the tex*, not that it is true of the artifacts.

### 4. The "decisive" third statistic in Table 1 is the sum of the two before it. VERIFIED arithmetically
`tab:floors`'s last column is the gap between the two floors, and the caption
calls it "the decisive one". Given the sign pattern the first two columns
report, `p - s = (p - F̄) + (F̄ - s)`, so it necessarily exceeds both. Checked
every row: no-CoT 0.193-0.154=0.039; r=8 0.587-0.408=0.179 (=0.139+0.040);
r=32 0.219; DT RL 0.270. The number is true and the framing as independent
decisive evidence is not.

### 5. `\bar{\mathcal{F}}` is two different statistics with two different values. VERIFIED
Body `tab:floors` caption: "the mean over the nine semantic families".
Appendix (`:95`, `:163`, `:257`, `:329`): "averaged over the 7 non-control edit
families". Same symbol, so ECoT-bridge reads 0.894 in the body and 0.869 in the
appendix; r=8 0.448 vs 0.407; no-CoT 0.187 vs 0.166. Two of the body's "nine
semantic families" are families the appendix files as Tier-0 controls, one of
them `cross_task_swap`, which is also the paper's maximum-effect ceiling.
The good news, and it should be stated in the paper: the 12/12 sign result holds
under **both** conventions (appendix range -0.028 to -0.179).

### 6. Neither half contains the judge protocol that certifies both floors. VERIFIED
`sec:judge_edits` is deferred. The body asserts at `:153` and in the
`tab:floors` caption that both floors are "certified meaning-preserving by the
same blind judge (0.975, 1.000)", and every headline reduces to that
certification. No judge model, prompt, blinding procedure or agreement
statistic appears in the 20 pages. `arr_appendix.tex:355` claims "no claim in
the body rests on one of them" — false as written.

### 7. Twelve cells of the flagship body table are outside the audit. VERIFIED
`grep` for `0.894`, `0.448`, `0.438`, `0.554` in `verify_paper_numbers.py`
returns nothing: the nine-family `\bar{\mathcal{F}}` column of `tab:floors` is
not among the 1,672 asserted claims, while the paper advertises the audit as
covering what it quotes.

### 8. `Bridge-4k` is a row in the flagship table with no support anywhere. VERIFIED
`grep -c '4k\|deconfound' arr_appendix.tex` = 0. The row carries two claims
("93% of its faithfulness signal is the single gripper bit", "65x smaller
translation magnitude") pointed at "Appendix A", and it is absent from
`tab:models`, which the abstract's "15 VLAs" rests on.

### 9. The style rule the user set is violated twice, in text I added. VERIFIED
`\textemdash` at `arr_appendix.tex:284` (twice in the `fig:paths` caption) and
`:358`. Body prose is clean; the remaining ` -- ` in the body are all inside
comments. Fix belongs in `cot_faith_iclr.tex`, since the appendix is generated.

---

## Tier 2 — verified structural problems

### 10. Ten floats, including Figure 1, are never referenced from the text. VERIFIED
Zero `\ref` in either file: `fig:overview`, `fig:collision`, `fig:dissociation`,
`fig:noise`, `fig:cross_corpus`, `fig:cross_corpus_edit`, `fig:directional`,
`tab:p3`. One reference only: `fig:attention`, `fig:edit_heatmap`, `tab:compare`,
`tab:crossfamily`, `tab:per_task`, `tab:fdirnull`. ACL reviewers name this every
time, and an uncited Figure 1 is a free hit.

### 11. The duplication the user complained about recurs three more times. R3, spot-checked
- `fig:directional` panels (a) and (b) redraw every numeric column of body
  Table 2 as bars, digit for digit. Only panel (c) is new data.
- `tab:leaderboard` (p14) and `fig:edit_heatmap` (p15) share 88 cells, and the
  figure's own caption opens "This is Table 6".
- `tab:crossfamily` (p20) is a strict subset of `tab:calibration` (p17) plus one
  subtraction.
- `fig:cross_corpus` and `fig:cross_corpus_edit`: all 24 bar heights are printed
  as prose in the paragraphs above them, and both floats are uncited.
This is the same standard by which the 12-panel taxonomy figure was cut, so it
should be applied uniformly or a reviewer will notice the inconsistency.

### 12. The body spends 40% of its 8 pages on floats and captions. R2, measured
Floats plus captions ~3.2 of 8 pages; caption prose alone ~1.6 pages, i.e. about
22% of the body's word budget. `fig:overview` 309 words, `fig:frames` 262,
`fig:attention` 403 in the appendix. Effective running text is roughly 4 pages
of 8, which is why the headline claim has no in-body significance test.

### 13. The headline claim has no inferential statistic in the body. R1
`tab:floors` is 12 rows of point estimates with no N, no CI, no p, while §8
reports that retraining moves `\bar{\mathcal{F}}` by up to 0.092 — larger than 8
of the 12 headline effects. The saving evidence exists and is appendix-only: the
per-task paired binomial over 85 LIBERO-90 tasks, p from 9.0e-11 to 0.020 on 7
of 8 models (`arr_appendix.tex:295`).

### 14. Legibility: type below 6pt in five figures. R3, measured against include widths
`fig:directional` VAL 4.8pt (smallest in the paper), `fig:dissociation` 5.0,
`fig:attention` LEG 5.0/TICK 5.3, `fig:noise` LEG 5.0, `fig:edit_heatmap`
CELL 5.4. Mathtext subscripts render at 0.7x base, so `F_dir`/`F_diff` in axis
labels land near 4.2pt. Separately, `fig:edit_heatmap`'s `RdYlBu_r` maps 0.00 and
0.96 to nearly the same greyscale luminance, so the figure's headline gestalt
inverts in a greyscale print, and about ten cells print white on pale fill.

### 15. Missing from both halves. R2
No artifact URL or "available on acceptance" statement; no dataset licensing
table (one sentence at `:369` is the only licensing text in 20 pages); no LLM
usage disclosure although an LLM judge is a method component; no training
compute (only ~3h on 8xA100 for evaluation); no multiple-comparison treatment
across 12 sign tests, 11 ratio tests and 16 binomial tests; no CI on the
headline Spearman rho = 0.476 at n=8.

---

## Tier 3 — flagged, lower impact

- `\mathcal{F}_{\text{dir}}`'s constructive result is confounded with policy
  competence: the six configurations that clear its null at 6.2-11.1x are
  exactly the six the body admits do not beat a constant predictor (`:353`).
  Not fixable today; pre-empt in Limitations, and add the conditional
  `\mathcal{F}_{\text{dir}} | \Delta_\infty > \tau` and the isotropic chance
  rate 0.25, both of which favour the paper.
- The "eight non-directional families" ceiling set as described has no slot for
  `cross_task_swap`, yet `tab:fdirnull` names it on six rows, and names
  `negation` on two rows although `:160` declares `negation` directional.
- "floor" and "ceiling" swap meanings between the abstract and §6 for the same
  quantity, and "ceiling" denotes three different constructs across the paper.
- `\mathcal{F}_{\text{mag}}` is used in the Figure 1 caption on page 1 and
  defined on page 4; `\bar{\mathcal{F}}` used at `:167`, defined in a caption at
  `:216`; `\bar{\mathcal{F}}_{\text{mag}}` never defined.
- Unit slip at `:346`: "0.092: about 3.9x the widest Wilson interval" binds the
  ratio to the wrong number (0.260/0.067 = 3.9; 0.092/0.067 = 1.4).
- `:326` "within 0.002 pp of data-50B" is almost certainly 0.002 absolute.
- Five self-errata inside captions ("an earlier version of this figure drew 11 of
  the 13 and did not say so", x4 floats). Both R3 and R4 judge these as hurting:
  a reviewer who never saw the earlier draft learns only that the figure
  pipeline silently dropped columns. Move to a change log.
- `tab:per_task` and `tab:calibration` disagree on ECoT-bridge (0.860/0.947 vs
  0.869/0.960) one line after the text says all eight reproduce "exactly"; the
  cause is a single-seed vs 3-seed run and is invisible to the reader.
- Appendix promises "five questions (F1-F3, O4, F5)" and delivers F5 and P3.
- Body `:231` promises `P(\Delta_\infty = 0)` "as a first-class column
  (Appendix A)"; the appendix has no such column.

---

## Scores

R1's mock ARR review: **Soundness 2.5/5, Excitement 4/5.** The measurements are
real, audited and unusually well controlled; the inference layer is where the
attack surface is. R1's single highest-lift change: rebuild §4 around the
appendix's per-task sign test, fix the count to one honest number, and drop the
"decisive" framing.

---

## Verdict on the two review questions asked

**Is anything in the appendix that belongs in the body?** Yes, ranked:
(1) the judge protocol certifying both floors (currently in neither half),
(2) the per-task sign test, which supplies the significance the headline lacks,
(3) the out-of-CoT specificity ratio, which the body introduces and never
scores.

**Is anything in the body that belongs in the appendix?** Yes: `fig:frames`
(44% of page 4 for an exhibit whose own caption says "This is not a
faithfulness measurement"), `fig:noise` (three numbers already in prose and
inside another figure's annotation), and the AUROC paragraph at `:334`, which
disclaims itself in four clauses and whose table is already in the appendix.
R3 proposes the specific swap `fig:frames` (body) <-> `fig:paths` (appendix) on
the grounds that the rollout claim is visible at print size in the line plot and
invisible in 18 thumbnails; that also pulls Table 1 back next to the §4 argument
that cites it, and empties the float-only page 20.

Free space measured from the PDF: **body 76pt (0.055 page), appendix 8pt**. Both
halves are saturated, so every promotion must be paid for. Cutting the four
uncited/duplicate appendix floats frees about 2.5 pages, which is more than
enough to pay for the judge section, the per-task promotion and a populated A.5.

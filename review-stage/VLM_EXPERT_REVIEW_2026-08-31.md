# VLA/VLM expert ARR review — CoT-Faith submission, 2026-08-31

Requested via `/research-review`: "suppose you are a reviewer who is an expert in
vision language model, read this arr submission, point out things can be
improved, weakness points, experiments we can add, the goal is we want the
paper to be accepted."

## Method note (read this before the findings below)

The skill's default reviewer backend is Codex MCP (`gpt-5.5`, xhigh reasoning).
Codex was **unreachable for this entire session**: both the MCP tool call and a
raw `curl` to `chatgpt.com/backend-api/codex/responses` were refused instantly
(HTTP 000, connection refused in <1ms — not a timeout), despite `codex login
status` showing a valid ChatGPT login. This looks like a network-level block
(the session also has several Apple-internal MCP servers configured, so this
network may be a corporate/VPN network that does not route to `chatgpt.com`).
Given the choice, the user asked to route the review through **Claude Opus 5**
agents instead of Codex, and through myself (Sonnet 5, orchestrating) rather
than through Oracle MCP (not installed in this environment).

This is a same-model-family review (Opus 5 / Sonnet 5, both Anthropic), not the
cross-model review the skill is designed around, and that matters for how much
independence it buys. To partially compensate:

1. **Round 1 and Round 2** were a single fresh Opus 5 `general-purpose` agent
   with *zero* prior context — it read the submission files itself (body,
   appendix, bibliography, datasheet, figure generators, compiled PDF, and ran
   `scripts/verify_paper_numbers.py` itself) rather than being handed any
   summary or interpretation from me, per this project's reviewer-independence
   protocol (`~/.claude/skills/shared-references/reviewer-independence.md`).
2. **A second, separate fresh Opus 5 agent** was launched with no visibility
   into Round 1/2's output, and tasked *only* with independently re-deriving
   the two most severe numeric claims from the raw JSON artifacts — deliberately
   withheld from seeing the first agent's numbers, so the two analyses could not
   anchor on each other.
3. **I independently confirmed the single most consequential factual claim**
   (that the "model's own CoT" is actually a teacher-forced ground-truth
   annotation, not a self-generated decode) myself, by reading
   `experiments/cotfaith_edit.py` directly — confirmed: the script's own header
   comment says "GT CoT, one with edited CoT," and `build_ecot_target_text()`
   constructs the CoT text from the `libero_reasonings.json` annotation fields.

The two Opus-5 agents converged tightly on the numbers without having seen each
other's work (e.g. both independently found bootstrap significance on the
`syntactic_scramble` floor comparison at **2 of 11–12 configurations**, computed
by completely different methods). That convergence is the strongest evidence
available here that the findings below are real rather than model artifacts,
short of an actual independent cross-model (e.g. GPT) review, which should
still be run once Codex/Oracle connectivity is available.

---

## Executive synthesis — cross-validated confidence

| Finding | Status | Severity |
|---|---|---|
| **The headline "sign flips on all 12 configurations" is convention-dependent.** Under the body's own 9-family convention it's 12/12; under the *appendix's own* 7-family ("core-7") convention it's **9–10/12**, and only **2–3/12** are statistically distinguishable from zero by bootstrap. `r=8` and `ECoT-bridge` flip sign under core-7. The paper's own caption asserts the convention swap "leaves the 12/12 result... intact" — it does not. | **Confirmed independently by 2 agents**, different methods, no shared context | **Fatal as currently stated; fixable by honest restatement** |
| **The headline is not robust to the faithfulness threshold τ.** Sign count degrades from 12/12 at τ=0.05–0.10 to 6–9/12 by τ=0.20–0.30. The paper's own threshold-sweep artifact sweeps only F̄'s rank ordering, never the sign this claim depends on. | **Confirmed independently by 2 agents** | **Fatal as currently stated; fixable** |
| **What gets edited on every LIBERO measurement (i.e. the entire headline) is a teacher-forced ground-truth dataset annotation, not any model's self-generated CoT**, despite the body and Figure 1 caption both saying "the model's own CoT." Self-generation happens only on the 3 cross-corpus runs, where the calibration nulls were never run. | **Confirmed by independent code read** (`experiments/cotfaith_edit.py`) | **Fatal as currently stated; fixable, and cheaply — the self-generation machinery already exists** |
| **7 of the 12 headline configurations have near-degenerate action spaces** (roll/pitch frozen at one bin on 6 of 7 LoRAs; 21–85% of actions are the exact no-op; 38–117 distinct actions over ~100–300 observations). On those rows, `F_dir`'s cosine criterion degenerates to a sign test on one active axis, inflating its apparent "clearance." | **Confirmed and deepened by independent re-derivation** with permutation nulls | **Severe; the proposed replacement metric is not comparable across the cohort as currently presented, but a real (smaller) signal survives** |
| **New finding, not in Round 1:** two of the paper's own tables (`tab:directional` vs `tab:fdirnull`) compute `F_dir` on **two different de-quantization grids**, producing different numbers for the same quantity (e.g. `r=8`: 0.649 vs 0.666). The caption's stated explanation for the discrepancy ("pooled vs. seed-mean") is shown to be arithmetically impossible — both aggregations agree to 4 decimals. | **Found and verified in the independent pass only** | **Moderate-severe; easy to fix once located, easy to miss otherwise** |
| Judge-validation pairs are truncated to 400 characters; 39/40 `syntactic_scramble` pairs in the release are byte-identical, so the one certification the entire sign-flip claim rests on cannot be checked from the release, contrary to the appendix's and datasheet's "verbatim... both traces." | Found by Round-1 agent only, with file:line specificity; not independently re-derived | Severe if accurate — should be spot-checked before acting, but the specificity (exact file, exact truncation length, exact counts) argues for high reliability |
| `syntactic_scramble`'s own judged fluency is 4.00 (vs. 4.92/5.00 for the two other nulls) and is never printed; F correlates with judged fluency at ρ=−0.60 across families — leaves "the metric tracks meaning" vs. "the metric tracks fluency" undecided | Round-1 agent only | Severe — directly actionable, cheap to check (rerun the judge report) |
| No successful rollouts (0/40 both arms); the validated rollout harness reaches 0.74–0.90 SR on *public* LIBERO checkpoints, so the 0/40 looks like a training-budget problem with the authors' own from-scratch LIBERO-90 LoRAs, not a harness or compute problem | Round-1 agent only | Severe for an embodied-AI reviewer, but well-supported by the datasheet's own numbers |
| Presentation: abstract cites a 5.2× "causal effect span" driven by one saturated OOD checkpoint over an action-only ablation (1.46× among comparable LoRAs); cites Spearman ρ=0.48 in the abstract when §5 says n=8 can't estimate it (CI spans 0); ~18% of body word count is caption text; confessional register repeatedly references a prior submitted version the reviewer never saw | Round-1 agent only | Moderate — free fixes, no new compute |

**Net effect on a mock ARR score (Round-1 agent's estimate, informed by the independent-verification pass):** Overall **2.5/5**, Soundness **2.5/5**, Excitement **3.0/5**, Confidence **4/5** — driven almost entirely by the convention-dependence and non-robustness of the headline claim, not by the paper's engineering quality, which both agents independently called exceptional (offline-reproducible audit script that actually passes: 1722/1722; datasheet documents its own failed attempts and retracted figures; license resolution from the HF Hub API rather than memory).

### Reconciliation with the existing `review-stage/OPUS5_PAPER_REVIEW.md` (2026-08-03)

No contradiction. That review's item #5 said the 12/12 sign result "holds under both conventions" — but it was checking only the **paraphrase**-floor side (appendix range −0.028 to −0.180), which both new independent analyses confirm is indeed robust (12/12 negative under both conventions, sign-count-robust to τ up to ~0.10). Nobody had previously checked whether the **`syntactic_scramble`**-floor side — the "12/12 positive, sign flips entirely" half of the headline — also survives the same convention swap. It does not (9–10/12, significant on 2–3/12). This is new information, not a re-litigation of settled ground, and the current `cot_faith_arr.tex`/`arr_appendix.tex` text does not state it anywhere (the `tab:floors` caption checks and reports the convention-swap range for the paraphrase side only, then asserts without checking that the scramble side is "intact").

---

## Recommended action plan (condensed from Round 2's full staged plan below)

| Stage | Cost | What it buys |
|---|---|---|
| **0 — Reanalysis & disclosure** | 0 GPU-h, ~1 eng-week | Mandatory, first. Canonicalize F̄ to one convention in both halves; add bootstrap CIs to Table 1; sweep τ over both floors; disclose action-space diagnostics per row; disclose `syntactic_scramble`'s fluency; release untruncated judge pairs; drop the unsupported `Bridge-4k` row; fix the abstract's 5.2×/ρ=0.48; add the missing citations (Jacovi & Goldberg 2020, ERASER, ROAR, Krishna et al., prompt-sensitivity literature). **This alone is deflationary — it turns "12/12" into the honest "9–10/12, significant on 2–3/12."** |
| **1 — Continuous effect size + own-likelihood covariate** (fused) | ~5 A100-h (~40 min on 8×A100) | The agent's revised top pick. Replaces the thresholded, argmax-collision-prone Δ∞ with a continuous, tokenizer-independent distributional measure, and regresses per-sample effect on the policy's own log-likelihood of the edited CoT. This is the only proposed experiment that can **restore statistical power** to the claim Stage 0 is forced to weaken — every outcome branch is an improvement or neutral, none is worse. |
| **2 — Self-generated CoT rerun** (3 checkpoints, + 3-timestep add-on) | ~9 A100-h (~1.2 h on 8×A100) | Fixes the teacher-forced-annotation construct-validity gap (the single most fundamental objection). Machinery already exists (`rollout_probe_ecot_bridge`). |
| **3 — Gate: competent CoT policy on `libero_spatial`** | ~80 A100-h (8–12 h on 8×A100) | Tests whether 0/40 rollouts is a training-budget artifact (current recipe: ~30k samples) rather than a fundamental limitation; gated — **stop here if SR < 0.4** and report the negative training result instead of burning Stage 4's budget. |
| **4 — Paired ΔSR rollout** (only if Stage 3 passes) | ~29 A100-h (3.6 h on 8×A100, using 10-step CoT-regen caching) | The only experiment that connects edit-sensitivity to actual behavior — closes the paper's largest "so what" gap for an embodied-safety audience. |
| **5 — Text-only replication** (optional) | ~3 A100-h | Tests whether the two-floors pathology reproduces outside action space; if yes, reframes the contribution from "a VLA benchmark problem" to "a general CoT-faithfulness metrology problem," which is a better fit for an ACL venue. |

Total ≈ 126 A100-h beyond Stage 0 — on the same order as compute already spent on
the existing 14 LoRA fine-tune runs, using the Bolt cluster access and job
configs already in this repo.

**If only one more round of compute is available: Stage 1** (not Stage 2, despite Round 1 calling self-generated-CoT its top recommendation) — because Stage 0's honest restatement leaves a statistically thin claim, and Stage 1 is the only candidate that can put power back into it; Stage 2 would just re-measure the same low-power thresholded statistic on different text. Full reasoning for this reversal is in Round 2's report below.

---

## What I have not independently verified

To calibrate trust correctly: **Round 1's findings on the judge-pair truncation (W5), the fluency confound (W6), the rollout/competence gap (W7), and all of the presentation/citation critiques were reported by one agent and not independently re-derived** by the verification pass (which was scoped narrowly to the two statistical claims judged most consequential) or by me. They come with specific file paths, line numbers, and exact figures, which is a good reliability signal, but "specific" is not the same as "independently confirmed" — treat them as high-confidence leads to spot-check before rewriting text around them, the same way you would a single sharp human reviewer's comment.

---

## Full reports, verbatim

### Round 1 — initial review (Opus 5, fresh context, read all submission files itself, ran the audit script itself)

<details><summary>Round 1 full text</summary>

**What I did.** Read `cot_faith_arr.tex`, `arr_appendix.tex`, `arr_bib.tex`, `DATASHEET.md`, all five submitted figure generators, and the 20-page `build/cot_faith_arr.pdf`. Ran `scripts/verify_paper_numbers.py` (**1722/1722 PASS**, exit 0). Then I did what this paper's own release strategy invites a reviewer to do: I recomputed the headline claims from `results_v2/canonical_runs/*.json` myself. That is where the problems are. Several load-bearing claims do not survive recomputation under the paper's own alternative conventions. Everything numeric below is something I computed from the released records, and I give the recipe so it can be reproduced.

**Bottom line up front.** This is the most rigorously *engineered* submission I have reviewed — the datasheet, the four-times-failed rollout gate, the retracted panels, the license resolution, the 1,722-claim audit are all genuinely exemplary. But the audit checks that the paper faithfully reports its own JSON; it does not check that the JSON licenses the claim. On that second question I found five things a determined reviewer will find too, at least two of which are, as written, fatal to the headline. They are all fixable, mostly without new compute.

#### 1. Prioritized weaknesses

**W1 — The headline "the sign flips on all 12 configurations" is an artifact of the F̄ convention. It is 9/11 under the paper's own appendix convention, and statistically significant on 2 of 11. [FATAL as written; highest priority]**

The body's Table 1 defines F̄ as "the mean over the nine semantic families, the *cross_task_swap* ceiling and the out-of-CoT *instr_random_sub* control among them." The appendix (`tab:calibration`, `tab:per_task`, and the `tab:leaderboard` caption) defines F̄ as "the mean over the seven non-control families." The body averages the designated maximum-effect ceiling family and an explicitly-labelled control into the treatment mean, and that is what produces the 12/12.

Recomputing from the released records (3-seed pooled for the seven `ours` rows, single run for ECoT-bridge and the three DeepThinkVLA checkpoints; 11 configurations, since Bridge-4k has no calibration file locatable):

| convention | negative vs `paraphrase_null` | positive vs `syntactic_scramble` |
|---|---|---|
| body's 9-family F̄ | 11/11 | 11/11 |
| appendix's core-7 F̄ | 11/11 | **9/11** |

The two failures under core-7 are `r=8` (−0.002) and `ECoT-bridge` (−0.011, i.e. it flips back to negative). And the magnitudes are tiny: under core-7, five configurations have |Δ| ≤ 0.012 — an order of magnitude below the 0.092 that §7 says retraining one recipe moves F̄.

A paired bootstrap (3,000 draws, resampling scored records within family) on the core-7 gap gives: **significant vs paraphrase 9/11; significant vs syntactic_scramble 2/11.** The sign-flip claim is statistically supported on two configurations, both from one third-party release. Table 1's caption says only that "the 12/12 result **below** [paraphrase side] intact" — it never states the scramble side is 9/11 under core-7.

*Fix.* Pick one F̄ (core-7; never average a ceiling and a control into a treatment mean) and use it in both halves; put bootstrap CIs in Table 1; restate the abstract to: "F̄ is below the paraphrase floor on 11/11, significantly on 9/11... and reverses against a length-exact floor on 9/11, significantly on 2" — which still supports "neither floor licenses a claim about meaning." Needle: large. As written this is a reject-level overclaim; restated honestly it is still publishable.

**W2 — The headline is not robust to τ either, and the paper claims τ-insensitivity as a feature. [FATAL as written]**

At τ=0.20 the core-7 floor gap sign count is neg-vs-paraphrase 9/11 (was 11/11), pos-vs-scramble 9/11; at τ=0.30, 8/11 and 9/11. `DT-SFT` goes to +0.070 and `DT-RL` to +0.020 on the paraphrase side at τ=0.20 — the paper's strongest claim reverses on both non-degenerate independent checkpoints at a threshold four bins wider. The released `threshold_sweep.json` sweeps τ but only over the 7 non-control families — it never sweeps the floors, so this analysis is not in the release.

*Fix.* Extend the existing sweep to both floors; report sign count vs τ in the body; justify τ=0.05 on decode-grid grounds. Needle: large — a reviewer who runs this and isn't told will conclude τ was chosen to make the result work.

**W3 — What is edited is not "the model's own CoT." It is a teacher-forced third-party dataset annotation. [FATAL to the construct as stated]**

`experiments/cotfaith_edit.py` takes `--reasoning-json libero_reasonings.json`; `judge_edit_families/args.json` confirms `reasoning_repo: Embodied-CoT/embodied_features_and_demos_libero`. So on LIBERO — every headline number — the "CoT" is the Embodied-CoT ground-truth annotation, injected into the prompt. Self-generation happens only on the three cross-corpus runs, and the nulls were never run there. The body says "we decode greedily twice (once on the model's own CoT...)" and Figure 1's caption says "the model's own CoT rewritten in one span." Both are inaccurate. A policy conditioned on reasoning text it did not produce is OOD-sensitive to any perturbation of it — which predicts the central finding.

*Fix, and it is cheap.* `results_v2/canonical_runs/rollout_probe_ecot_bridge/rollout_edit_probe.json` already proves the machinery works (`cot_structured: true`, 8 tags parsed, 255 tokens, 8.57s) and already applies 3 families to a self-generated trace. Run the full 13-family protocol on self-generated LIBERO CoT for ECoT-bridge and the LoRAs. ~2 GPU-hours of generation. Needle: very large — my #5 recommendation, and I would make it a condition of acceptance.

**W4 — Seven of the twelve headline configurations have a collapsed action space, and they are exactly the seven that carry the constructive result. [Severe]**

Inspecting `a_orig` in every scored record: on all seven LoRAs, 70–97% of predicted translations have at most one of three axes off the centre bin — one-hot-like, drawn from a codebook of 38–53 distinct vectors over ~100 observations, 26–47% (85% for no-CoT) the exact no-op. `ECoT-bridge` and all three DeepThinkVLA checkpoints: 0% single-axis, 89–100 distinct actions.

(a) F_dir's null is validated only on collapsed policies, and F_dir is maximised by a policy that ignores the image: on a one-hot action the cosine criterion degenerates to a sign test on one scalar. Conditional on the action moving at all, the six LoRAs reverse 88–95% of the time and all four independent checkpoints reverse 0–16%. A metric that reads ~0.9 on a CoT-parroting lookup table and ~0.1 on a competent policy cannot be recommended as an admission rule.

(b) The collision result's magnitude is a property of the LoRAs: P(Δ=0) is 0.46–0.83 on LoRAs vs 0.076–0.166 on independent checkpoints. The relation itself survives (R²=0.938 restricted to independent checkpoints; 0.975 pooled), so §4's identity is sound — only the "46.2%" headline is cohort-specific.

(c) The effective independent evidence base is two configurations, once the LoRA clones, the excluded Bridge-4k row, and the two `degen.`-marked rows are netted out.

*Fix.* Report the action-space diagnostic as a first-class column and gate the cohort on it; state the count honestly. Needle: large.

**W5 — The released judge pairs do not contain the edit for the three families that matter most, and the paper says they do verbatim. [Severe]**

`experiments/cotfaith_judge_edits.py:501` truncates every stored pair to 400 characters. Result: `direction_flip` 40/40 byte-identical heads, `negation` 39/40, `syntactic_scramble` 39/40, `paraphrase_null`/`bbox_jitter_null` 0/40. For `syntactic_scramble` — the floor whose 1.000 preserved rate flips the sign of the entire headline — the edit is not in the release at all. The judge saw full traces, so the validation itself is fine, but the appendix's "all 437 judged pairs with both traces" and the datasheet's "verbatim" are inaccurate, and a reviewer cannot check the one verdict everything rests on. Also: the "0/40 word-count change" length-exactness evidence is computed on these truncated heads, so for three of four families it is vacuous (true by construction, not by the cited measurement).

*Fix.* Store full traces (or a window around the diff); add an audit assertion that non-identity families' pairs differ. Needle: medium-large.

**W6 — `syntactic_scramble` is measurably dysfluent, its fluency is the one number the paper does not print, and dysfluency is an unexcluded alternative explanation for the sign flip. [Severe]**

`judge_report.json` mean_fluency: identity_control/bbox_jitter_null 5.00, paraphrase_null 4.92, **syntactic_scramble 4.00**, semantic families 2.58–3.66. The paper prints the first two fluencies and omits scramble's. Its `same_direction_rate` is 0.925, not 1.000. Across families F correlates with fluency at ρ=−0.601 (n=12). The two-floors result is currently ambiguous between "the metric cannot see meaning" and "the metric tracks surface fluency," and the paper cannot currently distinguish them.

*Fix, and this is the experiment I most want.* Regress per-sample/family effect on the policy's own Δ log-likelihood of the edited trace. Needle: large — converts a critique into a mechanism.

**W7 — Zero successful rollouts, and the attribution in Limitations is weaker than the authors' own artifacts support. [Severe for an embodied-AI reviewer]**

Limitations attributes 0/40 to "a checkpoint-competence problem." The datasheet shows the harness reaches 0.74/0.90/0.74/0.46 on public LIBERO-suite checkpoints against published 0.844/0.881/0.794/0.539 — the harness works on competent checkpoints. Why fine-tune on LIBERO-90 (no published checkpoint, no gate) instead of a suite with a public gate? 15,000 steps at batch size 2 (~30k samples) is not a serious adaptation budget for a 7B model on a new embodiment.

*Fix.* LoRA-tune on `libero_spatial` to nontrivial SR, report SR beside every F, run paired ΔSR. Needle: very large.

**W8 — F_dir's "N=299 vs N=100" makes the null test non-comparable across architecture families. [Moderate]**
Table 3's ceiling is a max over 8 families; the `ours` rows pool 3 seeds (N=299), DeepThinkVLA rows are N=100, so the second family's ceiling is more inflated (biasing the ratio *against* it, so the conclusion survives, but the ratio itself is not a comparable quantity).

**W9 — Figure 4(c) shows the CoT-stripped ablation as the second most faithful model, and the caption does not say so. [Moderate]**
Normalizing by each model's own ceiling, `no-CoT` reads 0.637 — above every CoT-trained LoRA (0.48–0.58). The generator's own docstring is candid about this; the printed caption is not.

**W10 — Figures 3/4: no error bars, 2-decimal value labels, most adjacent gaps inside the paper's own retraining bar. [Moderate]**
Panel (a) refuses this on principle; panels (b)/(c) do it anyway, where 5/7 and 6/7 adjacent gaps are smaller than the worst same-config retraining movement (0.0924).

**W11 — The abstract's two most quotable numbers are carried by pathological endpoints. [Moderate]**
"5.2×" = ECoT-bridge/no-CoT (saturated OOD checkpoint over an action-only ablation); among six comparable LoRAs it's 1.46×. "Spearman ρ=0.48" is in the abstract; §5 itself says n=8 can't estimate it (CI spans 0).

**W12 — The attention "pinning" is largely a token-share effect, and the correct null is in an appendix caption only. [Moderate]**
CoT tokens are ~47% of the prompt but draw only 0.34 attention share (ratio 0.73); the appendix's own panel (c) already shows this per-token effect, but the body's "pinned by architecture" claim doesn't carry the length null.

**W13 — "Resampling the decode" does not exist in this protocol. [Moderate, factual]**
The decode is greedy; the "3 seeds" reshuffle *sample-set* order, not the decode. §7's framing of three variance sources is really two.

**W14 — Bridge-4k is a headline row the datasheet says nothing depends on, built on a join the datasheet says cannot be done. [Moderate]**
Not in `tab:models`; `bridge_join_probe` shows the join succeeds on 2.1% of episode IDs and the rendered trace leaves two tags blank. *Fix: drop the row.*

**W15 — Stale numbers in the released datasheet. [Moderate]**
DATASHEET says 787 audited claims (paper says 1,722); 54.1MB (paper says 53.9MB); "(N=30)" cross-corpus in the license table (paper claims N=100).

**W16 — Writing density and caption budget. [Moderate-large on Excitement]**
18% of body words are caption text; Figure 1's caption is 290 words, Table 1's 237 (three times the height of its own table); abstract is 305 words with 12 numeric claims; confessional register ("We did not test this before submission," "the bar we submitted") repeatedly references a prior version the reviewer never saw.

**W17 — Terminology: "floor" and "ceiling" denote the same object in adjacent sentences. [Small]**

**W18 — Layout/formatting. [Small]** 20 pages exactly (zero headroom); ~432pt of unused body slack elsewhere; three unsignposted body→appendix pointers; Table 5 renders at 5.7pt; `figures/per_scene_data.json` is orphaned, invalid JSON, and contradicts the leaderboard table; one unaudited number (`R²=0.972` at line 241) I could not reproduce under 5 plausible definitions.

**W19 — Figure 2(a)'s "empty neighbourhood around τ" is a binning artifact; Figure 2(b)'s R² is not what the drawn line implies. [Moderate]** Panel (a)'s bins are unequal-width; in density terms there's no valley at τ. Panel (b)'s R²=0.926 is squared Pearson r next to a drawn y=x line; R² about the identity itself is 0.893, and it's a one-sided bound (0/324 cells above the line) not a two-sided identity.

#### 2. Missing/underpowered experiments

**2.1 Mechanism experiment (cheapest, highest value):** regress per-sample effect on edit-distance/length-delta/policy's-own-likelihood of the edited trace. Not currently answerable from the release (see W5).
**2.2 Continuous, tokenizer-independent effect measure:** replace Δ∞'s argmax-collision-prone bins with per-dimension TV/KL between action-token softmaxes.
**2.3 Self-generated CoT with the nulls (see W3, top recommendation).**
**2.4 Sample beyond t=0:** every number is first-step; free from data already on disk to extend to mid-trajectory steps.
**2.5 Rollout ΔSR on a competent CoT policy (see W7, most expensive, most valuable).**
**2.6 More independent CoT-VLA families:** `CoT-VLA` and `ECoT-lite` are cited but not evaluated.
**2.7 A protocol-comparison table:** the abstract's "every benchmark... scores by magnitude" universal rests on 3 citations; a small table would prove or soften it.
**2.8 Judge robustness:** one judge, n=40/family, 17.6% order-flip rate, and the entire headline turns on one family's 1.000. Add a second judge and/or human annotation.
**2.9 Statistics:** replace the tie-dropping per-task sign test with a mixed-effects model; add multiplicity correction across the enlarged claim set.

#### 3. Framing, novelty, related work

**3.1 Venue fit.** This is a robotics paper with an NLP hook — no LM experiments, evaluation entirely on 7-DoF action vectors, Related Work entirely in the appendix. The single highest-leverage fix for venue fit: replicate the two-floors pathology in a text-only CoT-faithfulness setting (see §2.5 experiment package below) — if it reproduces, the claim becomes general CoT-faithfulness metrology rather than VLA-specific, squarely in ACL scope.

**3.2 Missing related work — harshest likely reviewer criticism.** Absent and directly on-thesis: **Jacovi & Goldberg (2020)** "Towards Faithfully Interpretable NLP Systems" (the canonical faithfulness-desiderata paper — its absence from a construct-validity-of-faithfulness paper is the single most conspicuous gap); **DeYoung et al. (2020) ERASER** (comprehensiveness/sufficiency already pair with a random-rationale null); **Hooker et al. (2019) ROAR** (retrain-and-evaluate calibration, direct methodological ancestor of the floors); **Krishna et al. (2022)** "The Disagreement Problem in Explainability" and **Chan et al. (2022)** (both name the exact "two floors disagree with each other" phenomenon); prompt-sensitivity literature (**Lu et al. 2022, Sclar et al. 2024, Mizrahi et al. 2024**) — the central finding ("a paraphrase moves the output more than a semantic edit") is that literature's result transported to action space, and not citing it makes the result look more novel than it is; **Geiger et al.** causal-abstraction/interchange-intervention framework (Δ∞ *is* an interchange intervention read out in action space); action-tokenizer-artifact literature (FAST/π0-FAST) for the collision-counter finding.

**3.3 Novelty positioning.** Strongest available framing: *"perturbation-based CoT-faithfulness metrics, in the one modality where effect size is directly measurable, are dominated by surface-form sensitivity, and here is the calibration protocol that reveals it"* — a positive methodological contribution (the two-floor + out-of-CoT calibration triple, genuinely novel and reusable), with the negative result as evidence rather than the headline.

**3.4 Identity crisis.** Simultaneously a benchmark paper, a negative-result paper, and a position paper (the admission rule rejects all 8 of the authors' own submissions, so it can't be used as a benchmark). Recommend retitling away from "A Benchmark for" toward the measurement claim.

**3.5 Buried lede.** The `instr_random_sub` result ("no configuration clears its own out-of-CoT control by more than its own retraining noise") is cleaner, more damaging, and doesn't depend on the LLM judge at all — consider leading with it.

#### 4. Mock ARR review

**Summary:** [paraphrased above in the synthesis; full summary matches the paper's own abstract structure]

**Strengths:** (1) The question and framing are right — interrogating the measurement rather than adding another leaderboard. (2) Artifact discipline is the best of any paper in this area seen — offline-reproducible audit passes 1722/1722; datasheet documents 4 failed rollout-gate attempts and a retracted attention panel with root cause; upstream licenses resolved from the HF Hub API, not memory. (3) The calibration design (two independent meaning-preserving floors + an out-of-CoT specificity control) is a genuinely novel, reusable contribution. (4) Several self-critiques are unusually sharp and correct (τ-insensitivity-as-consequence-not-robustness; same-recipe retraining as the real error bar). (5) Negative results reported against interest throughout.

**Weaknesses:** the 10 numbered items map directly to W1–W7 and W9–W11 above (see the synthesis table for the condensed version); full text repeats each with its exact numbers.

**Questions for Authors:** 14 questions, each pinned to a specific file/line/number — matching W1 through W14 above, phrased as direct questions (e.g. "Under the appendix's core-7 F̄ convention I compute the syntactic_scramble sign as 9/11, not 12/12... Which convention is canonical, and does the abstract's claim hold under it with CIs?").

**Scores:**

| Axis | Score | Justification |
|---|---|---|
| Overall Assessment | 2.5 | Question, design and artifact discipline are publishable; headline as stated doesn't survive recomputation under the paper's own alternative convention/threshold; fixable in one revision cycle. |
| Soundness | 2.5 | Every number reproduces from the release (1722/1722), but reproducibility ≠ validity: the load-bearing "12/12 both directions" is 9/11 (2/11 significant) under the appendix's own convention, the edited CoT is a teacher-forced annotation not any model's output, and the constructive metric is validated only on collapsed-action-space checkpoints. |
| Excitement | 3.0 | Well-aimed at a premise the field is building safety machinery on; the calibration triple is reusable; but no behavioral evidence, only 2 independent CoT-VLAs after netting out clones/saturated rows, and no instrument a reader can pick up and use as-is. |
| Reviewer Confidence | 4 | Read everything, ran the audit, independently recomputed sign counts/bootstrap CIs/threshold curves/action-space stats/collision decomposition from raw JSON; did not run any model myself. |

#### 5. Single highest-leverage recommendation (Round 1)

**Run the full 13-family protocol — both floors included — on the model's own self-generated CoT, and make that the paper's primary result.** Repairs the paper's own construct validity (fatal to leave open in a construct-validity paper); the machinery already exists and costs ~2 GPU-hours; it's the only change that plausibly *rescues* the headline rather than merely qualifying it, and either outcome (reproduces / doesn't) is a publishable paper; composes with the other fixes (enables the likelihood-regression mechanism experiment for free). *(Revised in Round 2 below — the agent changed this recommendation once asked to rank by compute-adjusted acceptance lift.)*

</details>

### Round 2 — prioritized, compute-aware experiment package (same agent/thread, follow-up)

<details><summary>Round 2 full text</summary>

#### 0. Cost model
Anchors: full 13-family sweep ≈ 3h on 8×A100 = 24 A100-h across ~11 checkpoints (~2.2 A100-h/checkpoint, generation-based); LoRA fine-tune at bs=2/15k steps ≈ 10 A100-h; CoT self-generation ≈ 0.24 A100-h/checkpoint per 100 samples (`cot_gen_seconds: 8.57` from `rollout_probe_ecot_bridge`); teacher-forced-only pass (no generation) ≈ 0.5–0.7 A100-h/checkpoint, 3–5× cheaper than generation and batches well.

The full package below is two short 8×A100 blocks (~1h, ~2.5h), one 8–12h block, and one ~4h block — total ≈190 A100-h including Stage 0, against ~165 A100-h already sunk in the 14 existing fine-tunes. **You are proposing to spend roughly what you have already spent, on none of it being more LoRA rank ablations.**

#### Stage 0 — Reanalysis and disclosure (0 GPU-h, ~4–6 engineer-days). Not optional, must come first.
Canonicalize F̄ to core-7 in both halves with bootstrap CIs; extend the threshold sweep to both floors; add the action-space diagnostic and `F_dir | action moved` columns; print scramble's fluency/`same_direction_rate`; release untruncated judge traces + an audit assertion that non-identity pairs differ; drop Bridge-4k; fix the abstract (ρ, 5.2×); fix "resampling the decode," the stale datasheet numbers, floor/ceiling terminology, the unaudited R²=0.972, the unsignposted cross-references; enlarge Table 5; replace the tie-dropping sign test with a mixed-effects model; add the missing citations and a condensed Related Work in the body; retitle.

**Score effect of Stage 0 alone: Soundness 2.5→3.0, Overall 2.5→3.0 — but it is deflationary**, weakening the headline to "9/11, significant on 2/11." Everything after Stage 0 exists to buy back the strength given up by being honest.

#### Stage 1 — Threshold-free effect size + own-likelihood covariate (fused §2.1+§2.2)
Per-dimension TV/KL between action-token softmaxes (no argmax, no τ) + Δ log-p of the edited CoT under the policy's own LM head, as a byproduct of the same forward pass, on 9 families × 6 checkpoints (4 non-collapsed + 2 representative LoRAs) = 5,400 records. Fold in a second judge model on the existing 437 pairs.

**Compute:** ~5 A100-h (~40 min on 8×A100) — the cheapest job on the list.

**Results-to-claims matrix:**
| Outcome | Interpretation | Effect |
|---|---|---|
| Reproduces, likelihood not dominant (partial R²<0.15) | Headline survives on a threshold-free instrument, not a fluency artifact | Soundness/Overall 3.0→3.5 |
| Reproduces, likelihood dominant (R²>0.4) | "F is a surface-likelihood-perturbation detector" — a mechanism, indicts prior work more cleanly | 3.0→3.5, Excitement 3.0→3.5. **Best available outcome.** |
| Doesn't reproduce | Published magnitude gaps are largely quantization artifacts — painful (full rewrite) but stronger/more novel | Soundness 3.0 (new claim well-supported) |
| Inconclusive | Fall back to the honest-restatement paper | 3.0, neutral |

**No outcome makes the paper worse; only "inconclusive" is neutral.** This asymmetry, at 5 A100-h, is why the agent moved this to the top pick.

#### Stage 2 — Self-generated CoT, scored with both instruments
13 families on self-generated LIBERO CoT for 3 checkpoints (ECoT-bridge, r=32, no-CoT), scored with both Δ∞ and the Stage-1 continuous measure; +3-timestep add-on (t=0, T/3, 2T/3) on the 3 families carrying the headline. **Compute:** ~9 A100-h (~1.2h on 8×A100) +~2 A100-h for the timestep add-on. Excludes DeepThinkVLA (own `<think>` decode path; given the datasheet's own six-wrong-conventions history, scope this out rather than risk a week debugging it).

**Results-to-claims matrix:**
| Outcome | Interpretation | Effect |
|---|---|---|
| Reproduces on self-generated CoT | Construct is now the right one — strongest form of the claim | →4.0 |
| Weakens/vanishes | "The pathology is an OOD-text-conditioning artifact of teacher-forced protocols" — indicts the 3 cited precedents, novel, prescriptive | 3.5 (well-identified new claim) |
| ECoT-bridge de-saturates | Recovers a currently-`degen.`-marked configuration | +0.25 |

#### Stage 3 (gate) — competent CoT policy on `libero_spatial`
One LoRA at ~8× current token budget (~30k→~240k samples), effective batch 16. **Compute:** ~80 A100-h (8–12h on 8×A100). **Hard gate: if SR<0.4, stop — do not run Stage 4.** Publish the negative training result plus the action-space diagnostic as the honest finding instead.

#### Stage 4 — paired ΔSR rollout (only if Stage 3 passes)
4 arms (clean/direction_flip/paraphrase_null/no-CoT) × 10 tasks × 10 episodes, with CoT regenerated every K=10 steps (not every step, which would cost ~285 A100-h) — reduces cost to ~29 A100-h (3.6h on 8×A100); reduced-frequency regeneration is standard practice and is `ecotlite`'s own premise.

**Results-to-claims matrix:**
| Outcome | Interpretation | Effect |
|---|---|---|
| SR<0.4 (Stage 3 fails) | Structural cohort weakness, not a budget artifact — disclose, stop | unchanged |
| ΔSR(direction_flip) ≫ ΔSR(paraphrase) | CoT is behaviorally load-bearing; the first-step metric is what's broken — headline result | →4.0, main conference |
| ΔSR(paraphrase) ≈ ΔSR(direction_flip), both large | Construct-validity failure reaches behavior — strongest possible version of the paper's own claim, directly safety-relevant | →4.0–4.5, Excitement→4.0 |
| Both ≈0 on a competent policy | CoT is decorative at the actuator — a different, clean paper | →3.5–4.0 |

#### Stage 5 — text-only replication (optional, highest venue-fit lift/A100-h)
One 7–8B LM, ~500-item reasoning dataset, 3 conditions (unedited/paraphrase floor/scramble floor). **Compute:** ~3 A100-h. If the pathology reproduces: Excitement 3.0→4.0, secures ACL scope. If not: legitimate scoping finding (pathology specific to continuous action readout).

#### Dropped/deferred
More LoRA rank/data ablations — **drop entirely** ("you have seven; they are near-clones with collapsed action spaces and are the *source* of W4, not evidence. Any compute spent here is negative-value"). §2.4 (mid-trajectory) merged into Stage 2. §2.6 (more CoT-VLA families) deferred to camera-ready (engineering cost, not compute cost). §2.7/§2.9 (protocol table, stats fixes) moved into Stage 0 (zero compute). §2.8 (judge robustness) merged into Stage 1.

#### If only ONE round of compute: **Stage 1** — reversing the Round-1 recommendation
"Last round I named the self-generated-CoT rerun. I ranked by *depth of the conceptual hole*, and W3 is the deepest. With a compute budget and a one-shot constraint, the correct ranking criterion is different: which single job converts the largest number of un-defendable weaknesses into defended ones, per GPU-hour." Five reasons: (1) Stage 1 is the only experiment that can repair the statistical power Stage 0's honest restatement gives up — Δ∞ thresholding discards 46% of the mass to a point atom, and a continuous paired statistic typically recovers 2–4× effective power; self-generated CoT re-measures with the *same* low-power instrument and would return at the same ~2/11 significance. (2) It eliminates W2 (τ-robustness) and the "46.2% is cohort-specific" objection *by construction*, not by argument — arguments get rebutted in review, constructions don't. (3) It resolves the paper's central ambiguity (meaning vs. fluency) with a covariate that is a free byproduct of the same forward pass. (4) asymmetric payoff — every outcome branch improves or is neutral; Stage 2 has a genuine downside branch (full rewrite, no compensating power gain) if the effect vanishes on self-generated CoT. (5) cheapest job on the list (5 A100-h) and the one that fixes Excitement, which is the paper's second-weakest axis.

**Caveat offered by the agent itself:** Stage 1 leaves W3 fully open, and a reviewer may still lead with "you edited a teacher-forced annotation" — but W3 has a partial verbal defense at zero cost (the cross-corpus runs *do* self-generate and show direction_flip F of 0.91–0.97 vs. LIBERO's 0.96, direct if partial evidence the response transfers), whereas W1-after-restatement and W6 have no verbal defense — "spend your one round on the weaknesses that cannot be argued." Practical reconciliation: if building Stage 1's harness anyway, self-generating CoT for 3 checkpoints as an add-on costs only ~2 more A100-h — "ship Stage 1, but do not let the framing of the question make you choose when the marginal cost of not choosing is two A100-hours."

</details>

### Independent verification pass (separate fresh Opus 5 agent, no visibility into Round 1/2, forensic/statistical scope only)

<details><summary>Independent verification full text</summary>

**Data provenance.** Recomputed entirely from `results_v2/canonical_runs/*` per-sample records (13 source files/groups, listed with record counts and key fields in the original report). Did not use `derived_metrics.json` as a source for any reported number (opened once for provenance, to diagnose a grid mismatch — see Finding 9 — then re-derived that number from raw records too). Record-integrity pre-checks (delta_per_dim == a_edit − a_orig; delta_linf == max|Δ|; faithful == (delta_linf>0.05)) all pass, 0 mismatches across 19,974 records with action pairs. Reproduction check: this agent's Convention-A numbers reproduce `tab:floors` and its Convention-B numbers reproduce `tab:calibration` to printed precision on every row — i.e. the release is internally consistent and this pipeline agrees with the authors' own, so all findings below are about *derivation logic*, not arithmetic slips.

#### Analysis 1 — convention/threshold sensitivity

**Finding 1.** The "+12/12 against `syntactic_scramble`" claim is Convention-A-specific. Under the appendix's own Convention B it is **10/12** (across all 12 rows including Bridge-4k; `r=8` at −0.0007, `ECoT-bridge` at −0.0108). `tab:floors`'s caption asserts the convention swap "leaves the 12/12 result below intact" — the scramble column is not intact; the caption's stated Convention-B range for the *paraphrase* side (−0.013 to −0.180) reproduces exactly, meaning the caption checked that half and generalized to the scramble half without recomputing it.

**Finding 2.** |B−scram| ≤ 0.012 on 5/12 rows — decided in the third decimal place.

**Finding 3 (bootstrap, two designs — within-family and observation-clustered, 20,000 draws each).** Under Convention B, only **2/12** rows (3/12 under the more defensible observation-clustered design) have a scramble margin whose 95% CI excludes 0 — and both/all are DeepThinkVLA rows, which sit on a *different, non-normalized* Δ∞ scale (see Sensitivity S3 below). All eight ECoT-family rows are individually indistinguishable from zero. As a cohort-level binomial sign test (acknowledging the 12 rows are not independent — 7 are LoRA clones of one base scored on shared observations): 12/12 gives p=0.00049; 10/12 gives p=0.039. The paraphrase side is much more robust: CI excludes 0 on 9/12 (Convention B) vs. 2/12 for scramble.

**Finding 4.** The release's own same-config retraining replicates move (F̄_B − F(scram)) by 0.0115–0.0632 (median 0.0325) — larger in absolute value than the statistic itself on 8 of 12 rows.

**Finding 5.** The paraphrase claim is τ-robust to τ=0.10 but degrades: Convention A sign count 12/12 → 12/12 → **7/12** (τ=0.20) → **6/12** (τ=0.30); Convention B: 12/12→12/12→9/12→7/12. The rows that cross over are the near-saturated ones (r=64, DT SFT/RL/base, data-50B). The release's own `threshold_sweep.json` sweeps only F̄'s rank ordering, never the sign this claim depends on — the two facts (F falls smoothly with τ; the *sign* of a difference of two decaying quantities is not smooth) are compatible but the paper conflates them.

**Sensitivity S3 (flagged as material and underdocumented).** DeepThinkVLA's Δ∞ is in raw quantile-unnormalized robot units (axis spans differing 7.6×), not normalized to [−1,1] as Eq. (1) states; rescaling shrinks its (the only two individually-significant rows') effect sizes 20–25% without flipping any sign.

**Sensitivity S4.** Two released runs of the ECoT-bridge checkpoint give opposite-sign scramble margins (−0.0108 on the 13-family calibration run vs. +0.0034 on the 3-seed 11-family leaderboard sweep) — an additional demonstration the quantity is at noise level for that row.

#### Analysis 2 — action-space collapse and F_dir degeneracy

**Finding 6.** On six of seven "ours" LoRA rows, roll and pitch are frozen at exactly one bin across every scored sample; 21–85% of `a_orig` are the exact no-op; only 44–117 distinct actions appear across 300 observations. ECoT-bridge and DT base: 100/100 distinct, 0% no-op.

**Finding 7.** On the collapsed rows, one axis flipping sign alone gives cos≈−0.95…−1.00 (F_dir|1-flip = 0.98–1.00, because the other two frozen axes contribute nothing to the cosine); the identical event on ECoT-bridge's full 3-D action space gives mean cos=**+0.264** (F_dir|1-flip=0.097). **F_dir is not a common scale across the cohort.**

**Finding 8.** DeepThinkVLA's xyz action occupies only 3–4 sign octants (DT base: 97% in one octant). A permutation null (re-pairing each `a_orig` with a *different* sample's `a_edit`, 2,000 shuffles, same model/run/family) gives DT base a chance rate of exactly 0.000 — the instrument is geometrically near-unreachable on that support regardless of faithfulness, so "the second architecture family fails outright" is not fully licensed as the paper states it (though DT SFT/RL's *observed* F_dir does sit *below* their own permutation-null chance rate of 0.05/0.036, so the qualitative conclusion that DT fails does survive — just not for the reason given). The paper's own reasoning that "0.25 (isotropic chance) ≫ observed, therefore strongly clustered" has the collapse effect backwards: the six clearing LoRA rows' own model-specific chance rate under permutation is 0.376–0.487 (*above* the stated 0.25 isotropic reference), while ECoT/DT sit at 0.000–0.053 (*below* it).

**Finding 9 (favors the paper).** F_dir's clearance is *not* merely the argmax-collision confound the paper flags for F_mag: conditioning both treatment and ceiling on "the action actually moved" (Δ∞>τ) leaves a 5.9–10.7× clearance on the six LoRA rows and still correctly fails no-CoT (0.7×). Against the stricter permutation null (the model's own repertoire), the true margin is smaller — 1.5–1.8× on the LoRAs, 3.2× on ECoT-bridge — but still positive and still fails on no-CoT and DT.

**Finding 10 (new — not identified in Round 1).** `tab:directional` and `tab:fdirnull` compute F_dir on **two different de-quantization grids** (one re-grids through the checkpoint's own bin midpoints; the other uses the grid as released) and get different numbers for nominally the same quantity (e.g. `r=8`: 0.649 vs. 0.666; `data-50B`: 0.542 vs. 0.582). `tab:fdirnull`'s caption attributes the gap to "pooled vs. seed-mean" aggregation — verified arithmetically impossible: both aggregations agree to 4 decimal places on every row. The real cause (traced into `scripts/derive_metrics.py::_to_checkpoint_grid` vs. `scripts/fdir_null.py`) is the grid conversion, which snaps one of the two central (near-zero) bins to a true zero and shifts the other — exactly where the collapsed action space concentrates its mass. Separately, ECoT-bridge's printed 3.0× (vs. this agent's reproduced 3.8× on a single coherent run) comes from mixing the treatment value from one released run with supplementary-family values from a different released run, which is documented in the code but not in the paper, and contradicts `tab:calibration`'s caption's promise that "every quantity in a row is computed only against families from that same run."

**Finding on the direct question asked (§2.3):** Yes, with one instructive exception — every row clearing at 6.2–11.1× has a collapsed action space; the one clearing row with a rich action space (ECoT-bridge) clears at only 3.0–3.8×, the smallest positive margin; DT's 3–4-octant support doesn't clear at all. But collapse alone is not sufficient (no-CoT is the *most* collapsed row and has the *lowest* clearance, 0.8×) — what the collapsed rows additionally need is for the edit to actually flip the sign of their one active axis, which they do 61–96% of the time and no-CoT does only 2% of the time. So: "collapse is not what *creates* the clearance; it is what makes the clearance cheap and non-comparable across the cohort."

**Sensitivities flagged (S6–S10):** the no-op threshold definition, DT's incomparable grid, the permutation-null design choice, denominator behavior after re-gridding, and one unreproduced ±sd column in `tab:directional` are all called out explicitly as judgment calls the paper's schema does not resolve — full detail in the verbatim methodology.

**Closing note from this agent:** "No fallback occurred. Every number in this report is computed from `per_sample`/`per_sample_edit` records. `derived_metrics.json` was read once for provenance only... and its values were then re-derived from raw. No file in the repository was modified."

</details>

---

## Provenance

- Round 1 + Round 2: single Opus 5 `general-purpose` agent, agent id `a4d5595232b48ebd0`, ~445k combined output tokens, 58+ tool uses (Round 1); full transcripts in this session's task-output directory (ephemeral — not preserved beyond this session; this document and the trace files under `.aris/traces/research-review/2026-08-31_run01/` are the durable record).
- Independent verification: separate Opus 5 `general-purpose` agent, agent id `ac4fcd29f7b80b2c3`, ~223k output tokens, 60 tool uses, launched with no visibility into Round 1/2's output.
- Codex MCP (`gpt-5.5`, the skill's default backend) was unreachable all session; not used.

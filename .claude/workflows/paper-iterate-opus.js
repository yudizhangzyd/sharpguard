export const meta = {
  name: 'paper-iterate-opus5',
  description: 'Reviewer (Opus 5) + researcher loop on paper v6. 3 rounds.',
  phases: [
    { title: 'R1-review',   detail: 'Opus 5 reviewer scores paper v6' },
    { title: 'R1-research', detail: 'Researcher applies top-3 fixes' },
    { title: 'R2-review',   detail: 'Opus 5 blind re-reads' },
    { title: 'R2-research', detail: 'Researcher applies remaining fixes' },
    { title: 'R3-review',   detail: 'Opus 5 final verdict' },
    { title: 'Synthesize',  detail: 'Package final paper + verdicts' },
  ],
}

const REVIEWER_MODEL = 'claude-opus-5'

const PAPER = '/Users/yudizhang/Documents/sharpguard/cot_faith_iclr.tex'
const CTX_HEADER = `You are reviewing an ICLR 2026 D&B track submission on CoT faithfulness in manipulation VLAs.
Read the paper at ${PAPER} and the figures in /Users/yudizhang/Documents/sharpguard/figures/.
Look at raw data in /Users/yudizhang/Documents/sharpguard/results_v2/, /tmp/cf_done/, /tmp/cf_r3_all/, and /tmp/cf_full_sweep/.
Be brutally honest, senior-reviewer level. No hedging.

Paper v6 (current state) changes since prior review:
- 3-seed mean±std for ECoT-bridge row (leaderboard row 8)
- paraphrase_null edit family added (Section 6.7); ECoT-bridge shows F=0.95±0.02
- F5 cross-corpus now N=30 per non-LIBERO corpus (Fig 7, Fig 10 regenerated from real data)
- Real AUROC data (cot=0.410, visual=0.607, instr=0.459, prev=0.644 on N=200)
- Table 1 all cells filled (r=32 completed, ECoT-bridge completed with 3 seeds)
- Fig 1 hero regenerated with self-consistent 'turn on the stove' sample
- location_swap N expanded from 12 to 60-74 via C5 fix
- DT edit N=0 (FAST tokenizer incompatibility — acknowledged in limitations)
- Rollout Task SR 0/20 — deferred as future work`

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['score', 'top_5_critical', 'min_lift_experiments', 'strengths', 'accept_verdict'],
  properties: {
    score: { type: 'number', minimum: 1, maximum: 10 },
    top_5_critical: { type: 'array', items: { type: 'object', required: ['issue','why_it_kills','fix'],
      properties: { issue: {type:'string'}, why_it_kills: {type:'string'}, fix: {type:'string'} } }, maxItems: 5 },
    min_lift_experiments: { type: 'array', items: { type: 'object', required: ['name','compute_hours','expected_score_lift'],
      properties: { name: {type:'string'}, compute_hours: {type:'number'}, expected_score_lift: {type:'number'} } }, maxItems: 5 },
    strengths: { type: 'array', items: { type: 'string' }, maxItems: 5 },
    accept_verdict: { enum: ['strong_accept','accept','borderline_accept','borderline_reject','reject','strong_reject'] },
  },
}

const RESEARCH_SCHEMA = {
  type: 'object',
  required: ['changes_made','experiments_run','commit_sha'],
  properties: {
    changes_made: { type: 'array', items: { type: 'string' } },
    experiments_run: { type: 'array', items: { type: 'string' } },
    commit_sha: { type: 'string' },
    new_findings_added: { type: 'array', items: { type: 'string' } },
    remaining_gaps: { type: 'array', items: { type: 'string' } },
  },
}

// ────────── Round 1 ──────────
phase('R1-review')
const r1_review = await agent(
  `${CTX_HEADER}

   Round 1 review of paper v6. Blind read. Return structured JSON per schema.
   Focus on issues that would tank ICLR D&B. Prior reviews are irrelevant — assess as fresh reviewer.`,
  { schema: REVIEW_SCHEMA, label: 'reviewer/R1-opus5', model: REVIEWER_MODEL }
)

phase('R1-research')
log(`R1: score=${r1_review.score} verdict=${r1_review.accept_verdict}`)
const r1_research = await agent(
  `Round 1 researcher for paper v6. Opus reviewer gave score ${r1_review.score}/10 verdict=${r1_review.accept_verdict}.
   Top-5 critical: ${r1_review.top_5_critical.map((x,i)=>`(${i+1}) ${x.issue} → ${x.fix}`).join(' ; ')}
   Min-lift experiments: ${r1_review.min_lift_experiments.map((x,i)=>`(${i+1}) ${x.name} [${x.compute_hours}h, +${x.expected_score_lift}]`).join(' ; ')}

   Pick top-3 highest-lift-per-hour items and EXECUTE. Edit ${PAPER}, generate figures, submit bolt jobs.
   No scope creep. Commit + return structured JSON.`,
  { schema: RESEARCH_SCHEMA, label: 'researcher/R1' }
)

// ────────── Round 2 ──────────
phase('R2-review')
const r2_review = await agent(
  `${CTX_HEADER}

   Round 2 blind re-read after commit ${r1_research.commit_sha}. Return structured JSON per schema.`,
  { schema: REVIEW_SCHEMA, label: 'reviewer/R2-opus5', model: REVIEWER_MODEL }
)

if (r2_review.score >= 8 || r2_review.accept_verdict.includes('accept')) {
  log(`✅ Converged at R2: score=${r2_review.score} verdict=${r2_review.accept_verdict}`)
  return { rounds_completed: 2, final_score: r2_review.score, final_verdict: r2_review.accept_verdict,
           r1: r1_review, r2: r2_review, research_log: [r1_research] }
}

phase('R2-research')
const r2_research = await agent(
  `Round 2 researcher. Score ${r1_review.score} → ${r2_review.score}. Verdict: ${r2_review.accept_verdict}.
   Remaining top-5: ${r2_review.top_5_critical.map((x,i)=>`(${i+1}) ${x.issue} → ${x.fix}`).join(' ; ')}
   Apply top-3. Commit + return structured JSON.`,
  { schema: RESEARCH_SCHEMA, label: 'researcher/R2' }
)

// ────────── Round 3 ──────────
phase('R3-review')
const r3_review = await agent(
  `${CTX_HEADER}

   FINAL review. Commit ${r2_research.commit_sha} was the last edit. Score honestly for ICLR D&B track acceptance.
   Return structured JSON.`,
  { schema: REVIEW_SCHEMA, label: 'reviewer/R3-opus5', model: REVIEWER_MODEL }
)

phase('Synthesize')
log(`Trajectory: R1=${r1_review.score} → R2=${r2_review.score} → R3=${r3_review.score}`)
log(`Final verdict: ${r3_review.accept_verdict}`)

return {
  rounds_completed: 3,
  score_trajectory: [r1_review.score, r2_review.score, r3_review.score],
  final_verdict: r3_review.accept_verdict,
  final_top_issues: r3_review.top_5_critical,
  final_strengths: r3_review.strengths,
  research_diffs: [r1_research, r2_research],
  all_reviews: [r1_review, r2_review, r3_review],
}

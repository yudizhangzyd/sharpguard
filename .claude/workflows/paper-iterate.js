export const meta = {
  name: 'paper-iterate',
  description: 'Reviewer + researcher loop: critique paper, apply top-lift fixes, repeat until score ≥ 8/10 or 3 rounds.',
  phases: [
    { title: 'R1-review',   detail: 'Independent GPT-5 reviewer scores current paper' },
    { title: 'R1-research', detail: 'Researcher applies top-3 fixes' },
    { title: 'R2-review',   detail: 'Blind reviewer re-scores' },
    { title: 'R2-research', detail: 'Researcher applies remaining fixes' },
    { title: 'R3-review',   detail: 'Final judge score + accept/reject verdict' },
    { title: 'Synthesize',  detail: 'Package final paper + reviewer verdicts' },
  ],
}

const PAPER = '/Users/yudizhang/Documents/sharpguard/cot_faith_iclr.tex'
const CTX_HEADER = `You are reviewing an ICLR 2026 D&B track submission on CoT faithfulness in manipulation VLAs.
Read the paper at ${PAPER} and the figures in /Users/yudizhang/Documents/sharpguard/figures/.
Look at raw data in /Users/yudizhang/Documents/sharpguard/results_v2/ and /tmp/cf_done/.
Be brutally honest, senior-reviewer level. No hedging.`

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

   Round 1 initial review. Do NOT assume any prior context.
   Return structured JSON per schema. Focus on issues that would tank the paper at ICLR.`,
  { schema: REVIEW_SCHEMA, label: 'reviewer/R1' }
)

phase('R1-research')
log(`R1 verdict: ${r1_review.accept_verdict} (score ${r1_review.score}/10)`)
const r1_research = await agent(
  `Round 1 researcher. Reviewer gave score ${r1_review.score}/10 verdict=${r1_review.accept_verdict}.
   Top 5 critical issues:
   ${r1_review.top_5_critical.map((x,i)=>`   ${i+1}. ${x.issue} — fix: ${x.fix}`).join('\n')}
   Min-lift experiments to run:
   ${r1_review.min_lift_experiments.map((x,i)=>`   ${i+1}. ${x.name} (${x.compute_hours}h, +${x.expected_score_lift})`).join('\n')}

   Your job: pick the top-3 highest-lift-per-hour items and EXECUTE them. Edit ${PAPER}, generate figures, submit bolt jobs.
   Do NOT scope-creep. Commit changes when done. Return structured JSON.`,
  { schema: RESEARCH_SCHEMA, label: 'researcher/R1' }
)

// ────────── Round 2 ──────────
phase('R2-review')
const r2_review = await agent(
  `${CTX_HEADER}

   Round 2 review — blind re-read. Assume nothing from prior rounds.
   Commit ${r1_research.commit_sha} was applied since last review. Read paper fresh.
   Return structured JSON per schema.`,
  { schema: REVIEW_SCHEMA, label: 'reviewer/R2' }
)

if (r2_review.score >= 8 || r2_review.accept_verdict.includes('accept')) {
  log(`✅ Converged at R2: score ${r2_review.score}, verdict ${r2_review.accept_verdict}`)
  return { rounds_completed: 2, final_score: r2_review.score, final_verdict: r2_review.accept_verdict,
           r1: r1_review, r2: r2_review, research_log: [r1_research] }
}

phase('R2-research')
const r2_research = await agent(
  `Round 2 researcher. Score progressed ${r1_review.score} → ${r2_review.score}. Verdict: ${r2_review.accept_verdict}.
   Remaining top-5 critical:
   ${r2_review.top_5_critical.map((x,i)=>`   ${i+1}. ${x.issue} — fix: ${x.fix}`).join('\n')}

   Apply top-3 fixes. Commit. Return structured JSON.`,
  { schema: RESEARCH_SCHEMA, label: 'researcher/R2' }
)

// ────────── Round 3 (final judge) ──────────
phase('R3-review')
const r3_review = await agent(
  `${CTX_HEADER}

   FINAL round review. This is the verdict. Read paper fresh.
   Commit ${r2_research.commit_sha} was the last edit. Score honestly for ICLR D&B track acceptance.
   Return structured JSON.`,
  { schema: REVIEW_SCHEMA, label: 'reviewer/R3-final' }
)

phase('Synthesize')
log(`Trajectory: R1=${r1_review.score} → R2=${r2_review.score} → R3=${r3_review.score}`)
log(`Final verdict: ${r3_review.accept_verdict}`)

return {
  rounds_completed: 3,
  score_trajectory: [r1_review.score, r2_review.score, r3_review.score],
  final_verdict: r3_review.accept_verdict,
  final_top_issues: r3_review.top_5_critical,
  research_diffs: [r1_research, r2_research],
  all_reviews: [r1_review, r2_review, r3_review],
}

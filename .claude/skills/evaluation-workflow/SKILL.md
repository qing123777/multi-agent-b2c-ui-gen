---
name: evaluation-workflow
description: Use when running or extending the FYP's 9-metric evaluation framework — System Reliability (MAESTRO-grounded), Quality of System Output (WebCoderBench/Lighthouse/Baymard-grounded), and the Tier 3 MAS-vs-baseline comparison. This is the evaluation-campaign procedure, not a day-to-day development task.
---

# Running the evaluation campaign

Two categories plus a comparison tier, all grounded in published benchmarks so targets
have a stated justification rather than being arbitrary.

- **Category 1 — System Reliability**, grounded in MAESTRO (arXiv:2601.00481)
- **Category 2 — Quality of System Output**, grounded in WebCoderBench (arXiv:2601.02430)
- **Tier 3 — MAS vs. single-LLM baseline**, the strongest academic argument: it isolates
  what the multi-agent architecture itself contributes
- **Metric 9 — Cost & Latency Profile**, descriptive, attached to Tier 3

## Test set

15 B2C prompts, 5 page-type categories × 3 prompts each: product listing, product
detail, cart & checkout, storefront/homepage, user account. For **Metric 3 (run-to-run
stability)**, pick one prompt per category (5 total) and run each 3×.

## Procedure

1. **Run each of the 15 prompts through the full MAS once.** Log wall-clock time
   (timestamp before invoke → after final state) and token usage (LangSmith traces or
   the OpenAI usage field, summed across all agent calls in the run) for Metric 9.

2. **Category 1 — Reliability, from the same 15 runs:**
   - *Metric 1 (Task Completion Rate)*: completions / 15, target ≥80%.
     - *Lint Efficiency sub-metric*: first-pass lint rate (target ≥70%), average retry
       count, cap-hit rate (target <5%) — measures the developer agents' internal
       generate→lint→fix loop, not the modification loop.
     - *Modification Loop Efficiency sub-metric*: first-correction success rate,
       second-correction success rate, cap-exhaustion rate (target <10%). These three
       are mutually exclusive and **must sum to 100% of initially-failing runs** — this
       triage is the direct empirical answer to whether `modification_attempts` cap=2
       is well-calibrated (see [`playwright-test-pipeline`](../playwright-test-pipeline/SKILL.md)
       for the cap mechanics). Also compute regression-during-correction rate: diff the
       per-test pass/fail list at both transitions (baseline→attempt 1, attempt 1→attempt 2)
       for any test flipping PASSED→FAILED.
   - *Metric 2 (Silent Failure Rate)*, target <20% (contextualized against MAESTRO's
     75.17% baseline finding): two-pass check — (a) fresh Playwright render, count JS
     console errors; (b) LLM-as-judge given HTML+CSS+JS together, asked to flag selector
     mismatches, undefined DOM references, broken interactions.
   - *Metric 3 (Run-to-run Stability)*: run the 5 selected prompts 3× each, compute std
     dev of Category 2 scores, target <0.5 on the 1–5 scale.

3. **Category 2 — Quality, from the same 15 runs:**
   - *Metric 4 (Requirement Alignment)*: LLM-as-judge (GPT-4o-mini or Claude Haiku),
     pointwise scoring with chain-of-thought, JSON output, input = user prompt + HTML +
     CSS + JS. 4 dimensions × 1–5 scale: Requirement Coverage, Visual Coherence,
     Functional Completeness, Intent Alignment. Target ≥3.5/5 average (contextualized
     against Baymard's 64%-mediocre industry baseline).
   - *Metric 5 (Functional Correctness)*: a **fresh, independent** Playwright render
     check (`page.goto` → count console errors) — do NOT reuse the TestExec agent's own
     internal pass rate as quality evidence, it's partially circular (the system already
     looped to pass it). Target: 0 console errors on the fresh check; report TestExec's
     internal pass rate separately as a process-efficiency number, target ≥80%.
   - *Metrics 6/7 (Lighthouse Accessibility / Best Practices)*: `lighthouse
     http://localhost:8080/preview_{id}.html --output=json --quiet
     --chrome-flags=--headless`, parse `categories.accessibility.score` /
     `categories.best-practices.score`. Target ≥90 for both (Google's own "Good"
     classification threshold — the only metric here with an externally defined
     number). Lighthouse Performance is supplementary only (skewed by external CDN
     latency); Lighthouse SEO is irrelevant for a local demo, drop it.
   - *Metric 8 (Baymard UX Compliance)*: run UX-Ray 2.0 manually on 3–5 generated pages
     (one per page type). Report the observed compliance score against Baymard's own
     64%-mediocre industry finding as context, not a numeric pass/fail. This is
     validation evidence for the Compliance Mode system prompts, not a primary scored
     metric.

4. **Tier 3 — MAS vs. baseline**: repeat the same 15 prompts through a single-LLM
   baseline (one GPT-4o call, no agents, no correction loop). Compare Metrics 4–7
   between the MAS run and the baseline run — this comparison is the headline result.

5. **Metric 9 (Cost & Latency Profile)**: report per-prompt pairs (MAS vs. baseline on
   the same prompt) plus overall means, wall-clock latency and tokens/cost side by side.
   No pass/fail target — deliberately descriptive, answers "the MAS wins on quality, but
   at what price?"

## Circular metrics — do not use as primary quality evidence

| Metric | Why it's circular | Use instead |
|---|---|---|
| Final lint pass rate | The loop runs until it passes — always ~100% | Report as Lint Efficiency (first-pass rate + retry count) |
| Final TestExec Playwright pass rate | The correction loop already optimizes against this | Report as a process metric; use the fresh independent Playwright check (Metric 5) for quality evidence |

## Modification-mode caveats when scoring

- A user-triggered modification request consumes attempt 1 of the cap on its own
  verification — it does not get a free initial check the way a generation run does.
  State this asymmetry when interpreting observed correction rates.
- Per-instance style overrides (e.g. "make the 2nd card red") are applied but not
  covered by automated verification (`assert_css` checks the first selector match
  only) — exclude such prompts from scored modification test cases, or report under a
  documented limitation.
- A refused structural modification request (add/remove component, behavior change —
  intercepted at the Primary Orchestrator and redirected to generation) counts as
  **correct** behavior for Task Completion Rate, not a failed run.

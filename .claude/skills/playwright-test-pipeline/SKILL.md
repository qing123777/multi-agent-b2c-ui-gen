---
name: playwright-test-pipeline
description: Use when writing, debugging, or extending the Test Generation → Test Execution pipeline — the structured JSON test-case schema, the 9-action vocabulary, error classification, and how run_test_suite drives Playwright's Python API directly (no .spec.ts files, no LLM in the execution step).
---

# The test generation + execution pipeline

Two stages, deliberately different in kind:

- **Test Generation** is a real LLM agent — it judges *what* to test from the four
  specification schemas and emits structured JSON test cases.
- **Test Execution** is a **deterministic pipeline stage, not an LLM agent**.
  `call_test_exec_agent` calls `run_test_suite.func` directly; the case interpreter
  (`_execute_action`) drives Playwright's Python API straight from the JSON — there is
  no generated `.spec.ts`, no external test runner, no LLM judgment at execution time.
  (An older `test_exec_agent` that wrapped this in an LLM turn is marked SUPERSEDED
  in-notebook and kept only for reference — it added latency/tokens/a stochastic
  failure surface for zero benefit, since its only permitted behavior was calling the
  same tool once.)

## Test case schema

7 required keys + 1 optional:

| Key | Meaning |
|---|---|
| `id` | Unique test case identifier |
| `component` | Which spec component this exercises |
| `description` | Human-readable intent |
| `action` | One of the 9-action vocabulary below |
| `selector` | Per the Selector Contract — `.className` if the component has one, else `#id` |
| `expected` | Action-specific expected value (format depends on `action`, see below) |
| `passed` | Result field, written by execution, not generation |
| `viewport` *(optional)* | `"WIDTHxHEIGHT"`, e.g. `"375x812"` — triggers a `page.set_viewport_size` call before this case runs; omitted cases use the default 1280×720 |

`store_test_cases` is the syntactic + coverage validator: it checks the schema, the
action vocabulary, per-action expected-value format, the Selector Contract, and that
every spec component has at least one covering case. A malformed submission is rejected
with a specific error and the LLM retries — this validator already exists, don't add an
external syntax-checking tool on top of it.

## The 9-action vocabulary

| Action | `expected` format | What it does |
|---|---|---|
| `assert_visible` | (none / ignored) | Element exists and is visible |
| `assert_text` | literal text | Element's text content matches |
| `assert_css` | `property=value` pairs | Computed style matches |
| `assert_hover_css` | `property=value` pairs | `page.hover()` then computed style matches (shares the CSS-pair comparison helper with `assert_css`) |
| `assert_attribute` | `attr=value`, or `attr=` (empty) for presence-only | DOM attribute matches, or just exists |
| `assert_url` | substring | `page.url` contains this substring |
| `click` | (none) | `page.click()` |
| `type` | text to type | `page.fill()`/`page.type()` |
| `screenshot` | (none) | Capture only, no assertion |

## Error tagging and classification

Every `FAILED` line in the report is tagged with the action that produced it —
`FAILED: {id} [{action}] — {reason}` — written deterministically by `_build_report`,
never phrased by an LLM. `_error_type` maps the tag to one of 6 categories:

| Category | Source tag(s) |
|---|---|
| `style_mismatch` | `[assert_css]`, `[assert_hover_css]` |
| `text_content_error` | `[assert_text]` |
| `interaction_error` | `[click]`, `[type]`, `[assert_attribute]`, `[assert_url]` |
| `visibility_error` | `[assert_visible]`, or **any** assertion whose failure reason contains "not found" (this override takes priority over the tag) |
| `test_infra_error` | `[screenshot]`, `[page_load]` |
| `unclassified_error` | missing/unrecognized tag — kept separate from `test_infra_error` so an unexpected format is never silently mislabeled as "confirmed not a real bug" |

If you add a new action, add its category mapping here and to `_error_type` in the same
change — the write side (`_build_report`) and read side (`_error_type`) must stay in
sync, since nothing else enforces that.

## Retry / cap mechanics

The correction loop is capped at `modification_attempts == 2` per run, and the cap is
**code-enforced**, not just prompt-instructed: `call_test_exec_agent` refuses to run
further corrections once the cap is hit, returning a `"REFUSED"` `ToolMessage` instead
of invoking Playwright again. Don't rely on the system prompt alone if you're adding a
new correction path — add the same guard in code.

## Debugging a failing suite

1. Read the tagged failure lines directly — the `[action]` tag tells you which
   Playwright call failed, `_error_type`'s category tells you the likely root cause
   class without re-deriving it from free text.
2. Reproduce a single case in isolation with Playwright's Python API directly (same
   selector, same action) before assuming the generated HTML/CSS/JS is wrong — confirm
   it isn't a selector/Naming Contract mismatch first (see the CLAUDE.md Naming/Selector
   Contract section).
3. If the failure reason contains "not found," treat it as `visibility_error`
   regardless of which assertion surfaced it — the element genuinely isn't in the DOM,
   which is a different bug class than a style or text mismatch.

## Adding a new test action

1. Add the action to `VALID_ACTIONS` and `store_test_cases`'s per-action expected-format
   rule.
2. Add the interpreter branch in `_execute_action`.
3. Add its tag to `_build_report` and its category to `_error_type`.
4. Add Test Gen prompt guidance for when to emit it (which spec concept it verifies).
5. Add offline unit test cases per [`notebook-editing`](../notebook-editing/SKILL.md)'s
   test-cell convention — do not require a live LLM call to prove the interpreter and
   validator work.

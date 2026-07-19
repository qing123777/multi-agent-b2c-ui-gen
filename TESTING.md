# Testing & Evaluation — WebForge

Three-tier evaluation structure, executed in order:

| Tier | Scope | Verifies |
|---|---|---|
| **1 — Tool-level** | one tool / API | each tool called directly; correct behaviour plus failure cases (exception handling, external-process responses, tool safeguards) |
| **2 — Turn-based** | one sub-agent | single-turn invocation over a set of manual inputs; tool sequencing, output quality, fallback and feedback mechanisms |
| **3 — Task-based** | whole system | end-to-end scenarios; success rate, recovery rate, efficiency |

A second, independent axis cuts across all three: **offline-mocked** vs **live-real-deps**.
A mocked pass does not imply the live path works. This is not theoretical here —
`run_eslint` and `run_stylelint` were green under mocks for weeks while completely broken
against the real binaries (`run_eslint` passed flags removed in ESLint v9; stylelint v16+
moved its JSON report to stderr). Both tiers below state which axis they sit on.

All tests live inside `fyp.ipynb`, in a cell directly below the code they cover. Locate them
**by content, not index** — indices shift on every insertion.

---

## Tier 1 — Tool-level (offline-mocked) — COMPLETE

**13 of 13 registered developer-agent tools are tested, every one with ≥5 scenarios**, plus the shared multi-block diff helpers.
External processes (`npx html-validate`, `npx eslint`, `npx stylelint`) are mocked via
`unittest.mock.patch("subprocess.run")`, so the suite is deterministic and needs no toolchain.

| Tool | Agent | Scenarios | Failure cases covered |
|---|---|---:|---|
| `run_html_validate` | HTML | 5 | empty input guard; malformed HTML; validator error output |
| `store_html_code` | HTML | 9 | missing `<link>`/`<script>` include tags; spec component not addressable; modify-mode exemption; missing-`session_id` back-compat |
| `extract_component` | HTML | 5 | selector not found; `#id` / `.class` / bare-name resolution |
| `store_html_diff` | HTML | 7 | empty `html_code`; component not found; instance out of range; malformed patch |
| `validate_component_tags` | HTML | 5 | missing required attributes at depth |
| `run_eslint` | JS | 13 | empty input; **errors block / warnings never block**; `npx` missing; unparseable CLI output; flat-config invocation |
| `store_js_code` | JS | 13 | interaction target unreferenced; Naming-Contract className resolution; raw-id fallback; modify-mode exemption; no-spec back-compat |
| `extract_js_component` | JS | 7 | selector not found; `#id` / `.class` / bare-name resolution; **all referencing paragraphs returned**; single match stays unmarked |
| `store_js_diff` | JS | 6 | empty `js_code`; block not found; malformed patch; **non-contiguous paragraphs patched** (was a silent no-op) |
| `run_stylelint` | CSS | 12 | empty input; **report on stderr (v16+) and stdout (legacy)**; exit-78 config error; `npx` missing; unparseable output |
| `store_css_code` | CSS | 12 | decoration className with no matching rule; modify-mode exemption; no-spec back-compat |
| `extract_css_component` | CSS | 7 | selector not found; `#id` / `.class` / bare-name resolution; **all matching blocks returned** (base + `:hover`); single match stays unmarked |
| `store_css_diff` | CSS | 7 | empty `css_code`; block not found; malformed patch; **base and `:hover` patched together**; destroyed separator rejected |
| `_join_blocks` / `_split_blocks` / `_splice_spans` | shared | 7 | marker deleted / renumbered / duplicated; span-count mismatch; trailing-newline trim; identical and non-contiguous blocks |
| `_lint_failure` / `_lint_refused` | shared | 6 | off-by-one at the cap boundary; post-cap refusal not counted; escalation never fakes a clean lint; bounded-termination loop |

**How to run:** open `fyp.ipynb`, execute cells top-to-bottom, or run any tool's cell followed
immediately by its test cell. No API key, no network, no cost.

---

## Tier 2 — Turn-based (offline, scripted model) — COMPLETE for the three developer agents

**15 scenarios: 5 per agent, 0 LLM calls.** These exercise a whole sub-agent — tool
sequencing, the generate→lint→fix loop, write-time guard feedback, modify-mode routing, and
fallback on infrastructure failure — with every model reply pre-scripted.

The four cells sit in `fyp.ipynb`: a shared harness immediately before the HTML agent, and one
suite directly below each agent (sentinels `TB-HTML-1`, `TB-JS-1`, `TB-CSS-1`).

### Why a custom scripted model was necessary

No stock LangChain fake chat model can drive these agents. `FakeChatModel`,
`FakeListChatModel`, `FakeMessagesListChatModel` and `GenericFakeChatModel` all inherit
`BaseChatModel.bind_tools`, whose body is `raise NotImplementedError` — and `create_agent`
binds tools to the model. `ScriptedChatModel` overrides `bind_tools`, and additionally
replaces `FakeMessagesListChatModel`'s silent response-cycling with a hard failure, so a
runaway agent loop cannot masquerade as a pass.

### The five scenarios, applied identically to each agent

| # | Scenario | Verifies |
|---|---|---|
| 1 | Happy path (generation) | exact tool order, lint gate, file on disk matches state. Runs the **real** linter when `npx` is available — the only place real-linter integration meets full agent wiring without spending credits |
| 2 | Self-correction loop | linter fails once, agent revises, second pass succeeds. JS additionally proves warnings-only **never** blocks — a cosmetic `no-unused-vars` must not trap the fix loop |
| 3 | Coverage-guard rejection | incomplete output rejected at write time and the feedback **names the missing item**; retry accepted |
| 4 | Modify mode | Step 0 seed → extract → diff; asserts `store_*_code` is **not** called again after Step 0 |
| 5 | Infrastructure fallback | `npx` missing → error is distinguishable from a code error, gate holds, no file written |

### Validation of the suites themselves

- All 15 scenarios pass against the real tools and middleware.
- **Negative controls:** four deliberate mutations (drop the `write_file` turn, bypass the
  coverage guard, re-store instead of diffing in modify mode, and a tool-list drift) were each
  caught. A suite that always passes is worthless; these prove it does not.
- **Tool-drift guard:** each suite parses the production `create_agent` cell and asserts its
  rebuilt tool list matches, so adding a tool without updating the suite fails loudly.
- **No-network proof:** run with a deliberately invalid `OPENAI_API_KEY`; any real API call
  would raise an auth error.

### Defect found by Tier 2 on its first run — FIXED

**Symptom.** A modify-mode run extracted the target component, applied the diff, linted clean
and reported success — while the file on disk never changed. The user would see "done" and no
change in the preview.

**Cause.** `deepagents`' `FilesystemBackend.write()` refuses unconditionally when the target
path already exists (a plain `resolved_path.exists()` check, with no "was read first"
exemption). All three modify-mode prompts end with `write_file` on the very file read at
Step 0, so the write was silently rejected.

**Why tier 1 could not have caught it.** Every tool passes its own unit tests — each is
individually correct. The defect existed only in the *sequence*, which is exactly what
turn-based testing exercises.

**Fix.** `OverwritingFilesystemBackend`, a small subclass that removes the pre-existing file
before delegating to the parent's `write` (which already opens with `O_TRUNC`). All three
agents now construct it instead of the stock backend. Path resolution, traversal protection
and `virtual_mode` rooting are inherited unchanged — verified that `..` traversal is still
blocked and nested-directory creation still works.

Chosen over the alternatives because it fixes the root cause with no prompt change, no new
tool, and no change to agent behaviour. (`edit_file` was rejected: it is a find/replace needing
an exact unique `old_string`, not a whole-file overwrite, so it would have forced the model to
re-transcribe content.)

**Verification.** All three scenario-4 tests now assert the modification reaches disk *and*
that the file matches the code in state. The fix was confirmed **load-bearing** by reverting
just the backend swap in memory and observing the modify scenario fail again with the original
"refused" error — proving the tests detect the real defect rather than passing incidentally.
Each suite retains a regression guard that fails loudly if the stock backend ever returns.

---

## Multi-block modification — FIXED

**Limitation.** `extract_css_component` / `extract_js_component` returned only the *first*
matching block, and the diff tools patched only that one. A component's code is rarely one
block: a CSS rule normally has `:hover` and media-query variants, and a JS component can own
several listeners. "Make the button darker" therefore updated the base rule and left the hover
state on the old colour.

**A second, quieter defect found alongside it.** `store_js_diff` joined all matching
paragraphs with `"\n\n"` and spliced with `js_code.replace(joined, patched, 1)`. When the
matching paragraphs were **not adjacent**, the joined text was not a substring of `js_code`,
so `replace` matched nothing, `js_code` came back unchanged — and the tool still reported
"Diff applied." Same failure shape as the `write_file` bug: success reported, nothing changed.

**Fix.** Three shared helpers (`_join_blocks` / `_split_blocks` / `_splice_spans`) sitting
directly below `_apply_patch`:

- All matching blocks are returned as one document, separated by
  `/* @@ WebForge block N of M @@ */` lines — valid comment syntax in both CSS and JS. Diff
  line numbers stay unambiguous across block boundaries, and the tool message carries no
  preamble, since any header would shift those numbers.
- Splicing is by **character span**, not `str.replace`, so non-contiguous blocks return to
  their own positions and two textually identical blocks can never be confused.
- `_split_blocks` compares the surviving markers against the exact sequence handed out. A
  plain count check is not enough: deleting the *first* marker still leaves `M` sections and
  would silently merge two blocks. Tampering is rejected; state is left untouched.
- **One block behaves exactly as before** — no markers, byte-identical output — so the common
  case carries none of this machinery.

Two pre-existing wrinkles fell out of the rewrite: blocks extracted from a CRLF source are now
exact substrings (the old line-join re-emitted `\n`), and patching a block's last line no
longer injects a blank line (`_apply_patch` terminates every added line with `\n`, which the
old splice kept).

**Verification.** All 6 existing tool test cells pass unchanged; the 15 turn-based scenarios
still pass 5/5 per agent. The four updated test cells were confirmed **load-bearing** by
running them against the reverted `blocks[0]` tools — all four fail. A 19-check notebook audit
confirms 273 untouched cells are byte-identical, no caller indexes `[0]` any more, and the
marker the generator emits round-trips through the regex for 1–5 blocks.

**Prompt change.** Both modification prompts now describe the multi-block document and forbid
editing the separator lines. This is the "LLMs decide, Python guarantees" split again: the
prompt asks for the markers to be preserved, `_split_blocks` enforces it.

---

## Lint fix-loop cap — FIXED

**Limitation.** The generate → lint → fix loop had no bound in code — no `recursion_limit`,
`lint_attempts` or `max_iterations` existed anywhere in the notebook. The prompts said to
iterate until `"No issues found."`, so a defect the model could not fix had no stop condition.
The only backstop was LangGraph's default `recursion_limit`, which in the installed 1.0.10 is
**10000 supersteps** (`DEFAULT_RECURSION_LIMIT` in `langgraph/_internal/_config.py`) — roughly
5000 model calls of real spend before `GraphRecursionError`, producing no artifact and no
report. (LangGraph 0.x defaulted to 25; that figure does not apply here.)

**Fix.** `_LINT_ATTEMPT_CAP = 3` (1 initial lint run + 2 fix attempts), enforced by two shared
helpers used by all three linters. It mirrors `call_test_exec_agent`: a hard cap in Python and
a partial result instead of a failure.

Two rules make it correct rather than merely bounded:

- **Only a real lint verdict consumes an attempt.** `npx` missing, a stylelint config error,
  unparseable output and empty code are infrastructure problems and are reported without
  counting — the same reasoning that makes the Playwright pipeline retry transient page-load
  failures before entering the correction loop. ESLint warnings likewise never count, so a
  cosmetic `no-unused-vars` still cannot trap or exhaust the loop.
- **Past the cap the linter refuses to run at all.** The escalation message tells the model to
  stop, but a reasoning lapse could re-invoke the tool; refusing makes the cap non-negotiable
  and costs no subprocess.

**At escalation the agent saves the best-effort artifact** and ends with a summary of the
unresolved problems. This relaxes the previous absolute rule "never save unless lint_passed" —
deliberately, because a page with a known styling defect is more useful than no file at all.
`lint_passed` stays `False`, so escalation can never be mistaken for success.

**Recursion limits, set explicitly.** All five sub-agent invocations
(`call_html_agent`, `call_css_agent`, `call_js_agent`, `call_test_gen_agent` at 50;
`call_secondary_orchestrator` at 100) now pass `recursion_limit` rather than inheriting
10000. Sizing: the worst *lawful* developer-agent run is modify mode at 12 tool calls ≈ 25
supersteps, so 50 is 2x headroom. This is a **tightening** — a runaway now stops in seconds
instead of thousands of paid calls — and it makes the bound a design decision rather than an
inherited default.

**Verification.** 6 helper scenarios plus end-to-end checks against the real linter tools; all
10 existing tool test cells and all 15 turn-based scenarios still pass. Confirmed
**load-bearing** by reverting the cap and re-running the same unfixable defect: 30 lint calls
and still going, versus 3 and a clean stop with the cap in place. A 25-check audit confirms
278 untouched cells are byte-identical.

---

## Tier 3 — Task-based — NOT STARTED

Planned in two halves, both already designed:

- **System behaviour:** 24 tests across 6 groups — persistence/restoration, state management,
  agent routing, memory store, pipeline sequencing, robustness.
- **Output quality:** the 9-metric evaluation campaign over 15 B2C prompts (reliability metrics
  grounded in MAESTRO; quality metrics in WebCoderBench, Lighthouse and Baymard), plus the
  MAS-vs-single-LLM baseline comparison and a descriptive cost/latency profile.

Note for tier 3 — the two correction loops are bounded differently, so their metrics are not
equally circular:

- **Final lint pass rate is circular.** The lint fix loop has no cap in code; the prompts say
  to iterate until "No issues found." and the store tools gate on `lint_passed`. It reaches
  ~100% by construction. Report it as process efficiency: first-pass rate and retry count.
- **Final TestExec pass rate is NOT circular.** The bug-fix correction loop is capped at
  `modification_attempts == 2`, enforced in code in `call_test_exec_agent` (it returns a
  REFUSED ToolMessage past the cap, independent of what the LLM decides). A run can and does
  end with tests still failing, so this is a real measurement. Report it three ways: success
  rate (passed within the cap), recovery rate (failed first, passed after correction), and
  residual failure rate (still failing at cap).

Independent quality evidence — a fresh Playwright render check, Lighthouse, LLM-as-judge —
remains the headline, because the system has no knowledge of those during generation.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WebForge: an autonomous multi-agent system (LangGraph + `deepagents`) that generates a
complete frontend-only B2C e-commerce webapp (HTML/CSS/JS) from a natural-language
request, then self-tests and self-corrects it via a Playwright-based execution pipeline.
Frontend delivery only — no backend/server code is generated. FYP (final year project),
currently **~50% complete**: HTML/CSS/JS generation, linting, coverage guards, the test
pipeline, and episodic memory are implemented; persistence (SqliteSaver/restart
restoration), the evaluation campaign, and the Streamlit preview/select sub-window are
still in progress — see `.claude/skills/` for the procedures that apply to each area.

**`fyp.ipynb` is the single source of truth for all agent, tool, and pipeline code.**
`fyp.py` is a superseded/read-only snapshot (old version, schemas only) — **never edit
fyp.py**; if you need current Pydantic schemas or graph logic, read them from
`fyp.ipynb`, not `fyp.py`. `langgraph.json` still points at `fyp.py:graph`, so that
config is stale relative to the notebook.

The repo root also contains many one-off scratch/debug scripts and dumped text files
from earlier sessions (`_*.txt`, `agent_cells*.txt`, `patch_*.py`, `insert_tests*.py`,
`verify_*.py`, `cell_*.txt`, `test.ipynb`, `bin.ipynb`, etc.) — these are not part of
the shipped system, don't treat them as architecture references.

## Setup & running

```bash
pip install -r requirements.txt      # Python deps
npm install                          # Node deps (html-validate, stylelint, eslint)
streamlit run stream_lit.py          # Chat UI, talks to the LangGraph backend
python test_schemas.py               # Pydantic schema smoke tests (fyp.py schemas)
```

There is no top-level Python test runner. **All unit tests for notebook code live
inside `fyp.ipynb` itself**, in a plain code cell placed directly under the code cell
they test — not in a separate file. This is a deliberate project convention: tests must
be visible and manually runnable next to the code they cover. See
[`notebook-editing`](.claude/skills/notebook-editing/SKILL.md) for how to add cells and
tests safely.

**Never execute a notebook cell that calls the live LLM** (agent `.stream()`/`.invoke()`
calls) — that spends the user's OpenAI credits. Verify such cells with `ast.parse` or a
read-through; only the user runs them.

## Architecture

Hierarchical multi-agent system, all built with `deepagents.create_agent` on a shared
LLM backbone (GPT-5.4 via `langchain-openai`). Every agent has its own `AgentState`
schema, tool set, and middleware stack — no agent reaches into another's internals;
they communicate only through typed state fields passed at call time (a blackboard-style
contract, not direct agent-to-agent messaging — see Constraints below).

```
Streamlit UI (stream_lit.py) ── bridge.py ──> Primary Orchestrator
                                                 ├─ call_context_engineering_agent
                                                 ├─ call_system_design_planning_agent
                                                 └─ call_secondary_orchestrator
                                                       ├─ call_html_agent   ┐
                                                       ├─ call_css_agent    ├─ generate → lint → fix loop
                                                       ├─ call_js_agent     ┘
                                                       ├─ call_test_gen_agent   (LLM: decides what to test)
                                                       └─ call_test_exec_agent  (deterministic pipeline, not an LLM agent)
```

**Primary Orchestrator** — extracts a structured `design_schema` (Context Engineering
Agent), produces four cross-linked specification schemas (System Planning Agent), then
delegates the build to the Secondary Orchestrator. Only two `intent` values exist:
`"generate"` and `"modify"`. There is no incremental/add-a-page mode — a run produces
one page or several pages at once from one shared set of specs; anything after that is
a `"modify"` op, never a structural regeneration (structural changes are excluded from
modify mode because the original spec would go stale once the page is hand-edited).

**Four specification schemas** (produced once per generation run, shared by every
developer agent and Test Generation): `LayoutSpecification` (pages/regions/layout
types), `ComponentSpecification` (component tree — tags, `componentCategory`, states,
events, ids/classNames), `InteractionSpecification` (event → action flows),
`DecorationSpecification` (visual styling per component/state). Cross-spec consistency
is enforced by `UISpecificationBundle` validators. Two derived contracts matter
everywhere downstream: the **Naming Contract** (spec `ComponentNode.id`/`className` →
HTML `id`/`class`) and the **Selector Contract** (`.className` if present, else `#id`,
used by Test Generation). Component state is materialized onto the DOM as
`aria-*`/`data-state` attributes (State Materialization Contract).

**Secondary Orchestrator** — drives HTML → (CSS + JS + Test Gen in parallel) → Test
Execution, with a bug-fix correction loop capped at `modification_attempts == 2` per run
(code-enforced in `call_test_exec_agent`, not just prompt-instructed). Has
`recall`/`retain` store middleware around an episodic-memory store (`fyp_episodes.db`,
embeddings-indexed): `recall_node` surfaces past bug-fix episodes before dev-agent
calls; `retain_node` fires only when a correction loop happened and then passed,
storing the fix as a work-order instruction, not post-hoc insight.

**Developer agents (HTML/CSS/JS)** — each generates → self-lints → fixes until its
linter is clean before writing the final file to `static/preview/preview_{session_id}.*`
(the "LLMs decide, Python guarantees" pattern: prompt instructs, a Python guard tool
enforces). Each store tool also runs a coverage validator rejecting incomplete
submissions before linting starts (every spec component addressable, required
`<link>`/`<script>` includes present, every decoration/interaction target resolved).
See [`lint-generated-code`](.claude/skills/lint-generated-code/SKILL.md).

**Test Generation → Test Execution** — Test Gen is a real LLM agent emitting structured
JSON test cases (not `.spec.ts`); Test Execution is a **deterministic pipeline stage,
not an LLM agent** — it interprets those cases directly against Playwright's Python
API. See [`playwright-test-pipeline`](.claude/skills/playwright-test-pipeline/SKILL.md).

## Coding conventions

- Locate notebook cells **by content, not index** — insertions repeatedly shift every
  cell after them; any index cited in a comment or memory file may be stale.
- Every new tool/function gets its unit test in the cell immediately below it, meant to
  be run manually by the user.
- Two independent test axes apply throughout: scope (unit/component/integration/system)
  and environment (offline-mocked/live-real-deps). A mocked test passing does not mean
  the live behavior works — several linter bugs here were only caught by live checks.

## Constraints

- **Never edit `fyp.py`** — read-only snapshot; all real work happens in `fyp.ipynb`.
- **Never execute live-LLM notebook cells** yourself — costs the user's OpenAI credits.
- Sub-agents (HTML/CSS/JS/Test Gen) communicate only through typed state fields set by
  the orchestrator tools (blackboard architecture) — direct agent-to-agent handoffs
  were deliberately rejected because every guard (attempt cap, memory hooks, work-order
  capture) lives on that orchestrator boundary and a peer channel would bypass them.
- `FilesystemMiddleware` root dirs are hardcoded absolute paths
  (`C:/Users/Lenovo/Desktop/FYP/static/preview`) in each developer agent's
  `create_agent` cell — a known portability gap, not yet fixed.

## Skills

Step-by-step procedures live under `.claude/skills/` rather than here — load the
relevant one before doing that kind of work:

| Skill | Use when |
|---|---|
| [`notebook-editing`](.claude/skills/notebook-editing/SKILL.md) | Inserting/patching cells or tests in `fyp.ipynb` |
| [`lint-generated-code`](.claude/skills/lint-generated-code/SKILL.md) | Running/debugging `run_html_validate`/`run_eslint`/`run_stylelint` |
| [`playwright-test-pipeline`](.claude/skills/playwright-test-pipeline/SKILL.md) | Writing/debugging test cases or the Test Execution pipeline |
| [`live-agent-suite`](.claude/skills/live-agent-suite/SKILL.md) | Building a live (real-LLM) end-to-end suite for an agent |
| [`evaluation-workflow`](.claude/skills/evaluation-workflow/SKILL.md) | Running the 9-metric evaluation campaign |
| [`mas-research-writing`](.claude/skills/mas-research-writing/SKILL.md) | Writing/reviewing the FYP report, chapters, or any architecture documentation |

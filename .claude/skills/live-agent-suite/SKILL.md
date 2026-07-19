---
name: live-agent-suite
description: Use when building a new live end-to-end test suite for an agent in fyp.ipynb (one that makes real GPT-5.4 calls), or extending an existing one. Mirrors the established _stream_html_agent pattern. These suites catch bugs offline mocked tests structurally cannot (e.g. the eslint/stylelint failures that only live checks found).
---

# Building a live agent-level test suite

Offline unit tests (see [`notebook-editing`](../notebook-editing/SKILL.md)) verify a
tool's logic with mocked inputs. A **live agent suite** is different in kind: it invokes
the real agent end-to-end with a real GPT-5.4 call, to catch bugs that only surface when
the actual LLM, the actual linter binaries, and the actual filesystem are all in play.
This is how the previously-broken `run_eslint`/`run_stylelint` were discovered — every
mocked test was green, but the live tool had never worked.

**These suites cost the user's OpenAI credits. Never execute one yourself — write it,
verify it structurally, and tell the user to run it manually.**

## Placement

One new cell, directly after the target agent's `create_agent(...)` cell.

## The five steps

### 1. Pre-write a fixture

Generation-mode agents read an existing file at Step 0 (or need one to modify against).
Write a fixture HTML/CSS/JS file to `static/preview/preview_{sid}.{ext}` that satisfies
the **Naming Contract** — real `id`/`class` values the agent's coverage validator and
the fixture's own spec will expect (e.g. `#navbar`, `.product-card`,
`.add-to-cart-btn`), plus any required `<link>`/`<script>` include tags if the target
agent's store tool checks for them.

### 2. Build the payload

Mirror the real `call_<agent>_agent` tool's payload construction exactly — don't
invent a simplified version, since a live suite's whole point is exercising the real
contract:

```python
payload = {
    "messages": [{"role": "user", "content": query}],
    "input_query": query,
    "session_id": session_id,
    "intent": intent,                 # "generate" or "modify"
    "target_page": target_page,
    "target_component": target_component,
    "target_instance": target_instance,
    # language-specific spec field, forwarded verbatim as the real orchestrator does:
    "decoration_specification": _decoration_spec,   # CSS agent
    # "interaction_specification": _interaction_spec,  # JS agent
    "css_code": "",                   # or js_code / html_code — empty for generation mode
    "lint_passed": False,
}
```

### 3. Stream with a `_stream_<agent>_agent` helper

Mirror `_stream_html_agent` (defined in the notebook near the HTML agent's live suite):

```python
def _stream_css_agent(session_id, query, intent, target_page="", target_component="",
                       target_instance=1, css_code="", decoration_specification=None):
    payload = {...}  # per step 2
    captured = {}
    def _merge(update):
        if isinstance(update, dict):
            for k, v in update.items():
                if k != "messages":
                    captured[k] = v
    for chunk in css_agent.stream(
        payload, stream_mode=["updates", "custom"], version="v2",
        config={"configurable": {"session_id": session_id}},
    ):
        # tuple (mode, data) or dict chunk handling:
        # mode == "custom" -> print progress
        # mode == "updates" -> _merge(data)
        ...
    return captured
```

### 4. Assert deterministically after the stream completes

- `captured["lint_passed"] is True`
- the emitted code is non-empty
- coverage-guard content is actually present — e.g. for CSS, the fixture's
  `.add-to-cart-btn`/`.product-card` classes have real rules; for JS, interaction
  targets resolve to selectors that appear in the emitted code
- the file on disk (`static/preview/preview_{sid}.{ext}`) matches the code in state —
  proves the agent actually wrote the file, not just returned text

### 5. Treat State Materialization as a soft check only

`aria-*`/`data-state` attribute emission (State Materialization Contract) varies in
exact phrasing between LLM runs. **Print it for inspection, don't hard-assert it** — a
hard assertion here will produce false failures unrelated to the thing you're actually
testing.

## Cell hygiene

- Comment the approximate cost at the top of the cell (e.g. "~1-3 GPT-5.4 calls per
  run") so the user knows what running it will spend.
- Verify the cell's structure with `ast.parse` before handing it back; do not run it.
- When telling the user how to run it, note which earlier cells must be executed first
  (agent definition, spec fixtures, any shared helper cells).

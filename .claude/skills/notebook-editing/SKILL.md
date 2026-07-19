---
name: notebook-editing
description: Use when inserting, patching, or editing cells in fyp.ipynb — the project's single source of truth (278+ cells and growing). Covers safely locating cells, the Write-tool patch-script pattern, where unit tests go, and how to validate a change without spending the user's OpenAI credits.
---

# Editing fyp.ipynb safely

`fyp.ipynb` is the only place real implementation code lives (`fyp.py` is a frozen,
read-only snapshot — never edit it). The notebook is large and has been patched many
times; naive edits break in predictable ways. Follow this procedure.

## 1. Locate the target cell by content, never by index

Cell indices are not stable — every insertion shifts every cell after it, and multiple
sessions' worth of insertions have already happened. An index remembered from an
earlier conversation or a comment in a memory file is likely stale.

Find the cell by a unique substring of its source (function name, a distinctive prompt
line, a variable name):

```python
import json
nb = json.load(open(r"C:/Users/Lenovo/Desktop/FYP/fyp.ipynb", encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "def store_html_code" in src:
        print(i, src[:200])
```

Only use the resulting index for that single patch run — re-locate by content again next
time, don't hardcode it forward.

## 2. Patch via a Write-tool Python script, not bash heredocs

Bash heredocs corrupt on notebook edits (quote/newline escaping breaks silently, and the
failure often isn't obvious until the notebook fails to load). Always:

1. Use the Write tool to create a standalone `.py` patch script in the scratchpad
   directory.
2. The script: loads the notebook JSON, finds the cell(s) by content match, mutates
   `cell["source"]` (a list of lines — `nbformat` expects each line to end with `\n`
   except the last), writes the JSON back with `json.dump(..., ensure_ascii=False,
   indent=1)`.
3. Run the script with Bash/PowerShell.

Skeleton:

```python
import json

path = r"C:/Users/Lenovo/Desktop/FYP/fyp.ipynb"
nb = json.load(open(path, encoding="utf-8"))

target_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "def store_css_code" in "".join(c["source"])
)

new_source = '''def store_css_code(...):
    ...
'''
nb["cells"][target_idx]["source"] = new_source.splitlines(keepends=True)

json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"patched cell {target_idx}")
```

To **insert** a new cell (e.g. a test cell directly under a code cell), build a cell
dict and `nb["cells"].insert(idx + 1, new_cell)`:

```python
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": test_source.splitlines(keepends=True),
}
```

## 3. Where unit tests go

Every new tool/function gets its unit test in a **plain code cell placed directly below
it** in the notebook — not in an external `.py` test file. The user runs these manually.
This is in addition to, not instead of, any scratch verification script used while
developing the change (scratch scripts are your own internal check; the in-notebook
test cell is the deliverable).

Unit tests for notebook tools typically need lightweight shims instead of the real
`deepagents`/LangGraph runtime, so they run in plain Python without spinning up an
agent:

```python
class Command:
    def __init__(self, update=None): self.update = update or {}

class ToolMessage:
    def __init__(self, content, tool_call_id=None):
        self.content = content; self.tool_call_id = tool_call_id

def tool(fn=None, **kw):
    def wrap(f):
        return types.SimpleNamespace(func=f, name=f.__name__)
    return wrap(fn) if callable(fn) else wrap

State = dict
ToolRuntime = dict
```

Execute a cell's source with `exec(compile(source, f"<cell {i}>", "exec"), globals_dict)`
to run it standalone against these shims when writing/verifying a scratch check.

## 4. Validate before handing back

- `ast.parse(new_source)` on every cell you wrote or edited — catches syntax errors
  immediately, no notebook load required.
- After saving, load the notebook with `nbformat` (or `json.load` + spot check) to
  confirm it's still valid JSON and the cell count/order is what you expect.
- **Do not execute any cell that calls a live LLM** (`agent.stream(...)`,
  `agent.invoke(...)`, anything hitting `gpt-5.4`) — that spends the user's OpenAI
  credits. Verify those cells by reading them and `ast.parse`, then tell the user which
  cells to run manually and roughly how many LLM calls each will make.

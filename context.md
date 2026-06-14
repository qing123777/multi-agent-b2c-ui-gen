# FYP Project Context — AI-Driven B2C UI Generation System

Paste this into any new conversation to restore full context instantly.

---

## Project Overview

Multi-agent system that generates a complete **B2C e-commerce** single-page webapp
(HTML + CSS + JS) from a user's natural language request.

- **Framework**: `deepagents` + `LangGraph` + `LangChain`
- **LLM**: `ChatOpenAI(model="gpt-5.4", temperature=0)` — gpt-5.4 is valid (released March 2026)
- **Frontend**: Streamlit (chat-based UI)
- **Key files**: `fyp.py` (backend + all schemas), `fyp.ipynb` (notebook dev),
  `bridge.py`, `stream_lit.py`
- **Test file**: `test_schemas.py` — 48/48 passing
- **Visualization**: `spec_crossref.html` — interactive cross-reference diagram

---

## System Architecture

```
User (Streamlit)
    └── Orchestrator Agent
             ├── Context Engineering Agent   → extracts intent → design_schema
             ├── System Planning Agent       → generates 4 UI specifications
             └── Secondary Orchestrator      → delegates to developer agents
                      ├── HTML Agent         → generates + lints + saves HTML  ← current focus
                      ├── CSS Agent          → (pending)
                      ├── JS Agent           → (pending)
                      ├── Test Generation Agent  → (pending)
                      └── Test Executor Agent    → (pending)
```

---

## B2C Specification Schemas — Current State (fyp.py)

All four schemas were revised. Below is the authoritative summary.

### Enums / Types

```python
AgentType    = Literal["HTML", "CSS", "JavaScript"]

PageType     = Literal["plp", "pdp", "cart", "checkout", "account", "search"]

LayoutType   = Literal["grid","flex-row","flex-column","single-column",
                        "multi-column","absolute","stack"]

ActionType   = Literal[
    "updateState", "navigate", "toggleVisibility", "updateData", "showFeedback", "emit",
    # B2C e-commerce actions:
    "addToCart", "updateQuantity", "applyFilter", "clearFilter", "toggleWishlist",
]

ComponentCategory = Literal[
    "navbar", "footer", "breadcrumb",
    "product-card", "product-grid", "product-image-gallery", "product-price", "product-rating",
    "filter-panel", "sort-bar",
    "mini-cart", "cart-item", "add-to-cart-button", "order-summary", "checkout-form", "payment-form",
    "size-picker", "quantity-selector", "wishlist-button",
    "search-bar", "search-results", "pagination", "promo-banner",
    "account-profile", "order-history", "hero-section",
]

# REMOVED: ComponentRole, LayoutComponentRef
```

### New models

```python
class DataBindingSpec(BaseModel):
    source: str            # e.g. "product.price", "cart.items"
    format: Optional[str]  # "currency" | "list" | "count" | "date" | "percentage"

class ActionResultState(BaseModel):
    description: str       # e.g. "Cart count increments; button shows Added"
```

### LayoutSpecification

```python
class Region(BaseModel):
    name: str;  ownedBy: AgentType;  layoutType: LayoutType
    # componentRefs REMOVED — ComponentSpec owns placement

class Page(BaseModel):
    pageName: str;  pageType: PageType  # NEW
    pageLayoutType: LayoutType;  responsive: bool = True;  regions: List[Region]
```

### ComponentSpecification

```python
class ComponentNode(BaseModel):
    id: str;  tag: HTMLTag;  ownedBy: AgentType
    componentCategory: Optional[ComponentCategory]  # NEW — maps to data-component in HTML
    props: Dict[str, Any]
    dataBinding: Optional[DataBindingSpec]           # was Optional[str]
    states: List[ComponentState];  events: List[ComponentEvent]
    className: Optional[str];  children: List["ComponentNode"]
    # role REMOVED, layoutRef REMOVED
```

### InteractionSpecification

```python
class InteractionAction(BaseModel):
    type: ActionType                          # no "apiCall", no "log"
    targetId: Optional[str];  payload: Optional[Dict[str, Any]]
    result_state: Optional[ActionResultState] # was Optional[Dict] + coerce workaround
```

### DecorationSpecification — unchanged

### Cross-Spec Linking Keys (enforced by UISpecificationBundle)

| Key | From | To |
|-----|------|----|
| `pageName` | Layout | Component, Interaction, Decoration |
| `region.name` | Layout (defines) | Component (regionComponents dict key) |
| `component.id` | Component | Interaction (targetId), Decoration (componentId) |
| `className` | Component | Decoration (must match verbatim) |
| `state.name` | Component | Interaction (toState/fromState), Decoration (stateVariant) |
| `flow.name` | Component (events.flowName) | Interaction (InteractionFlow.name) |

---

## HTML Agent — Current State (fyp.ipynb)

### State Schema (Cell 102)

```python
class State(AgentState):
    user_query:         str
    html_specification: str      # ← NEEDS SPLITTING (see roadmap)
    html_code:          str
    lint_passed:        bool = False
    session_id:         str
    intent:             Literal["generate", "modify"] = "generate"
    target_page:        str = ""
    target_component:   str = ""
    target_instance:    int
```

### Tools

| Cell | Tool | Key behaviour |
|------|------|---------------|
| 105 | `run_html_validate` | Reads `html_code` from `runtime.state` (NOT a param); lints via `npx html-validate`; updates `lint_passed` |
| 108 | `store_html_code(html_code)` | Stores html_code into state |
| 111 | `extract_component()` | Reads target_page/component/instance from state; extracts block via BeautifulSoup |
| 114 | `store_html_diff(diff_output)` | Applies unified diff; splices back into html_code |
| 117 | `validate_component_tags()` | Checks ALL `nav/header/footer/section/main/aside/article` at any depth for `data-page`, `data-component`, `data-instance` |

```python
COMPONENT_TAGS = {"nav", "header", "footer", "section", "main", "aside", "article"}
```

### Current Prompt Structure (Cell 120)

```
_html_shared_base          → role, context fields, tagging convention, strict rules
_html_generation_prompt    = _html_shared_base + 7-step workflow
_html_modification_prompt  = _html_shared_base + 8-step workflow
```

**Generation workflow (do NOT change):**
1. Generate HTML from html_specification
2. `store_html_code(html_code)`
3. `validate_component_tags()` → fix issues, repeat until clean
4. `run_html_validate()` → fix issues, repeat until clean
5. `write_file(f"preview_{session_id}.html", html_code)` only when `lint_passed = True`

**Modification workflow (do NOT change):**
`extract_component()` → diff → `store_html_diff()` → `validate_component_tags()`
→ `run_html_validate()` → `write_file(...)`

### Dynamic Prompt Selector (Cell 125)

```python
@dynamic_prompt
def html_prompt_selector(request) -> str:
    intent = (request.state or {}).get("intent", "generate")
    return _html_modification_prompt if intent == "modify" else _html_generation_prompt
```

### create_agent (Cell 129)

```python
html_agent = create_agent(
    model=ChatOpenAI(model="gpt-5.4", temperature=0),
    tools=[store_html_code, run_html_validate, extract_component,
           store_html_diff, validate_component_tags],
    middleware=[
        FilesystemMiddleware(backend=FilesystemBackend(
            root_dir="C:/Users/Lenovo/Desktop/FYP/static/preview", virtual_mode=True)),
        html_prompt_selector,
        html_agent_progress,    # @before_agent
        html_tool_progress,     # @wrap_tool_call
    ],
    state_schema=State,
)
```

---

## HTML Agent System Prompt — What to Revise (NEXT TASK)

### Must-change (stale references to old schema fields)

| Location | Problem | Fix |
|----------|---------|-----|
| State schema Cell 102 | `html_specification: str` is a flat string | Split into `layout_specification: dict` + `components_specification: dict` |
| `_html_shared_base` Context section | Lists `html_specification` | Update to `layout_specification`, `components_specification` |
| `_html_shared_base` Tagging Convention | `data-page` examples use `"home"`, `"product"` | Use B2C `pageName` values (`plp`, `pdp`, `cart`…) |
| `_html_shared_base` Tagging Convention | `data-component` from agent's judgment | Must come from `componentCategory` in the spec |
| Generation Step 1 | "Read the full html_specification" | "Read `layout_specification` and `components_specification`" |

### Should-change (new fields the agent needs to know about)

| What | Instruction to add |
|------|--------------------|
| `componentCategory` → `data-component` | Use `componentCategory` value as `data-component`. If `componentCategory` is None on a COMPONENT_TAGS element, derive from component `id` in kebab-case |
| `dataBinding` → placeholder content | Use `dataBinding.source` + `format` to write placeholder content (e.g. `source="product.price"`, `format="currency"` → `<span>$0.00</span>`). Mock data only |
| `pageType` → structural guidance | Use `pageType` to know expected structure: `plp`=grid+filters+sort, `pdp`=gallery+details+add-to-cart, `cart`=line-items+quantity+summary, etc. Do not invent components not in the spec |

---

## Pending Notebook Fixes (fyp.ipynb)

| Cell | Bug | Fix | Status |
|------|-----|-----|--------|
| 117 | Docstring says "children not checked" — wrong, all depths checked | Replace with 4-line correct docstring | ✅ Done |
| 118 | `GOOD_HTML` uses `<div>` for product-cards | Change `<div>` → `<section>` | ✅ Done |
| 123 | `"run_html_validator"` typo — progress message never fires | Fix to `"run_html_validate"` | ✅ Done |
| 168 | Duplicate of Cell 127 instead of `call_html_agent` tests | Replace with Section 7 tests | ✅ Done |

### Supplementary HTML Agent Tool Tests — Not Yet Added to fyp.ipynb
Add as new cell after Cell 118. Sections:
- **S1** `_apply_patch` — 7 cases
- **S2** `run_html_validate` — rv-6 malformed JSON, rv-7 stderr fallback, rv-8 tempfile cleanup
- **S3** `store_html_code` — sc-4 empty string, sc-5 exact message
- **S4** `extract_component` — ec-6 instance=0 rejected, ec-7 first of three
- **S5** `store_html_diff` — sd-6 instance 2 only, instance 1 untouched
- **S6** `validate_component_tags` — vt-6 untagged, vt-7 nested flagged, vt-8 div-only passes, vt-9 5 issues
- **S7** `call_html_agent` — ✅ Done (Cell 168, 16 cases using `_run_cla()` helper)

---

## Secondary Orchestrator — call_html_agent (Cell 166)

```python
@tool
def call_html_agent(
    runtime: ToolRuntime[state],   # state = orchestrator state
    query: str,
    intent: str = "generate",
    target_page: str = "",
    target_component: str = "",
    target_instance: int = 1,
) -> Command:
    # streams html_agent, forwards custom chunks to stream writer
    # captured dict built but NOT returned in Command.update (intentional — file writes only)
```

**Cell 168 Bug**: duplicate of Cell 127. Replace with Section 7 `call_html_agent` tests.

---

## CSS / JS Agents — Design Decisions (not yet implemented)

| Tool | CSS | JS |
|------|-----|----|
| `store_css_code` / `store_js_code` | Copy-paste rename | Copy-paste rename |
| `run_stylelint` / `run_eslint` | Rewrite — different CLI + JSON schema | Rewrite |
| `extract_component`, `store_html_diff`, `validate_component_tags` | Drop | Drop |

**Linter JSON differences:**

| Linter | Error array key | Message key | Rule key |
|--------|----------------|-------------|----------|
| html-validate | `messages` | `message` | `ruleId` |
| stylelint | `warnings` | `text` | `rule` |
| eslint | `messages` | `message` | `ruleId` |

**`run_eslint` bug (Cell 134)**: takes `js_code` as direct param — must read from `runtime.state`.
**CSS/JS modification mode**: full regeneration only — no extract/diff.

---

## Key Patterns Reference

### Tool Return Pattern
```python
return Command(update={
    "field_name": value,
    "messages": [ToolMessage(content="...", tool_call_id=runtime.tool_call_id)]
})
```

### Testing Decorators
| Decorator | How to test |
|-----------|-------------|
| `@tool` | `tool_fn.func(args)` |
| `@dynamic_prompt` | call plain function directly |
| `@before_agent` | call plain function directly |
| `@wrap_tool_call` | call plain function directly |

### Preview Files
```
C:/Users/Lenovo/Desktop/FYP/static/preview/
    ├── preview_{session_id}.html
    ├── preview_{session_id}.css
    └── preview_{session_id}.js
```

### stream_lit.py Intent Logic (ready, not yet wired)
```python
intent = "modify" if st.session_state.get("target_component") else "generate"
graph.invoke({
    "messages": [...], "intent": intent,
    "target_page":      st.session_state.get("target_page", ""),
    "target_component": st.session_state.get("target_component", ""),
    "target_instance":  st.session_state.get("target_instance", 1),
})
st.session_state.pop("target_component", None)
st.session_state.pop("target_page",      None)
st.session_state.pop("target_instance",  None)
```

---

## Development Roadmap

```
✅ B2C Spec Schema Revision — DONE
   ├── Added: PageType, ComponentCategory (24 values), DataBindingSpec, ActionResultState
   ├── Removed: ComponentRole, LayoutComponentRef, componentRefs, role, layoutRef
   ├── ActionType: removed apiCall/log; added 5 B2C actions
   ├── test_schemas.py — 48/48 passing (10 sections)
   └── spec_crossref.html — interactive cross-reference diagram

✅ HTML Agent System Prompt Revision — DONE
   ├── Split State: html_specification → layout_specification + components_specification (Cell 102)
   ├── Updated _html_shared_base: context fields + tagging convention (componentCategory, B2C pageName)
   └── Updated generation Step 1: pageType structure, dataBinding placeholders (Cell 120)

⏳ Notebook Fixes — NEXT
   ├── ✅ Cell 117: docstring corrected (all-depths wording)
   ├── ✅ Cell 118: GOOD_HTML div → section for product-card
   ├── ✅ Cell 123: run_html_validator typo fixed
   ├── ✅ Cell 168: replaced duplicate with 12-case call_html_agent tests (S7)
   └── ⬜ Supplementary tool tests S1–S6 (new cell after Cell 118)

⬜ CSS Agent + JS Agent
   ├── store_css_code / store_js_code (rename)
   ├── run_stylelint / run_eslint (rewrite)
   ├── CSS/JS system prompts (generation only, no extract/diff)
   └── call_css_agent / call_js_agent in secondary orchestrator

⬜ Test Generation Agent + Test Executor Agent (Playwright)

⬜ Integration — stream_lit.py intent logic + end-to-end test
```

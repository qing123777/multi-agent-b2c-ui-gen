# UIGPT — AI-Driven B2C UI Generation System (In Progress)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-1C3C3C?logo=chainlink&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-1C3C3C?logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--5.4-412991?logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-chat%20UI-FF4B4B?logo=streamlit&logoColor=white)
![deepagents](https://img.shields.io/badge/deepagents-0.3+-4a90d9)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> A **multi-agent system** that generates a complete B2C e-commerce single-page webapp (HTML + CSS + JS) from a natural language request — orchestrated by a LangGraph agent graph, with per-language developer agents that generate, lint, and iteratively self-correct their own output.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Agent Hierarchy](#agent-hierarchy)
   - [B2C Specification Schemas](#b2c-specification-schemas)
   - [HTML Agent Detail](#html-agent-detail)
3. [Tech Stack](#tech-stack)
4. [Key Characteristics](#key-characteristics)
5. [Project Structure](#project-structure)
6. [Setup & Usage](#setup--usage)
7. [Development Roadmap](#development-roadmap)
8. [Contributors](#contributors)

---

## Overview

UIGPT takes a user's natural language description of a B2C e-commerce page and produces a fully linted, semantically tagged HTML/CSS/JS single-page webapp.

The pipeline is driven by a **hierarchical multi-agent system**:

1. A top-level **Orchestrator** receives the user request via a Streamlit chat UI.
2. A **Context Engineering Agent** extracts structured intent into a typed `design_schema`.
3. A **System Planning Agent** generates four cross-linked UI specification schemas (Layout, Component, Interaction, Decoration).
4. A **Secondary Orchestrator** delegates to per-language **Developer Agents** (HTML, CSS, JS) that each generate, self-lint, and iteratively fix their output before writing the final file.

The generated preview files are served live inside the Streamlit UI.

---

## Architecture

### Agent Hierarchy

```
User (Streamlit)
    └── Orchestrator Agent
             ├── Context Engineering Agent   → extracts intent → design_schema
             ├── System Planning Agent       → generates 4 UI specification schemas
             └── Secondary Orchestrator      → delegates to developer agents
                      ├── HTML Agent         → generate → lint → fix → save  ✅
                      ├── CSS Agent          → (in progress)
                      ├── JS Agent           → (in progress)
                      ├── Test Generation Agent  → (pending)
                      └── Test Executor Agent    → (pending)
```

Every agent is built with `deepagents.create_agent` + a LangGraph `AgentState`, giving each agent its own tool set, middleware stack, and state schema while sharing the same LLM backbone (`gpt-5.4`).

### B2C Specification Schemas

The System Planning Agent produces four strongly-typed Pydantic schemas that act as a structured contract between planning and implementation:

| Schema | Purpose |
|---|---|
| `LayoutSpecification` | Page-level layout: pages, regions, layout types (`grid`, `flex-row`, …) |
| `ComponentSpecification` | Component tree: tags, `componentCategory`, data bindings, states, events |
| `InteractionSpecification` | Event → action flows: `addToCart`, `applyFilter`, `toggleWishlist`, … |
| `DecorationSpecification` | Visual styling directives per component and state variant |

Cross-spec consistency (matching `pageName`, `region.name`, `component.id`, `className`, state names, flow names) is enforced at runtime by `UISpecificationBundle`.

**Supported B2C page types:** `plp` · `pdp` · `cart` · `checkout` · `account` · `search`

**Supported component categories (24):** `navbar`, `footer`, `product-card`, `product-grid`, `product-image-gallery`, `add-to-cart-button`, `filter-panel`, `sort-bar`, `mini-cart`, `cart-item`, `order-summary`, `checkout-form`, `payment-form`, `size-picker`, `quantity-selector`, `wishlist-button`, `search-bar`, `search-results`, `pagination`, `promo-banner`, `account-profile`, `order-history`, `hero-section`, `breadcrumb`

### HTML Agent Detail

The HTML Agent is a `deepagents` agent wired with five tools and two middleware layers:

| Tool | Behaviour |
|---|---|
| `store_html_code(html_code)` | Writes generated HTML into agent state |
| `run_html_validate()` | Lints via `npx html-validate`; sets `lint_passed` flag |
| `validate_component_tags()` | Checks all semantic tags (`nav`, `header`, `footer`, `section`, `main`, `aside`, `article`) for required `data-page`, `data-component`, `data-instance` attributes at all depths |
| `extract_component()` | Extracts a named component block for modification mode |
| `store_html_diff(diff_output)` | Applies a unified diff and splices it back into `html_code` |

**Generation workflow (self-correcting loop):**
1. Read `layout_specification` and `components_specification` from state
2. `store_html_code(html_code)`
3. `validate_component_tags()` → fix until clean
4. `run_html_validate()` → fix until `lint_passed = True`
5. `write_file(f"preview_{session_id}.html", html_code)`

**Modification workflow:**
`extract_component()` → produce unified diff → `store_html_diff()` → `validate_component_tags()` → `run_html_validate()` → `write_file(...)`

A `@dynamic_prompt` middleware selects between the generation and modification system prompts based on the `intent` field in state (`"generate"` | `"modify"`).

---

## Tech Stack

**Backend / Agents**
| Component | Library |
|---|---|
| Agent framework | `deepagents` |
| Agent orchestration | `langgraph` |
| LLM chains | `langchain`, `langchain-openai`, `langchain-anthropic` |
| LLM | `gpt-5.4` (OpenAI) via `langchain-openai` |
| Web search | `langchain-tavily`, `tavily-python` |
| PDF output | `reportlab` |
| Schema validation | Pydantic (via LangChain) |

**Frontend**
| Component | Technology |
|---|---|
| Chat UI | Streamlit |
| Preview serving | Static file server (`bridge.py`) |
| Theme | Custom CSS — dual light/dark palette, collapsible sidebar |

**Tooling**
| Component | Library / Tool |
|---|---|
| HTML linting | `html-validate` (Node, via `npx`) |
| CSS linting | `stylelint` (planned) |
| JS linting | `eslint` (planned) |
| E2E testing | Playwright (planned) |

---

## Key Characteristics

### 1. Hierarchical Multi-Agent Orchestration
The pipeline separates concerns across specialised agents: intent extraction, multi-spec planning, and per-language code generation are handled by distinct agents that communicate through typed state schemas. No agent "knows" about another's internals.

### 2. Four-Schema Structured Planning
Rather than generating code directly from a natural language request, the System Planning Agent first produces four cross-linked Pydantic schemas. These act as a verifiable, machine-readable contract that constrain what the developer agents are allowed to generate — preventing hallucinated components or inconsistent page structure.

### 3. Self-Correcting Developer Agents
Each developer agent (HTML, CSS, JS) runs a self-correcting loop: generate → lint → fix → repeat until the linter reports zero errors. The agent never writes the final file until `lint_passed = True`, making the generated output syntactically valid by construction.

### 4. Semantic Component Tagging
Every semantic HTML element (`nav`, `header`, `footer`, `section`, `main`, `aside`, `article`) is required to carry three `data-*` attributes — `data-page`, `data-component` (sourced from `componentCategory` in the spec), and `data-instance` — enforced by `validate_component_tags()` before any file is written. This makes the DOM directly addressable for modification mode and future test automation.

### 5. Generate / Modify Duality
The HTML Agent supports two intents controlled by a single `intent` state field. Modification mode surgically extracts the target component, applies a unified diff, and re-validates — leaving the rest of the page untouched. The Streamlit UI sets the intent automatically based on whether the user has selected a `target_component`.

### 6. Streaming Progress UI
Intermediate agent progress events (`[HTML Agent] Generating…`, `[Orchestrator] Planning…`) are streamed in real-time to a collapsible expander next to each message and persisted across `st.rerun()` so the processing log survives page interactions.

---

## Project Structure

```
.
├── fyp.ipynb                  # Main notebook — all agent definitions and tests
├── fyp.py                     # Backend: all Pydantic schemas + chain logic
├── stream_lit.py              # Streamlit chat UI
├── bridge.py                  # Streaming bridge between graph and Streamlit
├── agent_graph.py             # Standalone LangGraph agent example
├── agent_streaming.py         # Streaming utilities
├── requirements.txt           # Python dependencies
├── package.json               # Node dev dependencies (html-validate, stylelint, eslint)
├── langgraph.json             # LangGraph Studio config
│
├── static/
│   └── preview/               # Auto-generated — live preview files per session
│       ├── preview_{session_id}.html
│       ├── preview_{session_id}.css
│       └── preview_{session_id}.js
│
├── test_schemas.py            # Schema unit tests — 48/48 passing
├── spec_crossref.html         # Interactive cross-reference diagram for the 4 schemas
└── context.md                 # Project context snapshot for LLM onboarding
```

---

## Setup & Usage

### 1. Prerequisites

- Python 3.10+
- Node.js 18+ (for `html-validate`, `stylelint`, `eslint`)
- An OpenAI API key

### 2. Clone the repository

```bash
git clone https://github.com/qing123777/FYP.git
cd FYP
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Node dependencies

```bash
npm install
```

### 5. Set your API key

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."

# macOS / Linux
export OPENAI_API_KEY="sk-..."
```

### 6. Run the Streamlit UI

```bash
streamlit run stream_lit.py
```

The chat UI will open in your browser. Type a description of the B2C page you want to generate (e.g. *"A product listing page for a sneaker store with filters and a sort bar"*) and the agent pipeline will produce a live HTML preview.

### 7. Run schema tests

```bash
python test_schemas.py
```

---

## Development Roadmap

```
✅ B2C Spec Schema Revision
   ├── Added: PageType, ComponentCategory (24 values), DataBindingSpec, ActionResultState
   ├── Removed: ComponentRole, LayoutComponentRef, componentRefs, role, layoutRef
   ├── ActionType: removed apiCall/log; added 5 B2C-specific actions
   ├── test_schemas.py — 48/48 passing (10 sections)
   └── spec_crossref.html — interactive cross-reference diagram

✅ HTML Agent — Generation + Modification
   ├── Split State: html_specification → layout_specification + components_specification
   ├── Updated system prompt: componentCategory → data-component, B2C pageType guidance
   ├── validate_component_tags: all-depths check for semantic elements
   └── Self-correcting lint loop (html-validate)

⏳ HTML Agent — Supplementary Tests (S1–S6 tool unit tests)

⬜ CSS Agent
   ├── store_css_code, run_stylelint
   └── Generation-only system prompt (no extract/diff)

⬜ JS Agent
   ├── store_js_code, run_eslint
   └── Generation-only system prompt

⬜ Test Generation Agent + Test Executor Agent (Playwright)

⬜ End-to-end integration — stream_lit.py intent wiring + full pipeline test
```

---

## Contributors

**Name**
1. [Lim Qing](https://github.com/qing123777)

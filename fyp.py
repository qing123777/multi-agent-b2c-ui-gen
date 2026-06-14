"""Notebook-synced Context Engineering + Orchestrator agents.

This file is meant to be imported by `fyp.ipynb` (e.g., `importlib.reload(fyp)`).

Exports:
- `context_engineering_agent`
- `orchestrator_agent`
- `graph` (alias to `orchestrator_agent`)

Optional:
- LangSmith tracing helpers (`enable_langsmith`, `langsmith_config`).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_agent, wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator
from tavily import TavilyClient

load_dotenv()

def _bootstrap_langsmith_env() -> None:
    """Best-effort env var normalization for LangSmith tracing.

    Some setups (and older tutorials) use LANGSMITH_* vars while LangChain's
    tracer expects LANGCHAIN_* vars. This function maps between them without
    leaking secrets or overriding user-provided values.
    """

    # API key
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)

    # Tracing enabled flag
    tracing_flag = os.getenv("LANGSMITH_TRACING")
    if tracing_flag:
        os.environ.setdefault("LANGSMITH_TRACING", tracing_flag)
        # LangChain expects "LANGCHAIN_TRACING_V2=true"
        if tracing_flag.strip().lower() in {"1", "true", "yes", "on"}:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    # Endpoint
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    if endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)

    # Project
    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")
    if project:
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
        os.environ.setdefault("LANGSMITH_PROJECT", project)



def _disable_checkpointer(compiled_graph: Any) -> None:
    """LangGraph Studio/API manages persistence; clear any local checkpointer."""
    try:
        compiled_graph.checkpointer = None
    except Exception:
        return


# -----------------------------------------------------------------------------
# LangSmith (optional)
# -----------------------------------------------------------------------------


def enable_langsmith(*, project: Optional[str] = None) -> bool:
    """Enable LangSmith tracing via env vars.

    This is a no-op unless either `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY` is set.
    """

    _bootstrap_langsmith_env()

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return False

    # If project not explicitly provided, fall back to either env var.
    if not project:
        project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")

    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint or "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint or "https://api.smith.langchain.com")
    if project:
        # LangChain uses LANGCHAIN_PROJECT; many people set LANGSMITH_PROJECT.
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
        os.environ.setdefault("LANGSMITH_PROJECT", project)
    return True


# Bootstrap tracing if possible (keeps behavior deterministic across notebooks/Studio).
_bootstrap_langsmith_env()
enable_langsmith()


def langsmith_config(
    *,
    run_name: Optional[str] = None,
    project: Optional[str] = None,
    thread_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a runnable config dict you can pass to `.invoke(..., config=...)`."""

    enable_langsmith(project=project)
    cfg: Dict[str, Any] = {}
    if run_name:
        cfg["run_name"] = run_name
    if tags:
        cfg["tags"] = tags
    if metadata:
        cfg["metadata"] = metadata
    # Many LangGraph graphs require a thread_id when a checkpointer is present.
    # Auto-generate one so ad-hoc scripts/notebooks don't fail.
    if not thread_id:
        thread_id = f"run_{uuid.uuid4().hex[:12]}"
    cfg.setdefault("configurable", {})["thread_id"] = thread_id
    return cfg


def _merge_configs(base: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(base or {})
    extra = dict(extra or {})

    for k, v in extra.items():
        if k == "configurable" and isinstance(v, dict):
            base.setdefault("configurable", {})
            if isinstance(base["configurable"], dict):
                base["configurable"].update(v)
        else:
            base[k] = v
    return base


# -----------------------------------------------------------------------------
# Design schema
# -----------------------------------------------------------------------------


FeedbackComponent = Literal[
    "modal",
    "toast",
    "tooltip",
    "popover",
    "progress_indicator",
    "loading_indicator",
]


class UI(BaseModel):
    type: Optional[str] = Field(
        default=None,
        description=(
            "Primary page/UI components requested. "
            "(e.g., landing_page, product_listing, product_detail, cart, checkout, account, search)"
        ),
    )
    variant: Optional[str] = Field(
        default=None,
        description="Layout variant (e.g., grid, list, split, tabs, wizard).",
    )
    style: Optional[str] = Field(
        default=None,
        description="Overall visual style (e.g., modern, minimal, classic, bold).",
    )
    density: Optional[str] = Field(
        default=None,
        description="Information density / spacing (e.g., comfortable, compact).",
    )
    navigation: Optional[str] = Field(
        default=None,
        description="Navigation pattern (e.g., top_nav, side_nav, hamburger, none).",
    )
    feedback_components: List[FeedbackComponent] = Field(
        default_factory=list,
        description="Feedback UI elements explicitly requested.",
    )


class Theme(BaseModel):
    primary_color: Optional[str] = Field(default=None, description="Primary brand color.")
    secondary_color: Optional[str] = Field(default=None, description="Secondary/accent color.")
    mode: Optional[str] = Field(default=None, description="Light or dark mode preference.")


class Features(BaseModel):
    search: Optional[bool] = Field(default=None, description="Whether product search UI is needed.")
    filter: Optional[bool] = Field(default=None, description="Whether filtering UI is needed.")
    sorting: Optional[bool] = Field(default=None, description="Whether sorting UI is needed.")
    cart: Optional[bool] = Field(default=None, description="Whether cart UI is needed.")
    wishlist: Optional[bool] = Field(default=None, description="Whether wishlist UI is needed.")
    reviews: Optional[bool] = Field(default=None, description="Whether reviews/ratings UI is needed.")


FIXED_TECH_STACK: List[Literal["html", "css", "javascript"]] = ["html", "css", "javascript"]


class Constraints(BaseModel):
    tech_stack: List[Literal["html", "css", "javascript"]] = Field(
        default_factory=lambda: FIXED_TECH_STACK.copy(),
        description="Fixed constant. Always ['html','css','javascript']; do not change.",
    )

    @field_validator("tech_stack", mode="before")
    @classmethod
    def force_fixed_tech_stack(cls, _v):
        return FIXED_TECH_STACK.copy()


class DesignSchema(BaseModel):
    ui: UI = Field(default_factory=UI)
    theme: Theme = Field(default_factory=Theme)
    features: Features = Field(default_factory=Features)
    constraints: Constraints = Field(default_factory=Constraints)
    extra_requirements: List[str] = Field(
        default_factory=list,
        description=(
            "Catch-all requirements that don't fit other fields. "
            "Use short, specific strings (<= 12 words each)."
        ),
    )


class state(AgentState):
    input_query: str = Field(description="User request for this run.")
    design_schema: Dict[str, Any] = Field(default_factory=dict)
    tavily_search_result: str = Field(default="")


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


_schema_model = ChatOpenAI(model_name="gpt-5.4", temperature=0.0)

_extract_prompt = PromptTemplate.from_template(
    """Given query: {query}

Extract the user requirements into DesignSchema.

Rules:
- Return only valid JSON matching DesignSchema.
- Do not include markdown, explanation, or extra keys.
- Use null or [] when a field is not specified.
- Do not invent brand names, colors, or features.
- If a requirement does not cleanly map to a field, put it in extra_requirements as a short string (<= 12 words).
- If the query is ambiguous, make the most conservative extraction possible.

Now extract from the real query."""
)

_extract_chain = _extract_prompt | _schema_model.with_structured_output(DesignSchema, method="function_calling")
_extract_tool = _extract_chain.as_tool(
    name="extract_intent_signals",
    description="Extract renderable UI/UX intent signals into DesignSchema",
    arg_types={"query": str},
)


@tool
def extract_intent_signals(runtime: ToolRuntime[state]) -> Command:
    """Extracts intent signals."""
    s = runtime.state
    query = s.get("input_query", "")
    result_obj = _extract_tool.invoke({"query": query})
    result = result_obj.model_dump() if hasattr(result_obj, "model_dump") else result_obj
    return Command(
        update={
            "design_schema": result,
            "messages": [
                ToolMessage(
                    content="   [Context Agent] Storing design schema from intent extraction...",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


_str_parser = StrOutputParser()
_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

_tavily_search_query_prompt = PromptTemplate.from_template(
    "Given user query: {query}\nRewrite it into a short web search query (<= 12 words).\nReturn only the rewritten search query."
)
_tavily_search_query_chain = _tavily_search_query_prompt | _schema_model | _str_parser


def _safe_tavily_search(search_query: str) -> str:
    if not search_query.strip():
        return ""
    try:
        res = _tavily_client.search(search_query, max_results=5)
        return json.dumps(res, ensure_ascii=False)
    except Exception:
        return ""


_tavily_summary_prompt = PromptTemplate.from_template(
    "Given user query: {query}\nRaw Tavily results (JSON): {search_result}\n\n"
    "Task: Summarize ONLY time-sensitive UI/UX relevant context into <= 6 bullet points.\n"
    "Rules:\n- Output plain text only.\n- If results are empty/irrelevant, return empty string."
)

_tavily_chain = (
    {"query": lambda x: x["query"], "search_query": _tavily_search_query_chain}
    | RunnableLambda(lambda x: {"query": x["query"], "search_result": _safe_tavily_search(x["search_query"])})
    | _tavily_summary_prompt
    | _schema_model
    | _str_parser
)

_tavily_tool = _tavily_chain.as_tool(
    name="tavily_search_tool",
    description="Web search time-based needs and return a concise summary.",
    arg_types={"query": str},
)


@tool
def tavily_search_tool(runtime: ToolRuntime[state]) -> Command:
    """Run a Tavily web search and store a concise summary into state."""
    s = runtime.state
    query = s.get("input_query", "")
    result_obj = _tavily_tool.invoke({"query": query})
    result = result_obj.model_dump() if hasattr(result_obj, "model_dump") else result_obj
    return Command(
        update={
            "tavily_search_result": result,
            "messages": [
                ToolMessage(
                    content="   [Context Agent] Storing Tavily search result...",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class ExtraRequirementsUpdate(BaseModel):
    items: List[str] = Field(default_factory=list)


def _merge_extra_requirements(schema_payload: Any, new_items: List[str]) -> Dict[str, Any]:
    base = schema_payload if isinstance(schema_payload, dict) else {}
    if not base:
        base = DesignSchema().model_dump()
    validated = DesignSchema.model_validate(base).model_dump()
    existing: List[str] = list(validated.get("extra_requirements") or [])
    existing_lc = {e.strip().lower() for e in existing if isinstance(e, str) and e.strip()}

    for item in new_items or []:
        text = " ".join(str(item or "").strip().split()[:12]).strip()
        if not text:
            continue
        key = text.lower()
        if key in existing_lc:
            continue
        existing.append(text)
        existing_lc.add(key)

    validated["extra_requirements"] = existing
    return validated


_update_model = ChatOpenAI(model_name="gpt-5", temperature=0.0)
_update_prompt = PromptTemplate.from_template(
    "Given user query: {query}\n"
    "Tavily search result (time-related context, may be empty): {tavily_search_result}\n"
    "Existing extra_requirements (JSON array): {existing_extra_requirements}\n\n"
    "Task: Extract up to 5 short time-related UI/UX items to remember.\n"
    "Rules: each item <= 12 words; summarize; no duplicates; if irrelevant return items=[].\n"
    "Return ONLY the structured object."
)

_update_chain = _update_prompt | _update_model.with_structured_output(ExtraRequirementsUpdate, method="function_calling")
_update_tool = _update_chain.as_tool(
    name="update_extra_requirements",
    description="Append time-related Tavily findings into design_schema.extra_requirements (only).",
    arg_types={"query": str, "tavily_search_result": str, "existing_extra_requirements": str},
)


@tool
def update_extra_requirements(runtime: ToolRuntime[state]) -> Command:
    """Update `design_schema.extra_requirements` based on Tavily time-sensitive findings."""
    s = runtime.state
    query = str(s.get("input_query", "") or "")
    tavily_search_result = str(s.get("tavily_search_result", "") or "")
    current_schema = s.get("design_schema") if isinstance(s, dict) else {}
    if not isinstance(current_schema, dict) or not current_schema:
        current_schema = DesignSchema().model_dump()

    validated = DesignSchema.model_validate(current_schema).model_dump()
    existing_extra_json = json.dumps(validated.get("extra_requirements") or [], ensure_ascii=False)

    result_obj = _update_tool.invoke(
        {
            "query": query,
            "tavily_search_result": tavily_search_result,
            "existing_extra_requirements": existing_extra_json,
        }
    )

    items: List[str] = []
    if hasattr(result_obj, "items"):
        items = list(getattr(result_obj, "items") or [])
    elif isinstance(result_obj, dict):
        items = list(result_obj.get("items") or [])

    merged = _merge_extra_requirements(validated, items)
    return Command(
        update={
            "design_schema": merged,
            "messages": [
                ToolMessage(
                    content="   [Context Agent] Updating extra requirements from Tavily search results...",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# -----------------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------------


@before_agent
def reset_before(state, runtime):
    input_query = ""
    if isinstance(state, dict) and isinstance(state.get("input_query"), str):
        input_query = state.get("input_query", "") or ""

    if not input_query and isinstance(state, dict):
        msgs = state.get("messages") or []
        try:
            for m in reversed(list(msgs)):
                if hasattr(m, "type") and getattr(m, "type") == "human" and hasattr(m, "content"):
                    input_query = str(getattr(m, "content") or "")
                    break
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    input_query = m.get("content") or ""
                    break
        except Exception:
            pass

    return {"input_query": input_query, "design_schema": {}, "tavily_search_result": ""}


@before_agent
def context_agent_progress(state, runtime):
    get_stream_writer()("   [Context Agent] Processing request...")
    return None


@wrap_tool_call
def context_tool_progress(request, handler):
    writer = get_stream_writer()
    tool_name = ""
    try:
        tool_name = (request.tool_call or {}).get("name") or ""
    except Exception:
        tool_name = ""

    if tool_name == "extract_intent_signals":
        writer("   [Context Agent] Extracting intent signals...")
    elif tool_name == "tavily_search_tool":
        writer("   [Context Agent] Retrieving time-specific context...")
    elif tool_name == "update_extra_requirements":
        writer("   [Context Agent] Updating design requirements...")
    return handler(request)


@before_agent
def orchestrator_progress(state, runtime):
    get_stream_writer()("[Orchestrator] Processing request...")
    return None


@wrap_tool_call
def orchestrator_tool_progress(request, handler):
    writer = get_stream_writer()
    tool_name = ""
    try:
        tool_name = (request.tool_call or {}).get("name") or ""
    except Exception:
        tool_name = ""

    if tool_name == "update_input":
        writer("[Orchestrator] Recording request...")
    elif tool_name == "call_context_engineering_agent":
        writer("[Orchestrator] Invoking Context Agent...")
    return handler(request)


# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------


_context_system_prompt = """
ROLE
- You are the Context Engineering (Schema) Agent.
- Your job is to extract ONLY renderable UI/UX requirements into `design_schema` for HTML/CSS/JS generation.

TOOL EXECUTION RULES
- You MUST call `extract_intent_signals` before any other action.
- You MUST NOT manually construct or modify `design_schema`.

ROUTING RULES (Tavily)
Call `tavily_search_tool` ONLY if the query explicitly contains a year (e.g., 2024) or words like "latest"/"current".
"""

_ce_model = ChatOpenAI(model="gpt-5.4", reasoning={"effort": "medium", "summary": "auto"})

context_engineering_agent = create_agent(
    model=_ce_model,
    system_prompt=_context_system_prompt,
    tools=[extract_intent_signals, tavily_search_tool, update_extra_requirements],
    middleware=[context_agent_progress, reset_before, context_tool_progress],
    state_schema=state,
)

_disable_checkpointer(context_engineering_agent)


@tool
def update_input(runtime: ToolRuntime[state]) -> Command:
    """Persist the current input query into agent state."""
    s = runtime.state or {}
    input_query = s.get("input_query", "") if isinstance(s, dict) else ""
    return Command(
        update={
            "input_query": input_query,
            "messages": [ToolMessage(content="[Orchestrator] Recorded request.", tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def call_context_engineering_agent(runtime: ToolRuntime[state], query: str) -> Command:
    """Invoke the context agent and persist its design_schema into the orchestrator state."""
    writer = get_stream_writer()
    writer({"type": "indent", "delta": 1})
    try:
        payload = {
            "messages": [{"role": "user", "content": query}],
            "input_query": query,
            "design_schema": {},
            "tavily_search_result": "",
        }

        thread_id = f"ctx_{uuid.uuid4().hex[:8]}"
        # Disable callbacks so LangSmith traces ONLY the orchestrator run.
        # (Otherwise the context sub-agent appears as a nested run.)
        sub_config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}, "callbacks": []}

        captured: Dict[str, Any] = {}

        def _merge(update: Any) -> None:
            if not isinstance(update, dict):
                return
            for k, v in update.items():
                if k == "messages":
                    continue
                captured[k] = v

        for chunk in context_engineering_agent.stream(
            payload,
            stream_mode=["updates", "custom"],
            version="v2",
            config=sub_config,
        ):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type, data = chunk
            else:
                chunk_type = chunk.get("type")
                data = chunk.get("data")

            if chunk_type == "custom":
                writer(data if isinstance(data, (str, dict)) else str(data))
            elif chunk_type == "updates" and isinstance(data, dict):
                for _step, up in data.items():
                    _merge(up)

        return Command(
            update={
                "design_schema": captured.get("design_schema") or {},
                "messages": [ToolMessage(content="[Orchestrator] Context Engineering Agent completed. Returned design_schema backed to orchestrator.", tool_call_id=runtime.tool_call_id)],
            }
        )
    finally:
        writer({"type": "indent", "delta": -1})


_orchestrator_prompt = """
ROLE
You are the Orchestrator Agent within a multi-agent system designed to generate an e-commerce web application based solely on the user query.

PRIMARY GOAL
Populate state.design_schema by invoking the context-engineering sub-agent tool. Do NOT design the schema yourself.

WORKFLOW
1) Persist `state.input_query` using `update_input`.
2) Call `call_context_engineering_agent`.
3) Reply with a short summary starting with "[Orchestrator] Summary: ".

BEHAVIOR RULES

1. Valid Request (E-commerce related)
- Follow the WORKFLOW strictly.
- Always invoke the context-engineering agent.

2. Unclear / Gibberish Input
- Do NOT invoke any tools.
- Ask the user for clarification.
- Example: "Could you clarify your request for the e-commerce application?"

3. Out-of-Scope Request (e.g., unrelated questions like general knowledge)
- Do NOT invoke any tools.
- Politely reject the request.
- Clarify the system purpose:
  "This system is designed to generate e-commerce web applications. Please provide a relevant request."

4. Capability Inquiry (e.g., "what can you do?")
- Do NOT invoke any tools.
- Respond normally by explaining system capabilities.

5. User Introduction / Greeting
- Do NOT invoke any tools.
- Respond naturally and acknowledge the user.
- Store relevant information if applicable.

CONSTRAINTS
- Never design or modify the schema yourself.
- Never invoke tools unless the request is valid and relevant.
- Keep responses concise and aligned with the role of an orchestrator.
"""

_orc_model = ChatOpenAI(model="gpt-5.4", reasoning={"effort": "medium", "summary": "auto"})

orchestrator_agent = create_agent(
    model=_orc_model,
    system_prompt=_orchestrator_prompt,
    tools=[update_input, call_context_engineering_agent],
    middleware=[orchestrator_progress, reset_before, orchestrator_tool_progress],
    state_schema=state,
)

_disable_checkpointer(orchestrator_agent)


# Common export used by UIs
graph = orchestrator_agent

# LangGraph Studio/API requires no custom checkpointer on the exported graph.
_disable_checkpointer(graph)


# -----------------------------------------------------------------------------
# Notebook-synced runtime overrides (tests intentionally omitted)
# -----------------------------------------------------------------------------

from typing import Callable, Set

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import model_validator


class state(AgentState):
    input_query: str = Field(default="", description="User request for this run.")
    compiled_query: str = Field(default="", description="Optional concise query for downstream processing.")
    design_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured planning schema, including meta fields.",
    )
    tavily_search_result: str = Field(default="")
    layout_specification: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured layout specification for UI generation.",
    )
    components_specification: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured components specification for UI generation.",
    )
    interactions_specification: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured interactions specification for UI generation.",
    )
    decoration_specification: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured decorations specification for UI generation.",
    )


model = ChatOpenAI(model_name="gpt-5.4", temperature=0.0)

prompt = PromptTemplate.from_template(
    """Given query: {query}

    Extract the user requirements into DesignSchema.

    Rules:
    - Return only valid JSON matching DesignSchema.
    - Do not include markdown, explanation, or extra keys.
    - Use null or [] when a field is not specified.
    - Do not invent brand names, colors, or features.
    - If a requirement does not cleanly map to a field, put it in extra_requirements as a short string (<= 12 words).
    - If the query is ambiguous, make the most conservative extraction possible.
    """
    "Few-shot example:\n"
    "Query: 'Build a modern dark-mode product listing page for a sneaker store with filters, sorting, search, cart, and wishlist. Use a top nav and a compact grid. Also: add a sticky promo banner.'\n"
    "Example output (JSON-like):\n"
    "{{\n"
    "  'ui': {{\n"
    "    'type': 'product_listing',\n"
    "    'variant': 'grid',\n"
    "    'style': 'modern',\n"
    "    'density': 'compact',\n"
    "    'navigation': 'top_nav',\n"
    "    'feedback_components': []\n"
    "  }},\n"
    "  'theme': {{\n"
    "    'primary_color': null,\n"
    "    'secondary_color': null,\n"
    "    'mode': 'dark'\n"
    "  }},\n"
    "  'features': {{\n"
    "    'search': true,\n"
    "    'filter': true,\n"
    "    'sorting': true,\n"
    "    'cart': true,\n"
    "    'wishlist': true,\n"
    "    'reviews': null\n"
    "  }},\n"
    "  'constraints': {{\n"
    "    'tech_stack': ['html','css','javascript']\n"
    "  }},\n"
    "  'extra_requirements': ['Sticky promo banner on scroll']\n"
    "}}\n\n"
    "Now extract from the real query."
)

extract_intent_signals_chain = prompt | model.with_structured_output(DesignSchema, method="function_calling")

extract_intent_signals_tool = extract_intent_signals_chain.as_tool(
    name="extract_intent_signals",
    description="Extract renderable UI/UX intent signals from a query and persist a DesignSchema payload",
    arg_types={"query": str},
)


@tool
def extract_intent_signals(runtime: ToolRuntime[state]) -> Command:
    """Run intent signal extraction tool and persist the result into state.design_schema."""
    s = runtime.state
    query = s["input_query"]

    result_obj = extract_intent_signals_tool.invoke({"query": query})
    result = result_obj.model_dump() if hasattr(result_obj, "model_dump") else result_obj

    return Command(
        update={
            "design_schema": result,
            "messages": [
                ToolMessage(content="Stored design_schema from extract_intent_signals", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

tavily_search_query_prompt = PromptTemplate.from_template(
    "Given user query: {query}\n"
    "Rewrite it into a short web search query (<= 12 words).\n"
    "Return only the rewritten search query."
)

tavily_search_query_chain = tavily_search_query_prompt | model | _str_parser


def safe_tavily_search(search_query: str) -> str:
    if not search_query.strip():
        return ""
    try:
        res = tavily_client.search(search_query, max_results=5)
        return json.dumps(res, ensure_ascii=False)
    except Exception:
        return ""


prompt = PromptTemplate.from_template(
    "Given user query: {query}\n"
    "Raw Tavily results (JSON): {search_result}\n\n"
    "Task: Summarize ONLY time-sensitive UI/UX relevant context into <= 6 bullet points.\n"
    "Rules:\n"
    "- Output plain text only (no markdown fencing).\n"
    "- If results are empty/irrelevant, return empty string.\n"
)

tavily_chain = (
    {
        "query": lambda x: x["query"],
        "search_query": tavily_search_query_chain,
    }
    | RunnableLambda(
        lambda x: {
            "query": x["query"],
            "search_result": safe_tavily_search(x["search_query"]),
        }
    )
    | prompt
    | model
    | _str_parser
)

tavily_tool = tavily_chain.as_tool(
    name="tavily_search_tool",
    description="Web search the time-based needs (e.g., 'latest design trends') and return a concise summary.",
    arg_types={"query": str},
)


@tool
def tavily_search_tool(runtime: ToolRuntime[state]) -> Command:
    """Run Tavily search tool and persist the result into agent state."""
    s = runtime.state
    query = s["input_query"]

    result_obj = tavily_tool.invoke({"query": query})
    result = result_obj.model_dump() if hasattr(result_obj, "model_dump") else result_obj

    return Command(
        update={
            "tavily_search_result": result,
            "messages": [ToolMessage(content="Stored tavily_search_result", tool_call_id=runtime.tool_call_id)],
        }
    )


class ExtraRequirementsUpdate(BaseModel):
    items: List[str] = Field(
        default_factory=list,
        description="New time-related UI/UX items to append into DesignSchema.extra_requirements (<= 12 words each).",
    )


def _normalize_schema_payload(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _clean_item(text: Any, *, max_words: int = 12) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    words = [word for word in s.split() if word]
    if len(words) > max_words:
        s = " ".join(words[:max_words]).strip()
    return s


def _merge_extra_requirements(schema_payload: Any, new_items: List[str]) -> Dict[str, Any]:
    base = _normalize_schema_payload(schema_payload)
    if not base:
        base = DesignSchema().model_dump()
    validated = DesignSchema.model_validate(base).model_dump()
    existing: List[str] = list(validated.get("extra_requirements") or [])
    existing_lc = {e.strip().lower() for e in existing if isinstance(e, str) and e.strip()}

    for item in new_items or []:
        cleaned = _clean_item(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in existing_lc:
            continue
        existing.append(cleaned)
        existing_lc.add(key)

    validated["extra_requirements"] = existing
    return validated


model = ChatOpenAI(model_name="gpt-5", temperature=0.0)

update_extra_requirements_prompt = PromptTemplate.from_template(
    "Given user query: {query}\n"
    "Tavily search result (time-related context, may be empty): {tavily_search_result}\n"
    "Existing extra_requirements (JSON array): {existing_extra_requirements}\n\n"
    "Task: Extract up to 5 short time-related UI/UX requirements or trends from Tavily context that are useful to remember.\n"
    "Rules (strict):\n"
    "- Output must match ExtraRequirementsUpdate(items=[...]).\n"
    "- Each item must be <= 12 words.\n"
    "- Do NOT copy Tavily text verbatim; summarize instead.\n"
    "- Avoid duplicates with existing_extra_requirements.\n"
    "- If Tavily context is empty/irrelevant, return items=[].\n"
    "Return ONLY the structured object."
)

update_extra_requirements_chain = (
    update_extra_requirements_prompt
    | model.with_structured_output(ExtraRequirementsUpdate, method="function_calling")
)

update_extra_requirements_tool = update_extra_requirements_chain.as_tool(
    name="update_extra_requirements",
    description="Append time-related Tavily findings into design_schema.extra_requirements (only).",
    arg_types={
        "query": str,
        "tavily_search_result": str,
        "existing_extra_requirements": str,
    },
)


@tool
def update_extra_requirements(runtime: ToolRuntime[state]) -> Command:
    """Update ONLY design_schema.extra_requirements using the Tavily search result."""
    s = runtime.state
    query = str(s.get("input_query", "") or "")
    tavily_search_result = str(s.get("tavily_search_result", "") or "")
    current_schema = _normalize_schema_payload(s.get("design_schema"))
    if not current_schema:
        current_schema = DesignSchema().model_dump()

    validated = DesignSchema.model_validate(current_schema).model_dump()
    existing_extra = validated.get("extra_requirements") or []
    existing_extra_json = json.dumps(existing_extra, ensure_ascii=False)

    result_obj = update_extra_requirements_tool.invoke(
        {
            "query": query,
            "tavily_search_result": tavily_search_result,
            "existing_extra_requirements": existing_extra_json,
        }
    )

    items: List[str] = []
    if hasattr(result_obj, "items"):
        items = list(getattr(result_obj, "items") or [])
    elif isinstance(result_obj, dict):
        items = list(result_obj.get("items") or [])

    merged_schema = _merge_extra_requirements(validated, items)
    return Command(
        update={
            "design_schema": merged_schema,
            "messages": [
                ToolMessage(
                    content="Updated design_schema.extra_requirements from Tavily search result",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


AgentType = Literal["HTML", "CSS", "JavaScript"]

PageType = Literal["plp", "pdp", "cart", "checkout", "account", "search"]

LayoutType = Literal[
    "grid",
    "flex-row",
    "flex-column",
    "single-column",
    "multi-column",
    "absolute",
    "stack",
]

ResponsiveBreakpoint = Literal["mobile", "tablet", "desktop", "wide"]

HTMLTag = Literal[
    "div", "span", "section", "header", "footer", "nav", "main",
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "img", "video", "picture",
    "a",
    "ul", "ol", "li",
    "form", "input", "button", "label", "select", "option", "textarea",
    "article", "aside", "figure", "figcaption",
    "table", "thead", "tbody", "tr", "td", "th",
]

ActionType = Literal[
    "updateState", "navigate",
    "toggleVisibility", "updateData", "showFeedback", "emit",
    "addToCart", "updateQuantity",
    "applyFilter", "clearFilter", "toggleWishlist",
]

ComponentCategory = Literal[
    # Navigation & structure
    "navbar", "footer", "breadcrumb",
    # Product display
    "product-card", "product-grid", "product-image-gallery",
    "product-price", "product-rating",
    # Filtering & sorting
    "filter-panel", "sort-bar",
    # Cart & checkout
    "mini-cart", "cart-item", "add-to-cart-button",
    "order-summary", "checkout-form", "payment-form",
    # Product interaction
    "size-picker", "quantity-selector", "wishlist-button",
    # Search
    "search-bar", "search-results",
    # Pagination & promotions
    "pagination", "promo-banner",
    # Account
    "account-profile", "order-history",
    # General
    "hero-section",
]

RECOMMENDED_STATES = {
    "idle", "hover", "active", "focus", "focus-within",
    "disabled", "loading", "success", "error",
    "selected", "expanded", "collapsed", "empty",
    "checked", "unchecked", "indeterminate",
    "open", "closed",
}


class Region(BaseModel):
    name: str = Field(
        description="Region name. Unique within a page. Used as the dict key in ComponentSpec.regionComponents."
    )
    ownedBy: AgentType = Field(
        description="Default agent ownership for this region. Components inside MAY override this at the ComponentNode level."
    )
    layoutType: LayoutType = Field(
        description="How child components are arranged within this region (grid, flex-row, etc.)."
    )


class Page(BaseModel):
    pageName: str = Field(description="Unique page identifier. Primary linking key across ALL specs.")
    pageType: PageType = Field(
        description="B2C page type. Drives which components and interactions are expected."
    )
    pageLayoutType: LayoutType = Field(
        description="Top-level layout strategy for arranging regions on the page."
    )
    responsive: bool = Field(default=True)
    regions: List[Region] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_region_names(self):
        names = [r.name for r in self.regions]
        if len(names) != len(set(names)):
            raise ValueError(f"Page '{self.pageName}': region names must be unique.")
        return self


class LayoutSpecification(BaseModel):
    pages: List[Page] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_page_names(self):
        names = [p.pageName for p in self.pages]
        if len(names) != len(set(names)):
            raise ValueError("LayoutSpecification: pageName must be unique.")
        return self


class ComponentState(BaseModel):
    name: str = Field(
        description="State identifier (e.g., 'hover', 'loading'). Should typically come from RECOMMENDED_STATES."
    )
    description: Optional[str] = None


class ComponentEvent(BaseModel):
    event: str = Field(description="DOM-style event name (e.g., 'onClick', 'onSubmit').")
    flowName: Optional[str] = Field(
        default=None,
        description="Name of the InteractionFlow that implements this event. MUST match an InteractionFlow.name.",
    )


class DataBindingSpec(BaseModel):
    source: str = Field(
        description="Data path this component reads from (e.g. 'product.price', 'cart.items', 'filters.category')."
    )
    format: Optional[str] = Field(
        default=None,
        description="Display format hint (e.g. 'currency', 'list', 'count', 'date', 'percentage').",
    )


class ComponentNode(BaseModel):
    id: str = Field(description="Unique component identifier. Stable across all specs.")
    tag: HTMLTag
    ownedBy: AgentType = Field(description="Execution agent. May override the parent Region.ownedBy.")
    componentCategory: Optional[ComponentCategory] = Field(
        default=None,
        description=(
            "B2C component category — written as the HTML data-component attribute value. "
            "Pick the closest value from ComponentCategory. Omit for purely structural wrappers "
            "that are not individually addressable (use <div> for those instead)."
        ),
    )
    props: Dict[str, Any] = Field(default_factory=dict)
    dataBinding: Optional[DataBindingSpec] = None
    states: List[ComponentState] = Field(default_factory=list)
    events: List[ComponentEvent] = Field(default_factory=list)
    className: Optional[str] = Field(
        default=None,
        description="CSS class name. Primary linking key to DecorationSpec.ComponentDecoration.",
    )
    children: List["ComponentNode"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self):
        if self.ownedBy == "CSS" and not self.className:
            raise ValueError(f"{self.id}: CSS-owned component must define className.")
        if len({s.name for s in self.states}) != len(self.states):
            raise ValueError(f"{self.id}: duplicate state names.")
        if len({e.event for e in self.events}) != len(self.events):
            raise ValueError(f"{self.id}: duplicate event names.")
        return self


ComponentNode.model_rebuild()


class PageComponentTree(BaseModel):
    pageName: str
    regionComponents: Dict[str, List[ComponentNode]] = Field(
        description="Top-level components keyed by region name. Each key MUST match a Region.name in LayoutSpec."
    )


class ComponentSpecification(BaseModel):
    pages: List[PageComponentTree] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_page_names(self):
        names = [p.pageName for p in self.pages]
        if len(names) != len(set(names)):
            raise ValueError("ComponentSpecification: pageName must be unique.")
        return self


class PageComponentTreeDraft(BaseModel):
    pageName: str
    regionComponents: Dict[str, List[ComponentNode]] = Field(
        default_factory=dict,
        description="Draft top-level components keyed by region name. The final validator still requires full region coverage.",
    )


class ComponentSpecificationDraft(BaseModel):
    pages: List[PageComponentTreeDraft] = Field(default_factory=list)


class InteractionEvent(BaseModel):
    event: str
    targetId: str = Field(description="Component where the event originates. MUST match a ComponentNode.id.")


class ActionResultState(BaseModel):
    description: str = Field(
        description="Human-readable outcome of this action (e.g. 'Cart item count increments; button shows Added')."
    )


class InteractionAction(BaseModel):
    type: ActionType
    targetId: Optional[str] = Field(default=None, description="If specified, MUST match a ComponentNode.id.")
    payload: Optional[Dict[str, Any]] = None
    result_state: Optional[ActionResultState] = None


class InteractionStateTransition(BaseModel):
    componentId: str = Field(description="MUST match a ComponentNode.id.")
    fromState: Optional[str] = Field(default=None)
    toState: str = Field(description="MUST match a name in the target component's states[].")


class InteractionFlow(BaseModel):
    name: str = Field(description="Unique flow identifier. Referenced by ComponentEvent.flowName.")
    description: Optional[str] = None
    trigger: InteractionEvent
    actions: List[InteractionAction]
    stateTransitions: List[InteractionStateTransition] = Field(default_factory=list)


class PageInteraction(BaseModel):
    pageName: str
    flows: List[InteractionFlow]

    @model_validator(mode="after")
    def _unique_flow_names(self):
        names = [f.name for f in self.flows]
        if len(names) != len(set(names)):
            raise ValueError(f"Page '{self.pageName}': flow names must be unique.")
        return self


class InteractionSpecification(BaseModel):
    pages: List[PageInteraction] = Field(default_factory=list)


class DesignTokens(BaseModel):
    colors: Dict[str, str] = Field(default_factory=dict)
    typography: Dict[str, Any] = Field(default_factory=dict)
    spacing: Dict[str, str] = Field(default_factory=dict)
    borders: Dict[str, str] = Field(default_factory=dict)
    shadows: Dict[str, str] = Field(default_factory=dict)
    radii: Dict[str, str] = Field(default_factory=dict)


class TransitionSpec(BaseModel):
    property: str = Field(default="all")
    duration: str = Field(default="200ms")
    easing: str = Field(default="ease-in-out")
    delay: Optional[str] = None


class StateVariant(BaseModel):
    stateName: str = Field(description="MUST match a ComponentState.name on the linked ComponentNode.")
    styles: Dict[str, Any]


class ResponsiveVariant(BaseModel):
    breakpoint: ResponsiveBreakpoint
    minWidth: Optional[str] = None
    maxWidth: Optional[str] = None
    styles: Dict[str, Any]


class ComponentDecoration(BaseModel):
    componentId: str = Field(description="MUST match a ComponentNode.id.")
    className: str = Field(description="MUST match the target ComponentNode.className.")
    baseStyles: Dict[str, Any] = Field(default_factory=dict)
    stateVariants: List[StateVariant] = Field(default_factory=list)
    responsiveStyles: List[ResponsiveVariant] = Field(default_factory=list)
    transitions: List[TransitionSpec] = Field(default_factory=list)
    pseudoElements: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_dupes(self):
        names = [v.stateName for v in self.stateVariants]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.componentId}: duplicate stateVariant names.")
        breakpoints = [v.breakpoint for v in self.responsiveStyles]
        if len(breakpoints) != len(set(breakpoints)):
            raise ValueError(f"{self.componentId}: duplicate breakpoint in responsiveStyles.")
        return self


class PageDecoration(BaseModel):
    pageName: str
    pageTokens: Optional[DesignTokens] = Field(default=None)
    componentDecorations: List[ComponentDecoration] = Field(default_factory=list)


class DecorationSpecification(BaseModel):
    globalTokens: DesignTokens = Field(default_factory=DesignTokens)
    pages: List[PageDecoration] = Field(default_factory=list)


class UISpecificationBundle(BaseModel):
    layout: LayoutSpecification
    component: ComponentSpecification
    interaction: InteractionSpecification
    decoration: DecorationSpecification

    @model_validator(mode="after")
    def _cross_spec_consistency(self):
        def walk_ids(node: ComponentNode) -> Set[str]:
            ids = {node.id}
            for child in node.children:
                ids |= walk_ids(child)
            return ids

        def walk_state_index(node: ComponentNode, idx: Dict[str, Set[str]]):
            idx[node.id] = {s.name for s in node.states}
            for child in node.children:
                walk_state_index(child, idx)

        def walk_class_index(node: ComponentNode, idx: Dict[str, Optional[str]]):
            idx[node.id] = node.className
            for child in node.children:
                walk_class_index(child, idx)

        def walk_event_flows(node: ComponentNode, declared: Set[str]):
            for event in node.events:
                if event.flowName:
                    declared.add(event.flowName)
            for child in node.children:
                walk_event_flows(child, declared)

        layout_pages = {p.pageName for p in self.layout.pages}
        comp_pages = {p.pageName for p in self.component.pages}
        intr_pages = {p.pageName for p in self.interaction.pages}
        deco_pages = {p.pageName for p in self.decoration.pages}

        if not comp_pages.issubset(layout_pages):
            raise ValueError(f"ComponentSpec references unknown pages: {comp_pages - layout_pages}")
        if not intr_pages.issubset(layout_pages):
            raise ValueError(f"InteractionSpec references unknown pages: {intr_pages - layout_pages}")
        if not deco_pages.issubset(layout_pages):
            raise ValueError(f"DecorationSpec references unknown pages: {deco_pages - layout_pages}")

        for layout_page in self.layout.pages:
            page_name = layout_page.pageName
            comp_page = next((p for p in self.component.pages if p.pageName == page_name), None)
            intr_page = next((p for p in self.interaction.pages if p.pageName == page_name), None)
            deco_page = next((p for p in self.decoration.pages if p.pageName == page_name), None)

            if comp_page is None:
                continue

            layout_region_names = {r.name for r in layout_page.regions}
            comp_region_names = set(comp_page.regionComponents.keys())
            if not comp_region_names.issubset(layout_region_names):
                raise ValueError(
                    f"Page '{page_name}': ComponentSpec.regionComponents has unknown regions: {comp_region_names - layout_region_names}"
                )

            all_ids: Set[str] = set()
            state_index: Dict[str, Set[str]] = {}
            class_index: Dict[str, Optional[str]] = {}
            declared_flows: Set[str] = set()

            for nodes in comp_page.regionComponents.values():
                for node in nodes:
                    all_ids |= walk_ids(node)
                    walk_state_index(node, state_index)
                    walk_class_index(node, class_index)
                    walk_event_flows(node, declared_flows)

            if intr_page:
                actual_flow_names = {f.name for f in intr_page.flows}
                missing_flows = declared_flows - actual_flow_names
                if missing_flows:
                    raise ValueError(
                        f"Page '{page_name}': ComponentEvent.flowName refers to unknown flows: {missing_flows}"
                    )

                for flow in intr_page.flows:
                    if flow.trigger.targetId not in all_ids:
                        raise ValueError(
                            f"Page '{page_name}', flow '{flow.name}': trigger.targetId '{flow.trigger.targetId}' is not a known component."
                        )
                    for action in flow.actions:
                        if action.targetId and action.targetId not in all_ids:
                            raise ValueError(
                                f"Page '{page_name}', flow '{flow.name}': action.targetId '{action.targetId}' unknown."
                            )
                    for transition in flow.stateTransitions:
                        if transition.componentId not in all_ids:
                            raise ValueError(
                                f"Page '{page_name}', flow '{flow.name}': transition.componentId '{transition.componentId}' unknown."
                            )
                        comp_states = state_index.get(transition.componentId, set())
                        if transition.toState not in comp_states:
                            raise ValueError(
                                f"Page '{page_name}', flow '{flow.name}': toState '{transition.toState}' not declared on '{transition.componentId}' (declared: {comp_states})."
                            )
                        if transition.fromState is not None and transition.fromState not in comp_states:
                            raise ValueError(
                                f"Page '{page_name}', flow '{flow.name}': fromState '{transition.fromState}' not declared on '{transition.componentId}'."
                            )

            if deco_page:
                for deco in deco_page.componentDecorations:
                    if deco.componentId not in all_ids:
                        raise ValueError(
                            f"Page '{page_name}': Decoration componentId '{deco.componentId}' is not a known component."
                        )
                    target_class = class_index.get(deco.componentId)
                    if target_class != deco.className:
                        raise ValueError(
                            f"Page '{page_name}': Decoration className '{deco.className}' does not match component '{deco.componentId}'.className='{target_class}'."
                        )
                    comp_states = state_index.get(deco.componentId, set())
                    for variant in deco.stateVariants:
                        if variant.stateName not in comp_states:
                            raise ValueError(
                                f"Page '{page_name}': Decoration for '{deco.componentId}' has stateVariant '{variant.stateName}' not in declared states {comp_states}."
                            )

        return self


model = ChatOpenAI(model_name="gpt-5.4", temperature=0.0)


def _coerce_layout_spec(payload: Any) -> LayoutSpecification:
    if isinstance(payload, LayoutSpecification):
        return payload
    if payload is None:
        payload = {}
    return LayoutSpecification.model_validate(payload)


def _coerce_component_spec(payload: Any) -> ComponentSpecification:
    if isinstance(payload, ComponentSpecification):
        return payload
    if payload is None:
        payload = {}
    return ComponentSpecification.model_validate(payload)


def _coerce_interaction_spec(payload: Any) -> InteractionSpecification:
    if isinstance(payload, InteractionSpecification):
        return payload
    if payload is None:
        payload = {}
    if isinstance(payload, dict) and "pages" not in payload:
        payload = {"pages": []}
    return InteractionSpecification.model_validate(payload)


def _coerce_decoration_spec(payload: Any) -> DecorationSpecification:
    if isinstance(payload, DecorationSpecification):
        return payload
    if payload is None:
        payload = {}
    return DecorationSpecification.model_validate(payload)


def _walk_component_nodes(node: ComponentNode):
    yield node
    for child in node.children:
        yield from _walk_component_nodes(child)


def _validate_component_contract_for_downstream(
    layout: LayoutSpecification,
    component: ComponentSpecification,
) -> Optional[str]:
    layout_pages = {p.pageName: p for p in layout.pages}
    violations: List[str] = []

    for comp_page in component.pages:
        layout_page = layout_pages.get(comp_page.pageName)
        if layout_page is None:
            continue

        layout_region_names = {region.name for region in layout_page.regions}
        component_region_names = set(comp_page.regionComponents.keys())

        missing_regions = layout_region_names - component_region_names
        if missing_regions:
            violations.append(
                f"Page '{comp_page.pageName}' is missing regionComponents entries for: "
                + ", ".join(sorted(missing_regions))
            )

        extra_regions = component_region_names - layout_region_names
        if extra_regions:
            violations.append(
                f"Page '{comp_page.pageName}' has unknown regionComponents keys: "
                + ", ".join(sorted(extra_regions))
            )

        for region_name, nodes in comp_page.regionComponents.items():
            for root in nodes:
                for node in _walk_component_nodes(root):
                    if not str(node.ownedBy).strip():
                        violations.append(
                            f"Page '{comp_page.pageName}', region '{region_name}', component '{node.id}' is missing ownedBy."
                        )

                    for event in node.events:
                        if not str(event.event).strip():
                            violations.append(
                                f"Page '{comp_page.pageName}', region '{region_name}', component '{node.id}' has an event with an empty event name."
                            )
                        if not (event.flowName and str(event.flowName).strip()):
                            event_name = event.event if str(event.event).strip() else "<empty-event-name>"
                            violations.append(
                                f"Page '{comp_page.pageName}', region '{region_name}', component '{node.id}' has event '{event_name}' without a non-empty flowName."
                            )

                    if node.ownedBy == "CSS":
                        if not (node.className and str(node.className).strip()):
                            violations.append(
                                f"Page '{comp_page.pageName}', region '{region_name}', component '{node.id}' must define className because it is CSS-owned."
                            )

    if not violations:
        return None

    max_violations = 25
    rendered = [f"- {issue}" for issue in violations[:max_violations]]
    if len(violations) > max_violations:
        rendered.append(f"- ... and {len(violations) - max_violations} more issue(s).")
    return "[components] ComponentSpecification is incomplete for downstream:\n" + "\n".join(rendered)


def _build_component_scaffold_from_layout(layout: Any) -> Dict[str, Any]:
    layout_obj = _coerce_layout_spec(layout)
    return {
        "pages": [
            {
                "pageName": page.pageName,
                "regionComponents": {region.name: [] for region in page.regions},
            }
            for page in layout_obj.pages
        ]
    }


def _build_component_contract_from_layout(layout: Any) -> str:
    layout_obj = _coerce_layout_spec(layout)
    lines: List[str] = []

    for page in layout_obj.pages:
        lines.append(f"Page {page.pageName} ({page.pageType}):")
        for region in page.regions:
            lines.append(f"- {region.name} [{region.layoutType}] owned by {region.ownedBy}")

    return "\n".join(lines)


def _guess_component_tag(component_id: str, region_name: str) -> HTMLTag:
    token = f"{component_id} {region_name}".lower()
    if any(word in token for word in {"bottomnav", "bottom_nav", "nav", "navbar"}):
        return "nav"
    if any(word in token for word in {"header", "topbar", "top_bar"}):
        return "header"
    if any(word in token for word in {"search", "filter", "form", "checkout", "payment"}):
        return "form"
    if any(word in token for word in {"button", "cta", "action", "add_to_cart", "place_order"}):
        return "button"
    if any(word in token for word in {"list", "grid", "gallery", "carousel", "items", "products"}):
        return "section"
    return "section"


def _default_component_node_for_ref(component_id: str, region_name: str) -> Dict[str, Any]:
    return {
        "id": component_id,
        "tag": _guess_component_tag(component_id, region_name),
        "ownedBy": "HTML",
        "props": {},
        "children": [],
    }


def _repair_component_spec_from_layout(layout: Any, candidate: Any) -> Dict[str, Any]:
    layout_obj = _coerce_layout_spec(layout)

    if hasattr(candidate, "model_dump"):
        payload = candidate.model_dump()
    elif isinstance(candidate, dict):
        payload = dict(candidate)
    else:
        payload = {}

    raw_pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
    pages_by_name: Dict[str, Dict[str, Any]] = {}
    for page in raw_pages:
        if not isinstance(page, dict):
            continue
        page_name = str(page.get("pageName") or "").strip()
        if not page_name:
            continue
        region_components = page.get("regionComponents")
        if not isinstance(region_components, dict):
            region_components = {}
        pages_by_name[page_name] = {
            "pageName": page_name,
            "regionComponents": {key: list(value or []) for key, value in region_components.items() if isinstance(key, str)},
        }

    repaired_pages: List[Dict[str, Any]] = []
    for layout_page in layout_obj.pages:
        page_payload = pages_by_name.get(
            layout_page.pageName,
            {"pageName": layout_page.pageName, "regionComponents": {}},
        )
        repaired_regions: Dict[str, List[Any]] = {}

        for region in layout_page.regions:
            existing_nodes = list(page_payload["regionComponents"].get(region.name) or [])
            repaired_regions[region.name] = existing_nodes

        repaired_pages.append(
            {
                "pageName": layout_page.pageName,
                "regionComponents": repaired_regions,
            }
        )

    return {"pages": repaired_pages}


def _bundle_validate_or_warn(
    *,
    stage: str,
    layout: Any,
    component: Optional[Any] = None,
    interaction: Optional[Any] = None,
    decoration: Optional[Any] = None,
) -> Optional[str]:
    try:
        layout_obj = _coerce_layout_spec(layout)
        component_obj = _coerce_component_spec(component or {"pages": []})
        interaction_obj = _coerce_interaction_spec(interaction or {"pages": []})
        decoration_obj = _coerce_decoration_spec(decoration or {})

        layout_page_names = {p.pageName for p in layout_obj.pages}
        component_page_names = {p.pageName for p in component_obj.pages}
        missing_component_pages = layout_page_names - component_page_names
        if missing_component_pages:
            raise ValueError(
                "ComponentSpecification missing pages for layout: " + ", ".join(sorted(missing_component_pages))
            )

        component_contract_warning = _validate_component_contract_for_downstream(layout_obj, component_obj)
        if component_contract_warning:
            raise ValueError(component_contract_warning)

        UISpecificationBundle(
            layout=layout_obj,
            component=component_obj,
            interaction=interaction_obj,
            decoration=decoration_obj,
        )
        return None
    except Exception as e:
        return f"[{stage}] UISpecificationBundle validation warning: {e}"


def _invoke_structured_with_retries(
    *,
    stage: str,
    structured_tool: Any,
    base_payload: Dict[str, Any],
    validate_result: Callable[[Any], Optional[str]],
    max_attempts: int = 3,
) -> tuple[Optional[Any], Optional[str]]:
    validation_feedback = ""
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        payload = dict(base_payload)
        payload["validation_feedback"] = validation_feedback

        try:
            candidate = structured_tool.invoke(payload)
            if hasattr(candidate, "model_dump"):
                candidate = candidate.model_dump()
        except Exception as e:
            last_error = (
                f"[{stage}] structured output parse/invoke failed "
                f"(attempt {attempt}/{max_attempts}): {type(e).__name__}: {e}"
            )
            validation_feedback = last_error
            continue

        warn = validate_result(candidate)
        if warn is None:
            return candidate, None

        last_error = warn
        validation_feedback = warn

    return None, last_error or f"[{stage}] structured tool failed without details"


layout_prompt = PromptTemplate.from_template(
    """You are a UI LayoutSpecification generator.

INPUTS
- input_query: {input_query}
- design_schema (JSON): {design_schema}
- validation_feedback (may be empty): {validation_feedback}

TASK
Generate a LayoutSpecification that defines pages and regions, and places top-level components via componentRefs.

RETRY / FIXUP
- If validation_feedback is non-empty, it describes why the previous attempt failed validation.
- Correct the issues it mentions and regenerate a fully-valid LayoutSpecification.
- Keep identifiers stable where possible (pageName, region names, componentRef ids).

SCHEMA NOTES (STRICT)
- Output must match LayoutSpecification exactly.
- For each Page: pageName, pageType (one of: plp, pdp, cart, checkout, account, search), pageLayoutType, responsive, regions[].
- For each Region: name, ownedBy, layoutType. Do NOT include componentRefs — ComponentSpec owns component placement.
- pageName MUST be unique and stable.
- Region.name must be unique within a page.
- pages must be non-empty; each page must have at least one region.

GROUNDING
- Use design_schema only as constraints (features on/off, style hints).
- Do not invent features explicitly disabled in design_schema.

OUTPUT RULES
- Return ONLY a LayoutSpecification object (no markdown, no explanation).
"""
)

layout_specification_generator_chain = (
    layout_prompt | model.with_structured_output(LayoutSpecification, method="function_calling")
)

layout_specification_generator_tool = layout_specification_generator_chain.as_tool(
    name="layout_specification_generator",
    description="Generate LayoutSpecification from input_query + design_schema.",
    arg_types={"input_query": str, "design_schema": str, "validation_feedback": str},
)


@tool
def layout_specification_generator(runtime: ToolRuntime[state]) -> Command:
    """Generate and store layout_specification."""
    s = runtime.state
    input_query = str(s.get("input_query", "") or "")
    design_schema_payload: Any = s.get("design_schema") or {}

    def _validate(candidate: Any) -> Optional[str]:
        try:
            obj = _coerce_layout_spec(candidate)
        except Exception as e:
            return f"[layout] LayoutSpecification validation warning: {e}"
        if not obj.pages:
            return "[layout] LayoutSpecification.pages must be non-empty"
        for page in obj.pages:
            if not page.regions:
                return f"[layout] Page '{page.pageName}' must contain at least 1 region"
        return None

    base_payload = {
        "input_query": input_query,
        "design_schema": json.dumps(design_schema_payload, ensure_ascii=False),
    }
    candidate, error = _invoke_structured_with_retries(
        stage="layout",
        structured_tool=layout_specification_generator_tool,
        base_payload=base_payload,
        validate_result=_validate,
        max_attempts=3,
    )

    if error or candidate is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"FAILED to generate layout_specification.\n{error}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    result_obj = _coerce_layout_spec(candidate)
    result = result_obj.model_dump()
    return Command(
        update={
            "layout_specification": result,
            "messages": [ToolMessage(content="Stored layout_specification.", tool_call_id=runtime.tool_call_id)],
        }
    )


components_prompt = PromptTemplate.from_template(
    """You are a UI ComponentSpecification generator.

INPUTS
- input_query: {input_query}
- layout_specification (JSON): {layout_specification}
- required_component_contract:\n{required_component_contract}
- component_scaffold (JSON): {component_scaffold}
- validation_feedback (may be empty): {validation_feedback}

TASK
Generate a complete ComponentSpecification that provides full component trees per page, organized by regionComponents.
This output is consumed immediately by the interactions, decoration, and reorganizer stages, so all downstream-required fields must be present in the first valid result.
Start from component_scaffold exactly: keep every pageName and every regionComponents key unchanged, and replace each empty list with the required top-level ComponentNode trees.

RETRY / FIXUP
- If validation_feedback is non-empty, it describes why the previous attempt failed validation.
- Correct every issue it mentions in the same regenerated output.
- Keep identifiers stable where possible (component ids, flowName).

HARD CONSTRAINTS
- The set of pageName values MUST equal layout_specification.pages[].pageName (cover every layout page).
- Never emit a page object with only pageName. Every page object MUST include regionComponents as an object.
- For every page, regionComponents keys MUST exactly equal the Region.name values in the corresponding layout page. Include an empty list for any region that has no top-level components.
- ComponentNode.id must be globally unique and stable across all specs.
- Every ComponentNode MUST include ownedBy: exactly one of HTML, CSS, JavaScript. Never emit null, an empty string, or any other value.
- Every emitted ComponentEvent MUST include a non-empty event and a non-empty flowName. Do not leave either field blank.
- If ownedBy == 'CSS', className must be a non-empty CSS-safe token. If a node does not need CSS ownership, set ownedBy to HTML instead of leaving className blank.
- Do not emit empty strings or whitespace-only strings for ownedBy, event, flowName, or className.
- If a field is optional and truly not needed, omit it instead of emitting an empty string placeholder.
- componentCategory: MUST be one of the values in ComponentCategory (navbar, footer, breadcrumb, product-card, product-grid, product-image-gallery, product-price, product-rating, filter-panel, sort-bar, mini-cart, cart-item, add-to-cart-button, order-summary, checkout-form, payment-form, size-picker, quantity-selector, wishlist-button, search-bar, search-results, pagination, promo-banner, account-profile, order-history, hero-section). This value is written verbatim as the HTML data-component attribute. Omit for purely structural wrappers that are not individually addressable.
- dataBinding: if a component reads dynamic data, populate dataBinding.source (e.g. 'product.price') and optionally dataBinding.format (e.g. 'currency', 'list', 'count'). Do not use a plain string.

OUTPUT RULES
- Return ONLY a ComponentSpecification object (no markdown, no explanation).
"""
)

components_specification_generator_chain = (
    components_prompt | model.with_structured_output(ComponentSpecificationDraft, method="function_calling")
)

components_specification_generator_tool = components_specification_generator_chain.as_tool(
    name="components_specification_generator",
    description="Generate ComponentSpecification from input_query + layout_specification.",
    arg_types={
        "input_query": str,
        "layout_specification": str,
        "required_component_contract": str,
        "component_scaffold": str,
        "validation_feedback": str,
    },
)


@tool
def components_specification_generator(runtime: ToolRuntime[state]) -> Command:
    """Generate and store components_specification, then bundle-validate layout-component."""
    s = runtime.state
    input_query = str(s.get("input_query", "") or "")
    layout_payload = s.get("layout_specification") or {}
    validated_candidate: Dict[str, Any] | None = None

    def _validate(candidate: Any) -> Optional[str]:
        nonlocal validated_candidate
        validated_candidate = None
        try:
            repaired_candidate = _repair_component_spec_from_layout(layout_payload, candidate)
            obj = _coerce_component_spec(repaired_candidate)
        except Exception as e:
            return f"[components] ComponentSpecification validation warning: {e}"
        if not obj.pages:
            return "[components] ComponentSpecification.pages must be non-empty"
        try:
            layout_obj = _coerce_layout_spec(layout_payload)
        except Exception as e:
            return f"[components] Cannot validate against layout_specification: {e}"
        component_contract_warning = _validate_component_contract_for_downstream(layout_obj, obj)
        if component_contract_warning:
            return component_contract_warning
        validated_candidate = obj.model_dump()
        return _bundle_validate_or_warn(
            stage="components",
            layout=layout_payload,
            component=obj,
            interaction={"pages": []},
            decoration={},
        )

    base_payload = {
        "input_query": input_query,
        "layout_specification": json.dumps(layout_payload, ensure_ascii=False),
        "required_component_contract": _build_component_contract_from_layout(layout_payload),
        "component_scaffold": json.dumps(_build_component_scaffold_from_layout(layout_payload), ensure_ascii=False),
    }
    candidate, error = _invoke_structured_with_retries(
        stage="components",
        structured_tool=components_specification_generator_tool,
        base_payload=base_payload,
        validate_result=_validate,
        max_attempts=3,
    )

    if error or candidate is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"FAILED to generate components_specification.\n{error}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    result_obj = _coerce_component_spec(validated_candidate or candidate)
    result = result_obj.model_dump()
    return Command(
        update={
            "components_specification": result,
            "messages": [ToolMessage(content="Stored components_specification.", tool_call_id=runtime.tool_call_id)],
        }
    )


interactions_prompt = PromptTemplate.from_template(
    """You are a UI InteractionSpecification generator.

INPUTS
- input_query: {input_query}
- layout_specification (JSON): {layout_specification}
- components_specification (JSON): {components_specification}
- validation_feedback (may be empty): {validation_feedback}

TASK
Generate an InteractionSpecification with per-page flows.

RETRY / FIXUP
- If validation_feedback is non-empty, it describes why the previous attempt failed validation.
- Correct the issues it mentions and regenerate a fully-valid InteractionSpecification.
- Keep flow names stable where possible (InteractionFlow.name).

HARD CONSTRAINTS
- InteractionSpecification.pages MUST cover every layout pageName (include pages with empty flows if needed).
- pageName MUST reference existing pages.
- trigger.targetId and any action.targetId MUST be valid ComponentNode.id values.
- For any ComponentEvent.flowName present in components_specification, create a matching InteractionFlow.name on the same page.
- For stateTransitions, toState/fromState MUST match declared ComponentState.name for that component.
- action.type MUST be one of: updateState, navigate, toggleVisibility, updateData, showFeedback, emit, addToCart, updateQuantity, applyFilter, clearFilter, toggleWishlist.
- result_state, if present, must be an object with a non-empty description string only.

OUTPUT RULES
- Return ONLY an InteractionSpecification object (no markdown, no explanation).
"""
)

interactions_specification_generator_chain = (
    interactions_prompt | model.with_structured_output(InteractionSpecification, method="function_calling")
)

interactions_specification_generator_tool = interactions_specification_generator_chain.as_tool(
    name="interactions_specification_generator",
    description="Generate InteractionSpecification from input_query + layout_specification + components_specification.",
    arg_types={
        "input_query": str,
        "layout_specification": str,
        "components_specification": str,
        "validation_feedback": str,
    },
)


@tool
def interactions_specification_generator(runtime: ToolRuntime[state]) -> Command:
    """Generate and store interactions_specification, then bundle-validate layout-component-interaction."""
    s = runtime.state
    input_query = str(s.get("input_query", "") or "")
    layout_payload: Any = s.get("layout_specification") or {}
    components_payload: Any = s.get("components_specification") or {}

    def _validate(candidate: Any) -> Optional[str]:
        try:
            obj = _coerce_interaction_spec(candidate)
        except Exception as e:
            return f"[interactions] InteractionSpecification validation warning: {e}"
        if not obj.pages:
            return "[interactions] InteractionSpecification.pages must be non-empty"
        try:
            layout_obj = _coerce_layout_spec(layout_payload)
        except Exception as e:
            return f"[interactions] Cannot validate against layout_specification: {e}"
        layout_page_names = {p.pageName for p in layout_obj.pages}
        interaction_page_names = {p.pageName for p in obj.pages}
        missing_pages = layout_page_names - interaction_page_names
        if missing_pages:
            return "[interactions] InteractionSpecification missing pages for layout: " + ", ".join(sorted(missing_pages))
        return _bundle_validate_or_warn(
            stage="interactions",
            layout=layout_payload,
            component=components_payload,
            interaction=obj,
            decoration={},
        )

    base_payload = {
        "input_query": input_query,
        "layout_specification": json.dumps(layout_payload, ensure_ascii=False),
        "components_specification": json.dumps(components_payload, ensure_ascii=False),
    }
    candidate, error = _invoke_structured_with_retries(
        stage="interactions",
        structured_tool=interactions_specification_generator_tool,
        base_payload=base_payload,
        validate_result=_validate,
        max_attempts=3,
    )

    if error or candidate is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"FAILED to generate interactions_specification.\n{error}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    result_obj = _coerce_interaction_spec(candidate)
    result = result_obj.model_dump()
    return Command(
        update={
            "interactions_specification": result,
            "messages": [ToolMessage(content="Stored interactions_specification.", tool_call_id=runtime.tool_call_id)],
        }
    )


decoration_prompt = PromptTemplate.from_template(
    """You are a UI DecorationSpecification generator.

INPUTS
- input_query: {input_query}
- layout_specification (JSON): {layout_specification}
- components_specification (JSON): {components_specification}
- interactions_specification (JSON): {interactions_specification}
- validation_feedback (may be empty): {validation_feedback}

TASK
Generate a DecorationSpecification that defines global design tokens plus per-page componentDecorations.

RETRY / FIXUP
- If validation_feedback is non-empty, it describes why the previous attempt failed validation.
- Correct the issues it mentions and regenerate a fully-valid DecorationSpecification.
- Keep className tokens stable (do not invent new ones).

HARD CONSTRAINTS
- DecorationSpecification.pages MUST cover every layout pageName (include pages with empty componentDecorations if needed).
- Emit a ComponentDecoration ONLY for components whose className in components_specification is a non-empty string.
- If a component has className=null, is missing className, or has an empty/whitespace className, do NOT create a ComponentDecoration for it.
- ComponentDecoration.className is required and must always be a concrete non-empty string. Never emit null, an empty string, or whitespace.
- Every ComponentDecoration.componentId MUST be a valid ComponentNode.id.
- Every emitted ComponentDecoration.className MUST exactly match the target ComponentNode.className verbatim.
- Every StateVariant.stateName MUST be one of that component's declared states[].name.
- Do NOT invent classNames or componentIds.
- If a page has no components with a valid className, return that page with componentDecorations=[].

OUTPUT RULES
- Return ONLY a DecorationSpecification object (no markdown, no explanation).
"""
)

decoration_specification_generator_chain = (
    decoration_prompt | model.with_structured_output(DecorationSpecification, method="function_calling")
)

decoration_specification_generator_tool = decoration_specification_generator_chain.as_tool(
    name="decoration_specification_generator_llm",
    description="Generate DecorationSpecification from input_query + design_schema + layout_specification + components_specification + interactions_specification.",
    arg_types={
        "input_query": str,
        "design_schema": str,
        "layout_specification": str,
        "components_specification": str,
        "interactions_specification": str,
        "validation_feedback": str,
    },
)


@tool
def decoration_specification_generator(runtime: ToolRuntime[state]) -> Command:
    """Generate and store decoration_specification, then full bundle-validate."""
    s = runtime.state
    input_query = str(s.get("input_query", "") or "")
    design_schema_payload: Any = s.get("design_schema") or {}
    components_specification: Any = s.get("components_specification") or {}
    layout_specification: Any = s.get("layout_specification") or {}
    interactions_specification: Any = s.get("interactions_specification") or {"pages": []}

    def _validate(candidate: Any) -> Optional[str]:
        try:
            obj = _coerce_decoration_spec(candidate)
        except Exception as e:
            return f"[decoration] DecorationSpecification validation warning: {e}"
        try:
            layout_obj = _coerce_layout_spec(layout_specification)
        except Exception as e:
            return f"[decoration] Cannot validate against layout_specification: {e}"
        layout_page_names = {p.pageName for p in layout_obj.pages}
        decoration_page_names = {p.pageName for p in obj.pages}
        missing_pages = layout_page_names - decoration_page_names
        if missing_pages:
            return "[decoration] DecorationSpecification missing pages for layout: " + ", ".join(sorted(missing_pages))
        return _bundle_validate_or_warn(
            stage="decoration",
            layout=layout_specification,
            component=components_specification,
            interaction=interactions_specification,
            decoration=obj,
        )

    base_payload = {
        "input_query": input_query,
        "design_schema": json.dumps(design_schema_payload, ensure_ascii=False),
        "components_specification": json.dumps(components_specification, ensure_ascii=False),
        "layout_specification": json.dumps(layout_specification, ensure_ascii=False),
        "interactions_specification": json.dumps(interactions_specification, ensure_ascii=False),
    }
    candidate, error = _invoke_structured_with_retries(
        stage="decoration",
        structured_tool=decoration_specification_generator_tool,
        base_payload=base_payload,
        validate_result=_validate,
        max_attempts=3,
    )

    if error or candidate is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"FAILED to generate decoration_specification.\n{error}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    result_obj = _coerce_decoration_spec(candidate)
    result = result_obj.model_dump()
    return Command(
        update={
            "decoration_specification": result,
            "messages": [ToolMessage(content="Stored decoration_specification.", tool_call_id=runtime.tool_call_id)],
        }
    )


@before_agent
def reset_before(state, runtime):
    input_query = ""
    if isinstance(state, dict) and isinstance(state.get("input_query"), str):
        input_query = state.get("input_query", "") or ""

    if not input_query and isinstance(state, dict):
        msgs = state.get("messages") or []
        try:
            for m in reversed(list(msgs)):
                if hasattr(m, "type") and getattr(m, "type") == "human" and hasattr(m, "content"):
                    input_query = str(getattr(m, "content") or "")
                    break
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    input_query = m.get("content") or ""
                    break
        except Exception:
            pass

    return {
        "input_query": input_query,
        "design_schema": {},
        "tavily_search_result": "",
    }


@before_agent
def context_agent_progress(state, runtime):
    get_stream_writer()("   [Context Agent] Processing request...")
    return None


@wrap_tool_call
def context_tool_progress(request, handler):
    writer = get_stream_writer()
    tool_name = ""
    try:
        tool_name = (request.tool_call or {}).get("name") or ""
    except Exception:
        tool_name = ""

    if tool_name == "extract_intent_signals":
        writer("   [Context Agent] Extracting intent signals...")
    elif tool_name == "tavily_search_tool":
        writer("   [Context Agent] Retrieving time-specific context...")
    elif tool_name == "update_extra_requirements":
        writer("   [Context Agent] Updating design requirements...")
    return handler(request)


_context_system_prompt = """
ROLE
- You are the Context Engineering (Schema) Agent.
- Your job is to extract ONLY renderable UI/UX requirements into `design_schema` for HTML/CSS/JS generation.

BEHAVIOR RULES
- If the user query is unrelated to e-commerce software generation, reject the request clearly and do not proceed with the workflow.

INPUT
- The user request is provided in `state.input_query` and is the ONLY source of truth.

STATE (read/write)
- `input_query`
- `design_schema` (MUST strictly follow DesignSchema)
- `tavily_search_result` (external context, MUST remain separate)

AVAILABLE TOOLS (ONLY THESE)
- `extract_intent_signals` (REQUIRED FIRST STEP)
- `tavily_search_tool` (OPTIONAL)
- `update_extra_requirements` (OPTIONAL)

MIDDLEWARE
- `reset_before` runs automatically before each invocation:
  - Resets `design_schema` and `tavily_search_result`
  - Preserves `input_query` exactly

STRICT OUTPUT CONTRACT
- `design_schema` MUST strictly conform to DesignSchema.
- DO NOT add new fields.
- DO NOT modify enum values.
- Use null or [] for unspecified fields.
- DO NOT infer or assume unspecified attributes (e.g., colors, features, styles).
- Output must be minimal, explicit, and deterministic.

TOOL EXECUTION RULES
- You MUST call `extract_intent_signals` before any other action.
- You MUST NOT manually construct or modify `design_schema`.
- If `extract_intent_signals` is not called, the task is incomplete.

ROUTING RULES (Tavily)
Call `tavily_search_tool` ONLY if the query explicitly contains:
- a specific year (e.g., 2024, 2026)
- "latest", "current", "trending", "this year"
- explicit time comparison (e.g., "2018 vs 2026")

DO NOT trigger Tavily for vague terms like "modern", "clean", or "nice UI".

WORKFLOW (STRICT ORDER)
0) `reset_before` (automatic)
1) Read `state.input_query`
2) Call `extract_intent_signals`
3) IF time-aware intent -> call `tavily_search_tool`
4) IF `tavily_search_result` not empty -> call `update_extra_requirements`
5) STOP

HARD RULES
- Do NOT paste Tavily output into schema
- Only `update_extra_requirements` may modify `extra_requirements`
- `design_schema.constraints.tech_stack` MUST be ["html","css","javascript"]
- Only include UI-renderable requirements
- Keep outputs short, concrete, and structured
"""

reasoning = {"effort": "medium", "summary": "auto"}

model = ChatOpenAI(model="gpt-5.4", reasoning=reasoning)


context_engineering_agent = create_agent(
    model=model,
    system_prompt=_context_system_prompt,
    tools=[extract_intent_signals, tavily_search_tool, update_extra_requirements],
    middleware=[context_agent_progress, reset_before, context_tool_progress],
    state_schema=state,
    checkpointer=InMemorySaver(),
)


@before_agent
def design_agent_progress(state, runtime):
    writer = get_stream_writer()
    writer("\t[Design Agent] Processing request...")
    return None


@wrap_tool_call
def design_tool_progress(request, handler):
    writer = get_stream_writer()
    tool_name = ""
    try:
        tool_name = (request.tool_call or {}).get("name") or ""
    except Exception:
        tool_name = ""

    msg = None
    if tool_name == "layout_specification_generator":
        msg = "\t[Design Agent] Generating layout specification..."
    elif tool_name == "components_specification_generator":
        msg = "\t[Design Agent] Generating component specifications..."
    elif tool_name == "interactions_specification_generator":
        msg = "\t[Design Agent] Generating interaction specifications..."
    elif tool_name == "decoration_specification_generator":
        msg = "\t[Design Agent] Generating decoration specifications..."

    if msg:
        writer(msg)

    return handler(request)


_design_system_prompt = """
ROLE
- You are the System Planning & Design Agent.
- Your job is to produce a complete UI specification bundle (layout, components, interactions, decoration).

BEHAVIOR RULES
- If the user query is unrelated to e-commerce software generation, reject the request clearly and do not proceed with the workflow.

INPUT
- The user request is provided in `state.input_query` and is the ONLY source of truth for intent.
- `state.design_schema` (if present) is constraints-only input. DO NOT modify it.

STATE (read/write)
- Read: `input_query`, `design_schema`
- Write (pipeline artifacts):
  - `layout_specification`
  - `components_specification`
  - `interactions_specification`
  - `decoration_specification`

AVAILABLE TOOLS (ONLY THESE - DO NOT INVENT OTHERS)
1) `layout_specification_generator`
   - Inputs: `input_query`, `design_schema`
   - Output stored as: `layout_specification`
2) `components_specification_generator`
   - Inputs: `input_query`, `layout_specification`
   - Output stored as: `components_specification`
3) `interactions_specification_generator`
   - Inputs: `input_query`, `layout_specification`, `components_specification`
   - Output stored as: `interactions_specification`
4) `decoration_specification_generator`
   - Inputs: `input_query`, `layout_specification`, `components_specification`, `interactions_specification`
   - Output stored as: `decoration_specification`

RETRY POLICY (MANDATORY)
- Tools may fail due to schema/contract validation issues. These are retryable.
- If a tool did NOT write its expected output field into state, you MUST retry that SAME tool up to 2 additional times (total 3 tool calls) before stopping.
- When a tool returns a ToolMessage containing validation warnings / FAILED reasons, treat that as feedback: retry the same tool and expect it to correct the issues.
- Do not proceed to later stages unless the prerequisite state fields exist and are valid/non-empty.

WORKFLOW (FULLY STRUCTURED, STRICT, AND SEQUENTIAL)
0) Preconditions
   - Confirm `state.input_query` is a non-empty string.
   - If `input_query` is empty, STOP and ask the user for a concrete UI request.

1) Call `layout_specification_generator`
   1.1) Verify `layout_specification.pages` exists in Agent state and is non-empty; if missing, RETRY `layout_specification_generator` (up to 2 more times).
   1.2) If still missing after retries, STOP and report the failure reason from the latest tool message.

2) Call `components_specification_generator`
   2.1) Verify `components_specification.pages` exists in Agent state and is non-empty; if missing, RETRY `components_specification_generator` (up to 2 more times).
   2.2) If still missing after retries, STOP and report the failure reason from the latest tool message.

3) Call `interactions_specification_generator`
   3.1) Verify `interactions_specification.pages` exists in Agent state and is non-empty; if missing, RETRY `interactions_specification_generator` (up to 2 more times).
   3.2) If still missing after retries, STOP and report the failure reason from the latest tool message.

4) Call `decoration_specification_generator`
   4.1) Verify `decoration_specification.pages` exists in Agent state and is non-empty; if missing, RETRY `decoration_specification_generator` (up to 2 more times).
   4.2) If still missing after retries, STOP and report the failure reason from the latest tool message.

5) STOP

ERROR HANDLING (STRICT)
- If a required prerequisite is missing for a stage, do NOT skip ahead.
- Do NOT manually fabricate or patch a missing spec/schema in plain text.
- Use tools only.
"""


@before_agent
def reset_before_system_design_planning(state, runtime):
    """Reset transient fields while preserving input_query + design_schema."""
    input_query = ""
    if isinstance(state, dict) and isinstance(state.get("input_query"), str):
        input_query = state.get("input_query", "") or ""

    if not input_query and isinstance(state, dict):
        msgs = state.get("messages") or []
        try:
            for m in reversed(list(msgs)):
                if hasattr(m, "type") and getattr(m, "type") == "human" and hasattr(m, "content"):
                    input_query = str(getattr(m, "content") or "")
                    break
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    input_query = m.get("content") or ""
                    break
        except Exception:
            pass

    design_schema = {}
    if isinstance(state, dict) and isinstance(state.get("design_schema"), dict):
        design_schema = state.get("design_schema") or {}

    return {
        "input_query": input_query,
        "design_schema": design_schema,
        "tavily_search_result": "",
    }


reasoning = {"effort": "medium", "summary": "auto"}

model = ChatOpenAI(model="gpt-5.4", reasoning=reasoning)


system_design_planning_agent = create_agent(
    model=model,
    system_prompt=_design_system_prompt,
    tools=[
        layout_specification_generator,
        components_specification_generator,
        interactions_specification_generator,
        decoration_specification_generator,
    ],
    middleware=[design_agent_progress, reset_before_system_design_planning, design_tool_progress],
    state_schema=state,
    checkpointer=InMemorySaver(),
)


class WorkingMemory(BaseModel):
    design_schema: Dict[str, Any] = Field(default_factory=dict)
    layout_specification: Dict[str, Any] = Field(default_factory=dict)
    components_specification: Dict[str, Any] = Field(default_factory=dict)
    interactions_specification: Dict[str, Any] = Field(default_factory=dict)
    decoration_specification: Dict[str, Any] = Field(default_factory=dict)


class state(AgentState):
    input_query: str = ""
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)


@tool
def call_context_engineering_agent(runtime: ToolRuntime[state], query: str) -> Command:
    """Invoke the context_engineering_agent and persist its design_schema into the orchestrator state."""
    writer = get_stream_writer()
    s = runtime.state or {}
    working_memory = s.get("working_memory") if isinstance(s, dict) and isinstance(s.get("working_memory"), dict) else {}

    writer({"type": "indent", "delta": 1})
    try:
        payload = {
            "messages": [{"role": "user", "content": query}],
            "input_query": query,
            "design_schema": {},
            "tavily_search_result": "",
        }

        sub_config = {"configurable": {"thread_id": f"ctx_{uuid.uuid4().hex[:8]}"}}

        captured_sub_state: Dict[str, Any] = {}

        def _merge_state_update(update: Any) -> None:
            if not isinstance(update, dict):
                return
            for key, value in update.items():
                if key == "messages":
                    continue
                captured_sub_state[key] = value

        for chunk in context_engineering_agent.stream(
            payload,
            stream_mode=["updates", "custom"],
            version="v2",
            config=sub_config,
        ):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type, data = chunk
            else:
                chunk_type = chunk.get("type")
                data = chunk.get("data")

            if chunk_type == "custom":
                if isinstance(data, (str, dict)):
                    writer(data)
                else:
                    writer(str(data))
            elif chunk_type == "updates" and isinstance(data, dict):
                for _step, update in data.items():
                    _merge_state_update(update)

        design_schema = captured_sub_state.get("design_schema") or {}
        next_working_memory = {**working_memory, "design_schema": design_schema}

        return Command(
            update={
                "design_schema": design_schema,
                "working_memory": next_working_memory,
                "messages": [
                    ToolMessage(
                        content="\t[Orchestrator] Context Engineering Agent completed. Returned design_schema backed to orchestrator.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    finally:
        writer({"type": "indent", "delta": -1})


@tool
def call_system_design_planning_agent(runtime: ToolRuntime[state], query: str) -> Command:
    """Invoke the system_design_planning_agent and persist its outputs into orchestrator working_memory."""
    writer = get_stream_writer()
    s = runtime.state or {}
    working_memory = s.get("working_memory") if isinstance(s, dict) and isinstance(s.get("working_memory"), dict) else {}
    input_query = query or (s.get("input_query", "") if isinstance(s, dict) else "")
    design_schema = working_memory.get("design_schema") or (s.get("design_schema") if isinstance(s, dict) else {}) or {}

    writer({"type": "indent", "delta": 1})
    try:
        payload = {
            "messages": [{"role": "user", "content": input_query}],
            "input_query": input_query,
            "design_schema": design_schema,
        }

        sub_config = {"configurable": {"thread_id": f"design_{uuid.uuid4().hex[:8]}"}}

        captured_sub_state: Dict[str, Any] = {}

        def _merge_state_update(update: Any) -> None:
            if not isinstance(update, dict):
                return
            for key, value in update.items():
                if key == "messages":
                    continue
                captured_sub_state[key] = value

        for chunk in system_design_planning_agent.stream(
            payload,
            stream_mode=["updates", "custom"],
            version="v2",
            config=sub_config,
        ):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type, data = chunk
            else:
                chunk_type = chunk.get("type")
                data = chunk.get("data")

            if chunk_type == "custom":
                if isinstance(data, (str, dict)):
                    writer(data)
                else:
                    writer(str(data))
            elif chunk_type == "updates" and isinstance(data, dict):
                for _step, update in data.items():
                    _merge_state_update(update)

        next_working_memory = {
            "design_schema": captured_sub_state.get("design_schema") or design_schema,
            "layout_specification": captured_sub_state.get("layout_specification") or {},
            "components_specification": captured_sub_state.get("components_specification") or {},
            "interactions_specification": captured_sub_state.get("interactions_specification") or {},
            "decoration_specification": captured_sub_state.get("decoration_specification") or {},
        }

        return Command(
            update={
                "working_memory": next_working_memory,
                "messages": [
                    ToolMessage(
                        content="\t[Orchestrator] System Design & Planning Agent completed. Returned planning outputs backed to orchestrator.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    finally:
        writer({"type": "indent", "delta": -1})


@tool
def update_input(runtime: ToolRuntime[state]) -> Command:
    """Persist the current input query into agent state."""
    s = runtime.state or {}
    input_query = s.get("input_query", "") if isinstance(s, dict) else ""

    return Command(
        update={
            "input_query": input_query,
            "messages": [
                ToolMessage(
                    content="\t[Orchestrator] Recorded request.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@before_agent
def reset_before(state, runtime):
    input_query = ""
    if isinstance(state, dict) and isinstance(state.get("input_query"), str):
        input_query = state.get("input_query", "") or ""

    if not input_query and isinstance(state, dict):
        msgs = state.get("messages") or []
        try:
            for m in reversed(list(msgs)):
                if hasattr(m, "type") and getattr(m, "type") == "human" and hasattr(m, "content"):
                    input_query = str(getattr(m, "content") or "")
                    break
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    input_query = m.get("content") or ""
                    break
        except Exception:
            pass

    return {
        "input_query": input_query,
        "working_memory": {
            "design_schema": {},
            "layout_specification": {},
            "components_specification": {},
            "interactions_specification": {},
            "decoration_specification": {},
        },
    }


@before_agent
def orchestrator_progress(state, runtime):
    writer = get_stream_writer()
    writer("[Orchestrator] Processing request...")
    return None


@wrap_tool_call
def orchestrator_tool_progress(request, handler):
    writer = get_stream_writer()
    tool_name = ""
    try:
        tool_name = (request.tool_call or {}).get("name") or ""
    except Exception:
        tool_name = ""

    if tool_name == "update_input":
        writer("[Orchestrator] Recording request...")
    elif tool_name == "call_context_engineering_agent":
        writer("[Orchestrator] Invoking Context Agent...")
    elif tool_name == "call_system_design_planning_agent":
        writer("[Orchestrator] Invoking System Design & Planning Agent...")
    return handler(request)


_orchestrator_prompt = """
ROLE
You are the Orchestrator Agent for an e-commerce web app UI generation system using HTML, CSS, and JavaScript.

GENERAL WORKFLOW
1. Validate and clarify.
   - Check the request against the CHECKLIST.
   - If details are missing, ask only for the missing checklist items.
   - If the user delegates decisions, proceed with sensible defaults.
   - If the request is out of scope, reject briefly and ask for a relevant request.
   - If the user asks about your role or capabilities, answer directly.
2. Execute.
   - Follow the WORKFLOW below.

CHECKLIST
1. The request is relevant to your role.
2. The requested pages or core user flow are clear enough.
3. The required features or components are clear enough.
4. Responsiveness or device expectations are specified.
5. Design constraints or preferences are clear enough.

WORKFLOW
1. Call `update_input`.
2. Call `call_context_engineering_agent` and store the returned schema in `working_memory.design_schema`.
3. Call `call_system_design_planning_agent` and store `layout_specification`, `components_specification`, `interactions_specification`, and `decoration_specification` in working memory.
4. Return the layout specification, components specification, interactions specification, and decoration specification.
5. After finishing all the tools invocation, simply conclude what you have done.

RULES
- Do not design the schema yourself.
- Ask only for missing checklist items unless the user delegated those decisions.
- Keep responses concise.

TOOLS
- `update_input`
- `call_context_engineering_agent`
- `call_system_design_planning_agent`
"""

reasoning = {"effort": "medium", "summary": "auto"}

model = ChatOpenAI(model="gpt-5.4", reasoning=reasoning)


orchestrator_agent = create_agent(
    model=model,
    system_prompt=_orchestrator_prompt,
    tools=[
        update_input,
        call_context_engineering_agent,
        call_system_design_planning_agent,
    ],
    middleware=[orchestrator_progress, reset_before, orchestrator_tool_progress],
    state_schema=state,
    checkpointer=InMemorySaver(),
)


graph = orchestrator_agent

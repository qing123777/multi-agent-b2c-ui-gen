"""Bridge between the Streamlit frontend (``stream_lit.py``) and the
LangGraph backend (``fyp.py``).

Design rules
------------
- This module is a pure, importable Python module.
- It has NO dependency on Streamlit and NEVER touches ``st.session_state``.
- It has NO side effects on import: heavy backend imports (``fyp.graph``,
  LangChain message classes) happen lazily inside the functions that
  actually need them.
- It is stateless: it holds no module-level mutable state. All state
  flows in as arguments and out as return values.
- It only ever returns plain Python types — strings, dicts, lists of
  dicts, or a small ``StreamSession`` helper. LangChain ``BaseMessage``
  objects are converted to ``{"role", "content"}`` dicts before they
  leave this module.

Public functions
----------------
- ``initialize_session() -> dict``
- ``new_chat(current_threads: dict) -> dict``
- ``send_message(query, thread_id, threads, llm) -> dict``           # batch
- ``stream_message(query, thread_id, threads, llm) -> StreamSession``  # streaming
- ``switch_chat(thread_id: str) -> list[dict]``
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterator, List, Tuple


# Sidebar label shown for any thread that hasn't received its first
# message yet. The frontend writes this into ``st.session_state.threads``
# until ``send_message`` / ``stream_message`` returns a generated label.
_PLACEHOLDER_LABEL = "New Chat"


# ---------------------------------------------------------------------------
# Internal helpers (private)
# ---------------------------------------------------------------------------

def _get_graph():
    """Return the compiled LangGraph graph exported by ``fyp.py``.

    Imported lazily so that ``import bridge`` itself does not pull in the
    full backend (which loads models, env vars, tools, etc.).
    """
    from fyp import graph  # local import — avoids import-time side effects
    return graph


def _config(thread_id: str) -> Dict[str, Any]:
    """Build the per-thread config dict expected by LangGraph."""
    return {"configurable": {"thread_id": thread_id}}


def _pending_tool_call_ids(messages: List[Any]) -> List[str]:
    """Return tool call ids that do not yet have matching tool outputs."""
    pending_order: List[str] = []
    completed = set()

    for msg in messages or []:
        msg_type = getattr(msg, "type", None)
        if msg_type == "ai":
            for tool_call in getattr(msg, "tool_calls", None) or []:
                if isinstance(tool_call, dict):
                    tool_call_id = tool_call.get("id")
                else:
                    tool_call_id = getattr(tool_call, "id", None)
                if isinstance(tool_call_id, str) and tool_call_id:
                    pending_order.append(tool_call_id)
        elif msg_type == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None)
            if isinstance(tool_call_id, str) and tool_call_id:
                completed.add(tool_call_id)

    return [tool_call_id for tool_call_id in pending_order if tool_call_id not in completed]


def _prune_thread_tail_to_checkpoint(thread_id: str, checkpoint_ns: str, latest_id: str, safe_id: str) -> bool:
    """Drop checkpoint entries newer than ``safe_id`` for one thread."""
    graph = _get_graph()
    checkpointer = getattr(graph, "checkpointer", None)
    storage = getattr(checkpointer, "storage", None)
    writes = getattr(checkpointer, "writes", None)

    if not isinstance(storage, dict) or not latest_id or latest_id == safe_id:
        return False

    thread_storage = storage.get(thread_id)
    if not isinstance(thread_storage, dict):
        return False

    ns_storage = thread_storage.get(checkpoint_ns)
    if not isinstance(ns_storage, dict):
        return False

    current_id = latest_id
    pruned = False

    while current_id and current_id != safe_id:
        entry = ns_storage.pop(current_id, None)
        if isinstance(writes, dict):
            writes.pop((thread_id, checkpoint_ns, current_id), None)
        pruned = pruned or entry is not None
        if not entry:
            break
        current_id = entry[2] if len(entry) >= 3 else None

    return pruned


def _repair_incomplete_thread(thread_id: str) -> bool:
    """Rewind a thread if cancellation left an unmatched tool call in history."""
    graph = _get_graph()
    config = _config(thread_id)

    try:
        current_snapshot = graph.get_state(config=config)
    except Exception:
        return False

    current_values = getattr(current_snapshot, "values", None) or {}
    current_messages = list(current_values.get("messages", []) or [])
    if not _pending_tool_call_ids(current_messages):
        return False

    safe_snapshot = None
    try:
        for snapshot in graph.get_state_history(config):
            values = getattr(snapshot, "values", None) or {}
            messages = list(values.get("messages", []) or [])
            if not _pending_tool_call_ids(messages):
                safe_snapshot = snapshot
                break
    except Exception:
        return False

    if safe_snapshot is None:
        checkpointer = getattr(graph, "checkpointer", None)
        delete_thread = getattr(checkpointer, "delete_thread", None)
        if callable(delete_thread):
            delete_thread(thread_id)
            return True
        return False

    latest_cfg = getattr(current_snapshot, "config", None) or {}
    safe_cfg = getattr(safe_snapshot, "config", None) or {}
    latest_conf = latest_cfg.get("configurable", {}) if isinstance(latest_cfg, dict) else {}
    safe_conf = safe_cfg.get("configurable", {}) if isinstance(safe_cfg, dict) else {}

    latest_id = latest_conf.get("checkpoint_id")
    safe_id = safe_conf.get("checkpoint_id")
    checkpoint_ns = latest_conf.get("checkpoint_ns", "")

    if not isinstance(latest_id, str) or not isinstance(safe_id, str):
        return False

    return _prune_thread_tail_to_checkpoint(thread_id, checkpoint_ns, latest_id, safe_id)


def _get_thread_messages(thread_id: str) -> List[Any]:
    """Return the raw ``BaseMessage`` list stored for ``thread_id``.

    Returns an empty list if the thread has never been invoked (the
    InMemorySaver has no checkpoint for it yet).
    """
    _repair_incomplete_thread(thread_id)
    graph = _get_graph()
    try:
        state = graph.get_state(config=_config(thread_id))
    except Exception:
        return []
    values = getattr(state, "values", None) or {}
    return list(values.get("messages", []) or [])


def _msg_role(msg: Any) -> str:
    """Map a LangChain ``BaseMessage`` to a flat ``user``/``assistant`` role."""
    msg_type = getattr(msg, "type", None)
    if msg_type == "human":
        return "user"
    return "assistant"


def _msg_content(msg: Any) -> str:
    """Best-effort extraction of plain text from a ``BaseMessage``."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def _to_plain(messages: List[Any]) -> List[Dict[str, str]]:
    """Convert a list of ``BaseMessage`` objects to plain dicts."""
    plain: List[Dict[str, str]] = []
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type not in ("human", "ai"):
            continue
        content = _msg_content(msg)
        if not content:
            continue
        plain.append({"role": _msg_role(msg), "content": content})
    return plain


def _generate_label(query: str, llm) -> str:
    """Use ``llm`` to summarise ``query`` into a 5-word-or-fewer chat title."""
    if llm is None:
        return ""
    prompt = (
        "Summarize this in 5 words or fewer, as a chat title. "
        "Return only the title text - no surrounding quotes, no trailing "
        "punctuation, no prefix like 'Title:'.\n\n"
        f"Message: {query}"
    )
    try:
        result = llm.invoke(prompt)
    except Exception:
        return ""

    text = getattr(result, "content", result)
    if isinstance(text, list):
        parts: List[str] = []
        for block in text:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text") or ""
                if isinstance(t, str) and t:
                    parts.append(t)
        text = "".join(parts)
    if not isinstance(text, str):
        text = str(text)

    label = text.strip().strip('"').strip("'").strip()
    label = label[:60].strip()
    return label


# ---------------------------------------------------------------------------
# Public API — session lifecycle + non-streaming send
# ---------------------------------------------------------------------------

def initialize_session() -> Dict[str, Any]:
    """Produce a fresh session-state payload for first app load."""
    thread_id = str(uuid.uuid4())
    return {
        "active_thread": thread_id,
        "threads": {thread_id: _PLACEHOLDER_LABEL},
    }


def new_chat(current_threads: Dict[str, str]) -> Dict[str, Any]:
    """Append a new thread to the sidebar without hitting the backend."""
    threads = dict(current_threads or {})
    thread_id = str(uuid.uuid4())
    threads[thread_id] = _PLACEHOLDER_LABEL
    return {"active_thread": thread_id, "threads": threads}


def send_message(
    query: str,
    thread_id: str,
    threads: Dict[str, str],
    llm,
) -> Dict[str, Any]:
    """Non-streaming send. Kept for callers that still want a single dict.

    Use :func:`stream_message` for the live UI path.
    """
    from langchain_core.messages import HumanMessage  # lazy

    threads = dict(threads or {})
    graph = _get_graph()
    config = _config(thread_id)

    is_first_message = len(_get_thread_messages(thread_id)) == 0

    state = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    response_text = ""
    msgs = (state or {}).get("messages", []) or []
    for msg in reversed(msgs):
        if getattr(msg, "type", None) == "ai":
            text = _msg_content(msg)
            if text:
                response_text = text
                break

    label_updated = False
    if is_first_message:
        new_label = _generate_label(query, llm)
        if new_label:
            threads[thread_id] = new_label
            label_updated = True

    return {
        "response": response_text,
        "threads": threads,
        "label_updated": label_updated,
    }


def switch_chat(thread_id: str) -> List[Dict[str, str]]:
    """Fetch ``thread_id``'s history from LangGraph as plain dicts."""
    return _to_plain(_get_thread_messages(thread_id))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _extract_token_text(raw: Any) -> str:
    """Pull the user-visible text out of one ``graph.stream`` chunk.

    With ``stream_mode='messages'`` LangGraph yields tuples of
    ``(AIMessageChunk, metadata)``. ``AIMessageChunk.content`` may be a
    plain string or a list of provider content blocks (reasoning,
    tool_call, text, ...). We surface only the user-facing ``text``
    portion so the UI doesn't print intermediate reasoning or tool
    payloads as if they were the answer.
    """
    try:
        from langchain_core.messages import AIMessageChunk  # lazy
    except Exception:
        AIMessageChunk = None  # type: ignore

    token = raw[0] if isinstance(raw, tuple) and raw else raw

    if AIMessageChunk is not None and not isinstance(token, AIMessageChunk):
        return ""

    content = getattr(token, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)
    return ""


class StreamSession:
    """Iterable wrapper over ``graph.stream`` for one user query.

    Two ways to consume the session:

    - :meth:`stream_events` yields typed ``(kind, text)`` tuples where
      ``kind`` is either ``"answer"`` (real assistant tokens) or
      ``"processing"`` (orchestrator middleware progress strings). The
      Streamlit frontend uses this to render processing steps in a
      separate, collapsible block from the final answer.
    - ``__iter__`` is the back-compat path for ``st.write_stream``: it
      yields plain strings, formatting processing lines as italics so
      they read as inline status.

    After either iterator is exhausted these attributes are populated:

    - ``response``         — the full concatenated assistant reply.
    - ``processing_steps`` — list of progress strings emitted via
      ``get_stream_writer()`` from the LangGraph middlewares.
    - ``threads``          — the (possibly relabelled) threads dict.
    - ``label_updated``    — True iff this was the thread's first
      message and a non-empty label was generated.
    """

    def __init__(
        self,
        query: str,
        thread_id: str,
        threads: Dict[str, str],
        llm,
    ) -> None:
        self.query = query
        self.thread_id = thread_id
        self.threads: Dict[str, str] = dict(threads or {})
        self._llm = llm

        self.response: str = ""
        self.processing_steps: List[str] = []
        self.label_updated: bool = False
        self._consumed: bool = False

    def stream_events(self) -> Iterator[Tuple[str, str]]:
        """Yield typed ``(kind, text)`` tuples.

        ``kind`` is ``"answer"`` for real assistant tokens and
        ``"processing"`` for orchestrator progress strings. Frontends
        that want to render processing in a separate UI element should
        prefer this over ``__iter__``.
        """
        # Detect first-message status BEFORE invoking — once the graph
        # runs, the message list is no longer empty.
        was_first = len(_get_thread_messages(self.thread_id)) == 0

        from langchain_core.messages import HumanMessage  # lazy

        graph = _get_graph()
        config = _config(self.thread_id)

        chunks: List[str] = []
        try:
            # Multi-mode streaming: "messages" gives us LLM token
            # deltas; "custom" gives us anything the orchestrator's
            # middlewares pushed via ``get_stream_writer()`` (e.g.
            # "[Context Agent] Processing request..."). Without
            # "custom" those progress lines never reach the UI.
            stream_iter = graph.stream(
                {"messages": [HumanMessage(content=self.query)]},
                config=config,
                stream_mode=["messages", "custom"],
            )
            for raw in stream_iter:
                # When multiple modes are requested, LangGraph yields
                # (mode_name, payload) tuples. Defensive fallback: if
                # we somehow get a bare payload, treat it as messages.
                if (
                    isinstance(raw, tuple)
                    and len(raw) == 2
                    and isinstance(raw[0], str)
                    and raw[0] in ("messages", "custom", "updates")
                ):
                    mode, payload = raw
                else:
                    mode, payload = "messages", raw

                if mode == "messages":
                    text = _extract_token_text(payload)
                    if text:
                        chunks.append(text)
                        yield ("answer", text)
                elif mode == "custom":
                    # Middlewares emit strings ("Processing...") or
                    # formatting dicts ({"type": "indent", ...}). Only
                    # yield the human-readable strings; skip layout
                    # dicts entirely.
                    if isinstance(payload, str):
                        s = payload.strip()
                        if s:
                            self.processing_steps.append(s)
                            yield ("processing", s)
        finally:
            self.response = "".join(chunks)

            # Label generation runs strictly AFTER the stream finishes
            # so the user sees the assistant reply first.
            if was_first:
                new_label = _generate_label(self.query, self._llm)
                if new_label:
                    self.threads[self.thread_id] = new_label
                    self.label_updated = True

            self._consumed = True

    def __iter__(self) -> Iterator[str]:
        """Back-compat string iterator suitable for ``st.write_stream``.

        Processing lines are rendered inline as italics. New frontend
        code should call :meth:`stream_events` directly so processing
        progress can be routed to a separate, collapsible UI block.
        """
        for kind, text in self.stream_events():
            if kind == "answer":
                yield text
            elif kind == "processing":
                yield f"\n_{text}_\n"


def stream_message(
    query: str,
    thread_id: str,
    threads: "dict",
    llm,
) -> "StreamSession":
    """Streaming counterpart to :func:`send_message`.

    Returns an iterable :class:`StreamSession`. Iterating it yields text
    chunks produced by ``graph.stream(..., stream_mode='messages')``.
    Once exhausted, the session exposes ``response``, ``threads``, and
    ``label_updated`` for the frontend to write back into session state.
    """
    return StreamSession(query, thread_id, threads, llm)

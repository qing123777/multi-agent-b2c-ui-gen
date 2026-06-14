from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Literal, Optional, Sequence, Tuple

try:
    from langchain_core.messages import AIMessage, AIMessageChunk
except Exception:  # pragma: no cover
    AIMessage = None  # type: ignore
    AIMessageChunk = None  # type: ignore


EventType = Literal[
    "reasoning",
    "tool_call",
    "tool_result",
    "token",
    "update",
    "final",
    "error",
]


@dataclass(frozen=True)
class StreamEvent:
    """Structured event emitted by `stream_invoke_events`.

    Notes:
    - `reasoning` comes from provider-specific `content_blocks` (e.g. OpenAI reasoning summaries).
      Not all models/providers expose it.
    - `tool_call`/`tool_result` are best-effort: depending on agent/runtime, you may see them
      in message chunks (`content_blocks`) and/or in `updates` as ToolMessages.
    """

    type: EventType
    text: str = ""
    name: str = ""
    args: Any = None
    tool_call_id: str = ""
    data: Any = None


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _as_event_dict(event: StreamEvent) -> Dict[str, Any]:
    return {
        "type": event.type,
        "text": event.text,
        "name": event.name,
        "args": event.args,
        "tool_call_id": event.tool_call_id,
        "data": event.data,
    }


def _normalize_stream_chunk(chunk: Any) -> Tuple[str, Any]:
    """Normalize LangGraph `stream` outputs.

    Common shapes:
    - ("messages", (AIMessageChunk, metadata))
    - ("updates", {node: {"messages": [...]}})
    - {"type": "messages", "data": (AIMessageChunk, metadata)}
    """

    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        return chunk[0], chunk[1]

    if isinstance(chunk, dict) and "type" in chunk and "data" in chunk:
        return str(chunk["type"]), chunk["data"]

    # Fallback: treat as "messages" payload
    return "messages", chunk


def format_update_summary(
    data: Any,
    *,
    resummarizer: Optional[Callable[[str], str]] = None,
) -> str:
    """Extract a compact readable one-liner from a LangGraph/LangChain `updates` payload.

    This is intentionally conservative + provider-agnostic.
    """

    def _extract_tool_calls_from_content(content_obj: Any) -> list[str]:
        tool_calls: list[str] = []
        if not isinstance(content_obj, list):
            return tool_calls

        for block in content_obj:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype not in {"function_call", "tool_call"}:
                continue

            name = block.get("name")
            arguments = block.get("arguments")

            fn = block.get("function") if isinstance(block.get("function"), dict) else None
            if not name and fn:
                name = fn.get("name")
            if arguments is None and fn:
                arguments = fn.get("arguments")

            if name:
                if isinstance(arguments, str):
                    arg_text = arguments.strip()
                else:
                    arg_text = _safe_json_dumps(arguments)
                tool_calls.append(f"{name}({arg_text})")
        return tool_calls

    def _extract_tool_calls_from_message(message_obj: Any) -> list[str]:
        if message_obj is None:
            return []

        content_obj = getattr(message_obj, "content", None)
        tool_calls: list[str] = []
        tool_calls.extend(_extract_tool_calls_from_content(content_obj))

        msg_tool_calls = getattr(message_obj, "tool_calls", None)
        if isinstance(msg_tool_calls, list):
            for tc in msg_tool_calls:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name")
                args = tc.get("args")
                if name:
                    tool_calls.append(f"{name}({_safe_json_dumps(args)})")
        return tool_calls

    if not isinstance(data, dict) or not data:
        return f"[update] Update : {str(data).strip()}"

    node_name: Optional[str] = None
    inner: Any = None
    if len(data) == 1:
        node_name = next(iter(data.keys()))
        inner = data[node_name]
    else:
        inner = data

    messages = inner.get("messages") if isinstance(inner, dict) else None
    if not (isinstance(messages, list) and messages):
        return f"[update] {node_name or 'Update'} : <no messages>"

    msg = messages[-1]
    msg_type = type(msg).__name__

    # Prefer tool result text when message looks like a ToolMessage
    tool_result_text = None
    if getattr(msg, "tool_call_id", None) is not None:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            tool_result_text = content.strip()

    summary_chunks: list[str] = []

    def _collect_summary_text(obj: Any) -> None:
        if isinstance(obj, dict):
            summaries = obj.get("summary")
            if isinstance(summaries, list):
                for s in summaries:
                    if not isinstance(s, dict):
                        continue
                    text = s.get("text")
                    if text is None:
                        continue
                    if s.get("type") == "summary_text":
                        summary_chunks.append(str(text))
            return

        if isinstance(obj, list):
            for item in obj:
                _collect_summary_text(item)
            return

    content = getattr(msg, "content", None)
    _collect_summary_text(content)

    if tool_result_text is not None:
        main_line = f"[update] ToolMessage : {tool_result_text}"
    else:
        if summary_chunks:
            summary_text = "\n".join(chunk.strip() for chunk in summary_chunks if str(chunk).strip())
        elif isinstance(content, str):
            summary_text = content.strip()
        else:
            summary_text = ""

        if summary_text and callable(resummarizer):
            try:
                summary_text = str(resummarizer(summary_text)).strip()
            except Exception:
                pass

        summary_text = " ".join(summary_text.split())
        if not summary_text:
            summary_text = "<no summary_text found>"
        main_line = f"[update] {msg_type} : {summary_text}"

    tool_calls_texts: list[str] = []
    for m in messages:
        tool_calls_texts.extend(_extract_tool_calls_from_message(m))

    seen: set[str] = set()
    tool_calls_texts = [t for t in tool_calls_texts if not (t in seen or seen.add(t))]

    if tool_calls_texts:
        tool_lines = "\n".join(f"[update] ToolCall : {t}" for t in tool_calls_texts)
        return f"{main_line}\n{tool_lines}"
    return main_line


def _iter_tool_results_from_updates(update_data: Any) -> Iterator[StreamEvent]:
    """Best-effort extraction of ToolMessage-like results from `updates` payload."""

    if not isinstance(update_data, dict):
        return

    # Common: {node: {"messages": [...]}}
    for inner in update_data.values():
        if not isinstance(inner, dict):
            continue
        messages = inner.get("messages")
        if not isinstance(messages, list):
            continue

        for msg in messages:
            # ToolMessage has `tool_call_id`; but we avoid strict isinstance checks.
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id is None:
                continue
            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content)
            yield StreamEvent(
                type="tool_result",
                text=text.strip(),
                tool_call_id=str(tool_call_id),
                data=msg,
            )


def stream_invoke_events(
    agent: Any,
    agent_input: Any,
    *,
    config: Optional[dict] = None,
    stream_mode: Sequence[str] = ("messages", "updates"),
    version: str = "v2",
    include_reasoning: bool = True,
    include_tool_calls: bool = True,
    include_tool_results: bool = True,
    include_updates: bool = True,
    resummarizer: Optional[Callable[[str], str]] = None,
    redact: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream an agent run as a sequence of structured events.

    Emits event dicts with keys: type, text, name, args, tool_call_id, data.

    Customization hooks:
    - `resummarizer(text)->text`: compresses update summaries (good for UIs).
    - `redact(text)->text`: scrub secrets/PII from streamed text (applied to all `text` fields).
    - Turn on/off categories with `include_*` flags.

    Important limitation:
    - "Model reasoning" is only available if your model/provider emits it (e.g. OpenAI reasoning summaries
      via `AIMessageChunk.content_blocks`). Otherwise you will only get `token` text + tool events.
    """

    final_text_parts: list[str] = []
    seen_tool_calls: set[tuple[str, str, str]] = set()

    def _emit(ev: StreamEvent) -> Dict[str, Any]:
        if redact and ev.text:
            try:
                ev = StreamEvent(
                    type=ev.type,
                    text=str(redact(ev.text)),
                    name=ev.name,
                    args=ev.args,
                    tool_call_id=ev.tool_call_id,
                    data=ev.data,
                )
            except Exception:
                pass
        return _as_event_dict(ev)

    try:
        stream_iter = agent.stream(
            agent_input,
            config=config,
            stream_mode=list(stream_mode),
            version=version,
        )
    except TypeError:
        # Fallback: some runnables expose stream(input, config=...).
        stream_iter = agent.stream(agent_input, config=config)

    try:
        for raw_chunk in stream_iter:
            chunk_type, data = _normalize_stream_chunk(raw_chunk)

            if chunk_type == "messages":
                # Common: (AIMessageChunk, metadata)
                token = data[0] if isinstance(data, tuple) and data else data

                if AIMessageChunk is not None and not isinstance(token, AIMessageChunk):
                    continue

                # 1) Tool calls via `token.tool_calls` (some runtimes attach them here)
                if include_tool_calls:
                    token_tool_calls = getattr(token, "tool_calls", None)
                    if isinstance(token_tool_calls, list):
                        for tc in token_tool_calls:
                            if isinstance(tc, dict):
                                name = tc.get("name") or (tc.get("function", {}) or {}).get("name")
                                args = tc.get("args")
                                if args is None:
                                    args = (tc.get("function", {}) or {}).get("arguments")
                                tc_id = str(tc.get("id") or "")
                            else:
                                name = getattr(tc, "name", "")
                                args = getattr(tc, "args", None)
                                tc_id = str(getattr(tc, "id", ""))

                            if not name:
                                continue

                            args_text = args if isinstance(args, str) else _safe_json_dumps(args)
                            key = (str(name), str(args_text), str(tc_id))
                            if key in seen_tool_calls:
                                continue
                            seen_tool_calls.add(key)
                            yield _emit(StreamEvent(type="tool_call", name=str(name), args=args, text=args_text, tool_call_id=tc_id, data=tc))

                # 2) Provider content blocks: reasoning/text/tool_call/tool_result
                content_blocks = getattr(token, "content_blocks", None)
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")

                        if btype == "reasoning" and include_reasoning:
                            piece = (
                                block.get("summary")
                                or block.get("reasoning")
                                or block.get("text")
                                or block.get("content")
                            )
                            if piece:
                                yield _emit(StreamEvent(type="reasoning", text=str(piece)))

                        elif btype == "text":
                            text_piece = str(block.get("text", ""))
                            if text_piece:
                                final_text_parts.append(text_piece)
                                yield _emit(StreamEvent(type="token", text=text_piece))

                        elif btype in {"function_call", "tool_call"} and include_tool_calls:
                            fn = block.get("function") if isinstance(block.get("function"), dict) else None
                            name = block.get("name") or (fn.get("name") if fn else None)
                            args = block.get("arguments") if block.get("arguments") is not None else (fn.get("arguments") if fn else None)
                            tc_id = str(block.get("id") or "")

                            if name:
                                args_text = args if isinstance(args, str) else _safe_json_dumps(args)
                                key = (str(name), str(args_text), str(tc_id))
                                if key not in seen_tool_calls:
                                    seen_tool_calls.add(key)
                                    yield _emit(
                                        StreamEvent(
                                            type="tool_call",
                                            name=str(name),
                                            args=args,
                                            text=args_text,
                                            tool_call_id=tc_id,
                                            data=block,
                                        )
                                    )

                        elif btype == "tool_result" and include_tool_results:
                            content = block.get("content")
                            tool_call_id = str(block.get("tool_call_id") or "")
                            text = content if isinstance(content, str) else _safe_json_dumps(content)
                            yield _emit(
                                StreamEvent(
                                    type="tool_result",
                                    text=str(text).strip(),
                                    tool_call_id=tool_call_id,
                                    data=block,
                                )
                            )

            elif chunk_type == "updates":
                if include_updates:
                    summary = format_update_summary(data, resummarizer=resummarizer)
                    yield _emit(StreamEvent(type="update", text=summary, data=data))

                if include_tool_results:
                    for ev in _iter_tool_results_from_updates(data):
                        yield _emit(ev)

        final_text = "".join(final_text_parts).strip()
        yield _emit(StreamEvent(type="final", text=final_text))

    except Exception as e:
        yield _emit(StreamEvent(type="error", text=f"{type(e).__name__}: {e}"))
        raise


def stream_invoke_text(
    agent: Any,
    agent_input: Any,
    *,
    config: Optional[dict] = None,
    show_reasoning: bool = True,
    show_updates: bool = False,
    show_tools: bool = True,
    show_tool_results: bool = True,
    prefix_map: Optional[Dict[str, str]] = None,
) -> Iterator[str]:
    """UI-friendly text stream.

    This is a convenience wrapper over `stream_invoke_events` that yields *strings*.

    Customize:
    - `prefix_map`: change how different event types render.
    - Toggle `show_*` flags.

    Tip for Streamlit:
        st.write_stream(stream_invoke_text(agent, {"messages": [...] }))
    """

    prefixes = {
        "reasoning": "\n\n<sub>🤔 reasoning</sub>\n\n",
        "tool_call": "\n\n<sub>🛠 tool_call</sub> ",
        "tool_result": "\n\n<sub>📦 tool_result</sub> ",
        "update": "\n\n<sub>⚙ update</sub> ",
    }
    if prefix_map:
        prefixes.update(prefix_map)

    for ev in stream_invoke_events(agent, agent_input, config=config):
        et = ev.get("type")

        if et == "token":
            yield ev.get("text", "")

        elif et == "reasoning" and show_reasoning:
            txt = ev.get("text", "").strip()
            if txt:
                yield prefixes["reasoning"] + txt + "\n\n"

        elif et == "tool_call" and show_tools:
            name = ev.get("name", "")
            args = ev.get("text", "")
            yield f"{prefixes['tool_call']}{name}({args})\n\n"

        elif et == "tool_result" and show_tool_results:
            tool_call_id = ev.get("tool_call_id", "")
            txt = ev.get("text", "")
            label = f"[{tool_call_id}] " if tool_call_id else ""
            if txt:
                yield f"{prefixes['tool_result']}{label}{txt}\n\n"

        elif et == "update" and show_updates:
            txt = ev.get("text", "")
            if txt:
                yield prefixes["update"] + txt + "\n\n"

        # `final` is already represented by the streamed tokens; no need to yield again.

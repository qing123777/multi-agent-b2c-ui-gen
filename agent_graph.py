import os
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import InjectedState, ToolNode, create_react_agent

load_dotenv()


class EmailState(AgentState):
    """UI-friendly state schema for chat UIs.

    Inherits LangGraph's prebuilt `AgentState` which includes required keys
    (e.g., `messages`, `remaining_steps`) expected by `create_react_agent`.
    """


@tool
def read_email(state: Annotated[dict[str, Any], InjectedState]) -> str:
    """Read the most recent human message as the 'email to reply to'."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return "No email text found yet. Ask the user to paste the email content."


@tool
def send_email(body: str) -> str:
    """Send an email reply (demo stub)."""
    if not body.strip():
        return "Nothing to send: email body is empty."
    return "Email sent."


_MODEL_NAME = os.getenv("MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
_SYSTEM_PROMPT = (
    "You are an email assistant. "
    "When the user asks you to reply to an email, call `read_email` to retrieve the email text, "
    "draft a helpful reply, then call `send_email` with the final reply body."
)


def _hitl_only_send_email(request, execute):
    """Require human approval before executing `send_email`.

    In Studio/AgentChatUI, this shows as an interrupt. Resume with either:
    - `true`
    - or `{ "approved": true }`
    """
    tool_name = (request.tool_call or {}).get("name")
    if tool_name != "send_email":
        return execute(request)

    decision = interrupt(
        {
            "type": "tool_approval",
            "tool": tool_name,
            "tool_call": request.tool_call,
            "message": "Approve sending email?",
        }
    )

    approved = False
    if isinstance(decision, bool):
        approved = decision
    elif isinstance(decision, dict):
        approved = bool(decision.get("approved"))

    if not approved:
        return ToolMessage(
            content="send_email was not approved.",
            tool_call_id=(request.tool_call or {}).get("id", ""),
        )

    return execute(request)


tools = ToolNode([read_email, send_email], wrap_tool_call=_hitl_only_send_email)

graph = create_react_agent(
    model=ChatOpenAI(model=_MODEL_NAME, temperature=0),
    tools=tools,
    prompt=_SYSTEM_PROMPT,
    state_schema=EmailState,
)

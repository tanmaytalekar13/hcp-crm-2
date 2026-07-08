"""
The LangGraph agent that powers the conversational "Log Interaction" experience.

Role of the agent:
- Acts as the single orchestrator between the field rep's natural-language chat
  and the structured HCP interaction record in Postgres.
- Holds short-term conversational memory (per session_id) so a rep can say
  "log a visit with Dr. Mehta" then, a few turns later, "actually change the
  sentiment to positive" without repeating context.
- Decides which of its 5 tools to call based on intent: look up an HCP,
  create an interaction, edit one, pull history for context, or schedule a
  follow-up.
- Uses llama-3.3-70b-versatile (via Groq) as the reasoning/tool-calling model
  for orchestration, since it follows tool-use loops reliably. gemma2-9b-it
  (mandated by the task) does the actual required work: it's the model that
  runs inside log_interaction/edit_interaction to summarize the rep's notes
  and extract structured entities (topics, products, samples, sentiment).
"""
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from app.agent.llm import get_fallback_llm
from app.agent.tools import build_tools

# Hard ceiling on how many times the agent may call a tool in one turn.
# gemma2-9b-it is used inside the tools themselves (log_interaction /
# edit_interaction) for the mandated summarization/extraction step, but the
# *orchestration* LLM below is llama-3.3-70b-versatile: small models like
# gemma2-9b-it are unreliable at knowing when to stop calling tools, which
# was causing infinite agent<->tools loops and hitting LangGraph's recursion
# limit. This cap guarantees the graph always terminates.
MAX_TOOL_CALLS_PER_TURN = 6

SYSTEM_PROMPT = """You are the AI assistant embedded in the "Log Interaction" \
screen of a pharma CRM used by field representatives calling on Healthcare \
Professionals (HCPs).

Your job: help the rep log, review, or edit an interaction purely by chatting \
with them, using your tools rather than asking them to fill out a form.

Rules:
- If the rep mentions a doctor by name and you don't already know their hcp_id \
  from earlier in this conversation, call search_hcp first.
- When the rep describes what happened in a visit/call (products discussed, \
  samples left, doctor's reaction, next steps), call log_interaction with their \
  account as raw_notes - don't ask them to restate it in a rigid format.
- If the rep wants to correct or add detail to something already logged, use \
  edit_interaction.
- If they ask what was discussed last time, use get_interaction_history.
- If they mention a future commitment (send a study, another visit date, a \
  lunch-and-learn), use schedule_followup after logging.
- Keep replies brief, professional, and confirm what was recorded.

Critical rules about IDs and tool errors:
- NEVER invent, guess, or reuse a placeholder value for hcp_id or \
  interaction_id (e.g. do not write "logged_interaction_id" or "12345" or \
  similar). Only use an id exactly as it appears in the "id" field of a \
  tool's JSON result (from search_hcp, log_interaction, edit_interaction, or \
  get_interaction_history).
- If search_hcp returns an empty "matches" list, that HCP does not exist in \
  the system. Do NOT call any other tool with a made-up hcp_id in this case. \
  Instead, tell the rep plainly that no matching HCP was found and ask them \
  to check the spelling or confirm the HCP needs to be added first.
- If a tool's JSON result contains an "error" field, treat that step as \
  failed: do NOT call a follow-up tool that depends on its output (e.g. do \
  not call schedule_followup if log_interaction just returned an error). \
  Instead, tell the rep plainly what failed and, if it's something you can \
  fix (like a missing field), ask them for it or retry with corrected \
  arguments.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int


def build_graph(db: Session):
    tools = build_tools(db)
    llm = get_fallback_llm(temperature=0.1).bind_tools(tools)

    def call_model(state: AgentState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        count = state.get("tool_call_count", 0)
        if count >= MAX_TOOL_CALLS_PER_TURN:
            forced = SystemMessage(
                content="You have used the maximum number of tools for this "
                        "turn. Respond now in plain text summarizing what you "
                        "were able to do, with no further tool calls."
            )
            response = get_fallback_llm(temperature=0.1).invoke(messages + [forced])
            return {"messages": [response]}

        try:
            response = llm.invoke(messages)
        except Exception as exc:
            # Groq occasionally rejects a tool call generation outright (e.g.
            # strict schema mismatches like a numeric field coming back
            # quoted). Rather than let that exception blow up the whole SSE
            # stream, fall back to a plain-text (no tools) call so the turn
            # still ends with a coherent message to the rep.
            recovery = SystemMessage(
                content="Your previous attempt to call a tool failed due to "
                        f"a technical error ({type(exc).__name__}). Do not "
                        "retry the same tool call. Respond in plain text: "
                        "apologize briefly and tell the rep what you were "
                        "trying to do so they can retry, with no tool calls."
            )
            response = get_fallback_llm(temperature=0.1).invoke(messages + [recovery])
            return {"messages": [response]}

        new_count = count + (1 if getattr(response, "tool_calls", None) else 0)
        return {"messages": [response], "tool_call_count": new_count}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("tool_call_count", 0) <= MAX_TOOL_CALLS_PER_TURN:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
import json

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.database import get_db
from app import models, schemas
from app.agent.graph import build_graph

router = APIRouter()


def _load_history(db: Session, session_id: str, limit: int = 20):
    rows = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    messages = []
    for r in rows:
        if r.role == "user":
            messages.append(HumanMessage(content=r.content))
        elif r.role == "assistant":
            messages.append(AIMessage(content=r.content))
    return messages


def _save_message(db: Session, session_id: str, role: str, content: str, tool_name: str | None = None):
    try:
        msg = models.ChatMessage(session_id=session_id, role=role, content=content, tool_name=tool_name)
        db.add(msg)
        db.commit()
    except Exception:
        # Never let a chat-log write take down the response stream -
        # rollback so the session is usable again and swallow the error
        # (worst case: this one message isn't persisted to history).
        db.rollback()


@router.post("/chat/stream")
async def chat_stream(payload: schemas.ChatRequest, db: Session = Depends(get_db)):
    _save_message(db, payload.session_id, "user", payload.message)
    history = _load_history(db, payload.session_id)

    graph = build_graph(db)
    initial_state = {"messages": history + [HumanMessage(content=payload.message)]}

    def event_generator():
        final_text_parts = []
        try:
            for chunk in graph.stream(
                initial_state,
                stream_mode="updates",
                config={"recursion_limit": 40},
            ):
                for node_name, node_output in chunk.items():
                    for message in node_output.get("messages", []):
                        if isinstance(message, AIMessage):
                            if getattr(message, "tool_calls", None):
                                for tc in message.tool_calls:
                                    yield {
                                        "event": "tool_call",
                                        "data": json.dumps({"tool": tc["name"], "args": tc["args"]}),
                                    }
                            if message.content:
                                final_text_parts.append(message.content)
                                yield {"event": "token", "data": json.dumps({"text": message.content})}
                        elif isinstance(message, ToolMessage):
                            tool_result_raw = message.content
                            yield {
                                "event": "tool_result",
                                "data": json.dumps({"tool": message.name, "result": tool_result_raw}),
                            }
                            if message.name in (
                                "log_interaction",
                                "edit_interaction",
                                "schedule_followup",
                                "add_materials_shared",
                                "add_samples_distributed",
                                "record_outcome",
                            ):
                                try:
                                    parsed = json.loads(tool_result_raw)
                                    if "interaction" in parsed:
                                        yield {
                                            "event": "form_update",
                                            "data": json.dumps(parsed["interaction"]),
                                        }
                                except json.JSONDecodeError:
                                    pass
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client instead of hanging
            # A tool exception earlier in the stream can leave the session's
            # transaction aborted (Postgres refuses any further statements
            # until it's rolled back). Roll back here so the _save_message
            # call below doesn't also fail.
            db.rollback()
            error_text = (
                "Sorry, something went wrong while processing that "
                f"({type(exc).__name__}: {exc}). Please try again."
            )
            final_text_parts.append(error_text)
            yield {"event": "token", "data": json.dumps({"text": error_text})}
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

        final_text = "\n".join(final_text_parts).strip()
        if final_text:
            _save_message(db, payload.session_id, "assistant", final_text)
        yield {"event": "done", "data": json.dumps({"final_text": final_text})}

    return EventSourceResponse(event_generator())

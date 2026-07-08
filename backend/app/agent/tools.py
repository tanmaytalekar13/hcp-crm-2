"""
The 5 LangGraph tools available to the HCP interaction agent.

1. search_hcp            - find an HCP record to attach an interaction to
2. log_interaction       - create a new interaction record (LLM summarization + entity extraction)
3. edit_interaction      - modify a previously logged interaction (keeps an audit trail)
4. get_interaction_history - pull past interactions with an HCP for context ("what did we discuss last time?")
5. schedule_followup     - set/update the next-action + follow-up date on an interaction

Each tool is created as a closure bound to a live SQLAlchemy session so the
agent can be re-instantiated per request (per chat session) with its own DB
transaction, while still exposing a plain LangChain @tool interface to LangGraph.

IMPORTANT — error handling contract:
Every tool below is wrapped so that ANY exception (bad UUID, DB error, LLM
error) is caught, the DB session is rolled back, and a JSON {"error": ...}
string is returned to the agent instead of raising. This matters because a
single uncaught exception leaves the Postgres session in
`InFailedSqlTransaction` state — every subsequent query on that same session
(i.e. every other tool call in the same chat turn) then fails too, even if
unrelated. Rolling back immediately after any failure is what prevents one
bad call from cascading into a wall of "current transaction is aborted"
errors for the rest of the turn.
"""
import json
import uuid as uuid_lib
from datetime import datetime
from typing import Optional, List, Union

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.models import HCP, Interaction, InteractionType, Sentiment
from app.agent.llm import get_primary_llm


def _serialize_interaction(i: Interaction) -> dict:
    return {
        "id": i.id,
        "hcp_id": i.hcp_id,
        "interaction_type": i.interaction_type.value if i.interaction_type else None,
        "interaction_date": i.interaction_date.isoformat() if i.interaction_date else None,
        "summary": i.summary,
        "topics_discussed": i.topics_discussed,
        "products_discussed": i.products_discussed,
        "samples_distributed": i.samples_distributed,
        "sentiment": i.sentiment.value if i.sentiment else None,
        "next_action": i.next_action,
        "follow_up_date": i.follow_up_date.isoformat() if i.follow_up_date else None,
        "is_edited": i.is_edited,
    }


def _valid_uuid(value: str) -> bool:
    try:
        uuid_lib.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _heuristic_fallback(raw_text: str) -> dict:
    return {
        "summary": raw_text[:200],
        "topics_discussed": [],
        "products_discussed": [],
        "samples_distributed": [],
        "sentiment": "neutral",
        "next_action": "",
        "interaction_type": "visit",
    }


def _extract_structured_fields(raw_text: str) -> dict:
    """Use the primary LLM to turn free-text / chat transcript into structured
    CRM fields: summary, topics, products, samples, sentiment, next_action.

    Resilient by design: if the model call itself fails (deprecated model,
    rate limit, network blip, malformed JSON back), we fall back to a plain
    heuristic extraction rather than raising. That keeps log_interaction
    working (the record still gets saved) even if the LLM step degrades,
    instead of poisoning the whole turn."""
    try:
        llm = get_primary_llm(temperature=0)
        prompt = f"""You are a life-sciences CRM assistant. Extract structured data from
a field representative's account of a Healthcare Professional (HCP) interaction.

Return ONLY valid JSON, no markdown fences, no commentary, matching this schema:
{{
  "summary": "one or two sentence professional summary",
  "topics_discussed": ["list", "of", "clinical/topic", "keywords"],
  "products_discussed": ["list", "of", "drug/product", "names", "mentioned"],
  "samples_distributed": ["list", "of", "sample", "names", "if any, else empty list"],
  "sentiment": "positive | neutral | negative",
  "next_action": "a short recommended next step, or empty string",
  "interaction_type": "visit | call | email | conference | sample_drop"
}}

Field rep's account:
\"\"\"{raw_text}\"\"\"
"""
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json\n", "", 1) if text.startswith("json") else text
        return json.loads(text)
    except Exception:
        # Covers: model_decommissioned / other Groq API errors, network
        # errors, and malformed JSON from the model. Never let a bad LLM
        # call take down the whole tool call.
        return _heuristic_fallback(raw_text)


def build_tools(db: Session):
    """Return the list of 5 tools bound to this request's DB session."""

    @tool
    def search_hcp(query: str) -> str:
        """Search for a Healthcare Professional (HCP) by name, specialty, or city.
        Use this FIRST when the rep mentions a doctor's name, to find/confirm the
        hcp_id needed before logging an interaction. Returns a JSON list of matches."""
        try:
            like = f"%{query}%"
            results = (
                db.query(HCP)
                .filter(
                    (HCP.name.ilike(like))
                    | (HCP.specialty.ilike(like))
                    | (HCP.city.ilike(like))
                )
                .limit(5)
                .all()
            )
            if not results:
                return json.dumps({"matches": [], "message": "No HCP found. You may need to create one first."})
            return json.dumps({
                "matches": [
                    {"id": h.id, "name": h.name, "specialty": h.specialty, "hospital": h.hospital, "city": h.city}
                    for h in results
                ]
            })
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"search_hcp failed: {e}"})

    @tool
    def log_interaction(hcp_id: str, raw_notes: str, interaction_date: Optional[str] = None) -> str:
        """Log a NEW interaction with an HCP. Pass the hcp_id (from search_hcp)
        and raw_notes containing everything the rep said about the visit/call
        (in their own words, e.g. from the chat transcript). This tool calls the
        LLM internally to summarize the notes and extract structured entities:
        topics discussed, products discussed, samples distributed, sentiment,
        and a recommended next action. Returns the created interaction as JSON.
        The returned interaction's "id" field is the ONLY valid interaction_id
        to use in later tool calls (e.g. schedule_followup) for this record —
        never guess or invent one."""
        if not _valid_uuid(hcp_id):
            return json.dumps({"error": f"'{hcp_id}' is not a valid hcp_id. Call search_hcp first and use the exact id it returns."})
        try:
            hcp = db.query(HCP).filter(HCP.id == hcp_id).first()
            if not hcp:
                return json.dumps({"error": f"No HCP found with id {hcp_id}. Use search_hcp first."})

            extracted = _extract_structured_fields(raw_notes)

            try:
                i_type = InteractionType(extracted.get("interaction_type", "visit"))
            except ValueError:
                i_type = InteractionType.visit
            try:
                sentiment = Sentiment(extracted.get("sentiment", "neutral"))
            except ValueError:
                sentiment = Sentiment.neutral

            interaction = Interaction(
                hcp_id=hcp_id,
                interaction_type=i_type,
                interaction_date=datetime.fromisoformat(interaction_date) if interaction_date else datetime.utcnow(),
                raw_notes=raw_notes,
                summary=extracted.get("summary"),
                topics_discussed=extracted.get("topics_discussed", []),
                products_discussed=extracted.get("products_discussed", []),
                samples_distributed=extracted.get("samples_distributed", []),
                sentiment=sentiment,
                next_action=extracted.get("next_action"),
                created_via="chat",
            )
            db.add(interaction)
            db.commit()
            db.refresh(interaction)
            return json.dumps({"status": "logged", "interaction": _serialize_interaction(interaction)})
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"log_interaction failed: {e}"})

    @tool
    def edit_interaction(interaction_id: str, fields_to_update: str) -> str:
        """Edit/correct a previously logged interaction. `fields_to_update` must be
        a JSON string with any of: interaction_type, raw_notes, summary,
        topics_discussed, products_discussed, samples_distributed, sentiment,
        next_action, follow_up_date. Only the fields provided are changed; a
        snapshot of the prior values is kept in edit_history for audit purposes.
        If raw_notes is updated, summary/topics/products/sentiment are
        automatically re-extracted via the LLM unless explicitly overridden.
        interaction_id MUST be the exact "id" returned by a previous
        log_interaction / get_interaction_history call — never invent one."""
        if not _valid_uuid(interaction_id):
            return json.dumps({"error": f"'{interaction_id}' is not a valid interaction_id. Use the exact id from a prior log_interaction or get_interaction_history result."})
        try:
            interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
            if not interaction:
                return json.dumps({"error": f"No interaction found with id {interaction_id}"})

            try:
                updates = json.loads(fields_to_update)
            except json.JSONDecodeError:
                return json.dumps({"error": "fields_to_update must be a valid JSON object string"})

            prior_snapshot = _serialize_interaction(interaction)
            history = interaction.edit_history or []
            history.append({"before": prior_snapshot, "edited_at": datetime.utcnow().isoformat()})
            interaction.edit_history = history

            if "raw_notes" in updates and not any(
                k in updates for k in ("summary", "topics_discussed", "products_discussed", "sentiment")
            ):
                extracted = _extract_structured_fields(updates["raw_notes"])
                updates.setdefault("summary", extracted.get("summary"))
                updates.setdefault("topics_discussed", extracted.get("topics_discussed"))
                updates.setdefault("products_discussed", extracted.get("products_discussed"))
                updates.setdefault("sentiment", extracted.get("sentiment"))

            for field in (
                "raw_notes", "summary", "topics_discussed", "products_discussed",
                "samples_distributed", "next_action",
            ):
                if field in updates:
                    setattr(interaction, field, updates[field])

            if "interaction_type" in updates:
                try:
                    interaction.interaction_type = InteractionType(updates["interaction_type"])
                except ValueError:
                    pass
            if "sentiment" in updates:
                try:
                    interaction.sentiment = Sentiment(updates["sentiment"])
                except ValueError:
                    pass
            if "follow_up_date" in updates and updates["follow_up_date"]:
                interaction.follow_up_date = datetime.fromisoformat(updates["follow_up_date"])

            interaction.is_edited = True
            interaction.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(interaction)
            return json.dumps({"status": "updated", "interaction": _serialize_interaction(interaction)})
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"edit_interaction failed: {e}"})

    @tool
    def get_interaction_history(hcp_id: str, limit: Union[int, str] = 5) -> str:
        """Retrieve the most recent past interactions for an HCP, so the agent
        has context like 'what did we discuss last time' or to avoid duplicate
        sample drops. Returns a JSON list ordered most-recent-first.
        `limit` is a numeric string (e.g. "5") for how many past interactions
        to return."""
        if not _valid_uuid(hcp_id):
            return json.dumps({"error": f"'{hcp_id}' is not a valid hcp_id. Call search_hcp first and use the exact id it returns."})
        try:
            try:
                limit_int = int(limit)
            except (TypeError, ValueError):
                limit_int = 5
            rows = (
                db.query(Interaction)
                .filter(Interaction.hcp_id == hcp_id)
                .order_by(Interaction.interaction_date.desc())
                .limit(limit_int)
                .all()
            )
            return json.dumps({"history": [_serialize_interaction(r) for r in rows]})
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"get_interaction_history failed: {e}"})

    @tool
    def schedule_followup(interaction_id: str, follow_up_date: str, next_action: str) -> str:
        """Schedule a follow-up for a logged interaction: set the follow_up_date
        (ISO format, e.g. '2026-08-15') and next_action (e.g. 'Send updated
        efficacy study', 'Book lunch-and-learn'). Use this after log_interaction
        when the rep mentions a next step or future commitment. interaction_id
        MUST be the exact "id" field from that log_interaction call's result —
        never invent a placeholder id; if log_interaction failed, do not call
        this tool at all, tell the rep it failed instead."""
        if not _valid_uuid(interaction_id):
            return json.dumps({"error": f"'{interaction_id}' is not a valid interaction_id. Use the exact id returned by log_interaction — do not invent one."})
        try:
            interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
            if not interaction:
                return json.dumps({"error": f"No interaction found with id {interaction_id}"})
            interaction.follow_up_date = datetime.fromisoformat(follow_up_date)
            interaction.next_action = next_action
            interaction.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(interaction)
            return json.dumps({"status": "scheduled", "interaction": _serialize_interaction(interaction)})
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"schedule_followup failed: {e}"})

    return [search_hcp, log_interaction, edit_interaction, get_interaction_history, schedule_followup]
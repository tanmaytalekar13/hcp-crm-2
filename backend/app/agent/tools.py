"""
The LangGraph tools available to the HCP interaction agent.

1. search_hcp            - find an HCP record to attach an interaction to
2. create_hcp            - create a missing HCP profile from chat-provided details
3. log_interaction       - create a new interaction record (LLM summarization + entity extraction)
4. edit_interaction      - modify a previously logged interaction (keeps an audit trail)
5. get_interaction_history - pull past interactions with an HCP for context ("what did we discuss last time?")
6. schedule_followup     - set/update the next-action + follow-up date on an interaction
7. add_materials_shared  - add/search selected materials to an interaction
8. add_samples_distributed - add distributed samples to an interaction
9. record_outcome        - update outcomes/agreements from chat

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
import re
import uuid as uuid_lib
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional, List, Union

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.models import HCP, Interaction, InteractionType, Sentiment
from app.agent.llm import get_primary_llm

STOPWORDS = {
    "dr", "doctor", "the", "from", "at", "hospital", "clinic", "medical",
    "center", "centre", "city", "and", "of", "in",
}

# Generic/filler next_action phrases the orchestrator LLM sometimes invents
# when calling schedule_followup (e.g. just "Follow up" or "Follow-up visit").
# These should never be allowed to clobber a specific, already-extracted
# next_action like "Send long-term safety data" — see schedule_followup below.
GENERIC_NEXT_ACTION_PHRASES = {
    "follow up", "follow-up", "followup",
    "schedule follow up", "schedule a follow up", "schedule follow-up",
    "schedule a follow-up", "schedule follow up visit",
    "schedule a follow-up visit", "schedule follow-up visit",
    "follow up visit", "follow-up visit", "followup visit",
}


def _serialize_interaction(i: Interaction) -> dict:
    return {
        "id": i.id,
        "hcp_id": i.hcp_id,
        "hcp_name": i.hcp.name if i.hcp else None,
        "interaction_type": i.interaction_type.value if i.interaction_type else None,
        "interaction_date": i.interaction_date.isoformat() if i.interaction_date else None,
        "raw_notes": i.raw_notes,
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


def _serialize_hcp(hcp: HCP, score: float | None = None, reason: str | None = None) -> dict:
    data = {
        "id": hcp.id,
        "name": hcp.name,
        "specialty": hcp.specialty,
        "hospital": hcp.hospital,
        "city": hcp.city,
    }
    if score is not None:
        data["match_score"] = round(score, 3)
    if reason:
        data["match_reason"] = reason
    return data


def _normalize_text(value: Optional[str]) -> str:
    text = (value or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Optional[str]) -> set[str]:
    return {token for token in _normalize_text(value).split() if token not in STOPWORDS}


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _score_hcp_match(hcp: HCP, query: str) -> tuple[float, str]:
    query_norm = _normalize_text(query)
    query_tokens = _tokens(query)

    name_norm = _normalize_text(hcp.name)
    name_tokens = _tokens(hcp.name)
    hospital_tokens = _tokens(hcp.hospital)
    specialty_tokens = _tokens(hcp.specialty)
    city_tokens = _tokens(hcp.city)
    combined_text = " ".join(filter(None, [hcp.name, hcp.specialty, hcp.hospital, hcp.city]))
    combined_tokens = _tokens(combined_text)

    name_ratio = SequenceMatcher(None, query_norm, name_norm).ratio() if query_norm and name_norm else 0.0
    name_overlap = _token_overlap(query_tokens, name_tokens)
    combined_overlap = _token_overlap(query_tokens, combined_tokens)
    hospital_overlap = _token_overlap(query_tokens, hospital_tokens)
    specialty_overlap = _token_overlap(query_tokens, specialty_tokens)
    city_overlap = _token_overlap(query_tokens, city_tokens)

    score = max(
        name_ratio * 0.9,
        name_overlap,
        combined_overlap * 0.95,
        (name_overlap * 0.75) + (hospital_overlap * 0.2) + (city_overlap * 0.05),
        (hospital_overlap * 0.65) + (specialty_overlap * 0.2) + (city_overlap * 0.15),
    )
    reasons = []
    if name_ratio >= 0.82:
        reasons.append("similar name")
    if name_overlap >= 0.5:
        reasons.append("name token overlap")
    if hospital_overlap > 0:
        reasons.append("hospital overlap")
    if specialty_overlap > 0:
        reasons.append("specialty overlap")
    if city_overlap > 0:
        reasons.append("city overlap")
    return score, ", ".join(reasons) or "semantic/fuzzy match"


def _rank_hcps(candidates: list[HCP], query: str, minimum_score: float = 0.38) -> list[tuple[HCP, float, str]]:
    ranked = []
    for hcp in candidates:
        score, reason = _score_hcp_match(hcp, query)
        if score >= minimum_score:
            ranked.append((hcp, score, reason))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def _merge_unique(existing: list[str] | None, additions: list[str]) -> list[str]:
    merged = []
    seen = set()
    for item in (existing or []) + additions:
        normalized = _normalize_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(item)
    return merged


def _heuristic_fallback(raw_text: str) -> dict:
    sentiment = _infer_sentiment(raw_text)
    return {
        "summary": raw_text[:200],
        "topics_discussed": _infer_topics(raw_text),
        "products_discussed": _infer_products(raw_text),
        "samples_distributed": [],
        "sentiment": sentiment,
        "next_action": _infer_next_action(raw_text),
        "interaction_type": "visit",
        "interaction_date": None,
    }


def _infer_sentiment(raw_text: str, return_confidence: bool = False):
    """Heuristic sentiment reader.

    Returns just the sentiment string by default. Pass return_confidence=True
    to also get back "strong" or "weak":
      - "strong": an unambiguous arc pattern matched (e.g. "skeptical ...
        but became interested"). These are reliable enough to override even
        a non-neutral sentiment the extraction LLM already produced, because
        the LLM (a small/cheap model) tends to anchor on the first
        sentiment-coded word in a sentence ("skeptical") rather than reading
        the whole arc through to "became interested" / "agreed".
      - "weak": a plain keyword-count read, only trustworthy enough to fill
        in when the LLM returned "neutral" (i.e. said nothing useful).
    """
    text = _normalize_text(raw_text)
    positive_phrases = (
        "strong interest", "very interested", "showed interest", "interested in",
        "positive", "excited", "enthusiastic", "requested additional",
        "requested more", "asked for more", "wants more", "open to",
        "agreed", "approved", "accepted", "receptive",
    )
    negative_phrases = (
        "not interested", "no interest", "rejected", "declined", "concerned",
        "concerns", "skeptical", "not convinced", "unhappy", "negative",
        "refused", "pushed back", "pushback",
    )
    positive_matches = sum(1 for phrase in positive_phrases if phrase in text)
    negative_matches = sum(1 for phrase in negative_phrases if phrase in text)

    def _result(sentiment: str, confidence: str):
        return (sentiment, confidence) if return_confidence else sentiment

    # Strong, deterministic "arc" patterns — checked first and treated as
    # authoritative regardless of raw keyword counts.
    if "but became interested" in text or "but became" in text or "however became interested" in text:
        return _result("positive", "strong")
    if "initially skeptical" in text and "interested" in text:
        return _result("positive", "strong")
    if "initially skeptical" in text and "agreed" in text:
        return _result("positive", "strong")

    if positive_matches > negative_matches:
        return _result("positive", "weak")
    if negative_matches > positive_matches:
        return _result("negative", "weak")
    if any(phrase in text for phrase in positive_phrases):
        return _result("positive", "weak")
    if any(phrase in text for phrase in negative_phrases):
        return _result("negative", "weak")
    return _result("neutral", "weak")


def _infer_topics(raw_text: str) -> list[str]:
    text = _normalize_text(raw_text)
    topics = []
    if "clinical trial" in text or "trial data" in text:
        topics.append("clinical trial data")
    if "diabetes" in text:
        topics.append("diabetes portfolio")
    if "oncology" in text:
        topics.append("oncology portfolio")
    if "efficacy" in text:
        topics.append("efficacy")
    if "safety" in text:
        topics.append("safety")
    return topics


def _infer_products(raw_text: str) -> list[str]:
    text = _normalize_text(raw_text)
    products = []
    if "diabetes portfolio" in text:
        products.append("diabetes portfolio")
    if "oncology portfolio" in text:
        products.append("oncology portfolio")
    if "antibiotic" in text:
        products.append("antibiotic product")
    return products


def _infer_next_action(raw_text: str) -> str:
    """Build a next_action string from raw notes. Unlike the old version,
    this can combine more than one distinct commitment (e.g. "send safety
    data" AND "visit next Tuesday") instead of the first matching branch
    silently winning and dropping the rest, and it never falls back to the
    bare word "follow up" when a more specific action is identifiable."""
    text = _normalize_text(raw_text)
    actions: list[str] = []

    if "long term safety data" in text or ("safety data" in text and "requested" in text):
        actions.append("Send long-term safety data")
    elif "clinical trial data" in text or "trial data" in text:
        actions.append("Send additional clinical trial data")

    if "next tuesday" in text:
        actions.append("Schedule follow-up visit for next Tuesday")
    elif "follow up visit" in text or "follow-up visit" in text or "followup visit" in text:
        actions.append("Schedule follow-up visit")
    elif not actions and ("follow up" in text or "followup" in text or "follow-up" in text):
        actions.append("Schedule a follow-up visit")

    if not actions and ("evaluate the product" in text or "agreed to evaluate" in text):
        actions.append("Check back after their evaluation")

    return "; ".join(actions)


def _infer_follow_up_date(raw_text: str) -> Optional[str]:
    text = _normalize_text(raw_text)
    if "next tuesday" in text:
        today = datetime.utcnow().date()
        target_weekday = 1  # Tuesday
        days_ahead = (target_weekday - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).isoformat()
    if "tomorrow" in text:
        return (datetime.utcnow().date() + timedelta(days=1)).isoformat()
    if "next week" in text:
        return (datetime.utcnow().date() + timedelta(days=7)).isoformat()
    return None


def _stabilize_extracted_fields(raw_text: str, extracted: dict) -> dict:
    stabilized = dict(extracted or {})

    # Sentiment: strong deterministic arc-patterns always win, even over a
    # non-neutral sentiment the extraction LLM already produced (that's the
    # fix — previously this only ever filled in when the LLM said
    # "neutral", so a wrongly-guessed "negative" from the LLM was never
    # corrected). Weak/keyword-count heuristics keep the old, more
    # conservative behavior of only filling in a "neutral" gap.
    semantic_sentiment, confidence = _infer_sentiment(raw_text, return_confidence=True)
    if confidence == "strong":
        stabilized["sentiment"] = semantic_sentiment
    elif semantic_sentiment != "neutral" and stabilized.get("sentiment", "neutral") == "neutral":
        stabilized["sentiment"] = semantic_sentiment

    if not stabilized.get("topics_discussed"):
        stabilized["topics_discussed"] = _infer_topics(raw_text)
    if not stabilized.get("products_discussed"):
        stabilized["products_discussed"] = _infer_products(raw_text)
    if not stabilized.get("next_action"):
        stabilized["next_action"] = _infer_next_action(raw_text)
    if not stabilized.get("interaction_date"):
        stabilized["interaction_date"] = _infer_follow_up_date(raw_text)
    if not stabilized.get("follow_up_date"):
        stabilized["follow_up_date"] = _infer_follow_up_date(raw_text)
    return stabilized


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


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

Pay close attention to the FULL arc of the interaction, not just the first
sentiment-coded word. For example "she was initially skeptical but became
interested after reviewing the data" is POSITIVE overall, not negative -
read to the end of the sentence/paragraph before deciding sentiment.

If the rep mentions more than one distinct next step (e.g. both "send more
data" and "visit again next week"), include both, semicolon-separated, in
next_action - do not collapse them into a generic phrase like "Follow up".

Return ONLY valid JSON, no markdown fences, no commentary, matching this schema:
{{
  "summary": "one or two sentence professional summary",
  "topics_discussed": ["list", "of", "clinical/topic", "keywords"],
  "products_discussed": ["list", "of", "drug/product", "names", "mentioned"],
  "samples_distributed": ["list", "of", "sample", "names", "if any, else empty list"],
  "sentiment": "positive | neutral | negative",
  "next_action": "a short recommended next step (or several, semicolon-separated), or empty string",
  "interaction_type": "visit | call | email | conference | sample_drop",
  "interaction_date": "ISO-8601 datetime if the rep mentioned a date/time, else null"
}}

Field rep's account:
\"\"\"{raw_text}\"\"\"
"""
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json\n", "", 1) if text.startswith("json") else text
        return _stabilize_extracted_fields(raw_text, json.loads(text))
    except Exception:
        # Covers: model_decommissioned / other Groq API errors, network
        # errors, and malformed JSON from the model. Never let a bad LLM
        # call take down the whole tool call.
        return _heuristic_fallback(raw_text)


def build_tools(db: Session):
    """Return the LangGraph tools bound to this request's DB session."""

    @tool
    def search_hcp(query: str) -> str:
        """Search for a Healthcare Professional (HCP) by name, specialty,
        hospital, or city. Use this FIRST when the rep mentions a doctor's
        name, to find/confirm the hcp_id needed before logging an interaction.
        This performs direct text search plus fuzzy/token-overlap matching so
        minor spelling differences do not create duplicate HCPs. Returns a JSON
        list of matches ordered by confidence."""
        try:
            query = (query or "").strip()
            if not query:
                return json.dumps({"matches": [], "message": "Search query is required."})

            like = f"%{query}%"
            direct_results = (
                db.query(HCP)
                .filter(
                    (HCP.name.ilike(like))
                    | (HCP.specialty.ilike(like))
                    | (HCP.hospital.ilike(like))
                    | (HCP.city.ilike(like))
                )
                .limit(5)
                .all()
            )

            if direct_results:
                ranked = _rank_hcps(direct_results, query, minimum_score=0)
            else:
                candidates = db.query(HCP).limit(100).all()
                ranked = _rank_hcps(candidates, query)

            if not ranked:
                return json.dumps({"matches": [], "message": "No HCP found. You may need to create one first."})
            return json.dumps({
                "matches": [
                    _serialize_hcp(h, score=score, reason=reason)
                    for h, score, reason in ranked[:5]
                ]
            })
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"search_hcp failed: {e}"})

    @tool
    def create_hcp(
        name: str,
        specialty: Optional[str] = None,
        hospital: Optional[str] = None,
        city: Optional[str] = None,
    ) -> str:
        """Create a new HCP profile when search_hcp returns no match and the
        rep's message contains enough identifying details. Use the doctor's
        name exactly as provided. If specialty, hospital, or city are missing,
        pass null/None rather than inventing values. Returns the new hcp_id as
        JSON so log_interaction can immediately use it."""
        if not name or not name.strip():
            return json.dumps({"error": "name is required to create an HCP"})
        try:
            normalized_name = name.strip()
            duplicate_query = " ".join(
                value for value in (normalized_name, specialty, hospital, city) if value
            )
            direct_existing = db.query(HCP).filter(HCP.name.ilike(normalized_name)).first()
            candidates = db.query(HCP).limit(100).all()
            ranked_duplicates = _rank_hcps(candidates, duplicate_query, minimum_score=0.58)
            existing = direct_existing or (ranked_duplicates[0][0] if ranked_duplicates else None)

            if existing:
                score, reason = _score_hcp_match(existing, duplicate_query)
                return json.dumps({
                    "status": "already_exists",
                    "message": "A similar HCP already exists; use this hcp_id instead of creating a duplicate.",
                    "hcp": _serialize_hcp(existing, score=score, reason=reason),
                })

            hcp = HCP(
                name=normalized_name,
                specialty=specialty or "Unknown/Not Provided",
                hospital=hospital,
                city=city,
            )
            db.add(hcp)
            db.commit()
            db.refresh(hcp)
            return json.dumps({
                "status": "created",
                "hcp": {
                    "id": hcp.id,
                    "name": hcp.name,
                    "specialty": hcp.specialty,
                    "hospital": hcp.hospital,
                    "city": hcp.city,
                },
            })
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"create_hcp failed: {e}"})

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
                interaction_date=(
                    _parse_datetime(interaction_date)
                    or _parse_datetime(extracted.get("interaction_date"))
                    or datetime.utcnow()
                ),
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
    def add_materials_shared(interaction_id: str, materials: str) -> str:
        """Add materials shared during a logged interaction. `materials` may be
        a JSON array string or comma-separated text, e.g. 'Brochure, Trial Data'.
        This backs the UI's Materials Shared Search/Add control."""
        if not _valid_uuid(interaction_id):
            return json.dumps({"error": f"'{interaction_id}' is not a valid interaction_id"})
        additions = _as_list(materials)
        if not additions:
            return json.dumps({"error": "At least one material is required"})
        try:
            interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
            if not interaction:
                return json.dumps({"error": f"No interaction found with id {interaction_id}"})
            interaction.products_discussed = _merge_unique(interaction.products_discussed, additions)
            interaction.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(interaction)
            return json.dumps({"status": "materials_added", "interaction": _serialize_interaction(interaction)})
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"add_materials_shared failed: {e}"})

    @tool
    def add_samples_distributed(interaction_id: str, samples: str) -> str:
        """Add samples distributed during a logged interaction. `samples` may be
        a JSON array string or comma-separated text. This backs the UI's Add
        Sample control."""
        if not _valid_uuid(interaction_id):
            return json.dumps({"error": f"'{interaction_id}' is not a valid interaction_id"})
        additions = _as_list(samples)
        if not additions:
            return json.dumps({"error": "At least one sample is required"})
        try:
            interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
            if not interaction:
                return json.dumps({"error": f"No interaction found with id {interaction_id}"})
            interaction.samples_distributed = _merge_unique(interaction.samples_distributed, additions)
            interaction.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(interaction)
            return json.dumps({"status": "samples_added", "interaction": _serialize_interaction(interaction)})
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"add_samples_distributed failed: {e}"})

    @tool
    def record_outcome(interaction_id: str, outcome: str, next_action: Optional[str] = None) -> str:
        """Record the outcome/agreement for an interaction from chat, backing the
        UI's Outcomes field. If next_action is provided, update follow-up actions
        too. Only supplied fields are changed."""
        if not _valid_uuid(interaction_id):
            return json.dumps({"error": f"'{interaction_id}' is not a valid interaction_id"})
        if not outcome or not outcome.strip():
            return json.dumps({"error": "outcome is required"})
        try:
            interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
            if not interaction:
                return json.dumps({"error": f"No interaction found with id {interaction_id}"})
            interaction.summary = outcome.strip()
            if next_action and next_action.strip():
                interaction.next_action = next_action.strip()
            interaction.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(interaction)
            return json.dumps({"status": "outcome_recorded", "interaction": _serialize_interaction(interaction)})
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"record_outcome failed: {e}"})

    @tool
    def edit_interaction(interaction_id: str, fields_to_update: str) -> str:
        """Edit/correct a previously logged interaction. `fields_to_update` must be
        a JSON string with any of: interaction_type, interaction_date, raw_notes, summary,
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
            if "interaction_date" in updates and updates["interaction_date"]:
                parsed_interaction_date = _parse_datetime(updates["interaction_date"])
                if parsed_interaction_date:
                    interaction.interaction_date = parsed_interaction_date
            if "sentiment" in updates:
                try:
                    interaction.sentiment = Sentiment(updates["sentiment"])
                except ValueError:
                    pass
            if "follow_up_date" in updates and updates["follow_up_date"]:
                parsed_follow_up = _parse_datetime(updates["follow_up_date"])
                if parsed_follow_up:
                    interaction.follow_up_date = parsed_follow_up

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
        this tool at all, tell the rep it failed instead.

        Note on next_action: if the interaction already has a specific
        next_action recorded (e.g. "Send long-term safety data") and this call
        passes only a generic filler like "Follow up" or "Follow-up visit",
        the specific action is preserved (with the visit appended) instead of
        being overwritten — pass a specific value here only when you actually
        have new/more precise information from the rep."""
        if not _valid_uuid(interaction_id):
            return json.dumps({"error": f"'{interaction_id}' is not a valid interaction_id. Use the exact id returned by log_interaction — do not invent one."})
        try:
            interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
            if not interaction:
                return json.dumps({"error": f"No interaction found with id {interaction_id}"})
            parsed_follow_up = _parse_datetime(follow_up_date)
            if not parsed_follow_up:
                return json.dumps({"error": "follow_up_date must be a valid ISO date or datetime"})
            interaction.follow_up_date = parsed_follow_up

            normalized_new = _normalize_text(next_action)
            existing_next_action = (interaction.next_action or "").strip()
            if existing_next_action and normalized_new in GENERIC_NEXT_ACTION_PHRASES:
                if "follow" not in _normalize_text(existing_next_action):
                    interaction.next_action = f"{existing_next_action}; Schedule follow-up visit"
                # else: existing_next_action already mentions a follow-up visit,
                # keep it as-is rather than replacing it with the generic text.
            else:
                interaction.next_action = next_action

            interaction.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(interaction)
            return json.dumps({"status": "scheduled", "interaction": _serialize_interaction(interaction)})
        except Exception as e:
            db.rollback()
            return json.dumps({"error": f"schedule_followup failed: {e}"})

    return [
        search_hcp,
        create_hcp,
        log_interaction,
        edit_interaction,
        get_interaction_history,
        schedule_followup,
        add_materials_shared,
        add_samples_distributed,
        record_outcome,
    ]
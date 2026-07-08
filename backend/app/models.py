import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey, Enum, Float, Boolean
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class InteractionType(str, enum.Enum):
    visit = "visit"
    call = "call"
    email = "email"
    conference = "conference"
    sample_drop = "sample_drop"


class Sentiment(str, enum.Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class HCP(Base):
    __tablename__ = "hcps"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    specialty = Column(String, nullable=True)
    hospital = Column(String, nullable=True)
    city = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    tier = Column(String, nullable=True)  # e.g. A/B/C prescriber tier
    created_at = Column(DateTime, default=datetime.utcnow)

    interactions = relationship("Interaction", back_populates="hcp")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    hcp_id = Column(UUID(as_uuid=False), ForeignKey("hcps.id"), nullable=False)

    interaction_type = Column(Enum(InteractionType), default=InteractionType.visit)
    interaction_date = Column(DateTime, default=datetime.utcnow)

    raw_notes = Column(Text, nullable=True)          # original free-text / chat transcript
    summary = Column(Text, nullable=True)             # LLM-generated summary
    topics_discussed = Column(JSONB, nullable=True)   # list of products/topics (entity extraction)
    products_discussed = Column(JSONB, nullable=True)
    samples_distributed = Column(JSONB, nullable=True)
    sentiment = Column(Enum(Sentiment), nullable=True)
    next_action = Column(Text, nullable=True)
    follow_up_date = Column(DateTime, nullable=True)

    created_via = Column(String, default="chat")
    is_edited = Column(Boolean, default=False)
    edit_history = Column(JSONB, nullable=True)     # list of prior versions for audit trail

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    hcp = relationship("HCP", back_populates="interactions")


class ChatMessage(Base):
    """Stores conversational turns so the agent has memory per session."""
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # user | assistant | tool
    content = Column(Text, nullable=False)
    tool_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

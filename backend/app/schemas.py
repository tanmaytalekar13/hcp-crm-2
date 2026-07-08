from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel


class HCPBase(BaseModel):
    name: str
    specialty: Optional[str] = None
    hospital: Optional[str] = None
    city: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    tier: Optional[str] = None


class HCPCreate(HCPBase):
    pass


class HCPOut(HCPBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class InteractionBase(BaseModel):
    hcp_id: str
    interaction_type: str = "visit"
    interaction_date: Optional[datetime] = None
    raw_notes: Optional[str] = None
    summary: Optional[str] = None
    topics_discussed: Optional[List[str]] = None
    products_discussed: Optional[List[str]] = None
    samples_distributed: Optional[List[str]] = None
    sentiment: Optional[str] = None
    next_action: Optional[str] = None
    follow_up_date: Optional[datetime] = None


class InteractionCreate(InteractionBase):
    created_via: str = "chat"


class InteractionUpdate(BaseModel):
    interaction_type: Optional[str] = None
    interaction_date: Optional[datetime] = None
    raw_notes: Optional[str] = None
    summary: Optional[str] = None
    topics_discussed: Optional[List[str]] = None
    products_discussed: Optional[List[str]] = None
    samples_distributed: Optional[List[str]] = None
    sentiment: Optional[str] = None
    next_action: Optional[str] = None
    follow_up_date: Optional[datetime] = None


class InteractionOut(InteractionBase):
    id: str
    created_via: str
    is_edited: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    session_id: str
    message: str
    # Optional agent-controlled form snapshot for future merge workflows.
    current_form_state: Optional[dict[str, Any]] = None

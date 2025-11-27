from pydantic import BaseModel
from typing import Optional, List


class PersonData(BaseModel):
    """Data model for person information sent to frontend via SSE."""
    name: str
    description: str
    relationship: str
    person_id: Optional[str] = None


class ConversationEntry(BaseModel):
    """Data model for a conversation entry in the conversation history."""
    timestamp: str
    direction: str  # "to_patient" | "from_patient"
    text: str
    source: str  # "voice" or other sources


class ExtractedIdentity(BaseModel):
    """Data model for extracted identity with confidence."""
    value: Optional[str]
    confidence: float


class ExtractedIdentityResponse(BaseModel):
    """Data model for identity extraction response with confidence."""
    transcript: str
    extracted: dict  # {name: ExtractedIdentity, relationship: ExtractedIdentity}


class LastConversationResponse(BaseModel):
    """Data model for last conversation response with summary and keywords."""
    messages: List[dict]
    short_summary: str
    keywords: List[str]


class PromoteTemporaryPersonRequest(BaseModel):
    """Data model for promoting a temporary person."""
    person_id: str
    name: Optional[str] = None
    relationship: Optional[str] = None
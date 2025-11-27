"""Data models for the inference service - AR Glasses with two event types."""

from datetime import datetime
from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class ConversationUtterance(BaseModel):
    """Single utterance in a conversation - no timestamp needed."""

    speaker: str = Field(..., description="Speaker identifier ('patient' or person_id)")
    text: str = Field(..., description="What was said")

    class Config:
        json_schema_extra = {
            "example": {
                "speaker": "person_001",
                "text": "Hi dad, how are you feeling today?"
            }
        }


class ConversationEvent(BaseModel):
    """Event from speaker diarization metadata service - two types."""

    event_type: Literal["PERSON_DETECTED", "CONVERSATION_END"] = Field(
        ..., description="Type of event: PERSON_DETECTED or CONVERSATION_END"
    )
    person_id: str = Field(..., description="Person identifier from diarization")
    timestamp: datetime | None = Field(None, description="Event timestamp (auto-generated if not provided)")
    conversation: list[ConversationUtterance] | None = Field(
        None, description="Structured conversation array (only for CONVERSATION_END)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "event_type": "PERSON_DETECTED",
                    "person_id": "person_001",
                    "timestamp": "2025-10-11T14:30:00Z"
                },
                {
                    "event_type": "CONVERSATION_END",
                    "person_id": "person_001",
                    "timestamp": "2025-10-11T14:35:00Z",
                    "conversation": [
                        {
                            "speaker": "person_001",
                            "text": "Hi dad, how are you feeling today?"
                        },
                        {
                            "speaker": "patient",
                            "text": "I'm doing well, thanks for asking."
                        },
                        {
                            "speaker": "person_001",
                            "text": "I got that promotion at work I mentioned!"
                        },
                        {
                            "speaker": "patient",
                            "text": "That's wonderful news! Congratulations!"
                        }
                    ]
                }
            ]
        }


class InferenceResult(BaseModel):
    """Simple inference result for AR glasses display - shows who the person is and recent context."""

    person_id: str = Field(..., description="Person identifier for internal tracking")
    name: str = Field(..., description="Person's name to display")
    relationship: str = Field(..., description="Relationship to patient")
    description: str = Field(..., description="One-line context for AR display")

    class Config:
        json_schema_extra = {
            "example": {
                "person_id": "person_001",
                "name": "Sarah",
                "relationship": "Your daughter",
                "description": "Last spoke 3 days ago about her promotion and the grandchildren visiting"
            }
        }


# New models for face recognition and voice profiles
class FaceEmbedding(BaseModel):
    """Face embedding vector for face recognition."""
    vector: List[float] = Field(..., description="Face embedding vector")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_image_url: Optional[str] = Field(None, description="URL to the source image")
    model: str = Field("facenet-resnet50", description="Model used to generate embedding")


class VoiceProfile(BaseModel):
    """Voice profile for speaker recognition."""
    embedding: List[float] = Field(..., description="Voice embedding vector")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sample_count: int = Field(default=1, description="Number of voice samples used")


class PersonWithBiometrics(BaseModel):
    """Extended person model with face and voice recognition data."""
    person_id: str = Field(..., description="Unique person identifier")
    name: str = Field(..., description="Person's name")
    relationship: str = Field(..., description="Relationship to patient")
    aggregated_context: str = Field("", description="Running summary of conversations")
    cached_description: str = Field("No previous interactions", description="One-line description for AR display")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Biometric data for recognition
    face_embeddings: List[FaceEmbedding] = Field(default_factory=list, description="Face embeddings for recognition")
    voice_profile: Optional[VoiceProfile] = Field(None, description="Voice profile for speaker recognition")
    
    # Conversation history
    conversation_history: List[ConversationEvent] = Field(default_factory=list, description="History of conversations with this person")
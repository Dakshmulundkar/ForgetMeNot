"""Shared data models for the backend application."""

from pydantic import BaseModel
from typing import Optional


class PersonData(BaseModel):
    """Person information sent to frontend via SSE."""
    name: str
    description: str
    relationship: str
    person_id: Optional[str] = None
"""Utility functions for the backend application."""

import asyncio
from typing import Dict
from backend.app.models import PersonData

# Store SSE event queues for person detection events
event_queues: Dict[str, asyncio.Queue] = {}


async def broadcast_person(person: PersonData):
    """Broadcast person detection to all SSE connections."""
    # Send to all event queues
    for queue in event_queues.values():
        await queue.put(person)
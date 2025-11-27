"""FastAPI application providing WebRTC ingress and audio pipeline stubs."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Use absolute imports instead of relative imports
from backend.app.audio import AdaptiveDenoiser, AudioPipeline, PipelineConfig
from backend.app.core import AudioChunk
from backend.app.services.conversation_stream import ConversationEventBus
from backend.app.models import PersonData

# Import MongoDB database functions
from inference.database import (
    add_face_embedding,
    get_person_by_id,
    find_person_by_face_embedding,
    add_conversation_to_history,
    create_person
)

logger = logging.getLogger("webrtc")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

app = FastAPI(title="Multimodal Ingress Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_INDEX = ROOT_DIR / "frontend" / "out" / "index.html"

# Store active WebSocket connections
active_websockets: Set[WebSocket] = set()

# Import event_queues from utils
from backend.app.utils import event_queues

# Load environment variables from project root .env (if present)
load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("TORCHAUDIO_PYTHON_ONLY", "1")

pipeline_config = PipelineConfig()
conversation_bus = ConversationEventBus()
audio_pipeline = AudioPipeline(
    denoiser=AdaptiveDenoiser(),
    config=pipeline_config,
    conversation_bus=conversation_bus,
)

class SDPModel(BaseModel):
    sdp: str
    type: str


class FaceEmbeddingRequest(BaseModel):
    person_id: str
    face_embedding: List[float]
    source_image_url: Optional[str] = None
    model: str = "facenet-resnet50"


class FaceRecognitionRequest(BaseModel):
    face_embedding: List[float]


class VoiceProfileRequest(BaseModel):
    person_id: str
    voice_embedding: List[float]
    sample_count: int = 1


class CreatePersonRequest(BaseModel):
    person_id: str
    name: str
    relationship: str
    aggregated_context: str = ""
    cached_description: str = "No previous interactions"


class UpdatePersonRequest(BaseModel):
    name: str
    relationship: str
    aggregated_context: str = ""
    cached_description: str = "No previous interactions"


@app.put("/person/{person_id}")
async def update_person_endpoint(person_id: str, request: UpdatePersonRequest):
    """
    Update an existing person in the database.
    
    Args:
        person_id: Person identifier
        request: Updated person data
        
    Returns:
        Updated person document
    """
    try:
        # Check if person exists
        person = get_person_by_id(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Update person in database
        from inference.database import update_person_context, get_people_collection
        success = update_person_context(
            person_id=person_id,
            aggregated_context=request.aggregated_context,
            cached_description=request.cached_description
        )
        
        if success:
            # Update the person's name and relationship
            collection = get_people_collection()
            result = collection.update_one(
                {"person_id": person_id},
                {
                    "$set": {
                        "name": request.name,
                        "relationship": request.relationship,
                        "last_updated": datetime.utcnow()
                    }
                }
            )
            
            if result.matched_count > 0:
                # Get updated person data
                updated_person = get_person_by_id(person_id)
                if updated_person and '_id' in updated_person:
                    del updated_person['_id']
                return updated_person
            else:
                raise HTTPException(status_code=500, detail="Failed to update person")
        else:
            raise HTTPException(status_code=500, detail="Failed to update person context")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating person: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def broadcast_person(person: PersonData):
    """Broadcast person detection to all SSE connections."""
    # Send to all event queues
    for queue in event_queues.values():
        await queue.put(person)


@app.get("/stream/inference")
async def events_stream(session_id: str = "default"):
    """SSE endpoint for streaming person detection events."""
    
    async def event_generator():
        # Import broadcast_person here to avoid circular import
        from backend.app.utils import broadcast_person
        
        # Create a queue for this session
        queue: asyncio.Queue[PersonData] = asyncio.Queue()
        event_queues[session_id] = queue
        
        try:
            # Send initial connection event
            yield f"event: inference\ndata: {PersonData(name='Connected', description='Streaming service connected', relationship='System', person_id=None).model_dump_json()}\n\n"
            
            # Stream person data from queue
            while True:
                try:
                    person_data = await queue.get()
                    yield f"event: inference\ndata: {person_data.model_dump_json()}\n\n"
                except asyncio.CancelledError:
                    logger.info(f"📡 SSE client disconnected: {session_id}")
                    break
        finally:
            # Clean up queue when client disconnects
            event_queues.pop(session_id, None)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.on_event("startup")
async def on_startup() -> None:
    try:
        await audio_pipeline.warm_whisper()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper warm-up failed: %s", exc)


@app.get("/")
async def index() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=500, detail="Frontend bundle missing")
    return FileResponse(FRONTEND_INDEX)


# WebSocket endpoint for audio streaming
@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for audio streaming.
    
    Usage in frontend:
    ```js
    const ws = new WebSocket('ws://localhost:8000/ws/audio');
    // Send audio data as binary
    ws.send(audioBuffer);
    ```
    """
    await websocket.accept()
    active_websockets.add(websocket)
    session_id = f"sess-{uuid4().hex[:8]}"
    logger.info(f"🔌 WebSocket audio client connected (session: {session_id})")

    try:
        # Listen for incoming audio data
        while True:
            data = await websocket.receive_bytes()
            logger.debug(f"Received audio data: {len(data)} bytes")
            
            # Process audio data
            sample_rate = 16000  # Default sample rate
            chunk = AudioChunk(
                session_id=session_id,
                data=data,
                sample_rate=sample_rate,
                timestamp=datetime.utcnow(),
            )
            
            try:
                await audio_pipeline.process_chunk(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Audio pipeline error for session %s: %s",
                    session_id,
                    exc,
                )
                continue
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket audio client disconnected (session: {session_id})")
    except Exception as e:
        logger.error(f"🔌 WebSocket audio error: {e}")
    finally:
        active_websockets.discard(websocket)
        # Flush any remaining audio data
        try:
            await audio_pipeline.flush_session(session_id, 16000)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Error flushing audio buffer for session %s: %s",
                session_id,
                exc,
            )


@app.post("/face/embedding")
async def add_face_embedding_endpoint(request: FaceEmbeddingRequest):
    """
    Add a face embedding to a person's profile.
    
    Args:
        request: Face embedding data
        
    Returns:
        Success message
    """
    try:
        # Check if person exists
        person = get_person_by_id(request.person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Add face embedding
        face_embedding_data = {
            "vector": request.face_embedding,
            "created_at": datetime.utcnow(),
            "source_image_url": request.source_image_url,
            "model": request.model
        }
        
        success = add_face_embedding(request.person_id, face_embedding_data)
        if success:
            return {"message": "Face embedding added successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add face embedding")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding face embedding: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/face/recognize")
async def recognize_face_endpoint(request: FaceRecognitionRequest):
    """
    Recognize a person by face embedding.
    
    Args:
        request: Face embedding data
        
    Returns:
        Person information if found, otherwise unknown
    """
    try:
        # Try to find person by face embedding
        person = find_person_by_face_embedding(request.face_embedding)
        
        if person:
            # Return person information
            return {
                "known": True,
                "person_id": person["person_id"],
                "name": person["name"],
                "relationship": person["relationship"],
                "description": person["cached_description"],
                "conversation_history": person.get("conversation_history", [])[-5:]  # Last 5 conversations
            }
        else:
            # Return unknown person
            return {
                "known": False,
                "message": "Unknown person detected"
            }
    except Exception as e:
        logger.error(f"Error recognizing face: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/person")
async def create_person_endpoint(request: CreatePersonRequest):
    """
    Create a new person in the database.
    
    Args:
        request: Person data
        
    Returns:
        Created person document
    """
    try:
        # Create person in database
        person = create_person(
            person_id=request.person_id,
            name=request.name,
            relationship=request.relationship,
            aggregated_context=request.aggregated_context,
            cached_description=request.cached_description
        )
        
        # Broadcast the new person to frontend via SSE
        from backend.app.utils import broadcast_person
        person_data = PersonData(
            name=request.name,
            description=request.cached_description,
            relationship=request.relationship,
            person_id=request.person_id
        )
        await broadcast_person(person_data)
        
        # Convert MongoDB document to JSON-serializable format
        if '_id' in person:
            del person['_id']
        return person
    except Exception as e:
        logger.error(f"Error creating person: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/people")
async def list_people_endpoint():
    """
    List all people in the database.
    
    Returns:
        List of all people
    """
    try:
        from inference.database import list_all_people
        people = list_all_people()
        # Convert MongoDB documents to JSON-serializable format
        for person in people:
            if '_id' in person:
                del person['_id']
            # Handle datetime serialization for face embeddings
            if 'face_embeddings' in person:
                for embedding in person['face_embeddings']:
                    if 'created_at' in embedding and hasattr(embedding['created_at'], 'isoformat'):
                        embedding['created_at'] = embedding['created_at'].isoformat()
            # Handle datetime serialization for last_updated
            if 'last_updated' in person and person['last_updated'] is not None and hasattr(person['last_updated'], 'isoformat'):
                person['last_updated'] = person['last_updated'].isoformat()
            # Handle datetime serialization for voice_profile
            if 'voice_profile' in person and person['voice_profile']:
                if 'created_at' in person['voice_profile'] and person['voice_profile']['created_at'] is not None and hasattr(person['voice_profile']['created_at'], 'isoformat'):
                    person['voice_profile']['created_at'] = person['voice_profile']['created_at'].isoformat()
            # Handle datetime serialization for conversation_history
            if 'conversation_history' in person:
                for conversation in person['conversation_history']:
                    if 'timestamp' in conversation and conversation['timestamp'] is not None and hasattr(conversation['timestamp'], 'isoformat'):
                        conversation['timestamp'] = conversation['timestamp'].isoformat()
        return people
    except Exception as e:
        logger.error(f"Error listing people: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Shutdown complete")


@app.get("/stream/conversation")
async def stream_conversation() -> StreamingResponse:
    """Server-Sent Events stream of conversation metadata events."""

    async def event_generator():
        queue = await conversation_bus.subscribe()
        try:
            while True:
                try:
                    event = await queue.get()
                except asyncio.CancelledError:
                    break
                payload = event.model_dump_json()
                yield f"event: conversation\ndata: {payload}\n\n"
        finally:
            await conversation_bus.unsubscribe(queue)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
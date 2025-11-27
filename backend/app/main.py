"""FastAPI application providing WebRTC ingress and audio pipeline stubs."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
from uuid import uuid4

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import base64
import time
import re
from collections import Counter

# Use absolute imports instead of relative imports
from backend.app.audio import AdaptiveDenoiser, AudioPipeline, PipelineConfig
from backend.app.core import AudioChunk
from backend.app.services.conversation_stream import ConversationEventBus
from backend.app.models import PersonData, ExtractedIdentity, ExtractedIdentityResponse, LastConversationResponse, PromoteTemporaryPersonRequest

# Import MongoDB database functions
from inference.database import (
    add_face_embedding,
    get_person_by_id,
    find_person_by_face_embedding,
    add_conversation_to_history,
    create_person,
    add_conversation,
    get_last_conversation_summary
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


class TranscribeAndStoreRequest(BaseModel):
    audio: str  # Base64 encoded audio
    person_id: Optional[str] = None
    direction: str  # "to_patient" or "from_patient"


class TranscribeAndStoreResponse(BaseModel):
    transcript: str
    stored: bool
    person_id: Optional[str] = None


class InferIdentityRequest(BaseModel):
    audio: str  # Base64 encoded audio


class InferIdentityResponse(BaseModel):
    transcript: str
    extracted: dict  # Will be updated to use ExtractedIdentity model


class LogConversationRequest(BaseModel):
    """Request model for logging conversation endpoint."""
    person_id: Optional[str] = None
    direction: str  # "to_patient" | "from_patient" | "dialogue"
    audio: str  # Base64 encoded audio data


class LogConversationResponse(BaseModel):
    """Response model for logging conversation endpoint."""
    person_id: str
    text: str
    stored: bool


class VoiceContextResponse(BaseModel):
    """Response model for voice context endpoint."""
    person_id: str
    conversations: List[Dict[str, Any]]
    short_summary: str


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


def extract_name_and_relationship_with_confidence(transcript: str) -> dict:
    """
    Extract name and relationship from transcript using regex patterns with confidence scores.
    
    Args:
        transcript: Transcript text to analyze
        
    Returns:
        Dictionary with extracted name and relationship with confidence scores
    """
    import re
    
    # Convert to lowercase for easier matching
    text = transcript.lower().strip()
    
    # Track pattern matches for confidence calculation
    name_matches = []
    relationship_matches = []
    
    # Extract candidate name using patterns with confidence weights
    name = None
    name_patterns = [
        (r"my name is ([a-z]+)", 0.9),  # Strong pattern
        (r"you can call me ([a-z]+)", 0.8),
        (r"this is ([a-z]+)", 0.7),
        (r"i'm your [a-z]+ ([a-z]+)", 0.9),  # For cases like "I'm your daughter Emma"
        (r"i am your [a-z]+ ([a-z]+)", 0.9),  # For cases like "I am your daughter Emma"
        (r"i am ([a-z]+)", 0.6),  # Weaker pattern
        (r"i'm ([a-z]+)", 0.6)    # Weaker pattern
    ]
    
    matched_patterns = []  # For logging
    
    for pattern, confidence in name_patterns:
        match = re.search(pattern, text)
        if match:
            # For patterns with capture groups, use the last group as the name
            groups = match.groups()
            if groups:
                name_candidate = groups[-1].capitalize()  # Use the last captured group
                name_matches.append({
                    "value": name_candidate,
                    "confidence": confidence,
                    "pattern": pattern
                })
                matched_patterns.append(pattern)
                # Take the first strong match or continue to collect all matches
                if confidence >= 0.8 and name is None:
                    name = name_candidate
    
    # If no strong match found, take the first match regardless of confidence
    if name is None and name_matches:
        name = name_matches[0]["value"]
    
    # Calculate name confidence based on matches
    name_confidence = 0.0
    if name_matches:
        # Get highest confidence match for the name
        name_match_confidences = [m["confidence"] for m in name_matches if m["value"] == name]
        if name_match_confidences:
            name_confidence = max(name_match_confidences)
        else:
            # If name doesn't match any pattern exactly, use lowest confidence
            name_confidence = min([m["confidence"] for m in name_matches])
    
    # Extract candidate relationship using keywords with confidence weights
    relationship = None
    relationship_keywords = {
        "son": (["son"], 0.9),
        "daughter": (["daughter"], 0.9),
        "wife": (["wife"], 0.8),
        "husband": (["husband"], 0.8),
        "mother": (["mother", "mom", "mum"], 0.9),
        "father": (["father", "dad", "daddy"], 0.9),
        "sister": (["sister"], 0.8),
        "brother": (["brother"], 0.8),
        "friend": (["friend"], 0.7),
        "nurse": (["nurse"], 0.8),
        "doctor": (["doctor"], 0.8),
        "caregiver": (["caregiver"], 0.8),
        "neighbor": (["neighbor", "neighbour"], 0.7),
        "grandson": (["grandson"], 0.9),
        "granddaughter": (["granddaughter"], 0.9),
        "grandmother": (["grandmother", "grandma", "granny"], 0.9),
        "grandfather": (["grandfather", "grandpa"], 0.9)
    }
    
    matched_keywords = []  # For logging
    
    # First check for direct relationship statements
    for rel, (keywords, confidence) in relationship_keywords.items():
        for keyword in keywords:
            # Check for patterns like "I'm your [relationship]" or "I am your [relationship]"
            if re.search(r"i'?m your " + keyword, text) or re.search(r"i am your " + keyword, text):
                relationship_matches.append({
                    "value": rel,
                    "confidence": confidence,
                    "keyword": keyword
                })
                matched_keywords.append(keyword)
                if relationship is None:
                    relationship = rel
                break
            # Use word boundaries to match whole words only
            if re.search(r'\b' + keyword + r'\b', text):
                relationship_matches.append({
                    "value": rel,
                    "confidence": confidence * 0.8,  # Lower confidence for less specific matches
                    "keyword": keyword
                })
                matched_keywords.append(keyword)
                if relationship is None:
                    relationship = rel
                break
    
    # Calculate relationship confidence based on matches
    relationship_confidence = 0.0
    if relationship_matches:
        # Get highest confidence match for the relationship
        rel_match_confidences = [m["confidence"] for m in relationship_matches if m["value"] == relationship]
        if rel_match_confidences:
            relationship_confidence = max(rel_match_confidences)
        else:
            # If relationship doesn't match any pattern exactly, use lowest confidence
            relationship_confidence = min([m["confidence"] for m in relationship_matches])
    
    # Log extraction results
    logger.info(f"Identity extraction results - Transcript: '{transcript}'")
    logger.info(f"Matched name patterns: {matched_patterns}")
    logger.info(f"Matched relationship keywords: {matched_keywords}")
    logger.info(f"Extracted name: {name} (confidence: {name_confidence})")
    logger.info(f"Extracted relationship: {relationship} (confidence: {relationship_confidence})")
    
    return {
        "name": {
            "value": name,
            "confidence": round(name_confidence, 2)
        },
        "relationship": {
            "value": relationship,
            "confidence": round(relationship_confidence, 2)
        }
    }


# Simple in-memory cache for audio transcription
class AudioTranscriptionCache:
    def __init__(self, max_size: int = 100, ttl: int = 300):  # 5 minutes TTL
        self.cache: Dict[str, dict] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def _get_cache_key(self, audio_data: bytes) -> str:
        """Generate a short hash/fingerprint of the audio bytes."""
        return hashlib.md5(audio_data[:1000]).hexdigest()  # Use first 1000 bytes for efficiency
    
    def get(self, audio_data: bytes) -> str:
        """Get cached transcription if available and not expired."""
        key = self._get_cache_key(audio_data)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                logger.info(f"Cache hit for audio transcription")
                return entry["transcript"]
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def put(self, audio_data: bytes, transcript: str):
        """Store transcription in cache."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._get_cache_key(audio_data)
        self.cache[key] = {
            "transcript": transcript,
            "timestamp": time.time()
        }
        logger.info(f"Cached transcription for audio (key: {key[:8]}...)")

# Global cache instance
audio_cache = AudioTranscriptionCache()


async def transcribe_audio_with_whisper(audio_data: bytes) -> str:
    """
    Transcribe audio data using Whisper (locally) with caching.
    
    Args:
        audio_data: Audio data as bytes
        
    Returns:
        Transcribed text
    """
    try:
        import io
        import wave
        import numpy as np
        import whisper
        
        # Check cache first
        cached_transcript = audio_cache.get(audio_data)
        if cached_transcript is not None:
            logger.info("Using cached transcription result")
            return cached_transcript
        
        # Validate audio data
        if not audio_data:
            logger.warning("Empty audio data provided for transcription")
            return ""
        
        # Check audio duration (limit to 30 seconds for performance)
        # Assuming 16-bit PCM at 16kHz sample rate
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio_np) / 16000.0  # Duration in seconds
        
        if duration > 30.0:  # Limit to 30 seconds
            logger.warning(f"Audio too long ({duration:.2f}s), truncating to 30 seconds")
            # Truncate to 30 seconds of audio
            max_samples = int(30.0 * 16000)
            audio_np = audio_np[:max_samples]
        
        # Check if audio is silent or too short
        if len(audio_np) < 100:  # Arbitrary minimum length
            logger.info("Audio too short for meaningful transcription")
            return ""
            
        if np.max(np.abs(audio_np)) < 0.01:
            logger.info("Audio appears to be silent")
            return ""
        
        # Load Whisper model (using tiny model for speed)
        try:
            model = whisper.load_model("tiny")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return "Model loading failed"
        
        # Transcribe audio
        try:
            start_time = time.time()
            result = model.transcribe(audio_np, fp16=False)
            transcription_time = time.time() - start_time
            transcript = result["text"].strip()
            
            # Cache the result
            audio_cache.put(audio_data, transcript)
            
            # Log transcription performance
            logger.info(f"Transcribed audio ({duration:.2f}s) in {transcription_time:.2f}s: {transcript}")
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return "Transcription failed"
        
        return transcript
    except Exception as e:
        logger.error(f"Unexpected error in transcribe_audio_with_whisper: {e}")
        # Return a placeholder if transcription fails
        return "Transcription failed"


@app.post("/voice/transcribe_and_store", response_model=TranscribeAndStoreResponse)
async def transcribe_and_store_endpoint(request: TranscribeAndStoreRequest):
    """
    Transcribe audio and store the conversation.
    
    Args:
        request: Transcription and storage request
        
    Returns:
        Transcription result and storage status
    """
    try:
        # Validate input
        if not request.audio:
            raise HTTPException(status_code=400, detail="Audio data is required")
        
        if request.direction not in ["to_patient", "from_patient"]:
            raise HTTPException(status_code=400, detail="Direction must be 'to_patient' or 'from_patient'")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(request.audio)
        except Exception as e:
            logger.error(f"Error decoding base64 audio data: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        # Transcribe audio using Whisper
        transcript = await transcribe_audio_with_whisper(audio_data)
        
        stored = False
        person_id = request.person_id
        
        # If transcript is not empty and person_id is provided
        if transcript and request.person_id:
            # Insert a new document into conversations
            success = add_conversation(
                person_id=request.person_id,
                direction=request.direction,
                text=transcript,
                source="voice"
            )
            
            if success:
                # Push the same entry into that person's conversation_history array
                conversation_entry = {
                    "timestamp": datetime.utcnow(),
                    "direction": request.direction,
                    "text": transcript,
                    "source": "voice"
                }
                
                # Add to person's conversation history (keep last 20 entries)
                add_conversation_to_history(request.person_id, conversation_entry, max_history=20)
                stored = True
                logger.info(f"Successfully stored conversation for person {request.person_id}")
            else:
                logger.error(f"Failed to store conversation in database for person {request.person_id}")
        elif transcript and not request.person_id:
            logger.info("Transcript generated but no person_id provided, not storing conversation")
        elif not transcript:
            logger.info("No transcript generated from audio, nothing to store")
        
        return TranscribeAndStoreResponse(
            transcript=transcript,
            stored=stored,
            person_id=person_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in transcribe_and_store: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/voice/infer_identity_from_audio", response_model=InferIdentityResponse)
async def infer_identity_from_audio_endpoint(request: InferIdentityRequest):
    """
    Infer name and relationship from audio transcript with confidence scores.
    
    Args:
        request: Audio data for identity inference
        
    Returns:
        Transcript and extracted identity information with confidence
    """
    try:
        # Validate input
        if not request.audio:
            raise HTTPException(status_code=400, detail="Audio data is required")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(request.audio)
        except Exception as e:
            logger.error(f"Error decoding base64 audio data: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        # Transcribe audio using Whisper
        transcript = await transcribe_audio_with_whisper(audio_data)
        
        # Extract name and relationship with confidence using regex patterns
        extracted = extract_name_and_relationship_with_confidence(transcript)
        
        logger.info(f"Inferred identity - Name: {extracted['name']}, Relationship: {extracted['relationship']}")
        
        return InferIdentityResponse(
            transcript=transcript,
            extracted=extracted
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in infer_identity_from_audio: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class HandleUnknownSpeakerRequest(BaseModel):
    audio: str  # Base64 encoded audio

@app.post("/voice/handle_unknown_speaker")
async def handle_unknown_speaker(request: HandleUnknownSpeakerRequest):
    """
    Handle unknown speaker by inferring identity and creating temporary person.
    
    Args:
        request: HandleUnknownSpeakerRequest with audio data
        
    Returns:
        Created temporary person document
    """
    try:
        # Validate input
        if not request.audio:
            raise HTTPException(status_code=400, detail="Audio data is required")
        
        # Extract and decode audio data
        try:
            audio_data = base64.b64decode(request.audio)
        except Exception as e:
            logger.error(f"Error decoding base64 audio data: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        # Transcribe audio using Whisper
        transcript = await transcribe_audio_with_whisper(audio_data)
        
        # Extract name and relationship with confidence from transcript
        extracted = extract_name_and_relationship_with_confidence(transcript)
        
        # Create temporary person document
        from inference.database import create_temporary_person
        temp_person = create_temporary_person(
            name=extracted["name"]["value"] if extracted["name"]["value"] else "Unknown",
            relationship=extracted["relationship"]["value"] if extracted["relationship"]["value"] else "Unknown"
        )
        
        # Add confidence information to the temporary person
        temp_person["identity_confidence"] = {
            "name": extracted["name"],
            "relationship": extracted["relationship"]
        }
        
        # Store their conversation transcript
        if transcript:
            conversation_entry = {
                "timestamp": datetime.utcnow().isoformat(),  # Convert to string for JSON serialization
                "direction": "from_patient",  # Assuming unknown speaker is talking to patient
                "text": transcript,
                "source": "voice"
            }
            
            # Add to conversations collection
            success1 = add_conversation(
                person_id=temp_person["person_id"],
                direction="from_patient",
                text=transcript,
                source="voice"
            )
            
            # Add to person's conversation history (keep last 20 entries)
            success2 = add_conversation_to_history(temp_person["person_id"], conversation_entry, max_history=20)
            
            if not success1 or not success2:
                logger.warning(f"Partial failure storing conversation for temporary person {temp_person['person_id']}")
        
        logger.info(f"Created temporary person: {temp_person['name']} ({temp_person['person_id']}) with confidence name={extracted['name']['confidence']}, relationship={extracted['relationship']['confidence']}")
        return temp_person
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling unknown speaker: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/person/promote_temporary")
async def promote_temporary_person_endpoint(request: PromoteTemporaryPersonRequest):
    """
    Promote a temporary person to a permanent person.
    
    Args:
        request: Promotion request with person_id and optional updated fields
        
    Returns:
        Updated person document
    """
    try:
        from inference.database import promote_temporary_person, get_person_by_id
        
        # Promote the temporary person
        success = promote_temporary_person(
            person_id=request.person_id,
            name=request.name,
            relationship=request.relationship
        )
        
        if success:
            # Get the updated person document
            person = get_person_by_id(request.person_id)
            if person and '_id' in person:
                del person['_id']
            return person
        else:
            raise HTTPException(status_code=400, detail="Failed to promote temporary person")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error promoting temporary person: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def generate_short_summary(conversations: list) -> str:
    """
    Generate a short summary of conversations using lightweight, deterministic heuristics.
    
    Args:
        conversations: List of conversation entries
        
    Returns:
        Short summary string
    """
    if not conversations:
        return "No previous conversations found."
    
    # Simple concatenation with speaker identification
    summary_parts = []
    for conv in conversations[:3]:  # Limit to last 3 conversations for brevity
        # Make sure we have the required fields
        if "direction" in conv and "text" in conv:
            direction = "You said" if conv["direction"] == "from_patient" else "They said"
            summary_parts.append(f"{direction}: {conv['text']}")
    
    # Join with spaces and limit length for readability
    summary = " ".join(summary_parts)
    if len(summary) > 150:  # Limit summary length
        summary = summary[:150] + "..."
    
    return summary if summary else "No previous conversations found."


def extract_keywords(conversations: list) -> list:
    """
    Extract keywords from conversations using basic NLP techniques.
    
    Args:
        conversations: List of conversation entries
        
    Returns:
        List of extracted keywords
    """
    import re
    from collections import Counter
    
    if not conversations:
        return []
    
    # Combine all conversation text
    all_text = " ".join([conv.get("text", "") for conv in conversations])
    
    # Convert to lowercase
    text = all_text.lower()
    
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)  # Only words with 3+ characters
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
        'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old',
        'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'put', 'too', 'use', 'any', 'big', 'end',
        'far', 'got', 'let', 'lot', 'run', 'say', 'she', 'try', 'way', 'win', 'yes', 'yet', 'bit',
        'eat', 'hat', 'hit', 'law', 'lay', 'led', 'leg', 'let', 'lie', 'log', 'lot', 'low', 'map', 'net', 'new', 'nor', 'not', 'now', 'off',
        'old', 'one', 'our', 'out', 'own', 'pay', 'per', 'put', 'red', 'rid', 'run', 'say', 'see',
        'set', 'she', 'shy', 'sir', 'sit', 'six', 'son', 'sun', 'ten', 'the', 'tie', 'tip', 'too',
        'top', 'toy', 'try', 'two', 'use', 'war', 'way', 'wet', 'who', 'why', 'win', 'yes', 'yet'
    }
    
    # Filter out stop words and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get top 5 most common words
    top_words = [word for word, count in word_counts.most_common(5)]
    
    return top_words


def get_last_conversation_summary(person_id: str) -> dict:
    """
    Get a summary of the last conversation for a person with messages, summary, and keywords.
    
    Args:
        person_id: Person identifier
        
    Returns:
        Dictionary with messages, short_summary, and keywords
    """
    try:
        from inference.database import get_last_conversations
        conversations = get_last_conversations(person_id, 5)  # Get last 5 conversations
        
        if not conversations:
            return {
                "messages": [],
                "short_summary": "No previous conversations found.",
                "keywords": []
            }
        
        # Generate summary and keywords
        short_summary = generate_short_summary(conversations)
        keywords = extract_keywords(conversations)
        
        return {
            "messages": conversations,
            "short_summary": short_summary,
            "keywords": keywords
        }
    except Exception as e:
        logger.error(f"Error getting last conversation summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/voice/last_conversation/{person_id}", response_model=LastConversationResponse)
async def get_last_conversation_summary_endpoint(person_id: str):
    """
    Get a summary of the last conversation for a person with messages, summary, and keywords.
    
    Args:
        person_id: Person identifier
        
    Returns:
        Last conversation data with messages, short_summary, and keywords
    """
    try:
        result = get_last_conversation_summary(person_id)
        return LastConversationResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_last_conversation_summary_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/voice/log_conversation", response_model=LogConversationResponse)
async def log_conversation_endpoint(request: LogConversationRequest):
    """
    Log a conversation by transcribing audio and storing the full transcript.
    
    Args:
        request: LogConversationRequest with optional person_id, direction, and audio data
        
    Returns:
        LogConversationResponse with person_id, text, and stored status
    """
    try:
        # Validate input
        if not request.audio:
            raise HTTPException(status_code=400, detail="Audio data is required")
        
        # Validate and normalize direction
        if request.direction not in ["to_patient", "from_patient", "dialogue"]:
            raise HTTPException(status_code=400, detail="Direction must be 'to_patient', 'from_patient', or 'dialogue'")
        
        # As per user request, treat all conversations as "to_patient"
        normalized_direction = "to_patient"
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(request.audio)
        except Exception as e:
            logger.error(f"Error decoding base64 audio data: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        # Transcribe audio using Whisper
        transcript = await transcribe_audio_with_whisper(audio_data)
        
        stored = False
        person_id = request.person_id
        
        # If transcript is not empty
        if transcript:
            # Import required functions
            from inference.database import store_conversation, add_conversation_to_history, get_person_by_id
            
            # Store full conversation in conversations collection (always do this)
            # If person_id is provided, use it; otherwise use a special identifier for unknown persons
            conversation_person_id = person_id if person_id else "unknown_speaker"
            
            success = store_conversation(
                person_id=conversation_person_id,
                text=transcript,
                direction=normalized_direction,
                source="voice"
            )
            
            if success:
                stored = True
                logger.info(f"Successfully logged conversation {'for person ' + person_id if person_id else 'from unknown speaker'}")
                
                # If person_id is provided and exists, also update their conversation history
                if person_id:
                    person = get_person_by_id(person_id)
                    if person:
                        # Push the same entry into that person's conversation_history array
                        conversation_entry = {
                            "timestamp": datetime.utcnow(),
                            "direction": normalized_direction,
                            "text": transcript,
                            "source": "voice"
                        }
                        
                        # Add to person's conversation history (keep last 20 entries)
                        add_conversation_to_history(person_id, conversation_entry, max_history=20)
                        logger.info(f"Successfully updated conversation history for person {person_id}")
                    else:
                        logger.warning(f"Person {person_id} not found in database, only stored in conversations collection")
            else:
                logger.error(f"Failed to store conversation in database")
        elif not transcript:
            logger.info("No transcript generated from audio, nothing to store")
        
        return LogConversationResponse(
            person_id=person_id if person_id else "",
            text=transcript,
            stored=stored
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in log_conversation_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/voice/context/{person_id}", response_model=VoiceContextResponse)
async def get_voice_context_endpoint(person_id: str):
    """
    Get the recent conversation context for a person.
    
    Args:
        person_id: Person identifier
        
    Returns:
        VoiceContextResponse with person_id, conversations, and short_summary
    """
    try:
        # Import required functions
        from inference.database import get_recent_conversations
        
        # Get recent conversations (last 3 by default)
        conversations = get_recent_conversations(person_id, limit=3)
        
        if not conversations:
            return VoiceContextResponse(
                person_id=person_id,
                conversations=[],
                short_summary="No previous conversations found."
            )
        
        # Generate a simple short summary
        summary_parts = []
        for conv in conversations:
            # Make sure we have the required fields
            if "direction" in conv and "text" in conv:
                # As per user request, all conversations are treated as "to_patient"
                direction = "They said"  # Since all conversations are "to_patient"
                summary_parts.append(f"{direction}: {conv['text']}")
        
        # Join with spaces and limit length for readability
        short_summary = " ".join(summary_parts)
        if len(short_summary) > 200:  # Limit summary length
            short_summary = short_summary[:200] + "..."
        
        return VoiceContextResponse(
            person_id=person_id,
            conversations=conversations,
            short_summary=short_summary if short_summary else "No previous conversations found."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_voice_context_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

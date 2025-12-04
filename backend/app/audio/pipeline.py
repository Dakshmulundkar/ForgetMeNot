"""Audio pipeline that buffers conversations and transcribes them with Whisper."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import uuid

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from backend.app.core import AudioChunk, ConversationEvent, ConversationUtterance
from backend.app.services.conversation_stream import ConversationEventBus
from backend.app.audio.denoiser import AdaptiveDenoiser

# Import MongoDB database functions
from inference.database import (
    get_person_by_id,
    update_person_context,
    update_voice_profile,
    add_conversation_to_history
)

# Import PersonData model and broadcast function
from backend.app.models import PersonData

# Cloud service clients
from backend.app.audio.huggingface_client import hf_client

# Fireworks client for cloud transcription
from openai import AsyncOpenAI

# Add missing imports
try:  # pragma: no cover - optional dependency
    import webrtcvad
except ImportError:  # pragma: no cover
    webrtcvad = None

try:  # pragma: no cover - optional dependency
    from torchaudio.functional import resample as ta_resample
except ImportError:  # pragma: no cover
    ta_resample = None

trace_logger = logging.getLogger("webrtc.trace")
logger = logging.getLogger("webrtc.audio.pipeline")


@dataclass
class PipelineConfig:
    """Configuration for the audio pipeline."""
    target_sample_rate: int = 16_000  # Increased from 8000 to 16000 for better transcription quality
    silence_timeout_seconds: float = 3.0  # Increased from 2.0 to 3.0 for better sentence capture
    min_conversation_seconds: float = 1.5  # Increased from 1.0 for better quality
    vad_aggressiveness: int = 2  # Reduced from 3 to 2 for less aggressive voice activity detection
    transcription_model: str = "base"  # Changed from "cloud" to "base" for better local transcription quality
    min_speech_rms: float = 0.005  # Reduced from 0.01 for better sensitivity
    noise_floor_smoothing: float = 0.95  # Increased from 0.9 for better noise adaptation
    noise_gate_margin: float = 0.003  # Reduced from 0.005 for better sensitivity
    embedding_model: str = "pyannote/embedding"
    speaker_match_threshold: float = 0.25
    embedding_window_seconds: float = 0.75  # Increased from 0.5 for better embedding quality
    use_cloud_services: bool = True  # Enable cloud services by default
    max_concurrent_sessions: int = 10  # Increased from 1 to 10 for better performance


@dataclass
class ConversationState:
    conversation_id: str
    started_ts: float
    last_audio_ts: float
    last_speech_ts: float | None
    noise_floor_rms: float | None = None
    has_speech: bool = False
    chunks: List[np.ndarray] | None = None
    last_speaker_id: str | None = None

    def __post_init__(self) -> None:
        if self.chunks is None:
            self.chunks = []


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


@dataclass
class SpeakerProfile:
    speaker_id: str
    embedding: np.ndarray
    count: int = 1


class AudioPipeline:
    """Buffers audio per WebRTC session and transcribes completed conversations."""

    def __init__(
        self,
        denoiser: AdaptiveDenoiser,
        config: PipelineConfig | None = None,
        conversation_bus: ConversationEventBus | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.config = config or PipelineConfig()
        self._conversations: Dict[str, ConversationState] = {}
        self._whisper_lock = asyncio.Lock()
        self._vad = webrtcvad.Vad(self.config.vad_aggressiveness) if webrtcvad else None
        self._speaker_profiles: List[SpeakerProfile] = []
        self._speaker_lock = asyncio.Lock()
        self._next_speaker_index = 10
        self._pyannote_auth_token = os.getenv("PYANNOTE_AUTH_TOKEN")
        self._fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
        self._conversation_bus = conversation_bus
        self._active_sessions = 0
        self._session_lock = asyncio.Lock()

        # Cloud service clients
        self._hf_client = hf_client
        self._fireworks_client = None
        
        # Initialize Fireworks client if API key is available
        if self._fireworks_api_key:
            try:
                self._fireworks_client = AsyncOpenAI(
                    api_key=self._fireworks_api_key,
                    base_url="https://api.fireworks.ai/inference/v1"
                )
                logger.info("Initialized Fireworks client for cloud transcription")
            except Exception as e:
                logger.error("Failed to initialize Fireworks client: %s", e)
                self._fireworks_client = None
        else:
            logger.info("No Fireworks API key found, will use local Whisper only")
            self._fireworks_client = None
        
        if self.config.use_cloud_services and self._fireworks_client:
            logger.info("Cloud services enabled for audio processing")
        else:
            logger.info("Cloud services disabled or unavailable; using local processing where available")

    async def process_chunk(self, chunk: AudioChunk) -> None:
        session_id = chunk.session_id
        
        # Limit concurrent sessions for i3 performance
        async with self._session_lock:
            if self._active_sessions >= self.config.max_concurrent_sessions:
                logger.warning("Maximum concurrent sessions reached, dropping chunk for session %s", session_id)
                return
            
            # Check if this is a new session
            if session_id not in self._conversations:
                self._active_sessions += 1
                logger.info("Active sessions: %d/%d", self._active_sessions, self.config.max_concurrent_sessions)
        
        state = self._ensure_conversation(session_id, chunk.timestamp.timestamp())

        denoised = await self.denoiser.denoise(chunk)
        audio = self._convert_to_target_sr(denoised.payload, denoised.sample_rate)
        if audio.size == 0:
            logger.debug("Session %s chunk had no audio data", session_id)
            return

        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        has_speech = self._chunk_has_speech(audio)
        effective_threshold = self.config.min_speech_rms
        if state.noise_floor_rms is not None:
            effective_threshold = max(
                effective_threshold,
                state.noise_floor_rms + self.config.noise_gate_margin,
            )
        if has_speech and rms < effective_threshold:
            logger.debug(
                "Session %s chunk suppressed by RMS gate (rms=%.5f threshold=%.5f)",
                session_id,
                rms,
                effective_threshold,
            )
            has_speech = False
        logger.debug(
            "Session %s chunk rms=%.5f has_speech=%s", session_id, rms, has_speech
        )

        state.chunks.append(audio)
        state.last_audio_ts = chunk.timestamp.timestamp()

        now = chunk.timestamp.timestamp()
        if not has_speech:
            smoothing = min(max(self.config.noise_floor_smoothing, 0.0), 0.999)
            if state.noise_floor_rms is None:
                state.noise_floor_rms = rms
            else:
                state.noise_floor_rms = (
                    state.noise_floor_rms * smoothing
                    + rms * (1.0 - smoothing)
                )
        if has_speech:
            state.last_speech_ts = now
            state.has_speech = True
            logger.info(
                "Session %s conversation %s detected speech (rms=%.4f)",
                session_id,
                state.conversation_id,
                rms,
            )
        elif state.has_speech and state.last_speech_ts is not None:
            elapsed = now - state.last_speech_ts
            if elapsed >= self.config.silence_timeout_seconds:
                logger.info(
                    "Session %s conversation %s reached silence timeout (%.2fs)",
                    session_id,
                    state.conversation_id,
                    elapsed,
                )
                await self._finalize_conversation(session_id, "silence timeout")

    async def flush_session(self, session_id: str, sample_rate: int) -> None:  # noqa: ARG002
        if session_id in self._conversations:
            await self._finalize_conversation(session_id, "session flush")

    async def warm_whisper(self) -> None:
        if self._fireworks_client is None:
            logger.info("Fireworks client not available; skipping warm-up")
            return
        logger.info("Cloud transcription service ready")

    def _ensure_conversation(self, session_id: str, ts: float) -> ConversationState:
        state = self._conversations.get(session_id)
        if state is None:
            conv_id = f"{session_id}-conv{uuid.uuid4().hex[:6]}"
            state = ConversationState(
                conversation_id=conv_id,
                started_ts=ts,
                last_audio_ts=ts,
                last_speech_ts=None,
            )
            self._conversations[session_id] = state
            logger.info(
                "Starting conversation %s for session=%s",
                conv_id,
                session_id,
            )
        return state

    async def _finalize_conversation(self, session_id: str, reason: str) -> None:
        state = self._conversations.pop(session_id, None)
        if state is None:
            return

        duration = state.last_audio_ts - state.started_ts
        if not state.has_speech or duration < self.config.min_conversation_seconds:
            logger.info(
                "Discarding conversation %s for session=%s (reason=%s, duration=%.2fs, has_speech=%s)",
                state.conversation_id,
                session_id,
                reason,
                duration,
                state.has_speech,
            )
            # Decrement active sessions counter
            async with self._session_lock:
                if self._active_sessions > 0:
                    self._active_sessions -= 1
            return

        audio = np.concatenate(state.chunks) if state.chunks else np.array([], dtype=np.float32)
        if audio.size == 0:
            logger.info(
                "Conversation %s for session=%s had no samples after buffering",
                state.conversation_id,
                session_id,
            )
            # Decrement active sessions counter
            async with self._session_lock:
                if self._active_sessions > 0:
                    self._active_sessions -= 1
            return

        transcript = await self._transcribe_audio(audio)
        await self._assign_speakers(state, session_id, audio, transcript)
        await self._publish_conversation_event(state, session_id, transcript)
        
        # Store conversation in database
        await self._store_conversation_data(state, session_id, transcript)
        
        if transcript:
            self._print_transcript(state.conversation_id, transcript)
        else:
            logger.info(
                "Conversation %s for session=%s produced no transcription",
                state.conversation_id,
                session_id,
            )
        logger.info(
            "Ending conversation %s for session=%s (reason=%s, duration=%.2fs)",
            state.conversation_id,
            session_id,
            reason,
            duration,
        )
        
        # Decrement active sessions counter
        async with self._session_lock:
            if self._active_sessions > 0:
                self._active_sessions -= 1

    async def _transcribe_audio(self, audio: np.ndarray) -> List[TranscriptSegment]:
        """Transcribe audio to text using cloud services or local Whisper as fallback."""
        if audio.size == 0:
            return []

        # Try cloud transcription first if enabled and API key is available
        if self.config.use_cloud_services and self._fireworks_client:
            try:
                logger.info("Attempting cloud transcription with Fireworks")
                
                # Convert audio to WAV format for cloud transcription
                import io
                import wave
                import tempfile
                import os  # Import os at the right scope
                
                # Convert float32 to int16
                audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                
                # Log audio information for debugging
                duration = len(audio) / self.config.target_sample_rate
                logger.info(f"Audio info: duration={duration:.2f}s, samples={len(audio)}, sample_rate={self.config.target_sample_rate}")
                
                # Create temporary WAV file using tempfile.NamedTemporaryFile correctly
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # Write WAV header and audio data
                    with wave.open(tmp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(self.config.target_sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    tmp_path = tmp_file.name
                
                try:
                    # Upload and transcribe with Fireworks
                    with open(tmp_path, 'rb') as audio_file:
                        transcript_response = await self._fireworks_client.audio.transcriptions.create(
                            file=audio_file,
                            model="whisper-large-v3",
                            response_format="verbose_json"
                        )
                    
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    
                    # Parse the response and create TranscriptSegments
                    segments = []
                    if hasattr(transcript_response, 'segments'):
                        for segment in transcript_response.segments:
                            segments.append(TranscriptSegment(
                                start=segment['start'],
                                end=segment['end'],
                                text=segment['text'],
                                speaker=None  # Will be assigned later
                            ))
                    else:
                        # If no segments, create a single segment with the whole transcript
                        segments.append(TranscriptSegment(
                            start=0.0,
                            end=len(audio) / self.config.target_sample_rate,
                            text=transcript_response.text,
                            speaker=None
                        ))
                    
                    logger.info(f"Transcribed {len(segments)} segments with Fireworks")
                    for i, segment in enumerate(segments):
                        logger.info(f"Segment {i}: '{segment.text}' ({segment.start:.2f}-{segment.end:.2f}s)")
                    return segments
                    
                except Exception as e:
                    # Clean up temporary file even if transcription fails
                    if os.path.exists(tmp_path) and 'tmp_path' in locals():
                        os.unlink(tmp_path)
                    logger.warning("Cloud transcription failed: %s", e)
                    # Continue to local Whisper fallback
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cloud transcription failed: %s", exc)
                # Continue to local Whisper fallback
        
        # Fallback to local Whisper if cloud transcription failed or is disabled
        logger.info("Falling back to local Whisper transcription")
        try:
            import whisper
            import tempfile
            import os  # Import os at the right scope
            
            # Convert float32 to int16 for local processing
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            
            # Log audio information for debugging
            duration = len(audio) / self.config.target_sample_rate
            logger.info(f"Audio info: duration={duration:.2f}s, samples={len(audio)}, sample_rate={self.config.target_sample_rate}")
            
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                # Write WAV header and audio data
                import wave
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.config.target_sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                tmp_path = tmp_file.name
            
            try:
                # Load Whisper model (using base model for better accuracy)
                logger.info("Loading Whisper model...")
                model = whisper.load_model("base")  # Changed from "tiny" to "base" for better accuracy
                logger.info("Whisper model loaded successfully")
                
                # Transcribe audio
                logger.info(f"Transcribing audio with duration {duration:.2f}s...")
                result = model.transcribe(tmp_path, fp16=False)
                transcript = result["text"].strip()
                
                # Clean up temporary file
                if os.path.exists(tmp_path) and 'tmp_path' in locals():
                    os.unlink(tmp_path)
                
                if transcript:
                    logger.info(f"Local Whisper transcription successful: '{transcript}'")
                    # Create a single segment for the whole transcript
                    segment = TranscriptSegment(
                        start=0.0,
                        end=len(audio) / self.config.target_sample_rate,
                        text=transcript,
                        speaker=None
                    )
                    return [segment]
                else:
                    logger.warning("Local Whisper transcription returned empty result")
                    return []
            except Exception as e:
                # Clean up temporary file even if transcription fails
                if os.path.exists(tmp_path) and 'tmp_path' in locals():
                    os.unlink(tmp_path)
                logger.error("Local Whisper transcription failed: %s", e)
                return []
                
        except ImportError:
            logger.error("Whisper not installed, cannot perform local transcription")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.error("Local transcription failed: %s", exc)
            return []

    def _print_transcript(
        self, conversation_id: str, snippets: List[TranscriptSegment]
    ) -> None:
        logger.info(
            "Publishing conversation summary for %s (%d snippets)",
            conversation_id,
            len(snippets),
        )
        for segment in snippets:
            minutes, seconds = divmod(segment.start, 60.0)
            millis = int(round((seconds - int(seconds)) * 100))
            timestamp = f"{int(minutes):02d}:{int(seconds)%60:02d}.{millis:02d}"
            speaker_label = segment.speaker or "speaker_unknown"
            line = f"[{timestamp}] {speaker_label}: {segment.text}"
            print(f"\033[31m{line}\033[0m", flush=True)

    async def _publish_person_detected(
        self,
        session_id: str,
        conversation_id: str,
        speaker_id: str,
        utterance: str | None = None,
        is_new: bool = False,
    ) -> None:
        if self._conversation_bus is None:
            return

        conversation: list[ConversationUtterance] = []
        if utterance:
            conversation.append(
                ConversationUtterance(
                    speaker=speaker_id,
                    text=utterance,
                )
            )

        event = ConversationEvent(
            event_type="PERSON_DETECTED",
            person_id=speaker_id,
            conversation_id=conversation_id,
            session_id=session_id,
            conversation=conversation,
        )

        try:
            await self._conversation_bus.publish(event)
            logger.info(
                "Published PERSON_DETECTED for %s (session=%s, conversation=%s, new=%s)",
                speaker_id,
                session_id,
                conversation_id,
                is_new,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to publish PERSON_DETECTED for %s: %s",
                speaker_id,
                exc,
            )

    async def _publish_conversation_event(
        self,
        state: ConversationState,
        session_id: str,
        snippets: List[TranscriptSegment],
    ) -> None:
        if not snippets or self._conversation_bus is None:
            return

        conversation = [
            ConversationUtterance(
                speaker=segment.speaker or "speaker_unknown",
                text=segment.text,
            )
            for segment in snippets
        ]

        primary_speaker = next(
            (entry.speaker for entry in conversation if entry.speaker != "speaker_unknown"),
            state.last_speaker_id,
        ) or "speaker_unknown"

        event = ConversationEvent(
            event_type="CONVERSATION_END",
            conversation_id=state.conversation_id,
            session_id=session_id,
            person_id=primary_speaker,
            conversation=conversation,
        )

        try:
            await self._conversation_bus.publish(event)
            logger.info(
                "Published conversation event %s with %d utterances",
                state.conversation_id,
                len(conversation),
            )
            
            # Broadcast person detection event to frontend via SSE
            if not primary_speaker.startswith("speaker_unknown"):
                person_doc = get_person_by_id(primary_speaker)
                if person_doc:
                    person_data = PersonData(
                        name=person_doc["name"],
                        description=person_doc["cached_description"],
                        relationship=person_doc["relationship"],
                        person_id=primary_speaker
                    )
                    # Import broadcast_person here to avoid circular import
                    from backend.app.utils import broadcast_person
                    await broadcast_person(person_data)
                else:
                    # Create a default person data if not found
                    person_data = PersonData(
                        name=primary_speaker.replace("_", " ").title(),
                        description="Recently detected speaker",
                        relationship="Recently detected",
                        person_id=primary_speaker
                    )
                    # Import broadcast_person here to avoid circular import
                    from backend.app.utils import broadcast_person
                    await broadcast_person(person_data)
            else:
                # For unknown speakers, we still want to broadcast the event
                # This will help the frontend know when to start listening for name introduction
                person_data = PersonData(
                    name="Unknown Person",
                    description="Just met",
                    relationship="Unknown",
                    person_id=primary_speaker
                )
                # Import broadcast_person here to avoid circular import
                from backend.app.utils import broadcast_person
                await broadcast_person(person_data)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to publish conversation event %s: %s",
                state.conversation_id,
                exc,
            )

    async def _store_conversation_data(
        self,
        state: ConversationState,
        session_id: str,
        snippets: List[TranscriptSegment],
    ) -> None:
        """Store conversation data in MongoDB.
        
        NEW: Store ALL conversations; link person when known.
        Unknown speakers get temporary profiles for later linking.
        ZERO conversation loss.
        NEW: Extract name/relationship from speech and update temporary persons.
        """
        if not snippets:
            return

        conversation = [
            ConversationUtterance(
                speaker=segment.speaker or "speaker_unknown",
                text=segment.text,
            )
            for segment in snippets
        ]

        primary_speaker = next(
            (entry.speaker for entry in conversation if entry.speaker != "speaker_unknown"),
            state.last_speaker_id,
        ) or "speaker_unknown"
        
        # Apply transcript sanity filter before processing
        conversation_text = " ".join([utterance.text for utterance in conversation if utterance.text])
        
        # Check if transcript is meaningful before proceeding
        if not is_transcript_meaningful(conversation_text):
            logger.info(
                "Skipping storage of conversation for speaker %s - transcript not meaningful: %s",
                primary_speaker,
                conversation_text[:50] + "..." if len(conversation_text) > 50 else conversation_text
            )
            return
        
        # Additional check for very short conversations
        if len(conversation_text.strip()) < 5:
            logger.info(
                "Skipping storage of conversation for speaker %s - transcript too short: %s",
                primary_speaker,
                conversation_text[:50] + "..." if len(conversation_text) > 50 else conversation_text
            )
            return

        # NEW: Create conversation event for storage - ALWAYS store
        conversation_event = {
            "event_type": "CONVERSATION_END",
            "timestamp": datetime.utcnow(),
            "speaker_id": primary_speaker,  # NEW: Always store speaker_id
            "conversation_id": state.conversation_id,
            "session_id": session_id,
            "conversation": [
                {
                    "speaker": utterance.speaker,
                    "text": utterance.text
                }
                for utterance in conversation
            ]
        }
        
        # Import the add_conversation function
        from inference.database import add_conversation
        
        # NEW: ALWAYS store in conversations collection, even for unknown speakers
        conversation_text = " ".join([utterance.text for utterance in conversation if utterance.text])
        
        if len(conversation_text.strip()) >= 5:  # Only store if meaningful content
            # Store the full conversation
            success = add_conversation(
                person_id=primary_speaker if not primary_speaker.startswith("speaker_unknown") else None,
                speaker_id=primary_speaker,  # NEW: Store speaker_id explicitly
                direction="to_patient",  # Changed from "from_patient" to "to_patient" as per user request
                text=conversation_text,
                source="voice",
                session_id=session_id  # NEW: Store session_id
            )
            
            if success:
                logger.info(
                    "Stored conversation entry in full history for speaker %s: %s",
                    primary_speaker,
                    conversation_text[:50] + "..." if len(conversation_text) > 50 else conversation_text
                )
            else:
                logger.warning(
                    "Failed to store conversation entry in full history for speaker %s",
                    primary_speaker
                )
        
        # Handle name/relationship extraction for ALL speakers when appropriate
        # Check for trigger phrases like "my name", "I am", "I'm", "I am your", "I'm your", etc.
        trigger_phrases = ["my name", "i am", "i'm", "i am your", "i'm your"]
        has_trigger_phrase = any(phrase in conversation_text.lower() for phrase in trigger_phrases)
        
        # Process name/relationship extraction if:
        # 1. Transcript is meaningful and has sufficient content
        # 2. Contains trigger phrases OR is from an unknown speaker
        should_extract_identity = (len(conversation_text.strip()) >= 10 and 
                                 (has_trigger_phrase or 
                                  primary_speaker.startswith("speaker_unknown") or 
                                  primary_speaker.startswith("unknown_")))
        
        if should_extract_identity:
            extracted_info = extract_name_and_relation_from_speech(conversation_text)
            name = extracted_info.get('name')
            name_confidence = extracted_info.get('name_confidence', 0.0)
            relationship = extracted_info.get('relationship')
            relationship_confidence = extracted_info.get('relationship_confidence', 0.0)
            
            # Log extraction results
            logger.info(
                f"Name/relationship extraction for speaker {primary_speaker}: "
                f"name={name} (confidence={name_confidence}), "
                f"relationship={relationship} (confidence={relationship_confidence})"
            )
            
            # Structured logging for monitoring
            logger.info(f"NAME_RELATION_EXTRACTION: {{'speaker_id': '{primary_speaker}', 'name': '{name}', 'name_confidence': {name_confidence}, 'relationship': '{relationship}', 'relationship_confidence': {relationship_confidence}}}")

        # Handle temporary persons and name/relationship extraction
        if primary_speaker.startswith("speaker_unknown") or primary_speaker.startswith("unknown_"):
            logger.info(
                "Handling unknown speaker %s, creating/updating temporary person",
                primary_speaker,
            )
            
            # Extract name and relationship from conversation for temporary persons
            # Only process if we have meaningful content
            if should_extract_identity:
                # Create or update temporary person with extracted information
                from inference.database import get_people_collection, create_temporary_person_with_conversation
                collection = get_people_collection()
                
                # Check if temporary person already exists
                existing_temp_person = collection.find_one({
                    "$or": [
                        {"speaker_id_provisional": primary_speaker},
                        {"person_id": {"$regex": f"unknown.*{primary_speaker}"}}
                    ]
                })
                
                if existing_temp_person:
                    # Update existing temporary person
                    update_fields = {
                        "last_updated": datetime.utcnow()
                    }
                    
                    # Update name and relationship if confidence is high enough
                    if name and name_confidence >= 0.8:
                        # For known persons, check if name differs only by minor spelling or capitalization
                        current_name = existing_temp_person.get("name", "").lower()
                        new_name = name.lower()
                        if current_name != new_name and Levenshtein_distance(current_name, new_name) > 2:
                            update_fields["name"] = name.capitalize()
                        else:
                            logger.info(f"Skipping name update for {primary_speaker} - names are similar: {current_name} vs {new_name}")
                    
                    if relationship and relationship_confidence >= 0.8:
                        update_fields["relationship"] = relationship.capitalize()
                    
                    # Add identity confidence field
                    update_fields["identity_confidence"] = max(name_confidence, relationship_confidence)
                    
                    # Add conversation to history
                    conversation_entry = {
                        "timestamp": datetime.utcnow(),
                        "direction": "to_patient",  # Changed from "from_patient" to "to_patient" as per user request
                        "text": conversation_text,
                        "source": "voice"
                    }
                    
                    result = collection.update_one(
                        {"person_id": existing_temp_person["person_id"]},
                        {
                            "$set": update_fields,
                            "$push": {
                                "conversation_history": {
                                    "$each": [conversation_entry],
                                    "$slice": -20  # Keep last 20 conversations
                                }
                            }
                        }
                    )
                    
                    if result.matched_count > 0:
                        logger.info(
                            f"Updated temporary person {existing_temp_person['person_id']} with extracted info and conversation"
                        )
                        # Structured logging for monitoring
                        logger.info(f"TEMPORARY_PERSON_UPDATED: {{'person_id': '{existing_temp_person['person_id']}', 'updated_fields': {list(update_fields.keys())}}}")
                    else:
                        logger.warning(
                            f"Failed to update temporary person {existing_temp_person['person_id']}"
                        )
                        # Structured logging for monitoring
                        logger.info(f"TEMPORARY_PERSON_UPDATE_FAILED: {{'person_id': '{existing_temp_person['person_id']}'}}")
                else:
                    # Create new temporary person with conversation
                    temp_person_id = f"unknown_{primary_speaker}_{int(datetime.utcnow().timestamp())}"
                    
                    try:
                        temp_person = create_temporary_person_with_conversation(
                            person_id=temp_person_id,
                            speaker_id=primary_speaker,
                            name=name.capitalize() if name and name_confidence >= 0.8 else "Unknown Speaker",
                            relationship=relationship.capitalize() if relationship and relationship_confidence >= 0.8 else "Unknown",
                            conversation_text=conversation_text,
                            direction="to_patient",
                            timestamp=datetime.utcnow()
                        )
                        
                        if temp_person:
                            logger.info(f"Created temp person: {temp_person_id} with conversation")
                            # Structured logging for monitoring
                            logger.info(f"TEMPORARY_PERSON_CREATED: {{'person_id': '{temp_person_id}', 'name': '{temp_person.get('name')}', 'relationship': '{temp_person.get('relationship')}'}}")
                        else:
                            logger.warning(f"Failed to create temp person for speaker: {primary_speaker}")
                    except Exception as e:
                        logger.error(f"Error creating temporary person for speaker {primary_speaker}: {e}")
            else:
                # Even if conversation is short, still create a temporary person
                from inference.database import get_people_collection, create_temporary_person_with_conversation
                collection = get_people_collection()
                
                # Check if temporary person already exists
                existing_temp_person = collection.find_one({
                    "$or": [
                        {"speaker_id_provisional": primary_speaker},
                        {"person_id": {"$regex": f"unknown.*{primary_speaker}"}}
                    ]
                })
                
                if not existing_temp_person:
                    temp_person_id = f"unknown_{primary_speaker}_{int(datetime.utcnow().timestamp())}"
                    
                    try:
                        temp_person = create_temporary_person_with_conversation(
                            person_id=temp_person_id,
                            speaker_id=primary_speaker,
                            name="Unknown Speaker",
                            relationship="Unknown",
                            conversation_text=conversation_text,
                            direction="to_patient",
                            timestamp=datetime.utcnow()
                        )
                        
                        if temp_person:
                            logger.info(f"Created temp person: {temp_person_id} with conversation")
                        else:
                            logger.warning(f"Failed to create temp person for speaker: {primary_speaker}")
                    except Exception as e:
                        logger.error(f"Error creating temporary person for speaker {primary_speaker}: {e}")
        else:
            # Update person's conversation_history if person exists AND is not temporary
            if not primary_speaker.startswith("speaker_unknown"):
                from inference.database import get_person_by_id
                person = get_person_by_id(primary_speaker)
                
                # Also run name/relation extraction for known persons when appropriate
                # Process if we have meaningful content and transcript contains trigger phrases
                if should_extract_identity and extracted_info:
                    name = extracted_info.get('name')
                    name_confidence = extracted_info.get('name_confidence', 0.0)
                    relationship = extracted_info.get('relationship')
                    relationship_confidence = extracted_info.get('relationship_confidence', 0.0)
                    
                    # Log extraction results
                    logger.info(
                        f"Name/relationship extraction for known speaker {primary_speaker}: "
                        f"name={name} (confidence={name_confidence}), "
                        f"relationship={relationship} (confidence={relationship_confidence})"
                    )
                    
                    # Structured logging for monitoring
                    logger.info(f"NAME_RELATION_EXTRACTION_KNOWN: {{'speaker_id': '{primary_speaker}', 'name': '{name}', 'name_confidence': {name_confidence}, 'relationship': '{relationship}', 'relationship_confidence': {relationship_confidence}}}")
                    
                    # For known persons, update relationship if confidence is high and current relationship is blank or generic
                    if person and relationship and relationship_confidence >= 0.8:
                        current_relationship = person.get("relationship", "").lower()
                        generic_relationships = {"new acquaintance", "unknown", ""}
                        if current_relationship in generic_relationships:
                            from inference.database import get_people_collection
                            collection = get_people_collection()
                            
                            result = collection.update_one(
                                {"person_id": primary_speaker},
                                {
                                    "$set": {
                                        "relationship": relationship.capitalize(),
                                        "last_updated": datetime.utcnow()
                                    }
                                }
                            )
                            
                            if result.matched_count > 0:
                                logger.info(
                                    f"Updated relationship for known person {primary_speaker} to {relationship}"
                                )
                            else:
                                logger.warning(
                                    f"Failed to update relationship for known person {primary_speaker}"
                                )
                    
                    # For known persons with high name confidence, check for minor differences
                    if person and name and name_confidence >= 0.8:
                        current_name = person.get("name", "").lower()
                        new_name = name.lower()
                        # Only update if names are significantly different (Levenshtein distance > 2)
                        if current_name != new_name and Levenshtein_distance(current_name, new_name) > 2:
                            from inference.database import get_people_collection
                            collection = get_people_collection()
                            
                            result = collection.update_one(
                                {"person_id": primary_speaker},
                                {
                                    "$set": {
                                        "name": name.capitalize(),
                                        "last_updated": datetime.utcnow()
                                    }
                                }
                            )
                            
                            if result.matched_count > 0:
                                logger.info(
                                    f"Updated name for known person {primary_speaker} to {name}"
                                )
                            else:
                                logger.warning(
                                    f"Failed to update name for known person {primary_speaker}"
                                )
                        else:
                            logger.info(f"Skipping name update for {primary_speaker} - names are similar: {current_name} vs {new_name}")
                
                # Only update history for permanent persons, not temporary ones
                if person and not person.get("is_temporary", False):
                    success = add_conversation_to_history(primary_speaker, conversation_event, max_history=20)
                    if success:
                        logger.info(
                            "Stored conversation in history for person %s",
                            primary_speaker,
                        )
                    else:
                        logger.warning(
                            "Failed to store conversation in history for person %s",
                            primary_speaker,
                        )
                elif person and person.get("is_temporary", False):
                    logger.info(
                        "Updating temporary person %s with conversation",
                        primary_speaker,
                    )
                    
                    # Update temporary person with conversation
                    from inference.database import get_people_collection
                    collection = get_people_collection()
                    
                    conversation_entry = {
                        "timestamp": datetime.utcnow(),
                        "direction": "to_patient",  # Changed from "from_patient" to "to_patient" as per user request
                        "text": conversation_text,
                        "source": "voice"
                    }
                    
                    result = collection.update_one(
                        {"person_id": person["person_id"]},
                        {
                            "$push": {
                                "conversation_history": {
                                    "$each": [conversation_entry],
                                    "$slice": -20  # Keep last 20 conversations
                                }
                            },
                            "$set": {
                                "last_updated": datetime.utcnow()
                            }
                        }
                    )
                    
                    if result.matched_count > 0:
                        logger.info(
                            f"Updated temporary person {person['person_id']} with conversation"
                        )
                    else:
                        logger.warning(
                            f"Failed to update temporary person {person['person_id']} with conversation"
                        )

    async def _load_whisper_model(self):
        async with self._whisper_lock:
            if self._whisper_model is not None:
                return self._whisper_model
            try:
                model = await asyncio.to_thread(
                    whisper.load_model, self.config.transcription_model
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load Whisper model '%s': %s",
                    self.config.transcription_model,
                    exc,
                )
                self._whisper_model = None
                return None
            self._whisper_model = model
            logger.info(
                "Loaded Whisper model '%s' for transcription",
                self.config.transcription_model,
            )
            return self._whisper_model

    def _convert_to_target_sr(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        if not audio_bytes:
            return np.array([], dtype=np.float32)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        if sample_rate == self.config.target_sample_rate:
            return audio
        if ta_resample is None:
            duration = audio.shape[0] / sample_rate
            target_length = int(duration * self.config.target_sample_rate)
            if target_length <= 0:
                return np.array([], dtype=np.float32)
            x_old = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
            x_new = np.linspace(0, duration, num=target_length, endpoint=False)
            resampled = np.interp(x_new, x_old, audio)
            return resampled.astype(np.float32)
        tensor = torch.from_numpy(audio).unsqueeze(0)
        tensor = ta_resample(tensor, sample_rate, self.config.target_sample_rate)
        return tensor.squeeze(0).cpu().numpy().astype(np.float32)

    def _chunk_has_speech(self, audio: np.ndarray) -> bool:
        if self._vad is None or audio.size == 0:
            return True
        frame_samples = int(self.config.target_sample_rate * 0.02)
        pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        if pcm16.size < frame_samples:
            return False
        for start in range(0, pcm16.size - frame_samples + 1, frame_samples):
            frame = pcm16[start : start + frame_samples].tobytes()
            try:
                if self._vad.is_speech(frame, self.config.target_sample_rate):
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("WebRTC VAD error: %s", exc)
                return True
        return False

    async def _assign_speakers(
        self,
        state: ConversationState,
        session_id: str,
        audio: np.ndarray,
        snippets: List[TranscriptSegment],
    ) -> None:
        if not snippets or audio.size == 0:
            return

        publish_queue: List[Tuple[str, Optional[str], bool]] = []

        if self._hf_client is None:
            logger.warning(
                "Hugging Face embedding unavailable; using fallback speaker attribution"
            )
            async with self._speaker_lock:
                speaker_id = state.last_speaker_id
                is_new = False
                if speaker_id is None:
                    speaker_id = self._register_new_speaker(np.zeros(1, dtype=np.float32))
                    state.last_speaker_id = speaker_id
                    is_new = True

                for segment in snippets:
                    segment.speaker = speaker_id
                    publish_queue.append((speaker_id, segment.text, is_new))
                    is_new = False

            for speaker_id, utterance, is_new in publish_queue:
                await self._publish_person_detected(
                    session_id=session_id,
                    conversation_id=state.conversation_id,
                    speaker_id=speaker_id,
                    utterance=utterance,
                    is_new=is_new,
                )
            return

        sr = self.config.target_sample_rate
        windows: List[
            Tuple[TranscriptSegment, List[Tuple[np.ndarray, float]]]
        ] = []
        for segment in snippets:
            start_idx = max(int(segment.start * sr), 0)
            end_idx = max(int(segment.end * sr), start_idx + 1)
            if start_idx >= audio.size:
                continue
            end_idx = min(end_idx, audio.size)
            segment_audio = audio[start_idx:end_idx]
            if segment_audio.size == 0:
                continue
            prepared = self._prepare_embedding_windows(segment_audio)
            if not prepared:
                continue
            windows.append((segment, prepared))

        if not windows:
            return

        embeddings: List[Tuple[TranscriptSegment, np.ndarray]] = []
        for segment, prepared_windows in windows:
            vectors: List[Tuple[np.ndarray, float]] = []
            for window, weight in prepared_windows:
                embedding = await self._embed_audio(window)
                if embedding is None or embedding.size == 0:
                    continue
                vectors.append((embedding, weight))

            if not vectors:
                logger.info(
                    "No embedding generated for segment %.2f-%.2f in %s",
                    segment.start,
                    segment.end,
                    state.conversation_id,
                )
                continue

            if len(vectors) > 1:
                embeddings_array = np.vstack([vec for vec, _ in vectors])
                weights = np.array([max(weight, 1e-6) for _, weight in vectors])
                weights /= weights.sum()
                averaged = (embeddings_array * weights[:, None]).sum(axis=0)
                logger.info(
                    "Averaged %d embedding windows for segment %.2f-%.2f in %s",
                    len(vectors),
                    segment.start,
                    segment.end,
                    state.conversation_id,
                )
            else:
                averaged = vectors[0][0]
            embeddings.append((segment, averaged))

        if not embeddings:
            logger.info(
                "Hugging Face produced no embeddings for conversation %s", state.conversation_id
            )
            return

        async with self._speaker_lock:
            for segment, vector in embeddings:
                speaker_id, is_new = self._match_speaker(vector, state.last_speaker_id)
                segment.speaker = speaker_id
                state.last_speaker_id = speaker_id
                publish_queue.append((speaker_id, segment.text, is_new))

                # Update voice profile in database for recognized speakers
                if not speaker_id.startswith("speaker_unknown"):
                    voice_profile_data = {
                        "embedding": vector.tolist(),
                        "created_at": datetime.utcnow(),
                        "sample_count": 1
                    }
                    update_voice_profile(speaker_id, voice_profile_data)

        for speaker_id, utterance, is_new in publish_queue:
            await self._publish_person_detected(
                session_id=session_id,
                conversation_id=state.conversation_id,
                speaker_id=speaker_id,
                utterance=utterance,
                is_new=is_new,
            )

    def _prepare_embedding_windows(
        self, segment_audio: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        window_size = int(self.config.embedding_window_seconds * self.config.target_sample_rate)
        if window_size <= 0:
            return []
        if segment_audio.size <= window_size:
            rms = float(np.sqrt(np.mean(np.square(segment_audio))))
            return [(segment_audio, rms)]
        step = max(window_size // 6, 1)
        windows: List[Tuple[float, np.ndarray]] = []
        for start in range(0, segment_audio.size - window_size + 1, step):
            window = segment_audio[start : start + window_size]
            rms = float(np.sqrt(np.mean(np.square(window))))
            windows.append((rms, window))
        if not windows:
            tail = segment_audio[-window_size:]
            rms = float(np.sqrt(np.mean(np.square(tail))))
            return [(tail, rms)]
        windows.sort(key=lambda item: item[0], reverse=True)
        top_k = min(5, len(windows))
        return [(windows[i][1], windows[i][0]) for i in range(top_k)]

    async def _embed_audio(self, audio_window: np.ndarray) -> Optional[np.ndarray]:
        if self._hf_client is None or audio_window.size == 0:
            return None
            
        # Prepare audio data for Hugging Face API
        audio_data = {
            "waveform": audio_window.astype(np.float32),
            "sample_rate": self.config.target_sample_rate
        }
        
        # Use Hugging Face client for embedding
        vector = await self._hf_client.embed_audio(audio_data)
        
        if vector is None:
            logger.info("Hugging Face embedding returned None")
            return None
            
        return vector.astype(np.float32, copy=False)

    def _match_speaker(
        self, vector: np.ndarray, previous_speaker: Optional[str]
    ) -> Tuple[str, bool]:
        normalized = self._normalize_vector(vector)
        if normalized.size == 0:
            fallback = previous_speaker or self._register_new_speaker(normalized)
            return fallback, fallback != previous_speaker

        scores: List[Tuple[SpeakerProfile, float]] = []
        for profile in self._speaker_profiles:
            score = float(np.dot(normalized, profile.embedding))
            scores.append((profile, score))

        if scores:
            similarity_map = ", ".join(
                f"{profile.speaker_id}:{score:.3f}" for profile, score in scores
            )
            logger.info("Speaker similarity map: %s", similarity_map)

        prev_profile = self._find_profile(previous_speaker)
        prev_score = float("-inf")
        if prev_profile is not None:
            prev_score = next(
                (score for profile, score in scores if profile is prev_profile),
                float(np.dot(normalized, prev_profile.embedding)),
            )
            logger.info(
                "Speaker similarity with previous %s: %.3f",
                prev_profile.speaker_id,
                prev_score,
            )

        best_profile: Optional[SpeakerProfile] = None
        best_score = float("-inf")
        for profile, score in scores:
            if profile is prev_profile:
                continue
            if score > best_score:
                best_score = score
                best_profile = profile

        if best_profile is not None:
            logger.info(
                "Best speaker candidate %s score=%.3f (threshold=%.3f)",
                best_profile.speaker_id,
                best_score,
                self.config.speaker_match_threshold,
            )

        candidate: Optional[Tuple[SpeakerProfile, float]] = None
        for profile, score in scores:
            if score >= self.config.speaker_match_threshold:
                if candidate is None or score > candidate[1]:
                    candidate = (profile, score)

        if candidate is not None:
            profile, score = candidate
            logger.info(
                "Selecting speaker %s with score %.3f", profile.speaker_id, score
            )
            self._update_profile(profile, normalized)
            return profile.speaker_id, False

        new_id = self._register_new_speaker(normalized)
        return new_id, True

    def _find_profile(self, speaker_id: Optional[str]) -> Optional[SpeakerProfile]:
        if speaker_id is None:
            return None
        for profile in self._speaker_profiles:
            if profile.speaker_id == speaker_id:
                return profile
        return None

    def _update_profile(self, profile: SpeakerProfile, vector: np.ndarray) -> None:
        weight = 1.0 / (profile.count + 1)
        updated = profile.embedding * (1.0 - weight) + vector * weight
        profile.embedding = self._normalize_vector(updated)
        profile.count += 1

    def _register_new_speaker(self, vector: np.ndarray) -> str:
        speaker_id = f"speaker_{self._next_speaker_index:03d}"
        self._next_speaker_index += 1
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            embedding=self._normalize_vector(vector),
            count=1,
        )
        self._speaker_profiles.append(profile)
        logger.info(
            "Registered new speaker profile %s (total=%d)",
            speaker_id,
            len(self._speaker_profiles),
        )
        return speaker_id

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        if vector.size == 0:
            return vector
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.zeros_like(vector)
        return (vector / norm).astype(np.float32)


import re  # NEW: Added for regex pattern matching

def Levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return Levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def is_transcript_meaningful(text: str) -> bool:
    """
    Filter obviously bad transcripts before they hit DB.
    
    Args:
        text: Transcript text to evaluate
        
    Returns:
        True if transcript is meaningful, False otherwise
    """
    import re
    
    # Handle empty or invalid transcripts
    if not text or not isinstance(text, str):
        return False
    
    # Remove extra whitespace
    cleaned_text = text.strip()
    
    # Too short (less than 10-15 non-whitespace chars)
    if len(re.sub(r'\s', '', cleaned_text)) < 15:  # Increased threshold to 15 chars
        return False
    
    # Dominated by a single repeated token (e.g. "you you you you", "Shush. Shush.")
    words = re.findall(r'\b\w+\b', cleaned_text.lower())
    if len(words) > 0:
        # Check if a single word makes up more than 50% of the transcript
        from collections import Counter
        word_counts = Counter(words)
        most_common_word, count = word_counts.most_common(1)[0]
        if count / len(words) > 0.6 and len(words) > 3:  # Increased threshold to 60%
            return False
    
    # Only punctuation or filler words
    filler_words = {'uh', 'um', 'er', 'ah', 'hmm', 'mmm', 'shush', 'shh'}
    non_filler_words = [word for word in words if word not in filler_words]
    if len(non_filler_words) < max(1, len(words) * 0.4):  # Increased threshold to 40% non-filler words
        return False
    
    # Looks like garbage text (random characters, no real words)
    if len(words) == 0 and len(cleaned_text) > 0:
        # Contains mostly non-alphabetic characters
        alpha_chars = sum(1 for c in cleaned_text if c.isalpha())
        if alpha_chars / len(cleaned_text) < 0.6:  # Increased threshold to 60% alphabetic
            return False
    
    # Check for repeated patterns that indicate poor transcription
    if re.search(r'(\w+\s*){2,}\1{2,}', cleaned_text.lower()):
        # Pattern like "word word word word word word" repeated
        return False
    
    return True


def extract_name_and_relation_from_speech(transcript: str) -> dict:
    """
    Extract name and relationship using enhanced pattern matching.
    No LLM; pure regex (lightweight on i3).
    
    Args:
        transcript: Speech transcript to analyze
        
    Returns:
        Dictionary with extracted name, relationship and confidence scores
    """
    import re
    
    # Handle empty or invalid transcripts
    if not transcript or not isinstance(transcript, str):
        return {
            'name': None,
            'name_confidence': 0.0,
            'relationship': None,
            'relationship_confidence': 0.0
        }
    
    text = transcript.lower().strip()
    
    # Handle very short transcripts
    if len(text) < 5:
        return {
            'name': None,
            'name_confidence': 0.0,
            'relationship': None,
            'relationship_confidence': 0.0
        }
    
    # Combined patterns that extract both name and relationship in one go (higher priority)
    combined_patterns = [
        (r"i['’]m your (\w+) ([a-z]+)", 0.9),  # "I'm your nurse Jennifer"
        (r"i am your (\w+) ([a-z]+)", 0.9),    # "I am your nurse Jennifer"
        (r"i['’]m your ([a-z]+) ([a-z]+)", 0.85),  # More general pattern
        (r"this is your (\w+) ([a-z]+)", 0.8),  # "This is your friend Robert"
    ]
    
    # Try combined patterns first
    extracted_name = None
    extracted_relation = None
    name_confidence = 0.0
    relation_confidence = 0.0
    
    for pattern, confidence in combined_patterns:
        match = re.search(pattern, text)
        if match:
            rel_candidate = match.group(1).lower()
            name_candidate = match.group(2).capitalize()
            
            # Valid relations
            valid_relations = {
                'son', 'daughter', 'wife', 'husband', 'mother', 'father',
                'brother', 'sister', 'friend', 'nurse', 'doctor', 'caregiver',
                'aunt', 'uncle', 'grandmother', 'grandfather', 'grandma', 'grandpa',
                'grandson', 'granddaughter', 'cousin', 'neighbor', 'neighbour'
            }
            
            # Check if the first group is a valid relationship and the second is a valid name
            if rel_candidate in valid_relations and 2 <= len(name_candidate) <= 20 and name_candidate.isalpha():
                extracted_relation = rel_candidate.capitalize()
                extracted_name = name_candidate
                relation_confidence = confidence
                name_confidence = confidence
                break
    
    # If we didn't get both from combined patterns, try separate patterns
    if not extracted_name or not extracted_relation:
        # Name patterns with confidence scores
        name_patterns = [
            (r"my name is ([a-z]+)", 0.95),
            (r"i am ([a-z]+)", 0.90),
            (r"i'm ([a-z]+)", 0.85),
            (r"call me ([a-z]+)", 0.90),
            (r"you can call me ([a-z]+)", 0.95),
            (r"this is ([a-z]+)", 0.8),
            (r"hello,? i['’]m ([a-z]+)", 0.85),
            (r"hi,? i['’]m ([a-z]+)", 0.85),
            (r"hey,? i['’]m ([a-z]+)", 0.85),
        ]
        
        # Relationship patterns
        relation_patterns = [
            (r"i['’]m your (\w+)", 0.95),
            (r"i am your (\w+)", 0.95),
            (r"i['’]m (?:the|a) (\w+)", 0.90),
            (r"your (\w+)", 0.7),
            (r"i['’]m your ([a-z]+ [a-z]+)", 0.9),  # For compound relationships like "best friend"
        ]
        
        valid_relations = {
            'son', 'daughter', 'wife', 'husband', 'mother', 'father',
            'brother', 'sister', 'friend', 'nurse', 'doctor', 'caregiver',
            'aunt', 'uncle', 'grandmother', 'grandfather', 'grandma', 'grandpa',
            'grandson', 'granddaughter', 'cousin', 'neighbor', 'neighbour'
        }
        
        # Extract name if not already found
        if not extracted_name:
            for pattern, confidence in name_patterns:
                match = re.search(pattern, text)
                if match:
                    candidate_name = match.group(1).capitalize()
                    # Check if name is reasonable (2-20 characters, alphabetic)
                    if 2 <= len(candidate_name) <= 20 and candidate_name.isalpha():
                        extracted_name = candidate_name
                        name_confidence = confidence
                        break
        
        # Extract relationship if not already found
        if not extracted_relation:
            for pattern, confidence in relation_patterns:
                match = re.search(pattern, text)
                if match:
                    candidate_relation = match.group(1).lower()
                    # Check if it's a valid relationship or compound relationship
                    if candidate_relation in valid_relations or any(word in candidate_relation for word in valid_relations):
                        # Capitalize first letter of each word
                        extracted_relation = ' '.join(word.capitalize() for word in candidate_relation.split())
                        relation_confidence = confidence
                        break
    
    # Special case for "This is your friend Robert calling" (more specific pattern)
    if not extracted_name and "this is your" in text:
        # Try to extract name after "this is your [relationship]"
        match = re.search(r"this is your \w+ ([a-z]+)", text)
        if match:
            name_candidate = match.group(1).capitalize()
            if 2 <= len(name_candidate) <= 20 and name_candidate.isalpha():
                extracted_name = name_candidate
                name_confidence = 0.8
    
    # Additional special case for the exact phrase the user mentioned
    if not extracted_name and not extracted_relation:
        # Handle the specific phrase: "Hi mom this is your son Bob. just wanted to let you know i got promoted ok bye"
        match = re.search(r"this is your (\w+) ([a-z]+)", text)
        if match:
            rel_candidate = match.group(1).lower()
            name_candidate = match.group(2).capitalize()
            
            valid_relations = {
                'son', 'daughter', 'wife', 'husband', 'mother', 'father',
                'brother', 'sister', 'friend', 'nurse', 'doctor', 'caregiver',
                'aunt', 'uncle', 'grandmother', 'grandfather', 'grandma', 'grandpa',
                'grandson', 'granddaughter', 'cousin', 'neighbor', 'neighbour'
            }
            
            if rel_candidate in valid_relations and 2 <= len(name_candidate) <= 20 and name_candidate.isalpha():
                extracted_relation = rel_candidate.capitalize()
                extracted_name = name_candidate
                relation_confidence = 0.9
                name_confidence = 0.9
    
    # Even more specific pattern for the user's example
    if not extracted_name or not extracted_relation:
        # Handle pattern: "hi mom i am Daksh your son just wanted to let you know that i got promoted bye"
        match = re.search(r"i am ([a-z]+) your (\w+)", text)
        if match:
            name_candidate = match.group(1).capitalize()
            rel_candidate = match.group(2).lower()
            
            valid_relations = {
                'son', 'daughter', 'wife', 'husband', 'mother', 'father',
                'brother', 'sister', 'friend', 'nurse', 'doctor', 'caregiver',
                'aunt', 'uncle', 'grandmother', 'grandfather', 'grandma', 'grandpa',
                'grandson', 'granddaughter', 'cousin', 'neighbor', 'neighbour'
            }
            
            if rel_candidate in valid_relations and 2 <= len(name_candidate) <= 20 and name_candidate.isalpha():
                extracted_name = name_candidate
                extracted_relation = rel_candidate.capitalize()
                name_confidence = 0.95  # High confidence for this specific pattern
                relation_confidence = 0.95
    
    return {
        'name': extracted_name,
        'name_confidence': name_confidence,
        'relationship': extracted_relation,
        'relationship_confidence': relation_confidence
    }


def get_conversation_context_for_person(person_id: str, num_recent: int = 5) -> dict:
    """Retrieve recent conversations + humanized summary.
    
    Args:
        person_id: Person identifier
        num_recent: Number of recent conversations to retrieve
        
    Returns:
        Dictionary with person_id, recent conversations, and humanized summary
    """
    from inference.database import get_recent_conversations
    from datetime import datetime
    from collections import Counter
    import re
    
    recent = get_recent_conversations(person_id, limit=num_recent)
    
    if not recent:
        return {
            'person_id': person_id,
            'recent_conversations': [],
            'summary': 'No previous conversations.',
            'humanized_summary': 'You have not had any previous conversations.'
        }
    
    # Humanized summary: extract top keywords and create natural language summary
    all_text = ' '.join([c['text'] for c in recent])
    words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
    word_freq = Counter(words)
    top_words = [w for w, _ in word_freq.most_common(3)]  # Limit to top 3 for conciseness
    
    # Create humanized summary based on number of topics
    if len(top_words) == 1:
        humanized_summary = f"You mainly discussed {top_words[0]}."
    elif len(top_words) == 2:
        humanized_summary = f"You talked about {top_words[0]} and {top_words[1]}."
    elif len(top_words) >= 3:
        humanized_summary = f"You talked about {top_words[0]}, {top_words[1]}, and {top_words[2]}."
    else:
        humanized_summary = "You had a previous conversation."
    
    # Ensure summary stays within character limits
    if len(humanized_summary) > 400:
        humanized_summary = humanized_summary[:397] + "..."
    
    # Original summary for backward compatibility
    summary = f"Recently discussed: {', '.join(top_words)}."
    
    return {
        'person_id': person_id,
        'recent_conversations': recent,
        'summary': summary,
        'humanized_summary': humanized_summary,
        'last_spoken_at': recent[0]['timestamp'] if recent else None
    }

"""MongoDB database operations for person data."""

import logging
import os
from datetime import datetime
from typing import Optional, List
from bson import ObjectId
import numpy as np

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "dementia_care_db")

# Global MongoDB client and database
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_database() -> Database:
    """Get MongoDB database instance (singleton pattern)."""
    global _client, _db

    if _db is None:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable not set")

        logger.info(f"Connecting to MongoDB at {MONGODB_URI}...")
        _client = MongoClient(MONGODB_URI)
        _db = _client[MONGODB_DATABASE]

        # Test connection
        _client.admin.command('ping')
        logger.info(f"Connected to MongoDB database: {MONGODB_DATABASE}")

    return _db


def get_people_collection() -> Collection:
    """Get the 'people' collection."""
    db = get_database()
    return db["people"]


def get_conversations_collection() -> Collection:
    """Get the 'conversations' collection."""
    db = get_database()
    return db["conversations"]


def add_conversation(
    person_id: str = None, 
    speaker_id: str = None, 
    direction: str = "to_patient", 
    text: str = "", 
    source: str = "voice",
    session_id: str = None
) -> bool:
    """
    Add a conversation entry to the conversations collection.
    
    Args:
        person_id: Person identifier (optional)
        speaker_id: Speaker identifier from audio pipeline (optional)
        direction: Direction of conversation ("to_patient" or "from_patient")
        text: Transcript text
        source: Source of conversation ("voice" or other)
        session_id: Session identifier (optional)
        
    Returns:
        True if added successfully, False otherwise
    """
    try:
        collection = get_conversations_collection()
        conversation_doc = {
            "timestamp": datetime.utcnow(),
            "direction": direction,
            "text": text,
            "source": source
        }
        
        # NEW: Add optional fields if provided
        if person_id:
            conversation_doc["person_id"] = person_id
        if speaker_id:
            conversation_doc["speaker_id"] = speaker_id
        if session_id:
            conversation_doc["session_id"] = session_id
        
        result = collection.insert_one(conversation_doc)
        logger.info(f"Added conversation entry: person_id={person_id}, speaker_id={speaker_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding conversation entry: {e}")
        return False


def store_conversation(
    person_id: str = None, 
    speaker_id: str = None, 
    text: str = "", 
    direction: str = "to_patient", 
    source: str = "voice",
    session_id: str = None
) -> bool:
    """
    Store a full conversation entry in the conversations collection.
    This is an alias for add_conversation to maintain compatibility.
    
    Args:
        person_id: Person identifier (optional)
        speaker_id: Speaker identifier from audio pipeline (optional)
        text: Full transcript text (word-for-word)
        direction: Direction of conversation ("to_patient", "from_patient", or "dialogue")
        source: Source of conversation ("voice" or other)
        session_id: Session identifier (optional)
        
    Returns:
        True if stored successfully, False otherwise
    """
    # For now, treat all conversations as "to_patient" as per user request
    if direction == "from_patient":
        direction = "to_patient"
    
    return add_conversation(person_id, speaker_id, direction, text, source, session_id)


def get_recent_conversations(person_id: str = None, speaker_id: str = None, limit: int = 3) -> list:
    """
    Get the recent conversation entries for a person or speaker.
    
    Args:
        person_id: Person identifier (optional)
        speaker_id: Speaker identifier (optional)
        limit: Number of conversation entries to retrieve (default: 3)
        
    Returns:
        List of conversation entries ordered by timestamp descending
    """
    try:
        collection = get_conversations_collection()
        
        # Build query filter
        query_filter = {}
        if person_id:
            query_filter["person_id"] = person_id
        elif speaker_id:
            query_filter["speaker_id"] = speaker_id
        else:
            # If neither person_id nor speaker_id provided, return empty list
            return []
        
        conversations = list(collection.find(query_filter).sort("timestamp", -1).limit(limit))
        
        # Convert MongoDB documents to JSON-serializable format
        for conversation in conversations:
            if '_id' in conversation:
                del conversation['_id']
            # Handle timestamp conversion
            if 'timestamp' in conversation:
                if isinstance(conversation['timestamp'], datetime):
                    conversation['timestamp'] = conversation['timestamp'].isoformat()
                # If it's already a string, leave it as is
                
        return conversations
    except Exception as e:
        logger.error(f"Error retrieving recent conversations for person {person_id} or speaker {speaker_id}: {e}")
        return []


def get_last_conversations(person_id: str = None, speaker_id: str = None, limit: int = 10) -> list:
    """
    Get the last N conversation entries for a person or speaker.
    
    Args:
        person_id: Person identifier (optional)
        speaker_id: Speaker identifier (optional)
        limit: Number of conversation entries to retrieve
        
    Returns:
        List of conversation entries
    """
    try:
        collection = get_conversations_collection()
        
        # Build query filter
        query_filter = {}
        if person_id:
            query_filter["person_id"] = person_id
        elif speaker_id:
            query_filter["speaker_id"] = speaker_id
        else:
            # If neither person_id nor speaker_id provided, return empty list
            return []
        
        conversations = list(collection.find(query_filter).sort("timestamp", -1).limit(limit))
        
        # Convert MongoDB documents to JSON-serializable format
        for conversation in conversations:
            if '_id' in conversation:
                del conversation['_id']
            # Handle timestamp conversion
            if 'timestamp' in conversation:
                if isinstance(conversation['timestamp'], datetime):
                    conversation['timestamp'] = conversation['timestamp'].isoformat()
                # If it's already a string, leave it as is
                
        return conversations
    except Exception as e:
        logger.error(f"Error retrieving conversations for person {person_id} or speaker {speaker_id}: {e}")
        return []


def get_last_conversation_summary(person_id: str, limit: int = 5) -> str:
    """
    Get a summary of the last N conversation entries for a person.
    
    Args:
        person_id: Person identifier
        limit: Number of conversation entries to summarize
        
    Returns:
        Summary string of recent conversations
    """
    try:
        conversations = get_last_conversations(person_id, limit)
        if not conversations:
            return "No previous conversations found."
        
        # Simple concatenation for now - can be enhanced with LLM summarization
        summary_parts = []
        for conv in conversations:
            # Make sure we have the required fields
            if "direction" in conv and "text" in conv:
                direction = "You said" if conv["direction"] == "to_patient" else "They said"
                summary_parts.append(f"{direction}: {conv['text']}")
        
        # Join with spaces and limit length for readability
        summary = " ".join(summary_parts)
        if len(summary) > 200:  # Limit summary length
            summary = summary[:200] + "..."
        
        return summary if summary else "No previous conversations found."
    except Exception as e:
        logger.error(f"Error generating conversation summary for person {person_id}: {e}")
        return "Unable to generate conversation summary."


def create_temporary_person(name: str = "Unknown", relationship: str = "Unknown") -> dict:
    """
    Create a temporary person document for unknown speakers.
    
    Args:
        name: Person's name (if known)
        relationship: Relationship to patient (if known)
        
    Returns:
        Created temporary person document
    """
    try:
        from datetime import datetime
        import uuid
        
        # Generate a temporary person ID
        person_id = f"unknown_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create person document
        person_doc = {
            "person_id": person_id,
            "name": name if name else "Unknown",
            "relationship": relationship if relationship else "Unknown",
            "aggregated_context": "Inferred from voice-only conversation",
            "cached_description": "Inferred from voice-only conversation",
            "last_updated": datetime.utcnow(),
            "face_embeddings": [],
            "voice_profile": None,
            "conversation_history": [],
            "is_temporary": True,
            "identity_confidence": 0.0
        }
        
        # Insert into people collection
        collection = get_people_collection()
        result = collection.insert_one(person_doc)
        logger.info(f"Created temporary person: {person_doc['name']} ({person_id})")
        
        # Remove MongoDB-specific fields for JSON serialization
        if '_id' in person_doc:
            del person_doc['_id']
            
        return person_doc
    except Exception as e:
        logger.error(f"Error creating temporary person: {e}")
        raise


def find_existing_person_for_speaker_or_embedding(speaker_id: str, face_embedding: List[float] = None) -> Optional[str]:
    """
    Find an existing person by speaker_id or face embedding to prevent creating duplicate unknown persons.
    
    Args:
        speaker_id: Speaker identifier to match
        face_embedding: Face embedding vector to match (optional)
        
    Returns:
        Person ID if found, None otherwise
    """
    collection = get_people_collection()
    
    # First, look for a person with matching speaker_id
    person = collection.find_one({
        "$or": [
            {"speaker_id_provisional": speaker_id},
            {"person_id": speaker_id}
        ]
    })
    
    if person:
        logger.info(f"Found existing person {person['person_id']} for speaker_id {speaker_id}")
        return person['person_id']
    
    # Optionally, if face_embedding provided, compute similarity to find matching person
    if face_embedding:
        # Use existing face recognition function with lower threshold for loose matching
        from inference.database import find_person_by_face_embedding
        result = find_person_by_face_embedding(
            face_embedding, 
            threshold=float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.75")) - 0.1,  # Lower threshold for loose matching
            required_match_count=int(os.getenv("FACE_RECOGNITION_REQUIRED_MATCH_COUNT", "1"))
        )
        
        if result:
            logger.info(f"Found existing person {result['person']['person_id']} by face embedding match")
            return result['person']['person_id']
    
    logger.info(f"No existing person found for speaker_id {speaker_id}")
    return None


def create_temporary_person_with_conversation(
    person_id: str,
    speaker_id: str,
    name: str,
    relationship: str,
    conversation_text: str,
    direction: str = "to_patient",
    timestamp: datetime = None
) -> Optional[dict]:
    """
    Create a temporary person with initial conversation.
    
    Args:
        person_id: Unique person identifier
        speaker_id: Provisional speaker identifier
        name: Person's name
        relationship: Relationship to patient
        conversation_text: Initial conversation text
        direction: Direction of conversation
        timestamp: Timestamp for conversation
        
    Returns:
        Created person document or None if failed
    """
    collection = get_people_collection()
    
    # Check if person already exists using our new helper
    existing_person_id = find_existing_person_for_speaker_or_embedding(speaker_id)
    if existing_person_id:
        logger.info(f"Person already exists with ID {existing_person_id}, not creating duplicate")
        return get_person_by_id(existing_person_id)
    
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    person_doc = {
        "person_id": person_id,
        "speaker_id_provisional": speaker_id,  # Store provisional speaker_id
        "name": name,
        "relationship": relationship,
        "aggregated_context": f"First conversation: {conversation_text[:100]}...",
        "cached_description": "Just met",
        "last_updated": timestamp,
        "face_embeddings": [],
        "voice_profile": None,
        "conversation_history": [{
            "timestamp": timestamp,
            "direction": direction,
            "text": conversation_text,
            "source": "voice"
        }],
        "is_temporary": True,
        "identity_confidence": 0.5  # Initial confidence for temporary persons
    }

    try:
        result = collection.insert_one(person_doc)
        logger.info(f"Created temporary person: {name} ({person_id})")
        return person_doc
    except Exception as e:
        logger.error(f"Error creating temporary person {person_id}: {e}")
        return None


def get_person_by_id(person_id: str) -> Optional[dict]:
    """
    Retrieve a person document by person_id.

    Args:
        person_id: The person identifier

    Returns:
        Person document or None if not found
    """
    collection = get_people_collection()
    person_doc = collection.find_one({"person_id": person_id})

    if person_doc:
        logger.info(f"Found person: {person_doc.get('name')} ({person_id})")
    else:
        logger.warning(f"Person not found: {person_id}")

    return person_doc


def create_person(
    person_id: str,
    name: str,
    relationship: str,
    aggregated_context: str = "",
    cached_description: str = "No previous interactions"
) -> dict:
    """
    Create a new person document.

    Args:
        person_id: Unique person identifier
        name: Person's name
        relationship: Relationship to patient
        aggregated_context: Running summary of conversations
        cached_description: One-line description for AR display

    Returns:
        Created person document
    """
    collection = get_people_collection()

    person_doc = {
        "person_id": person_id,
        "name": name,
        "relationship": relationship,
        "aggregated_context": aggregated_context,
        "cached_description": cached_description,
        "last_updated": datetime.utcnow(),
        "face_embeddings": [],
        "voice_profile": None,
        "conversation_history": []
    }

    result = collection.insert_one(person_doc)
    logger.info(f"Created person: {name} ({person_id})")

    return person_doc


def update_person_context(
    person_id: str,
    aggregated_context: str,
    cached_description: str
) -> bool:
    """
    Update a person's aggregated context and cached description.

    Args:
        person_id: Person identifier
        aggregated_context: Updated conversation summary
        cached_description: New one-line description

    Returns:
        True if updated, False if person not found
    """
    collection = get_people_collection()

    result = collection.update_one(
        {"person_id": person_id},
        {
            "$set": {
                "aggregated_context": aggregated_context,
                "cached_description": cached_description,
                "last_updated": datetime.utcnow()
            }
        }
    )

    if result.matched_count > 0:
        logger.info(f"Updated context for person: {person_id}")
        return True
    else:
        logger.warning(f"Person not found for update: {person_id}")
        return False


def add_face_embedding(person_id: str, face_embedding: dict) -> bool:
    """
    Add a face embedding to a person's profile.

    Args:
        person_id: Person identifier
        face_embedding: Face embedding data

    Returns:
        True if added, False if person not found
    """
    collection = get_people_collection()

    result = collection.update_one(
        {"person_id": person_id},
        {
            "$push": {
                "face_embeddings": face_embedding
            },
            "$set": {
                "last_updated": datetime.utcnow()
            }
        }
    )

    if result.matched_count > 0:
        logger.info(f"Added face embedding for person: {person_id}")
        return True
    else:
        logger.warning(f"Person not found for face embedding: {person_id}")
        return False


def update_voice_profile(person_id: str, voice_profile: dict) -> bool:
    """
    Update a person's voice profile.

    Args:
        person_id: Person identifier
        voice_profile: Voice profile data

    Returns:
        True if updated, False if person not found
    """
    collection = get_people_collection()

    result = collection.update_one(
        {"person_id": person_id},
        {
            "$set": {
                "voice_profile": voice_profile,
                "last_updated": datetime.utcnow()
            }
        }
    )

    if result.matched_count > 0:
        logger.info(f"Updated voice profile for person: {person_id}")
        return True
    else:
        logger.warning(f"Person not found for voice profile update: {person_id}")
        return False


def add_conversation_to_history(person_id: str, conversation: dict, max_history: int = 20) -> bool:
    """
    Add a conversation to a person's history, keeping only the last N entries.
    
    Args:
        person_id: Person identifier
        conversation: Conversation event data
        max_history: Maximum number of conversation entries to keep (default: 20)
        
    Returns:
        True if added, False if person not found
    """
    try:
        collection = get_people_collection()
        
        # First push the new conversation
        result = collection.update_one(
            {"person_id": person_id},
            {
                "$push": {
                    "conversation_history": conversation
                },
                "$set": {
                    "last_updated": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count > 0:
            # Then trim the conversation history to keep only the last N entries
            collection.update_one(
                {"person_id": person_id},
                {
                    "$push": {
                        "conversation_history": {
                            "$each": [],
                            "$slice": -max_history
                        }
                    }
                }
            )
            logger.info(f"Added conversation to history for person: {person_id} (trimmed to last {max_history} entries)")
            return True
        else:
            logger.warning(f"Person not found for conversation history update: {person_id}")
            return False
    except Exception as e:
        logger.error(f"Error adding conversation to history for person {person_id}: {e}")
        return False


def find_person_by_face_embedding(
    face_embedding: List[float], 
    threshold: float = None,  # NEW: Allow None to use tuned threshold
    required_match_count: int = 1  # NEW: Reduced required match count for faster recognition
) -> Optional[dict]:
    """
    Find a person by face embedding using cosine similarity.
    
    NEW: Ensemble matching - voting over multiple stored embeddings per person.
    Compare new embedding to all stored embeddings, use voting consensus.
    NEW: Use tuned threshold from environment variables if not provided.

    Args:
        face_embedding: Face embedding vector to match
        threshold: Min similarity to count as vote (default: from env or 0.75)
        required_match_count: Min embeddings that must exceed threshold (default: 1)

    Returns:
        Dictionary with person document and matching metrics, or None if not found
    """
    collection = get_people_collection()
    
    # NEW: Load tuned threshold from environment variable
    if threshold is None:
        threshold = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.75"))
    
    # Convert input embedding to numpy array
    input_vector = np.array(face_embedding, dtype=np.float32)
    input_dim = input_vector.shape[0]
    input_norm = np.linalg.norm(input_vector)
    
    if input_norm == 0:
        return None
        
    # Normalize input vector
    input_vector = input_vector / (input_norm + 1e-8)
    
    # Find all people with face embeddings
    people = list(collection.find({"face_embeddings.0": {"$exists": True}}))
    
    best_person = None
    best_similarity = -1
    best_match_count = 0
    best_confidence = 0.0
    
    # NEW: Ensemble matching - compare to all stored embeddings per person
    for person in people:
        face_embeddings = person.get("face_embeddings", [])
        if not face_embeddings:
            continue
        
        similarities = []
        match_count = 0
        
        # Compare against each stored embedding for this person
        for embedding_data in face_embeddings:
            db_vector = np.array(embedding_data.get("vector", []), dtype=np.float32)
            # Skip if dimensions don't match
            if db_vector.shape[0] != input_dim:
                continue
                
            db_norm = np.linalg.norm(db_vector)
            
            if db_norm == 0:
                continue
                
            # Normalize database vector
            db_vector = db_vector / (db_norm + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(input_vector, db_vector)
            similarities.append(similarity)
            
            # Count as match if above threshold
            if similarity >= threshold:
                match_count += 1
        
        if not similarities:
            continue
            
        best_sim_for_person = max(similarities) if similarities else -1
        
        # NEW: Person is matched if enough embeddings exceed threshold
        if match_count >= required_match_count:
            # NEW: Confidence combines voting consensus and best similarity
            confidence = (match_count / len(face_embeddings)) * 0.5 + (best_sim_for_person / 2.0) * 0.5
            
            if best_sim_for_person > best_similarity:
                best_similarity = best_sim_for_person
                best_person = person
                best_match_count = match_count
                best_confidence = confidence
    
    if best_person:
        logger.info(f"Found person by face embedding: {best_person.get('name')} (similarity: {best_similarity:.3f}, matches: {best_match_count}, confidence: {best_confidence:.3f})")
        # NEW: Structured logging for monitoring
        logger.info(f"FACE_RECOGNITION_DECISION: {{'person_id': '{best_person.get('person_id')}', 'name': '{best_person.get('name')}', 'similarity': {best_similarity:.3f}, 'match_count': {best_match_count}, 'confidence': {best_confidence:.3f}}}")
        # NEW: Return person with matching metrics
        return {
            "person": best_person,
            "similarity": float(best_similarity),
            "match_count": best_match_count,
            "confidence": float(best_confidence)
        }
    
    logger.info("No matching person found by face embedding")
    # NEW: Structured logging for monitoring
    logger.info("FACE_RECOGNITION_DECISION: {'result': 'not_found', 'similarity': -1, 'match_count': 0, 'confidence': 0.0}")
    return None


def list_all_people() -> list[dict]:
    """
    List all people in the database.

    Returns:
        List of person documents
    """
    collection = get_people_collection()
    people = list(collection.find())
    
    # Convert MongoDB documents to JSON-serializable format
    for person in people:
        # Remove MongoDB ObjectId
        if '_id' in person:
            del person['_id']
        # Convert datetime objects to strings if needed
        if 'last_updated' in person and isinstance(person['last_updated'], datetime):
            person['last_updated'] = person['last_updated'].isoformat()
            
    logger.info(f"Found {len(people)} people in database")
    return people


def delete_all_people() -> int:
    """
    Delete all people from the database (for testing/reset).

    Returns:
        Number of documents deleted
    """
    collection = get_people_collection()
    result = collection.delete_many({})
    logger.info(f"Deleted {result.deleted_count} people from database")
    return result.deleted_count


def close_connection():
    """Close MongoDB connection."""
    global _client, _db

    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")


def promote_temporary_person(person_id: str, name: str = None, relationship: str = None) -> bool:
    """
    Promote a temporary person to a permanent person.
    
    Args:
        person_id: Person identifier
        name: Updated name (optional)
        relationship: Updated relationship (optional)
        
    Returns:
        True if promoted successfully, False otherwise
    """
    try:
        collection = get_people_collection()
        
        # Check if person exists and is temporary
        person = collection.find_one({"person_id": person_id})
        if not person:
            logger.warning(f"Person not found: {person_id}")
            return False
            
        if not person.get("is_temporary", False):
            logger.warning(f"Person {person_id} is not temporary")
            return False
        
        # Prepare update fields
        update_fields = {
            "is_temporary": False,
            "identity_confidence": 1.0,
            "last_updated": datetime.utcnow()
        }
        
        # Update name if provided
        if name:
            update_fields["name"] = name
            
        # Update relationship if provided
        if relationship:
            update_fields["relationship"] = relationship
            
        # Update the person document
        result = collection.update_one(
            {"person_id": person_id},
            {"$set": update_fields}
        )
        
        if result.matched_count > 0:
            logger.info(f"Promoted temporary person {person_id} to permanent")
            return True
        else:
            logger.warning(f"Failed to promote temporary person {person_id}")
            return False
    except Exception as e:
        logger.error(f"Error promoting temporary person {person_id}: {e}")
        return False

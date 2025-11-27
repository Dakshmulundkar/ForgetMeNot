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


def add_conversation(person_id: str, direction: str, text: str, source: str = "voice") -> bool:
    """
    Add a conversation entry to the conversations collection.
    
    Args:
        person_id: Person identifier
        direction: Direction of conversation ("to_patient" or "from_patient")
        text: Transcript text
        source: Source of conversation ("voice" or other)
        
    Returns:
        True if added successfully, False otherwise
    """
    try:
        collection = get_conversations_collection()
        conversation_doc = {
            "person_id": person_id,
            "timestamp": datetime.utcnow(),
            "direction": direction,
            "text": text,
            "source": source
        }
        
        result = collection.insert_one(conversation_doc)
        logger.info(f"Added conversation entry for person: {person_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding conversation entry: {e}")
        return False


def store_conversation(person_id: str, text: str, direction: str, source: str = "voice") -> bool:
    """
    Store a full conversation entry in the conversations collection.
    This is an alias for add_conversation to maintain compatibility.
    
    Args:
        person_id: Person identifier
        text: Full transcript text (word-for-word)
        direction: Direction of conversation ("to_patient", "from_patient", or "dialogue")
        source: Source of conversation ("voice" or other)
        
    Returns:
        True if stored successfully, False otherwise
    """
    # For now, treat all conversations as "to_patient" as per user request
    if direction == "from_patient":
        direction = "to_patient"
    
    return add_conversation(person_id, direction, text, source)


def get_recent_conversations(person_id: str, limit: int = 3) -> list:
    """
    Get the recent conversation entries for a person.
    
    Args:
        person_id: Person identifier
        limit: Number of conversation entries to retrieve (default: 3)
        
    Returns:
        List of conversation entries ordered by timestamp descending
    """
    try:
        collection = get_conversations_collection()
        conversations = list(collection.find({"person_id": person_id}).sort("timestamp", -1).limit(limit))
        
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
        logger.error(f"Error retrieving recent conversations for person {person_id}: {e}")
        return []


def get_last_conversations(person_id: str, limit: int = 10) -> list:
    """
    Get the last N conversation entries for a person.
    
    Args:
        person_id: Person identifier
        limit: Number of conversation entries to retrieve
        
    Returns:
        List of conversation entries
    """
    try:
        collection = get_conversations_collection()
        conversations = list(collection.find({"person_id": person_id}).sort("timestamp", -1).limit(limit))
        
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
        logger.error(f"Error retrieving conversations for person {person_id}: {e}")
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
                direction = "You said" if conv["direction"] == "from_patient" else "They said"
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


def find_person_by_face_embedding(face_embedding: List[float], threshold: float = 0.8) -> Optional[dict]:
    """
    Find a person by face embedding using cosine similarity.

    Args:
        face_embedding: Face embedding vector to match
        threshold: Similarity threshold (0-1)

    Returns:
        Person document or None if not found
    """
    collection = get_people_collection()
    
    # Convert input embedding to numpy array
    input_vector = np.array(face_embedding)
    input_dim = input_vector.shape[0]
    input_norm = np.linalg.norm(input_vector)
    
    if input_norm == 0:
        return None
        
    input_vector = input_vector / input_norm
    
    # Find all people with face embeddings
    people = collection.find({"face_embeddings.0": {"$exists": True}})
    
    best_match = None
    best_similarity = -1
    
    for person in people:
        # Check each face embedding for this person
        for embedding_data in person.get("face_embeddings", []):
            db_vector = np.array(embedding_data.get("vector", []))
            # Skip if dimensions don't match
            if db_vector.shape[0] != input_dim:
                continue
                
            db_norm = np.linalg.norm(db_vector)
            
            if db_norm == 0:
                continue
                
            db_vector = db_vector / db_norm
            
            # Calculate cosine similarity
            similarity = np.dot(input_vector, db_vector)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = person
    
    if best_match:
        logger.info(f"Found person by face embedding: {best_match.get('name')} (similarity: {best_similarity:.3f})")
        return best_match
    
    logger.info("No matching person found by face embedding")
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

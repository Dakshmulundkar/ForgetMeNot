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


def add_conversation_to_history(person_id: str, conversation: dict) -> bool:
    """
    Add a conversation to a person's history.

    Args:
        person_id: Person identifier
        conversation: Conversation event data

    Returns:
        True if added, False if person not found
    """
    collection = get_people_collection()

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
        logger.info(f"Added conversation to history for person: {person_id}")
        return True
    else:
        logger.warning(f"Person not found for conversation history update: {person_id}")
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
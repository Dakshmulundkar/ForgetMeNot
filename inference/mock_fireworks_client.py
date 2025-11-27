"""Mock Fireworks.ai client for conversation processing - for testing without API key."""

import logging
from typing import List
from models import ConversationUtterance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_fireworks_client")


async def aggregate_conversation_context(
    person_name: str,
    current_context: str,
    new_conversation: List[ConversationUtterance]
) -> str:
    """
    Mock Model #1: Context Aggregation
    Takes the current aggregated context and a new conversation, returns updated summary.
    
    Args:
        person_name: The person's name for context
        current_context: Previous aggregated conversation summary
        new_conversation: New conversation to incorporate
        
    Returns:
        Updated aggregated context
    """
    logger.info(f"Mock context aggregation for {person_name}")
    
    # Format the conversation for the summary
    conversation_text = "\n".join([
        f"{utt.speaker}: {utt.text}"
        for utt in new_conversation
    ])
    
    # Simple mock implementation - just append the new conversation
    updated_context = f"{current_context} Recently discussed: {conversation_text[:100]}..."
    
    return updated_context


async def generate_ar_description(
    person_name: str,
    relationship: str,
    aggregated_context: str
) -> str:
    """
    Mock Model #2: AR Description Generation
    Takes person info and aggregated context, returns one-line AR display description.
    
    Args:
        person_name: The person's name
        relationship: Their relationship to the patient
        aggregated_context: Full conversation history summary
        
    Returns:
        One-line description for AR display
    """
    logger.info(f"Mock description generation for {person_name}")
    
    # Simple mock implementation
    description = f"Recently interacted with {person_name}"
    
    return description


async def infer_new_person_details(conversation: List[ConversationUtterance]) -> dict:
    """
    Mock Model #3: New Person Inference
    Analyzes a first-time conversation to infer person details.
    
    Args:
        conversation: The first conversation with this person
        
    Returns:
        Dictionary with keys: name, relationship, aggregated_context, cached_description
    """
    logger.info("Mock new person inference")
    
    # Simple mock implementation
    return {
        "name": "New Person",
        "relationship": "New acquaintance",
        "aggregated_context": "First conversation with this person.",
        "cached_description": "Just met"
    }


async def test_fireworks_connection() -> bool:
    """
    Mock test that Fireworks.ai API is accessible.
    
    Returns:
        True if connection successful, False otherwise
    """
    logger.info("Mock Fireworks.ai connection test")
    return True
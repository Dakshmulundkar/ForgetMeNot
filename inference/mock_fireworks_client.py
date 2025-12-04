"""Mock Fireworks.ai client for conversation processing - for testing without API key."""

import logging
from typing import List
from models import ConversationUtterance
import re
from collections import Counter

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
    
    # Extract keywords from the conversation
    words = re.findall(r'\b[a-zA-Z]{3,}\b', conversation_text.lower())
    word_freq = Counter(words)
    top_words = [w for w, _ in word_freq.most_common(5)]
    
    # Create a humanized summary
    if top_words:
        if len(top_words) == 1:
            summary = f"You mainly discussed {top_words[0]}."
        elif len(top_words) <= 3:
            summary = f"You talked about {', '.join(top_words[:-1])} and {top_words[-1]}."
        else:
            summary = f"You talked about {', '.join(top_words[:3])} and more."
    else:
        summary = "You had a conversation."
    
    # Combine with existing context, but keep it bounded
    if current_context and "Recently discussed:" in current_context:
        # Extract the recent topics from current context
        recent_topics = current_context.split("Recently discussed:")[1].strip() if "Recently discussed:" in current_context else ""
        updated_context = f"Recently discussed: {summary} {recent_topics}"
    else:
        updated_context = f"Recently discussed: {summary}"
    
    # Keep context length bounded
    if len(updated_context) > 400:
        # Truncate to keep it readable
        updated_context = updated_context[:400] + "..."
    
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
    
    # Extract keywords from the context
    if "Recently discussed:" in aggregated_context:
        context_content = aggregated_context.split("Recently discussed:")[1].strip()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', context_content.lower())
        word_freq = Counter(words)
        top_words = [w for w, _ in word_freq.most_common(3)]
        
        if top_words:
            if len(top_words) == 1:
                description = f"Recently talked about {top_words[0]}"
            elif len(top_words) == 2:
                description = f"Recently talked about {top_words[0]} and {top_words[1]}"
            else:
                description = f"Recently talked about {', '.join(top_words)}"
        else:
            description = "Recently interacted"
    else:
        description = "Recently interacted"
    
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
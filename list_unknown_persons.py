#!/usr/bin/env python3
"""List all unknown persons in the database."""

import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_unknown_persons():
    """List all unknown persons in the database."""
    try:
        # Make request to get all people
        response = requests.get("http://localhost:8000/people")
        
        if response.status_code == 200:
            people = response.json()
            
            # Filter for unknown persons
            unknown_persons = [person for person in people if person.get("name") == "Unknown Person"]
            
            if unknown_persons:
                print(f"Found {len(unknown_persons)} unknown person(s):")
                print("-" * 50)
                for person in unknown_persons:
                    print(f"ID: {person.get('person_id')}")
                    print(f"Name: {person.get('name')}")
                    print(f"Relationship: {person.get('relationship')}")
                    print(f"Created: {person.get('last_updated')}")
                    print(f"Face Embeddings: {len(person.get('face_embeddings', []))}")
                    print("-" * 50)
            else:
                print("No unknown persons found in the database.")
                
            # Also show total count
            print(f"\nTotal people in database: {len(people)}")
        else:
            logger.error(f"Failed to get people list: {response.status_code}")
            logger.error(response.text)
            
    except Exception as e:
        logger.error(f"Error listing unknown persons: {e}")

if __name__ == "__main__":
    list_unknown_persons()
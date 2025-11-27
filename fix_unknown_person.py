#!/usr/bin/env python3
"""Script to fix an unknown person record by updating their information and adding face embedding."""

import requests
import json
from datetime import datetime
import sys

def fix_unknown_person(person_id, name, relationship):
    """Fix an unknown person record by updating their information and adding face embedding."""
    base_url = "http://localhost:8000"
    
    print(f"Fixing person ID: {person_id}")
    print(f"Updating name to: {name}")
    print(f"Updating relationship to: {relationship}")
    
    # Step 1: Update the person's information
    print("\n1. Updating person information...")
    update_data = {
        "name": name,
        "relationship": relationship,
        "aggregated_context": f"Identified as {name}",
        "cached_description": f"Identified as {name}"
    }
    
    try:
        response = requests.put(f"{base_url}/person/{person_id}", json=update_data)
        if response.status_code == 200:
            print("✅ Person information updated successfully")
            updated_person = response.json()
            print(f"   Updated person: {updated_person['name']} ({updated_person['relationship']})")
        else:
            print(f"❌ Failed to update person information. Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error updating person information: {e}")
        return False
    
    # Step 2: Add a face embedding
    print("\n2. Adding face embedding...")
    # Generate a dummy face embedding (64-dimensional vector)
    dummy_embedding = [round(float(i)/100, 3) for i in range(64)]
    
    embedding_data = {
        "person_id": person_id,
        "face_embedding": dummy_embedding,
        "source_image_url": "http://example.com/identified_person.jpg"
    }
    
    try:
        response = requests.post(f"{base_url}/face/embedding", json=embedding_data)
        if response.status_code == 200:
            print("✅ Face embedding added successfully")
        else:
            print(f"❌ Failed to add face embedding. Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error adding face embedding: {e}")
        return False
    
    print("\n✅ Person record fixed successfully!")
    return True

if __name__ == "__main__":
    # Fix the specific person you mentioned
    person_id = "unknown_1764140700577"
    name = "John Doe"  # Change this to the actual name
    relationship = "Friend"  # Change this to the actual relationship
    
    # Allow command line arguments to override defaults
    if len(sys.argv) >= 4:
        person_id = sys.argv[1]
        name = sys.argv[2]
        relationship = sys.argv[3]
    elif len(sys.argv) >= 2:
        person_id = sys.argv[1]
        name = input("Enter the person's name: ")
        relationship = input("Enter the person's relationship: ")
    
    print("=== MongoDB Hackathon - Fix Unknown Person Script ===")
    fix_unknown_person(person_id, name, relationship)
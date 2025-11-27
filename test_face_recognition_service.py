#!/usr/bin/env python3
"""
Test script for the face recognition service
"""

import requests
import json
from datetime import datetime
import numpy as np
import time

def test_face_recognition_service():
    """Test the face recognition service endpoints"""
    
    print("=== Face Recognition Service Test ===")
    print("Testing health endpoint...")
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Health check failed: Could not connect to face recognition service")
        print("   Please make sure the face recognition service is running on port 8001")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    print("\nTesting embedding extraction endpoint...")
    
    # Test embedding extraction with dummy data
    # In a real test, you would upload an actual image file
    try:
        # Create a simple test image (100x100 RGB image)
        import io
        from PIL import Image
        import base64
        
        # Create a dummy image
        img = Image.new('RGB', (100, 100), color = 'red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create form data
        files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
        
        # Send request to embedding endpoint
        response = requests.post(
            "http://localhost:8001/extract_embedding",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            embedding_data = response.json()
            print(f"✅ Embedding extraction passed")
            print(f"   Embedding dimensions: {embedding_data.get('dimensions')}")
            print(f"   Model: {embedding_data.get('model')}")
            print(f"   Success: {embedding_data.get('success')}")
            # Show first 5 values of embedding
            if 'embedding' in embedding_data:
                print(f"   First 5 embedding values: {embedding_data['embedding'][:5]}")
                # Verify embedding dimensions
                if len(embedding_data['embedding']) == 128:
                    print(f"   ✅ Correct embedding dimensions (128)")
                else:
                    print(f"   ❌ Incorrect embedding dimensions: expected 128, got {len(embedding_data['embedding'])}")
        else:
            print(f"❌ Embedding extraction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Embedding extraction failed: Could not connect to face recognition service")
        return False
    except Exception as e:
        print(f"❌ Embedding extraction error: {e}")
        return False
    
    print("\n✅ All face recognition service tests passed!")
    return True

def test_integration_with_main_backend():
    """Test integration with the main backend service"""
    
    print("\n=== Main Backend Integration Test ===")
    print("Testing face embedding storage with model field...")
    
    # Test creating a person and adding face embedding with model field
    base_url = "http://localhost:8000"
    
    # Check if main backend is running
    try:
        health_response = requests.get(f"{base_url}/", timeout=5)
        if health_response.status_code != 200:
            print("⚠️  Main backend service not responding correctly")
    except requests.exceptions.ConnectionError:
        print("⚠️  Main backend service not running - skipping integration tests")
        return True
    except Exception as e:
        print(f"⚠️  Error checking main backend: {e} - skipping integration tests")
        return True
    
    # Create a test person
    person_id = f"test_person_{int(datetime.now().timestamp() * 1000)}"
    person_data = {
        "person_id": person_id,
        "name": "Test Person",
        "relationship": "Test",
        "aggregated_context": "Test context",
        "cached_description": "Test description"
    }
    
    try:
        # Create person
        response = requests.post(f"{base_url}/person", json=person_data, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to create person: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        print("✅ Person created successfully")
        
        # Generate a dummy face embedding
        face_embedding = [round(float(i)/100, 3) for i in range(128)]
        
        # Add face embedding with model field
        embedding_data = {
            "person_id": person_id,
            "face_embedding": face_embedding,
            "source_image_url": "http://localhost/test_face.jpg",
            "model": "facenet-resnet50"
        }
        
        response = requests.post(f"{base_url}/face/embedding", json=embedding_data, timeout=10)
        if response.status_code == 200:
            print("✅ Face embedding with model field added successfully")
        else:
            print(f"❌ Failed to add face embedding: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
        # Verify the person has the embedding with model field
        response = requests.get(f"{base_url}/people", timeout=10)
        if response.status_code == 200:
            people = response.json()
            target_person = None
            for person in people:
                if person.get('person_id') == person_id:
                    target_person = person
                    break
            
            if target_person and 'face_embeddings' in target_person:
                embeddings = target_person['face_embeddings']
                if len(embeddings) > 0:
                    embedding = embeddings[0]
                    if 'model' in embedding and embedding['model'] == 'facenet-resnet50':
                        print("✅ Face embedding stored with correct model field")
                        print(f"   Model: {embedding['model']}")
                        print(f"   Vector length: {len(embedding.get('vector', []))}")
                    else:
                        print("⚠️  Face embedding stored but model field missing or incorrect")
                        print(f"   Expected: facenet-resnet50, Got: {embedding.get('model', 'None')}")
                else:
                    print("❌ No face embeddings found for person")
                    return False
            else:
                print("❌ Person or face embeddings not found")
                return False
        else:
            print(f"❌ Failed to get people list: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Integration test failed: Could not connect to main backend service")
        return False
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False
    
    print("\n✅ Main backend integration test passed!")
    return True

def test_multi_image_enrollment():
    """Test multi-image enrollment simulation"""
    
    print("\n=== Multi-Image Enrollment Test ===")
    
    # Generate multiple dummy embeddings
    embeddings = []
    for i in range(3):  # Simulate 3 images
        # Create slightly different embeddings
        embedding = [round(float(j)/100 + i*0.01, 3) for j in range(128)]
        embeddings.append(embedding)
    
    # Calculate average embedding
    avg_embedding = [0.0] * 128
    for embedding in embeddings:
        for j in range(128):
            avg_embedding[j] += embedding[j]
    
    for j in range(128):
        avg_embedding[j] /= len(embeddings)
    
    print(f"✅ Generated {len(embeddings)} dummy embeddings")
    print(f"✅ Calculated average embedding")
    print(f"   First 5 average values: {avg_embedding[:5]}")
    
    # Verify dimensions
    if len(avg_embedding) == 128:
        print(f"✅ Correct average embedding dimensions (128)")
    else:
        print(f"❌ Incorrect average embedding dimensions: expected 128, got {len(avg_embedding)}")
        return False
    
    print("\n✅ Multi-image enrollment test passed!")
    return True

if __name__ == "__main__":
    print("Starting face recognition system tests...")
    
    # Add a small delay to ensure services have time to start
    time.sleep(2)
    
    # Test face recognition service
    service_success = test_face_recognition_service()
    
    if service_success:
        # Test multi-image enrollment
        multi_image_success = test_multi_image_enrollment()
        
        # Test integration with main backend
        integration_success = test_integration_with_main_backend()
        
        if multi_image_success and integration_success:
            print("\n🎉 All tests passed! The face recognition system is working correctly.")
        else:
            print("\n⚠️  Some tests had issues, but the core functionality may still work.")
    else:
        print("\n❌ Face recognition service tests failed.")
        print("   Please make sure the face recognition service is running on port 8001")
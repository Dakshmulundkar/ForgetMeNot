"""Configuration for Fireworks.ai fine-tuning."""

import os
import numpy as np  # NEW: Added for threshold tuning
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fireworks API Configuration
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Base model to fine-tune
# Options: "accounts/fireworks/models/llama-v3p1-8b-instruct"
#          "accounts/fireworks/models/qwen2p5-7b-instruct"
#          "accounts/fireworks/models/gemma-2-9b-it"
BASE_MODEL = os.getenv("BASE_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct")

# Fine-tuning parameters
LORA_RANK = int(os.getenv("LORA_RANK", "8"))  # Must be power of 2, max 64
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))  # Default learning rate
EPOCHS = int(os.getenv("EPOCHS", "3"))  # Number of training epochs
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))  # Batch size for training

# Dataset paths
TRAINING_DATA_PATH = "data/context_aggregation_training.jsonl"

# Model naming
FINETUNE_JOB_NAME = os.getenv("FINETUNE_JOB_NAME", "context-aggregation-model-v1")

# Validation
if not FIREWORKS_API_KEY:
    raise ValueError(
        "FIREWORKS_API_KEY not found in environment variables. "
        "Please create a .env file with your API key."
    )

# NEW: Threshold tuning function for face recognition
def tune_face_threshold_from_test_data(positive_pairs, negative_pairs):
    """
    Grid search to find optimal threshold on your test data.
    Run once with 10-20 positive + 10-20 negative pairs from your environment.
    """
    def compute_similarity(e1, e2):
        e1_norm = e1 / np.linalg.norm(e1)
        e2_norm = e2 / np.linalg.norm(e2)
        return np.dot(e1_norm, e2_norm)
    
    positive_sims = [compute_similarity(e1, e2) for e1, e2 in positive_pairs]
    negative_sims = [compute_similarity(e1, e2) for e1, e2 in negative_pairs]
    
    best_threshold = 0.5
    best_accuracy = 0
    
    for threshold in np.arange(0.5, 1.0, 0.01):
        tp = sum(1 for s in positive_sims if s >= threshold)
        tn = sum(1 for s in negative_sims if s < threshold)
        accuracy = (tp + tn) / (len(positive_sims) + len(negative_sims))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f}, Accuracy: {best_accuracy:.1%}")
    return best_threshold


# NEW: Script to run threshold tuning and save result
def run_threshold_tuning_and_save(positive_pairs_file="positive_pairs.json", negative_pairs_file="negative_pairs.json", output_file="tuned_threshold.txt"):
    """
    Run threshold tuning with JSON files containing positive and negative pairs,
    and save the optimal threshold to a file.
    
    Args:
        positive_pairs_file: Path to JSON file with positive pairs
        negative_pairs_file: Path to JSON file with negative pairs
        output_file: Path to output file for tuned threshold
    """
    import json
    
    # Load pairs from JSON files
    with open(positive_pairs_file, 'r') as f:
        positive_pairs = json.load(f)
    
    with open(negative_pairs_file, 'r') as f:
        negative_pairs = json.load(f)
    
    # Run tuning
    optimal_threshold = tune_face_threshold_from_test_data(positive_pairs, negative_pairs)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(f"{optimal_threshold:.3f}")
    
    # Also print for immediate use
    print(f"Tuned threshold saved to {output_file}")
    print(f"Set FACE_RECOGNITION_THRESHOLD={optimal_threshold:.3f} in your environment variables")
    
    return optimal_threshold

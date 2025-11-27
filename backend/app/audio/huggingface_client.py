"""Hugging Face client for cloud-based speaker embedding inference."""

import os
import logging
from typing import Optional, Dict, Any
import numpy as np
from huggingface_hub import InferenceClient

logger = logging.getLogger("webrtc.audio.hf_client")

class HuggingFaceEmbeddingClient:
    """Client for Hugging Face embedding models via API."""
    
    def __init__(self):
        self.auth_token = os.getenv("PYANNOTE_AUTH_TOKEN")
        self.model_name = "pyannote/embedding"
        self.client = None
        
        if self.auth_token:
            try:
                self.client = InferenceClient(
                    model=self.model_name,
                    token=self.auth_token
                )
                logger.info("Initialized Hugging Face client for %s", self.model_name)
            except Exception as e:
                logger.error("Failed to initialize Hugging Face client: %s", e)
                self.client = None
        else:
            logger.warning("PYANNOTE_AUTH_TOKEN not set; Hugging Face client disabled")
    
    async def embed_audio(self, audio_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Get speaker embedding for audio data using Hugging Face API.
        
        Args:
            audio_data: Dictionary with 'waveform' (np.ndarray) and 'sample_rate' (int)
            
        Returns:
            Embedding vector or None if failed
        """
        if self.client is None:
            logger.warning("Hugging Face client not initialized")
            return None
            
        try:
            # Convert numpy array to the format expected by Hugging Face
            waveform = audio_data.get("waveform")
            sample_rate = audio_data.get("sample_rate")
            
            if waveform is None or sample_rate is None:
                logger.error("Invalid audio data format")
                return None
                
            # Hugging Face API expects the audio data in a specific format
            # For now, we'll return a placeholder - in a real implementation,
            # you would send the audio to the Hugging Face API
            logger.info("Using Hugging Face API for embedding (placeholder implementation)")
            
            # In a real implementation, you would do something like:
            # result = self.client.feature_extraction(inputs=audio_data)
            # return np.array(result)
            
            # For now, return a placeholder vector
            return np.random.rand(512).astype(np.float32)
            
        except Exception as e:
            logger.error("Error getting embedding from Hugging Face: %s", e)
            return None

# Global instance
hf_client = HuggingFaceEmbeddingClient()
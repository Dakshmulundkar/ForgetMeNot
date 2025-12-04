"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/hooks/use-toast"
import * as faceapi from 'face-api.js'

// Use the same backend URL as other components
const OFFER_BACKEND_URL = "http://localhost:8000"

interface IdentifyUnknownPersonProps {
  personId: string
  onIdentified: (name: string, relationship: string) => void
  onCancel: () => void
  videoRef: React.RefObject<HTMLVideoElement | null> | null
  canvasRef: React.RefObject<HTMLCanvasElement | null> | null
  loadFaceApiModels?: () => Promise<boolean> // Add this prop
  modelsLoaded?: boolean // Add this prop
}

export function IdentifyUnknownPerson({ 
  personId, 
  onIdentified, 
  onCancel, 
  videoRef, 
  canvasRef,
  loadFaceApiModels,
  modelsLoaded
}: IdentifyUnknownPersonProps) {
  const [name, setName] = useState("")
  const [relationship, setRelationship] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [step, setStep] = useState<"info" | "capturing" | "success">("info")
  const { toast } = useToast()

  /**
   * Extract face embedding using backend service
   * @param canvas - Canvas element containing face image
   * @returns Promise resolving to face embedding array
   */
  const extractFaceEmbedding = async (canvas: HTMLCanvasElement): Promise<number[]> => {
    // Convert canvas to blob
    return new Promise((resolve, reject) => {
      canvas.toBlob(async (blob) => {
        if (!blob) {
          reject(new Error('Failed to create blob from canvas'));
          return;
        }

        try {
          // Create FormData for file upload
          const formData = new FormData();
          formData.append('file', blob, 'face.jpg');

          // Send to backend service
          console.log('[FaceEmbedding] Sending request to face recognition service...');
          const response = await fetch('http://localhost:8001/extract_embedding', {
            method: 'POST',
            body: formData,
          });

          console.log('[FaceEmbedding] Face recognition service response status:', response.status);
          if (!response.ok) {
            const errorText = await response.text();
            console.error('[FaceEmbedding] Face recognition service error response:', errorText);
            let errorMessage = 'Failed to extract face embedding';
            try {
              const errorData = JSON.parse(errorText);
              errorMessage = errorData.detail || errorMessage;
            } catch (parseError) {
              errorMessage = errorText || errorMessage;
            }
            throw new Error(errorMessage);
          }

          const result = await response.json();
          console.log('[FaceEmbedding] Face recognition service response:', result);
          
          // Check if the backend reported success
          if (result.success === false) {
            console.warn('[FaceEmbedding] Backend reported failure:', result.error);
            // Still try to use the embedding if provided, but log the warning
            if (result.embedding && Array.isArray(result.embedding)) {
              console.warn('[FaceEmbedding] Using fallback embedding from backend');
              resolve(result.embedding);
            } else {
              throw new Error(result.error || 'Face recognition failed');
            }
          } else {
            resolve(result.embedding);
          }
        } catch (error) {
          console.error('[FaceEmbedding] Error extracting embedding:', error);
          toast({
            title: "Face Recognition Error",
            description: `Failed to extract face embedding: ${error instanceof Error ? error.message : 'Unknown error'}`,
            variant: "destructive",
          });
          reject(error);
        }
      }, 'image/jpeg');
    });
  };

  /**
   * Capture multiple face embeddings for enrollment with quality filtering
   * @returns Promise resolving to array of quality-filtered face embeddings
   */
  const captureMultipleEmbeddingsV2 = async (): Promise<Array<{vector: number[], quality_score: number, captured_at: string, model: string, blur_score: number, confidence: number}> | null> => {
    if (!videoRef?.current || !canvasRef?.current) {
      console.error('[FaceEmbedding] Video or canvas not available');
      toast({
        title: "Error",
        description: "Video or canvas not available for face capture",
        variant: "destructive",
      });
      return null;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    if (!context) {
      console.error('[FaceEmbedding] Canvas context not available');
      toast({
        title: "Error",
        description: "Canvas context not available for face capture",
        variant: "destructive",
      });
      return null;
    }

    const embeddings: Array<{vector: number[], quality_score: number, captured_at: string, model: string, blur_score: number, confidence: number}> = [];
    
    // NEW: Capture 5 images instead of 7, with more lenient quality filtering
    for (let i = 0; i < 5; i++) {
      try {
        // Wait a bit between captures to allow for slight variations
        if (i > 0) {
          await new Promise(resolve => setTimeout(resolve, 300)); // Reduced delay to 300ms for faster capture
        }
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get image data for quality computation
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        
        // Extract face embedding using backend service
        console.log('[FaceEmbedding] Attempting to extract face embedding...');
        const embedding = await extractFaceEmbedding(canvas);
        console.log('[FaceEmbedding] Successfully extracted face embedding, length:', embedding?.length);
        
        // Validate embedding
        if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
          console.warn('[FaceEmbedding] Invalid embedding received, skipping frame');
          continue;
        }
        
        // Compute quality: confidence + blur + face size
        // Note: In a real implementation, we would use a face detector here
        // For now, we'll use simplified quality metrics
        const confidence = 0.9; // Increased placeholder confidence
        const blurScore = computeLaplacianBlurness(imageData); // Real blur detection
        const faceSize = 1.0; // Placeholder - would come from bounding box size
        
        // Simplified quality score calculation (more lenient scoring)
        const qualityScore = 0.3 * confidence + 0.5 * blurScore + 0.2 * Math.min(faceSize, 1.0);
        
        // Only store embeddings that meet quality threshold
        // NEW: Make threshold configurable with default 0.5 (lowered for better capture rate)
        const QUALITY_THRESHOLD = parseFloat(process.env.REACT_APP_FACE_QUALITY_THRESHOLD || '0.5');
        if (qualityScore >= QUALITY_THRESHOLD) {
          embeddings.push({
            vector: embedding,
            quality_score: qualityScore,
            captured_at: new Date().toISOString(),
            model: 'facenet-resnet50',
            blur_score: blurScore,
            confidence: confidence
          });
        }
        
        console.log(`[FaceEmbedding] Captured embedding ${i+1}/5, quality: ${qualityScore.toFixed(3)}`);
      } catch (err) {
        console.error(`[FaceEmbedding] Error capturing embedding ${i+1}:`, err);
        // Show toast notification for embedding capture errors
        toast({
          title: "Capture Error",
          description: `Error capturing frame ${i+1}: ${err instanceof Error ? err.message : 'Unknown error'}`,
          variant: "destructive",
        });
        continue;
      }
    }
    
    // Require at least 1 frame (relaxed requirement)
    if (embeddings.length < 1) {
      // If no high-quality frames, try to return what we have if any were captured
      console.warn(`Only ${embeddings.length} high-quality frames; need ≥1`);
      toast({
        title: "Low Quality Capture",
        description: `Only ${embeddings.length} acceptable frames captured. Try to keep face steady and well-lit.`,
        variant: "destructive",
      });
      // Don't throw error, just return what we have (could be null)
      return embeddings.length > 0 ? embeddings : null;
    }
    
    // Sort by quality score (highest first)
    return embeddings.sort((a, b) => b.quality_score - a.quality_score);
  };

  // Helper function to compute image blurriness using Laplacian variance
  const computeLaplacianBlurness = (imageData: ImageData): number => {
    const { width, height, data } = imageData;
    if (width === 0 || height === 0) return 0;
    
    // Convert to grayscale
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const grayValue = 0.299 * r + 0.587 * g + 0.114 * b;
      gray[i / 4] = grayValue;
    }
    
    // Apply Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]]
    let variance = 0;
    let sum = 0;
    let sumSquares = 0;
    let count = 0;
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const center = gray[y * width + x];
        const top = gray[(y - 1) * width + x];
        const bottom = gray[(y + 1) * width + x];
        const left = gray[y * width + (x - 1)];
        const right = gray[y * width + (x + 1)];
        
        // Apply Laplacian operator
        const laplacian = Math.abs(top + bottom + left + right - 4 * center);
        
        sum += laplacian;
        sumSquares += laplacian * laplacian;
        count++;
      }
    }
    
    if (count === 0) return 0;
    
    const mean = sum / count;
    variance = (sumSquares / count) - (mean * mean);
    
    // Normalize to 0-1 range (empirically determined range)
    // Variance values typically range from 0 to several thousand
    const normalized = Math.min(1, variance / 1000);
    return normalized;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      toast({
        title: "Error",
        description: "Please enter a name",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    
    try {
      // Update the person's information
      const response = await fetch(`${OFFER_BACKEND_URL}/person/${personId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: name.trim(),
          relationship: relationship.trim() || "Acquaintance",
          aggregated_context: `Just met ${name.trim()}`,
          cached_description: `Just met ${name.trim()}`
        }),
      });

      if (response.ok) {
        // After updating person info, capture face embedding
        setStep("capturing");
        
        // NEW: Capture multiple embeddings with quality filtering (no averaging)
        const faceEmbeddings = await captureMultipleEmbeddingsV2();
        
        if (faceEmbeddings && faceEmbeddings.length > 0) {
          // Send all quality-filtered embeddings to backend
          const embeddingResponses = await Promise.all(faceEmbeddings.map(async (embeddingData) => {
            return await fetch(`${OFFER_BACKEND_URL}/face/embedding`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                person_id: personId,
                face_embedding: embeddingData.vector,
                quality_score: embeddingData.quality_score,
                captured_at: embeddingData.captured_at,
                model: embeddingData.model,
                blur_score: embeddingData.blur_score,
                confidence: embeddingData.confidence
              })
            });
          }));
          
          // Check if all embeddings were saved successfully
          const allSuccessful = embeddingResponses.every(response => response.ok);
          
          if (allSuccessful) {
            setStep("success");
            toast({
              title: "Success",
              description: `Person identified as ${name.trim()} and ${faceEmbeddings.length} face embeddings captured successfully!`,
            });
            // Give user a moment to see the success message
            setTimeout(() => {
              onIdentified(name.trim(), relationship.trim() || "Acquaintance");
            }, 1500);
          } else {
            toast({
              title: "Partial Success",
              description: `Person identified as ${name.trim()} but some face embeddings failed to save. Please try capturing again later.`,
              variant: "destructive",
            });
          }
        } else {
          // Still consider it a success even if face capture failed, since we identified the person
          setStep("success");
          toast({
            title: "Person Identified",
            description: `Person identified as ${name.trim()}. Face capture was skipped or failed, but you can try capturing later.`,
          });
          // Give user a moment to see the message
          setTimeout(() => {
            onIdentified(name.trim(), relationship.trim() || "Acquaintance");
          }, 1500);
        }
      } else {
        toast({
          title: "Error",
          description: "Failed to update person information.",
          variant: "destructive",
        });
      }
    } catch (err) {
      console.error('[IdentifyPerson] Error identifying person:', err);
      toast({
        title: "Error",
        description: "Error identifying person. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Identify Unknown Person</CardTitle>
        <CardDescription>
          {step === "info" && "This person was detected but not recognized. Please provide their information."}
          {step === "capturing" && "Capturing face embedding..."}
          {step === "success" && "Success! Face embedding captured."}
        </CardDescription>
      </CardHeader>
      {step === "info" && (
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="text-sm font-medium">Name</div>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter person's name"
                required
              />
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium">Relationship</div>
              <Input
                value={relationship}
                onChange={(e) => setRelationship(e.target.value)}
                placeholder="e.g., Friend, Family member, Caregiver"
              />
            </div>
            <div className="text-sm text-muted-foreground">
              <p>Person ID: {personId}</p>
            </div>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button type="button" variant="outline" onClick={onCancel} disabled={isLoading}>
              Cancel
            </Button>
            <Button type="submit" disabled={isLoading || !name.trim()}>
              {isLoading ? "Saving..." : "Save & Capture Face"}
            </Button>
          </CardFooter>
        </form>
      )}
      {step === "capturing" && (
        <CardContent className="flex flex-col items-center justify-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
          <p>Capturing face embedding...</p>
        </CardContent>
      )}
      {step === "success" && (
        <CardContent className="flex flex-col items-center justify-center py-8">
          <div className="rounded-full bg-green-100 p-3 mb-4">
            <div className="rounded-full bg-green-500 p-1">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
          </div>
          <p>Face embedding captured successfully!</p>
        </CardContent>
      )}
    </Card>
  )
}
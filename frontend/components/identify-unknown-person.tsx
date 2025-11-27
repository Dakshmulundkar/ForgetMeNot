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
          const response = await fetch('http://localhost:8001/extract_embedding', {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to extract face embedding');
          }

          const result = await response.json();
          resolve(result.embedding);
        } catch (error) {
          console.error('[FaceEmbedding] Error extracting embedding:', error);
          reject(error);
        }
      }, 'image/jpeg');
    });
  };

  /**
   * Capture multiple face embeddings for enrollment and compute average
   * @returns Promise resolving to averaged face embedding
   */
  const captureMultipleEmbeddings = async (): Promise<number[] | null> => {
    if (!videoRef?.current || !canvasRef?.current) {
      console.error('[FaceEmbedding] Video or canvas not available');
      return null;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    if (!context) {
      console.error('[FaceEmbedding] Canvas context not available');
      return null;
    }

    const embeddings: number[][] = [];
    
    // Capture 3-5 slightly varied face images
    for (let i = 0; i < 3; i++) {
      try {
        // Wait a bit between captures to allow for slight variations
        if (i > 0) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Extract face embedding using backend service
        const embedding = await extractFaceEmbedding(canvas);
        embeddings.push(embedding);
        
        console.log(`[FaceEmbedding] Captured embedding ${i+1}/3`);
      } catch (err) {
        console.error(`[FaceEmbedding] Error capturing embedding ${i+1}:`, err);
        continue;
      }
    }
    
    if (embeddings.length === 0) {
      return null;
    }
    
    // Compute average embedding
    const avgEmbedding: number[] = [];
    for (let i = 0; i < embeddings[0].length; i++) {
      const sum = embeddings.reduce((acc, emb) => acc + emb[i], 0);
      avgEmbedding.push(sum / embeddings.length);
    }
    
    // L2 normalize the averaged embedding
    const norm = Math.sqrt(avgEmbedding.reduce((sum, val) => sum + val * val, 0));
    const normalizedEmbedding = avgEmbedding.map(val => val / norm);
    
    return normalizedEmbedding;
  };

  const captureFaceEmbedding = async (): Promise<boolean> => {
    if (!videoRef?.current || !canvasRef?.current) {
      console.error('[FaceEmbedding] Video or canvas not available')
      return false
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    
    if (!context) {
      console.error('[FaceEmbedding] Canvas context not available')
      return false
    }

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    
    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    
    try {
      // Extract face embedding using backend service
      const faceEmbedding = await extractFaceEmbedding(canvas);
      
      // Create face embedding data
      const faceEmbeddingData = {
        person_id: personId,
        face_embedding: faceEmbedding,
        source_image_url: '',
        model: 'facenet-vgg2' // Add model version field
      }

      try {
        // Send face embedding to backend
        const response = await fetch(`${OFFER_BACKEND_URL}/face/embedding`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(faceEmbeddingData)
        })

        if (response.ok) {
          console.log('[FaceEmbedding] Successfully sent face embedding to backend')
          return true
        } else {
          console.error('[FaceEmbedding] Failed to send face embedding:', response.status)
          return false
        }
      } catch (err) {
        console.error('[FaceEmbedding] Error sending face embedding:', err)
        toast({
          title: "Error",
          description: "Failed to capture face embedding. Network error occurred.",
          variant: "destructive",
        })
        return false
      }
    } catch (err) {
      console.error('[IdentifyPerson] Error generating face embedding:', err)
      toast({
        title: "Error",
        description: "Error generating face embedding. Please try again.",
        variant: "destructive",
      })
      return false
    }
  }

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
          aggregated_context: `Identified as ${name.trim()}`,
          cached_description: `Identified as ${name.trim()}`
        }),
      });

      if (response.ok) {
        // After updating person info, capture face embedding
        setStep("capturing");
        
        // Capture multiple embeddings and compute average
        const faceEmbedding = await captureMultipleEmbeddings();
        
        if (faceEmbedding && faceEmbedding.length > 0) {
          // Send averaged embedding to backend
          const embeddingResponse = await fetch(`${OFFER_BACKEND_URL}/face/embedding`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              person_id: personId,
              face_embedding: faceEmbedding,
              model: 'facenet-vgg2' // Add model version field
            })
          });
          
          if (embeddingResponse.ok) {
            setStep("success");
            toast({
              title: "Success",
              description: `Person identified as ${name.trim()} and face captured successfully!`,
            });
            // Give user a moment to see the success message
            setTimeout(() => {
              onIdentified(name.trim(), relationship.trim() || "Acquaintance");
            }, 1500);
          } else {
            toast({
              title: "Error",
              description: "Failed to save face embedding. Please try again.",
              variant: "destructive",
            });
          }
        } else {
          toast({
            title: "Partial Success",
            description: `Person identified as ${name.trim()} but face capture failed. You can try capturing again later.`,
            variant: "destructive",
          });
          // Still consider it a success since the person info was updated
          setTimeout(() => {
            onIdentified(name.trim(), relationship.trim() || "Acquaintance");
          }, 1500);
        }
      } else {
        const errorData = await response.json();
        toast({
          title: "Error",
          description: errorData.detail || "Failed to identify person. Please try again.",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('[IdentifyPerson] Network error:', error);
      toast({
        title: "Error",
        description: "Network error occurred. Please check your connection and try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

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
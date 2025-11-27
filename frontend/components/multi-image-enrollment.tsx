"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Camera, Upload, X } from "lucide-react"

interface MultiImageEnrollmentProps {
  personId: string
  onEnrollmentComplete: (averageEmbedding: number[]) => void
  onCancel: () => void
}

export function MultiImageEnrollment({ personId, onEnrollmentComplete, onCancel }: MultiImageEnrollmentProps) {
  const [images, setImages] = useState<{ id: string; url: string; file: File }[]>([])
  const [embeddings, setEmbeddings] = useState<number[][]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [isCapturing, setIsCapturing] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const MAX_IMAGES = 5
  const MIN_IMAGES = 3

  useEffect(() => {
    startWebcam()
    return () => {
      stopWebcam()
    }
  }, [])

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
      }
    } catch (err) {
      console.error("[MultiImageEnrollment] Error accessing webcam:", err)
      setError("Failed to access webcam. Please check permissions.")
    }
  }

  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
  }

  const captureImage = () => {
    if (!videoRef.current || images.length >= MAX_IMAGES) return
    
    const video = videoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], `face_${Date.now()}.jpg`, { type: 'image/jpeg' })
          const url = URL.createObjectURL(blob)
          setImages(prev => [...prev, { id: Date.now().toString(), url, file }])
        }
      }, 'image/jpeg')
    }
  }

  const removeImage = (id: string) => {
    setImages(prev => prev.filter(img => img.id !== id))
    setEmbeddings(prev => {
      const index = prev.findIndex((_, i) => images[i].id === id)
      if (index !== -1) {
        const newEmbeddings = [...prev]
        newEmbeddings.splice(index, 1)
        return newEmbeddings
      }
      return prev
    })
  }

  const extractEmbedding = async (file: File): Promise<number[]> => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch('http://localhost:8001/extract_embedding', {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Failed to extract face embedding')
    }

    const result = await response.json()
    return result.embedding
  }

  const processImages = async () => {
    if (images.length < MIN_IMAGES) {
      setError(`Please capture at least ${MIN_IMAGES} images`)
      return
    }

    setIsProcessing(true)
    setError(null)
    
    try {
      // Extract embeddings for all images
      const newEmbeddings: number[][] = []
      for (let i = 0; i < images.length; i++) {
        setCurrentStep(i + 1)
        try {
          const embedding = await extractEmbedding(images[i].file)
          newEmbeddings.push(embedding)
        } catch (err) {
          console.error(`[MultiImageEnrollment] Error extracting embedding for image ${i}:`, err)
          throw new Error(`Failed to process image ${i + 1}: ${(err as Error).message}`)
        }
      }
      
      setEmbeddings(newEmbeddings)
      
      // Calculate average embedding
      if (newEmbeddings.length > 0) {
        const avgEmbedding = newEmbeddings[0].map((_, idx) => {
          const sum = newEmbeddings.reduce((acc, embedding) => acc + embedding[idx], 0)
          return sum / newEmbeddings.length
        })
        
        onEnrollmentComplete(avgEmbedding)
      }
    } catch (err) {
      console.error("[MultiImageEnrollment] Error processing images:", err)
      setError((err as Error).message || "Failed to process images")
    } finally {
      setIsProcessing(false)
      setCurrentStep(0)
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    const newImages = Array.from(files).slice(0, MAX_IMAGES - images.length)
    newImages.forEach(file => {
      const url = URL.createObjectURL(file)
      setImages(prev => [...prev, { id: Date.now().toString() + Math.random(), url, file }])
    })
    
    // Clear the input
    e.target.value = ''
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <CardHeader>
          <CardTitle>Multi-Image Face Enrollment</CardTitle>
          <p className="text-sm text-muted-foreground">
            Capture {MIN_IMAGES}-{MAX_IMAGES} images from different angles for better recognition accuracy
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          {error && (
            <div className="bg-destructive/10 border border-destructive/50 text-destructive rounded-md p-3 text-sm">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Camera Preview */}
            <div className="space-y-4">
              <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                {isCapturing && (
                  <div className="absolute inset-0 border-4 border-white rounded-lg animate-pulse" />
                )}
              </div>
              
              <div className="flex flex-wrap gap-2">
                <Button
                  onClick={captureImage}
                  disabled={images.length >= MAX_IMAGES || isProcessing}
                  className="flex-1"
                >
                  <Camera className="w-4 h-4 mr-2" />
                  Capture Image ({images.length}/{MAX_IMAGES})
                </Button>
                
                <label className="flex-1">
                  <Button
                    variant="outline"
                    className="w-full"
                    disabled={images.length >= MAX_IMAGES || isProcessing}
                    asChild
                  >
                    <span>
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Image
                    </span>
                  </Button>
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={handleFileUpload}
                    disabled={images.length >= MAX_IMAGES || isProcessing}
                  />
                </label>
              </div>
            </div>

            {/* Image Gallery */}
            <div className="space-y-4">
              <h3 className="font-medium">Captured Images</h3>
              <div className="grid grid-cols-2 gap-2">
                {images.map((image) => (
                  <div key={image.id} className="relative group">
                    <img
                      src={image.url}
                      alt="Captured face"
                      className="w-full h-32 object-cover rounded-md border"
                    />
                    <Button
                      variant="destructive"
                      size="icon"
                      className="absolute top-1 right-1 h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => removeImage(image.id)}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
                {images.length === 0 && (
                  <div className="col-span-2 flex items-center justify-center h-32 border border-dashed rounded-md text-muted-foreground">
                    No images captured yet
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Progress */}
          {isProcessing && (
            <div className="space-y-2">
              <p className="text-sm">
                Processing image {currentStep} of {images.length}...
              </p>
              <Progress value={(currentStep / images.length) * 100} />
            </div>
          )}

          {/* Actions */}
          <div className="flex flex-wrap gap-3 justify-end">
            <Button variant="outline" onClick={onCancel} disabled={isProcessing}>
              Cancel
            </Button>
            <Button
              onClick={processImages}
              disabled={images.length < MIN_IMAGES || isProcessing}
              className="bg-green-600 hover:bg-green-700"
            >
              {isProcessing ? "Processing..." : "Complete Enrollment"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
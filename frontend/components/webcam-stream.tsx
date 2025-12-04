"use client"

import React from "react"
import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Mic, MicOff, Video, VideoOff } from "lucide-react"
import { FaceNotification } from "@/components/face-notification"
import { useFaceDetection } from "@/hooks/use-face-detection"
import { RayBanOverlay } from "@/components/rayban-overlay"
import { cn } from "@/lib/utils"
import { IdentifyUnknownPerson } from "@/components/identify-unknown-person"
import { MultiImageEnrollment } from "@/components/multi-image-enrollment"
import {
  calculateVideoTransform,
  mapBoundingBoxToOverlay,
  calculateNotificationPosition,
} from "@/lib/coordinate-mapper"
import * as faceapi from 'face-api.js'
import type { DetectedFace } from '@/lib/face-tracker'

interface PersonData {
  name: string
  description: string
  relationship: string
  person_id?: string
}

interface FaceEmbeddingData {
  person_id: string
  face_embedding: number[]
  source_image_url?: string
  model?: string
}

interface FaceRecognitionData {
  face_embedding: number[]
}

interface FaceRecognitionResponse {
  known: boolean
  person_id?: string
  name?: string
  relationship?: string
  description?: string
  conversation_history?: any[]
  message?: string
  similarity?: number  // NEW: Add similarity property
}

type FacePersonMap = Map<string, PersonData>

export function WebcamStream() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const dataChannelRef = useRef<RTCDataChannel | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  
  const [isStreaming, setIsStreaming] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [isRayBanMode, setIsRayBanMode] = useState(false)
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [latestPersonData, setLatestPersonData] = useState<PersonData | null>(null)
  const [unknownPersonId, setUnknownPersonId] = useState<string | null>(null)
  const [showIdentificationForm, setShowIdentificationForm] = useState(false)
  const [isListeningForName, setIsListeningForName] = useState(false)
  const [facePersonData, setFacePersonData] = useState<FacePersonMap>(new Map())
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [showMultiImageEnrollment, setShowMultiImageEnrollment] = useState(false)
  
  // State for face-api.js model loading
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [modelsLoading, setModelsLoading] = useState(false)
  const [isModelLoading, setIsModelLoading] = useState(false)

  // State for automatic face recognition
  const [lastRecognizedFaces, setLastRecognizedFaces] = useState<Set<string>>(new Set())
  const [isAutoRecognizing, setIsAutoRecognizing] = useState(false)
  // Track the timestamp of last recognition for each face to allow re-recognition after a timeout
  const [faceLastRecognitionTime, setFaceLastRecognitionTime] = useState<Map<string, number>>(new Map())
  
  // NEW: Add face recognition cache
  const [faceRecognitionCache, setFaceRecognitionCache] = useState<Map<string, {
    person_id: string;
    similarity: number;
    timestamp: number;
  }>>(new Map())

  // NEW: Activity tracking for session management
  const [lastActivityTimestamp, setLastActivityTimestamp] = useState<number>(Date.now())
  const [isSessionActive, setIsSessionActive] = useState<boolean>(true)

  const INFERENCE_BACKEND_URL = "http://localhost:8000"
  const OFFER_BACKEND_URL = "http://localhost:8000"

  const { detectedFaces, isLoading: isFaceDetectionLoading, error: faceDetectionError } = useFaceDetection(
    videoRef.current,
    {
      enabled: isStreaming && !isRayBanMode,
      minDetectionConfidence: 0.5,
      targetFps: 20,
      useWorker: true,
    }
  )

  useEffect(() => {
    if (!latestPersonData || detectedFaces.length === 0) {
      return
    }

    const mostProminentFace = detectedFaces.reduce((prev, current) => {
      const prevScore = (prev.boundingBox.width * prev.boundingBox.height) * prev.confidence
      const currentScore = (current.boundingBox.width * current.boundingBox.height) * current.confidence
      return currentScore > prevScore ? current : prev
    })

    setFacePersonData((prev: FacePersonMap) => {
      const newMap = new Map(prev)
      newMap.set(mostProminentFace.id, latestPersonData)
      return newMap
    })

    console.log(`[FaceDetection] Associated "${latestPersonData.name}" with face ${mostProminentFace.id}`)
    
    // Only clear latestPersonData for known persons, not unknown ones
    if (latestPersonData.name !== "Unknown Person") {
      setLatestPersonData(null)
    }
  }, [latestPersonData, detectedFaces])

  useEffect(() => {
    const currentFaceIds = new Set(detectedFaces.map(f => f.id))
    setFacePersonData((prev: FacePersonMap) => {
      const newMap = new Map(prev)
      for (const faceId of newMap.keys()) {
        // Only remove face data if it's not the latest detected person
        if (!currentFaceIds.has(faceId) && (!latestPersonData || latestPersonData.person_id !== faceId)) {
          newMap.delete(faceId)
        }
      }
      return newMap
    })
  }, [detectedFaces, latestPersonData])

  // NEW: Update activity timestamp when recognition result comes back
  useEffect(() => {
    if (latestPersonData) {
      // Recognition result received - update timestamp
      setLastActivityTimestamp(Date.now());
      if (!isSessionActive) {
        console.log('[Session] Recognition result received, marking session as active');
        setIsSessionActive(true);
      }
    }
  }, [latestPersonData, isSessionActive]);

  useEffect(() => {
    startWebcam()
    connectSSE()

    return () => {
      stopWebcam()
      disconnectSSE()
    }
  }, [])

  // NEW: Activity monitor effect
  useEffect(() => {
    // Only run if we're streaming
    if (!isStreaming) return;

    // Set up activity monitoring interval
    const activityMonitorInterval = setInterval(() => {
      const now = Date.now();
      const timeSinceLastActivity = now - lastActivityTimestamp;
      
      // If no activity for 30 seconds, mark session as idle
      if (timeSinceLastActivity > 30000 && isSessionActive) {
        console.log('[Session] No activity for 30 seconds, marking session as idle');
        setIsSessionActive(false);
        // Clear active person state but keep WebSocket and camera running
        setLatestPersonData(null);
        setFacePersonData(new Map());
        setUnknownPersonId(null);
        setIsListeningForName(false);
      }
    }, 5000); // Check every 5 seconds

    return () => {
      clearInterval(activityMonitorInterval);
    };
  }, [isStreaming, lastActivityTimestamp, isSessionActive]);

  // NEW: Update activity timestamp when face is detected
  useEffect(() => {
    if (detectedFaces.length > 0) {
      // Face activity detected - update timestamp
      setLastActivityTimestamp(Date.now());
      if (!isSessionActive) {
        console.log('[Session] Face detected, marking session as active');
        setIsSessionActive(true);
      }
    }
  }, [detectedFaces, isSessionActive]);

  const connectSSE = () => {
    try {
      const es = new EventSource(`${INFERENCE_BACKEND_URL}/stream/inference`)
      
      console.log('[SSE] Connecting to:', `${INFERENCE_BACKEND_URL}/stream/inference`)
      
      es.onopen = () => {
        console.log('[SSE] Connected')
      }
      
      es.addEventListener('inference', (event) => {
        try {
          const message = JSON.parse(event.data)
          console.log('[SSE] Received inference event:', message)
          
          // Handle both the old format and new format
          if (message.name && message.description && message.relationship) {
            setLatestPersonData({
              name: message.name,
              description: message.description,
              relationship: message.relationship,
              person_id: message.person_id,
            })
          } else if (message.event_type === 'PERSON_DETECTED' && message.person_id) {
            // Handle the conversation event format
            setLatestPersonData({
              name: message.person_id.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
              description: "Recently detected speaker",
              relationship: "Recently detected",
              person_id: message.person_id,
            })
          }
        } catch (err) {
          console.error('[SSE] Error parsing message:', err)
        }
      })
      
      es.onerror = (error) => {
        console.error('[SSE] Error:', error)
        console.log('[SSE] This error occurs when the SSE connection fails. Make sure the backend is running at http://localhost:8000')
      }
      
      setEventSource(es)
    } catch (err) {
      console.error('[SSE] Connection error:', err)
    }
  }

  const disconnectSSE = () => {
    if (eventSource) {
      eventSource.close()
      setEventSource(null)
    }
  }

  const waitForIceGathering = (pc: RTCPeerConnection): Promise<void> => {
    return new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') {
        resolve()
        return
      }

      const checkState = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', checkState)
          resolve()
        }
      }

      pc.addEventListener('icegatheringstatechange', checkState)
    })
  }

  const setupWebSocket = async (stream: MediaStream) => {
    // Track reconnection attempts
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 10;
    const baseReconnectDelay = 1000; // 1 second
    
    const connect = () => {
      try {
        console.log('[WebSocket] Setting up WebSocket connection (attempt ' + (reconnectAttempts + 1) + ')');
        const websocket = new WebSocket('ws://localhost:8000/ws/audio');
        
        // Store the WebSocket instance
        let heartbeatInterval: NodeJS.Timeout | null = null;
        
        websocket.onopen = () => {
          console.log('[WebSocket] Connected to audio streaming');
          setIsConnected(true);
          setWs(websocket);
          reconnectAttempts = 0; // Reset reconnect attempts on successful connection
          
          // Start heartbeat to prevent disconnection - MAXIMUM FREQUENCY FOR STABILITY
          heartbeatInterval = setInterval(() => {
            if (websocket.readyState === WebSocket.OPEN) {
              websocket.send(JSON.stringify({ type: 'heartbeat' }));
              
              // NEW: Update activity timestamp on heartbeat send (indicates connection is alive)
              setLastActivityTimestamp(Date.now());
            }
          }, 5000); // Send heartbeat every 5 seconds for MAXIMUM stability
        };

        websocket.onclose = (event) => {
          console.log('[WebSocket] Disconnected from audio streaming', event.reason);
          setIsConnected(false);
          setWs(null);
          
          // Clear heartbeat interval
          if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
          }
          
          // Attempt to reconnect with exponential backoff
          if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = Math.min(baseReconnectDelay * Math.pow(2, reconnectAttempts), 30000); // Max 30s delay
            console.log(`[WebSocket] Attempting to reconnect in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
            setTimeout(connect, delay);
          } else {
            console.error('[WebSocket] Maximum reconnection attempts reached');
            setIsConnected(false);
          }
        };

        websocket.onerror = (error) => {
          console.error('[WebSocket] Error:', error);
          console.log('[WebSocket] This error occurs when the WebSocket connection fails. Make sure the backend is running at http://localhost:8000');
          setIsConnected(false);
          
          // Clear heartbeat interval
          if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
          }
        };

        // Set up audio processing
        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length > 0) {
          const audioTrack = audioTracks[0];
          const audioContext = new AudioContext();
          const source = audioContext.createMediaStreamSource(stream);
          const processor = audioContext.createScriptProcessor(4096, 1, 1);
          
          source.connect(processor);
          processor.connect(audioContext.destination);
          
          processor.onaudioprocess = (e) => {
            if (websocket.readyState === WebSocket.OPEN) {
              const inputData = e.inputBuffer.getChannelData(0);
              // Convert float32 to int16
              const buffer = new ArrayBuffer(inputData.length * 2);
              const view = new DataView(buffer);
              let index = 0;
              for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                index += 2;
              }
              websocket.send(buffer);
              
              // NEW: Update activity timestamp when audio is sent
              setLastActivityTimestamp(Date.now());
              if (!isSessionActive) {
                console.log('[Session] Audio chunk sent, marking session as active');
                setIsSessionActive(true);
              }
            }
          };
        }
      } catch (err) {
        console.error('[WebSocket] Setup error:', err);
        setIsConnected(false);
        
        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          const delay = Math.min(baseReconnectDelay * Math.pow(2, reconnectAttempts), 30000);
          console.log(`[WebSocket] Retrying connection in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
          setTimeout(connect, delay);
        }
      }
    };
    
    // Initial connection
    connect();
  };

  const setupWebRTC = async (stream: MediaStream) => {
    // This function is now deprecated since we're using WebSocket
    // But keeping it for backward compatibility
    try {
      console.log('[WebRTC] Setting up peer connection')
      const pc = new RTCPeerConnection()
      pcRef.current = pc

      stream.getTracks().forEach(track => {
        console.log('[WebRTC] Adding track:', track.kind)
        pc.addTrack(track, stream)
      })

      pc.onconnectionstatechange = () => {
        console.log('[WebRTC] Connection state:', pc.connectionState)
        setIsConnected(pc.connectionState === 'connected')
      }

      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      console.log('[WebRTC] Created offer, waiting for ICE gathering...')

      await waitForIceGathering(pc)
      console.log('[WebRTC] ICE gathering complete, sending offer to backend')

      const response = await fetch(`${OFFER_BACKEND_URL}/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription!.sdp,
          type: pc.localDescription!.type
        })
      })

      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`)
      }

      const answer = await response.json()
      console.log('[WebRTC] Received answer from backend')
      await pc.setRemoteDescription(answer)
      console.log('[WebRTC] Connection established!')
    } catch (err) {
      console.error('[WebRTC] Setup error:', err)
      setIsConnected(false)
    }
  }

  // Effect for automatic face recognition when new faces are detected
  useEffect(() => {
    if (!isStreaming || isAutoRecognizing || detectedFaces.length === 0) {
      return
    }

    // Check if we have new faces that haven't been recognized recently
    // NEW: Increase re-recognition timeout from 30s to 60s
    const RECOGNITION_TIMEOUT = 60000 // 60 seconds
    const currentTime = Date.now()
    
    const newFaces = detectedFaces.filter(face => {
      const lastRecognitionTime = faceLastRecognitionTime.get(face.id)
      // If face was never recognized, or was recognized more than 60 seconds ago
      return !lastRecognitionTime || (currentTime - lastRecognitionTime > RECOGNITION_TIMEOUT)
    })
    
    // NEW: Limit parallel recognitions to a reasonable number (e.g., 3 faces) to avoid CPU overload
    const MAX_PARALLEL_RECOGNITIONS = 3
    const facesToProcess = newFaces.slice(0, MAX_PARALLEL_RECOGNITIONS)
    
    // NEW: Process ALL detected faces in parallel instead of just the most prominent one
    if (facesToProcess.length > 0 && videoRef.current && canvasRef.current) {
      // Start automatic recognition
      setIsAutoRecognizing(true)
      
      console.log(`[AutoFaceRecognition] New faces detected: ${newFaces.length}, triggering automatic recognition for up to ${MAX_PARALLEL_RECOGNITIONS} faces`)
      
      // Perform face recognition for ALL new faces in parallel
      Promise.all(facesToProcess.map(async (face) => {
        try {
          await recognizeFaceAutomaticallyForFace(face)
          // Update the last recognition time for this face
          setFaceLastRecognitionTime(prev => {
            const newMap = new Map(prev)
            newMap.set(face.id, Date.now())
            return newMap
          })
        } catch (error) {
          console.error(`[AutoFaceRecognition] Error recognizing face ${face.id}:`, error)
        }
      })).finally(() => {
        setIsAutoRecognizing(false)
      })
    }
  }, [detectedFaces, isStreaming, isAutoRecognizing, faceLastRecognitionTime])

  // NEW: Function to manually trigger face recognition for all detected faces
  const recognizeFaceAutomatically = async () => {
    if (!isStreaming || detectedFaces.length === 0) {
      console.log('[ManualFaceRecognition] No faces detected or streaming not active');
      return;
    }

    console.log(`[ManualFaceRecognition] Manually triggering recognition for ${detectedFaces.length} faces`);
    
    // Process ALL detected faces in parallel
    setIsAutoRecognizing(true);
    
    try {
      await Promise.all(detectedFaces.map(async (face) => {
        try {
          await recognizeFaceAutomaticallyForFace(face);
          // Update the last recognition time for this face
          setFaceLastRecognitionTime(prev => {
            const newMap = new Map(prev);
            newMap.set(face.id, Date.now());
            return newMap;
          });
        } catch (error) {
          console.error(`[ManualFaceRecognition] Error recognizing face ${face.id}:`, error);
        }
      }));
    } finally {
      setIsAutoRecognizing(false);
    }
  };

  // NEW: Function to recognize a specific face
  const recognizeFaceAutomaticallyForFace = async (face: DetectedFace): Promise<void> => {
    if (!videoRef.current || !canvasRef.current) {
      console.error('[AutoFaceRecognition] Video or canvas not available');
      return;
    }

    // NEW: Check cache first
    const cacheKey = `${face.boundingBox.originX.toFixed(0)}_${face.boundingBox.originY.toFixed(0)}_${face.boundingBox.width.toFixed(0)}_${face.boundingBox.height.toFixed(0)}`;
    const cachedResult = faceRecognitionCache.get(cacheKey);
    
    // If cached result exists and is not too old (5 minutes), and has high similarity, use it
    if (cachedResult && 
        Date.now() - cachedResult.timestamp < 50000 && // 5 minutes
        cachedResult.similarity > 0.85) {
      console.log(`[AutoFaceRecognition] Using cached recognition result for face ${face.id}`);
      
      // Get person data from facePersonData or create temporary person
      const cachedPersonData = facePersonData.get(face.id);
      if (cachedPersonData) {
        setLatestPersonData(cachedPersonData);
        // Update last recognition time
        setFaceLastRecognitionTime(prev => {
          const newMap = new Map(prev);
          newMap.set(face.id, Date.now());
          return newMap;
        });
        return;
      }
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    if (!context) {
      console.error('[AutoFaceRecognition] Canvas context not available');
      return;
    }

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    try {
      // Extract face embedding using backend service
      const faceEmbedding = await extractFaceEmbedding(canvas);
      
      // Create face recognition data
      const faceRecognitionData: FaceRecognitionData = {
        face_embedding: faceEmbedding
      }

      try {
        // Send face embedding to backend for recognition
        const response = await fetch(`${OFFER_BACKEND_URL}/face/recognize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(faceRecognitionData)
        })
        
        if (response.ok) {
          const result: FaceRecognitionResponse = await response.json()
          console.log(`[AutoFaceRecognition] Recognition result for face ${face.id}:`, result)
          
          // NEW: Cache the recognition result
          if (result.known && result.person_id) {
            const cacheKey = `${face.boundingBox.originX.toFixed(0)}_${face.boundingBox.originY.toFixed(0)}_${face.boundingBox.width.toFixed(0)}_${face.boundingBox.height.toFixed(0)}`;
            setFaceRecognitionCache(prev => {
              const newMap = new Map(prev);
              newMap.set(cacheKey, {
                person_id: result.person_id!,
                similarity: result.similarity || 0.9, // Default high similarity for known persons
                timestamp: Date.now()
              });
              return newMap;
            });
          }
          
          if (result.known && result.person_id && result.name && result.relationship) {
            // Known person detected
            setLatestPersonData({
              name: result.name || "Unknown",
              description: result.description || "No previous interactions",
              relationship: result.relationship || "Unknown",
              person_id: result.person_id
            })
            console.log(`[AutoFaceRecognition] Known person detected: ${result.name}`)
            
            // Associate this person with this specific face
            setFacePersonData((prev: FacePersonMap) => {
              const newMap = new Map(prev)
              newMap.set(face.id, {
                name: result.name || "Unknown",
                description: result.description || "No previous interactions",
                relationship: result.relationship || "Unknown",
                person_id: result.person_id
              })
              return newMap
            })
          } else {
            // Unknown person detected - check if we already have a temporary person for this session
            if (!unknownPersonId) {
              const newUnknownId = `unknown_${Date.now()}_${face.id}`
              setUnknownPersonId(newUnknownId)
              setIsListeningForName(true)
              console.log('[AutoFaceRecognition] Unknown person detected, listening for name')
              
              // Set latestPersonData for the unknown person so buttons appear
              setLatestPersonData({
                name: "Unknown Person",
                description: "Just met",
                relationship: "Unknown",
                person_id: newUnknownId
              })
              
              // Associate this unknown person with this specific face
              setFacePersonData((prev: FacePersonMap) => {
                const newMap = new Map(prev)
                newMap.set(face.id, {
                  name: "Unknown Person",
                  description: "Just met",
                  relationship: "Unknown",
                  person_id: newUnknownId
                })
                return newMap
              })
              
              // Create a temporary person entry for the unknown person
              // BUT first check if a person with this name already exists
              await new Promise(resolve => setTimeout(resolve, 1000));
              
              // Check if person already exists
              let shouldCreateTempPerson = true;
              try {
                const peopleResponse = await fetch(`${OFFER_BACKEND_URL}/people`);
                if (peopleResponse.ok) {
                  const people = await peopleResponse.json();
                  // Look for a person with this person_id
                  const existingUnknown = people.find((p: any) => 
                    p.person_id === newUnknownId
                  );
                  
                  if (existingUnknown) {
                    console.log('[AutoFaceRecognition] Temporary person already exists, skipping creation');
                    shouldCreateTempPerson = false;
                  }
                }
              } catch (error) {
                console.error('[AutoFaceRecognition] Error checking for existing temporary person:', error);
              }
              
              if (shouldCreateTempPerson) {
                const tempPersonResponse = await fetch(`${OFFER_BACKEND_URL}/person`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    person_id: newUnknownId,
                    name: "Unknown Person",
                    relationship: "Unknown",
                    aggregated_context: "Just met",
                    cached_description: "Just met"
                  })
                }).catch(error => {
                  console.error('[AutoFaceRecognition] Network error creating person:', error);
                  // Continue even if person creation fails
                });
                
                if (tempPersonResponse && tempPersonResponse.ok) {
                  console.log('[AutoFaceRecognition] Created temporary person entry for unknown person')
                  // Immediately capture face embedding for this unknown person
                  await captureFaceEmbedding(newUnknownId).catch(embeddingError => {
                    console.error('[AutoFaceRecognition] Failed to capture face embedding:', embeddingError)
                    // Continue even if face embedding fails
                  })
                }
              }
            }
          }
        } else {
          console.error('[AutoFaceRecognition] Failed to recognize face:', response.status)
          // Even if recognition fails, we still want to create a temporary person
          if (!unknownPersonId) {
            const newUnknownId = `unknown_${Date.now()}_${face.id}`
            setUnknownPersonId(newUnknownId)
            setIsListeningForName(true)
            console.log('[AutoFaceRecognition] Face recognition failed, creating temporary person')
            
            // Set latestPersonData for the unknown person so buttons appear
            setLatestPersonData({
              name: "Unknown Person",
              description: "Just met",
              relationship: "Unknown",
              person_id: newUnknownId
            })
            
            // Associate this unknown person with this specific face
            setFacePersonData((prev: FacePersonMap) => {
              const newMap = new Map(prev)
              newMap.set(face.id, {
                name: "Unknown Person",
                description: "Just met",
                relationship: "Unknown",
                person_id: newUnknownId
              })
              return newMap
            })
            
            // Create a temporary person entry for the unknown person
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Check if person already exists
            let shouldCreateTempPerson = true;
            try {
              const peopleResponse = await fetch(`${OFFER_BACKEND_URL}/people`);
              if (peopleResponse.ok) {
                const people = await peopleResponse.json();
                // Look for a person with this person_id
                const existingUnknown = people.find((p: any) => 
                  p.person_id === newUnknownId
                );
                
                if (existingUnknown) {
                  console.log('[AutoFaceRecognition] Temporary person already exists, skipping creation');
                  shouldCreateTempPerson = false;
                }
              }
            } catch (error) {
              console.error('[AutoFaceRecognition] Error checking for existing temporary person:', error);
            }
            
            if (shouldCreateTempPerson) {
              const tempPersonResponse = await fetch(`${OFFER_BACKEND_URL}/person`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  person_id: newUnknownId,
                  name: "Unknown Person",
                  relationship: "Unknown",
                  aggregated_context: "Just met",
                  cached_description: "Just met"
                })
              }).catch(error => {
                console.error('[AutoFaceRecognition] Network error creating person:', error);
                // Continue even if person creation fails
              });
              
              if (tempPersonResponse && tempPersonResponse.ok) {
                console.log('[AutoFaceRecognition] Created temporary person entry for unknown person after recognition failure')
              }
            }
          }
        }
      } catch (error) {
        console.error('[AutoFaceRecognition] Error during face recognition:', error)
        // Even if recognition fails, we still want to create a temporary person
        if (!unknownPersonId) {
          const newUnknownId = `unknown_${Date.now()}_${face.id}`
          setUnknownPersonId(newUnknownId)
          setIsListeningForName(true)
          console.log('[AutoFaceRecognition] Face recognition error, creating temporary person')
          
          // Set latestPersonData for the unknown person so buttons appear
          setLatestPersonData({
            name: "Unknown Person",
            description: "Just met",
            relationship: "Unknown",
            person_id: newUnknownId
          })
          
          // Associate this unknown person with this specific face
          setFacePersonData((prev: FacePersonMap) => {
            const newMap = new Map(prev)
            newMap.set(face.id, {
              name: "Unknown Person",
              description: "Just met",
              relationship: "Unknown",
              person_id: newUnknownId
            })
            return newMap
          })
          
          // Create a temporary person entry for the unknown person
          // BUT first check if a person with this name already exists
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          // Check if person already exists
          let shouldCreateTempPerson = true;
          try {
            const peopleResponse = await fetch(`${OFFER_BACKEND_URL}/people`);
            if (peopleResponse.ok) {
              const people = await peopleResponse.json();
              // Look for a person with this person_id
              const existingUnknown = people.find((p: any) => 
                p.person_id === newUnknownId
              );
              
              if (existingUnknown) {
                console.log('[AutoFaceRecognition] Temporary person already exists, skipping creation');
                shouldCreateTempPerson = false;
              }
            }
          } catch (error) {
            console.error('[AutoFaceRecognition] Error checking for existing temporary person:', error);
          }
          
          if (shouldCreateTempPerson) {
            const tempPersonResponse = await fetch(`${OFFER_BACKEND_URL}/person`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                person_id: newUnknownId,
                name: "Unknown Person",
                relationship: "Unknown",
                aggregated_context: "Just met",
                cached_description: "Just met"
              })
            }).catch(error => {
              console.error('[AutoFaceRecognition] Network error creating person:', error);
              // Continue even if person creation fails
            });
            
            if (tempPersonResponse && tempPersonResponse.ok) {
              console.log('[AutoFaceRecognition] Created temporary person entry for unknown person after recognition error')
            }
          }
        }
      }
    } catch (error) {
      console.error('[AutoFaceRecognition] Error extracting face embedding:', error)
    }
  }

  const extractNameAndRelationshipFromSpeech = (text: string): {name: string | null, relationship: string | null} => {
    // Handle empty or failed transcriptions
    if (!text || text.trim() === "" || 
        text.includes("Model loading failed") || 
        text.includes("Transcription failed") || 
        text.includes("Whisper not installed")) {
      console.log('[NameExtraction] Empty or failed transcription, cannot extract name/relationship');
      return {name: null, relationship: null};
    }
    
    // Extract name
    const namePatterns = [
      /i\s+am\s+([a-zA-Z\s]+)/i,
      /my\s+name\s+is\s+([a-zA-Z\s]+)/i,
      /i'm\s+([a-zA-Z\s]+)/i,
      /you\s+can\s+call\s+me\s+([a-zA-Z\s]+)/i,
      /i'm\s+called\s+([a-zA-Z\s]+)/i,
      /people\s+call\s+me\s+([a-zA-Z\s]+)/i
    ]
    
    let name: string | null = null
    for (const pattern of namePatterns) {
      const match = text.match(pattern)
      if (match && match[1]) {
        // Clean up the name
        let extractedName = match[1].trim()
        // Remove trailing words like "and", "so", etc.
        extractedName = extractedName.replace(/\s+(and|so|the|a|an|myself|me)$/i, '').trim()
        // Make sure we have a valid name (at least 2 characters)
        if (extractedName.length >= 2) {
          // Capitalize first letter of each word
          name = extractedName.replace(/\b\w/g, l => l.toUpperCase())
          break
        }
      }
    }
    
    // Extract relationship
    const relationshipPatterns = [
      /i\s+am\s+your\s+([a-zA-Z\s]+)/i,
      /i'm\s+your\s+([a-zA-Z\s]+)/i,
      /my\s+relationship\s+to\s+you\s+is\s+([a-zA-Z\s]+)/i,
      /i\s+am\s+([a-zA-Z\s]+)\s+to\s+you/i
    ]
    
    let relationship: string | null = null
    for (const pattern of relationshipPatterns) {
      const match = text.match(pattern)
      if (match && match[1]) {
        // Clean up the relationship
        let extractedRelationship = match[1].trim()
        // Remove trailing words like "and", "so", etc.
        extractedRelationship = extractedRelationship.replace(/\s+(and|so|the|a|an)$/i, '').trim()
        // Make sure we have a valid relationship (at least 2 characters)
        if (extractedRelationship.length >= 2) {
          // Capitalize first letter of each word
          relationship = extractedRelationship.replace(/\b\w/g, l => l.toUpperCase())
          break
        }
      }
    }
    
    return {name, relationship}
  }

  const createPersonFromName = async (name: string, existingPersonId?: string, relationship: string = "New acquaintance") => {
    try {
      // Add a small delay to ensure backend is ready
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // First, check if a person with this name already exists in the database
      let existingPerson = null;
      try {
        const peopleResponse = await fetch(`${OFFER_BACKEND_URL}/people`);
        if (peopleResponse.ok) {
          const people = await peopleResponse.json();
          // Look for a person with the same name (case insensitive)
          existingPerson = people.find((p: any) => 
            p.name && p.name.toLowerCase() === name.toLowerCase()
          );
          
          if (existingPerson) {
            console.log(`[PersonCreation] Found existing person with name ${name}:`, existingPerson);
            // If we found an existing person, use their ID instead of creating a new one
            existingPersonId = existingPerson.person_id;
          }
        }
      } catch (error) {
        console.error('[PersonCreation] Error checking for existing person:', error);
      }
      
      // If we have an existing person ID, update the existing person
      if (existingPersonId) {
        console.log(`[PersonCreation] Updating existing person ${existingPersonId} with name: ${name}`)
        
        const response = await fetch(`${OFFER_BACKEND_URL}/person/${existingPersonId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: name,
            relationship: relationship, // Use provided relationship or default
            aggregated_context: `Just met ${name}`,
            cached_description: `Just met ${name}`
          })
        }).catch(error => {
          console.error('[PersonCreation] Network error updating person:', error);
          // Continue with fallback approach
          return null;
        })
        
        if (response && response.ok) {
          const person = await response.json()
          console.log('[PersonCreation] Updated person:', person)
          return person
        } else if (response) {
          console.error('[PersonCreation] Failed to update person:', response.status)
          // Try to get more details about the error
          try {
            const errorText = await response.text();
            console.error('[PersonCreation] Error details:', errorText);
          } catch (e) {
            console.error('[PersonCreation] Could not read error details');
          }
          // Still return a basic person object so the process can continue
          return {
            person_id: existingPersonId,
            name: name,
            relationship: relationship
          }
        } else {
          // Network error occurred, but we still want to continue
          console.log('[PersonCreation] Network error, continuing with basic person object')
          return {
            person_id: existingPersonId,
            name: name,
            relationship: relationship
          }
        }
      } else {
        // Create a new person
        // But first check if a person with this name already exists
        let shouldCreatePerson = true;
        try {
          const peopleResponse = await fetch(`${OFFER_BACKEND_URL}/people`);
          if (peopleResponse.ok) {
            const people = await peopleResponse.json();
            // Look for a person with this name
            const existingPerson = people.find((p: any) => 
              p.name.toLowerCase() === name.toLowerCase()
            );
            
            if (existingPerson) {
              console.log('[PersonCreation] Person with this name already exists, updating existing person');
              existingPersonId = existingPerson.person_id;
              shouldCreatePerson = false;
              
              // Update the existing person
              const response = await fetch(`${OFFER_BACKEND_URL}/person/${existingPersonId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  name: name,
                  relationship: relationship, // Use provided relationship or default
                  aggregated_context: `Just met ${name}`,
                  cached_description: `Just met ${name}`
                })
              }).catch(error => {
                console.error('[PersonCreation] Network error updating person:', error);
                // Continue with fallback approach
                return null;
              })
              
              if (response && response.ok) {
                const person = await response.json()
                console.log('[PersonCreation] Updated existing person:', person)
                return person
              } else if (response) {
                console.error('[PersonCreation] Failed to update existing person:', response.status)
                // Try to get more details about the error
                try {
                  const errorText = await response.text();
                  console.error('[PersonCreation] Error details:', errorText);
                } catch (e) {
                  console.error('[PersonCreation] Could not read error details');
                }
                // Still return a basic person object so the process can continue
                return {
                  person_id: existingPersonId,
                  name: name,
                  relationship: relationship
                }
              } else {
                // Network error occurred, but we still want to continue
                console.log('[PersonCreation] Network error, continuing with basic person object')
                return {
                  person_id: existingPersonId,
                  name: name,
                  relationship: relationship
                }
              }
            }
          }
        } catch (error) {
          console.error('[PersonCreation] Error checking for existing person:', error);
        }
        
        if (shouldCreatePerson) {
          const personId = `person_${Date.now()}`
          const response = await fetch(`${OFFER_BACKEND_URL}/person`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              person_id: personId,
              name: name,
              relationship: relationship, // Use provided relationship or default
              aggregated_context: `Meet ${name}`,
              cached_description: `Meet ${name}`
            })
          }).catch(error => {
            console.error('[PersonCreation] Network error creating person:', error);
            // Continue with fallback approach
            return null;
          })
          
          if (response && response.ok) {
            const person = await response.json()
            console.log('[PersonCreation] Created person:', person)
            return person
          } else if (response) {
            console.error('[PersonCreation] Failed to create person:', response.status)
            // Try to get more details about the error
            try {
              const errorText = await response.text();
              console.error('[PersonCreation] Error details:', errorText);
            } catch (e) {
              console.error('[PersonCreation] Could not read error details');
            }
            // Still return a basic person object so the process can continue
            return {
              person_id: personId,
              name: name,
              relationship: relationship
            }
          } else {
            // Network error occurred, but we still want to continue
            console.log('[PersonCreation] Network error, continuing with basic person object')
            return {
              person_id: personId,
              name: name,
              relationship: relationship
            }
          }
        }
      }
    } catch (err) {
      console.error('[PersonCreation] Error creating/updating person:', err)
      // Return a basic person object so the process can continue
      const personId = existingPersonId || `person_${Date.now()}`
      return {
        person_id: personId,
        name: name,
        relationship: relationship
      }
    }
  }

  const startWebcam = async () => {
    try {
      console.log('[Webcam] Requesting media access...')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      })

      if (videoRef.current) {
        const video = videoRef.current

        const handleMetadataLoaded = () => {
          console.log('[Webcam] Video metadata loaded:', {
            width: video.videoWidth,
            height: video.videoHeight
          })
          setIsVideoReady(true)
        }

        video.addEventListener('loadedmetadata', handleMetadataLoaded)

        if (video.videoWidth > 0 && video.videoHeight > 0) {
          handleMetadataLoaded()
        }

        video.srcObject = stream
        streamRef.current = stream
        stream.getAudioTracks().forEach((track) => {
          track.enabled = !isMuted
        })
        setIsStreaming(true)
        console.log('[Webcam] Stream started')

        // Use WebSocket instead of WebRTC
        setTimeout(() => setupWebSocket(stream), 1000)
      }
    } catch (err) {
      console.error("[Webcam] Error accessing webcam:", err)
    }
  }

  const stopWebcam = () => {
    console.log('[Webcam] Stopping stream')

    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }

    if (pcRef.current) {
      pcRef.current.close()
      pcRef.current = null
    }

    // Close WebSocket connection
    if (ws) {
      ws.close()
      setWs(null)
    }

    streamRef.current = null
    setIsStreaming(false)
    setIsVideoReady(false)
    setIsConnected(false)
    setIsRayBanMode(false)
    setIsMuted(false)
  }

  const toggleMute = () => {
    if (!streamRef.current) return
    
    const stream = streamRef.current
    setIsMuted((prev: boolean) => {
      const newMuted = !prev
      stream.getAudioTracks().forEach((track: MediaStreamTrack) => {
        track.enabled = !newMuted
      })
      return newMuted
    })
  }

  // Handle speech recognition for unknown persons
  useEffect(() => {
    // Only listen for names when we're actually listening and have an unknown person
    if (!isListeningForName || !unknownPersonId) return

    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data)
        if (data.event_type === 'CONVERSATION_END' && data.conversation) {
          // Check if the unknown person said their name
          for (const utterance of data.conversation) {
            const {name, relationship} = extractNameAndRelationshipFromSpeech(utterance.text)
            if (name) {
              console.log(`[NameDetection] Detected name: ${name}`)
              // Use detected relationship or default to "New acquaintance"
              const personRelationship = relationship || "New acquaintance"
              // Create a new person with the detected name
              createPersonFromName(name, unknownPersonId, personRelationship).then((person) => {
                if (person) {
                  // Update latestPersonData with the identified person's information
                  setLatestPersonData({
                    name: name,
                    description: `Meet ${name}`,
                    relationship: personRelationship,
                    person_id: unknownPersonId
                  })
                  
                  // After creating/updating the person, capture their face embedding
                  if ((person.person_id || unknownPersonId) && videoRef.current && canvasRef.current) {
                    // Capture face embedding for the updated person
                    captureFaceEmbedding(person.person_id || unknownPersonId!).catch((embeddingError) => {
                      console.error('[FaceEmbedding] Failed to capture face embedding:', embeddingError)
                      // Don't fail the whole process if face embedding fails
                      alert(`Successfully identified ${name} but failed to capture face embedding. You can try again later.`)
                    })
                  }
                }
              })
              setIsListeningForName(false)
              setUnknownPersonId(null)
              break
            } else if (utterance.text && 
                      !utterance.text.includes("Model loading failed") && 
                      !utterance.text.includes("Transcription failed") && 
                      !utterance.text.includes("Whisper not installed") &&
                      utterance.text.trim() !== "") {
              // If we got a transcription but couldn't extract name, log it for debugging
              console.log(`[NameDetection] Transcription received but no name found: "${utterance.text}"`)
            } else {
              // Log failed transcription
              console.log(`[NameDetection] Transcription failed: "${utterance.text}"`)
            }
          }
        }
      } catch (err) {
        console.error('[NameDetection] Error parsing message:', err)
      }
    }

    // Listen for conversation events
    const eventSource = new EventSource(`${OFFER_BACKEND_URL}/stream/conversation`)
    eventSource.addEventListener('conversation', handleMessage)

    return () => {
      eventSource.close()
    }
  }, [isListeningForName, unknownPersonId])

  // Load face-api.js models with better error handling
  const loadFaceApiModels = useCallback(async () => {
    if (modelsLoaded || modelsLoading) return Promise.resolve(modelsLoaded)
    
    setModelsLoading(true)
    setIsModelLoading(true) // Set our custom loading state
    try {
      // Check if models directory exists and has files
      const modelCheckResponse = await fetch('/models/face_recognition_model-weights_manifest.json')
      if (!modelCheckResponse.ok) {
        console.warn('[FaceAPI] Model files not found, face detection may be limited')
        setModelsLoaded(true)
        return true
      }
      
      // Load face-api.js models with timeout
      const loadWithTimeout = (loadFn: () => Promise<any>, timeoutMs: number) => {
        return Promise.race([
          loadFn(),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Model loading timeout')), timeoutMs)
          )
        ]);
      };
      
      // Try to load models, but don't fail if they don't exist
      try {
        await loadWithTimeout(() => faceapi.nets.ssdMobilenetv1.loadFromUri('/models'), 10000)
      } catch (e) {
        console.warn('[FaceAPI] Failed to load SSD MobileNet model, using fallback')
      }
      
      try {
        await loadWithTimeout(() => faceapi.nets.tinyFaceDetector.loadFromUri('/models'), 5000)
      } catch (e) {
        console.warn('[FaceAPI] Failed to load Tiny Face Detector model, using fallback')
      }
      
      try {
        await loadWithTimeout(() => faceapi.nets.faceLandmark68Net.loadFromUri('/models'), 5000)
      } catch (e) {
        console.warn('[FaceAPI] Failed to load Face Landmark model')
      }
      
      try {
        await loadWithTimeout(() => faceapi.nets.faceRecognitionNet.loadFromUri('/models'), 5000)
      } catch (e) {
        console.warn('[FaceAPI] Failed to load Face Recognition model, using backend service')
      }
      
      setModelsLoaded(true)
      console.log('[FaceAPI] Models loading attempt completed')
      return true
    } catch (error) {
      console.error('[FaceAPI] Error loading models:', error)
      // Don't show alert here, just log the error
      console.error('Face recognition models failed to load. Face recognition may be limited.')
      // Still mark as loaded to prevent infinite retry loop
      setModelsLoaded(true)
      return false
    } finally {
      setModelsLoading(false)
      setIsModelLoading(false) // Reset our custom loading state
    }
  }, [modelsLoaded, modelsLoading])

  // Effect to load face-api.js models - but don't block the UI
  useEffect(() => {
    // Load models in the background without blocking UI
    // But only after the component has mounted and UI is responsive
    const loadModelsInBackground = async () => {
      // Small delay to ensure UI is responsive
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      try {
        await loadFaceApiModels()
      } catch (err) {
        console.error('[FaceAPI] Background model loading failed:', err)
      }
    }
    
    loadModelsInBackground()
  }, [loadFaceApiModels])

  const captureFaceEmbedding = async (personId: string) => {
    if (!videoRef.current || !canvasRef.current) {
      console.error('[FaceEmbedding] Video or canvas not available')
      alert('Video or canvas not available. Please start the webcam first.')
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    
    if (!context) {
      console.error('[FaceEmbedding] Canvas context not available')
      alert('Canvas context not available. Please try again.')
      return
    }

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    
    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    
    try {
      // Extract face embedding using backend service
      const faceEmbedding = await extractFaceEmbedding(canvas)
      
      // Create face embedding data
      const faceEmbeddingData: FaceEmbeddingData = {
        person_id: personId,
        face_embedding: faceEmbedding,
        source_image_url: '',
        model: 'facenet-resnet50' // Updated model version field
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
          // Show success message to user
          alert(`Face embedding captured for ${latestPersonData?.name || 'person'}!`)
        } else {
          const errorText = await response.text();
          console.error('[FaceEmbedding] Failed to send face embedding:', response.status, errorText)
          alert('Failed to capture face embedding. Please try again.')
        }
      } catch (err) {
        console.error('[FaceEmbedding] Error sending face embedding:', err)
        alert('Error capturing face embedding. Please check your network connection and try again.')
      }
    } catch (err: any) {
      console.error('[FaceEmbedding] Error generating face embedding:', err)
      alert(`Error generating face embedding: ${err.message || 'Please try again.'}`)
    }
  }

  // New function to handle multi-image enrollment
  const handleMultiImageEnrollment = async (personId: string) => {
    setShowMultiImageEnrollment(true)
  }

  // Handler for when multi-image enrollment is complete
  const handleEnrollmentComplete = async (personId: string, averageEmbedding: number[]) => {
    try {
      // Create face embedding data with average embedding
      const faceEmbeddingData: FaceEmbeddingData = {
        person_id: personId,
        face_embedding: averageEmbedding,
        source_image_url: '',
        model: 'facenet-resnet50'
      }

      // Send face embedding to backend
      const response = await fetch(`${OFFER_BACKEND_URL}/face/embedding`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(faceEmbeddingData)
      })

      if (response.ok) {
        console.log('[FaceEmbedding] Successfully sent averaged face embedding to backend')
        alert(`Face embeddings captured and averaged for ${latestPersonData?.name || 'person'}!`)
        setShowMultiImageEnrollment(false)
      } else {
        const errorText = await response.text();
        console.error('[FaceEmbedding] Failed to send averaged face embedding:', response.status, errorText)
        alert('Failed to capture averaged face embedding. Please try again.')
      }
    } catch (err) {
      console.error('[FaceEmbedding] Error sending averaged face embedding:', err)
      alert('Error capturing averaged face embedding. Please check your network connection and try again.')
    }
  }

  const faceNotifications = detectedFaces.map((face) => {
    const video = videoRef.current
    const overlay = overlayRef.current

    if (!video || !overlay) return null

    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight
    const overlayWidth = overlay.clientWidth
    const overlayHeight = overlay.clientHeight

    if (videoWidth === 0 || videoHeight === 0) return null

    const transform = calculateVideoTransform(
      videoWidth,
      videoHeight,
      overlayWidth,
      overlayHeight
    )

    const overlayBox = mapBoundingBoxToOverlay(
      face.boundingBox,
      transform,
      overlayWidth,
      true
    )

    const position = calculateNotificationPosition(
      overlayBox,
      overlayWidth,
      overlayHeight
    )

    return {
      face,
      position,
    }
  }).filter((n) => n !== null)

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      {/* Hidden canvas for face embedding capture */}
      <canvas ref={canvasRef} className="hidden" />
      
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={cn(
          "absolute inset-0 h-full w-full object-cover transition-all duration-300",
          isRayBanMode ? "scale-[1.08] blur-[13px]" : "scale-100"
        )}
        style={{ transform: 'scaleX(-1)' }}
      />

      {!isRayBanMode && (
        <>
          <div ref={overlayRef} className="absolute inset-0 pointer-events-none">
            {faceNotifications.map((notification) => {
              const person = facePersonData.get(notification!.face.id)
              return (
                <FaceNotification
                  key={notification!.face.id}
                  faceId={notification!.face.id}
                  left={notification!.position.left}
                  top={notification!.position.top}
                  confidence={notification!.face.confidence}
                  name={person?.name}
                  description={person?.description}
                  relationship={person?.relationship}
                />
              )
            })}
          </div>

          <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
            <div className={`w-2 h-2 rounded-full ${isSessionActive ? 'bg-green-500' : 'bg-yellow-500'} ${isSessionActive ? 'animate-pulse' : ''}`} />
            <span className="text-xs text-white/80">{isSessionActive ? 'Active Session' : 'Idle Session'}</span>
          </div>

          {/* Connection Status Indicator - Shows only Connected or Disconnected */}
          <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-xs text-white/80">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>

          {isModelLoading && (
            <div className="absolute top-16 left-4 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
              <span className="text-xs text-white/80">Loading face recognition models...</span>
            </div>
          )}
      
          {isFaceDetectionLoading && (
            <div className="absolute top-16 left-4 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
              <span className="text-xs text-white/80">Loading face detection...</span>
            </div>
          )}
          {faceDetectionError && (
            <div className="absolute top-4 left-4 px-3 py-2 bg-red-500/60 backdrop-blur-sm rounded-full">
              <span className="text-xs text-white/80">Face detection error</span>
            </div>
          )}

          {isListeningForName && (
            <div className="absolute top-16 left-4 px-3 py-2 bg-blue-500/60 backdrop-blur-sm rounded-full">
              <span className="text-xs text-white/80">Listening for name from unknown person...</span>
            </div>
          )}
          
          {showMultiImageEnrollment && latestPersonData?.person_id && (
            <MultiImageEnrollment
              personId={latestPersonData.person_id}
              onEnrollmentComplete={(averageEmbedding) => 
                handleEnrollmentComplete(latestPersonData.person_id!, averageEmbedding)
              }
              onCancel={() => setShowMultiImageEnrollment(false)}
            />
          )}
          
          {showIdentificationForm && unknownPersonId && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-50">
              <IdentifyUnknownPerson 
                personId={unknownPersonId}
                onIdentified={(name, relationship) => {
                  // Update the latest person data with the identified information
                  setLatestPersonData({
                    name,
                    description: `Just met ${name}`,
                    relationship,
                    person_id: unknownPersonId
                  })
                  setShowIdentificationForm(false)
                  setUnknownPersonId(null)
                }}
                onCancel={() => {
                  setShowIdentificationForm(false)
                  setUnknownPersonId(null)
                }}
                videoRef={videoRef}
                canvasRef={canvasRef}
                loadFaceApiModels={loadFaceApiModels}
                modelsLoaded={modelsLoaded}
              />
            </div>
          )}
        </>
      )}

      <RayBanOverlay stream={streamRef.current} videoRef={videoRef} visible={isRayBanMode} />

      <div className="absolute bottom-0 left-0 right-0 flex items-center justify-center px-6 py-4 bg-gradient-to-t from-black/80 to-transparent">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="icon"
            variant={isMuted ? "default" : "secondary"}
            onClick={toggleMute}
            className="h-12 w-12 rounded-full"
            disabled={!isStreaming}
            aria-pressed={isMuted}
          >
            {isMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
          </Button>
          <span className="text-sm text-white/80">{isMuted ? "Unmute" : "Mute"}</span>

          <Button
            size="icon"
            variant={isStreaming ? "default" : "secondary"}
            onClick={isStreaming ? stopWebcam : startWebcam}
            className="h-12 w-12 rounded-full"
          >
            {isStreaming ? <Video className="h-5 w-5" /> : <VideoOff className="h-5 w-5" />}
          </Button>
          <span className="text-sm text-white/80">{isStreaming ? "Stop Video" : "Start Video"}</span>

          <Button
            variant={isRayBanMode ? "default" : "secondary"}
            className="rounded-full px-4"
            disabled={!isStreaming}
            onClick={() => setIsRayBanMode((prev) => !prev)}
          >
            {isRayBanMode ? "Exit Ray-Ban Mode" : "Enter Ray-Ban Mode"}
          </Button>
          
          {/* Button to recognize face */}
          <Button
            variant="default"
            className="rounded-full px-4"
            onClick={recognizeFaceAutomatically}
          >
            Recognize Face
          </Button>
          
          {/* Button to capture face embedding for the latest detected person */}
          {latestPersonData?.person_id && (
            <>
              <Button
                variant="default"
                className="rounded-full px-4"
                onClick={() => captureFaceEmbedding(latestPersonData.person_id!)}
              >
                Capture Face
              </Button>
              <Button
                variant="secondary"
                className="rounded-full px-4"
                onClick={() => handleMultiImageEnrollment(latestPersonData.person_id!)}
              >
                Multi-Image Enrollment
              </Button>
            </>
          )}
          
          {/* Button to manually identify unknown person */}
          {unknownPersonId && !showIdentificationForm && (
            <Button
              variant="default"
              className="rounded-full px-4"
              onClick={() => setShowIdentificationForm(true)}
            >
              Identify Unknown Person
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

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
        reject(new Error('Failed to create image data from video feed'));
        return;
      }

      try {
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('file', blob, 'face.jpg');

        // Send to backend service with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 30 second timeout
        
        try {
          const response = await fetch('http://localhost:8001/extract_embedding', {
            method: 'POST',
            body: formData,
            signal: controller.signal
          });
          
          clearTimeout(timeoutId);
          
          if (!response.ok) {
            let errorMessage = `Face recognition service error: ${response.status}`;
            try {
              const errorData = await response.json();
              errorMessage = errorData.detail || `Face recognition service error: ${response.status} ${response.statusText}`;
            } catch (e) {
              // If we can't parse JSON, use status text
              errorMessage = `Face recognition service error: ${response.status} ${response.statusText}`;
            }
            throw new Error(errorMessage);
          }

          const result = await response.json();
          resolve(result.embedding);
        } catch (error: any) {
          clearTimeout(timeoutId);
          
          if (error.name === 'AbortError') {
            reject(new Error('Face recognition service timeout - please try again'));
          } else if (error.message && error.message.includes('fetch')) {
            reject(new Error('Unable to connect to face recognition service - please ensure it is running on port 8001'));
          } else {
            reject(new Error(`Face embedding extraction failed: ${error.message || 'Unknown error'}`));
          }
        }
      } catch (error: any) {
        console.error('[FaceEmbedding] Error extracting embedding:', error);
        reject(new Error(`Face embedding extraction failed: ${error.message || 'Unknown error'}`));
      }
    }, 'image/jpeg');
  });
};

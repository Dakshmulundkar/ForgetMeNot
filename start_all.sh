#!/bin/bash

# Shell script to start all services for the dementia-assistant project

echo "Starting dementia-assistant services..."
echo "========================================"

# Check if MongoDB is running
echo "Checking MongoDB..."
if nc -z localhost 27017; then
    echo "✓ MongoDB is running"
else
    echo "⚠️  MongoDB is not running. Please start MongoDB service manually."
    echo "   You can start it with: sudo systemctl start mongod"
    echo "   Or install MongoDB and start it as a service."
fi

echo ""
echo "Starting Backend API service..."
cd "$(dirname "$0")"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload --timeout-keep-alive 60 --timeout-graceful-shutdown 60 &
BACKEND_PID=$!
echo "✓ Backend API service started (PID: $BACKEND_PID)"

echo "Starting Face Recognition service..."
cd "$(dirname "$0")"
python -m uvicorn backend.face_recognition_service.main:app --host 127.0.0.1 --port 8001 --reload --timeout-keep-alive 60 --timeout-graceful-shutdown 60 &
FACE_PID=$!
echo "✓ Face Recognition service started (PID: $FACE_PID)"

echo "Starting Inference service..."
cd "$(dirname "$0")/inference"
python -m uvicorn main:app --host 127.0.0.1 --port 8002 --reload --timeout-keep-alive 60 --timeout-graceful-shutdown 60 &
INFERENCE_PID=$!
echo "✓ Inference service started (PID: $INFERENCE_PID)"
cd "$(dirname "$0")"

echo ""
echo "Services started:"
echo "- Backend API:     http://127.0.0.1:8000"
echo "- Face Service:    http://127.0.0.1:8001"
echo "- Inference Service: http://127.0.0.1:8002"
echo ""
echo "To start the frontend, run:"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo "Press Ctrl+C to stop services..."

# Wait for services to be terminated
trap "echo 'Stopping services...'; kill $BACKEND_PID $FACE_PID $INFERENCE_PID 2>/dev/null; exit" INT TERM

# Wait indefinitely
while true; do
    sleep 1
done
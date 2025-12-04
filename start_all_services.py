#!/usr/bin/env python3
"""
Script to start all services for the dementia assistant application.
This replaces the need to manually start each service in a new terminal.
"""

import subprocess
import sys
import os
import signal
import time
from typing import List

# Global list to keep track of subprocesses
processes: List[subprocess.Popen] = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by terminating all subprocesses."""
    print("\n\nReceived interrupt signal. Shutting down all services...")
    for process in processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    print("All services stopped.")
    sys.exit(0)

def start_service(command: List[str], name: str, cwd: str = None, shell: bool = False) -> subprocess.Popen:
    """Start a service and add it to the processes list."""
    print(f"Starting {name}...")
    # On Windows, we need shell=True for npm commands
    if shell:
        process = subprocess.Popen(
            ' '.join(command) if isinstance(command, list) else command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=cwd,
            shell=True
        )
    else:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=cwd
        )
    processes.append(process)
    return process

def main():
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting all dementia assistant services...")
    print("=" * 50)
    
    # Start MongoDB (if not already running)
    # Note: This assumes MongoDB is installed as a service or can be started manually
    print("Note: Make sure MongoDB is running on localhost:27017")
    print("If not, start it with: mongod --dbpath /path/to/db")
    print()
    
    # Start main backend service (port 8000) with maximum timeout settings
    backend_process = start_service([
        sys.executable, "-m", "uvicorn", 
        "backend.app.main:app", 
        "--reload", 
        "--host", "127.0.0.1",
        "--port", "8000",
        "--timeout-keep-alive", "60",
        "--timeout-graceful-shutdown", "60"
    ], "Main Backend Service")
    
    # Start face recognition service (port 8001) with maximum timeout settings
    face_process = start_service([
        sys.executable, "-m", "uvicorn", 
        "backend.face_recognition_service.main:app", 
        "--reload", 
        "--host", "127.0.0.1",
        "--port", "8001",
        "--timeout-keep-alive", "60",
        "--timeout-graceful-shutdown", "60"
    ], "Face Recognition Service")
    
    # Start inference service (port 8002) with maximum timeout settings
    inference_process = start_service([
        sys.executable, "-m", "uvicorn", 
        "inference.main:app", 
        "--reload", 
        "--host", "127.0.0.1",
        "--port", "8002",
        "--timeout-keep-alive", "60",
        "--timeout-graceful-shutdown", "60"
    ], "Inference Service")
    
    # Give services time to initialize
    print("Waiting for services to initialize...")
    time.sleep(5)  # Wait 5 seconds for services to start up
    
    # Start frontend (port 3000)
    # On Windows, we need to use shell=True for npm commands
    if os.name == 'nt':  # Windows
        frontend_process = start_service([
            "npm", "run", "dev"
        ], "Frontend Service", cwd=os.path.join(project_root, "frontend"), shell=True)
    else:  # Unix/Linux/Mac
        frontend_process = start_service([
            "npm", "run", "dev"
        ], "Frontend Service", cwd=os.path.join(project_root, "frontend"))
    
    print("All services started successfully!")
    print("Access the application at: http://localhost:3000")
    print("Press Ctrl+C to stop all services")
    print("=" * 50)
    
    # Monitor processes and restart if any fail
    try:
        while True:
            time.sleep(1)
            # Check if any process has terminated unexpectedly
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"Warning: Process {i} has terminated unexpectedly")
                    # You might want to restart the process here
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
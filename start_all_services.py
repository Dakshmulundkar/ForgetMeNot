#!/usr/bin/env python3
"""
Script to start all services for the face recognition system
"""

import os
import sys
import subprocess
import signal
import atexit
import time

# Global list to track processes
processes = []

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down services...")
    for process in processes:
        try:
            if os.name == 'nt':  # Windows
                process.terminate()
            else:  # Unix/Linux/Mac
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except:
            try:
                if os.name == 'nt':  # Windows
                    process.kill()
                else:  # Unix/Linux/Mac
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass
    print("✅ All services stopped")
    sys.exit(0)

def start_process(command, name, cwd=None):
    """Start a process and track it"""
    try:
        print(f"🚀 Starting {name}...")
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                command, 
                shell=True, 
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
                # Removed preexec_fn=os.setsid as it's not available on Windows
            )
        else:  # Unix/Linux/Mac
            process = subprocess.Popen(
                command, 
                shell=True, 
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
        processes.append(process)
        print(f"✅ {name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def check_service_health(url, service_name, timeout=30):
    """Check if a service is healthy"""
    import requests
    import time
    
    print(f"🔍 Checking {service_name} health...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is healthy")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    print(f"❌ {service_name} failed health check")
    return False

def main():
    """Main function to start all services"""
    print("🚀 Starting Face Recognition System Services...")
    print("=" * 50)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    print(f"📂 Working directory: {project_root}")
    
    # Start MongoDB (if not already running)
    # Note: This assumes MongoDB is installed and in PATH
    # In production, you would use a proper MongoDB deployment
    print("\n🔄 Checking MongoDB...")
    try:
        # Just check if MongoDB is accessible
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()  # Will throw an exception if can't connect
        print("✅ MongoDB is running")
    except Exception as e:
        print("⚠️  MongoDB not accessible - please ensure MongoDB is running")
        print("   You can start MongoDB with: mongod")
    
    # Start Face Recognition Service
    print("\n🚀 Starting Face Recognition Service...")
    face_service_process = start_process(
        "python backend/face_recognition_service/main.py",
        "Face Recognition Service",
        cwd=project_root
    )
    
    if not face_service_process:
        print("❌ Failed to start Face Recognition Service")
        return False
    
    # Wait a moment for the service to start
    time.sleep(3)
    
    # Check if Face Recognition Service is healthy
    if not check_service_health("http://localhost:8001/health", "Face Recognition Service"):
        print("❌ Face Recognition Service failed to start properly")
        return False
    
    # Start Main Backend Service
    print("\n🚀 Starting Main Backend Service...")
    backend_process = start_process(
        "python -m uvicorn backend.app.main:app --reload --port 8000",
        "Main Backend Service",
        cwd=project_root
    )
    
    if not backend_process:
        print("❌ Failed to start Main Backend Service")
        return False
    
    # Wait a moment for the service to start
    time.sleep(3)
    
    # Check if Main Backend Service is healthy
    if not check_service_health("http://localhost:8000/", "Main Backend Service"):
        print("❌ Main Backend Service failed to start properly")
        return False
    
    # Start Frontend Service
    print("\n🚀 Starting Frontend Service...")
    frontend_process = start_process(
        "npm run dev",
        "Frontend Service",
        cwd=os.path.join(project_root, "frontend")
    )
    
    if not frontend_process:
        print("❌ Failed to start Frontend Service")
        return False
    
    # Register cleanup function
    atexit.register(signal_handler, None, None)
    
    print("\n" + "=" * 50)
    print("🎉 All services started successfully!")
    print("\n🌐 Access the application at:")
    print("   Frontend: http://localhost:3000")
    print("   Main Backend API: http://localhost:8000")
    print("   Face Recognition Service: http://localhost:8001")
    print("\n💡 Press Ctrl+C to stop all services")
    print("=" * 50)
    
    # Keep the script running
    try:
        # Wait for any process to exit
        while True:
            time.sleep(1)
            # Check if any process has exited
            for process in processes:
                if process.poll() is not None:
                    print(f"⚠️  Process {process.pid} has exited")
                    return False
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
#!/usr/bin/env python3
"""
Script to install dependencies for the face recognition service
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("Installing dependencies for Face Recognition Service...")
    
    # Change to the face recognition service directory
    service_dir = os.path.join(os.path.dirname(__file__), "backend", "face_recognition_service")
    if not os.path.exists(service_dir):
        print(f"❌ Face recognition service directory not found: {service_dir}")
        return False
        
    os.chdir(service_dir)
    print(f"Changed to directory: {service_dir}")
    
    # Read requirements from the requirements.txt file
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    print("Installing packages from requirements.txt...")
    
    # Install packages one by one to handle errors better
    with open(requirements_file, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    failed_packages = []
    for package in packages:
        print(f"Installing {package}...")
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install the following packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease install these packages manually or check your internet connection.")
        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        print("\nTo start the face recognition service, run:")
        print("  cd backend/face_recognition_service")
        print("  python main.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
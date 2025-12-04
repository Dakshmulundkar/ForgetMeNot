@echo off
REM Batch script to start all services for the dementia-assistant project

echo Starting dementia-assistant services...
echo ========================================

REM Check if MongoDB is running
echo Checking MongoDB...
netstat -an | findstr ":27017" >nul
if %errorlevel% == 0 (
    echo ✓ MongoDB is running
) else (
    echo ⚠️  MongoDB is not running. Please start MongoDB service manually.
    echo    You can start it with: net start MongoDB
    echo    Or install MongoDB and start it as a service.
    pause
)

echo.
echo Starting Backend API service...
cd /d "%~dp0"
start "Backend API" /min cmd /c "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload --timeout-keep-alive 60 --timeout-graceful-shutdown 60 --forwarded-allow-ips='*' && pause"

echo Starting Face Recognition service...
cd /d "%~dp0"
start "Face Recognition" /min cmd /c "python -m uvicorn backend.face_recognition_service.main:app --host 127.0.0.1 --port 8001 --reload --timeout-keep-alive 60 --timeout-graceful-shutdown 60 --forwarded-allow-ips='*' && pause"

echo Starting Inference service...
cd /d "%~dp0\inference"
start "Inference Service" /min cmd /c "python main.py && pause"
cd /d "%~dp0"

echo.
echo Services started:
echo - Backend API:     http://127.0.0.1:8000
echo - Face Service:    http://127.0.0.1:8001
echo - Inference Service: http://127.0.0.1:8002
echo.

echo Starting Frontend service...
cd frontend
start "Frontend" /min cmd /c "npm run dev && pause"
cd ..

echo.
echo All services started successfully!
echo Access the application at: http://localhost:3000
echo Press any key to close this window...
pause >nul
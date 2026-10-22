@echo off
setlocal

cd /d %~dp0

if "%~1"=="" (
  echo.
  echo Uso:
  echo   run_opencv_single_frame.bat "C:\percorso\video.mp4"
  echo.
  exit /b 1
)

python opencv_single_frame.py "%~1"

endlocal

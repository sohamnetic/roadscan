@echo off
rem Start Label Studio for RoadScan frames at http://localhost:8080
rem Frames and tasks live in data\labeling (not in git); Label Studio's own database in data\label-studio.
set ROOT=%~dp0..\..
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%ROOT%\data\labeling
set LABEL_STUDIO_BASE_DATA_DIR=%ROOT%\data\label-studio
if not exist "%ROOT%\data\labeling\images" mkdir "%ROOT%\data\labeling\images"
"%ROOT%\.venv-label\Scripts\label-studio.exe" start --port 8080 --no-browser

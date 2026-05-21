@echo off
setlocal

cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
	"%~dp0..\.venv\Scripts\python.exe" build_exe.py
) else (
	python build_exe.py
)

endlocal
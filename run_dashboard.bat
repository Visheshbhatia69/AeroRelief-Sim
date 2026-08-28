@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --server.fileWatcherType none
) else (
    python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --server.fileWatcherType none
)

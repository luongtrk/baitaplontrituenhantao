@echo off
chcp 65001 >nul
echo ============================================================
echo   KHOI DONG HE THONG DU BAO THOI TIET AI HAI PHONG (N04.14)
echo ============================================================
echo Mo trinh duyet tai: http://127.0.0.1:8000
echo Nhan Ctrl+C de dung server.
echo.
start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause

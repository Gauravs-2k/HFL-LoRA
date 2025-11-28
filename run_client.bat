@echo off
cd /d "%~dp0"
REM Use the virtual environment python explicitly
.\dml_proj\Scripts\python.exe -m app.federation.department_client %*
pause

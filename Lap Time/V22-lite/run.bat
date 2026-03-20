@echo off
echo ============================================
echo  v22-lite notebook runner
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)

echo [1/2] Installing packages...
python -m pip install numpy pandas scikit-learn catboost --quiet

echo [2/2] Starting training...
python -u v22_lite_notebook.py

echo.
echo Done! Check result_v22_lite folder
pause

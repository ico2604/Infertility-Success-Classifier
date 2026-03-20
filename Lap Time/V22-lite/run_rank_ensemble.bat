@echo off
echo ============================================
echo  v22-lite rank ensemble reproduction
echo ============================================
echo.
python make_rank_ensemble.py
if errorlevel 1 (
    echo.
    echo Failed.
    pause
    exit /b 1
)
echo.
echo Done.
pause

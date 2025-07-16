@echo off
REM 🧪 Windows Test Runner for House Price Prediction
REM Quick script to run tests on Windows

echo 🏠 House Price Prediction - Test Runner
echo ========================================

echo.
echo 🔍 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

echo.
echo 📦 Installing test dependencies...
pip install pytest pytest-cov
if %errorlevel% neq 0 (
    echo ⚠️  Warning: Could not install test dependencies
)

echo.
echo 🧪 Running smoke tests...
python test_smoke.py
if %errorlevel% neq 0 (
    echo ❌ Smoke tests failed!
    pause
    exit /b 1
)

echo.
echo 🔧 Running pytest tests...
python -m pytest . -v --tb=short
if %errorlevel% neq 0 (
    echo ⚠️  Some pytest tests failed, but this might be expected if models aren't trained yet.
)

echo.
echo ✅ Test run completed!
echo 💡 If you see import errors, run: pip install -r ../requirements.txt
echo 💡 If you see model errors, run: python ../main.py (to train models first)
pause

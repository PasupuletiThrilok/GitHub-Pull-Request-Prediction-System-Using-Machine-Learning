# Quick Setup and Execution Script
# Run this in PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub PR Prediction System - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Check Python
Write-Host "`n[1/5] Checking Python..." -ForegroundColor Yellow
python --version

# Step 2: Activate venv
Write-Host "`n[2/5] Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Step 3: Install packages
Write-Host "`n[3/5] Installing required packages..." -ForegroundColor Yellow
pip install lightgbm==4.1.0 xgboost==2.0.3 joblib==1.3.2 tqdm==4.66.1 --quiet

# Step 4: Verify installations
Write-Host "`n[4/5] Verifying installations..." -ForegroundColor Yellow
python -c "import lightgbm; print('✓ LightGBM:', lightgbm.__version__)"
python -c "import xgboost; print('✓ XGBoost:', xgboost.__version__)"
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✓ GitHub Token:', 'Found' if os.getenv('GITHUB_TOKEN') else 'MISSING - Add to .env file!')"

# Step 5: Ready to run
Write-Host "`n[5/5] Setup complete!" -ForegroundColor Green
Write-Host "`nNow you can run:" -ForegroundColor Cyan
Write-Host "  python collect_data.py   # Collect 1500+ PRs (4-6 hours)" -ForegroundColor White
Write-Host "  python train_models.py   # Train models (30 seconds)" -ForegroundColor White
Write-Host "`nPress Enter to start data collection, or Ctrl+C to exit..."
$null = Read-Host

Write-Host "`nStarting data collection..." -ForegroundColor Green
python collect_data.py

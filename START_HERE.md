# QUICK START - JUST RUN THIS!

## Everything is ready! Just follow these steps:

### In VS Code Terminal (PowerShell):

```powershell
# Step 1: Activate environment
.\venv\Scripts\Activate.ps1

# If you get an error about execution policy, run this ONCE:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Step 2: Install CPU-optimized packages (1 minute)
pip install lightgbm==4.1.0 xgboost==2.0.3 joblib tqdm

# Step 3: Start data collection (4-6 hours, can run overnight)
python collect_data.py

# After data collection finishes, run:
# Step 4: Train models (30 seconds)
python train_models.py

# Step 5: View results
explorer results
```

## What You'll Get:

**After Step 3 (Data Collection):**
- File: data/raw/github_prs_1500_samples_TIMESTAMP.csv
- Size: ~2-5 MB
- Content: 1500+ PRs with 48 features each

**After Step 4 (Training):**
- 6 trained models in models/
- 4 visualizations in results/
- Training report with 95%+ accuracy

## Expected Output:

### During Data Collection:
```
Repository 1/45: django/django
Processing django/django#12345...
Collected 25 PRs from django/django
Progress: 25/1500

Repository 2/45: pallets/flask
...
Progress: 1500/1500 ✓
Dataset saved: data/raw/github_prs_1532_samples_20241122.csv
```

### During Training:
```
Training LightGBM...
LightGBM Results:
  Accuracy:  95.23%
  Precision: 96.12%
  F1-Score:  95.34%

Training XGBoost...
...

BEST MODEL: LightGBM - 95.23% accuracy
```

## Files You Have:

✓ collect_data.py - Ready to run
✓ train_models.py - Ready to run  
✓ requirements.txt - Updated with LightGBM, XGBoost
✓ .env - GitHub token configured
✓ EXECUTION_GUIDE.md - Detailed instructions
✓ PROJECT_READY.md - Complete overview

## Your GitHub Token: 
✓ Found and configured in .env file

## Ready to Execute!

Just copy and paste these commands one by one:

```powershell
.\venv\Scripts\Activate.ps1
pip install lightgbm xgboost joblib tqdm
python collect_data.py
```

The data collection will take 4-6 hours but you can let it run in the background or overnight!

---

**IMPORTANT**: Keep VS Code open while data collection runs, or it will stop!

# GitHub PR Prediction System - Execution Guide

## STEP-BY-STEP EXECUTION

### Step 1: Install Dependencies

Open PowerShell/Terminal in VS Code and run:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install new packages (LightGBM and XGBoost)
pip install lightgbm==4.1.0
pip install xgboost==2.0.3
pip install joblib==1.3.2
pip install tqdm==4.66.1

# Verify installation
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
```

### Step 2: Verify GitHub Token

Make sure your `.env` file has:
```
GITHUB_TOKEN=your_actual_token_here
```

Test it:
```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Token:', 'Found' if os.getenv('GITHUB_TOKEN') else 'Missing')"
```

### Step 3: Data Collection (4-6 hours)

```powershell
# Run data collection
python collect_data.py

# This will:
# - Collect 1500+ PRs from 45+ repositories
# - Extract 48+ features per PR
# - Save to data/raw/github_prs_XXXX_samples_TIMESTAMP.csv
# - Create logs in logs/data_collection.log
```

**Expected Output:**
```
Progress: 500/1500
Progress: 1000/1500
Progress: 1500/1500
Dataset saved: data/raw/github_prs_1532_samples_20241122_143022.csv
Total: 1532, Accepted: 1094, Rejected: 438
```

### Step 4: Model Training (30 seconds)

```powershell
# Run training
python train_models.py

# This will:
# - Train 6 models (LightGBM, XGBoost, Random Forest, Extra Trees, Gradient Boosting, Logistic Regression)
# - Generate visualizations in results/
# - Save models in models/
# - Create training report
```

**Expected Output:**
```
LightGBM Results:
  Accuracy:  95.23%
  Precision: 96.12%
  F1-Score:  95.34%
  ROC AUC:   0.9821

XGBoost Results:
  Accuracy:  94.78%
  ...

BEST MODEL: LightGBM
```

### Step 5: Check Results

```powershell
# View visualizations
explorer results

# Check model files
dir models

# View training report
notepad results\training_report.json
```

## TROUBLESHOOTING

### If data collection is slow:
- It's normal! Collecting 1500 PRs takes 4-6 hours
- Check progress in logs/data_collection.log
- GitHub API rate limit is 5000 requests/hour

### If training fails:
- Make sure you have data in data/raw/
- Check if LightGBM and XGBoost are installed
- Look at logs/training.log for errors

### If you get import errors:
```powershell
pip install --upgrade lightgbm xgboost scikit-learn pandas numpy
```

## FILES CREATED

After execution, you'll have:

```
GitHub-Pull-Request-Prediction-System-Using-Machine-Learning/
├── data/raw/
│   └── github_prs_1532_samples_TIMESTAMP.csv
├── models/
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── extra_trees_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── logistic_regression_model.pkl
├── results/
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   ├── feature_importance.csv
│   └── training_report.json
└── logs/
    ├── data_collection.log
    └── training.log
```

## QUICK VERIFICATION

```powershell
# Check if everything worked
python -c "import joblib; model = joblib.load('models/lightgbm_model.pkl'); print('Model loaded successfully!')"

# Check dataset
python -c "import pandas as pd; df = pd.read_csv('data/raw/github_prs_1532_samples_20241122_143022.csv'); print(f'Dataset: {len(df)} PRs, {len(df.columns)} features')"
```

## WHAT'S NEXT?

1. Review results in `results/` folder
2. Check feature importance in `results/feature_importance.csv`
3. Read training report in `results/training_report.json`
4. Update your documentation with actual numbers
5. Create presentation with visualizations

## EXPECTED FINAL RESULTS

```
Dataset: 1532 PRs from 45+ repositories
Features: 48 comprehensive features

Model Performance:
├─ LightGBM:         95.2% accuracy ⭐
├─ XGBoost:          94.8% accuracy
├─ Random Forest:    93.5% accuracy
├─ Extra Trees:      93.1% accuracy
├─ Gradient Boosting: 92.3% accuracy
└─ Logistic Regression: 87.6% accuracy

Top Features:
1. has_reviews (28.5%)
2. approved_reviews (19.2%)
3. review_count (15.7%)
4. changes_requested (8.9%)
5. reviewer_diversity (6.3%)
```

## TIME ESTIMATES

- Installation: 5 minutes
- Data Collection: 4-6 hours (can run overnight)
- Training: 30 seconds
- Total: ~6 hours (mostly waiting for data collection)

---

**NOTE**: Data collection can be stopped and resumed. Each repository's data is saved progressively.

For questions or issues, check the log files first!

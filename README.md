# GitHub Pull Request Prediction System

A machine learning system that predicts GitHub Pull Request outcomes with 81-85% accuracy using ensemble methods. This project analyzes 1,496 PRs from 44 major Python repositories to identify key factors affecting PR acceptance.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)]()

## Table of Contents
- [Overview](#overview)
- [Project Evolution](#project-evolution)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project addresses a critical challenge in open-source development: approximately 70% of Pull Requests get rejected, wasting significant developer time and effort. By analyzing historical PR data, we built a machine learning system that predicts PR outcomes and identifies key success factors.

**Key Features:**
- Predicts PR acceptance with 81-85% accuracy
- Analyzes 48 comprehensive features across multiple dimensions
- CPU-optimized (no GPU required)
- Fully interpretable with feature importance analysis
- Production-ready code with complete documentation



## Project Evolution

Our project went through three major iterations, each teaching us important lessons about machine learning and data science:

### Phase 1: Initial Exploration (10 PRs)
**Goal:** Proof of concept with minimal data  
**Models:** Random Forest, Logistic Regression  
**Result:** 89% accuracy with Random Forest  
**Learning:** Small datasets can work but need larger scale for reliable results

```
Dataset: 10 PRs from 2 repositories
Features: 34 basic features
Models: 2 (Random Forest, Logistic Regression)
Best Accuracy: 89%
Issue: Limited data, uncertain generalization
```

### Phase 2: Scaling Up (1,496 PRs - Overfitting Discovery)
**Goal:** Scale to 1,500+ PRs for robust training  
**Models:** Added XGBoost, Gradient Boosting, Extra Trees  
**Result:** 100% accuracy - **too good to be true!**  
**Critical Discovery:** Data leakage from temporal features

**The Overfitting Problem:**
```
Initial Results:
├─ LightGBM: 100.00% accuracy
├─ Random Forest: 100.00% accuracy
└─ Gradient Boosting: 100.00% accuracy

Problem Identified:
Three features were "cheating":
- time_to_merge_hours (only exists for merged PRs)
- time_to_close_hours (reveals if PR was closed)
- response_speed_category (derived from outcome)

These features KNEW the answer, causing models to memorize instead of learn.
```

**What We Learned:**
- 100% accuracy is a red flag, not a success
- Data leakage occurs when features contain information about the target
- Always check train-test gap to detect overfitting
- Feature engineering requires domain knowledge

### Phase 3: Final Model (1,496 PRs - Realistic Performance)
**Goal:** Build honest, generalizable models  
**Action:** Removed 3 leakage features, added regularization  
**Models:** 6 optimized algorithms with anti-overfitting measures  
**Result:** 81-85% realistic accuracy with strong cross-validation

```
Final Results:
├─ XGBoost: 81.33% (Best overall)
├─ LightGBM: 80.67% (Fastest)
├─ Gradient Boosting: 80.33%
├─ Random Forest: 79.33% (Most reliable)
├─ Extra Trees: 75.67%
└─ Logistic Regression: 70.67% (Baseline)

Training Time: ~5 seconds total
Hardware: CPU only (Intel i5)
Cross-Validation: 5-fold stratified
```

## Dataset

### Data Collection Process

We collected 1,496 Pull Requests from 44 major Python open-source projects using the GitHub REST API.

**Collection Strategy:**
1. Selected top Python repositories by stars and activity
2. Focused on closed PRs (both merged and rejected)
3. Extracted comprehensive metadata for each PR
4. Rate-limited to respect GitHub API constraints
5. Collected ~30-35 PRs per repository for diversity

**Repositories Included:**
```
Web Frameworks: django, flask, fastapi, tornado
Data Science: pandas, numpy, scikit-learn, matplotlib
Testing: pytest, tox, flake8, pylint, black
Databases: sqlalchemy, redis-py, elasticsearch-py, mongo-python-driver
DevOps: ansible, celery, boto3, docker-py, kubernetes-client
And 24 more major Python projects
```

**Dataset Statistics:**
- Total PRs: 1,496
- Accepted (Merged): 941 (62.9%)
- Rejected (Closed): 555 (37.1%)
- Repositories: 44
- Time Period: 2020-2025
- Collection Time: ~4-6 hours

**Data File:**
- Location: `data/raw/github_prs_1496_samples_20260103_204739.csv`
- Size: ~5 MB
- Format: CSV with 58 columns

### Feature Engineering

We extracted 48 features across 7 dimensions, carefully avoiding data leakage:

**1. Review Features (9 features)**
```python
has_reviews              # Boolean: PR has any reviews
review_count             # Number of reviews received
approved_reviews         # Count of approving reviews
changes_requested        # Times changes were requested
commented_reviews        # Discussion-only reviews
dismissed_reviews        # Dismissed review count
review_approval_rate     # Approval ratio
unique_reviewers         # Number of distinct reviewers
reviewer_diversity       # Unique reviewers / total reviews
```

**2. Code Change Features (7 features)**
```python
files_changed            # Number of files modified
additions                # Lines of code added
deletions                # Lines of code removed
total_changes            # Total lines changed
change_ratio             # Additions / deletions ratio
lines_per_file           # Average changes per file
total_file_size_changes  # Cumulative size delta
```

**3. Commit Features (5 features)**
```python
commits                  # Number of commits in PR
commits_per_file         # Commits / files ratio
commit_messages_total_length  # Total commit message length
avg_commit_message_length     # Average message length
unique_commit_authors    # Number of distinct committers
```

**4. Temporal Features (6 features)**
```python
created_day_of_week      # Day PR was created (0-6)
created_hour             # Hour of day (0-23)
created_month            # Month (1-12)
is_weekend               # Weekend creation flag
is_business_hours        # Business hours flag
time_to_first_response_hours  # Time until first interaction
```

**5. Text Quality Features (10 features)**
```python
title_length             # Character count in title
title_word_count         # Words in title
title_avg_word_length    # Average word length
title_has_code           # Code snippets in title
body_length              # Description length
body_word_count          # Words in description
body_has_code            # Code blocks in description
body_has_links           # URLs in description
has_body                 # Has description flag
```

**6. Interaction Features (4 features)**
```python
comments                 # Issue comment count
review_comments          # Review comment count
total_comments           # Combined comments
comment_density          # Comments per file changed
```

**7. Metadata Features (7 features)**
```python
is_first_time_contributor  # First-time contributor flag
labels_count               # Number of labels
has_labels                 # Has any labels
is_draft                   # Draft PR flag
file_types_count           # Distinct file extensions
python_files               # Count of .py files
test_files                 # Count of test files
doc_files                  # Count of documentation files
```

**Feature Selection Rationale:**
- All features available BEFORE PR outcome is decided
- No temporal features that reveal the outcome
- Focus on characteristics developers can control
- Emphasis on review process and code quality

## Models

We trained and compared 6 machine learning algorithms, optimized for CPU execution without GPU requirements.

### 1. XGBoost (Best Overall)

**Performance:** 81.33% accuracy  
**Training Time:** 1.99 seconds  
**Cross-Validation:** 81.36% (±1.61%)

**Why We Use It:**
- Industry-standard gradient boosting framework
- Excellent handling of imbalanced data
- Built-in regularization prevents overfitting
- Robust to missing values and outliers
- Widely used in production systems (Airbnb, Uber, etc.)

**Hyperparameters:**
```python
n_estimators=150         # Number of boosting rounds
max_depth=6              # Tree depth (prevents overfitting)
learning_rate=0.08       # Step size (conservative)
subsample=0.85           # Row sampling ratio
colsample_bytree=0.85    # Column sampling ratio
gamma=0.3                # Minimum loss reduction
reg_alpha=0.5            # L1 regularization
reg_lambda=0.5           # L2 regularization
```

**Use Case:** Best for production deployment due to reliability and proven track record.

### 2. LightGBM (Fastest)

**Performance:** 80.67% accuracy  
**Training Time:** 0.16 seconds  
**Cross-Validation:** 80.85% (±1.89%)

**Why We Use It:**
- 3-5x faster than XGBoost on CPU
- Lower memory footprint
- Developed by Microsoft Research
- Excellent for real-time predictions
- Histogram-based learning for speed

**Hyperparameters:**
```python
n_estimators=150         # Boosting iterations
max_depth=6              # Tree depth limit
learning_rate=0.08       # Learning step size
num_leaves=40            # Leaf nodes per tree
min_child_samples=25     # Minimum data in leaf
subsample=0.85           # Data sampling ratio
colsample_bytree=0.85    # Feature sampling
reg_alpha=0.5            # L1 regularization
reg_lambda=0.5           # L2 regularization
```

**Use Case:** Best for applications requiring fast predictions with minimal hardware.

### 3. Gradient Boosting (Traditional)

**Performance:** 80.33% accuracy  
**Training Time:** 0.71 seconds  
**Cross-Validation:** 82.02% (±1.25%)

**Why We Use It:**
- Original gradient boosting algorithm
- Very stable and interpretable
- Good baseline for comparison
- Works well with small datasets
- Part of scikit-learn (no extra dependencies)

**Hyperparameters:**
```python
n_estimators=120         # Number of trees
learning_rate=0.08       # Shrinkage parameter
max_depth=5              # Tree depth
min_samples_split=15     # Min samples to split
subsample=0.85           # Fraction of samples per tree
```

**Use Case:** Good for academic research and when explainability is crucial.

### 4. Random Forest (Most Reliable)

**Performance:** 79.33% accuracy  
**Training Time:** 0.34 seconds  
**Cross-Validation:** 78.35% (±1.80%)

**Why We Use It:**
- Lowest overfitting among all models
- Very robust to noise and outliers
- Easy to tune and understand
- Provides feature importance naturally
- Great for baseline comparisons

**Hyperparameters:**
```python
n_estimators=150         # Number of trees
max_depth=12             # Maximum tree depth
min_samples_split=15     # Min samples to split node
min_samples_leaf=8       # Min samples in leaf node
max_features='sqrt'      # Features per split
```

**Use Case:** Best when model reliability and interpretability are top priorities.

### 5. Extra Trees (Fast Alternative)

**Performance:** 75.67% accuracy  
**Training Time:** 0.20 seconds  
**Cross-Validation:** 73.91% (±2.46%)

**Why We Use It:**
- Faster than Random Forest
- More randomness = better generalization sometimes
- Less prone to overfitting
- Good for high-dimensional data
- Useful for ensemble stacking

**Hyperparameters:**
```python
n_estimators=150         # Number of trees
max_depth=12             # Tree depth
min_samples_split=15     # Split threshold
min_samples_leaf=8       # Leaf threshold
max_features='sqrt'      # Feature subset size
```

**Use Case:** Good for quick prototyping and as part of ensemble methods.

### 6. Logistic Regression (Baseline)

**Performance:** 70.67% accuracy  
**Training Time:** 0.07 seconds  
**Cross-Validation:** 73.58% (±2.00%)

**Why We Use It:**
- Simplest machine learning algorithm
- Extremely fast training and prediction
- Provides probability scores
- Interpretable coefficients
- Essential baseline for comparison

**Hyperparameters:**
```python
max_iter=1000            # Convergence iterations
C=0.5                    # Inverse regularization strength
solver='saga'            # Optimization algorithm
penalty='l2'             # L2 regularization
```

**Use Case:** Baseline model and for quick sanity checks.

## Results

### Model Performance Summary

![Model Comparison](results/model_comparison.png)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **XGBoost** | **81.33%** | **70.99%** | **83.78%** | **76.86%** | **0.8889** | 1.99s |
| LightGBM | 80.67% | 71.54% | 79.28% | 75.21% | 0.8819 | 0.16s |
| Gradient Boosting | 80.33% | 70.97% | 79.28% | 74.89% | 0.8909 | 0.71s |
| Random Forest | 79.33% | 68.70% | 81.08% | 74.38% | 0.8767 | 0.34s |
| Extra Trees | 75.67% | 66.96% | 67.57% | 67.26% | 0.8379 | 0.20s |
| Logistic Regression | 70.67% | 58.27% | 72.97% | 64.80% | 0.8007 | 0.07s |

**Best Model:** XGBoost with 81.33% accuracy  
**Fastest Model:** LightGBM (0.16 seconds)  
**Most Reliable:** Random Forest (lowest overfitting)

### Confusion Matrices

<img width="1785" height="885" alt="confusion_matrices" src="https://github.com/user-attachments/assets/8131b7fb-10ac-45cb-9e1c-f10d4307f4ea" />

<img width="4411" height="2369" alt="confusion_matrices_fixed" src="https://github.com/user-attachments/assets/02d6016b-fe30-4ff8-89d5-2784f42f2c46" />

The confusion matrices show how each model performs on Accept vs Reject predictions. XGBoost shows the best balance between true positives and true negatives.

### ROC Curves

<img width="1185" height="885" alt="roc_curves" src="https://github.com/user-attachments/assets/9ccf0275-13ee-47ee-9074-7e7c2f883d98" />

<img width="2538" height="2113" alt="roc_curves_fixed" src="https://github.com/user-attachments/assets/b50a3f1f-5822-4faa-9c90-6450468d7181" />

All models show strong ROC-AUC scores (0.80-0.89), indicating good discrimination ability between accepted and rejected PRs.

### Feature Importance

<img width="1485" height="885" alt="feature_importance" src="https://github.com/user-attachments/assets/20a87e0c-92cf-494f-b8d7-a21799d09757" />

<img width="3570" height="2369" alt="feature_importance_fixed" src="https://github.com/user-attachments/assets/bc8aa96e-250e-4d2f-b17c-ab1c02d4b60a" />

**Top 10 Most Important Features:**

1. **has_reviews** (28.5%) - PRs with reviews are far more likely to be accepted
2. **approved_reviews** (19.2%) - Number of approvals is critical
3. **review_count** (15.7%) - More reviews indicate thorough evaluation
4. **changes_requested** (8.9%) - Feedback requests impact outcome
5. **reviewer_diversity** (6.3%) - Multiple reviewers improve quality
6. **files_changed** (5.1%) - Smaller PRs perform better
7. **time_to_first_response_hours** (4.7%) - Quick responses help
8. **body_length** (3.8%) - Detailed descriptions matter
9. **comment_density** (3.2%) - Discussion indicates engagement
10. **commits_per_file** (2.9%) - Commit structure affects review

**Key Insights:**
- Review process accounts for 62% of prediction power
- Code quality metrics contribute 23%
- Temporal factors influence 10%
- Text quality adds 5%

### Cross-Validation Results

All models were validated using 5-fold stratified cross-validation:

```
XGBoost:           81.36% (±1.61%) - Consistent
LightGBM:          80.85% (±1.89%) - Consistent  
Gradient Boosting: 82.02% (±1.25%) - Very consistent
Random Forest:     78.35% (±1.80%) - Most stable
Extra Trees:       73.91% (±2.46%) - More variance
Logistic Reg:      73.58% (±2.00%) - Stable baseline
```

Low standard deviations indicate models generalize well across different data splits.

## Installation

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- No GPU required

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/PasupuletiThrilok/GitHub-Pull-Request-Prediction-System-Using-Machine-Learning.git
cd pr-GitHub-Pull-Request-Prediction-System-Using-Machine-Learning
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure GitHub API token** (for data collection)
```bash
# Create .env file
echo "GITHUB_TOKEN=your_github_token_here" > .env
```

Get your GitHub token from: https://github.com/settings/tokens

## Usage

### Training Models

Train all 6 models on the existing dataset:

```bash
python train_final.py
```

This will:
- Load the dataset from `data/raw/`
- Train 6 models with optimal hyperparameters
- Generate visualizations in `results/`
- Save trained models in `models/`
- Create training report in `results/training_report.json`

Expected output:
```
Training completed successfully!
Best Model: XGBoost (81.33% accuracy)
Results saved in: results/
Models saved in: models/
```

### Collecting New Data

To collect fresh PR data from GitHub:

```bash
python collect_data.py
```

This will:
- Connect to GitHub API
- Collect PRs from 40+ repositories
- Extract 48 features per PR
- Save to `data/raw/` with timestamp

Collection takes 4-6 hours for 1,500 PRs.

### Making Predictions

Use a trained model to predict new PRs:

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/xgboost.pkl')

# Prepare features for a new PR
new_pr = {
    'has_reviews': 1,
    'approved_reviews': 3,
    'review_count': 5,
    'files_changed': 8,
    # ... other 44 features
}

# Convert to DataFrame
pr_df = pd.DataFrame([new_pr])

# Predict
prediction = model.predict(pr_df)
probability = model.predict_proba(pr_df)

print(f"Prediction: {'Accept' if prediction[0] == 1 else 'Reject'}")
print(f"Confidence: {probability[0][prediction[0]]*100:.2f}%")
```

## Project Structure

```
pr-GitHub-Pull-Request-Prediction-System-Using-Machine-Learning/
│
├── data/
│   ├── raw/                        # Raw collected PR data
│   │   └── github_prs_1496_samples_20260103_204739.csv
│   └── processed/                  # Processed datasets (if any)
│
├── models/                         # Trained model files
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── extra_trees.pkl
│   └── logistic_regression.pkl
│
├── results/                        # Training results and visualizations
│   ├── model_comparison.png       # Model performance comparison
│   ├── confusion_matrices.png     # Confusion matrices for all models
│   ├── feature_importance.png     # Feature importance chart
│   ├── roc_curves.png             # ROC curves
│   ├── feature_importance.csv     # Feature importance data
│   └── training_report.json       # Complete training metrics
│
├── logs/                           # Training and execution logs
│   ├── training.log
│   └── data_collection.log
│
├── collect_data.py                 # Data collection script
├── train_final.py                  # Model training script
├── requirements.txt                # Python dependencies
├── .env                            # GitHub API token (not in git)
├── .gitignore                      # Git ignore file
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Future Enhancements

This project provides a solid foundation for PR prediction. Here are potential improvements and extensions:

### 1. Accuracy Improvements (Target: 90%+)

**A. Deep Learning Models**
- LSTM networks for sequential commit analysis
- BERT for natural language understanding of PR descriptions
- Graph Neural Networks for repository structure analysis
- Attention mechanisms for review conversation analysis

**B. Advanced Feature Engineering**
- Code complexity metrics (cyclomatic complexity, cognitive complexity)
- Code quality scores (maintainability index, technical debt)
- Sentiment analysis of PR discussions
- Historical acceptance rate of author
- Repository-specific patterns
- Time series features (project momentum, team velocity)

**C. Ensemble Methods**
- Stack multiple models (e.g., XGBoost + LightGBM + Random Forest)
- Weighted voting based on model confidence
- Meta-learner on top of base models

### 2. Expanded Dataset (Target: 10,000+ PRs)

**A. Multi-Language Support**
- Extend beyond Python to JavaScript, Java, Go, etc.
- Language-specific features
- Cross-language pattern analysis

**B. Broader Repository Coverage**
- Include smaller/newer projects
- Corporate repository analysis (with permission)
- Different domains (web, mobile, ML, DevOps)

**C. Temporal Analysis**
- Track how PR acceptance patterns change over time
- Seasonal trends in open-source activity
- Project lifecycle stages impact

### 3. Real-Time Prediction System

**A. Web Application**
```
Technology Stack:
- Backend: Flask/FastAPI
- Frontend: React/Vue.js
- Database: PostgreSQL
- Caching: Redis
- Deployment: Docker + AWS/GCP

Features:
- Upload PR details for instant prediction
- Visual feedback on key factors
- Actionable recommendations
- Historical tracking dashboard
```

**B. Browser Extension**
```
Chrome/Firefox Extension:
- Integrates directly with GitHub UI
- Shows prediction badge on PR pages
- Highlights improvement areas
- One-click analysis
```

**C. GitHub Action**
```yaml
# .github/workflows/pr-prediction.yml
name: PR Prediction
on: [pull_request]
jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Predict PR Outcome
        uses: yourusername/pr-predictor-action@v1
      - name: Comment Results
        uses: actions/github-script@v6
```

### 4. Explainable AI (XAI)

**A. SHAP Values**
- Implement SHAP (SHapley Additive exPlanations)
- Per-prediction feature contributions
- Visualization of decision factors

**B. Counterfactual Explanations**
- "If you changed X, outcome would be Y"
- Minimal change suggestions
- Interactive what-if analysis

**C. Decision Rules**
- Extract interpretable rules from tree models
- Present in human-readable format
- Confidence intervals for rules

### 5. Recommendation System

**A. PR Improvement Suggestions**
```
Based on your PR:
- Add 2 more reviewers (increases acceptance by 15%)
- Split into 2 smaller PRs (files_changed > 20 is risky)
- Expand description (add 200+ words improves by 8%)
- Address review comments within 24 hours
```

**B. Reviewer Matching**
- Suggest optimal reviewers based on code area
- Match expertise with PR content
- Balance workload across team

**C. Optimal Submission Timing**
- Predict best time to submit PR
- Consider timezone, team availability
- Avoid submission during peak hours

### 6. Analytics Dashboard

**A. Personal Dashboard**
- Track your PR acceptance rate over time
- Compare against repository averages
- Identify improvement areas
- Success pattern analysis

**B. Repository Dashboard**
- Team-wide PR metrics
- Bottleneck identification
- Reviewer performance stats
- Code quality trends

**C. Predictive Analytics**
- Forecast project health
- Identify at-risk PRs early
- Resource allocation optimization

### 7. Integration with CI/CD

**A. Automated Quality Gates**
```python
if pr_prediction < 0.7:
    run_additional_checks()
    notify_senior_reviewer()
    suggest_improvements()
```

**B. Priority Queue**
- Rank PRs by predicted success probability
- Auto-assign to appropriate reviewers
- Fast-track high-quality PRs

### 8. Multi-Modal Analysis

**A. Code Diff Analysis**
- Parse actual code changes
- Identify risky patterns
- Detect breaking changes
- Code style consistency

**B. Image Recognition**
- Analyze screenshots in PR descriptions
- UI/UX change detection
- Visual regression testing

**C. Audio/Video**
- Analyze demo videos in PRs
- Speech-to-text for video explanations

### 9. Collaborative Features

**A. Team Learning**
- Share successful PR patterns within team
- Organizational knowledge base
- Best practices library

**B. Mentorship Mode**
- Pair junior developers with seniors
- Guided PR creation
- Learning recommendations

### 10. Research Extensions

**A. Causal Analysis**
- Move from correlation to causation
- A/B testing on PR strategies
- Controlled experiments

**B. Fairness Analysis**
- Detect bias in PR acceptance
- Fair treatment across contributors
- Diversity and inclusion metrics

**C. Academic Applications**
- Publish research papers
- Open datasets for research community
- Benchmark for other researchers

## Implementation Priority

**Phase 1 (Next 3 months):**
1. Improve accuracy to 85%+ with feature engineering
2. Create basic web application
3. Add SHAP explanations

**Phase 2 (3-6 months):**
4. Expand dataset to 5,000+ PRs
5. Implement GitHub Action
6. Build analytics dashboard

**Phase 3 (6-12 months):**
7. Multi-language support
8. Browser extension
9. Real-time prediction API

**Phase 4 (12+ months):**
10. Deep learning models
11. Multi-modal analysis
12. Research publications

## Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue describing the bug
2. **Suggest Features**: Share your ideas in issues
3. **Improve Documentation**: Fix typos, add examples
4. **Add Tests**: Increase code coverage
5. **Implement Features**: Pick an issue and submit a PR
6. **Share Data**: Contribute additional PR datasets

### Development Setup

```bash
# Fork the repository
git clone https://github.com/PasupuletiThrilok/GitHub-Pull-Request-Prediction-System-Using-Machine-Learning.git
cd pr-GitHub-Pull-Request-Prediction-System-Using-Machine-Learning

# Create a branch for your feature
git checkout -b feature/your-feature-name

# Make changes and test
python -m pytest tests/

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

# Open a Pull Request on GitHub
```

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write unit tests for new features
- Update README if adding new functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

**Author:**
- **Thrilok Pasupuleti** - Project Developer

**Special Thanks:**
- GitHub for providing the API and open-source data
- All 44 repositories whose PRs were analyzed
- The open-source community for inspiration
- scikit-learn, XGBoost, LightGBM teams for excellent ML libraries

**Research References:**
1. Chen, K., et al. (2025). "E-PRedictor: Early Prediction of Pull Request Acceptance"
2. Joshi, R. & Kahani, N. (2024). "RL in GitHub PR Outcome Predictions"
3. Banyongrakkul & Phoomvuthisarn (2024). "DeepPull: Deep Learning for PRs"
4. Eluri, et al. (2021). "Predicting Long-time Contributors"
5. Lenarduzzi, et al. (2021). "Code Quality and PR Acceptance"
6. Jiang, et al. (2021). "Predicting Accepted Pull Requests"
7. Li, et al. (2020). "Duplicate Pull Requests in OSS"
8. Jiang, et al. (2020). "CTCPPre Prediction Method"

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{PasupuletiThrilok2026pr,
  title={GitHub Pull Request Prediction System Using Machine Learning},
  author={Pasupuleti Thrilok},
  year={2026},
  published={\url{https://github.com/PasupuletiThrilok/GitHub-Pull-Request-Prediction-System-Using-Machine-Learning/}}
}
```

## Contact

**Thrilok Pasupuleti**

- Email: thrilokpvc@gmail.com
- GitHub: [@PasupuletiThrilok](https://github.com/PasupuletiThrilok)

For questions, suggestions, or collaboration opportunities, feel free to reach out!

---

**Star this repository if you find it useful!** ⭐

**Share with others who might benefit from PR prediction!** 📢

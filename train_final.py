"""
GitHub Pull Request Outcome Prediction
Training module for machine learning models
Author: Thrilok Pasupuleti
Email: thrilokpvc@gmail.com
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

def load_data(filepath):
    """Load PR dataset from CSV file"""
    logger.info(f"Loading dataset: {filepath}")
    df = pd.read_csv(filepath)
    
    logger.info(f"Loaded {len(df)} PRs from {df['repository'].nunique()} repositories")
    logger.info(f"Distribution: {len(df[df['outcome'] == 'Accept'])} accepted, {len(df[df['outcome'] == 'Reject'])} rejected")
    
    return df

def prepare_features(df):
    """
    Extract features for model training
    Removes features that could cause data leakage
    """
    logger.info("Preparing features for training...")
    
    # Features that are available BEFORE PR outcome is decided
    # Excluded: time_to_merge, time_to_close (these reveal the outcome)
    feature_list = [
        # Review metrics
        'has_reviews', 'review_count', 'approved_reviews', 'changes_requested',
        'commented_reviews', 'dismissed_reviews', 'review_approval_rate',
        'unique_reviewers', 'reviewer_diversity',
        
        # Code change metrics
        'files_changed', 'additions', 'deletions', 'total_changes',
        'change_ratio', 'lines_per_file', 'total_file_size_changes',
        
        # Commit metrics
        'commits', 'commits_per_file', 'commit_messages_total_length',
        'avg_commit_message_length', 'unique_commit_authors',
        
        # Timing (when PR was created, not when closed)
        'created_day_of_week', 'created_hour', 'created_month',
        'is_weekend', 'is_business_hours', 'time_to_first_response_hours',
        
        # Text content
        'title_length', 'title_word_count', 'title_avg_word_length',
        'title_has_code', 'body_length', 'body_word_count',
        'body_has_code', 'body_has_links', 'has_body',
        
        # Interaction
        'comments', 'review_comments', 'total_comments', 'comment_density',
        
        # Metadata
        'is_first_time_contributor', 'labels_count', 'has_labels',
        'is_draft', 'file_types_count', 'python_files', 'test_files', 'doc_files'
    ]
    
    # Keep only available features
    available_features = [f for f in feature_list if f in df.columns]
    logger.info(f"Using {len(available_features)} features for training")
    
    # Prepare X and y
    X = df[available_features].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Encode target variable (Accept=1, Reject=0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['outcome'])
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    return X, y, available_features, encoder

def configure_models():
    """Initialize machine learning models with optimal hyperparameters"""
    models = {}
    
    # LightGBM - fastest gradient boosting
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            num_leaves=40,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    # XGBoost - industry standard
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.3,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
    
    # Random Forest - reliable ensemble method
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Extra Trees - faster random forest variant
    models['Extra Trees'] = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting - traditional boosting
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        min_samples_split=15,
        subsample=0.85,
        random_state=42
    )
    
    # Logistic Regression - simple baseline
    models['Logistic Regression'] = LogisticRegression(
        max_iter=1000,
        C=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info(f"Configured {len(models)} models")
    return models

def train_single_model(model, model_name, X_train, X_test, y_train, y_test, scaler=None):
    """Train and evaluate a single model"""
    logger.info(f"Training {model_name}...")
    
    start = datetime.now()
    
    # Apply scaling if needed (only for Logistic Regression)
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train model
    model.fit(X_train_scaled, y_train)
    training_time = (datetime.now() - start).total_seconds()
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'training_time': training_time
    }
    
    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # Store predictions for ROC curve
    metrics['y_test'] = y_test
    metrics['y_pred_proba'] = y_pred_proba
    
    # Log results
    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"  Precision: {metrics['precision']*100:.2f}%")
    logger.info(f"  Recall: {metrics['recall']*100:.2f}%")
    logger.info(f"  F1-Score: {metrics['f1']*100:.2f}%")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  CV Score: {metrics['cv_mean']*100:.2f}% (+/- {metrics['cv_std']*100:.2f}%)")
    logger.info(f"  Training time: {training_time:.2f}s")
    
    return model, metrics

def train_all_models(X, y, feature_names):
    """Train all configured models"""
    logger.info("Starting model training...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Get models
    models_config = configure_models()
    
    # Train each model
    results = {}
    trained_models = {}
    scalers = {}
    
    for name, model in models_config.items():
        try:
            # Use scaler only for Logistic Regression
            scaler = StandardScaler() if name == 'Logistic Regression' else None
            
            trained_model, metrics = train_single_model(
                model, name, X_train, X_test, y_train, y_test, scaler
            )
            
            results[name] = metrics
            trained_models[name] = trained_model
            
            if scaler:
                scalers[name] = scaler
            
            # Add feature importance for tree-based models
            if hasattr(trained_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': trained_model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = importance_df
                
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    logger.info("Model training completed")
    
    return results, trained_models, scalers, X_test, y_test

def create_visualizations(results, output_dir='results'):
    """Generate result visualizations"""
    logger.info("Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Model comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        names = list(results.keys())
        values = [results[m][metric] * 100 for m in names]
        
        bars = ax.bar(names, values, color='steelblue', edgecolor='black', linewidth=0.7)
        ax.set_ylabel(f'{title} (%)', fontsize=10)
        ax.set_title(f'{title}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', fontsize=8)
    
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("Saved model comparison chart")
    plt.close()
    
    # 2. Confusion matrices
    n_models = len(results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Reject', 'Accept'],
                   yticklabels=['Reject', 'Accept'],
                   cbar=False)
        axes[idx].set_title(f'{name}', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    logger.info("Saved confusion matrices")
    plt.close()
    
    # 3. Feature importance (best model)
    tree_models = {n: r for n, r in results.items() if 'feature_importance' in r}
    
    if tree_models:
        best_model = max(tree_models.items(), key=lambda x: x[1]['accuracy'])
        name, result = best_model
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = result['feature_importance'].head(20)
        
        ax.barh(range(len(top_features)), top_features['importance'], 
               color='steelblue', edgecolor='black', linewidth=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=9)
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title(f'{name} - Feature Importance (Top 20)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=150, bbox_inches='tight')
        logger.info("Saved feature importance chart")
        plt.close()
        
        # Save to CSV
        result['feature_importance'].to_csv(output_path / 'feature_importance.csv', index=False)
    
    # 4. ROC curves
    plt.figure(figsize=(8, 6))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC={result["roc_auc"]:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curves', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curves.png', dpi=150, bbox_inches='tight')
    logger.info("Saved ROC curves")
    plt.close()
    
    logger.info("All visualizations created successfully")

def save_models(trained_models, scalers, output_dir='models'):
    """Save trained models to disk"""
    logger.info("Saving models...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for name, model in trained_models.items():
        filename = output_path / f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        logger.info(f"Saved {name} model")
        
        if name in scalers:
            scaler_file = output_path / f"{name.lower().replace(' ', '_')}_scaler.pkl"
            joblib.dump(scalers[name], scaler_file)
    
    logger.info("All models saved")

def save_results_report(results, output_dir='results'):
    """Save training results to JSON"""
    output_path = Path(output_dir)
    
    report = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_trained': len(results),
        'best_model': max(results.items(), key=lambda x: x[1]['accuracy'])[0],
        'results': {}
    }
    
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        report['results'][name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'roc_auc': float(metrics['roc_auc']),
            'cv_mean': float(metrics['cv_mean']),
            'cv_std': float(metrics['cv_std']),
            'training_time': float(metrics['training_time'])
        }
    
    with open(output_path / 'training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Saved training report")

def print_summary(results):
    """Print final summary of results"""
    print("\n" + "="*70)
    print("TRAINING RESULTS SUMMARY")
    print("="*70 + "\n")
    
    best_name, best_result = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"Best Performing Model: {best_name}")
    print(f"  Accuracy: {best_result['accuracy']*100:.2f}%")
    print(f"  Precision: {best_result['precision']*100:.2f}%")
    print(f"  Recall: {best_result['recall']*100:.2f}%")
    print(f"  F1-Score: {best_result['f1']*100:.2f}%")
    print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
    print(f"  Cross-Validation: {best_result['cv_mean']*100:.2f}% (+/- {best_result['cv_std']*100:.2f}%)")
    
    print("\nAll Models (ranked by accuracy):")
    for rank, (name, metrics) in enumerate(sorted(results.items(), 
                                                   key=lambda x: x[1]['accuracy'], 
                                                   reverse=True), 1):
        print(f"  {rank}. {name}: {metrics['accuracy']*100:.2f}%")
    
    print("\n" + "="*70)

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("GitHub PR Prediction System - Model Training")
    print("Author: Thrilok Pasupuleti")
    print("="*70 + "\n")
    
    # Find latest dataset
    data_files = sorted(Path('data/raw').glob('github_prs_*.csv'))
    if not data_files:
        logger.error("No dataset found in data/raw/")
        return
    
    data_path = data_files[-1]
    
    # Load and prepare data
    df = load_data(data_path)
    X, y, feature_names, encoder = prepare_features(df)
    
    # Train models
    results, trained_models, scalers, X_test, y_test = train_all_models(X, y, feature_names)
    
    # Create visualizations
    create_visualizations(results)
    
    # Save models
    save_models(trained_models, scalers)
    
    # Save report
    save_results_report(results)
    
    # Print summary
    print_summary(results)
    
    print("\nTraining completed successfully!")
    print(f"Models saved in: models/")
    print(f"Results saved in: results/")
    print(f"Check results/training_report.json for detailed metrics\n")

if __name__ == "__main__":
    main()

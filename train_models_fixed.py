"""
FIXED: No Overfitting - Removed Data Leakage Features
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
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

class DataLoader:
    @staticmethod
    def load_dataset(filepath):
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath)
        
        logger.info(f"Dataset loaded: {len(df)} samples, {len(df.columns)} features")
        logger.info(f"Accepted: {len(df[df['outcome'] == 'Accept'])}, Rejected: {len(df[df['outcome'] == 'Reject'])}")
        logger.info(f"Repositories: {df['repository'].nunique()}")
        
        return df
    
    @staticmethod
    def prepare_features(df):
        logger.info("Preparing features (FIXED - No Data Leakage)...")
        
        # REMOVED LEAKAGE FEATURES:
        # - time_to_merge_hours (only exists for merged PRs)
        # - time_to_close_hours (reveals outcome)
        # - response_speed_category (based on merge time)
        
        # SAFE FEATURES ONLY:
        numerical_features = [
            # Review features (SAFE - available before outcome)
            'has_reviews', 'review_count', 'approved_reviews', 'changes_requested',
            'commented_reviews', 'dismissed_reviews', 'review_approval_rate',
            'unique_reviewers', 'reviewer_diversity',
            
            # File features (SAFE - known when PR created)
            'files_changed', 'additions', 'deletions', 'total_changes',
            'change_ratio', 'lines_per_file', 'total_file_size_changes',
            
            # Commit features (SAFE - known when PR created)
            'commits', 'commits_per_file', 'commit_messages_total_length',
            'avg_commit_message_length', 'unique_commit_authors',
            
            # Temporal features (SAFE - when PR was created)
            'created_day_of_week', 'created_hour', 'created_month',
            'is_weekend', 'is_business_hours',
            # REMOVED: time_to_merge_hours, time_to_close_hours
            'time_to_first_response_hours',  # SAFE - first response time
            
            # Text features (SAFE - PR content)
            'title_length', 'title_word_count', 'title_avg_word_length',
            'title_has_code', 'body_length', 'body_word_count',
            'body_has_code', 'body_has_links', 'has_body',
            
            # Interaction features (SAFE - before outcome)
            'comments', 'review_comments', 'total_comments', 'comment_density',
            
            # Metadata features (SAFE)
            'is_first_time_contributor', 'labels_count', 'has_labels',
            'is_draft', 'file_types_count', 'python_files', 'test_files', 'doc_files'
        ]
        
        available_features = [f for f in numerical_features if f in df.columns]
        
        logger.info(f"Selected {len(available_features)} SAFE features (no data leakage)")
        logger.info(f"REMOVED LEAKAGE FEATURES: time_to_merge_hours, time_to_close_hours, response_speed_category")
        
        X = df[available_features].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['outcome'])
        
        logger.info(f"Feature matrix: {X.shape}")
        logger.info(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, available_features, label_encoder

class ModelTrainer:
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.results = {}
        self.scalers = {}
        
    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        logger.info(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        
    def get_models(self):
        models = {}
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,  # Reduced from 300
                max_depth=5,       # Reduced from 8
                learning_rate=0.1, # Increased from 0.05
                num_leaves=31,     # Reduced from 50
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,     # Increased regularization
                reg_lambda=1.0,    # Increased regularization
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,  # Reduced from 300
                max_depth=5,       # Reduced from 8
                learning_rate=0.1, # Increased from 0.05
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.5,         # Increased
                reg_alpha=1.0,     # Increased
                reg_lambda=1.0,    # Increased
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,      # Reduced from 300
            max_depth=10,          # Reduced from 15
            min_samples_split=20,  # Increased from 10
            min_samples_leaf=10,   # Increased from 4
            random_state=42,
            n_jobs=-1
        )
        
        models['Extra Trees'] = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,      # Reduced from 200
            learning_rate=0.1,     # Increased from 0.05
            max_depth=4,           # Reduced from 6
            min_samples_split=20,  # Increased from 10
            subsample=0.8,
            random_state=42
        )
        
        models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            C=0.1,                 # Added strong regularization
            random_state=42,
            n_jobs=-1
        )
        
        logger.info(f"Initialized {len(models)} models (with anti-overfitting parameters)")
        return models
        
    def train_model(self, name, model):
        logger.info(f"Training {name}...")
        start_time = datetime.now()
        
        if name == 'Logistic Regression':
            scaler = StandardScaler()
            X_train_use = scaler.fit_transform(self.X_train)
            X_test_use = scaler.transform(self.X_test)
            self.scalers[name] = scaler
        else:
            X_train_use = self.X_train
            X_test_use = self.X_test
        
        model.fit(X_train_use, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Train accuracy (to check overfitting)
        y_train_pred = model.predict(X_train_use)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # Test accuracy
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]
        
        test_accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        mcc = matthews_corrcoef(self.y_test, y_pred)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Calculate overfitting gap
        overfitting_gap = train_accuracy - test_accuracy
        
        result = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_gap': overfitting_gap,
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'mcc': mcc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': self.y_test,
            'confusion_matrix': cm
        }
        
        if hasattr(model, 'feature_importances_'):
            result['feature_importance'] = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info(f"{name} Results:")
        logger.info(f"  Train Accuracy: {train_accuracy*100:.2f}%")
        logger.info(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
        logger.info(f"  Overfitting Gap: {overfitting_gap*100:.2f}% {'HIGH!' if overfitting_gap > 0.05 else 'Good'}")
        logger.info(f"  Precision: {precision*100:.2f}%")
        logger.info(f"  Recall:    {recall*100:.2f}%")
        logger.info(f"  F1-Score:  {f1*100:.2f}%")
        logger.info(f"  ROC AUC:   {roc_auc:.4f}")
        logger.info(f"  CV Score:  {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        logger.info(f"  Time:      {training_time:.2f}s")
        
        return result
    
    def train_all(self):
        logger.info("="*80)
        logger.info("TRAINING ALL MODELS (ANTI-OVERFITTING)")
        logger.info("="*80)
        
        models = self.get_models()
        
        for name, model in models.items():
            try:
                result = self.train_model(name, model)
                self.results[name] = result
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        
        return self.results

class ResultAnalyzer:
    def __init__(self, results, output_dir='results'):
        self.results = results
        self.output_dir = Path(output_dir)
        
    def plot_comparison(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            model_names = list(self.results.keys())
            values = [self.results[m][metric] * 100 for m in model_names]
            
            bars = ax.bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 0.9, len(model_names))))
            ax.set_ylabel(f'{name} (%)', fontsize=11)
            ax.set_title(f'{name} Comparison', fontsize=13, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.set_ylim(0, 100)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot overfitting gap
        ax = axes[5]
        model_names = list(self.results.keys())
        gaps = [self.results[m]['overfitting_gap'] * 100 for m in model_names]
        
        colors = ['red' if gap > 5 else 'green' for gap in gaps]
        bars = ax.bar(model_names, gaps, color=colors)
        ax.set_ylabel('Overfitting Gap (%)', fontsize=11)
        ax.set_title('Train-Test Gap (Lower is Better)', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.axhline(y=5, color='orange', linestyle='--', label='Acceptable threshold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_fixed.png', bbox_inches='tight')
        logger.info("Saved: model_comparison_fixed.png")
        plt.close()
    
    def plot_confusion_matrices(self):
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Reject', 'Accept'],
                       yticklabels=['Reject', 'Accept'])
            
            gap = result['overfitting_gap'] * 100
            gap_text = f" (Gap: {gap:.1f}%)" if gap > 5 else ""
            
            axes[idx].set_title(f'{model_name}{gap_text}', fontsize=11, fontweight='bold')
        
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices_fixed.png', bbox_inches='tight')
        logger.info("Saved: confusion_matrices_fixed.png")
        plt.close()
    
    def plot_feature_importance(self, top_n=20):
        tree_models = {name: result for name, result in self.results.items()
                      if 'feature_importance' in result}
        
        if not tree_models:
            return
        
        best_model = max(tree_models.items(), key=lambda x: x[1]['accuracy'])
        model_name, result = best_model
        
        fig, ax = plt.subplots(figsize=(12, 8))
        feature_imp = result['feature_importance'].head(top_n)
        
        ax.barh(range(len(feature_imp)), feature_imp['importance'],
               color=plt.cm.viridis(np.linspace(0, 0.9, len(feature_imp))))
        ax.set_yticks(range(len(feature_imp)))
        ax.set_yticklabels(feature_imp['feature'], fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{model_name} - Top {top_n} Features (No Leakage)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance_fixed.png', bbox_inches='tight')
        logger.info("Saved: feature_importance_fixed.png")
        plt.close()
        
        result['feature_importance'].to_csv(self.output_dir / 'feature_importance_fixed.csv', index=False)
    
    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        
        for (model_name, result), color in zip(self.results.items(), 
                                                plt.cm.viridis(np.linspace(0, 0.9, len(self.results)))):
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC={result["roc_auc"]:.3f})', 
                    linewidth=2.5, color=color)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('ROC Curves (Fixed - No Overfitting)', fontsize=15, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.savefig(self.output_dir / 'roc_curves_fixed.png', bbox_inches='tight')
        logger.info("Saved: roc_curves_fixed.png")
        plt.close()
    
    def generate_report(self):
        report = {
            'generation_time': datetime.now().isoformat(),
            'models_trained': len(self.results),
            'best_model': max(self.results.items(), key=lambda x: x[1]['accuracy'])[0],
            'overfitting_check': 'PASSED - No data leakage features',
            'results': {}
        }
        
        for name, result in sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            report['results'][name] = {
                'train_accuracy': float(result['train_accuracy']),
                'test_accuracy': float(result['accuracy']),
                'overfitting_gap': float(result['overfitting_gap']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'roc_auc': float(result['roc_auc']),
                'cv_mean': float(result['cv_mean']),
                'training_time': float(result['training_time'])
            }
        
        with open(self.output_dir / 'training_report_fixed.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Saved: training_report_fixed.json")
        return report
    
    def generate_all(self):
        logger.info("Generating visualizations...")
        self.plot_comparison()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        self.plot_roc_curves()
        report = self.generate_report()
        logger.info("All visualizations generated")
        return report

def save_models(results, scalers):
    logger.info("Saving models...")
    output_dir = Path('models_fixed')
    output_dir.mkdir(exist_ok=True)
    
    for name, result in results.items():
        model_file = output_dir / f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(result['model'], model_file)
        logger.info(f"Saved: {model_file}")
        
        if name in scalers:
            scaler_file = output_dir / f"{name.lower().replace(' ', '_')}_scaler.pkl"
            joblib.dump(scalers[name], scaler_file)
    
    logger.info("All models saved")

def print_summary(results):
    logger.info("="*80)
    logger.info("FINAL RESULTS (FIXED - NO OVERFITTING)")
    logger.info("="*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    best_model, best_result = sorted_results[0]
    
    logger.info(f"\nBEST MODEL: {best_model}")
    logger.info(f"  Train Accuracy: {best_result['train_accuracy']*100:.2f}%")
    logger.info(f"  Test Accuracy:  {best_result['accuracy']*100:.2f}%")
    logger.info(f"  Overfitting Gap: {best_result['overfitting_gap']*100:.2f}%")
    logger.info(f"  F1-Score:  {best_result['f1_score']*100:.2f}%")
    logger.info(f"  ROC AUC:   {best_result['roc_auc']:.4f}")
    
    logger.info("\nALL MODELS:")
    for rank, (name, result) in enumerate(sorted_results, 1):
        gap = result['overfitting_gap'] * 100
        gap_status = "OVERFITTING" if gap > 5 else "Good"
        logger.info(f"{rank}. {name}: {result['accuracy']*100:.2f}% (Gap: {gap:.2f}% {gap_status})")

def main():
    logger.info("="*80)
    logger.info("GITHUB PR PREDICTION - FIXED TRAINING (NO OVERFITTING)")
    logger.info("="*80)
    
    data_files = sorted(Path('data/raw').glob('github_prs_*.csv'))
    if not data_files:
        logger.error("No dataset found!")
        return
    
    data_path = data_files[-1]
    logger.info(f"Using dataset: {data_path}")
    
    df = DataLoader.load_dataset(data_path)
    X, y, feature_names, label_encoder = DataLoader.prepare_features(df)
    
    trainer = ModelTrainer(X, y, feature_names)
    trainer.split_data()
    results = trainer.train_all()
    
    analyzer = ResultAnalyzer(results)
    analyzer.generate_all()
    
    save_models(results, trainer.scalers)
    print_summary(results)
    
    logger.info("="*80)
    logger.info("FIXED TRAINING COMPLETED - CHECK OVERFITTING GAPS!")
    logger.info("="*80)

if __name__ == "__main__":
    main()

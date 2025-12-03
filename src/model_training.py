"""
Model Training Module
=====================

Enterprise-grade model training for rare disease enrollment prediction.
Includes:
- Multiple model training (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning with cross-validation
- Class imbalance handling (SMOTE, class weights)
- Model evaluation and comparison
- Feature importance analysis
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config import (
    MODEL_DIR, CHARTS_DIR, OUTPUT_DIR,
    model_config, rare_disease_config
)


class ModelTrainer:
    """
    Enterprise-grade model training for enrollment prediction.
    
    Handles the complete model training pipeline with special
    consideration for rare disease pharmaceutical context.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.training_results = {}
        self.feature_names = []
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = None,
                    handle_imbalance: str = None) -> Tuple:
        """
        Prepare data for training with train/test split and imbalance handling.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        test_size : float
            Test set proportion
        handle_imbalance : str
            Method to handle imbalance: 'SMOTE', 'ADASYN', 'undersample', None
        """
        
        print("\n" + "=" * 80)
        print("STEP 6: MODEL TRAINING")
        print("Training enterprise-grade models for rare disease enrollment prediction")
        print("=" * 80)
        
        if test_size is None:
            test_size = model_config.test_size
        
        if handle_imbalance is None:
            handle_imbalance = rare_disease_config.imbalance_strategy
        
        self.feature_names = X.columns.tolist()
        
        # Store original class distribution
        print(f"\n📊 Original class distribution:")
        for cls, count in y.value_counts().items():
            print(f"   Class {cls}: {count:,} ({count/len(y)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=model_config.random_state,
            stratify=y
        )
        
        print(f"\n📊 Train/Test Split:")
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Handle class imbalance
        if handle_imbalance and handle_imbalance != 'class_weight':
            print(f"\n🔧 Handling class imbalance with {handle_imbalance}...")
            
            if handle_imbalance == 'SMOTE':
                sampler = SMOTE(random_state=model_config.random_state)
            elif handle_imbalance == 'ADASYN':
                sampler = ADASYN(random_state=model_config.random_state)
            elif handle_imbalance == 'undersample':
                sampler = RandomUnderSampler(random_state=model_config.random_state)
            
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
            
            print(f"   After resampling: {len(X_train_resampled):,} samples")
            for cls, count in pd.Series(y_train_resampled).value_counts().items():
                print(f"   Class {cls}: {count:,}")
            
            return X_train_resampled, X_test_scaled, y_train_resampled, y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_models(self) -> Dict[str, Any]:
        """Get configured models for training."""
        
        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=model_config.random_state
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=model_config.random_state,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=5,  # For imbalanced data
                random_state=model_config.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                is_unbalance=True,
                random_state=model_config.random_state,
                verbose=-1
            )
        }
        
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train all models and evaluate performance."""
        
        models = self.get_models()
        results = {}
        
        print(f"\n🚀 Training {len(models)} models...")
        
        for name, model in models.items():
            print(f"\n   Training {name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Evaluate
                metrics = {
                    'accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
                    'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
                    'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
                    'roc_auc': round(roc_auc_score(y_test, y_pred_proba), 4),
                    'pr_auc': round(average_precision_score(y_test, y_pred_proba), 4)
                }
                
                # Cross-validation
                cv = StratifiedKFold(n_splits=model_config.cv_folds, shuffle=True, 
                                    random_state=model_config.random_state)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
                metrics['cv_f1_mean'] = round(cv_scores.mean(), 4)
                metrics['cv_f1_std'] = round(cv_scores.std(), 4)
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                self.models[name] = model
                
                print(f"      ✓ F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}, "
                      f"CV-F1: {metrics['cv_f1_mean']:.4f} (±{metrics['cv_f1_std']:.4f})")
                
            except Exception as e:
                print(f"      ✗ Error: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.training_results = results
        return results
    
    def tune_best_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                       model_name: str = None) -> Tuple[Any, Dict]:
        """Tune hyperparameters for the best model."""
        
        if model_name is None:
            # Find best model by F1 score
            best_f1 = 0
            for name, result in self.training_results.items():
                if 'metrics' in result and result['metrics']['f1'] > best_f1:
                    best_f1 = result['metrics']['f1']
                    model_name = name
        
        print(f"\n🔧 Tuning hyperparameters for {model_name}...")
        
        param_grid = model_config.param_grids.get(model_name, {})
        
        if not param_grid:
            print(f"   No parameter grid defined for {model_name}")
            return self.models[model_name], {}
        
        # Use reduced grid for speed
        reduced_grid = {}
        for key, values in param_grid.items():
            if isinstance(values, list) and len(values) > 3:
                reduced_grid[key] = values[:3]  # Take first 3 values
            else:
                reduced_grid[key] = values
        
        base_model = self.get_models()[model_name]
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=model_config.random_state)
        
        grid_search = RandomizedSearchCV(
            base_model,
            reduced_grid,
            n_iter=10,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            random_state=model_config.random_state,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"   ✓ Best parameters: {grid_search.best_params_}")
        print(f"   ✓ Best CV F1: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        self.best_model_name = model_name
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_final_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Comprehensive evaluation of the final model."""
        
        if self.best_model is None:
            raise ValueError("No best model trained yet")
        
        print(f"\n📊 Final Model Evaluation: {self.best_model_name}")
        print("=" * 50)
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        print("\n   Performance Metrics:")
        for metric, value in metrics.items():
            print(f"      {metric}: {value:.4f}")
        
        # Classification report
        print("\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'High']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"      True Negatives: {cm[0,0]}")
        print(f"      False Positives: {cm[0,1]}")
        print(f"      False Negatives: {cm[1,0]}")
        print(f"      True Positives: {cm[1,1]}")
        
        return metrics
    
    def plot_model_comparison(self) -> str:
        """Plot model comparison visualization."""
        
        if not self.training_results:
            return None
        
        # Prepare data
        models_data = []
        for name, result in self.training_results.items():
            if 'metrics' in result:
                models_data.append({
                    'Model': name,
                    **result['metrics']
                })
        
        if not models_data:
            return None
        
        df = pd.DataFrame(models_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # F1 scores
        ax1 = axes[0, 0]
        bars = ax1.bar(df['Model'], df['f1'], color='steelblue', edgecolor='white')
        ax1.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1 Score')
        ax1.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, df['f1']):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                    ha='center', fontsize=10)
        
        # ROC-AUC
        ax2 = axes[0, 1]
        bars = ax2.bar(df['Model'], df['roc_auc'], color='coral', edgecolor='white')
        ax2.set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ROC-AUC')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, df['roc_auc']):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                    ha='center', fontsize=10)
        
        # Precision vs Recall
        ax3 = axes[1, 0]
        x = np.arange(len(df))
        width = 0.35
        ax3.bar(x - width/2, df['precision'], width, label='Precision', color='green', alpha=0.7)
        ax3.bar(x + width/2, df['recall'], width, label='Recall', color='purple', alpha=0.7)
        ax3.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(df['Model'], rotation=45)
        ax3.legend()
        
        # CV F1 with error bars
        ax4 = axes[1, 1]
        ax4.bar(df['Model'], df['cv_f1_mean'], yerr=df['cv_f1_std'], 
               color='teal', edgecolor='white', capsize=5)
        ax4.set_title('Cross-Validation F1 (with std)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('CV F1 Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Model Performance Comparison - Enrollment Prediction', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = CHARTS_DIR / 'model_comparison.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {filepath}")
        return str(filepath)
    
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """Plot ROC curves for all models."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for idx, (name, model) in enumerate(self.models.items()):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                ax.plot(fpr, tpr, color=colors[idx % len(colors)], 
                       linewidth=2, label=f'{name} (AUC = {auc:.3f})')
            except Exception as e:
                print(f"Could not plot ROC for {name}: {e}")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = CHARTS_DIR / 'roc_curves.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filepath}")
        return str(filepath)
    
    def save_model(self, filepath: str = None) -> str:
        """Save the best model to disk."""
        
        if self.best_model is None:
            raise ValueError("No best model to save")
        
        if filepath is None:
            filepath = MODEL_DIR / f'best_model_{self.best_model_name}.joblib'
        
        # Save model package
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_results': {
                k: v['metrics'] if 'metrics' in v else v
                for k, v in self.training_results.items()
            }
        }
        
        joblib.dump(model_package, filepath)
        print(f"\n✓ Model saved to: {filepath}")
        
        return str(filepath)
    
    def run_training_pipeline(self, X: pd.DataFrame, y: pd.Series,
                             tune_hyperparameters: bool = True) -> Dict:
        """Execute complete training pipeline."""
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train models
        self.train_models(X_train, y_train, X_test, y_test)
        
        # Tune best model
        if tune_hyperparameters:
            self.tune_best_model(X_train, y_train)
        else:
            # Select best model without tuning
            best_f1 = 0
            for name, result in self.training_results.items():
                if 'metrics' in result and result['metrics']['f1'] > best_f1:
                    best_f1 = result['metrics']['f1']
                    self.best_model = self.models[name]
                    self.best_model_name = name
        
        # Evaluate
        final_metrics = self.evaluate_final_model(X_test, y_test)
        
        # Plots
        self.plot_model_comparison()
        self.plot_roc_curves(X_test, y_test)
        
        # Save model
        model_path = self.save_model()
        
        return {
            'best_model_name': self.best_model_name,
            'final_metrics': final_metrics,
            'model_path': model_path,
            'all_results': self.training_results
        }


def run_model_training(X: pd.DataFrame, y: pd.Series) -> Tuple[ModelTrainer, Dict]:
    """Execute the complete model training process."""
    trainer = ModelTrainer()
    results = trainer.run_training_pipeline(X, y)
    return trainer, results


if __name__ == "__main__":
    print("Model Training module - run from main pipeline")

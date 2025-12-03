"""
Scoring Module
==============

Enterprise-grade scoring and inference for enrollment prediction.
Includes:
- Model loading and inference
- Batch scoring
- Real-time prediction API
- Score explanation with SHAP
- Prediction output formatting
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_DIR, OUTPUT_DIR, CHARTS_DIR


class ModelScorer:
    """
    Enterprise-grade scoring for enrollment prediction.
    
    Provides inference capabilities for the trained model
    including batch scoring and real-time predictions.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_name = None
        self.scaler = None
        self.feature_names = []
        self.explainer = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str = None) -> 'ModelScorer':
        """Load trained model from disk."""
        
        if model_path is None:
            # Find latest model
            model_files = list(MODEL_DIR.glob('best_model_*.joblib'))
            if not model_files:
                raise FileNotFoundError("No trained model found in models directory")
            model_path = sorted(model_files)[-1]
        
        print(f"\n📦 Loading model from: {model_path}")
        
        model_package = joblib.load(model_path)
        
        self.model = model_package['model']
        self.model_name = model_package['model_name']
        self.scaler = model_package['scaler']
        self.feature_names = model_package['feature_names']
        
        print(f"   ✓ Loaded {self.model_name} model")
        print(f"   ✓ Features: {len(self.feature_names)}")
        
        return self
    
    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for scoring."""
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            print(f"   ⚠ Missing features (filling with 0): {missing_features}")
            for feat in missing_features:
                X[feat] = 0
        
        # Select and order features
        X = X[self.feature_names].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        X_prepared = self.prepare_features(X)
        predictions = self.model.predict(X_prepared)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        X_prepared = self.prepare_features(X)
        probabilities = self.model.predict_proba(X_prepared)[:, 1]
        
        return probabilities
    
    def score_batch(self, X: pd.DataFrame, 
                   include_explanation: bool = False) -> pd.DataFrame:
        """
        Score a batch of records.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix to score
        include_explanation : bool
            Whether to include SHAP explanations
            
        Returns:
        --------
        pd.DataFrame with predictions and probabilities
        """
        
        print("\n" + "=" * 80)
        print("STEP 7: MODEL SCORING")
        print("Generating enrollment predictions")
        print("=" * 80)
        
        print(f"\n📊 Scoring {len(X):,} records...")
        
        X_prepared = self.prepare_features(X)
        
        # Generate predictions
        predictions = self.model.predict(X_prepared)
        probabilities = self.model.predict_proba(X_prepared)[:, 1]
        
        # Create output dataframe
        results = X.copy()
        results['prediction'] = predictions
        results['probability'] = probabilities.round(4)
        results['prediction_label'] = results['prediction'].map({
            0: 'Low Enrollment Potential',
            1: 'High Enrollment Potential'
        })
        
        # Add risk tier
        results['risk_tier'] = pd.cut(
            results['probability'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Add score timestamp
        results['scored_at'] = datetime.now().isoformat()
        
        # Summary
        print(f"\n📊 Scoring Results:")
        print(f"   High Enrollment Potential: {(predictions == 1).sum():,} ({(predictions == 1).mean()*100:.1f}%)")
        print(f"   Low Enrollment Potential: {(predictions == 0).sum():,} ({(predictions == 0).mean()*100:.1f}%)")
        print(f"   Mean probability: {probabilities.mean():.4f}")
        
        print(f"\n📊 Risk Tier Distribution:")
        for tier, count in results['risk_tier'].value_counts().sort_index().items():
            print(f"   {tier}: {count:,}")
        
        if include_explanation:
            self._add_explanations(results, X_prepared)
        
        return results
    
    def _add_explanations(self, results: pd.DataFrame, 
                         X_prepared: pd.DataFrame) -> pd.DataFrame:
        """Add SHAP explanations to results."""
        
        print("\n🔍 Generating SHAP explanations...")
        
        try:
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)
            
            # Use smaller sample for SHAP to avoid timeout
            sample_size = min(1000, len(X_prepared))
            X_sample = X_prepared.iloc[:sample_size]
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Get top contributing features for sampled predictions
            top_features = []
            for i in range(len(shap_values)):
                feature_contributions = list(zip(self.feature_names, shap_values[i]))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_3 = [(f, round(v, 4)) for f, v in feature_contributions[:3]]
                top_features.append(str(top_3))
            
            # For remaining records, use empty string (SHAP only computed on sample)
            results['top_contributing_features'] = top_features + [''] * (len(results) - len(top_features))
            
            print("   ✓ Added SHAP explanations")
            
        except Exception as e:
            print(f"   ⚠ Could not generate SHAP explanations: {str(e)}")
        
        return results
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict:
        """
        Generate detailed explanation for a single prediction.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        index : int
            Index of the record to explain
            
        Returns:
        --------
        Dictionary with prediction details and feature contributions
        """
        
        X_prepared = self.prepare_features(X)
        
        # Get prediction
        prediction = self.model.predict(X_prepared.iloc[[index]])[0]
        probability = self.model.predict_proba(X_prepared.iloc[[index]])[0, 1]
        
        explanation = {
            'prediction': int(prediction),
            'prediction_label': 'High Enrollment Potential' if prediction == 1 else 'Low Enrollment Potential',
            'probability': round(probability, 4),
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.15 else 'Low'
        }
        
        # SHAP explanation
        try:
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)
            
            shap_values = self.explainer.shap_values(X_prepared.iloc[[index]])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            feature_contributions = list(zip(self.feature_names, shap_values[0]))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation['top_positive_factors'] = [
                {'feature': f, 'contribution': round(v, 4)}
                for f, v in feature_contributions if v > 0
            ][:5]
            
            explanation['top_negative_factors'] = [
                {'feature': f, 'contribution': round(v, 4)}
                for f, v in feature_contributions if v < 0
            ][:5]
            
        except Exception as e:
            explanation['explanation_error'] = str(e)
        
        return explanation
    
    def plot_shap_summary(self, X: pd.DataFrame, max_samples: int = 1000) -> str:
        """Generate SHAP summary plot."""
        
        print("\n📈 Generating SHAP summary plot...")
        
        X_prepared = self.prepare_features(X)
        
        if len(X_prepared) > max_samples:
            X_sample = X_prepared.sample(max_samples, random_state=42)
        else:
            X_sample = X_prepared
        
        try:
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)
            
            shap_values = self.explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
            
            plt.tight_layout()
            
            filepath = CHARTS_DIR / 'shap_summary.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"   ⚠ Could not generate SHAP plot: {str(e)}")
            return None
    
    def save_scores(self, scores: pd.DataFrame, 
                   filename: str = None) -> str:
        """Save scoring results to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'enrollment_predictions_{timestamp}.csv'
        
        filepath = OUTPUT_DIR / filename
        
        # Select output columns
        output_cols = [
            col for col in scores.columns 
            if col not in self.feature_names or col in ['territory_name', 'hcp_id', 'hcp_segment', 'region_name']
        ]
        
        scores[output_cols].to_csv(filepath, index=False)
        
        print(f"\n✓ Predictions saved to: {filepath}")
        
        return str(filepath)
    
    def generate_scoring_report(self, scores: pd.DataFrame) -> str:
        """Generate comprehensive scoring report."""
        
        report = []
        report.append("=" * 80)
        report.append("ENROLLMENT PREDICTION SCORING REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        report.append(f"\nModel: {self.model_name}")
        report.append(f"Features used: {len(self.feature_names)}")
        report.append(f"Records scored: {len(scores):,}")
        
        report.append("\n## PREDICTION SUMMARY")
        report.append("-" * 40)
        
        for pred, count in scores['prediction'].value_counts().items():
            label = 'High Enrollment Potential' if pred == 1 else 'Low Enrollment Potential'
            pct = count / len(scores) * 100
            report.append(f"{label}: {count:,} ({pct:.1f}%)")
        
        report.append("\n## RISK TIER DISTRIBUTION")
        report.append("-" * 40)
        
        for tier in ['Very High', 'High', 'Medium', 'Low']:
            if tier in scores['risk_tier'].values:
                count = (scores['risk_tier'] == tier).sum()
                pct = count / len(scores) * 100
                report.append(f"{tier}: {count:,} ({pct:.1f}%)")
        
        report.append("\n## PROBABILITY STATISTICS")
        report.append("-" * 40)
        report.append(f"Mean: {scores['probability'].mean():.4f}")
        report.append(f"Median: {scores['probability'].median():.4f}")
        report.append(f"Std: {scores['probability'].std():.4f}")
        report.append(f"Min: {scores['probability'].min():.4f}")
        report.append(f"Max: {scores['probability'].max():.4f}")
        
        if 'territory_name' in scores.columns:
            report.append("\n## TOP TERRITORIES BY ENROLLMENT POTENTIAL")
            report.append("-" * 40)
            
            territory_scores = scores.groupby('territory_name')['probability'].mean()
            top_territories = territory_scores.nlargest(10)
            
            for territory, prob in top_territories.items():
                report.append(f"{territory}: {prob:.4f}")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = OUTPUT_DIR / 'reports' / 'scoring_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Scoring report saved to: {report_path}")
        
        return report_text


def run_scoring(X: pd.DataFrame, model_path: str = None) -> Tuple[ModelScorer, pd.DataFrame]:
    """Execute the complete scoring process."""
    
    scorer = ModelScorer()
    scorer.load_model(model_path)
    scores = scorer.score_batch(X, include_explanation=True)
    scorer.save_scores(scores)
    scorer.generate_scoring_report(scores)
    
    return scorer, scores


if __name__ == "__main__":
    print("Scoring module - run from main pipeline")

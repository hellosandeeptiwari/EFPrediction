"""
Main Pipeline Orchestrator
==========================

End-to-end pipeline for Enrollment Form Prediction
for Rare Disease Pharmaceutical Company.

This module orchestrates all steps:
1. Data Discovery
2. Exploratory Data Analysis
3. Feature Engineering
4. Feature Selection
5. Target Engineering
6. Model Training
7. Scoring
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, CHARTS_DIR, MODEL_DIR
from data_discovery import DataDiscovery
from eda import RareDiseaseEDA
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector
from target_engineering import TargetEngineer
from model_training import ModelTrainer
from scoring import ModelScorer
from pptx_generator import PPTXReportGenerator


class EnrollmentPredictionPipeline:
    """
    Enterprise-grade ML pipeline for enrollment prediction.
    
    Orchestrates the complete workflow from data discovery
    to model scoring for rare disease pharmaceutical context.
    """
    
    def __init__(self):
        self.discovery = None
        self.eda = None
        self.feature_engineer = None
        self.feature_selector = None
        self.target_engineer = None
        self.model_trainer = None
        self.scorer = None
        
        self.data_dict = {}
        self.feature_df = None
        self.target = None
        self.X_selected = None
        self.selected_features = []
        self.training_results = {}
        self.charts = []
        self.insights = {}
        
    def run_discovery(self):
        """Step 1: Data Discovery"""
        
        print("\n" + "=" * 80)
        print("STEP 1: DATA DISCOVERY")
        print("=" * 80)
        
        self.discovery = DataDiscovery()
        self.data_dict = self.discovery.load_all_data()
        self.discovery.discover_all()
        self.discovery.generate_discovery_report()
        
        return self
    
    def run_eda(self):
        """Step 2: Exploratory Data Analysis"""
        
        print("\n" + "=" * 80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        self.eda = RareDiseaseEDA(self.data_dict)
        self.insights = self.eda.run_complete_eda()
        self.charts = self.eda.charts_generated
        
        return self
    
    def run_feature_engineering(self):
        """Step 3: Feature Engineering"""
        
        self.feature_engineer = FeatureEngineer(self.data_dict)
        self.feature_df = self.feature_engineer.run_feature_engineering()
        
        return self
    
    def run_target_engineering(self, target_type: str = 'binary'):
        """Step 5: Target Engineering"""
        
        self.target_engineer = TargetEngineer(self.data_dict)
        self.feature_df, self.target = self.target_engineer.create_enrollment_target(
            self.feature_df,
            target_type=target_type
        )
        self.target_engineer.analyze_target_characteristics(self.target)
        
        return self
    
    def run_feature_selection(self, n_features: int = 25):
        """Step 4: Feature Selection"""
        
        # Get numeric features only
        feature_cols = self.feature_engineer.get_feature_list()
        X = self.feature_df[feature_cols].copy()
        
        self.feature_selector = FeatureSelector(X)
        self.X_selected, self.selected_features = self.feature_selector.run_feature_selection(
            X, self.target, final_n_features=n_features
        )
        
        # Add importance plot to charts
        if hasattr(self.feature_selector, 'plot_feature_importance'):
            importance_chart = CHARTS_DIR / 'feature_importance.png'
            if importance_chart.exists():
                self.charts.append(str(importance_chart))
        
        return self
    
    def run_model_training(self, tune_hyperparameters: bool = True):
        """Step 6: Model Training"""
        
        self.model_trainer = ModelTrainer()
        self.training_results = self.model_trainer.run_training_pipeline(
            self.X_selected,
            self.target,
            tune_hyperparameters=tune_hyperparameters
        )
        
        # Add model charts
        model_comparison_chart = CHARTS_DIR / 'model_comparison.png'
        roc_chart = CHARTS_DIR / 'roc_curves.png'
        
        if model_comparison_chart.exists():
            self.charts.append(str(model_comparison_chart))
        if roc_chart.exists():
            self.charts.append(str(roc_chart))
        
        return self
    
    def run_scoring(self, X: pd.DataFrame = None):
        """Step 7: Scoring"""
        
        if X is None:
            X = self.X_selected
        
        self.scorer = ModelScorer()
        self.scorer.load_model()
        scores = self.scorer.score_batch(X, include_explanation=True)
        self.scorer.save_scores(scores)
        self.scorer.generate_scoring_report(scores)
        
        # Add SHAP chart
        try:
            shap_chart = self.scorer.plot_shap_summary(X)
            if shap_chart:
                self.charts.append(shap_chart)
        except Exception as e:
            print(f"Could not generate SHAP plot: {e}")
        
        return scores
    
    def generate_presentation(self):
        """Generate PowerPoint presentation"""
        
        generator = PPTXReportGenerator()
        pptx_path = generator.generate_eda_presentation(self.insights, self.charts)
        
        return pptx_path
    
    def run_full_pipeline(self, 
                         target_type: str = 'binary',
                         n_features: int = 25,
                         tune_hyperparameters: bool = True):
        """Execute the complete pipeline."""
        
        print("\n" + "=" * 80)
        print("ENROLLMENT FORM PREDICTION PIPELINE")
        print("Rare Disease Pharmaceutical Company")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Execute all steps
        self.run_discovery()
        self.run_eda()
        self.run_feature_engineering()
        self.run_target_engineering(target_type=target_type)
        self.run_feature_selection(n_features=n_features)
        self.run_model_training(tune_hyperparameters=tune_hyperparameters)
        scores = self.run_scoring()
        pptx_path = self.generate_presentation()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   Data files processed: {len(self.data_dict)}")
        print(f"   Features engineered: {len(self.feature_engineer.get_feature_list())}")
        print(f"   Features selected: {len(self.selected_features)}")
        print(f"   Best model: {self.training_results.get('best_model_name', 'N/A')}")
        print(f"   Charts generated: {len(self.charts)}")
        
        print(f"\n📁 Output Files:")
        print(f"   Reports: {OUTPUT_DIR / 'reports'}")
        print(f"   Charts: {CHARTS_DIR}")
        print(f"   Models: {MODEL_DIR}")
        print(f"   Presentation: {pptx_path}")
        
        print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'scores': scores,
            'training_results': self.training_results,
            'selected_features': self.selected_features,
            'insights': self.insights,
            'charts': self.charts,
            'presentation': pptx_path
        }


def main():
    """Run the complete enrollment prediction pipeline."""
    
    import pandas as pd
    
    pipeline = EnrollmentPredictionPipeline()
    results = pipeline.run_full_pipeline(
        target_type='binary',
        n_features=25,
        tune_hyperparameters=True
    )
    
    return results


if __name__ == "__main__":
    results = main()

"""
Configuration Module for Enrollment Prediction Pipeline
========================================================

Contains all configuration settings for the ML pipeline including:
- File paths
- Model hyperparameters
- Feature engineering settings
- Rare disease specific configurations
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Reporting Tables"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, MODEL_DIR, CHARTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data files and loading"""
    
    # Data file mappings
    data_files: Dict[str, str] = field(default_factory=lambda: {
        'hcp_universe': 'icpt_ai_hcp_universe_202509260526.csv',
        'monthly_kpis': 'monthly_base_kpis__c_202509260532.csv',
        'daily_kpis': 'daily_base_kpis__c_202509260540.csv',
        'territory_hierarchy': 'territory_hierarchy__c_202509260537.csv',
        'monthly_calls_territory': 'monthly_calls_by_territory__c_202509260536.csv',
        'monthly_meetings': 'monthly_meetings__c_202509260534.csv',
        'writers_prescribers': 'writers_prescribers_count__c_202509260536.csv',
        'tbm_goals': 'tbm_goals__c_202509260535.csv',
        'monthly_email': 'monthly_email_sent__c_202509260535.csv',
        'monthly_units': 'monthly_units_by_location__c_202509260540.csv',
        'daily_calls_territory': 'daily_calls_by_territory__c_202509260538.csv',
        'daily_meetings': 'daily_meetings__c_202509260538.csv',
        'daily_calls_tgt_wrtr': 'daily_calls_tgt_wrtr__c_202509260539.csv',
        'monthly_calls_tgt_wrtr': 'monthly_calls_tgt_wrtr__c_202509260532.csv',
        'monthly_tot_summary': 'monthly_tot_summary__c_202509260533.csv',
        'trimester_calls': 'trimester_calls_by_territory__c_202509260537.csv'
    })
    
    # Date columns for parsing
    date_columns: Dict[str, List[str]] = field(default_factory=lambda: {
        'monthly_kpis': ['month_begin_date__c', 'month_ending_date__c'],
        'daily_kpis': ['date__c'],
        'monthly_calls_territory': ['calendar_month_begin_date__c', 'calendar_month_end_date__c'],
        'monthly_meetings': ['month_start_date__c', 'month_end_date__c'],
        'writers_prescribers': ['trimester_start_date__c', 'trimester_end_date__c'],
        'monthly_email': ['calendar_month_begin_date__c', 'calendar_month_end_date__c'],
        'monthly_units': ['month_begin_date__c', 'month_ending_date__c'],
        'daily_calls_territory': ['date__c'],
        'daily_meetings': ['date__c'],
        'daily_calls_tgt_wrtr': ['date__c'],
        'monthly_calls_tgt_wrtr': ['calendar_month_begin_date__c', 'calendar_month_end_date__c'],
        'monthly_tot_summary': ['calendar_month_begin_date__c', 'calendar_month_end_date__c'],
        'trimester_calls': ['trimester_start_date__c', 'trimester_end_date__c']
    })


@dataclass
class RareDiseaseConfig:
    """Configuration specific to rare disease pharmaceutical context"""
    
    # HCP segments with priority for rare disease
    hcp_segments: List[str] = field(default_factory=lambda: [
        'Writer', 'Potential Writer', 'Lapsed Writer'
    ])
    
    # Regions for analysis
    regions: List[str] = field(default_factory=lambda: [
        'NORTHEAST', 'SOUTHEAST', 'Midwest', 'MID-ATLANTIC', 
        'South Central', 'Great Plains', 'West'
    ])
    
    # Enrollment channels
    enrollment_channels: List[str] = field(default_factory=lambda: [
        'HUB', 'WALGREENS', 'CVS CAREMARK', 'ACCREDO', 'ACARIA', 'CURASCRIPT'
    ])
    
    # Key behavioral flags for rare disease patients
    behavioral_flags: List[str] = field(default_factory=lambda: [
        'refill_flag', 'early_stop_flag', 'switch_flag', 'line2_flag', 'line3_flag'
    ])
    
    # Minimum enrollments threshold for rare disease (low volume expected)
    min_enrollments_threshold: int = 1
    
    # Class imbalance handling
    imbalance_strategy: str = 'SMOTE'  # Options: 'SMOTE', 'ADASYN', 'class_weight', 'undersample'


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    
    # Temporal features
    temporal_windows: List[int] = field(default_factory=lambda: [1, 3, 6, 12])  # months
    
    # Aggregation functions
    aggregations: List[str] = field(default_factory=lambda: [
        'sum', 'mean', 'std', 'min', 'max', 'count'
    ])
    
    # Lag features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 6])
    
    # Rolling window sizes
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])


@dataclass
class ModelConfig:
    """Configuration for model training"""
    
    # Models to train
    models: List[str] = field(default_factory=lambda: [
        'LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM'
    ])
    
    # Cross-validation
    cv_folds: int = 5
    stratify: bool = True
    
    # Test split
    test_size: float = 0.2
    
    # Random state for reproducibility
    random_state: int = 42
    
    # Hyperparameter grids
    param_grids: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['saga'],
            'max_iter': [1000],
            'class_weight': ['balanced']
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [1, 5, 10]  # For imbalanced data
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'is_unbalance': [True]
        }
    })
    
    # Metrics for evaluation (rare disease focus)
    primary_metric: str = 'f1'  # F1 is better for imbalanced rare disease data
    secondary_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc'
    ])


@dataclass 
class EDAConfig:
    """Configuration for EDA and visualization"""
    
    # Chart settings
    figure_size: tuple = (12, 8)
    dpi: int = 150
    style: str = 'seaborn-v0_8-whitegrid'
    
    # Color palette for rare disease theme
    color_palette: List[str] = field(default_factory=lambda: [
        '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
        '#95C623', '#6B4E71', '#2E4057', '#048A81', '#54C6EB'
    ])
    
    # Statistical tests
    significance_level: float = 0.05
    
    # Correlation threshold
    correlation_threshold: float = 0.7


# Initialize configuration objects
data_config = DataConfig()
rare_disease_config = RareDiseaseConfig()
feature_config = FeatureConfig()
model_config = ModelConfig()
eda_config = EDAConfig()

"""
Target Engineering Module
=========================

Enterprise-grade target variable engineering for rare disease enrollment prediction.
Includes:
- Target definition for rare disease context
- Binary/multi-class target creation
- Temporal target alignment
- Class imbalance handling strategies
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, OUTPUT_DIR,
    rare_disease_config
)


class TargetEngineer:
    """
    Enterprise-grade target engineering for enrollment prediction.
    
    Handles the unique challenges of rare disease pharmaceutical
    enrollment prediction including low event rates.
    """
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame] = None):
        self.data_dict = data_dict or {}
        self.target_df = None
        self.target_stats = {}
        
    def create_enrollment_target(self, 
                                 feature_df: pd.DataFrame,
                                 target_type: str = 'binary',
                                 threshold: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create enrollment target variable based on HCP behavior.
        
        For rare disease pharma, we predict HCP enrollment potential:
        - Binary: Writer vs Non-Writer (Potential Writer + Lapsed Writer)
        - Multiclass: Writer, Potential Writer, Lapsed Writer
        
        Parameters:
        -----------
        feature_df : pd.DataFrame
            Feature dataframe with hcp_segment
        target_type : str
            'binary' for Writer/Non-Writer, 'multiclass' for all segments
        threshold : int
            Not used for segment-based target (kept for compatibility)
        
        Returns:
        --------
        Tuple of (feature_df_with_target, target_series)
        """
        
        print("\n" + "=" * 80)
        print("STEP 5: TARGET ENGINEERING")
        print("Creating enrollment prediction target for rare disease context")
        print("=" * 80)
        
        df = feature_df.copy()
        
        # Check if hcp_segment exists
        if 'hcp_segment' not in df.columns:
            # Try to get from hcp_universe
            if 'hcp_universe' in self.data_dict:
                hcp_df = self.data_dict['hcp_universe'][['hcp_id', 'hcp_segment']].copy()
                df = df.merge(hcp_df, on='hcp_id', how='left')
        
        if 'hcp_segment' not in df.columns:
            raise ValueError("hcp_segment column required for target engineering")
        
        print(f"\n📊 HCP Segment Distribution:")
        segment_counts = df['hcp_segment'].value_counts()
        for seg, count in segment_counts.items():
            print(f"   {seg}: {count:,} ({count/len(df)*100:.1f}%)")
        
        if target_type == 'binary':
            # Binary target: Writer (1) vs Non-Writer (0)
            # Writers are HCPs who are actively prescribing
            df['target'] = (df['hcp_segment'] == 'Writer').astype(int)
            
            print(f"\n📊 Binary Target (Writer vs Non-Writer):")
            print(f"   Class 0 (Non-Writer): {(df['target'] == 0).sum():,} ({(df['target'] == 0).mean()*100:.1f}%)")
            print(f"   Class 1 (Writer): {(df['target'] == 1).sum():,} ({(df['target'] == 1).mean()*100:.1f}%)")
            
        elif target_type == 'multiclass':
            # Multi-class target: 0=Potential Writer, 1=Lapsed Writer, 2=Writer
            segment_map = {
                'Potential Writer': 0,
                'Lapsed Writer': 1,
                'Writer': 2
            }
            df['target'] = df['hcp_segment'].map(segment_map).fillna(0).astype(int)
            
            print(f"\n📊 Multi-class Target:")
            for seg, label in segment_map.items():
                count = (df['target'] == label).sum()
                print(f"   Class {label} ({seg}): {count:,}")
        
        elif target_type == 'regression':
            # For regression, use power_score or enrollment-related metric
            if 'power_score' in df.columns:
                df['target'] = df['power_score'].fillna(0)
                print(f"\n📊 Regression Target (Power Score):")
                print(f"   Mean: {df['target'].mean():.2f}")
                print(f"   Std: {df['target'].std():.2f}")
            else:
                # Use target_trx as regression target
                df['target'] = df.get('target_trx', 0).fillna(0)
                print(f"\n📊 Regression Target (Target TRx):")
                print(f"   Mean: {df['target'].mean():.2f}")
                print(f"   Std: {df['target'].std():.2f}")
            print(f"   Range: [{df['target'].min():.0f}, {df['target'].max():.0f}]")
        
        # Store statistics
        self.target_stats = {
            'type': target_type,
            'threshold': threshold if target_type == 'binary' else None,
            'class_distribution': df['target'].value_counts().to_dict(),
            'total_samples': len(df)
        }
        
        self.target_df = df
        
        return df, df['target']
    
    def create_hcp_level_target(self,
                                feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create HCP-level enrollment potential target.
        
        Uses HCP segment and behavioral indicators to predict
        enrollment likelihood at the individual HCP level.
        """
        
        print("\n🔧 Creating HCP-level enrollment potential target...")
        
        df = feature_df.copy()
        
        # Create target based on HCP segment and behavior
        # Writer = high potential, Potential Writer = medium, Lapsed = needs reactivation
        
        segment_scores = {
            'Writer': 3,
            'Potential Writer': 2,
            'Lapsed Writer': 1
        }
        
        df['segment_score'] = df['hcp_segment'].map(segment_scores).fillna(0)
        
        # Combine with behavioral indicators
        if 'power_score' in df.columns:
            df['power_score_norm'] = (df['power_score'] - df['power_score'].min()) / \
                                     (df['power_score'].max() - df['power_score'].min() + 0.001)
        else:
            df['power_score_norm'] = 0.5
        
        # Composite score
        df['enrollment_potential_score'] = (
            df['segment_score'] * 0.4 +
            df['power_score_norm'] * 0.3 +
            (df.get('called_when_suggested_pct', 50) / 100) * 0.15 +
            (df.get('total_calls', 0) / (df.get('total_calls', 1).max() + 1)) * 0.15
        )
        
        # Binary target: High potential vs Others
        median_score = df['enrollment_potential_score'].median()
        df['target'] = (df['enrollment_potential_score'] >= median_score).astype(int)
        
        print(f"   ✓ Created HCP-level target")
        print(f"   Class 0 (Lower Potential): {(df['target'] == 0).sum():,}")
        print(f"   Class 1 (Higher Potential): {(df['target'] == 1).sum():,}")
        
        self.target_df = df
        
        return df, df['target']
    
    def create_next_period_target(self,
                                  feature_df: pd.DataFrame,
                                  forecast_horizon: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create forward-looking target for next period enrollment prediction.
        
        This creates a more actionable target that predicts future enrollments
        rather than historical patterns.
        """
        
        print(f"\n🔧 Creating next {forecast_horizon}-month enrollment target...")
        
        if 'monthly_kpis' not in self.data_dict:
            raise ValueError("Monthly KPIs data required")
        
        kpi_df = self.data_dict['monthly_kpis'].copy()
        enrollment_df = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments'].copy()
        enrollment_df['month_begin_date__c'] = pd.to_datetime(enrollment_df['month_begin_date__c'])
        
        # Get max date and define forecast period
        max_date = enrollment_df['month_begin_date__c'].max()
        forecast_start = max_date - pd.DateOffset(months=forecast_horizon)
        
        # Future enrollments (last N months as proxy for future)
        future_enrollments = enrollment_df[
            enrollment_df['month_begin_date__c'] >= forecast_start
        ].groupby('territory_name__c')['kpi_value__c'].sum()
        
        future_enrollments = future_enrollments.reset_index()
        future_enrollments.columns = ['territory_name', 'future_enrollments']
        
        df = feature_df.merge(future_enrollments, on='territory_name', how='left')
        df['future_enrollments'] = df['future_enrollments'].fillna(0)
        
        # Binary target
        threshold = df['future_enrollments'].median()
        df['target'] = (df['future_enrollments'] >= threshold).astype(int)
        
        print(f"   ✓ Created next-period target (threshold={threshold:.0f})")
        print(f"   Class 0: {(df['target'] == 0).sum():,}")
        print(f"   Class 1: {(df['target'] == 1).sum():,}")
        
        self.target_df = df
        
        return df, df['target']
    
    def get_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        
        class_counts = y.value_counts()
        total = len(y)
        n_classes = len(class_counts)
        
        weights = {}
        for cls, count in class_counts.items():
            weights[cls] = total / (n_classes * count)
        
        print(f"\n📊 Class Weights for Imbalanced Learning:")
        for cls, weight in weights.items():
            print(f"   Class {cls}: {weight:.4f}")
        
        return weights
    
    def analyze_target_characteristics(self, y: pd.Series) -> dict:
        """Analyze target variable characteristics for rare disease context."""
        
        analysis = {
            'n_samples': len(y),
            'n_classes': y.nunique(),
            'class_distribution': y.value_counts().to_dict(),
            'class_proportions': (y.value_counts() / len(y) * 100).round(2).to_dict(),
            'is_imbalanced': y.value_counts().min() / y.value_counts().max() < 0.3,
            'minority_class': y.value_counts().idxmin(),
            'minority_pct': round(y.value_counts().min() / len(y) * 100, 2)
        }
        
        print(f"\n📊 Target Analysis:")
        print(f"   Total samples: {analysis['n_samples']:,}")
        print(f"   Number of classes: {analysis['n_classes']}")
        print(f"   Is imbalanced: {analysis['is_imbalanced']}")
        print(f"   Minority class: {analysis['minority_class']} ({analysis['minority_pct']}%)")
        
        # Recommendations for rare disease
        if analysis['is_imbalanced']:
            print(f"\n   ⚠ IMBALANCED DATA DETECTED - Recommendations:")
            print(f"      1. Use SMOTE or ADASYN for oversampling")
            print(f"      2. Apply class weights in model training")
            print(f"      3. Use F1-score or PR-AUC instead of accuracy")
            print(f"      4. Consider cost-sensitive learning")
        
        return analysis


def run_target_engineering(feature_df: pd.DataFrame, 
                          data_dict: Dict[str, pd.DataFrame],
                          target_type: str = 'binary') -> Tuple[pd.DataFrame, pd.Series]:
    """Execute the complete target engineering process."""
    
    engineer = TargetEngineer(data_dict)
    df, target = engineer.create_enrollment_target(feature_df, target_type=target_type)
    engineer.analyze_target_characteristics(target)
    
    return engineer, df, target


if __name__ == "__main__":
    print("Target Engineering module - run from main pipeline")

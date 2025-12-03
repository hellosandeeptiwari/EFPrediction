"""
Feature Selection Module
========================

Enterprise-grade feature selection for rare disease enrollment prediction.
Includes:
- Correlation-based selection
- Variance threshold
- Feature importance ranking
- Recursive feature elimination
- Domain-knowledge based selection
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import (
    OUTPUT_DIR, CHARTS_DIR,
    eda_config, model_config
)


class FeatureSelector:
    """
    Enterprise-grade feature selection for enrollment prediction.
    
    Provides multiple selection methods optimized for rare disease
    pharmaceutical context with class imbalance.
    """
    
    def __init__(self, feature_df: pd.DataFrame = None):
        self.feature_df = feature_df
        self.selected_features = []
        self.feature_importance = {}
        self.selection_results = {}
        
    def remove_low_variance(self, X: pd.DataFrame, 
                            threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with low variance."""
        
        print("\n🔍 Removing low variance features...")
        
        # Scale first (variance threshold works better on scaled data)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_scaled)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        removed_features = X.columns[~selected_mask].tolist()
        
        print(f"   ✓ Kept {len(selected_features)} features, removed {len(removed_features)}")
        
        if removed_features:
            print(f"   Removed: {removed_features[:10]}{'...' if len(removed_features) > 10 else ''}")
        
        self.selection_results['variance_threshold'] = {
            'kept': len(selected_features),
            'removed': len(removed_features),
            'removed_features': removed_features
        }
        
        return X[selected_features], selected_features
    
    def remove_highly_correlated(self, X: pd.DataFrame,
                                 threshold: float = None) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        
        if threshold is None:
            threshold = eda_config.correlation_threshold
        
        print(f"\n🔍 Removing highly correlated features (threshold={threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        selected_features = [col for col in X.columns if col not in to_drop]
        
        print(f"   ✓ Kept {len(selected_features)} features, removed {len(to_drop)}")
        
        if to_drop:
            print(f"   Removed: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")
        
        self.selection_results['correlation'] = {
            'kept': len(selected_features),
            'removed': len(to_drop),
            'removed_features': to_drop
        }
        
        return X[selected_features], selected_features
    
    def select_by_importance(self, X: pd.DataFrame, y: pd.Series,
                            n_features: int = 30) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Select features using tree-based importance."""
        
        print(f"\n🔍 Selecting top {n_features} features by importance...")
        
        # Use Random Forest for importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=model_config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Handle any remaining NaNs
        X_clean = X.fillna(0)
        
        rf.fit(X_clean, y)
        
        # Get feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        selected_features = importance_df.head(n_features)['feature'].tolist()
        
        # Store importance for all features
        self.feature_importance = importance_df.set_index('feature')['importance'].to_dict()
        
        print(f"   ✓ Selected top {len(selected_features)} features")
        print(f"\n   Top 10 features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        self.selection_results['importance'] = {
            'n_features': len(selected_features),
            'top_features': importance_df.head(20).to_dict('records')
        }
        
        return X[selected_features], selected_features, self.feature_importance
    
    def select_by_mutual_information(self, X: pd.DataFrame, y: pd.Series,
                                     n_features: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using mutual information (good for non-linear relationships)."""
        
        print(f"\n🔍 Selecting features by mutual information...")
        
        X_clean = X.fillna(0)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_clean, y, random_state=model_config.random_state)
        
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        selected_features = mi_df.head(n_features)['feature'].tolist()
        
        print(f"   ✓ Selected top {len(selected_features)} features by MI")
        
        self.selection_results['mutual_information'] = {
            'n_features': len(selected_features),
            'top_features': mi_df.head(20).to_dict('records')
        }
        
        return X[selected_features], selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                      n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Perform recursive feature elimination."""
        
        print(f"\n🔍 Performing Recursive Feature Elimination (target: {n_features} features)...")
        
        X_clean = X.fillna(0)
        
        # Use Gradient Boosting for RFE
        estimator = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=model_config.random_state
        )
        
        rfe = RFE(estimator, n_features_to_select=n_features, step=5)
        rfe.fit(X_clean, y)
        
        selected_mask = rfe.support_
        selected_features = X.columns[selected_mask].tolist()
        
        # Get ranking
        ranking_df = pd.DataFrame({
            'feature': X.columns,
            'ranking': rfe.ranking_
        }).sort_values('ranking')
        
        print(f"   ✓ Selected {len(selected_features)} features via RFE")
        
        self.selection_results['rfe'] = {
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'ranking': ranking_df.head(30).to_dict('records')
        }
        
        return X[selected_features], selected_features
    
    def domain_based_selection(self, feature_list: List[str]) -> List[str]:
        """Apply domain knowledge for feature selection (rare disease context)."""
        
        print("\n🔍 Applying domain-based feature selection...")
        
        # Critical features for rare disease enrollment prediction
        priority_patterns = [
            'enrollment',  # Historical enrollments
            'power_score',  # HCP influence
            'writer',  # Writer/prescriber metrics
            'prescriber',
            'call',  # Engagement metrics
            'meeting',
            'segment',  # HCP segmentation
            'behavioral',  # Behavioral flags
            'retention',  # Retention metrics
            'goal',  # Territory goals
            'trend'  # Trend features
        ]
        
        # Ensure priority features are included
        priority_features = []
        for pattern in priority_patterns:
            matches = [f for f in feature_list if pattern.lower() in f.lower()]
            priority_features.extend(matches)
        
        priority_features = list(set(priority_features))
        
        print(f"   ✓ Identified {len(priority_features)} domain-priority features")
        
        self.selection_results['domain'] = {
            'priority_features': priority_features
        }
        
        return priority_features
    
    def plot_feature_importance(self, top_n: int = 30) -> str:
        """Plot feature importance visualization."""
        
        if not self.feature_importance:
            print("No feature importance calculated yet.")
            return None
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance.keys()),
            'importance': list(self.feature_importance.values())
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        bars = ax.barh(importance_df['feature'], importance_df['importance'],
                      color='steelblue', edgecolor='white')
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance Scores', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, importance_df['importance']):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        filepath = CHARTS_DIR / 'feature_importance.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filepath}")
        return str(filepath)
    
    def run_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                             final_n_features: int = 25) -> Tuple[pd.DataFrame, List[str]]:
        """Execute complete feature selection pipeline."""
        
        print("\n" + "=" * 80)
        print("STEP 4: FEATURE SELECTION")
        print("Selecting optimal features for rare disease enrollment prediction")
        print("=" * 80)
        
        print(f"\n📊 Starting with {X.shape[1]} features")
        
        # Step 1: Remove low variance
        X, features = self.remove_low_variance(X)
        
        # Step 2: Remove highly correlated
        X, features = self.remove_highly_correlated(X)
        
        # Step 3: Select by importance
        X_imp, features_imp, importance = self.select_by_importance(X, y, n_features=40)
        
        # Step 4: Apply domain knowledge
        priority_features = self.domain_based_selection(features_imp)
        
        # Step 5: Final selection - combine importance and domain
        # Ensure we have all priority features plus top remaining by importance
        remaining_slots = final_n_features - len(priority_features)
        
        if remaining_slots > 0:
            other_features = [f for f in features_imp if f not in priority_features]
            other_by_importance = sorted(
                other_features,
                key=lambda x: importance.get(x, 0),
                reverse=True
            )[:remaining_slots]
            final_features = list(set(priority_features + other_by_importance))
        else:
            # Sort priority features by importance and take top
            final_features = sorted(
                priority_features,
                key=lambda x: importance.get(x, 0),
                reverse=True
            )[:final_n_features]
        
        self.selected_features = final_features
        
        print(f"\n✓ Feature Selection Complete!")
        print(f"  Final feature count: {len(final_features)}")
        print(f"  Selected features: {final_features[:10]}...")
        
        # Plot importance
        self.plot_feature_importance()
        
        # Save selection results
        results_df = pd.DataFrame({
            'feature': final_features,
            'importance': [importance.get(f, 0) for f in final_features]
        }).sort_values('importance', ascending=False)
        
        results_path = OUTPUT_DIR / 'selected_features.csv'
        results_df.to_csv(results_path, index=False)
        print(f"  Selection results saved to: {results_path}")
        
        return X[final_features], final_features


def run_feature_selection(feature_df: pd.DataFrame, target: pd.Series):
    """Execute the complete feature selection process."""
    selector = FeatureSelector(feature_df)
    X_selected, features = selector.run_feature_selection(feature_df, target)
    return selector, X_selected, features


if __name__ == "__main__":
    print("Feature Selection module - run from main pipeline")

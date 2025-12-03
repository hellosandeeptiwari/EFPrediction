"""
Feature Engineering Module
==========================

Enterprise-grade feature engineering for rare disease enrollment prediction.
Includes:
- Temporal feature creation
- Aggregation features
- Lag features
- Interaction features
- Domain-specific features for rare disease pharma

RARE DISEASE SPECIFIC FEATURES:
- HCP Power Score (calculated from multiple engagement metrics)
- HCO Segmentation (hospital/clinic classification)
- Patient Adherence Score (refill patterns, early stops)
- HCP Influence Score (writing patterns, reach)
- Territory Performance Index
- Specialty Alignment Score
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, OUTPUT_DIR, 
    feature_config, rare_disease_config, data_config
)
from data_discovery import DataDiscovery


class FeatureEngineer:
    """
    Enterprise-grade feature engineering for enrollment prediction.
    
    Creates comprehensive feature set for rare disease pharmaceutical
    enrollment prediction model.
    """
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame] = None):
        self.data_dict = data_dict or {}
        self.feature_df = None
        self.feature_names = []
        
    def load_data(self):
        """Load data using DataDiscovery."""
        discovery = DataDiscovery()
        self.data_dict = discovery.load_all_data()
        return self
    
    def create_base_features(self) -> pd.DataFrame:
        """Create base feature dataset from HCP universe."""
        
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 60)
        
        if 'hcp_universe' not in self.data_dict:
            self.load_data()
        
        # Start with HCP universe as base
        base_df = self.data_dict['hcp_universe'].copy()
        
        print(f"\n📊 Base HCP Universe: {len(base_df):,} records")
        
        # NOTE: We do NOT encode hcp_segment as a feature since it IS the target
        # This prevents data leakage in the model
        
        # Region encoding
        if 'region_name' in base_df.columns:
            region_dummies = pd.get_dummies(base_df['region_name'], prefix='region')
            base_df = pd.concat([base_df, region_dummies], axis=1)
        
        self.feature_df = base_df
        return base_df
    
    # ==================== RARE DISEASE SPECIFIC FEATURES ====================
    
    def calculate_hcp_power_score(self) -> pd.DataFrame:
        """
        Calculate HCP Power Score - a composite metric indicating HCP's 
        influence and potential for rare disease enrollment.
        
        Components:
        - Writing history (current/past writer status)
        - Prescription volume (TRx)
        - Patient reach (unique patients)
        - Engagement responsiveness
        - Specialty alignment for rare disease
        """
        
        print("\n🔧 Calculating HCP Power Score...")
        
        df = self.feature_df.copy()
        
        # Initialize power score components
        power_components = []
        
        # NOTE: We do NOT use hcp_segment for power score as it leaks target info
        # Instead, we use behavioral and prescription-based indicators only
        
        # 1. Prescription Volume Score (0-25 points) - normalized TRx
        if 'target_trx' in df.columns:
            trx_normalized = df['target_trx'].fillna(0)
            trx_percentile = trx_normalized.rank(pct=True) * 25
            power_components.append(('trx_volume_score', trx_percentile))
            df['trx_volume_score'] = trx_percentile
        
        # 3. Engagement Score (0-20 points)
        if 'total_calls' in df.columns:
            calls_normalized = df['total_calls'].fillna(0).rank(pct=True) * 10
            df['calls_engagement_score'] = calls_normalized
        else:
            df['calls_engagement_score'] = 5
        
        if 'called_when_suggested_pct' in df.columns:
            responsiveness = df['called_when_suggested_pct'].fillna(0) / 100 * 10
            df['responsiveness_score'] = responsiveness
        else:
            df['responsiveness_score'] = 5
        
        df['engagement_score'] = df['calls_engagement_score'] + df['responsiveness_score']
        power_components.append(('engagement_score', df['engagement_score']))
        
        # 4. Specialty Alignment Score (0-15 points) - for rare disease
        if 'specialty' in df.columns:
            rare_disease_specialties = {
                'GASTROENTEROLOGY': 15,
                'HEPATOLOGY': 15,
                'INTERNAL MEDICINE': 10,
                'FAMILY MEDICINE': 8,
                'TRANSPLANT': 12,
                'ONCOLOGY': 10
            }
            specialty_score = df['specialty'].str.upper().map(rare_disease_specialties).fillna(5)
            power_components.append(('specialty_alignment_score', specialty_score))
            df['specialty_alignment_score'] = specialty_score
        else:
            df['specialty_alignment_score'] = 7.5
        
        # 5. Behavioral Indicators Score (0-15 points)
        behavioral_cols = ['refill_flag', 'switch_flag', 'line2_flag', 'line3_flag']
        available_behavioral = [c for c in behavioral_cols if c in df.columns]
        
        if available_behavioral:
            # Positive behaviors (refill) add points, risk behaviors (switch) reduce
            if 'refill_flag' in df.columns:
                df['refill_score'] = df['refill_flag'].fillna(0) * 8
            else:
                df['refill_score'] = 4
            
            if 'switch_flag' in df.columns:
                df['loyalty_score'] = (1 - df['switch_flag'].fillna(0)) * 7
            else:
                df['loyalty_score'] = 3.5
            
            df['behavioral_indicator_score'] = df['refill_score'] + df['loyalty_score']
        else:
            df['behavioral_indicator_score'] = 7.5
        
        power_components.append(('behavioral_indicator_score', df['behavioral_indicator_score']))
        
        # Calculate Final HCP Power Score (0-100)
        df['hcp_power_score_calculated'] = (
            df.get('writer_status_score', 12.5) +
            df.get('trx_volume_score', 12.5) +
            df['engagement_score'] +
            df['specialty_alignment_score'] +
            df['behavioral_indicator_score']
        )
        
        # Normalize to 0-1 scale
        df['hcp_power_score_normalized'] = df['hcp_power_score_calculated'] / 100
        
        # Create power score tiers
        df['hcp_power_tier'] = pd.cut(
            df['hcp_power_score_normalized'],
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=['Low', 'Medium', 'High', 'Elite']
        )
        
        print(f"   ✓ HCP Power Score calculated (mean: {df['hcp_power_score_normalized'].mean():.3f})")
        print(f"   ✓ Power Tier distribution:")
        print(df['hcp_power_tier'].value_counts().to_string().replace('\n', '\n      '))
        
        self.feature_df = df
        return df
    
    def calculate_hco_segmentation(self) -> pd.DataFrame:
        """
        Calculate HCO (Healthcare Organization) Segmentation features.
        
        Segments organizations based on:
        - Organization type (hospital, clinic, practice)
        - Size indicators
        - Rare disease focus potential
        - Geographic coverage
        """
        
        print("\n🔧 Creating HCO Segmentation features...")
        
        df = self.feature_df.copy()
        
        # Territory-level HCO analysis (since HCO data is at territory level)
        if 'territory_hierarchy' in self.data_dict:
            territory_df = self.data_dict['territory_hierarchy'].copy()
            
            # Create territory size indicators
            territory_counts = df.groupby('territory_name').size().reset_index(name='hcp_count_in_territory')
            
            df = df.merge(territory_counts, on='territory_name', how='left')
            
            # Territory size tier
            df['territory_size_tier'] = pd.cut(
                df['hcp_count_in_territory'].fillna(0),
                bins=[0, 50, 100, 200, 500, float('inf')],
                labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
            )
        
        # HCO Type Score based on enrollment patterns (from monthly_units)
        if 'monthly_units' in self.data_dict:
            units_df = self.data_dict['monthly_units'].copy()
            
            # Analyze channel distribution as HCO proxy
            if 'kpi_value_for__c' in units_df.columns:
                # HCO channel affinity
                channel_volume = units_df.groupby(['territory_name__c', 'kpi_value_for__c'])['kpi_value__c'].sum().unstack(fill_value=0)
                
                # Calculate channel concentration (HHI-like metric)
                channel_pcts = channel_volume.div(channel_volume.sum(axis=1), axis=0)
                channel_concentration = (channel_pcts ** 2).sum(axis=1)
                channel_concentration = channel_concentration.reset_index()
                channel_concentration.columns = ['territory_name', 'hco_channel_concentration']
                
                df = df.merge(channel_concentration, on='territory_name', how='left')
                
                # Dominant channel flag
                if 'HUB' in channel_volume.columns:
                    hub_pct = (channel_volume['HUB'] / channel_volume.sum(axis=1)).reset_index()
                    hub_pct.columns = ['territory_name', 'hco_hub_reliance_pct']
                    df = df.merge(hub_pct, on='territory_name', how='left')
        
        # HCO Rare Disease Alignment Score
        # Based on territory's historical rare disease enrollment success
        if 'monthly_kpis' in self.data_dict:
            kpi_df = self.data_dict['monthly_kpis']
            enrollments = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments']
            
            territory_enroll = enrollments.groupby('territory_name__c')['kpi_value__c'].agg(['sum', 'mean', 'std']).reset_index()
            territory_enroll.columns = ['territory_name', 'hco_total_enrollments', 'hco_avg_monthly_enrollments', 'hco_enrollment_volatility']
            
            df = df.merge(territory_enroll, on='territory_name', how='left')
            
            # HCO Rare Disease Score (normalized enrollment success)
            if 'hco_total_enrollments' in df.columns:
                df['hco_rare_disease_score'] = df['hco_total_enrollments'].rank(pct=True)
        
        # Fill NaN with defaults
        hco_cols = [c for c in df.columns if c.startswith('hco_')]
        for col in hco_cols:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        
        print(f"   ✓ Created {len(hco_cols)} HCO segmentation features")
        
        self.feature_df = df
        return df
    
    def calculate_patient_adherence_score(self) -> pd.DataFrame:
        """
        Calculate Patient Adherence Score - predicts how well patients
        in a territory/HCP's care adhere to rare disease treatment.
        
        Components:
        - Refill patterns
        - Early discontinuation rates
        - Treatment line progression
        - Switch behavior
        """
        
        print("\n🔧 Calculating Patient Adherence Score...")
        
        df = self.feature_df.copy()
        
        # Adherence Score Components (0-100 scale)
        adherence_score = pd.Series(50.0, index=df.index)  # Base score of 50
        
        # 1. Refill Behavior (+20 points max)
        if 'refill_flag' in df.columns:
            adherence_score += df['refill_flag'].fillna(0) * 20
            df['adherence_refill_component'] = df['refill_flag'].fillna(0) * 20
        
        # 2. Early Stop Penalty (-25 points max)
        if 'early_stop_flag' in df.columns:
            adherence_score -= df['early_stop_flag'].fillna(0) * 25
            df['adherence_early_stop_penalty'] = -df['early_stop_flag'].fillna(0) * 25
        
        # 3. Switch Behavior Penalty (-15 points max)
        if 'switch_flag' in df.columns:
            adherence_score -= df['switch_flag'].fillna(0) * 15
            df['adherence_switch_penalty'] = -df['switch_flag'].fillna(0) * 15
        
        # 4. Treatment Line Progression (+15 points for staying on therapy)
        line_flags = ['line2_flag', 'line3_flag']
        available_lines = [c for c in line_flags if c in df.columns]
        if available_lines:
            # Patients progressing through lines indicates ongoing treatment
            line_progression = df[available_lines].sum(axis=1)
            adherence_score += (line_progression > 0).astype(int) * 15
            df['adherence_line_progression'] = (line_progression > 0).astype(int) * 15
        
        # 5. Territory-level adherence patterns
        if 'writers_prescribers' in self.data_dict:
            wp_df = self.data_dict['writers_prescribers']
            
            # Repeat writers indicate better patient adherence
            repeat_ratio = wp_df.groupby('territory_name__c').apply(
                lambda x: x['number_of_repeat_writers__c'].sum() / (x['number_of_new_writers__c'].sum() + 1)
            ).reset_index()
            repeat_ratio.columns = ['territory_name', 'territory_writer_retention_ratio']
            
            df = df.merge(repeat_ratio, on='territory_name', how='left')
            
            # Add retention bonus to adherence score
            retention_bonus = df['territory_writer_retention_ratio'].fillna(0.5).clip(0, 1) * 10
            adherence_score += retention_bonus
            df['adherence_territory_bonus'] = retention_bonus
        
        # Normalize to 0-100 and then 0-1
        adherence_score = adherence_score.clip(0, 100)
        df['patient_adherence_score'] = adherence_score
        df['patient_adherence_score_normalized'] = adherence_score / 100
        
        # Adherence Risk Categories
        df['adherence_risk_category'] = pd.cut(
            df['patient_adherence_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['High Risk', 'Moderate Risk', 'Low Risk', 'Excellent']
        )
        
        print(f"   ✓ Patient Adherence Score calculated (mean: {df['patient_adherence_score'].mean():.1f})")
        print(f"   ✓ Adherence Risk distribution:")
        print(df['adherence_risk_category'].value_counts().to_string().replace('\n', '\n      '))
        
        self.feature_df = df
        return df
    
    def calculate_hcp_influence_score(self) -> pd.DataFrame:
        """
        Calculate HCP Influence/Reach Score - measures the potential
        impact of an HCP on rare disease enrollment.
        
        Components:
        - Patient volume (unique patients)
        - Prescription volume
        - Referral network proxy
        - KOL indicators
        """
        
        print("\n🔧 Calculating HCP Influence Score...")
        
        df = self.feature_df.copy()
        
        influence_score = pd.Series(0.0, index=df.index)
        
        # 1. Prescription Influence (0-30 points)
        if 'target_trx' in df.columns:
            trx_pct = df['target_trx'].fillna(0).rank(pct=True) * 30
            influence_score += trx_pct
            df['influence_rx_component'] = trx_pct
        
        if 'shared_trx' in df.columns:
            shared_pct = df['shared_trx'].fillna(0).rank(pct=True) * 10
            influence_score += shared_pct
            df['influence_shared_rx_component'] = shared_pct
        
        # 2. Engagement Reach (0-25 points)
        if 'total_calls' in df.columns:
            # High call volume indicates active engagement
            call_pct = df['total_calls'].fillna(0).rank(pct=True) * 15
            influence_score += call_pct
            df['influence_call_component'] = call_pct
        
        if 'territory_number_of_meetings_sum' in df.columns:
            meeting_rank = df['territory_number_of_meetings_sum'].fillna(0).rank(pct=True) * 10
            influence_score += meeting_rank
            df['influence_meeting_component'] = meeting_rank
        
        # 3. Writer Network Effect (0-20 points)
        if 'new_writers_sum' in df.columns and 'repeat_writers_sum' in df.columns:
            total_writers = df['new_writers_sum'].fillna(0) + df['repeat_writers_sum'].fillna(0)
            writer_pct = total_writers.rank(pct=True) * 20
            influence_score += writer_pct
            df['influence_network_component'] = writer_pct
        
        # 4. Specialty Leadership (0-15 points) - KOL proxy
        if 'specialty' in df.columns:
            kol_specialties = {
                'GASTROENTEROLOGY': 15,
                'HEPATOLOGY': 15,
                'TRANSPLANT': 12,
                'INTERNAL MEDICINE': 8
            }
            specialty_influence = df['specialty'].str.upper().map(kol_specialties).fillna(5)
            influence_score += specialty_influence
            df['influence_specialty_component'] = specialty_influence
        
        # Normalize to 0-100
        max_possible = 100
        influence_score = (influence_score / max_possible * 100).clip(0, 100)
        
        df['hcp_influence_score'] = influence_score
        df['hcp_influence_score_normalized'] = influence_score / 100
        
        # Influence Tiers
        df['hcp_influence_tier'] = pd.cut(
            df['hcp_influence_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low Influence', 'Moderate Influence', 'High Influence', 'Key Opinion Leader']
        )
        
        print(f"   ✓ HCP Influence Score calculated (mean: {df['hcp_influence_score'].mean():.1f})")
        
        self.feature_df = df
        return df
    
    def calculate_territory_performance_index(self) -> pd.DataFrame:
        """
        Calculate Territory Performance Index - composite metric for
        territory-level rare disease performance.
        """
        
        print("\n🔧 Calculating Territory Performance Index...")
        
        df = self.feature_df.copy()
        
        # Territory metrics aggregation
        territory_metrics = df.groupby('territory_name').agg({
            'hcp_power_score_normalized': 'mean',
            'patient_adherence_score_normalized': 'mean',
            'hcp_influence_score_normalized': 'mean' if 'hcp_influence_score_normalized' in df.columns else 'first'
        }).reset_index()
        
        territory_metrics.columns = [
            'territory_name', 
            'territory_avg_power_score',
            'territory_avg_adherence_score',
            'territory_avg_influence_score'
        ]
        
        # Calculate Territory Performance Index
        territory_metrics['territory_performance_index'] = (
            territory_metrics['territory_avg_power_score'] * 0.35 +
            territory_metrics['territory_avg_adherence_score'] * 0.35 +
            territory_metrics['territory_avg_influence_score'] * 0.30
        )
        
        # Rank territories
        territory_metrics['territory_performance_rank'] = territory_metrics['territory_performance_index'].rank(ascending=False)
        territory_metrics['territory_performance_percentile'] = territory_metrics['territory_performance_index'].rank(pct=True)
        
        df = df.merge(territory_metrics, on='territory_name', how='left')
        
        print(f"   ✓ Territory Performance Index calculated")
        
        self.feature_df = df
        return df
    
    def calculate_specialty_alignment_features(self) -> pd.DataFrame:
        """
        Create specialty-specific features for rare disease targeting.
        """
        
        print("\n🔧 Creating Specialty Alignment features...")
        
        df = self.feature_df.copy()
        
        if 'specialty' not in df.columns:
            print("   ⚠ Specialty column not found, skipping specialty features")
            return df
        
        # Rare disease specialty tiers
        specialty_tiers = {
            # Tier 1 - Primary rare disease specialists
            'GASTROENTEROLOGY': 1,
            'HEPATOLOGY': 1,
            'TRANSPLANT': 1,
            # Tier 2 - Secondary specialists
            'INTERNAL MEDICINE': 2,
            'FAMILY MEDICINE': 2,
            'ONCOLOGY': 2,
            # Tier 3 - Supporting specialists
            'PEDIATRICS': 3,
            'SURGERY': 3
        }
        
        df['specialty_tier'] = df['specialty'].str.upper().map(specialty_tiers).fillna(4)
        
        # One-hot encode specialty for modeling
        specialty_dummies = pd.get_dummies(df['specialty'].str.upper(), prefix='specialty')
        df = pd.concat([df, specialty_dummies], axis=1)
        
        # Specialty concentration in territory
        specialty_counts = df.groupby(['territory_name', 'specialty']).size().unstack(fill_value=0)
        specialty_concentration = specialty_counts.div(specialty_counts.sum(axis=1), axis=0)
        
        # Get dominant specialty per territory
        df['territory_dominant_specialty'] = df.groupby('territory_name')['specialty'].transform(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'UNKNOWN'
        )
        
        print(f"   ✓ Created specialty alignment features")
        
        self.feature_df = df
        return df
    
    def create_engagement_features(self) -> pd.DataFrame:
        """Create engagement-based features from calls and meetings data."""
        
        print("\n🔧 Creating engagement features...")
        
        if self.feature_df is None:
            self.create_base_features()
        
        # Add territory-level engagement metrics
        if 'monthly_calls_territory' in self.data_dict:
            calls_df = self.data_dict['monthly_calls_territory'].copy()
            
            # Aggregate by territory
            territory_engagement = calls_df.groupby('territory_name__c').agg({
                'calls_to_hcps__c': ['sum', 'mean', 'std', 'max'],
                'calls_to_targets__c': ['sum', 'mean', 'max'],
                'num_of_hcps_called_on__c': ['sum', 'mean', 'max'],
                'num_of_targets_called_on__c': ['sum', 'mean', 'max']
            }).round(2)
            
            # Flatten column names
            territory_engagement.columns = [
                f'territory_{col[0].replace("__c", "")}_{col[1]}'
                for col in territory_engagement.columns
            ]
            territory_engagement = territory_engagement.reset_index()
            territory_engagement.rename(columns={'territory_name__c': 'territory_name'}, inplace=True)
            
            # Merge with base features
            self.feature_df = self.feature_df.merge(
                territory_engagement,
                on='territory_name',
                how='left'
            )
            
            print(f"   ✓ Added {len(territory_engagement.columns) - 1} territory engagement features")
        
        # Add meeting features
        if 'monthly_meetings' in self.data_dict:
            meetings_df = self.data_dict['monthly_meetings'].copy()
            
            territory_meetings = meetings_df.groupby('territory_name__c').agg({
                'number_of_meetings__c': ['sum', 'mean', 'max'],
                'number_of_attendees__c': ['sum', 'mean', 'max']
            }).round(2)
            
            territory_meetings.columns = [
                f'territory_{col[0].replace("__c", "")}_{col[1]}'
                for col in territory_meetings.columns
            ]
            territory_meetings = territory_meetings.reset_index()
            territory_meetings.rename(columns={'territory_name__c': 'territory_name'}, inplace=True)
            
            self.feature_df = self.feature_df.merge(
                territory_meetings,
                on='territory_name',
                how='left'
            )
            
            print(f"   ✓ Added {len(territory_meetings.columns) - 1} meeting features")
        
        return self.feature_df
    
    def create_enrollment_history_features(self) -> pd.DataFrame:
        """Create historical enrollment features by territory."""
        
        print("\n🔧 Creating enrollment history features...")
        
        if 'monthly_kpis' in self.data_dict:
            kpi_df = self.data_dict['monthly_kpis'].copy()
            enrollment_df = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments'].copy()
            enrollment_df['month_begin_date__c'] = pd.to_datetime(enrollment_df['month_begin_date__c'])
            
            # Total historical enrollments by territory
            territory_enrollments = enrollment_df.groupby('territory_name__c').agg({
                'kpi_value__c': ['sum', 'mean', 'std', 'count', 'max']
            }).round(2)
            
            territory_enrollments.columns = [
                f'territory_enrollment_{col[1]}'
                for col in territory_enrollments.columns
            ]
            territory_enrollments = territory_enrollments.reset_index()
            territory_enrollments.rename(columns={'territory_name__c': 'territory_name'}, inplace=True)
            
            # Recent enrollment trends (last 6 months, last 12 months)
            enrollment_df = enrollment_df.sort_values('month_begin_date__c')
            max_date = enrollment_df['month_begin_date__c'].max()
            
            # Last 6 months
            recent_6m = enrollment_df[
                enrollment_df['month_begin_date__c'] >= max_date - pd.DateOffset(months=6)
            ]
            territory_recent_6m = recent_6m.groupby('territory_name__c')['kpi_value__c'].agg([
                'sum', 'mean'
            ]).round(2)
            territory_recent_6m.columns = ['enrollment_6m_sum', 'enrollment_6m_mean']
            territory_recent_6m = territory_recent_6m.reset_index()
            territory_recent_6m.rename(columns={'territory_name__c': 'territory_name'}, inplace=True)
            
            # Last 12 months
            recent_12m = enrollment_df[
                enrollment_df['month_begin_date__c'] >= max_date - pd.DateOffset(months=12)
            ]
            territory_recent_12m = recent_12m.groupby('territory_name__c')['kpi_value__c'].agg([
                'sum', 'mean'
            ]).round(2)
            territory_recent_12m.columns = ['enrollment_12m_sum', 'enrollment_12m_mean']
            territory_recent_12m = territory_recent_12m.reset_index()
            territory_recent_12m.rename(columns={'territory_name__c': 'territory_name'}, inplace=True)
            
            # Channel mix features
            channel_mix = enrollment_df.pivot_table(
                index='territory_name__c',
                columns='kpi_value_for__c',
                values='kpi_value__c',
                aggfunc='sum'
            ).fillna(0)
            channel_mix.columns = [f'enrollment_channel_{col.replace(" ", "_").lower()}' 
                                   for col in channel_mix.columns]
            channel_mix = channel_mix.reset_index()
            channel_mix.rename(columns={'territory_name__c': 'territory_name'}, inplace=True)
            
            # Merge all enrollment features
            self.feature_df = self.feature_df.merge(
                territory_enrollments, on='territory_name', how='left'
            )
            self.feature_df = self.feature_df.merge(
                territory_recent_6m, on='territory_name', how='left'
            )
            self.feature_df = self.feature_df.merge(
                territory_recent_12m, on='territory_name', how='left'
            )
            self.feature_df = self.feature_df.merge(
                channel_mix, on='territory_name', how='left'
            )
            
            print(f"   ✓ Added enrollment history features (total, 6m, 12m trends, channel mix)")
        
        return self.feature_df
    
    def create_writer_prescriber_features(self) -> pd.DataFrame:
        """Create writer and prescriber count features."""
        
        print("\n🔧 Creating writer/prescriber features...")
        
        if 'writers_prescribers' in self.data_dict:
            wp_df = self.data_dict['writers_prescribers'].copy()
            
            territory_writers = wp_df.groupby('territory_name__c').agg({
                'number_of_new_writers__c': ['sum', 'mean'],
                'number_of_repeat_writers__c': ['sum', 'mean'],
                'number_of_new_prescribers__c': ['sum', 'mean'],
                'number_of_repeat_prescribers__c': ['sum', 'mean']
            }).round(2)
            
            territory_writers.columns = [
                f'{col[0].replace("__c", "").replace("number_of_", "")}_{col[1]}'
                for col in territory_writers.columns
            ]
            territory_writers = territory_writers.reset_index()
            territory_writers.rename(columns={'territory_name__c': 'territory_name'}, inplace=True)
            
            # Create derived features
            territory_writers['writer_retention_rate'] = (
                territory_writers['repeat_writers_sum'] / 
                (territory_writers['new_writers_sum'] + territory_writers['repeat_writers_sum'] + 0.1)
            ).round(4)
            
            territory_writers['prescriber_retention_rate'] = (
                territory_writers['repeat_prescribers_sum'] / 
                (territory_writers['new_prescribers_sum'] + territory_writers['repeat_prescribers_sum'] + 0.1)
            ).round(4)
            
            self.feature_df = self.feature_df.merge(
                territory_writers, on='territory_name', how='left'
            )
            
            print(f"   ✓ Added writer/prescriber features with retention rates")
        
        return self.feature_df
    
    def create_email_features(self) -> pd.DataFrame:
        """Create email engagement features."""
        
        print("\n🔧 Creating email engagement features...")
        
        if 'monthly_email' in self.data_dict:
            email_df = self.data_dict['monthly_email'].copy()
            
            territory_email = email_df.groupby('territory_name__c').agg({
                'num_of_emails_sent__c': ['sum', 'mean', 'std', 'max']
            }).round(2)
            
            territory_email.columns = [
                f'email_{col[1]}'
                for col in territory_email.columns
            ]
            territory_email = territory_email.reset_index()
            territory_email.rename(columns={'territory_name__c': 'territory_name'}, inplace=True)
            
            self.feature_df = self.feature_df.merge(
                territory_email, on='territory_name', how='left'
            )
            
            print(f"   ✓ Added email engagement features")
        
        return self.feature_df
    
    def create_goal_features(self) -> pd.DataFrame:
        """Create goal and target features."""
        
        print("\n🔧 Creating goal/target features...")
        
        if 'tbm_goals' in self.data_dict:
            goals_df = self.data_dict['tbm_goals'].copy()
            
            # Get latest goals by territory
            latest_goals = goals_df.sort_values('file_load_time__v', ascending=False)
            latest_goals = latest_goals.drop_duplicates(subset=['territory_name__c'])
            
            goal_features = latest_goals[[
                'territory_name__c', 'total_units_baseline1__c', 'total_units_baseline2__c',
                'initial_goals__c', 'rbd_adjustments__c', 'final_goals__c'
            ]].copy()
            
            goal_features.columns = ['territory_name', 'units_baseline1', 'units_baseline2',
                                    'initial_goal', 'rbd_adjustment', 'final_goal']
            
            # Create derived goal features
            goal_features['goal_adjustment_pct'] = (
                goal_features['rbd_adjustment'] / (goal_features['initial_goal'] + 0.1) * 100
            ).round(2)
            
            goal_features['goal_vs_baseline1_ratio'] = (
                goal_features['final_goal'] / (goal_features['units_baseline1'] + 0.1)
            ).round(4)
            
            self.feature_df = self.feature_df.merge(
                goal_features, on='territory_name', how='left'
            )
            
            print(f"   ✓ Added goal/target features")
        
        return self.feature_df
    
    def create_derived_features(self) -> pd.DataFrame:
        """Create derived and interaction features."""
        
        print("\n🔧 Creating derived and interaction features...")
        
        df = self.feature_df.copy()
        
        # HCP-level derived features
        if 'total_calls' in df.columns and 'tenure_years' in df.columns:
            df['calls_per_year'] = (df['total_calls'] / (df['tenure_years'] + 0.1)).round(2)
        
        if 'target_trx' in df.columns and 'shared_trx' in df.columns:
            df['trx_share_pct'] = (
                df['shared_trx'] / (df['target_trx'] + df['shared_trx'] + 0.1) * 100
            ).round(2)
        
        if 'power_score' in df.columns:
            df['power_score_positive'] = (df['power_score'] > 0).astype(int)
            df['power_score_quartile'] = pd.qcut(
                df['power_score'].rank(method='first'), 
                q=4, 
                labels=['Q1', 'Q2', 'Q3', 'Q4']
            )
        
        # Behavioral flag combinations (important for rare disease)
        flag_cols = ['refill_flag', 'early_stop_flag', 'switch_flag', 'line2_flag', 'line3_flag']
        available_flags = [col for col in flag_cols if col in df.columns]
        
        if len(available_flags) >= 2:
            # Total behavioral score
            df['behavioral_score'] = df[available_flags].sum(axis=1)
            
            # Risk indicators
            if 'early_stop_flag' in df.columns and 'switch_flag' in df.columns:
                df['discontinuation_risk'] = df['early_stop_flag'] + df['switch_flag']
        
        # Engagement intensity
        if 'called_when_suggested_pct' in df.columns:
            df['high_responsiveness'] = (df['called_when_suggested_pct'] >= 70).astype(int)
        
        # Territory performance indicators
        if 'territory_enrollment_sum' in df.columns and 'territory_calls_to_hcps_sum' in df.columns:
            df['enrollment_per_call'] = (
                df['territory_enrollment_sum'] / (df['territory_calls_to_hcps_sum'] + 0.1)
            ).round(4)
        
        # Trend features
        if 'enrollment_6m_sum' in df.columns and 'enrollment_12m_sum' in df.columns:
            df['enrollment_trend_6m_vs_12m'] = (
                df['enrollment_6m_sum'] / (df['enrollment_12m_sum'] + 0.1)
            ).round(4)
        
        self.feature_df = df
        print(f"   ✓ Created derived and interaction features")
        
        return self.feature_df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values appropriately."""
        
        print("\n🔧 Handling missing values...")
        
        df = self.feature_df.copy()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fill missing with 0 for count/sum columns
        zero_fill_patterns = ['sum', 'count', 'flag', 'enrollment_channel']
        for col in numeric_cols:
            if any(pattern in col.lower() for pattern in zero_fill_patterns):
                df[col] = df[col].fillna(0)
        
        # Fill missing with median for other numeric columns
        remaining_missing = df[numeric_cols].isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]
        
        for col in remaining_missing.index:
            df[col] = df[col].fillna(df[col].median())
        
        self.feature_df = df
        
        total_missing = self.feature_df.isnull().sum().sum()
        print(f"   ✓ Remaining missing values: {total_missing}")
        
        return self.feature_df
    
    def get_feature_list(self) -> List[str]:
        """Get list of engineered features."""
        
        if self.feature_df is None:
            return []
        
        # Exclude non-feature columns and target-leaking columns
        exclude_cols = [
            'hcp_id', 'hcp_segment', 'territory_name', 'region_name',
            'power_score_quartile',  # categorical
            'hcp_segment_encoded',   # LEAKS TARGET - do not use
            'writer_status_score',   # LEAKS TARGET - derived from hcp_segment
            'target'                 # target variable
        ]
        
        feature_cols = [
            col for col in self.feature_df.columns
            if col not in exclude_cols and 
            self.feature_df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]
        
        self.feature_names = feature_cols
        return feature_cols
    
    def run_feature_engineering(self) -> pd.DataFrame:
        """Execute complete feature engineering pipeline."""
        
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE ENGINEERING")
        print("Creating enterprise-grade features for rare disease enrollment prediction")
        print("=" * 80)
        
        if not self.data_dict:
            self.load_data()
        
        # Build features step by step
        self.create_base_features()
        self.create_engagement_features()
        self.create_enrollment_history_features()
        self.create_writer_prescriber_features()
        self.create_email_features()
        self.create_goal_features()
        
        # ===== RARE DISEASE SPECIFIC FEATURES =====
        print("\n" + "-" * 60)
        print("RARE DISEASE SPECIFIC FEATURE ENGINEERING")
        print("-" * 60)
        
        self.calculate_hcp_power_score()
        self.calculate_hco_segmentation()
        self.calculate_patient_adherence_score()
        self.calculate_hcp_influence_score()
        self.calculate_territory_performance_index()
        self.calculate_specialty_alignment_features()
        
        # ===== DERIVED AND INTERACTION FEATURES =====
        self.create_derived_features()
        self.handle_missing_values()
        
        # Get feature list
        features = self.get_feature_list()
        
        # Print feature summary by category
        print(f"\n✓ Feature Engineering Complete!")
        print(f"  Total records: {len(self.feature_df):,}")
        print(f"  Total features: {len(features)}")
        
        # Categorize features
        power_features = [f for f in features if 'power' in f.lower()]
        adherence_features = [f for f in features if 'adherence' in f.lower()]
        influence_features = [f for f in features if 'influence' in f.lower()]
        hco_features = [f for f in features if 'hco' in f.lower()]
        territory_features = [f for f in features if 'territory' in f.lower()]
        enrollment_features = [f for f in features if 'enrollment' in f.lower()]
        
        print(f"\n  Feature Categories:")
        print(f"    • HCP Power Score features: {len(power_features)}")
        print(f"    • Patient Adherence features: {len(adherence_features)}")
        print(f"    • HCP Influence features: {len(influence_features)}")
        print(f"    • HCO Segmentation features: {len(hco_features)}")
        print(f"    • Territory features: {len(territory_features)}")
        print(f"    • Enrollment features: {len(enrollment_features)}")
        
        # Save features
        feature_path = OUTPUT_DIR / 'engineered_features.csv'
        self.feature_df.to_csv(feature_path, index=False)
        print(f"\n  Features saved to: {feature_path}")
        
        return self.feature_df


def run_feature_engineering():
    """Execute the complete feature engineering process."""
    fe = FeatureEngineer()
    df = fe.run_feature_engineering()
    return fe, df


if __name__ == "__main__":
    fe, df = run_feature_engineering()

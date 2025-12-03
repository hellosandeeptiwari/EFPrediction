"""
Exploratory Data Analysis (EDA) Module
======================================

Comprehensive EDA for rare disease pharmaceutical enrollment prediction.
Includes:
- Statistical analysis
- Distribution analysis
- Correlation analysis
- Rare disease specific tests
- Visualization generation for leadership presentations
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, spearmanr
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, CHARTS_DIR, OUTPUT_DIR, 
    eda_config, rare_disease_config, data_config
)
from data_discovery import DataDiscovery


class RareDiseaseEDA:
    """
    Enterprise-grade EDA for rare disease enrollment prediction.
    
    Provides comprehensive statistical analysis and visualizations
    specifically designed for rare disease pharmaceutical context.
    """
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame] = None):
        self.data_dict = data_dict or {}
        self.insights = {}
        self.charts_generated = []
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = eda_config.color_palette
        
    def load_data(self):
        """Load data using DataDiscovery."""
        discovery = DataDiscovery()
        self.data_dict = discovery.load_all_data()
        return self
    
    # ==================== STATISTICAL TESTS ====================
    
    def test_normality(self, data: pd.Series, name: str = "variable") -> dict:
        """Test for normality using multiple tests."""
        clean_data = data.dropna()
        
        if len(clean_data) < 8:
            return {'error': 'Insufficient data for normality test'}
        
        # Sample if data is too large
        if len(clean_data) > 5000:
            clean_data = clean_data.sample(5000, random_state=42)
        
        results = {
            'variable': name,
            'n_samples': len(clean_data),
            'mean': round(clean_data.mean(), 4),
            'std': round(clean_data.std(), 4),
            'skewness': round(clean_data.skew(), 4),
            'kurtosis': round(clean_data.kurtosis(), 4)
        }
        
        # Shapiro-Wilk test
        try:
            stat, p_value = stats.shapiro(clean_data[:5000])
            results['shapiro_wilk'] = {
                'statistic': round(stat, 4),
                'p_value': round(p_value, 6),
                'is_normal': p_value > eda_config.significance_level
            }
        except Exception as e:
            results['shapiro_wilk'] = {'error': str(e)}
        
        # D'Agostino-Pearson test
        try:
            if len(clean_data) >= 20:
                stat, p_value = stats.normaltest(clean_data)
                results['dagostino_pearson'] = {
                    'statistic': round(stat, 4),
                    'p_value': round(p_value, 6),
                    'is_normal': p_value > eda_config.significance_level
                }
        except Exception as e:
            results['dagostino_pearson'] = {'error': str(e)}
        
        return results
    
    def test_group_differences(self, data: pd.DataFrame, 
                               numeric_col: str, 
                               group_col: str) -> dict:
        """Test for significant differences between groups (rare disease context)."""
        
        results = {
            'numeric_variable': numeric_col,
            'grouping_variable': group_col
        }
        
        # Get groups
        groups = [group[numeric_col].dropna().values 
                  for name, group in data.groupby(group_col)]
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        # Kruskal-Wallis H-test (non-parametric, good for rare disease data)
        try:
            stat, p_value = kruskal(*groups)
            results['kruskal_wallis'] = {
                'statistic': round(stat, 4),
                'p_value': round(p_value, 6),
                'significant': p_value < eda_config.significance_level,
                'interpretation': 'Groups differ significantly' if p_value < eda_config.significance_level else 'No significant difference'
            }
        except Exception as e:
            results['kruskal_wallis'] = {'error': str(e)}
        
        return results
    
    def test_rare_event_association(self, data: pd.DataFrame,
                                    var1: str, var2: str) -> dict:
        """Test association between categorical variables (Chi-square with Yates correction for rare events)."""
        
        contingency_table = pd.crosstab(data[var1], data[var2])
        
        # Use Yates correction for rare events (small cell counts)
        min_expected = contingency_table.min().min()
        use_yates = min_expected < 5
        
        try:
            chi2, p_value, dof, expected = chi2_contingency(
                contingency_table, 
                correction=use_yates
            )
            
            # Cramér's V for effect size
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            return {
                'test': 'Chi-square',
                'variables': [var1, var2],
                'chi2_statistic': round(chi2, 4),
                'p_value': round(p_value, 6),
                'degrees_of_freedom': dof,
                'yates_correction': use_yates,
                'cramers_v': round(cramers_v, 4),
                'effect_size': 'small' if cramers_v < 0.1 else 'medium' if cramers_v < 0.3 else 'large',
                'significant': p_value < eda_config.significance_level
            }
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== ENROLLMENT ANALYSIS ====================
    
    def analyze_enrollments(self) -> dict:
        """Comprehensive enrollment analysis for rare disease context."""
        
        print("\n" + "=" * 60)
        print("ENROLLMENT ANALYSIS - Rare Disease Context")
        print("=" * 60)
        
        results = {}
        
        if 'monthly_kpis' not in self.data_dict:
            return {'error': 'Monthly KPIs data not loaded'}
        
        kpi_df = self.data_dict['monthly_kpis'].copy()
        enrollment_df = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments'].copy()
        
        # Parse dates if not already
        if not pd.api.types.is_datetime64_any_dtype(enrollment_df['month_begin_date__c']):
            enrollment_df['month_begin_date__c'] = pd.to_datetime(enrollment_df['month_begin_date__c'])
        
        # Basic statistics
        results['basic_stats'] = {
            'total_enrollment_records': len(enrollment_df),
            'total_enrollments': int(enrollment_df['kpi_value__c'].sum()),
            'mean_monthly_enrollment': round(enrollment_df['kpi_value__c'].mean(), 2),
            'median_monthly_enrollment': round(enrollment_df['kpi_value__c'].median(), 2),
            'std_monthly_enrollment': round(enrollment_df['kpi_value__c'].std(), 2),
            'max_monthly_enrollment': int(enrollment_df['kpi_value__c'].max()),
            'min_date': str(enrollment_df['month_begin_date__c'].min().date()),
            'max_date': str(enrollment_df['month_begin_date__c'].max().date())
        }
        
        print(f"\n📊 Basic Enrollment Statistics:")
        for key, value in results['basic_stats'].items():
            print(f"   {key}: {value}")
        
        # Channel analysis
        results['channel_analysis'] = enrollment_df.groupby('kpi_value_for__c')['kpi_value__c'].agg([
            'sum', 'mean', 'count'
        ]).round(2).to_dict('index')
        
        print(f"\n📊 Enrollment by Channel:")
        for channel, stats in results['channel_analysis'].items():
            print(f"   {channel}: Total={stats['sum']:.0f}, Avg={stats['mean']:.2f}")
        
        # Territory analysis
        territory_enrollments = enrollment_df.groupby('territory_name__c')['kpi_value__c'].sum()
        results['territory_analysis'] = {
            'unique_territories': territory_enrollments.nunique(),
            'top_5_territories': territory_enrollments.nlargest(5).to_dict(),
            'bottom_5_territories': territory_enrollments.nsmallest(5).to_dict()
        }
        
        # Temporal patterns
        enrollment_df['year'] = enrollment_df['month_begin_date__c'].dt.year
        enrollment_df['month'] = enrollment_df['month_begin_date__c'].dt.month
        
        yearly_trend = enrollment_df.groupby('year')['kpi_value__c'].sum()
        results['temporal_analysis'] = {
            'yearly_enrollments': yearly_trend.to_dict(),
            'yoy_growth': yearly_trend.pct_change().dropna().to_dict()
        }
        
        # Rare disease specific: Check for zero-inflation
        zero_enrollments = (enrollment_df['kpi_value__c'] == 0).sum()
        results['rare_disease_indicators'] = {
            'zero_enrollment_records': int(zero_enrollments),
            'zero_enrollment_pct': round(zero_enrollments / len(enrollment_df) * 100, 2),
            'is_zero_inflated': zero_enrollments / len(enrollment_df) > 0.3
        }
        
        print(f"\n📊 Rare Disease Indicators:")
        print(f"   Zero-inflation: {results['rare_disease_indicators']['zero_enrollment_pct']}%")
        
        self.insights['enrollments'] = results
        return results
    
    def analyze_hcp_segments(self) -> dict:
        """Analyze HCP segments and their enrollment potential."""
        
        print("\n" + "=" * 60)
        print("HCP SEGMENT ANALYSIS")
        print("=" * 60)
        
        if 'hcp_universe' not in self.data_dict:
            return {'error': 'HCP universe data not loaded'}
        
        hcp_df = self.data_dict['hcp_universe'].copy()
        results = {}
        
        # Segment distribution
        segment_dist = hcp_df['hcp_segment'].value_counts()
        results['segment_distribution'] = segment_dist.to_dict()
        
        print(f"\n📊 HCP Segment Distribution:")
        for segment, count in results['segment_distribution'].items():
            print(f"   {segment}: {count:,} ({count/len(hcp_df)*100:.1f}%)")
        
        # Segment characteristics
        numeric_cols = ['power_score', 'target_trx', 'total_calls', 'specialty_score', 
                       'tenure_years', 'referral_score', 'switch_flag', 'early_stop_flag']
        
        available_cols = [col for col in numeric_cols if col in hcp_df.columns]
        
        segment_stats = hcp_df.groupby('hcp_segment')[available_cols].agg(['mean', 'std']).round(2)
        results['segment_characteristics'] = segment_stats.to_dict()
        
        # Test if power_score differs by segment
        if 'power_score' in hcp_df.columns:
            test_result = self.test_group_differences(hcp_df, 'power_score', 'hcp_segment')
            results['power_score_by_segment_test'] = test_result
            
            print(f"\n📊 Power Score Difference by Segment:")
            print(f"   Kruskal-Wallis p-value: {test_result.get('kruskal_wallis', {}).get('p_value', 'N/A')}")
            print(f"   Interpretation: {test_result.get('kruskal_wallis', {}).get('interpretation', 'N/A')}")
        
        # Regional analysis
        if 'region_name' in hcp_df.columns:
            regional_segments = pd.crosstab(hcp_df['region_name'], hcp_df['hcp_segment'])
            results['regional_segment_distribution'] = regional_segments.to_dict()
        
        self.insights['hcp_segments'] = results
        return results
    
    def analyze_call_effectiveness(self) -> dict:
        """Analyze sales call effectiveness for rare disease context."""
        
        print("\n" + "=" * 60)
        print("CALL EFFECTIVENESS ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        if 'monthly_calls_territory' in self.data_dict:
            calls_df = self.data_dict['monthly_calls_territory'].copy()
            
            # Call volume statistics
            results['call_volume_stats'] = {
                'total_calls_to_hcps': int(calls_df['calls_to_hcps__c'].sum()),
                'total_calls_to_targets': int(calls_df['calls_to_targets__c'].sum()),
                'avg_calls_per_territory_month': round(calls_df['calls_to_hcps__c'].mean(), 2),
                'target_coverage_rate': round(
                    calls_df['calls_to_targets__c'].sum() / calls_df['calls_to_hcps__c'].sum() * 100, 2
                ) if calls_df['calls_to_hcps__c'].sum() > 0 else 0
            }
            
            print(f"\n📊 Call Volume Statistics:")
            for key, value in results['call_volume_stats'].items():
                print(f"   {key}: {value}")
        
        if 'monthly_calls_tgt_wrtr' in self.data_dict:
            tgt_calls = self.data_dict['monthly_calls_tgt_wrtr'].copy()
            
            # Call type analysis
            call_types = tgt_calls.groupby('call_type__c')['num_of_calls__c'].sum()
            results['call_type_distribution'] = call_types.to_dict()
            
            # Writer vs Prescriber engagement
            results['writer_prescriber_engagement'] = {
                'total_calls_to_writers': int(tgt_calls['calls_to_writers__c'].sum()),
                'total_calls_to_prescribers': int(tgt_calls['calls_to_prescribers__c'].sum()),
                'writer_call_ratio': round(
                    tgt_calls['calls_to_writers__c'].sum() / tgt_calls['num_of_calls__c'].sum() * 100, 2
                ) if tgt_calls['num_of_calls__c'].sum() > 0 else 0
            }
        
        self.insights['call_effectiveness'] = results
        return results
    
    # ==================== VISUALIZATION ====================
    
    def plot_enrollment_trends(self) -> str:
        """Generate enrollment trend visualization."""
        
        if 'monthly_kpis' not in self.data_dict:
            return None
        
        kpi_df = self.data_dict['monthly_kpis'].copy()
        enrollment_df = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments'].copy()
        enrollment_df['month_begin_date__c'] = pd.to_datetime(enrollment_df['month_begin_date__c'])
        
        monthly_total = enrollment_df.groupby('month_begin_date__c')['kpi_value__c'].sum().reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall trend
        ax1 = axes[0, 0]
        ax1.plot(monthly_total['month_begin_date__c'], monthly_total['kpi_value__c'], 
                 color=self.colors[0], linewidth=2)
        ax1.fill_between(monthly_total['month_begin_date__c'], monthly_total['kpi_value__c'], 
                        alpha=0.3, color=self.colors[0])
        ax1.set_title('Monthly Enrollment Trend', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Enrollments')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Channel distribution
        ax2 = axes[0, 1]
        channel_totals = enrollment_df.groupby('kpi_value_for__c')['kpi_value__c'].sum().sort_values(ascending=True)
        bars = ax2.barh(channel_totals.index, channel_totals.values, color=self.colors[:len(channel_totals)])
        ax2.set_title('Enrollments by Channel', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Total Enrollments')
        for bar, val in zip(bars, channel_totals.values):
            ax2.text(val + 50, bar.get_y() + bar.get_height()/2, f'{int(val):,}', 
                    va='center', fontsize=10)
        
        # 3. Year-over-Year comparison
        ax3 = axes[1, 0]
        enrollment_df['year'] = enrollment_df['month_begin_date__c'].dt.year
        yearly = enrollment_df.groupby('year')['kpi_value__c'].sum()
        bars = ax3.bar(yearly.index.astype(str), yearly.values, color=self.colors[2])
        ax3.set_title('Yearly Enrollment Totals', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Total Enrollments')
        for bar, val in zip(bars, yearly.values):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 20, f'{int(val):,}', 
                    ha='center', fontsize=9)
        
        # 4. Distribution (histogram)
        ax4 = axes[1, 1]
        ax4.hist(enrollment_df['kpi_value__c'], bins=30, color=self.colors[3], edgecolor='white', alpha=0.7)
        ax4.axvline(enrollment_df['kpi_value__c'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {enrollment_df["kpi_value__c"].mean():.1f}')
        ax4.axvline(enrollment_df['kpi_value__c'].median(), color='orange', linestyle='--',
                   label=f'Median: {enrollment_df["kpi_value__c"].median():.1f}')
        ax4.set_title('Enrollment Distribution (Rare Disease Pattern)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Enrollment Count per Territory-Month')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save
        filepath = CHARTS_DIR / 'enrollment_trends.png'
        plt.savefig(filepath, dpi=eda_config.dpi, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(str(filepath))
        print(f"✓ Saved: {filepath}")
        return str(filepath)
    
    def plot_hcp_analysis(self) -> str:
        """Generate HCP segment analysis visualization."""
        
        if 'hcp_universe' not in self.data_dict:
            return None
        
        hcp_df = self.data_dict['hcp_universe'].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Segment distribution
        ax1 = axes[0, 0]
        segment_counts = hcp_df['hcp_segment'].value_counts()
        wedges, texts, autotexts = ax1.pie(segment_counts.values, labels=segment_counts.index,
                                           autopct='%1.1f%%', colors=self.colors[:len(segment_counts)],
                                           explode=[0.05] * len(segment_counts))
        ax1.set_title('HCP Segment Distribution', fontsize=14, fontweight='bold')
        
        # 2. Power score by segment
        ax2 = axes[0, 1]
        segments = hcp_df['hcp_segment'].unique()
        segment_data = [hcp_df[hcp_df['hcp_segment'] == s]['power_score'].dropna() for s in segments]
        bp = ax2.boxplot(segment_data, labels=segments, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title('Power Score Distribution by Segment', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Power Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Regional distribution
        ax3 = axes[1, 0]
        if 'region_name' in hcp_df.columns:
            region_counts = hcp_df['region_name'].value_counts()
            bars = ax3.bar(region_counts.index, region_counts.values, color=self.colors[4])
            ax3.set_title('HCPs by Region', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Region')
            ax3.set_ylabel('Number of HCPs')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Total calls distribution
        ax4 = axes[1, 1]
        ax4.hist(hcp_df['total_calls'], bins=50, color=self.colors[5], edgecolor='white', alpha=0.7)
        ax4.set_title('Distribution of Total Calls per HCP', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Total Calls')
        ax4.set_ylabel('Frequency')
        ax4.set_yscale('log')  # Log scale for rare disease (many zeros, few high values)
        
        plt.tight_layout()
        
        filepath = CHARTS_DIR / 'hcp_analysis.png'
        plt.savefig(filepath, dpi=eda_config.dpi, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(str(filepath))
        print(f"✓ Saved: {filepath}")
        return str(filepath)
    
    def plot_correlation_matrix(self) -> str:
        """Generate correlation matrix for key features."""
        
        if 'hcp_universe' not in self.data_dict:
            return None
        
        hcp_df = self.data_dict['hcp_universe'].copy()
        
        # Select numeric columns
        numeric_cols = ['power_score', 'target_trx', 'shared_trx', 'called_when_suggested_pct',
                       'total_calls', 'specialty_score', 'tenure_years', 'referral_score',
                       'refill_flag', 'early_stop_flag', 'switch_flag', 'line2_flag', 'line3_flag']
        
        available_cols = [col for col in numeric_cols if col in hcp_df.columns]
        corr_matrix = hcp_df[available_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5, ax=ax,
                   annot_kws={'size': 9})
        
        ax.set_title('Feature Correlation Matrix - HCP Universe', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = CHARTS_DIR / 'correlation_matrix.png'
        plt.savefig(filepath, dpi=eda_config.dpi, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(str(filepath))
        print(f"✓ Saved: {filepath}")
        return str(filepath)
    
    def plot_temporal_patterns(self) -> str:
        """Generate temporal pattern analysis."""
        
        if 'monthly_kpis' not in self.data_dict:
            return None
        
        kpi_df = self.data_dict['monthly_kpis'].copy()
        enrollment_df = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments'].copy()
        enrollment_df['month_begin_date__c'] = pd.to_datetime(enrollment_df['month_begin_date__c'])
        enrollment_df['month'] = enrollment_df['month_begin_date__c'].dt.month
        enrollment_df['year'] = enrollment_df['month_begin_date__c'].dt.year
        enrollment_df['quarter'] = enrollment_df['month_begin_date__c'].dt.quarter
        enrollment_df['day_of_week'] = enrollment_df['month_begin_date__c'].dt.dayofweek
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Monthly seasonality
        ax1 = axes[0, 0]
        monthly_avg = enrollment_df.groupby('month')['kpi_value__c'].mean()
        bars = ax1.bar(monthly_avg.index, monthly_avg.values, color=self.colors[0])
        ax1.set_title('Average Enrollments by Month (Seasonality)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Enrollments')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # 2. Quarterly trends
        ax2 = axes[0, 1]
        quarterly = enrollment_df.groupby(['year', 'quarter'])['kpi_value__c'].sum().reset_index()
        quarterly['period'] = quarterly['year'].astype(str) + '-Q' + quarterly['quarter'].astype(str)
        quarterly = quarterly.tail(16)  # Last 4 years
        ax2.plot(quarterly['period'], quarterly['kpi_value__c'], marker='o', 
                color=self.colors[1], linewidth=2, markersize=6)
        ax2.set_title('Quarterly Enrollment Trends', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_xlabel('Quarter')
        ax2.set_ylabel('Total Enrollments')
        
        # 3. Heatmap by Year-Month
        ax3 = axes[1, 0]
        pivot = enrollment_df.pivot_table(values='kpi_value__c', index='year', 
                                          columns='month', aggfunc='sum')
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', ax=ax3, 
                   cbar_kws={'label': 'Enrollments'})
        ax3.set_title('Enrollment Heatmap (Year x Month)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Year')
        
        # 4. Channel trends over time
        ax4 = axes[1, 1]
        channel_yearly = enrollment_df.groupby(['year', 'kpi_value_for__c'])['kpi_value__c'].sum().unstack()
        channel_yearly = channel_yearly.fillna(0)
        channel_yearly.plot(kind='area', stacked=True, ax=ax4, alpha=0.7, 
                           color=self.colors[:len(channel_yearly.columns)])
        ax4.set_title('Enrollment Channel Mix Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Total Enrollments')
        ax4.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        filepath = CHARTS_DIR / 'temporal_patterns.png'
        plt.savefig(filepath, dpi=eda_config.dpi, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(str(filepath))
        print(f"✓ Saved: {filepath}")
        return str(filepath)
    
    def plot_territory_performance(self) -> str:
        """Generate territory performance analysis."""
        
        if 'monthly_kpis' not in self.data_dict:
            return None
        
        kpi_df = self.data_dict['monthly_kpis'].copy()
        enrollment_df = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments'].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top territories
        ax1 = axes[0, 0]
        territory_totals = enrollment_df.groupby('territory_name__c')['kpi_value__c'].sum()
        top_territories = territory_totals.nlargest(15)
        bars = ax1.barh(top_territories.index, top_territories.values, color=self.colors[2])
        ax1.set_title('Top 15 Territories by Total Enrollments', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Total Enrollments')
        for bar, val in zip(bars, top_territories.values):
            ax1.text(val + 5, bar.get_y() + bar.get_height()/2, f'{int(val):,}', 
                    va='center', fontsize=9)
        
        # 2. Territory distribution
        ax2 = axes[0, 1]
        ax2.hist(territory_totals, bins=30, color=self.colors[3], edgecolor='white', alpha=0.7)
        ax2.axvline(territory_totals.mean(), color='red', linestyle='--', 
                   label=f'Mean: {territory_totals.mean():.1f}')
        ax2.axvline(territory_totals.median(), color='orange', linestyle='--',
                   label=f'Median: {territory_totals.median():.1f}')
        ax2.set_title('Distribution of Enrollments Across Territories', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Total Enrollments')
        ax2.set_ylabel('Number of Territories')
        ax2.legend()
        
        # 3. Bottom territories (opportunity areas)
        ax3 = axes[1, 0]
        bottom_territories = territory_totals.nsmallest(15)
        bars = ax3.barh(bottom_territories.index, bottom_territories.values, color=self.colors[4])
        ax3.set_title('Bottom 15 Territories (Growth Opportunities)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Total Enrollments')
        
        # 4. Pareto chart (80/20 rule)
        ax4 = axes[1, 1]
        sorted_territories = territory_totals.sort_values(ascending=False)
        cumulative_pct = sorted_territories.cumsum() / sorted_territories.sum() * 100
        x = range(len(sorted_territories))
        ax4.bar(x, sorted_territories.values, color=self.colors[5], alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(x, cumulative_pct.values, color='red', linewidth=2, marker='')
        ax4_twin.axhline(80, color='orange', linestyle='--', alpha=0.7)
        ax4.set_title('Pareto Analysis: Territory Enrollments', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Territory Rank')
        ax4.set_ylabel('Enrollments')
        ax4_twin.set_ylabel('Cumulative %')
        ax4.set_xticks([])
        
        plt.tight_layout()
        
        filepath = CHARTS_DIR / 'territory_performance.png'
        plt.savefig(filepath, dpi=eda_config.dpi, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(str(filepath))
        print(f"✓ Saved: {filepath}")
        return str(filepath)
    
    def plot_rare_disease_indicators(self) -> str:
        """Generate rare disease specific indicator visualizations."""
        
        if 'hcp_universe' not in self.data_dict:
            return None
        
        hcp_df = self.data_dict['hcp_universe'].copy()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Behavioral flags analysis
        flag_cols = ['refill_flag', 'early_stop_flag', 'switch_flag', 'line2_flag', 'line3_flag']
        available_flags = [col for col in flag_cols if col in hcp_df.columns]
        
        for idx, flag in enumerate(available_flags[:6]):
            ax = axes.flat[idx]
            
            # Distribution by segment
            segment_flag = hcp_df.groupby('hcp_segment')[flag].mean() * 100
            bars = ax.bar(segment_flag.index, segment_flag.values, color=self.colors[idx])
            ax.set_title(f'{flag.replace("_", " ").title()} by Segment', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Value')
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Rare Disease Behavioral Indicators', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filepath = CHARTS_DIR / 'rare_disease_indicators.png'
        plt.savefig(filepath, dpi=eda_config.dpi, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(str(filepath))
        print(f"✓ Saved: {filepath}")
        return str(filepath)
    
    # ==================== RARE DISEASE SPECIFIC EDA ====================
    
    def analyze_patient_adherence_patterns(self) -> dict:
        """
        Analyze patient adherence patterns specific to rare disease.
        Includes refill behavior, early stops, treatment line progression.
        """
        
        print("\n" + "=" * 60)
        print("PATIENT ADHERENCE PATTERN ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        if 'hcp_universe' not in self.data_dict:
            return {'error': 'HCP universe data not loaded'}
        
        hcp_df = self.data_dict['hcp_universe'].copy()
        
        # Adherence-related flags
        adherence_cols = ['refill_flag', 'early_stop_flag', 'switch_flag', 'line2_flag', 'line3_flag']
        available_cols = [c for c in adherence_cols if c in hcp_df.columns]
        
        if available_cols:
            # Overall adherence statistics
            adherence_stats = {}
            for col in available_cols:
                stats = {
                    'mean': round(hcp_df[col].mean(), 4),
                    'sum': int(hcp_df[col].sum()),
                    'pct': round(hcp_df[col].mean() * 100, 2)
                }
                adherence_stats[col] = stats
            
            results['adherence_flag_stats'] = adherence_stats
            
            print(f"\n📊 Patient Adherence Indicators:")
            for flag, stats in adherence_stats.items():
                print(f"   {flag}: {stats['pct']}% ({stats['sum']:,} HCPs)")
            
            # By segment analysis
            segment_adherence = hcp_df.groupby('hcp_segment')[available_cols].mean().round(4)
            results['adherence_by_segment'] = segment_adherence.to_dict()
            
            print(f"\n📊 Adherence by HCP Segment:")
            print(segment_adherence.to_string())
            
            # Adherence correlations
            if len(available_cols) > 1:
                adherence_corr = hcp_df[available_cols].corr().round(3)
                results['adherence_correlations'] = adherence_corr.to_dict()
                
                # Key finding: early_stop vs refill correlation
                if 'early_stop_flag' in available_cols and 'refill_flag' in available_cols:
                    es_refill_corr = adherence_corr.loc['early_stop_flag', 'refill_flag']
                    results['early_stop_refill_relationship'] = {
                        'correlation': es_refill_corr,
                        'interpretation': 'Negative correlation expected (patients who refill are less likely to stop early)'
                    }
                    print(f"\n📊 Early Stop vs Refill Correlation: {es_refill_corr}")
        
        # Treatment line progression analysis
        line_cols = [c for c in ['line2_flag', 'line3_flag'] if c in hcp_df.columns]
        if line_cols:
            line_progression = hcp_df[line_cols].sum()
            results['treatment_line_progression'] = {
                'line2_patients': int(line_progression.get('line2_flag', 0)),
                'line3_patients': int(line_progression.get('line3_flag', 0)),
                'line2_pct': round(hcp_df['line2_flag'].mean() * 100, 2) if 'line2_flag' in hcp_df else 0,
                'line3_pct': round(hcp_df['line3_flag'].mean() * 100, 2) if 'line3_flag' in hcp_df else 0
            }
            print(f"\n📊 Treatment Line Progression:")
            print(f"   Line 2 Patients: {results['treatment_line_progression']['line2_pct']}%")
            print(f"   Line 3 Patients: {results['treatment_line_progression']['line3_pct']}%")
        
        self.insights['patient_adherence'] = results
        return results
    
    def analyze_hcp_power_distribution(self) -> dict:
        """
        Analyze HCP power score distribution and its relationship
        with enrollment success.
        """
        
        print("\n" + "=" * 60)
        print("HCP POWER SCORE ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        if 'hcp_universe' not in self.data_dict:
            return {'error': 'HCP universe data not loaded'}
        
        hcp_df = self.data_dict['hcp_universe'].copy()
        
        if 'power_score' in hcp_df.columns:
            # Power score distribution
            power_stats = {
                'mean': round(hcp_df['power_score'].mean(), 4),
                'median': round(hcp_df['power_score'].median(), 4),
                'std': round(hcp_df['power_score'].std(), 4),
                'min': round(hcp_df['power_score'].min(), 4),
                'max': round(hcp_df['power_score'].max(), 4),
                'skewness': round(hcp_df['power_score'].skew(), 4),
                'positive_score_pct': round((hcp_df['power_score'] > 0).mean() * 100, 2),
                'high_score_pct': round((hcp_df['power_score'] > 0.5).mean() * 100, 2)
            }
            results['power_score_distribution'] = power_stats
            
            print(f"\n📊 Power Score Distribution:")
            for key, val in power_stats.items():
                print(f"   {key}: {val}")
            
            # Power score by segment
            power_by_segment = hcp_df.groupby('hcp_segment')['power_score'].agg([
                'mean', 'median', 'std', 'count'
            ]).round(4)
            results['power_by_segment'] = power_by_segment.to_dict()
            
            print(f"\n📊 Power Score by HCP Segment:")
            print(power_by_segment.to_string())
            
            # Power score quartiles
            hcp_df['power_quartile'] = pd.qcut(
                hcp_df['power_score'].rank(method='first'),
                q=4, labels=['Q1-Low', 'Q2', 'Q3', 'Q4-High']
            )
            quartile_dist = hcp_df['power_quartile'].value_counts().sort_index()
            results['power_quartile_distribution'] = quartile_dist.to_dict()
            
            # Statistical test: Power score difference by segment
            test_result = self.test_group_differences(hcp_df, 'power_score', 'hcp_segment')
            results['power_segment_test'] = test_result
            
            print(f"\n📊 Statistical Test - Power Score by Segment:")
            if 'kruskal_wallis' in test_result:
                kw = test_result['kruskal_wallis']
                print(f"   Kruskal-Wallis H: {kw.get('statistic', 'N/A')}")
                print(f"   p-value: {kw.get('p_value', 'N/A')}")
                print(f"   Interpretation: {kw.get('interpretation', 'N/A')}")
        
        self.insights['hcp_power_analysis'] = results
        return results
    
    def analyze_channel_performance(self) -> dict:
        """
        Analyze pharmacy channel performance for rare disease enrollment.
        """
        
        print("\n" + "=" * 60)
        print("PHARMACY CHANNEL PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        if 'monthly_kpis' not in self.data_dict:
            return {'error': 'Monthly KPIs data not loaded'}
        
        kpi_df = self.data_dict['monthly_kpis'].copy()
        
        # Filter to enrollments
        enrollments = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments'].copy()
        
        if 'kpi_value_for__c' in enrollments.columns:
            # Channel summary
            channel_summary = enrollments.groupby('kpi_value_for__c')['kpi_value__c'].agg([
                'sum', 'mean', 'std', 'count', 'max'
            ]).round(2)
            channel_summary['pct_of_total'] = (
                channel_summary['sum'] / channel_summary['sum'].sum() * 100
            ).round(2)
            
            results['channel_summary'] = channel_summary.to_dict()
            
            print(f"\n📊 Channel Enrollment Summary:")
            print(channel_summary.sort_values('sum', ascending=False).to_string())
            
            # Channel concentration (HHI)
            channel_shares = channel_summary['pct_of_total'] / 100
            hhi = (channel_shares ** 2).sum()
            results['channel_concentration_hhi'] = round(hhi, 4)
            
            print(f"\n📊 Channel Concentration (HHI): {hhi:.4f}")
            print(f"   Interpretation: {'Highly Concentrated' if hhi > 0.25 else 'Moderate' if hhi > 0.15 else 'Diversified'}")
            
            # Top channel dominance
            top_channel = channel_summary['sum'].idxmax()
            top_channel_pct = channel_summary.loc[top_channel, 'pct_of_total']
            results['dominant_channel'] = {
                'name': top_channel,
                'pct': top_channel_pct,
                'is_dominant': top_channel_pct > 50
            }
            
            print(f"   Dominant Channel: {top_channel} ({top_channel_pct}%)")
        
        self.insights['channel_performance'] = results
        return results
    
    def analyze_writer_prescriber_patterns(self) -> dict:
        """
        Analyze writer and prescriber patterns for rare disease.
        """
        
        print("\n" + "=" * 60)
        print("WRITER/PRESCRIBER PATTERN ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        if 'writers_prescribers' not in self.data_dict:
            return {'error': 'Writers/Prescribers data not loaded'}
        
        wp_df = self.data_dict['writers_prescribers'].copy()
        
        # Overall writer statistics
        writer_stats = {
            'total_new_writers': int(wp_df['number_of_new_writers__c'].sum()),
            'total_repeat_writers': int(wp_df['number_of_repeat_writers__c'].sum()),
            'total_new_prescribers': int(wp_df['number_of_new_prescribers__c'].sum()),
            'total_repeat_prescribers': int(wp_df['number_of_repeat_prescribers__c'].sum())
        }
        
        # Derived metrics
        total_writers = writer_stats['total_new_writers'] + writer_stats['total_repeat_writers']
        writer_stats['writer_retention_rate'] = round(
            writer_stats['total_repeat_writers'] / total_writers * 100, 2
        ) if total_writers > 0 else 0
        
        total_prescribers = writer_stats['total_new_prescribers'] + writer_stats['total_repeat_prescribers']
        writer_stats['prescriber_retention_rate'] = round(
            writer_stats['total_repeat_prescribers'] / total_prescribers * 100, 2
        ) if total_prescribers > 0 else 0
        
        results['overall_stats'] = writer_stats
        
        print(f"\n📊 Writer/Prescriber Statistics:")
        for key, val in writer_stats.items():
            print(f"   {key}: {val}")
        
        # By territory analysis
        territory_wp = wp_df.groupby('territory_name__c').agg({
            'number_of_new_writers__c': 'sum',
            'number_of_repeat_writers__c': 'sum',
            'number_of_new_prescribers__c': 'sum',
            'number_of_repeat_prescribers__c': 'sum'
        }).round(0)
        
        territory_wp['writer_retention'] = (
            territory_wp['number_of_repeat_writers__c'] / 
            (territory_wp['number_of_new_writers__c'] + territory_wp['number_of_repeat_writers__c'] + 0.1)
        ).round(4)
        
        results['top_territories_by_writers'] = territory_wp.nlargest(10, 'number_of_new_writers__c').to_dict()
        results['top_territories_by_retention'] = territory_wp.nlargest(10, 'writer_retention').to_dict()
        
        print(f"\n📊 Top 5 Territories by New Writers:")
        print(territory_wp.nlargest(5, 'number_of_new_writers__c')[['number_of_new_writers__c', 'writer_retention']].to_string())
        
        self.insights['writer_prescriber_patterns'] = results
        return results
    
    def generate_eda_summary_report(self) -> dict:
        """
        Generate a comprehensive EDA summary report for leadership.
        """
        
        print("\n" + "=" * 60)
        print("GENERATING EDA SUMMARY REPORT")
        print("=" * 60)
        
        summary = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'key_findings': [],
            'recommendations': [],
            'data_quality_notes': []
        }
        
        # Key findings from each analysis
        if 'enrollments' in self.insights:
            enr = self.insights['enrollments']
            summary['key_findings'].append(
                f"Total historical enrollments: {enr.get('basic_stats', {}).get('total_enrollments', 'N/A'):,}"
            )
            if enr.get('rare_disease_indicators', {}).get('is_zero_inflated'):
                summary['data_quality_notes'].append(
                    "Data shows zero-inflation typical of rare disease (many zero enrollment months)"
                )
        
        if 'hcp_segments' in self.insights:
            seg = self.insights['hcp_segments']
            summary['key_findings'].append(
                f"HCP Universe: {sum(seg.get('segment_distribution', {}).values()):,} providers"
            )
        
        if 'patient_adherence' in self.insights:
            adh = self.insights['patient_adherence']
            if 'adherence_flag_stats' in adh:
                refill_pct = adh['adherence_flag_stats'].get('refill_flag', {}).get('pct', 0)
                summary['key_findings'].append(f"Patient refill rate: {refill_pct}%")
        
        if 'hcp_power_analysis' in self.insights:
            pwr = self.insights['hcp_power_analysis']
            high_power = pwr.get('power_score_distribution', {}).get('high_score_pct', 0)
            summary['key_findings'].append(f"High power score HCPs: {high_power}%")
        
        if 'channel_performance' in self.insights:
            ch = self.insights['channel_performance']
            dom = ch.get('dominant_channel', {})
            summary['key_findings'].append(
                f"Dominant channel: {dom.get('name', 'N/A')} ({dom.get('pct', 0)}%)"
            )
        
        # Generate recommendations
        summary['recommendations'] = [
            "Focus engagement on high power score HCPs for maximum enrollment impact",
            "Monitor patient adherence indicators to identify at-risk populations early",
            "Consider territory-specific strategies based on performance analysis",
            "Leverage repeat writer relationships for sustainable growth"
        ]
        
        print(f"\n📋 EDA Summary Report Generated")
        print(f"   Key Findings: {len(summary['key_findings'])}")
        print(f"   Recommendations: {len(summary['recommendations'])}")
        
        self.insights['summary_report'] = summary
        return summary
    
    def run_complete_eda(self) -> dict:
        """Execute complete EDA pipeline."""
        
        print("\n" + "=" * 80)
        print("STEP 2: COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("Rare Disease Pharmaceutical - Enrollment Prediction")
        print("=" * 80)
        
        if not self.data_dict:
            self.load_data()
        
        # Run all analyses
        print("\n📊 Running Statistical Analyses...")
        self.analyze_enrollments()
        self.analyze_hcp_segments()
        self.analyze_call_effectiveness()
        
        # Rare Disease Specific Analyses
        print("\n📊 Running Rare Disease Specific Analyses...")
        self.analyze_patient_adherence_patterns()
        self.analyze_hcp_power_distribution()
        self.analyze_channel_performance()
        self.analyze_writer_prescriber_patterns()
        
        # Generate visualizations
        print("\n📈 Generating Visualizations...")
        self.plot_enrollment_trends()
        self.plot_hcp_analysis()
        self.plot_correlation_matrix()
        self.plot_temporal_patterns()
        self.plot_territory_performance()
        self.plot_rare_disease_indicators()
        
        # Generate summary report
        self.generate_eda_summary_report()
        
        print(f"\n✓ EDA Complete! Generated {len(self.charts_generated)} charts")
        print(f"  Charts saved to: {CHARTS_DIR}")
        
        return self.insights


def run_eda():
    """Execute the complete EDA process."""
    eda = RareDiseaseEDA()
    insights = eda.run_complete_eda()
    return eda, insights


if __name__ == "__main__":
    eda, insights = run_eda()

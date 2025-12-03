"""
Data Discovery Module
=====================

Comprehensive data understanding for rare disease pharmaceutical enrollment prediction.
This module provides:
- Data loading and validation
- Schema discovery
- Data profiling
- Relationship mapping
- Quality assessment
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import DATA_DIR, data_config, OUTPUT_DIR


class DataDiscovery:
    """
    Enterprise-grade data discovery for enrollment prediction.
    
    Attributes:
        data_dict: Dictionary containing all loaded dataframes
        profiles: Dictionary containing data profiles for each dataset
    """
    
    def __init__(self):
        self.data_dict: Dict[str, pd.DataFrame] = {}
        self.profiles: Dict[str, dict] = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files from the Reporting Tables directory."""
        
        print("=" * 60)
        print("STEP 1: DATA DISCOVERY - Loading Data Files")
        print("=" * 60)
        
        for key, filename in data_config.data_files.items():
            filepath = DATA_DIR / filename
            if filepath.exists():
                # Get date columns for this file
                date_cols = data_config.date_columns.get(key, None)
                
                try:
                    df = pd.read_csv(filepath, parse_dates=date_cols if date_cols else False)
                    self.data_dict[key] = df
                    print(f"[OK] Loaded {key}: {len(df):,} rows, {len(df.columns)} columns")
                except Exception as e:
                    print(f"[ERROR] Error loading {key}: {str(e)}")
            else:
                print(f"[ERROR] File not found: {filename}")
        
        print(f"\nTotal datasets loaded: {len(self.data_dict)}")
        return self.data_dict
    
    def profile_dataset(self, name: str, df: pd.DataFrame) -> dict:
        """Generate comprehensive profile for a dataset."""
        
        profile = {
            'name': name,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_info': {},
            'missing_summary': {},
            'duplicates': df.duplicated().sum(),
            'date_range': {}
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_pct': round(df[col].isnull().sum() / len(df) * 100, 2),
                'unique_count': df[col].nunique()
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'mean': round(df[col].mean(), 4) if df[col].notna().any() else None,
                    'std': round(df[col].std(), 4) if df[col].notna().any() else None,
                    'min': df[col].min() if df[col].notna().any() else None,
                    'max': df[col].max() if df[col].notna().any() else None,
                    'median': df[col].median() if df[col].notna().any() else None
                })
            
            # Add info for datetime columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                profile['date_range'][col] = {
                    'min': str(df[col].min()),
                    'max': str(df[col].max())
                }
            
            profile['column_info'][col] = col_info
            
            if col_info['null_pct'] > 0:
                profile['missing_summary'][col] = col_info['null_pct']
        
        return profile
    
    def discover_all(self) -> Dict[str, dict]:
        """Run comprehensive discovery on all datasets."""
        
        if not self.data_dict:
            self.load_all_data()
        
        print("\n" + "=" * 60)
        print("STEP 1: DATA DISCOVERY - Profiling Datasets")
        print("=" * 60)
        
        for name, df in self.data_dict.items():
            self.profiles[name] = self.profile_dataset(name, df)
            print(f"\n📊 {name.upper()}")
            print(f"   Rows: {self.profiles[name]['rows']:,}")
            print(f"   Columns: {self.profiles[name]['columns']}")
            print(f"   Memory: {self.profiles[name]['memory_mb']:.2f} MB")
            print(f"   Duplicates: {self.profiles[name]['duplicates']:,}")
            if self.profiles[name]['date_range']:
                for date_col, range_info in self.profiles[name]['date_range'].items():
                    print(f"   Date Range ({date_col}): {range_info['min']} to {range_info['max']}")
            
            # Show missing data summary
            if self.profiles[name]['missing_summary']:
                high_missing = {k: v for k, v in self.profiles[name]['missing_summary'].items() if v > 5}
                if high_missing:
                    print(f"   ⚠ High missing (>5%): {list(high_missing.keys())}")
        
        return self.profiles
    
    def get_domain_insights(self) -> dict:
        """Extract domain-specific insights for rare disease pharma."""
        
        insights = {
            'hcp_universe': {},
            'enrollment_patterns': {},
            'territory_coverage': {},
            'engagement_metrics': {}
        }
        
        if 'hcp_universe' in self.data_dict:
            hcp_df = self.data_dict['hcp_universe']
            insights['hcp_universe'] = {
                'total_hcps': len(hcp_df),
                'segment_distribution': hcp_df['hcp_segment'].value_counts().to_dict(),
                'regions': hcp_df['region_name'].dropna().nunique(),
                'territories': hcp_df['territory_name'].dropna().nunique(),
                'avg_power_score': round(hcp_df['power_score'].mean(), 2),
                'avg_total_calls': round(hcp_df['total_calls'].mean(), 2)
            }
        
        if 'monthly_kpis' in self.data_dict:
            kpi_df = self.data_dict['monthly_kpis']
            enrollment_df = kpi_df[kpi_df['kpi_name__c'] == 'Enrollments']
            insights['enrollment_patterns'] = {
                'total_enrollment_records': len(enrollment_df),
                'unique_territories': enrollment_df['territory_name__c'].nunique(),
                'enrollment_channels': enrollment_df['kpi_value_for__c'].unique().tolist(),
                'total_enrollments': enrollment_df['kpi_value__c'].sum()
            }
        
        if 'territory_hierarchy' in self.data_dict:
            territory_df = self.data_dict['territory_hierarchy']
            insights['territory_coverage'] = {
                'total_territories': len(territory_df),
                'parent_territories': territory_df['parent_territory_name__v'].nunique(),
                'field_forces': territory_df['territory_field_force__c'].dropna().unique().tolist()
            }
        
        return insights
    
    def generate_discovery_report(self) -> str:
        """Generate comprehensive discovery report."""
        
        if not self.profiles:
            self.discover_all()
        
        domain_insights = self.get_domain_insights()
        
        report = []
        report.append("=" * 80)
        report.append("ENROLLMENT FORM PREDICTION - DATA DISCOVERY REPORT")
        report.append("Rare Disease Pharmaceutical Company")
        report.append("=" * 80)
        
        report.append("\n## EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Datasets: {len(self.profiles)}")
        total_rows = sum(p['rows'] for p in self.profiles.values())
        total_memory = sum(p['memory_mb'] for p in self.profiles.values())
        report.append(f"Total Records: {total_rows:,}")
        report.append(f"Total Memory: {total_memory:.2f} MB")
        
        report.append("\n## DOMAIN INSIGHTS")
        report.append("-" * 40)
        
        if domain_insights['hcp_universe']:
            report.append("\n### HCP Universe")
            for key, value in domain_insights['hcp_universe'].items():
                report.append(f"  - {key}: {value}")
        
        if domain_insights['enrollment_patterns']:
            report.append("\n### Enrollment Patterns")
            for key, value in domain_insights['enrollment_patterns'].items():
                report.append(f"  - {key}: {value}")
        
        report.append("\n## DATA QUALITY ASSESSMENT")
        report.append("-" * 40)
        
        for name, profile in self.profiles.items():
            report.append(f"\n### {name}")
            report.append(f"  Rows: {profile['rows']:,}, Columns: {profile['columns']}")
            if profile['missing_summary']:
                report.append(f"  Missing Data: {list(profile['missing_summary'].keys())}")
            report.append(f"  Duplicates: {profile['duplicates']}")
        
        report.append("\n## KEY RELATIONSHIPS")
        report.append("-" * 40)
        report.append("  - territory_name links across all datasets")
        report.append("  - rolling_month_num/rolling_trimester_num for temporal joins")
        report.append("  - parent_territory for hierarchical analysis")
        report.append("  - hcp_id for individual HCP tracking")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = OUTPUT_DIR / "reports" / "data_discovery_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Discovery report saved to: {report_path}")
        return report_text


def run_discovery():
    """Execute the complete data discovery process."""
    discovery = DataDiscovery()
    discovery.load_all_data()
    discovery.discover_all()
    report = discovery.generate_discovery_report()
    print(report)
    return discovery


if __name__ == "__main__":
    discovery = run_discovery()

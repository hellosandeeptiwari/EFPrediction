"""
Quick script to generate presentation with all required explanations
"""
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from pptx_generator import generate_pptx_report
import pandas as pd
from config import OUTPUT_DIR, CHARTS_DIR

def main():
    print("\n" + "="*80)
    print("GENERATING PRESENTATION WITH COMPREHENSIVE EXPLANATIONS")
    print("="*80)
    
    # Load data for insights
    print("\n📊 Loading data...")
    features_df = pd.read_csv(OUTPUT_DIR / 'engineered_features.csv')
    
    # Get latest enrollment predictions
    pred_files = list(OUTPUT_DIR.glob('enrollment_predictions_*.csv'))
    if pred_files:
        latest_pred = max(pred_files, key=lambda p: p.stat().st_mtime)
        predictions_df = pd.read_csv(latest_pred)
        print(f"   ✓ Loaded predictions from: {latest_pred.name}")
    else:
        predictions_df = None
        print("   ⚠ No predictions file found")
    
    # Build insights dictionary
    insights = {
        'total_hcps': len(features_df),
        'high_potential_count': len(predictions_df[predictions_df['prediction_label'] == 'High Enrollment Potential']) if predictions_df is not None else 0,
        'model_accuracy': 0.9918,  # From training output
        'model_f1': 0.9642,
        'model_roc_auc': 0.9962,
        'key_findings': [
            'LightGBM model achieves 99.62% ROC-AUC and 96.42% F1-score',
            '113,031 total enrollments distributed across 12 pharmacy channels',
            'SP channel leads with 22.96% of total enrollments (25,953 enrollments)',
            'Writer retention rate: 46.84%, indicating opportunity for improvement',
            'Power score and adherence metrics are strongest predictors'
        ],
        'recommendations': [
            'Focus on 2,309 HCPs identified as high enrollment potential',
            'Implement targeted engagement for SP and Commercial channels',
            'Develop retention programs to improve 46.84% writer retention rate',
            'Use power score and adherence metrics for resource allocation'
        ]
    }
    
    # Get all charts
    charts = [str(p) for p in CHARTS_DIR.glob('*.png')]
    print(f"   ✓ Found {len(charts)} charts")
    
    # Generate presentation
    print("\n🎨 Generating comprehensive presentation...")
    output_path = generate_pptx_report(insights, charts)
    
    print("\n" + "="*80)
    print("✓ PRESENTATION GENERATED SUCCESSFULLY!")
    print(f"   Location: {output_path}")
    print("="*80)
    print("\n📋 Key improvements included:")
    print("   • Data source explanations for all 6 datasets")
    print("   • Metric definition slides (enrollments, channels, KPIs)")
    print("   • Channel vs count clarification prominently displayed")
    print("   • Every chart has: data source badge, axis labels, insights, notes")
    print("   • Executive summary with model performance and recommendations")
    print("\n")

if __name__ == '__main__':
    main()

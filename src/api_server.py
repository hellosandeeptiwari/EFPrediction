"""
Flask API Server for Enrollment Prediction
==========================================

RESTful API for model inference and visualization.
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_DIR, OUTPUT_DIR, CHARTS_DIR

app = Flask(__name__, static_folder='../ui/build', static_url_path='')
CORS(app)

# Global model reference
model_package = None


def load_model():
    """Load the trained model."""
    global model_package
    
    model_files = list(MODEL_DIR.glob('best_model_*.joblib'))
    if not model_files:
        return None
    
    model_path = sorted(model_files)[-1]
    model_package = joblib.load(model_path)
    return model_package


@app.route('/')
def serve():
    """Serve the React frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_package is not None
    })


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information."""
    if model_package is None:
        load_model()
    
    if model_package is None:
        return jsonify({'error': 'No model loaded'}), 404
    
    return jsonify({
        'model_name': model_package.get('model_name', 'Unknown'),
        'n_features': len(model_package.get('feature_names', [])),
        'features': model_package.get('feature_names', [])[:20],
        'training_results': model_package.get('training_results', {})
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Generate prediction for input data."""
    if model_package is None:
        load_model()
    
    if model_package is None:
        return jsonify({'error': 'No model loaded'}), 404
    
    try:
        data = request.json
        
        if 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        # Create feature dataframe
        feature_names = model_package['feature_names']
        features_dict = data['features']
        
        # Fill missing features with 0
        input_features = {f: features_dict.get(f, 0) for f in feature_names}
        X = pd.DataFrame([input_features])
        
        # Scale features
        scaler = model_package['scaler']
        X_scaled = scaler.transform(X)
        
        # Predict
        model = model_package['model']
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0, 1]
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'High Enrollment Potential' if prediction == 1 else 'Low Enrollment Potential',
            'probability': round(float(probability), 4),
            'risk_tier': (
                'Very High' if probability > 0.7 else
                'High' if probability > 0.5 else
                'Medium' if probability > 0.3 else 'Low'
            ),
            'confidence': round(abs(float(probability) - 0.5) * 200, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Generate predictions for batch input."""
    if model_package is None:
        load_model()
    
    if model_package is None:
        return jsonify({'error': 'No model loaded'}), 404
    
    try:
        data = request.json
        
        if 'records' not in data:
            return jsonify({'error': 'No records provided'}), 400
        
        records = data['records']
        feature_names = model_package['feature_names']
        
        # Create dataframe
        processed_records = []
        for record in records:
            input_features = {f: record.get(f, 0) for f in feature_names}
            processed_records.append(input_features)
        
        X = pd.DataFrame(processed_records)
        
        # Scale and predict
        scaler = model_package['scaler']
        X_scaled = scaler.transform(X)
        
        model = model_package['model']
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        results = []
        for i in range(len(predictions)):
            results.append({
                'index': i,
                'prediction': int(predictions[i]),
                'prediction_label': 'High Enrollment Potential' if predictions[i] == 1 else 'Low Enrollment Potential',
                'probability': round(float(probabilities[i]), 4)
            })
        
        # Summary
        summary = {
            'total_records': len(results),
            'high_potential_count': int((predictions == 1).sum()),
            'low_potential_count': int((predictions == 0).sum()),
            'avg_probability': round(float(probabilities.mean()), 4)
        }
        
        return jsonify({
            'predictions': results,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts', methods=['GET'])
def list_charts():
    """List available charts."""
    charts = []
    
    if CHARTS_DIR.exists():
        for chart_file in CHARTS_DIR.glob('*.png'):
            charts.append({
                'name': chart_file.stem,
                'filename': chart_file.name,
                'url': f'/api/charts/{chart_file.name}'
            })
    
    return jsonify({'charts': charts})


@app.route('/api/charts/<filename>', methods=['GET'])
def get_chart(filename):
    """Serve a chart image."""
    chart_path = CHARTS_DIR / filename
    
    if not chart_path.exists():
        return jsonify({'error': 'Chart not found'}), 404
    
    return send_file(chart_path, mimetype='image/png')


@app.route('/api/scores', methods=['GET'])
def get_scores():
    """Get latest scoring results with HCP details."""
    score_files = list(OUTPUT_DIR.glob('enrollment_predictions_*.csv'))
    features_file = OUTPUT_DIR / 'engineered_features.csv'
    
    if not score_files:
        return jsonify({'error': 'No scoring results found'}), 404
    
    latest_file = sorted(score_files)[-1]
    predictions_df = pd.read_csv(latest_file)
    
    # Load engineered features to get HCP details
    if features_file.exists():
        features_df = pd.read_csv(features_file)
        # Merge predictions with HCP info
        hcp_cols = ['hcp_id', 'hcp_segment', 'territory_name', 'region_name', 'power_score', 
                    'total_calls', 'specialty_score', 'trx_volume_score']
        available_cols = [c for c in hcp_cols if c in features_df.columns]
        
        if len(predictions_df) == len(features_df):
            for col in available_cols:
                predictions_df[col] = features_df[col].values
    
    # Return summary and sample
    summary = {
        'total_records': len(predictions_df),
        'high_potential_count': int((predictions_df['prediction'] == 1).sum()) if 'prediction' in predictions_df.columns else 0,
        'low_potential_count': int((predictions_df['prediction'] == 0).sum()) if 'prediction' in predictions_df.columns else 0,
        'avg_probability': float(predictions_df['probability'].mean()) if 'probability' in predictions_df.columns else 0
    }
    
    # Top records by probability
    if 'probability' in predictions_df.columns:
        top_records = predictions_df.nlargest(20, 'probability').to_dict('records')
    else:
        top_records = predictions_df.head(20).to_dict('records')
    
    return jsonify({
        'summary': summary,
        'top_records': top_records,
        'file': latest_file.name
    })


@app.route('/api/hcp-predictions', methods=['GET'])
def get_hcp_predictions():
    """Get all HCP predictions with filtering and pagination."""
    score_files = list(OUTPUT_DIR.glob('enrollment_predictions_*.csv'))
    features_file = OUTPUT_DIR / 'engineered_features.csv'
    
    if not score_files:
        return jsonify({'error': 'No scoring results found'}), 404
    
    latest_file = sorted(score_files)[-1]
    predictions_df = pd.read_csv(latest_file)
    
    # Load engineered features for HCP details
    if features_file.exists():
        features_df = pd.read_csv(features_file)
        hcp_cols = ['hcp_id', 'hcp_segment', 'territory_name', 'region_name', 'power_score', 
                    'total_calls', 'specialty_score', 'trx_volume_score', 'refill_flag',
                    'adherence_line_progression', 'patient_adherence_score']
        available_cols = [c for c in hcp_cols if c in features_df.columns]
        
        if len(predictions_df) == len(features_df):
            for col in available_cols:
                predictions_df[col] = features_df[col].values
    
    # Query parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    sort_by = request.args.get('sort_by', 'probability')
    sort_order = request.args.get('sort_order', 'desc')
    filter_segment = request.args.get('segment', None)
    filter_risk = request.args.get('risk_tier', None)
    filter_region = request.args.get('region', None)
    min_prob = request.args.get('min_probability', None)
    
    # Apply filters
    filtered_df = predictions_df.copy()
    
    if filter_segment and 'hcp_segment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['hcp_segment'].str.contains(filter_segment, case=False, na=False)]
    
    if filter_risk and 'risk_tier' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['risk_tier'] == filter_risk]
    
    if filter_region and 'region_name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['region_name'].str.contains(filter_region, case=False, na=False)]
    
    if min_prob:
        filtered_df = filtered_df[filtered_df['probability'] >= float(min_prob)]
    
    # Sort
    if sort_by in filtered_df.columns:
        ascending = sort_order != 'desc'
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    # Pagination
    total_records = len(filtered_df)
    total_pages = (total_records + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    # Summary stats
    summary = {
        'total_records': total_records,
        'total_pages': total_pages,
        'current_page': page,
        'per_page': per_page,
        'high_potential_count': int((filtered_df['prediction'] == 1).sum()),
        'low_potential_count': int((filtered_df['prediction'] == 0).sum()),
        'avg_probability': round(float(filtered_df['probability'].mean()), 4),
        'risk_tier_breakdown': filtered_df['risk_tier'].value_counts().to_dict() if 'risk_tier' in filtered_df.columns else {}
    }
    
    # Get unique values for filters
    filters = {
        'segments': predictions_df['hcp_segment'].dropna().unique().tolist() if 'hcp_segment' in predictions_df.columns else [],
        'regions': predictions_df['region_name'].dropna().unique().tolist() if 'region_name' in predictions_df.columns else [],
        'risk_tiers': predictions_df['risk_tier'].dropna().unique().tolist() if 'risk_tier' in predictions_df.columns else []
    }
    
    # Convert to records
    records = page_df.fillna('').to_dict('records')
    
    return jsonify({
        'status': 'success',
        'summary': summary,
        'filters': filters,
        'records': records
    })


@app.route('/api/features/importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from trained model."""
    if model_package is None:
        load_model()
    
    if model_package is None:
        return jsonify({'error': 'No model loaded'}), 404
    
    model = model_package['model']
    feature_names = model_package['feature_names']
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return jsonify({'error': 'Model does not have feature importances'}), 400
    
    importance_list = [
        {'feature': name, 'importance': round(float(imp), 6)}
        for name, imp in zip(feature_names, importances)
    ]
    
    importance_list.sort(key=lambda x: x['importance'], reverse=True)
    
    return jsonify({
        'feature_importances': importance_list[:20]
    })


def run_server(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask server."""
    load_model()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()

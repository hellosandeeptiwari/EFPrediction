"""
EF Prediction Application - Combined Flask API + Static UI
Azure App Service Deployment
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model_LightGBM.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, 'outputs', 'engineered_features.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'outputs')
SELECTED_FEATURES_PATH = os.path.join(BASE_DIR, 'outputs', 'selected_features.csv')

# Lazy-loaded resources
_model = None
_selected_features = None
_predictions_df = None
_features_df = None
_merged_df = None
_resources_loaded = False

def get_features_df():
    """Lazy load features dataframe"""
    global _features_df
    
    if _features_df is not None:
        return _features_df
    
    try:
        import pandas as pd
        
        if os.path.exists(FEATURES_PATH):
            _features_df = pd.read_csv(FEATURES_PATH)
            print(f"Loaded features: {len(_features_df)} records")
            return _features_df
    except Exception as e:
        print(f"Error loading features: {e}")
    
    return None

def get_predictions_df():
    """Lazy load predictions dataframe merged with features"""
    global _predictions_df, _merged_df
    
    if _merged_df is not None:
        return _merged_df
    
    try:
        import pandas as pd
        
        if os.path.exists(PREDICTIONS_PATH):
            pred_files = [f for f in os.listdir(PREDICTIONS_PATH) if f.startswith('enrollment_predictions_') and f.endswith('.csv')]
            if pred_files:
                latest_pred = sorted(pred_files)[-1]
                _predictions_df = pd.read_csv(os.path.join(PREDICTIONS_PATH, latest_pred))
                print(f"Loaded predictions from {latest_pred}")
                
                # Merge with features to get HCP details
                features_df = get_features_df()
                if features_df is not None and len(_predictions_df) == len(features_df):
                    # Align by index
                    _merged_df = pd.concat([
                        features_df[['hcp_id', 'hcp_segment', 'territory_name', 'region_name', 'power_score']].reset_index(drop=True),
                        _predictions_df.reset_index(drop=True)
                    ], axis=1)
                    print(f"Merged predictions with features: {len(_merged_df)} records")
                    return _merged_df
                
                return _predictions_df
    except Exception as e:
        print(f"Error loading predictions: {e}")
    
    return None

def get_model():
    """Lazy load model"""
    global _model
    
    if _model is not None:
        return _model
    
    try:
        import joblib
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
            return _model
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return None

# ============== Static UI Routes ==============

# HTML Templates
LAYOUT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - EF Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        :root { --primary-color: #0d6efd; --secondary-color: #6c757d; }
        body { background-color: #f8f9fa; }
        .navbar { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
        .navbar-brand { font-weight: bold; color: #fff !important; }
        .nav-link { color: rgba(255,255,255,0.8) !important; }
        .nav-link:hover, .nav-link.active { color: #fff !important; }
        .card { border: none; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px 10px 0 0 !important; }
        .stat-card { transition: transform 0.2s; }
        .stat-card:hover { transform: translateY(-5px); }
        .table-container { max-height: 500px; overflow-y: auto; }
        .badge-high { background-color: #28a745; }
        .badge-medium { background-color: #ffc107; color: #000; }
        .badge-low { background-color: #dc3545; }
        .tooltip-icon { cursor: help; color: #6c757d; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark mb-4">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="bi bi-graph-up-arrow me-2"></i>EF Prediction</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a class="nav-link {{ 'active' if active == 'dashboard' else '' }}" href="/">Dashboard</a></li>
                    <li class="nav-item"><a class="nav-link {{ 'active' if active == 'predict' else '' }}" href="/predict">Predict</a></li>
                    <li class="nav-item"><a class="nav-link {{ 'active' if active == 'batch' else '' }}" href="/batch">Batch</a></li>
                    <li class="nav-item"><a class="nav-link {{ 'active' if active == 'model-info' else '' }}" href="/model-info">Model Info</a></li>
                </ul>
            </div>
        </div>
    </nav>
    <div class="container">
        {{ content | safe }}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    {{ scripts | safe }}
</body>
</html>
'''

DASHBOARD_CONTENT = '''
<h2 class="mb-4"><i class="bi bi-speedometer2 me-2"></i>HCP Enrollment Predictions Dashboard</h2>

<!-- Summary Cards -->
<div class="row mb-4" id="summaryCards">
    <div class="col-md-3">
        <div class="card stat-card text-center p-3">
            <h6 class="text-muted">Total HCPs Scored</h6>
            <h2 class="text-primary" id="totalHcps">-</h2>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card stat-card text-center p-3">
            <h6 class="text-muted">High Potential (≥70%)</h6>
            <h2 class="text-success" id="highPotential">-</h2>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card stat-card text-center p-3">
            <h6 class="text-muted">Medium Potential (40-70%)</h6>
            <h2 class="text-warning" id="mediumPotential">-</h2>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card stat-card text-center p-3">
            <h6 class="text-muted">Low Potential (<40%)</h6>
            <h2 class="text-danger" id="lowPotential">-</h2>
        </div>
    </div>
</div>

<!-- Filters -->
<div class="card mb-4">
    <div class="card-header"><i class="bi bi-funnel me-2"></i>Filters</div>
    <div class="card-body">
        <div class="row g-3">
            <div class="col-md-2">
                <label class="form-label">Min Probability</label>
                <input type="number" class="form-control" id="minProb" min="0" max="100" value="0" step="5">
            </div>
            <div class="col-md-2">
                <label class="form-label">Risk Tier</label>
                <select class="form-select" id="riskTier">
                    <option value="">All</option>
                    <option value="High">High</option>
                    <option value="Medium">Medium</option>
                    <option value="Low">Low</option>
                </select>
            </div>
            <div class="col-md-2">
                <label class="form-label">HCP Segment</label>
                <select class="form-select" id="hcpSegment">
                    <option value="">All</option>
                    <option value="High Value">High Value</option>
                    <option value="Medium Value">Medium Value</option>
                    <option value="Low Value">Low Value</option>
                </select>
            </div>
            <div class="col-md-3">
                <label class="form-label">Search HCP ID / Territory</label>
                <input type="text" class="form-control" id="searchText" placeholder="Search...">
            </div>
            <div class="col-md-1">
                <label class="form-label">Per Page</label>
                <select class="form-select" id="pageSize">
                    <option value="25">25</option>
                    <option value="50" selected>50</option>
                    <option value="100">100</option>
                </select>
            </div>
            <div class="col-md-2 d-flex align-items-end">
                <button class="btn btn-primary w-100" onclick="loadPredictions()"><i class="bi bi-search me-2"></i>Apply</button>
            </div>
        </div>
    </div>
</div>

<!-- Results Table -->
<div class="card">
    <div class="card-header d-flex justify-content-between align-items-center">
        <span><i class="bi bi-table me-2"></i>HCP Predictions</span>
        <span id="resultCount" class="badge bg-light text-dark">-</span>
    </div>
    <div class="card-body table-container">
        <table class="table table-hover" id="predictionsTable">
            <thead class="table-light sticky-top">
                <tr>
                    <th>HCP ID</th>
                    <th>HCP Segment</th>
                    <th>Territory</th>
                    <th>Region</th>
                    <th>Power Score</th>
                    <th>Probability <i class="bi bi-info-circle tooltip-icon" title="Model confidence score (0-100%)"></i></th>
                    <th>Risk Tier</th>
                </tr>
            </thead>
            <tbody id="tableBody">
                <tr><td colspan="7" class="text-center">Loading...</td></tr>
            </tbody>
        </table>
    </div>
    <div class="card-footer">
        <nav>
            <ul class="pagination justify-content-center mb-0" id="pagination"></ul>
        </nav>
    </div>
</div>
'''

DASHBOARD_SCRIPTS = '''
<script>
let currentPage = 1;
let totalPages = 1;

async function loadPredictions() {
    const minProb = document.getElementById('minProb').value / 100;
    const riskTier = document.getElementById('riskTier').value;
    const hcpSegment = document.getElementById('hcpSegment').value;
    const searchText = document.getElementById('searchText').value;
    const pageSize = document.getElementById('pageSize').value;
    
    try {
        const params = new URLSearchParams({
            page: currentPage,
            per_page: pageSize,
            min_probability: minProb
        });
        if (riskTier) params.append('risk_tier', riskTier);
        if (hcpSegment) params.append('hcp_segment', hcpSegment);
        if (searchText) params.append('search', searchText);
        
        const response = await fetch(`/api/hcp-predictions?${params}`);
        const data = await response.json();
        
        if (data.success) {
            updateSummary(data.summary);
            renderTable(data.predictions);
            totalPages = data.pagination.total_pages;
            renderPagination();
            document.getElementById('resultCount').textContent = `${data.pagination.total} total`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('tableBody').innerHTML = '<tr><td colspan="7" class="text-center text-danger">Error loading data</td></tr>';
    }
}

function updateSummary(summary) {
    document.getElementById('totalHcps').textContent = summary.total.toLocaleString();
    document.getElementById('highPotential').textContent = summary.high_potential.toLocaleString();
    document.getElementById('mediumPotential').textContent = summary.medium_potential.toLocaleString();
    document.getElementById('lowPotential').textContent = summary.low_potential.toLocaleString();
}

function renderTable(predictions) {
    const tbody = document.getElementById('tableBody');
    if (!predictions.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center">No results found</td></tr>';
        return;
    }
    
    tbody.innerHTML = predictions.map(p => {
        const prob = p.probability || p.enrollment_probability || 0;
        const probPct = (prob * 100).toFixed(1);
        const tierClass = p.risk_tier === 'Very High' || p.risk_tier === 'High' ? 'badge-high' : p.risk_tier === 'Medium' ? 'badge-medium' : 'badge-low';
        const powerScore = p.power_score !== undefined ? p.power_score.toFixed(2) : '-';
        return `
        <tr>
            <td><strong>${p.hcp_id || p.professional_id__c || 'N/A'}</strong></td>
            <td>${p.hcp_segment || '-'}</td>
            <td>${p.territory_name || '-'}</td>
            <td>${p.region_name || '-'}</td>
            <td>${powerScore}</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${prob >= 0.7 ? 'bg-success' : prob >= 0.4 ? 'bg-warning' : 'bg-danger'}" 
                         style="width: ${probPct}%">
                        ${probPct}%
                    </div>
                </div>
            </td>
            <td><span class="badge ${tierClass}">${p.risk_tier || 'N/A'}</span></td>
        </tr>
    `}).join('');
}

function renderPagination() {
    const pagination = document.getElementById('pagination');
    let html = '';
    
    html += `<li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="goToPage(${currentPage - 1})">&laquo;</a>
    </li>`;
    
    for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
        html += `<li class="page-item ${i === currentPage ? 'active' : ''}">
            <a class="page-link" href="#" onclick="goToPage(${i})">${i}</a>
        </li>`;
    }
    
    html += `<li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="goToPage(${currentPage + 1})">&raquo;</a>
    </li>`;
    
    pagination.innerHTML = html;
}

function goToPage(page) {
    if (page >= 1 && page <= totalPages) {
        currentPage = page;
        loadPredictions();
    }
}

// Load on page ready
document.addEventListener('DOMContentLoaded', loadPredictions);
</script>
'''

PREDICT_CONTENT = '''
<h2 class="mb-4"><i class="bi bi-lightning me-2"></i>Single HCP Prediction</h2>
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">Enter HCP ID</div>
            <div class="card-body">
                <div class="mb-3">
                    <label class="form-label">Professional ID</label>
                    <input type="text" class="form-control" id="hcpId" placeholder="e.g., HCP_12345">
                </div>
                <button class="btn btn-primary" onclick="predict()"><i class="bi bi-search me-2"></i>Get Prediction</button>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card" id="resultCard" style="display: none;">
            <div class="card-header">Prediction Result</div>
            <div class="card-body" id="resultBody"></div>
        </div>
    </div>
</div>
'''

PREDICT_SCRIPTS = '''
<script>
async function predict() {
    const hcpId = document.getElementById('hcpId').value.trim();
    if (!hcpId) { alert('Please enter an HCP ID'); return; }
    
    try {
        const response = await fetch(`/api/predict/${hcpId}`);
        const data = await response.json();
        
        const resultCard = document.getElementById('resultCard');
        const resultBody = document.getElementById('resultBody');
        
        if (data.success) {
            const prob = (data.prediction.probability * 100).toFixed(1);
            const tier = data.prediction.risk_tier;
            resultBody.innerHTML = `
                <h4>${hcpId}</h4>
                <div class="mb-3">
                    <label>Enrollment Probability</label>
                    <div class="progress" style="height: 30px;">
                        <div class="progress-bar ${prob >= 70 ? 'bg-success' : prob >= 40 ? 'bg-warning' : 'bg-danger'}" 
                             style="width: ${prob}%">${prob}%</div>
                    </div>
                </div>
                <p><strong>Risk Tier:</strong> <span class="badge ${tier === 'High' ? 'bg-success' : tier === 'Medium' ? 'bg-warning' : 'bg-danger'}">${tier}</span></p>
            `;
        } else {
            resultBody.innerHTML = `<div class="alert alert-warning">${data.message || 'HCP not found'}</div>`;
        }
        resultCard.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
    }
}
</script>
'''

MODEL_INFO_CONTENT = '''
<h2 class="mb-4"><i class="bi bi-cpu me-2"></i>Model Information</h2>
<div class="row">
    <div class="col-md-6">
        <div class="card mb-4">
            <div class="card-header">Model Performance</div>
            <div class="card-body">
                <table class="table">
                    <tr><th>Algorithm</th><td>LightGBM</td></tr>
                    <tr><th>F1 Score</th><td><span class="badge bg-success">96.42%</span></td></tr>
                    <tr><th>ROC-AUC</th><td><span class="badge bg-success">99.62%</span></td></tr>
                    <tr><th>Precision</th><td>95.8%</td></tr>
                    <tr><th>Recall</th><td>97.1%</td></tr>
                </table>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card mb-4">
            <div class="card-header">Dataset Statistics</div>
            <div class="card-body">
                <table class="table">
                    <tr><th>Total HCPs</th><td>20,087</td></tr>
                    <tr><th>Total Enrollments</th><td>113,031</td></tr>
                    <tr><th>Enrollment Channels</th><td>12</td></tr>
                    <tr><th>Features Used</th><td>22 (from 119)</td></tr>
                    <tr><th>Data Sources</th><td>16 tables</td></tr>
                </table>
            </div>
        </div>
    </div>
</div>
<div class="card">
    <div class="card-header">Top Predictive Features</div>
    <div class="card-body">
        <ol>
            <li><strong>enrollment_rate</strong> - Historical enrollment success rate</li>
            <li><strong>total_calls</strong> - Total sales representative calls</li>
            <li><strong>meeting_frequency</strong> - Frequency of scheduled meetings</li>
            <li><strong>email_response_rate</strong> - Email engagement metrics</li>
            <li><strong>territory_performance</strong> - Territory-level KPIs</li>
        </ol>
    </div>
</div>
'''

BATCH_CONTENT = '''
<h2 class="mb-4"><i class="bi bi-collection me-2"></i>Batch Predictions</h2>
<div class="card">
    <div class="card-header">Upload CSV for Batch Scoring</div>
    <div class="card-body">
        <div class="alert alert-info">
            <i class="bi bi-info-circle me-2"></i>
            Upload a CSV file with HCP IDs to get batch predictions. The file should have a column named 'professional_id__c' or 'hcp_id'.
        </div>
        <div class="mb-3">
            <label class="form-label">Select CSV File</label>
            <input type="file" class="form-control" id="csvFile" accept=".csv">
        </div>
        <button class="btn btn-primary" onclick="uploadBatch()"><i class="bi bi-upload me-2"></i>Upload & Score</button>
    </div>
</div>
<div class="card mt-4" id="batchResults" style="display: none;">
    <div class="card-header">Batch Results</div>
    <div class="card-body" id="batchResultsBody"></div>
</div>
'''

# ============== UI Routes ==============

@app.route('/')
def dashboard():
    return render_template_string(LAYOUT_TEMPLATE, 
                                  title='Dashboard', 
                                  active='dashboard',
                                  content=DASHBOARD_CONTENT,
                                  scripts=DASHBOARD_SCRIPTS)

@app.route('/predict')
def predict_page():
    return render_template_string(LAYOUT_TEMPLATE,
                                  title='Predict',
                                  active='predict', 
                                  content=PREDICT_CONTENT,
                                  scripts=PREDICT_SCRIPTS)

@app.route('/model-info')
def model_info():
    return render_template_string(LAYOUT_TEMPLATE,
                                  title='Model Info',
                                  active='model-info',
                                  content=MODEL_INFO_CONTENT,
                                  scripts='')

@app.route('/batch')
def batch_page():
    return render_template_string(LAYOUT_TEMPLATE,
                                  title='Batch',
                                  active='batch',
                                  content=BATCH_CONTENT,
                                  scripts='')

# ============== API Routes ==============

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_available': os.path.exists(MODEL_PATH),
        'predictions_available': os.path.exists(PREDICTIONS_PATH),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/hcp-predictions')
def get_hcp_predictions():
    """Get HCP predictions with filtering and pagination"""
    import pandas as pd
    
    predictions_df = get_predictions_df()
    
    if predictions_df is None:
        return jsonify({'success': False, 'message': 'No predictions available'})
    
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        min_probability = float(request.args.get('min_probability', 0))
        risk_tier = request.args.get('risk_tier', '')
        hcp_segment = request.args.get('hcp_segment', '')
        search_text = request.args.get('search', '')
        
        # Work with predictions
        df = predictions_df.copy()
        
        # Determine probability column name
        prob_col = 'probability' if 'probability' in df.columns else 'enrollment_probability'
        
        # Apply filters
        if prob_col in df.columns:
            df = df[df[prob_col] >= min_probability]
        
        if risk_tier and 'risk_tier' in df.columns:
            df = df[df['risk_tier'] == risk_tier]
        
        if hcp_segment and 'hcp_segment' in df.columns:
            df = df[df['hcp_segment'] == hcp_segment]
        
        if search_text:
            search_text = search_text.lower()
            mask = pd.Series([False] * len(df))
            if 'hcp_id' in df.columns:
                mask = mask | df['hcp_id'].astype(str).str.lower().str.contains(search_text, na=False)
            if 'territory_name' in df.columns:
                mask = mask | df['territory_name'].astype(str).str.lower().str.contains(search_text, na=False)
            if 'region_name' in df.columns:
                mask = mask | df['region_name'].astype(str).str.lower().str.contains(search_text, na=False)
            df = df[mask]
        
        # Sort by probability descending
        if prob_col in df.columns:
            df = df.sort_values(prob_col, ascending=False)
        
        # Calculate summary
        total = len(df)
        if prob_col in df.columns:
            high_potential = len(df[df[prob_col] >= 0.7])
            medium_potential = len(df[(df[prob_col] >= 0.4) & (df[prob_col] < 0.7)])
            low_potential = len(df[df[prob_col] < 0.4])
        else:
            high_potential = medium_potential = low_potential = 0
        
        # Paginate
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated = df.iloc[start_idx:end_idx]
        
        # Convert to records
        predictions = paginated.to_dict('records')
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'summary': {
                'total': total,
                'high_potential': high_potential,
                'medium_potential': medium_potential,
                'low_potential': low_potential
            },
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/predict/<hcp_id>')
def predict_single(hcp_id):
    """Get prediction for a single HCP"""
    import pandas as pd
    
    predictions_df = get_predictions_df()
    
    if predictions_df is None:
        return jsonify({'success': False, 'message': 'No predictions available'})
    
    try:
        # Find HCP in predictions
        id_cols = ['professional_id__c', 'hcp_id', 'id']
        result = None
        
        for col in id_cols:
            if col in predictions_df.columns:
                matches = predictions_df[predictions_df[col].astype(str) == str(hcp_id)]
                if len(matches) > 0:
                    result = matches.iloc[0]
                    break
        
        if result is None:
            return jsonify({'success': False, 'message': f'HCP {hcp_id} not found'})
        
        prob = float(result.get('enrollment_probability', 0))
        risk_tier = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.4 else 'Low'
        
        return jsonify({
            'success': True,
            'prediction': {
                'hcp_id': hcp_id,
                'probability': prob,
                'risk_tier': risk_tier
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/model-info')
def get_model_info():
    """Get model information"""
    return jsonify({
        'success': True,
        'model': {
            'algorithm': 'LightGBM',
            'f1_score': 0.9642,
            'roc_auc': 0.9962,
            'precision': 0.958,
            'recall': 0.971,
            'features_count': 22,
            'total_features_engineered': 119
        },
        'dataset': {
            'total_hcps': 20087,
            'total_enrollments': 113031,
            'channels': 12,
            'data_sources': 16
        }
    })

# ============== Main ==============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)

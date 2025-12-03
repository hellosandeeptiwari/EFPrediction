/**
 * Model Info Page JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
});

async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        if (data.status === 'success' && data.model_info) {
            displayModelInfo(data);
        } else {
            showDefaultModelInfo();
        }
    } catch (error) {
        console.log('Model info not available:', error);
        showDefaultModelInfo();
    }
}

function displayModelInfo(data) {
    document.getElementById('model-loading').classList.add('d-none');
    document.getElementById('model-content').classList.remove('d-none');
    
    const info = data.model_info;
    const metrics = info.metrics || {};
    
    // Model overview
    document.getElementById('model-type').textContent = info.model_type || 'XGBoost Classifier';
    document.getElementById('model-version').textContent = info.version || '1.0.0';
    document.getElementById('training-date').textContent = formatDate(info.training_date);
    document.getElementById('training-duration').textContent = info.training_duration || '--';
    document.getElementById('best-score').textContent = (metrics.auc_roc * 100).toFixed(1) + '%';
    
    // Performance metrics
    document.getElementById('metric-accuracy').textContent = formatPercent(metrics.accuracy);
    document.getElementById('metric-precision').textContent = formatPercent(metrics.precision);
    document.getElementById('metric-recall').textContent = formatPercent(metrics.recall);
    document.getElementById('metric-f1').textContent = formatPercent(metrics.f1_score);
    document.getElementById('metric-auc').textContent = formatPercent(metrics.auc_roc);
    document.getElementById('metric-log-loss').textContent = metrics.log_loss?.toFixed(4) || '--';
    
    // Confusion matrix
    const cm = info.confusion_matrix || [[0, 0], [0, 0]];
    document.getElementById('cm-tn').textContent = formatNumber(cm[0][0]);
    document.getElementById('cm-fp').textContent = formatNumber(cm[0][1]);
    document.getElementById('cm-fn').textContent = formatNumber(cm[1][0]);
    document.getElementById('cm-tp').textContent = formatNumber(cm[1][1]);
    
    // Feature importance chart
    createFeatureImportanceChart(data.feature_importance);
    
    // Hyperparameters
    populateHyperparameters(info.hyperparameters);
    
    // Selected features
    populateSelectedFeatures(info.selected_features);
    
    // Model comparison
    populateModelComparison(info.model_comparison);
}

function showDefaultModelInfo() {
    document.getElementById('model-loading').classList.add('d-none');
    document.getElementById('model-content').classList.remove('d-none');
    
    // Default values
    document.getElementById('model-type').textContent = 'XGBoost Classifier';
    document.getElementById('model-version').textContent = '1.0.0';
    document.getElementById('training-date').textContent = '--';
    document.getElementById('training-duration').textContent = '--';
    document.getElementById('best-score').textContent = '--';
    
    // Default metrics
    document.getElementById('metric-accuracy').textContent = '--';
    document.getElementById('metric-precision').textContent = '--';
    document.getElementById('metric-recall').textContent = '--';
    document.getElementById('metric-f1').textContent = '--';
    document.getElementById('metric-auc').textContent = '--';
    document.getElementById('metric-log-loss').textContent = '--';
    
    // Default confusion matrix
    document.getElementById('cm-tn').textContent = '--';
    document.getElementById('cm-fp').textContent = '--';
    document.getElementById('cm-fn').textContent = '--';
    document.getElementById('cm-tp').textContent = '--';
    
    // Sample feature importance
    createFeatureImportanceChart();
    
    // Sample hyperparameters
    populateHyperparameters();
    
    // Sample features
    populateSelectedFeatures();
    
    // Sample model comparison
    populateModelComparison();
}

function createFeatureImportanceChart(importance) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;
    
    const defaultImportance = {
        'power_score': 0.18,
        'prev_enrollments': 0.15,
        'call_count': 0.12,
        'meeting_count': 0.10,
        'segment_HIGH': 0.09,
        'email_count': 0.08,
        'days_since_contact': 0.07,
        'territory_rank': 0.06,
        'specialty_index': 0.05,
        'writer_status_REPEAT': 0.04
    };
    
    const data = importance || defaultImportance;
    const sorted = Object.entries(data)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);
    
    createChart(ctx, 'bar', {
        labels: sorted.map(([name]) => name),
        datasets: [{
            label: 'Importance',
            data: sorted.map(([, value]) => value),
            backgroundColor: chartColors.palette,
            borderWidth: 0
        }]
    }, {
        indexAxis: 'y',
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: {
                beginAtZero: true,
                title: { display: true, text: 'Importance Score' }
            }
        }
    });
}

function populateHyperparameters(params) {
    const tbody = document.getElementById('hyperparameters-body');
    if (!tbody) return;
    
    const defaultParams = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'scale_pos_weight': 7.13
    };
    
    const hyperparams = params || defaultParams;
    
    tbody.innerHTML = '';
    Object.entries(hyperparams).forEach(([key, value]) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><code>${key}</code></td>
            <td>${typeof value === 'number' ? value.toFixed(4).replace(/\.?0+$/, '') : value}</td>
        `;
        tbody.appendChild(tr);
    });
}

function populateSelectedFeatures(features) {
    const container = document.getElementById('selected-features');
    if (!container) return;
    
    const defaultFeatures = [
        'power_score', 'call_count', 'meeting_count', 'email_count',
        'prev_enrollments', 'days_since_contact', 'territory_rank',
        'specialty_index', 'segment_HIGH', 'segment_MEDIUM', 'segment_LOW',
        'writer_status_NEW', 'writer_status_REPEAT', 'region_A1', 'region_A2',
        'region_A3', 'region_A4', 'region_A5', 'region_A6', 'region_A7',
        'channel_HUB', 'channel_CVS', 'channel_WALGREENS', 'channel_ACCREDO'
    ];
    
    const featureList = features || defaultFeatures;
    
    container.innerHTML = '';
    featureList.forEach(f => {
        const span = document.createElement('span');
        span.className = 'feature-badge';
        span.textContent = f;
        container.appendChild(span);
    });
}

function populateModelComparison(comparison) {
    const tbody = document.getElementById('model-comparison-body');
    if (!tbody) return;
    
    const defaultComparison = [
        { name: 'XGBoost', accuracy: 0.87, precision: 0.72, recall: 0.68, f1: 0.70, auc: 0.89, best: true },
        { name: 'LightGBM', accuracy: 0.86, precision: 0.70, recall: 0.66, f1: 0.68, auc: 0.88, best: false },
        { name: 'Random Forest', accuracy: 0.85, precision: 0.68, recall: 0.62, f1: 0.65, auc: 0.86, best: false },
        { name: 'Logistic Regression', accuracy: 0.82, precision: 0.58, recall: 0.55, f1: 0.56, auc: 0.79, best: false }
    ];
    
    const models = comparison || defaultComparison;
    
    tbody.innerHTML = '';
    models.forEach(m => {
        const tr = document.createElement('tr');
        tr.className = m.best ? 'best-model' : '';
        tr.innerHTML = `
            <td>
                <strong>${m.name}</strong>
                ${m.best ? '<i class="bi bi-trophy text-warning ms-2"></i>' : ''}
            </td>
            <td>${formatPercent(m.accuracy)}</td>
            <td>${formatPercent(m.precision)}</td>
            <td>${formatPercent(m.recall)}</td>
            <td>${formatPercent(m.f1)}</td>
            <td>${formatPercent(m.auc)}</td>
            <td>
                ${m.best ? 
                    '<span class="badge bg-success">Selected</span>' : 
                    '<span class="badge bg-secondary">Evaluated</span>'
                }
            </td>
        `;
        tbody.appendChild(tr);
    });
}

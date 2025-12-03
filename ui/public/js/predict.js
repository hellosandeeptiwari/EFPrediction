/**
 * Prediction Page JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    setupPredictionForm();
});

function setupPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        const features = {};
        
        formData.forEach((value, key) => {
            if (value) {
                // Try to parse as number, otherwise keep as string
                const numValue = parseFloat(value);
                features[key] = isNaN(numValue) ? value : numValue;
            }
        });
        
        const submitBtn = form.querySelector('button[type="submit"]');
        setLoading(submitBtn, true);
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features })
            });
            
            const data = await response.json();
            displayResult(data);
        } catch (error) {
            showToast('Prediction failed: ' + error.message, 'danger');
            displayError(error.message);
        } finally {
            setLoading(submitBtn, false);
        }
    });
}

function displayResult(data) {
    const placeholder = document.getElementById('result-placeholder');
    const content = document.getElementById('result-content');
    
    placeholder.classList.add('d-none');
    content.classList.remove('d-none');
    
    if (data.status === 'success') {
        const prediction = data.prediction;
        const isEnroll = prediction.predicted_class === 1;
        const confidence = prediction.confidence || 0;
        const probabilities = prediction.probabilities || [1 - confidence, confidence];
        
        // Update icon and label
        const icon = document.getElementById('result-icon');
        const label = document.getElementById('result-label');
        
        if (isEnroll) {
            icon.innerHTML = '<i class="bi bi-check-circle-fill text-success"></i>';
            label.textContent = 'Likely to Enroll';
            label.className = 'text-success';
        } else {
            icon.innerHTML = '<i class="bi bi-x-circle-fill text-danger"></i>';
            label.textContent = 'Unlikely to Enroll';
            label.className = 'text-danger';
        }
        
        // Update confidence bar
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceValue = document.getElementById('confidence-value');
        const confidencePercent = (confidence * 100).toFixed(1);
        
        confidenceBar.style.width = confidencePercent + '%';
        confidenceBar.className = `progress-bar ${isEnroll ? 'bg-success' : 'bg-danger'}`;
        confidenceValue.textContent = confidencePercent + '%';
        
        // Update probabilities
        document.getElementById('prob-enroll').textContent = formatPercent(probabilities[1] || confidence);
        document.getElementById('prob-no-enroll').textContent = formatPercent(probabilities[0] || (1 - confidence));
        document.getElementById('pred-time').textContent = formatDate(new Date());
        
    } else {
        displayError(data.message || 'Prediction failed');
    }
}

function displayError(message) {
    const placeholder = document.getElementById('result-placeholder');
    const content = document.getElementById('result-content');
    
    placeholder.classList.add('d-none');
    content.classList.remove('d-none');
    
    const icon = document.getElementById('result-icon');
    const label = document.getElementById('result-label');
    
    icon.innerHTML = '<i class="bi bi-exclamation-triangle-fill text-warning"></i>';
    label.textContent = 'Error';
    label.className = 'text-warning';
    
    document.getElementById('confidence-bar').style.width = '0%';
    document.getElementById('confidence-value').textContent = '--';
    document.getElementById('prob-enroll').textContent = '--';
    document.getElementById('prob-no-enroll').textContent = '--';
    document.getElementById('pred-time').textContent = message;
}

/**
 * Batch Prediction Page JavaScript
 */

let uploadedData = null;
let results = null;

document.addEventListener('DOMContentLoaded', () => {
    setupFileUpload();
    setupDownloadSample();
});

function setupFileUpload() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const predictBtn = document.getElementById('predict-btn');
    const removeBtn = document.getElementById('remove-file');
    
    // Browse button click
    browseBtn.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('click', (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });
    
    // Remove file
    removeBtn.addEventListener('click', () => {
        uploadedData = null;
        fileInput.value = '';
        document.getElementById('file-info').classList.add('d-none');
        predictBtn.disabled = true;
        showPlaceholder();
    });
    
    // Run prediction
    predictBtn.addEventListener('click', runBatchPrediction);
    
    // Download results
    document.getElementById('download-btn').addEventListener('click', downloadResults);
}

function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file', 'warning');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            uploadedData = parseCSV(e.target.result);
            displayFileInfo(file);
            document.getElementById('predict-btn').disabled = false;
        } catch (error) {
            showToast('Error parsing CSV: ' + error.message, 'danger');
        }
    };
    reader.readAsText(file);
}

function displayFileInfo(file) {
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = `Size: ${(file.size / 1024).toFixed(2)} KB | Rows: ${uploadedData.data.length}`;
    document.getElementById('file-info').classList.remove('d-none');
}

async function runBatchPrediction() {
    if (!uploadedData || !uploadedData.data.length) {
        showToast('No data to process', 'warning');
        return;
    }
    
    showLoading();
    
    try {
        // Prepare batch data
        const batchData = uploadedData.data.map(row => {
            const features = {};
            Object.entries(row).forEach(([key, value]) => {
                if (value) {
                    const numValue = parseFloat(value);
                    features[key] = isNaN(numValue) ? value : numValue;
                }
            });
            return features;
        });
        
        const response = await fetch('/api/batch-predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: batchData })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            results = data.predictions;
            displayResults(results);
        } else {
            showToast('Batch prediction failed: ' + (data.message || 'Unknown error'), 'danger');
            showPlaceholder();
        }
    } catch (error) {
        showToast('Batch prediction failed: ' + error.message, 'danger');
        showPlaceholder();
    }
}

function showLoading() {
    document.getElementById('results-placeholder').classList.add('d-none');
    document.getElementById('results-content').classList.add('d-none');
    document.getElementById('results-loading').classList.remove('d-none');
    
    // Animate progress bar
    const progressBar = document.getElementById('progress-bar');
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressBar.style.width = progress + '%';
    }, 200);
    
    // Store interval ID for cleanup
    progressBar.dataset.intervalId = interval;
}

function showPlaceholder() {
    document.getElementById('results-placeholder').classList.remove('d-none');
    document.getElementById('results-content').classList.add('d-none');
    document.getElementById('results-loading').classList.add('d-none');
    
    // Clear progress interval
    const progressBar = document.getElementById('progress-bar');
    if (progressBar.dataset.intervalId) {
        clearInterval(parseInt(progressBar.dataset.intervalId));
    }
}

function displayResults(predictions) {
    // Complete progress bar
    const progressBar = document.getElementById('progress-bar');
    if (progressBar.dataset.intervalId) {
        clearInterval(parseInt(progressBar.dataset.intervalId));
    }
    progressBar.style.width = '100%';
    
    setTimeout(() => {
        document.getElementById('results-loading').classList.add('d-none');
        document.getElementById('results-content').classList.remove('d-none');
        document.getElementById('download-btn').classList.remove('d-none');
        
        // Calculate summary stats
        const total = predictions.length;
        const enrollCount = predictions.filter(p => p.predicted_class === 1).length;
        const noEnrollCount = total - enrollCount;
        
        document.getElementById('total-records').textContent = formatNumber(total);
        document.getElementById('enroll-count').textContent = formatNumber(enrollCount);
        document.getElementById('no-enroll-count').textContent = formatNumber(noEnrollCount);
        
        // Populate table
        const tbody = document.getElementById('results-body');
        tbody.innerHTML = '';
        
        predictions.forEach((pred, index) => {
            const originalRow = uploadedData.data[index] || {};
            const isEnroll = pred.predicted_class === 1;
            
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td>${originalRow.territory_id || '--'}</td>
                <td>${originalRow.segment || '--'}</td>
                <td>
                    <span class="badge ${isEnroll ? 'bg-success' : 'bg-secondary'}">
                        ${isEnroll ? 'Enroll' : 'No Enroll'}
                    </span>
                </td>
                <td>${formatPercent(pred.confidence)}</td>
            `;
            tbody.appendChild(tr);
        });
    }, 500);
}

function downloadResults() {
    if (!results || !uploadedData) {
        showToast('No results to download', 'warning');
        return;
    }
    
    // Merge original data with predictions
    const outputData = uploadedData.data.map((row, index) => {
        const pred = results[index] || {};
        return {
            ...row,
            predicted_class: pred.predicted_class === 1 ? 'Enroll' : 'No Enroll',
            confidence: pred.confidence ? (pred.confidence * 100).toFixed(2) + '%' : '--',
            prob_enroll: pred.probabilities ? (pred.probabilities[1] * 100).toFixed(2) + '%' : '--',
            prob_no_enroll: pred.probabilities ? (pred.probabilities[0] * 100).toFixed(2) + '%' : '--'
        };
    });
    
    const headers = [...uploadedData.headers, 'predicted_class', 'confidence', 'prob_enroll', 'prob_no_enroll'];
    const csv = dataToCSV(outputData, headers);
    
    const timestamp = new Date().toISOString().slice(0, 10);
    downloadFile(csv, `enrollment_predictions_${timestamp}.csv`);
    
    showToast('Results downloaded successfully', 'success');
}

function setupDownloadSample() {
    document.getElementById('download-sample').addEventListener('click', () => {
        const sampleData = `territory_id,segment,power_score,call_count,meeting_count,email_count
A1-NORTHEAST-001,HIGH,0.85,25,5,12
A2-SOUTHEAST-002,MEDIUM,0.65,15,3,8
A3-MIDWEST-003,LOW,0.35,8,1,4
A4-SOUTHWEST-004,HIGH,0.92,30,7,15
A5-WEST-005,MEDIUM,0.55,12,2,6`;
        
        downloadFile(sampleData, 'sample_enrollment_data.csv');
        showToast('Sample file downloaded', 'success');
    });
}

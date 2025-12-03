/**
 * Common JavaScript utilities for Enrollment Prediction Dashboard
 */

// API Status Check
async function checkApiStatus() {
    const statusBadge = document.getElementById('api-status');
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusBadge.innerHTML = '<i class="bi bi-circle-fill me-1"></i>API Online';
            statusBadge.className = 'badge bg-success';
        } else {
            statusBadge.innerHTML = '<i class="bi bi-circle-fill me-1"></i>API Degraded';
            statusBadge.className = 'badge bg-warning';
        }
    } catch (error) {
        statusBadge.innerHTML = '<i class="bi bi-circle-fill me-1"></i>API Offline';
        statusBadge.className = 'badge bg-danger';
    }
}

// Format number with commas
function formatNumber(num) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return num.toLocaleString();
}

// Format percentage
function formatPercent(num, decimals = 1) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return (num * 100).toFixed(decimals) + '%';
}

// Format date
function formatDate(dateString) {
    if (!dateString) return '--';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Show toast notification
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast, { delay: 5000 });
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', () => toast.remove());
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    document.body.appendChild(container);
    return container;
}

// Loading state helper
function setLoading(element, isLoading) {
    if (isLoading) {
        element.disabled = true;
        element.dataset.originalText = element.innerHTML;
        element.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
    } else {
        element.disabled = false;
        element.innerHTML = element.dataset.originalText || element.innerHTML;
    }
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Create chart with consistent styling
function createChart(ctx, type, data, options = {}) {
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 20,
                    usePointStyle: true
                }
            }
        }
    };
    
    return new Chart(ctx, {
        type: type,
        data: data,
        options: { ...defaultOptions, ...options }
    });
}

// Color palette
const chartColors = {
    primary: 'rgba(13, 110, 253, 0.8)',
    success: 'rgba(25, 135, 84, 0.8)',
    danger: 'rgba(220, 53, 69, 0.8)',
    warning: 'rgba(255, 193, 7, 0.8)',
    info: 'rgba(13, 202, 240, 0.8)',
    secondary: 'rgba(108, 117, 125, 0.8)',
    
    primaryLight: 'rgba(13, 110, 253, 0.2)',
    successLight: 'rgba(25, 135, 84, 0.2)',
    dangerLight: 'rgba(220, 53, 69, 0.2)',
    warningLight: 'rgba(255, 193, 7, 0.2)',
    infoLight: 'rgba(13, 202, 240, 0.2)',
    
    palette: [
        'rgba(102, 126, 234, 0.8)',
        'rgba(118, 75, 162, 0.8)',
        'rgba(17, 153, 142, 0.8)',
        'rgba(56, 239, 125, 0.8)',
        'rgba(252, 70, 107, 0.8)',
        'rgba(63, 94, 251, 0.8)',
        'rgba(252, 176, 69, 0.8)'
    ]
};

// CSV parsing utility
function parseCSV(text) {
    const lines = text.split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        
        const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
        const row = {};
        
        headers.forEach((header, index) => {
            row[header] = values[index];
        });
        
        data.push(row);
    }
    
    return { headers, data };
}

// Convert data to CSV
function dataToCSV(data, headers) {
    let csv = headers.join(',') + '\n';
    
    data.forEach(row => {
        const values = headers.map(header => {
            const val = row[header] || '';
            return typeof val === 'string' && val.includes(',') ? `"${val}"` : val;
        });
        csv += values.join(',') + '\n';
    });
    
    return csv;
}

// Download file
function downloadFile(content, filename, mimeType = 'text/csv') {
    const blob = new Blob([content], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Initialize API status check on page load
document.addEventListener('DOMContentLoaded', () => {
    checkApiStatus();
    // Check API status every 30 seconds
    setInterval(checkApiStatus, 30000);
});

/**
 * Dashboard JavaScript - HCP Predictions View
 */

let currentPage = 1;
let currentSort = { column: 'probability', order: 'desc' };
let allRecords = [];
let riskTierChart = null;
let segmentChart = null;

document.addEventListener('DOMContentLoaded', () => {
    loadHCPPredictions();
    setupFilters();
    setupSorting();
    setupExport();
});

// Load HCP predictions from API
async function loadHCPPredictions(page = 1) {
    currentPage = page;
    
    const segment = document.getElementById('filter-segment')?.value || '';
    const region = document.getElementById('filter-region')?.value || '';
    const riskTier = document.getElementById('filter-risk')?.value || '';
    const minProb = document.getElementById('filter-min-prob')?.value || '';
    
    const params = new URLSearchParams({
        page: page,
        per_page: 50,
        sort_by: currentSort.column,
        sort_order: currentSort.order
    });
    
    if (segment) params.append('segment', segment);
    if (region) params.append('region', region);
    if (riskTier) params.append('risk_tier', riskTier);
    if (minProb) params.append('min_probability', minProb);
    
    try {
        const response = await fetch(`/api/hcp-predictions?${params}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            updateSummaryStats(data.summary);
            populateFilters(data.filters);
            renderTable(data.records);
            renderPagination(data.summary);
            renderCharts(data.summary);
            updateSortIndicator();
        } else {
            showError(data.error || 'Failed to load predictions');
        }
    } catch (error) {
        console.error('Error loading predictions:', error);
        showError('Failed to connect to API. Make sure the backend is running.');
    }
}

// Update summary statistics cards
function updateSummaryStats(summary) {
    document.getElementById('total-hcps').textContent = formatNumber(summary.total_records);
    document.getElementById('high-potential').textContent = formatNumber(summary.high_potential_count);
    document.getElementById('low-potential').textContent = formatNumber(summary.low_potential_count);
    document.getElementById('avg-prob').textContent = (summary.avg_probability * 100).toFixed(1) + '%';
    
    const sortInfo = currentSort.order === 'desc' ? '↓' : '↑';
    document.getElementById('showing-count').textContent = 
        `Showing ${Math.min(summary.per_page, summary.total_records)} of ${summary.total_records} | Sorted by ${currentSort.column} ${sortInfo}`;
}

// Populate filter dropdowns
function populateFilters(filters) {
    const segmentSelect = document.getElementById('filter-segment');
    const regionSelect = document.getElementById('filter-region');
    
    // Only populate if empty (first load)
    if (segmentSelect && segmentSelect.options.length <= 1) {
        filters.segments.forEach(seg => {
            const option = document.createElement('option');
            option.value = seg;
            option.textContent = seg;
            segmentSelect.appendChild(option);
        });
    }
    
    if (regionSelect && regionSelect.options.length <= 1) {
        filters.regions.forEach(region => {
            const option = document.createElement('option');
            option.value = region;
            option.textContent = region;
            regionSelect.appendChild(option);
        });
    }
}

// Render the predictions table
function renderTable(records) {
    const tbody = document.getElementById('predictions-body');
    
    if (!records || records.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center py-5 text-muted">
                    <i class="bi bi-inbox display-4"></i>
                    <p class="mt-2">No predictions found matching your criteria</p>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = records.map(record => `
        <tr>
            <td><strong>${record.hcp_id || 'N/A'}</strong></td>
            <td>
                <span class="badge ${getSegmentBadgeClass(record.hcp_segment)}" 
                      data-bs-toggle="tooltip" title="${getSegmentTooltip(record.hcp_segment)}">
                    ${record.hcp_segment || 'Unknown'}
                </span>
            </td>
            <td>${record.territory_name || 'N/A'}</td>
            <td>${record.region_name || 'N/A'}</td>
            <td>
                <span class="${getPowerScoreClass(record.power_score)}"
                      data-bs-toggle="tooltip" title="${getPowerScoreTooltip(record.power_score)}">
                    ${typeof record.power_score === 'number' ? record.power_score.toFixed(2) : 'N/A'}
                </span>
            </td>
            <td>
                <div class="d-flex align-items-center" data-bs-toggle="tooltip" 
                     title="Model confidence: ${(record.probability * 100).toFixed(1)}% chance of enrollment form submission">
                    <div class="progress flex-grow-1 me-2" style="height: 8px; width: 80px;">
                        <div class="progress-bar ${getProbabilityBarClass(record.probability)}" 
                             style="width: ${(record.probability * 100).toFixed(0)}%"></div>
                    </div>
                    <span class="fw-bold">${(record.probability * 100).toFixed(1)}%</span>
                </div>
            </td>
            <td>
                <span class="badge ${record.prediction === 1 ? 'bg-success' : 'bg-secondary'}"
                      data-bs-toggle="tooltip" title="${record.prediction === 1 ? 'Likely to submit enrollment form - prioritize for outreach' : 'Unlikely to submit enrollment form - may need different approach'}">
                    ${record.prediction === 1 ? 'High' : 'Low'}
                </span>
            </td>
            <td>
                <span class="badge ${getRiskTierBadgeClass(record.risk_tier)}"
                      data-bs-toggle="tooltip" title="${getRiskTierTooltip(record.risk_tier)}">
                    ${record.risk_tier || 'N/A'}
                </span>
            </td>
        </tr>
    `).join('');
    
    // Reinitialize tooltips for new elements
    initTooltips();
}

// Render pagination
function renderPagination(summary) {
    const pagination = document.getElementById('pagination');
    const { current_page, total_pages } = summary;
    
    if (total_pages <= 1) {
        pagination.innerHTML = '';
        return;
    }
    
    let html = '';
    
    // Previous button
    html += `
        <li class="page-item ${current_page === 1 ? 'disabled' : ''}">
            <a class="page-link" href="#" data-page="${current_page - 1}">Previous</a>
        </li>
    `;
    
    // Page numbers
    const startPage = Math.max(1, current_page - 2);
    const endPage = Math.min(total_pages, current_page + 2);
    
    if (startPage > 1) {
        html += `<li class="page-item"><a class="page-link" href="#" data-page="1">1</a></li>`;
        if (startPage > 2) {
            html += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
        }
    }
    
    for (let i = startPage; i <= endPage; i++) {
        html += `
            <li class="page-item ${i === current_page ? 'active' : ''}">
                <a class="page-link" href="#" data-page="${i}">${i}</a>
            </li>
        `;
    }
    
    if (endPage < total_pages) {
        if (endPage < total_pages - 1) {
            html += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
        }
        html += `<li class="page-item"><a class="page-link" href="#" data-page="${total_pages}">${total_pages}</a></li>`;
    }
    
    // Next button
    html += `
        <li class="page-item ${current_page === total_pages ? 'disabled' : ''}">
            <a class="page-link" href="#" data-page="${current_page + 1}">Next</a>
        </li>
    `;
    
    pagination.innerHTML = html;
    
    // Add click handlers
    pagination.querySelectorAll('a[data-page]').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = parseInt(e.target.dataset.page);
            if (page >= 1 && page <= total_pages) {
                loadHCPPredictions(page);
            }
        });
    });
}

// Render charts
function renderCharts(summary) {
    // Risk Tier Chart
    const riskCtx = document.getElementById('riskTierChart');
    if (riskCtx && summary.risk_tier_breakdown) {
        if (riskTierChart) riskTierChart.destroy();
        
        const riskData = summary.risk_tier_breakdown;
        riskTierChart = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(riskData),
                datasets: [{
                    data: Object.values(riskData),
                    backgroundColor: ['#198754', '#0d6efd', '#ffc107', '#dc3545'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
}

// Setup filter form
function setupFilters() {
    const form = document.getElementById('filter-form');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            loadHCPPredictions(1);
        });
    }
    
    const resetBtn = document.getElementById('reset-filters');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            document.getElementById('filter-segment').value = '';
            document.getElementById('filter-region').value = '';
            document.getElementById('filter-risk').value = '';
            document.getElementById('filter-min-prob').value = '';
            loadHCPPredictions(1);
        });
    }
}

// Setup column sorting
function setupSorting() {
    document.querySelectorAll('.sortable').forEach(th => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => {
            const column = th.dataset.sort;
            if (currentSort.column === column) {
                currentSort.order = currentSort.order === 'desc' ? 'asc' : 'desc';
            } else {
                currentSort.column = column;
                currentSort.order = 'desc';
            }
            loadHCPPredictions(1);
        });
    });
}

// Update sort indicator in table header
function updateSortIndicator() {
    document.querySelectorAll('.sortable').forEach(th => {
        const icon = th.querySelector('i');
        if (th.dataset.sort === currentSort.column) {
            icon.className = currentSort.order === 'desc' ? 'bi bi-chevron-down' : 'bi bi-chevron-up';
            th.classList.add('table-active');
        } else {
            icon.className = 'bi bi-chevron-expand';
            th.classList.remove('table-active');
        }
    });
}

// Setup export button
function setupExport() {
    const exportBtn = document.getElementById('export-csv');
    if (exportBtn) {
        exportBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/hcp-predictions?per_page=10000');
                const data = await response.json();
                
                if (data.records) {
                    downloadCSV(data.records, 'hcp_predictions.csv');
                }
            } catch (error) {
                console.error('Export failed:', error);
            }
        });
    }
}

// Helper functions
function getSegmentBadgeClass(segment) {
    if (!segment) return 'bg-secondary';
    const s = segment.toLowerCase();
    if (s.includes('writer') && !s.includes('lapsed') && !s.includes('potential')) return 'bg-success';
    if (s.includes('potential')) return 'bg-info';
    if (s.includes('lapsed')) return 'bg-warning text-dark';
    return 'bg-secondary';
}

function getSegmentTooltip(segment) {
    if (!segment) return 'Unknown segment';
    const s = segment.toLowerCase();
    if (s.includes('writer') && !s.includes('lapsed') && !s.includes('potential')) 
        return 'Active prescriber currently writing prescriptions for this product';
    if (s.includes('potential')) 
        return 'HCP with potential to become a prescriber based on specialty and patient volume';
    if (s.includes('lapsed')) 
        return 'Previously prescribed but has not written in recent period - re-engagement opportunity';
    return segment;
}

function getPowerScoreClass(score) {
    if (score === null || score === undefined) return '';
    if (score > 20) return 'text-success fw-bold';
    if (score > 0) return 'text-primary';
    if (score < -20) return 'text-danger';
    return 'text-muted';
}

function getPowerScoreTooltip(score) {
    if (score === null || score === undefined) return 'Power score not available';
    if (score > 30) return `High influence HCP (${score.toFixed(1)}) - Top priority for engagement`;
    if (score > 10) return `Above average influence (${score.toFixed(1)}) - Good engagement candidate`;
    if (score > 0) return `Moderate influence (${score.toFixed(1)}) - Standard engagement`;
    if (score > -20) return `Below average influence (${score.toFixed(1)}) - Lower priority`;
    return `Low influence HCP (${score.toFixed(1)}) - May need different approach`;
}

function getProbabilityBarClass(prob) {
    if (prob >= 0.7) return 'bg-success';
    if (prob >= 0.5) return 'bg-info';
    if (prob >= 0.3) return 'bg-warning';
    return 'bg-danger';
}

function getRiskTierBadgeClass(tier) {
    switch (tier) {
        case 'Very High': return 'bg-success';
        case 'High': return 'bg-info';
        case 'Medium': return 'bg-warning text-dark';
        case 'Low': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

function getRiskTierTooltip(tier) {
    switch (tier) {
        case 'Very High': return 'Probability >70% - Highest priority for immediate outreach';
        case 'High': return 'Probability 50-70% - Strong candidate for engagement';
        case 'Medium': return 'Probability 30-50% - Consider for targeted campaigns';
        case 'Low': return 'Probability <30% - May need nurturing before enrollment push';
        default: return 'Risk tier not calculated';
    }
}

function initTooltips() {
    // Dispose existing tooltips first
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
        const existing = bootstrap.Tooltip.getInstance(el);
        if (existing) existing.dispose();
    });
    // Initialize new tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(el) {
        return new bootstrap.Tooltip(el, { placement: 'top', html: true });
    });
}

function formatNumber(num) {
    if (num === null || num === undefined) return '--';
    return num.toLocaleString();
}

function showError(message) {
    const tbody = document.getElementById('predictions-body');
    tbody.innerHTML = `
        <tr>
            <td colspan="8" class="text-center py-5 text-danger">
                <i class="bi bi-exclamation-triangle display-4"></i>
                <p class="mt-2">${message}</p>
                <button class="btn btn-outline-primary" onclick="loadHCPPredictions()">
                    <i class="bi bi-arrow-clockwise me-1"></i>Retry
                </button>
            </td>
        </tr>
    `;
}

function downloadCSV(data, filename) {
    if (!data || !data.length) return;
    
    const headers = Object.keys(data[0]);
    const csv = [
        headers.join(','),
        ...data.map(row => headers.map(h => JSON.stringify(row[h] ?? '')).join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function displayQuickPrediction(data) {
    const resultDiv = document.getElementById('prediction-result');
    const alert = document.getElementById('prediction-alert');
    
    resultDiv.classList.remove('d-none');
    
    if (data.status === 'success') {
        const prediction = data.prediction;
        const isEnroll = prediction.predicted_class === 1;
        
        alert.className = `alert ${isEnroll ? 'alert-success' : 'alert-warning'}`;
        document.getElementById('pred-class').textContent = isEnroll ? 'Will Enroll' : 'Will Not Enroll';
        document.getElementById('pred-confidence').textContent = formatPercent(prediction.confidence);
    } else {
        alert.className = 'alert alert-danger';
        document.getElementById('pred-class').textContent = 'Error';
        document.getElementById('pred-confidence').textContent = data.message || 'Unknown error';
    }
}

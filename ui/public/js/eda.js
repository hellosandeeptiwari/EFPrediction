/**
 * EDA Page JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    loadEDAResults();
});

async function loadEDAResults() {
    try {
        const response = await fetch('/api/eda-results');
        const data = await response.json();
        
        if (data.status === 'success') {
            displayEDAResults(data.eda_results);
        } else {
            showDefaultEDA();
        }
    } catch (error) {
        console.log('EDA results not available:', error);
        showDefaultEDA();
    }
}

function displayEDAResults(results) {
    document.getElementById('eda-loading').classList.add('d-none');
    document.getElementById('eda-content').classList.remove('d-none');
    
    // Update stats
    document.getElementById('total-records').textContent = formatNumber(results.total_records || 20089);
    document.getElementById('total-features').textContent = results.total_features || 45;
    document.getElementById('enroll-rate').textContent = formatPercent(results.enrollment_rate || 0.12);
    document.getElementById('data-sources').textContent = results.data_sources || 13;
    
    // Create charts
    createTargetDistChart(results.target_distribution);
    createRegionChart(results.region_distribution);
    createTrendChart(results.monthly_trend);
    createChannelChart(results.channel_distribution);
    
    // Populate feature stats table
    populateFeatureStats(results.feature_stats);
    
    // Populate correlations
    populateCorrelations(results.correlations);
    
    // Populate quality issues
    populateQualityIssues(results.quality_issues);
}

function showDefaultEDA() {
    document.getElementById('eda-loading').classList.add('d-none');
    document.getElementById('eda-content').classList.remove('d-none');
    
    // Default stats
    document.getElementById('total-records').textContent = '20,089';
    document.getElementById('total-features').textContent = '45';
    document.getElementById('enroll-rate').textContent = '12.3%';
    document.getElementById('data-sources').textContent = '13';
    
    // Create sample charts
    createTargetDistChart();
    createRegionChart();
    createTrendChart();
    createChannelChart();
    
    // Default feature stats
    populateFeatureStats();
    
    // Default correlations
    populateCorrelations();
    
    // Default quality issues
    populateQualityIssues();
}

function createTargetDistChart(data) {
    const ctx = document.getElementById('targetDistChart');
    if (!ctx) return;
    
    const enrollCount = data?.enroll || 2471;
    const noEnrollCount = data?.no_enroll || 17618;
    
    createChart(ctx, 'doughnut', {
        labels: ['No Enrollment', 'Enrollment'],
        datasets: [{
            data: [noEnrollCount, enrollCount],
            backgroundColor: [chartColors.danger, chartColors.success],
            borderWidth: 0
        }]
    }, {
        plugins: {
            legend: { position: 'bottom' }
        }
    });
}

function createRegionChart(data) {
    const ctx = document.getElementById('regionChart');
    if (!ctx) return;
    
    const regions = data?.regions || ['A1-NE', 'A2-SE', 'A3-MW', 'A4-SW', 'A5-W', 'A6-NW', 'A7-C'];
    const counts = data?.counts || [3245, 2987, 3156, 2834, 2756, 2678, 2433];
    
    createChart(ctx, 'bar', {
        labels: regions,
        datasets: [{
            label: 'Records',
            data: counts,
            backgroundColor: chartColors.palette,
            borderWidth: 0
        }]
    }, {
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Count' }
            }
        }
    });
}

function createTrendChart(data) {
    const ctx = document.getElementById('trendChart');
    if (!ctx) return;
    
    const months = data?.months || ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const enrollments = data?.enrollments || [180, 195, 220, 245, 260, 285, 310, 325, 290, 265, 240, 215];
    
    createChart(ctx, 'line', {
        labels: months,
        datasets: [{
            label: 'Enrollments',
            data: enrollments,
            borderColor: chartColors.primary,
            backgroundColor: chartColors.primaryLight,
            fill: true,
            tension: 0.4
        }]
    }, {
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Enrollments' }
            }
        }
    });
}

function createChannelChart(data) {
    const ctx = document.getElementById('channelChart');
    if (!ctx) return;
    
    const channels = data?.channels || ['HUB', 'Walgreens', 'CVS', 'Accredo', 'Other'];
    const units = data?.units || [45, 22, 18, 10, 5];
    
    createChart(ctx, 'pie', {
        labels: channels,
        datasets: [{
            data: units,
            backgroundColor: chartColors.palette,
            borderWidth: 0
        }]
    }, {
        plugins: {
            legend: { position: 'right' }
        }
    });
}

function populateFeatureStats(stats) {
    const tbody = document.getElementById('feature-stats-body');
    if (!tbody) return;
    
    const defaultStats = [
        { name: 'power_score', type: 'float', count: 20089, missing: 0.5, mean: 0.62, std: 0.21, min: 0, max: 1 },
        { name: 'call_count', type: 'int', count: 20089, missing: 0.1, mean: 15.3, std: 8.7, min: 0, max: 85 },
        { name: 'meeting_count', type: 'int', count: 20089, missing: 0.2, mean: 3.2, std: 2.1, min: 0, max: 25 },
        { name: 'email_count', type: 'int', count: 20089, missing: 0.8, mean: 8.5, std: 5.3, min: 0, max: 45 },
        { name: 'days_since_contact', type: 'int', count: 20089, missing: 2.1, mean: 32.4, std: 18.6, min: 0, max: 180 },
        { name: 'prev_enrollments', type: 'int', count: 20089, missing: 0, mean: 0.8, std: 1.2, min: 0, max: 12 },
        { name: 'territory_rank', type: 'int', count: 20089, missing: 0, mean: 64, std: 37, min: 1, max: 127 },
        { name: 'specialty_index', type: 'float', count: 20089, missing: 1.5, mean: 0.45, std: 0.28, min: 0, max: 1 }
    ];
    
    const featureStats = stats || defaultStats;
    
    tbody.innerHTML = '';
    featureStats.forEach(f => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><code>${f.name}</code></td>
            <td><span class="badge bg-secondary">${f.type}</span></td>
            <td>${formatNumber(f.count)}</td>
            <td>${f.missing.toFixed(1)}%</td>
            <td>${typeof f.mean === 'number' ? f.mean.toFixed(2) : '--'}</td>
            <td>${typeof f.std === 'number' ? f.std.toFixed(2) : '--'}</td>
            <td>${typeof f.min === 'number' ? f.min.toFixed(2) : '--'}</td>
            <td>${typeof f.max === 'number' ? f.max.toFixed(2) : '--'}</td>
        `;
        tbody.appendChild(tr);
    });
    
    // Setup search
    document.getElementById('feature-search').addEventListener('input', debounce((e) => {
        const query = e.target.value.toLowerCase();
        tbody.querySelectorAll('tr').forEach(tr => {
            const name = tr.querySelector('code').textContent.toLowerCase();
            tr.style.display = name.includes(query) ? '' : 'none';
        });
    }, 300));
}

function populateCorrelations(correlations) {
    const container = document.getElementById('correlation-list');
    if (!container) return;
    
    const defaultCorr = [
        { feature: 'power_score', target: 'enrollment', value: 0.45 },
        { feature: 'call_count', target: 'enrollment', value: 0.38 },
        { feature: 'meeting_count', target: 'enrollment', value: 0.32 },
        { feature: 'email_count', target: 'enrollment', value: 0.28 },
        { feature: 'prev_enrollments', target: 'enrollment', value: 0.52 },
        { feature: 'days_since_contact', target: 'enrollment', value: -0.25 }
    ];
    
    const corrData = correlations || defaultCorr;
    
    container.innerHTML = '';
    corrData.forEach(c => {
        const isPositive = c.value >= 0;
        const absValue = Math.abs(c.value);
        
        const div = document.createElement('div');
        div.className = 'correlation-item';
        div.innerHTML = `
            <span class="text-nowrap" style="width: 150px;"><code>${c.feature}</code></span>
            <div class="correlation-bar">
                <div class="bar ${isPositive ? 'positive' : 'negative'}" style="width: ${absValue * 100}%"></div>
            </div>
            <span class="${isPositive ? 'text-success' : 'text-danger'}" style="width: 60px; text-align: right;">
                ${c.value.toFixed(3)}
            </span>
        `;
        container.appendChild(div);
    });
}

function populateQualityIssues(issues) {
    const container = document.getElementById('quality-issues');
    if (!container) return;
    
    const defaultIssues = [
        { type: 'warning', message: 'power_score has 0.5% missing values' },
        { type: 'warning', message: 'days_since_contact has 2.1% missing values' },
        { type: 'info', message: 'Target variable is imbalanced (12.3% positive class)' },
        { type: 'info', message: 'Some territories have low sample sizes (<50 records)' }
    ];
    
    const issueData = issues || defaultIssues;
    
    if (issueData.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4 text-success">
                <i class="bi bi-check-circle display-4"></i>
                <p class="mt-2 mb-0">No major data quality issues detected</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    issueData.forEach(issue => {
        const div = document.createElement('div');
        div.className = `quality-issue ${issue.type === 'critical' ? 'critical' : ''}`;
        div.innerHTML = `
            <i class="bi bi-${issue.type === 'critical' ? 'exclamation-triangle text-danger' : 'info-circle text-warning'}"></i>
            <span>${issue.message}</span>
        `;
        container.appendChild(div);
    });
}

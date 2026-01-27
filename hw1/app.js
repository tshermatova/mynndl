// app.js - Titanic EDA Explorer
// This is a reusable EDA tool that can be adapted for other datasets
// Change the schema variables below to use with different datasets

// ==================== CONFIGURATION ====================
// DATASET SCHEMA - Change these variables to adapt for another dataset
const TARGET_COLUMN = 'Survived';           // Change this to your target column name
const IDENTIFIER_COLUMN = 'PassengerId';    // Change this to your ID column (excluded from analysis)
const NUMERIC_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];  // Update with your numeric columns
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Update with your categorical columns
const ALL_FEATURES = [...NUMERIC_FEATURES, ...CATEGORICAL_FEATURES];
// =======================================================

// Global variables
let mergedData = [];
let trainData = [];
let testData = [];
let charts = {}; // Store chart instances for cleanup

// DOM Elements
const trainFileInput = document.getElementById('trainFile');
const testFileInput = document.getElementById('testFile');
const loadBtn = document.getElementById('loadBtn');
const runEDABtn = document.getElementById('runEDABtn');
const exportCSVBtn = document.getElementById('exportCSVBtn');
const exportJSONBtn = document.getElementById('exportJSONBtn');
const loadStatus = document.getElementById('loadStatus');
const overviewContent = document.getElementById('overviewContent');
const missingValuesContent = document.getElementById('missingValuesContent');
const statsContent = document.getElementById('statsContent');
const vizContent = document.getElementById('vizContent');
const exportStatus = document.getElementById('exportStatus');

// Event Listeners
loadBtn.addEventListener('click', loadAndMergeData);
runEDABtn.addEventListener('click', runFullEDA);
exportCSVBtn.addEventListener('click', exportMergedCSV);
exportJSONBtn.addEventListener('click', exportStatsJSON);

// ==================== DATA LOADING & MERGING ====================
async function loadAndMergeData() {
    const trainFile = trainFileInput.files[0];
    const testFile = testFileInput.files[0];
    
    if (!trainFile || !testFile) {
        showAlert('Please select both train.csv and test.csv files', 'error');
        return;
    }
    
    try {
        showAlert('Loading and parsing CSV files...', 'info', loadStatus);
        
        // Parse both files in parallel
        const [trainResult, testResult] = await Promise.all([
            parseCSV(trainFile),
            parseCSV(testFile)
        ]);
        
        trainData = trainResult.data;
        testData = testResult.data;
        
        // Add source column
        trainData.forEach(row => row.source = 'train');
        testData.forEach(row => row.source = 'test');
        
        // Merge datasets
        mergedData = [...trainData, ...testData];
        
        // Validate data structure
        validateDataStructure();
        
        showAlert(`Successfully loaded ${trainData.length} train + ${testData.length} test = ${mergedData.length} total rows`, 'success', loadStatus);
        
        // Enable EDA button
        runEDABtn.disabled = false;
        exportCSVBtn.disabled = false;
        
        // Show initial overview
        showOverview();
        
    } catch (error) {
        showAlert(`Error loading data: ${error.message}`, 'error', loadStatus);
        console.error('Data loading error:', error);
    }
}

function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            quotes: true, // Handle quoted commas correctly
            complete: resolve,
            error: reject
        });
    });
}

function validateDataStructure() {
    // Check for required columns in train data
    if (trainData.length > 0) {
        const firstRow = trainData[0];
        if (!(TARGET_COLUMN in firstRow)) {
            console.warn(`Target column "${TARGET_COLUMN}" not found in train data`);
        }
    }
    
    // Check feature columns exist in merged data
    const sampleRow = mergedData[0] || {};
    ALL_FEATURES.forEach(feature => {
        if (!(feature in sampleRow)) {
            console.warn(`Feature column "${feature}" not found in data`);
        }
    });
}

// ==================== OVERVIEW ====================
function showOverview() {
    if (mergedData.length === 0) return;
    
    const columns = Object.keys(mergedData[0]).filter(col => col !== 'source');
    const trainRows = trainData.length;
    const testRows = testData.length;
    
    // Create overview HTML
    overviewContent.innerHTML = `
        <div class="alert success">
            <strong>Dataset Shape:</strong> ${mergedData.length} rows × ${columns.length} columns<br>
            <strong>Train:</strong> ${trainRows} rows | <strong>Test:</strong> ${testRows} rows
        </div>
        <h3>Data Preview (first 5 rows)</h3>
        <div class="data-preview">
            ${createPreviewTable(mergedData.slice(0, 5))}
        </div>
    `;
}

function createPreviewTable(data) {
    if (data.length === 0) return '<p>No data to display</p>';
    
    const columns = Object.keys(data[0]);
    
    let html = '<table><thead><tr>';
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    data.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            html += `<td>${value === null || value === undefined ? '<em>null</em>' : value}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    return html;
}

// ==================== FULL EDA PIPELINE ====================
function runFullEDA() {
    if (mergedData.length === 0) {
        showAlert('Please load data first', 'error');
        return;
    }
    
    try {
        // Clear existing charts
        Object.values(charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        charts = {};
        
        // Run all EDA components
        analyzeMissingValues();
        calculateStatistics();
        generateVisualizations();
        
        showAlert('EDA completed successfully!', 'success');
        
        // Enable JSON export
        exportJSONBtn.disabled = false;
        
    } catch (error) {
        showAlert(`Error during EDA: ${error.message}`, 'error');
        console.error('EDA error:', error);
    }
}

// ==================== MISSING VALUES ANALYSIS ====================
function analyzeMissingValues() {
    if (mergedData.length === 0) return;
    
    const columns = Object.keys(mergedData[0]).filter(col => 
        col !== 'source' && col !== IDENTIFIER_COLUMN
    );
    
    // Calculate missing values
    const missingStats = columns.map(col => {
        const total = mergedData.length;
        const missing = mergedData.filter(row => 
            row[col] === null || 
            row[col] === undefined || 
            row[col] === '' || 
            (typeof row[col] === 'number' && isNaN(row[col]))
        ).length;
        const percentage = ((missing / total) * 100).toFixed(1);
        
        return {
            column: col,
            missing: missing,
            total: total,
            percentage: parseFloat(percentage)
        };
    });
    
    // Sort by percentage descending
    missingStats.sort((a, b) => b.percentage - a.percentage);
    
    // Create HTML table
    let tableHTML = '<div class="stats-table"><table><thead><tr><th>Column</th><th>Missing</th><th>Total</th><th>% Missing</th></tr></thead><tbody>';
    
    missingStats.forEach(stat => {
        const rowClass = stat.percentage > 20 ? 'style="background-color: #fed7d7;"' : 
                        stat.percentage > 5 ? 'style="background-color: #feebc8;"' : '';
        
        tableHTML += `
            <tr ${rowClass}>
                <td><strong>${stat.column}</strong></td>
                <td>${stat.missing}</td>
                <td>${stat.total}</td>
                <td>${stat.percentage}%</td>
            </tr>
        `;
    });
    
    tableHTML += '</tbody></table></div>';
    
    // Create chart
    missingValuesContent.innerHTML = `
        <h3>Missing Values by Column</h3>
        <div class="chart-container">
            <canvas id="missingChart"></canvas>
        </div>
        ${tableHTML}
    `;
    
    // Render chart
    const ctx = document.getElementById('missingChart').getContext('2d');
    charts.missingChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: missingStats.map(stat => stat.column),
            datasets: [{
                label: '% Missing',
                data: missingStats.map(stat => stat.percentage),
                backgroundColor: missingStats.map(stat => 
                    stat.percentage > 20 ? '#fc8181' : 
                    stat.percentage > 5 ? '#f6ad55' : '#68d391'
                ),
                borderColor: '#2d3748',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.parsed.y}% missing`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Percentage Missing (%)'
                    },
                    max: 100
                },
                x: {
                    ticks: {
                        maxRotation: 45
                    }
                }
            }
        }
    });
}

// ==================== STATISTICS CALCULATION ====================
function calculateStatistics() {
    if (mergedData.length === 0 || trainData.length === 0) return;
    
    // 1. Numeric statistics for merged data
    const numericStats = {};
    NUMERIC_FEATURES.forEach(feature => {
        const values = mergedData
            .map(row => row[feature])
            .filter(val => val !== null && val !== undefined && !isNaN(val) && typeof val === 'number');
        
        if (values.length > 0) {
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const sorted = [...values].sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            const variance = values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length;
            const std = Math.sqrt(variance);
            const min = Math.min(...values);
            const max = Math.max(...values);
            
            numericStats[feature] = {
                mean: mean.toFixed(2),
                median: median.toFixed(2),
                std: std.toFixed(2),
                min: min.toFixed(2),
                max: max.toFixed(2),
                count: values.length,
                missing: mergedData.length - values.length
            };
        }
    });
    
    // 2. Categorical value counts for merged data
    const categoricalStats = {};
    CATEGORICAL_FEATURES.forEach(feature => {
        const counts = {};
        mergedData.forEach(row => {
            const val = row[feature];
            if (val !== null && val !== undefined) {
                counts[val] = (counts[val] || 0) + 1;
            }
        });
        
        // Convert to array and sort by count
        categoricalStats[feature] = Object.entries(counts)
            .map(([value, count]) => ({
                value,
                count,
                percentage: ((count / mergedData.length) * 100).toFixed(1)
            }))
            .sort((a, b) => b.count - a.count);
    });
    
    // 3. Statistics by Survived (train data only)
    let survivalStats = null;
    if (trainData.some(row => TARGET_COLUMN in row)) {
        survivalStats = {};
        
        // For each numeric feature, calculate stats by survival
        NUMERIC_FEATURES.forEach(feature => {
            const survivedValues = trainData
                .filter(row => row[TARGET_COLUMN] === 1 && row[feature] !== null && !isNaN(row[feature]))
                .map(row => row[feature]);
            
            const notSurvivedValues = trainData
                .filter(row => row[TARGET_COLUMN] === 0 && row[feature] !== null && !isNaN(row[feature]))
                .map(row => row[feature]);
            
            if (survivedValues.length > 0 && notSurvivedValues.length > 0) {
                const survivedMean = survivedValues.reduce((a, b) => a + b, 0) / survivedValues.length;
                const notSurvivedMean = notSurvivedValues.reduce((a, b) => a + b, 0) / notSurvivedValues.length;
                
                survivalStats[feature] = {
                    survived_mean: survivedMean.toFixed(2),
                    not_survived_mean: notSurvivedMean.toFixed(2),
                    difference: (survivedMean - notSurvivedMean).toFixed(2)
                };
            }
        });
        
        // Survival rates by categorical features
        CATEGORICAL_FEATURES.forEach(feature => {
            const groups = {};
            trainData.forEach(row => {
                if (row[feature] !== null && row[feature] !== undefined) {
                    const key = row[feature];
                    if (!groups[key]) {
                        groups[key] = { total: 0, survived: 0 };
                    }
                    groups[key].total++;
                    if (row[TARGET_COLUMN] === 1) {
                        groups[key].survived++;
                    }
                }
            });
            
            // Calculate survival rates
            const rates = Object.entries(groups).map(([value, stats]) => ({
                value,
                total: stats.total,
                survived: stats.survived,
                survival_rate: ((stats.survived / stats.total) * 100).toFixed(1)
            })).sort((a, b) => b.survival_rate - a.survival_rate);
            
            survivalStats[`${feature}_survival`] = rates;
        });
    }
    
    // Store stats globally for export
    window.edaStats = {
        numeric: numericStats,
        categorical: categoricalStats,
        survival: survivalStats,
        timestamp: new Date().toISOString()
    };
    
    // Display statistics
    displayStatistics(numericStats, categoricalStats, survivalStats);
}

function displayStatistics(numericStats, categoricalStats, survivalStats) {
    let html = '<div class="stats-container">';
    
    // 1. Numeric Statistics
    html += '<h3>Numeric Features Statistics</h3>';
    if (Object.keys(numericStats).length > 0) {
        html += '<div class="stats-table"><table><thead><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Count</th></tr></thead><tbody>';
        
        Object.entries(numericStats).forEach(([feature, stats]) => {
            html += `
                <tr>
                    <td><strong>${feature}</strong></td>
                    <td>${stats.mean}</td>
                    <td>${stats.median}</td>
                    <td>${stats.std}</td>
                    <td>${stats.min}</td>
                    <td>${stats.max}</td>
                    <td>${stats.count}</td>
                </tr>
            `;
        });
        
        html += '</tbody></table></div>';
    } else {
        html += '<p>No numeric features found</p>';
    }
    
    // 2. Categorical Statistics
    html += '<h3 style="margin-top: 30px;">Categorical Features</h3>';
    Object.entries(categoricalStats).forEach(([feature, counts]) => {
        html += `<h4>${feature} Distribution</h4>`;
        html += '<div class="stats-table"><table><thead><tr><th>Value</th><th>Count</th><th>Percentage</th></tr></thead><tbody>';
        
        counts.forEach(item => {
            html += `
                <tr>
                    <td>${item.value}</td>
                    <td>${item.count}</td>
                    <td>${item.percentage}%</td>
                </tr>
            `;
        });
        
        html += '</tbody></table></div>';
    });
    
    // 3. Survival Statistics (if available)
    if (survivalStats) {
        html += '<h3 style="margin-top: 30px;">Statistics by Survival (Train Data)</h3>';
        
        // Numeric features by survival
        if (Object.keys(survivalStats).some(key => NUMERIC_FEATURES.includes(key))) {
            html += '<h4>Average Values by Survival</h4>';
            html += '<div class="stats-table"><table><thead><tr><th>Feature</th><th>Survived Mean</th><th>Not Survived Mean</th><th>Difference</th></tr></thead><tbody>';
            
            NUMERIC_FEATURES.forEach(feature => {
                if (survivalStats[feature]) {
                    const diff = parseFloat(survivalStats[feature].difference);
                    const diffClass = diff > 0 ? 'style="color: #38a169;"' : 
                                    diff < 0 ? 'style="color: #e53e3e;"' : '';
                    
                    html += `
                        <tr>
                            <td><strong>${feature}</strong></td>
                            <td>${survivalStats[feature].survived_mean}</td>
                            <td>${survivalStats[feature].not_survived_mean}</td>
                            <td ${diffClass}>${survivalStats[feature].difference}</td>
                        </tr>
                    `;
                }
            });
            
            html += '</tbody></table></div>';
        }
        
        // Survival rates by categorical features
        CATEGORICAL_FEATURES.forEach(feature => {
            const key = `${feature}_survival`;
            if (survivalStats[key]) {
                html += `<h4>Survival Rate by ${feature}</h4>`;
                html += '<div class="stats-table"><table><thead><tr><th>Value</th><th>Total</th><th>Survived</th><th>Survival Rate</th></tr></thead><tbody>';
                
                survivalStats[key].forEach(item => {
                    html += `
                        <tr>
                            <td>${item.value}</td>
                            <td>${item.total}</td>
                            <td>${item.survived}</td>
                            <td>${item.survival_rate}%</td>
                        </tr>
                    `;
                });
                
                html += '</tbody></table></div>';
            }
        });
    }
    
    html += '</div>';
    statsContent.innerHTML = html;
}

// ==================== VISUALIZATIONS ====================
function generateVisualizations() {
    if (mergedData.length === 0) return;
    
    vizContent.innerHTML = `
        <div class="container" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            <div>
                <h3>Categorical Distributions</h3>
                <div class="chart-container">
                    <canvas id="sexChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="pclassChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="embarkedChart"></canvas>
                </div>
            </div>
            <div>
                <h3>Numeric Distributions</h3>
                <div class="chart-container">
                    <canvas id="ageChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="fareChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="correlationChart"></canvas>
                </div>
            </div>
        </div>
    `;
    
    // Create all charts
    createCategoricalCharts();
    createNumericCharts();
    createCorrelationHeatmap();
}

function createCategoricalCharts() {
    // Sex distribution
    const sexCounts = countCategories('Sex');
    const sexCtx = document.getElementById('sexChart').getContext('2d');
    charts.sexChart = new Chart(sexCtx, {
        type: 'pie',
        data: {
            labels: Object.keys(sexCounts),
            datasets: [{
                data: Object.values(sexCounts),
                backgroundColor: ['#4299e1', '#ed64a6'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Sex Distribution' }
            }
        }
    });
    
    // Pclass distribution
    const pclassCounts = countCategories('Pclass');
    const pclassCtx = document.getElementById('pclassChart').getContext('2d');
    charts.pclassChart = new Chart(pclassCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(pclassCounts).map(k => `Class ${k}`),
            datasets: [{
                label: 'Count',
                data: Object.values(pclassCounts),
                backgroundColor: ['#48bb78', '#4299e1', '#9f7aea'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Passenger Class Distribution' }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
    
    // Embarked distribution
    const embarkedCounts = countCategories('Embarked');
    const embarkedCtx = document.getElementById('embarkedChart').getContext('2d');
    charts.embarkedChart = new Chart(embarkedCtx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(embarkedCounts),
            datasets: [{
                data: Object.values(embarkedCounts),
                backgroundColor: ['#ecc94b', '#4299e1', '#48bb78', '#a0aec0'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Embarkation Port' }
            }
        }
    });
}

function countCategories(column) {
    const counts = {};
    mergedData.forEach(row => {
        const val = row[column];
        if (val !== null && val !== undefined) {
            counts[val] = (counts[val] || 0) + 1;
        }
    });
    return counts;
}

function createNumericCharts() {
    // Age histogram
    const ages = mergedData
        .map(row => row.Age)
        .filter(age => age !== null && !isNaN(age) && typeof age === 'number');
    
    const ageCtx = document.getElementById('ageChart').getContext('2d');
    charts.ageChart = new Chart(ageCtx, {
        type: 'histogram',
        data: {
            datasets: [{
                label: 'Age Distribution',
                data: ages,
                backgroundColor: 'rgba(66, 153, 225, 0.6)',
                borderColor: 'rgba(66, 153, 225, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Age Distribution' }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Age' },
                    min: 0,
                    max: Math.ceil(Math.max(...ages) / 10) * 10
                },
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Frequency' }
                }
            }
        }
    });
    
    // Fare histogram (with log scale for better visualization)
    const fares = mergedData
        .map(row => row.Fare)
        .filter(fare => fare !== null && !isNaN(fare) && typeof fare === 'number' && fare > 0);
    
    const fareCtx = document.getElementById('fareChart').getContext('2d');
    charts.fareChart = new Chart(fareCtx, {
        type: 'histogram',
        data: {
            datasets: [{
                label: 'Fare Distribution',
                data: fares,
                backgroundColor: 'rgba(72, 187, 120, 0.6)',
                borderColor: 'rgba(72, 187, 120, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Fare Distribution' }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Fare' },
                    min: 0
                },
                y: {
                    type: 'logarithmic',
                    beginAtZero: true,
                    title: { display: true, text: 'Frequency (log scale)' }
                }
            }
        }
    });
}

function createCorrelationHeatmap() {
    // Calculate correlations between numeric features
    const numericData = mergedData.map(row => {
        const obj = {};
        NUMERIC_FEATURES.forEach(feature => {
            obj[feature] = row[feature];
        });
        return obj;
    }).filter(row => Object.values(row).every(val => val !== null && !isNaN(val)));
    
    if (numericData.length === 0) return;
    
    const correlations = calculateCorrelations(numericData);
    
    const correlationCtx = document.getElementById('correlationChart').getContext('2d');
    charts.correlationChart = new Chart(correlationCtx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Correlation Matrix',
                data: correlations.points,
                backgroundColor(context) {
                    const value = context.dataset.data[context.dataIndex].v;
                    const alpha = Math.abs(value);
                    return `rgba(159, 122, 234, ${alpha})`;
                },
                borderColor: 'rgba(159, 122, 234, 1)',
                borderWidth: 1,
                width: ({ chart }) => (chart.chartArea.width / NUMERIC_FEATURES.length) - 1,
                height: ({ chart }) => (chart.chartArea.height / NUMERIC_FEATURES.length) - 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Correlation Heatmap' },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const point = context.dataset.data[context.dataIndex];
                            return `${point.x} vs ${point.y}: ${point.v.toFixed(3)}`;
                        }
                    }
                },
                legend: { display: false }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: NUMERIC_FEATURES,
                    offset: true,
                    ticks: {
                        maxRotation: 45
                    },
                    title: {
                        display: true,
                        text: 'Features'
                    }
                },
                y: {
                    type: 'category',
                    labels: NUMERIC_FEATURES,
                    offset: true,
                    title: {
                        display: true,
                        text: 'Features'
                    }
                }
            }
        }
    });
}

function calculateCorrelations(data) {
    const features = NUMERIC_FEATURES;
    const points = [];
    
    for (let i = 0; i < features.length; i++) {
        for (let j = 0; j < features.length; j++) {
            const xVals = data.map(row => row[features[i]]);
            const yVals = data.map(row => row[features[j]]);
            const correlation = pearsonCorrelation(xVals, yVals);
            
            points.push({
                x: features[i],
                y: features[j],
                v: isNaN(correlation) ? 0 : correlation
            });
        }
    }
    
    return { points, features };
}

function pearsonCorrelation(x, y) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

// ==================== EXPORT FUNCTIONS ====================
function exportMergedCSV() {
    if (mergedData.length === 0) {
        showAlert('No data to export', 'error', exportStatus);
        return;
    }
    
    try {
        const csv = Papa.unparse(mergedData);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        
        link.href = url;
        link.setAttribute('download', `titanic_merged_${new Date().toISOString().split('T')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showAlert('CSV exported successfully!', 'success', exportStatus);
    } catch (error) {
        showAlert(`Export failed: ${error.message}`, 'error', exportStatus);
    }
}

function exportStatsJSON() {
    if (!window.edaStats) {
        showAlert('Run EDA first to generate statistics', 'error', exportStatus);
        return;
    }
    
    try {
        const json = JSON.stringify(window.edaStats, null, 2);
        const blob = new Blob([json], { type: 'application/json;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        
        link.href = url;
        link.setAttribute('download', `titanic_stats_${new Date().toISOString().split('T')[0]}.json`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showAlert('JSON exported successfully!', 'success', exportStatus);
    } catch (error) {
        showAlert(`Export failed: ${error.message}`, 'error', exportStatus);
    }
}

// ==================== UTILITY FUNCTIONS ====================
function showAlert(message, type = 'info', element = null) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert ${type}`;
    alertDiv.innerHTML = message;
    
    if (element) {
        // Clear existing content and add alert
        element.innerHTML = '';
        element.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds for success/info messages
        if (type !== 'error') {
            setTimeout(() => {
                if (alertDiv.parentNode === element) {
                    element.removeChild(alertDiv);
                }
            }, 5000);
        }
    } else {
        // Show as temporary alert at top of page
        alertDiv.style.position = 'fixed';
        alertDiv.style.top = '20px';
        alertDiv.style.right = '20px';
        alertDiv.style.zIndex = '1000';
        alertDiv.style.minWidth = '300px';
        alertDiv.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
        document.body.appendChild(alertDiv);
        
        setTimeout(() => {
            if (alertDiv.parentNode === document.body) {
                document.body.removeChild(alertDiv);
            }
        }, 5000);
    }
}

// Add histogram chart type to Chart.js
Chart.register({
    id: 'histogram',
    beforeInit(chart) {
        chart.options.scales.x.type = 'linear';
    }
});

// Initial state
showAlert('Please load train.csv and test.csv files to begin analysis', 'info');







// app.js - Titanic EDA Explorer
// To reuse this app for other datasets:
// 1. Update the schema definition (lines 40-50)
// 2. Update feature lists for numeric/categorical (lines 53-55)
// 3. Update column exclusions if needed (line 51)
// 4. Update target variable if needed (line 52)

document.addEventListener('DOMContentLoaded', () => {
    // ==================== CONFIGURATION ====================
    // DATASET SCHEMA CONFIGURATION - Update these for different datasets
    const DATASET_SCHEMA = {
        // Features to analyze (update for different datasets)
        features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        
        // Target variable (only in train data)
        target: 'Survived',
        
        // Identifier column to exclude from analysis
        identifier: 'PassengerId',
        
        // Column types (update based on your dataset)
        numericFeatures: ['Age', 'Fare', 'SibSp', 'Parch'],
        categoricalFeatures: ['Pclass', 'Sex', 'Embarked'],
        
        // Source column name
        sourceCol: 'Dataset_Source'
    };
    
    // ==================== STATE MANAGEMENT ====================
    let mergedData = null;
    let trainData = null;
    let testData = null;
    let charts = {};
    let analysisResults = {};
    
    // ==================== DOM ELEMENTS ====================
    const trainFileInput = document.getElementById('trainFile');
    const testFileInput = document.getElementById('testFile');
    const loadBtn = document.getElementById('loadBtn');
    const runEDABtn = document.getElementById('runEDABtn');
    const exportBtn = document.getElementById('exportBtn');
    const vizBarBtn = document.getElementById('vizBarBtn');
    const vizHistBtn = document.getElementById('vizHistBtn');
    const vizCorrBtn = document.getElementById('vizCorrBtn');
    const loadAlert = document.getElementById('loadAlert');
    const exportAlert = document.getElementById('exportAlert');
    const overviewStats = document.getElementById('overviewStats');
    const previewTable = document.getElementById('previewTable');
    const missingValues = document.getElementById('missingValues');
    const numericStats = document.getElementById('numericStats');
    const categoricalStats = document.getElementById('categoricalStats');
    const missingChartCtx = document.getElementById('missingChart').getContext('2d');
    const mainChartCtx = document.getElementById('mainChart').getContext('2d');
    
    // ==================== EVENT LISTENERS ====================
    loadBtn.addEventListener('click', loadAndMergeData);
    runEDABtn.addEventListener('click', runFullEDA);
    exportBtn.addEventListener('click', exportResults);
    vizBarBtn.addEventListener('click', () => createBarCharts());
    vizHistBtn.addEventListener('click', () => createHistograms());
    vizCorrBtn.addEventListener('click', () => createCorrelationHeatmap());
    
    // ==================== DATA LOADING & MERGING ====================
    function loadAndMergeData() {
        showAlert(loadAlert, 'Loading datasets...', 'info');
        
        const trainFile = trainFileInput.files[0];
        const testFile = testFileInput.files[0];
        
        if (!trainFile || !testFile) {
            showAlert(loadAlert, 'Please select both train and test CSV files', 'error');
            return;
        }
        
        // Parse train data
        Papa.parse(trainFile, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (trainResults) => {
                if (trainResults.errors.length > 0) {
                    showAlert(loadAlert, `Train CSV error: ${trainResults.errors[0].message}`, 'error');
                    return;
                }
                
                trainData = trainResults.data.map(row => ({
                    ...row,
                    [DATASET_SCHEMA.sourceCol]: 'train'
                }));
                
                // Parse test data
                Papa.parse(testFile, {
                    header: true,
                    dynamicTyping: true,
                    skipEmptyLines: true,
                    complete: (testResults) => {
                        if (testResults.errors.length > 0) {
                            showAlert(loadAlert, `Test CSV error: ${testResults.errors[0].message}`, 'error');
                            return;
                        }
                        
                        testData = testResults.data.map(row => ({
                            ...row,
                            [DATASET_SCHEMA.sourceCol]: 'test'
                        }));
                        
                        // Merge datasets
                        mergedData = [...trainData, ...testData];
                        
                        // Update UI
                        showAlert(loadAlert, `Successfully loaded ${mergedData.length} rows`, 'success');
                        runEDABtn.disabled = false;
                        exportBtn.disabled = false;
                        vizBarBtn.disabled = false;
                        vizHistBtn.disabled = false;
                        vizCorrBtn.disabled = false;
                        
                        // Show overview
                        updateOverview();
                    },
                    error: (error) => {
                        showAlert(loadAlert, `Test file error: ${error.message}`, 'error');
                    }
                });
            },
            error: (error) => {
                showAlert(loadAlert, `Train file error: ${error.message}`, 'error');
            }
        });
    }
    
    // ==================== EDA FUNCTIONS ====================
    function runFullEDA() {
        if (!mergedData) {
            showAlert(loadAlert, 'Please load data first', 'error');
            return;
        }
        
        // Clear existing charts
        Object.values(charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        charts = {};
        
        // Run all analyses
        analyzeMissingValues();
        calculateStatistics();
        createBarCharts();
        
        showAlert(loadAlert, 'EDA analysis complete!', 'success');
    }
    
    function updateOverview() {
        if (!mergedData) return;
        
        // Calculate overview statistics
        const trainRows = mergedData.filter(row => row[DATASET_SCHEMA.sourceCol] === 'train').length;
        const testRows = mergedData.filter(row => row[DATASET_SCHEMA.sourceCol] === 'test').length;
        const columns = mergedData.length > 0 ? Object.keys(mergedData[0]).length : 0;
        
        // Update overview stats
        overviewStats.innerHTML = `
            <div class="stat-card">
                <div class="stat-title">Total Rows</div>
                <div class="stat-value">${mergedData.length}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Train Rows</div>
                <div class="stat-value">${trainRows}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Test Rows</div>
                <div class="stat-value">${testRows}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Columns</div>
                <div class="stat-value">${columns}</div>
            </div>
        `;
        
        // Show data preview
        showDataPreview();
    }
    
    function showDataPreview() {
        if (!mergedData || mergedData.length === 0) return;
        
        const previewRows = mergedData.slice(0, 5);
        const headers = Object.keys(previewRows[0]);
        
        let tableHTML = '<table><thead><tr>';
        headers.forEach(header => {
            tableHTML += `<th>${header}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';
        
        previewRows.forEach(row => {
            tableHTML += '<tr>';
            headers.forEach(header => {
                const value = row[header];
                tableHTML += `<td>${value === null || value === undefined ? 'NaN' : value}</td>`;
            });
            tableHTML += '</tr>';
        });
        
        tableHTML += '</tbody></table>';
        previewTable.innerHTML = tableHTML;
    }
    
    function analyzeMissingValues() {
        if (!mergedData) return;
        
        // Calculate missing values for each column
        const columns = Object.keys(mergedData[0]);
        const missingCounts = {};
        
        columns.forEach(col => {
            const missing = mergedData.filter(row => 
                row[col] === null || row[col] === undefined || 
                row[col] === '' || isNaN(row[col])
            ).length;
            missingCounts[col] = {
                count: missing,
                percentage: (missing / mergedData.length * 100).toFixed(1)
            };
        });
        
        // Store for export
        analysisResults.missingValues = missingCounts;
        
        // Update UI
        updateMissingValuesUI(missingCounts);
        createMissingChart(missingCounts);
    }
    
    function updateMissingValuesUI(missingCounts) {
        missingValues.innerHTML = '';
        
        Object.entries(missingCounts)
            .sort((a, b) => b[1].percentage - a[1].percentage)
            .forEach(([col, data]) => {
                const bar = document.createElement('div');
                bar.className = 'missing-bar';
                bar.innerHTML = `
                    <div class="missing-label">${col}</div>
                    <div class="missing-value">${data.percentage}%</div>
                    <div class="missing-visual">
                        <div class="missing-fill" style="width: ${data.percentage}%"></div>
                    </div>
                    <div>(${data.count})</div>
                `;
                missingValues.appendChild(bar);
            });
    }
    
    function createMissingChart(missingCounts) {
        const columns = Object.keys(missingCounts);
        const percentages = columns.map(col => parseFloat(missingCounts[col].percentage));
        
        if (charts.missingChart) charts.missingChart.destroy();
        
        charts.missingChart = new Chart(missingChartCtx, {
            type: 'bar',
            data: {
                labels: columns,
                datasets: [{
                    label: 'Missing Values (%)',
                    data: percentages,
                    backgroundColor: columns.map(p => 
                        p > 20 ? 'rgba(231, 76, 60, 0.8)' : 
                        p > 5 ? 'rgba(241, 196, 15, 0.8)' : 
                        'rgba(46, 204, 113, 0.8)'
                    ),
                    borderColor: columns.map(p => 
                        p > 20 ? 'rgba(192, 57, 43, 1)' : 
                        p > 5 ? 'rgba(243, 156, 18, 1)' : 
                        'rgba(39, 174, 96, 1)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Percentage Missing'
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Missing Values by Column'
                    }
                }
            }
        });
    }
    
    function calculateStatistics() {
        if (!mergedData || !trainData) return;
        
        // Numeric statistics
        const numericStatsResult = {};
        DATASET_SCHEMA.numericFeatures.forEach(feature => {
            const values = trainData
                .map(row => row[feature])
                .filter(val => val !== null && !isNaN(val));
            
            if (values.length > 0) {
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const sorted = [...values].sort((a, b) => a - b);
                const median = sorted[Math.floor(sorted.length / 2)];
                const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
                const stdDev = Math.sqrt(variance);
                
                numericStatsResult[feature] = {
                    mean: mean.toFixed(2),
                    median: median.toFixed(2),
                    stdDev: stdDev.toFixed(2),
                    min: Math.min(...values).toFixed(2),
                    max: Math.max(...values).toFixed(2)
                };
            }
        });
        
        // Categorical statistics
        const categoricalStatsResult = {};
        DATASET_SCHEMA.categoricalFeatures.forEach(feature => {
            const valueCounts = {};
            trainData.forEach(row => {
                const val = row[feature];
                if (val !== null && val !== undefined) {
                    valueCounts[val] = (valueCounts[val] || 0) + 1;
                }
            });
            
            categoricalStatsResult[feature] = valueCounts;
        });
        
        // Group by target variable (Survived)
        const groupedStats = {};
        if (trainData.some(row => row[DATASET_SCHEMA.target] !== undefined)) {
            [0, 1].forEach(targetValue => {
                const subset = trainData.filter(row => row[DATASET_SCHEMA.target] === targetValue);
                const groupStats = {};
                
                DATASET_SCHEMA.numericFeatures.forEach(feature => {
                    const values = subset
                        .map(row => row[feature])
                        .filter(val => val !== null && !isNaN(val));
                    
                    if (values.length > 0) {
                        const mean = values.reduce((a, b) => a + b, 0) / values.length;
                        groupStats[feature] = {
                            mean: mean.toFixed(2),
                            count: values.length
                        };
                    }
                });
                
                groupedStats[`Survived_${targetValue}`] = groupStats;
            });
        }
        
        // Store for export
        analysisResults.statistics = {
            numeric: numericStatsResult,
            categorical: categoricalStatsResult,
            grouped: groupedStats
        };
        
        // Update UI
        updateStatisticsUI(numericStatsResult, categoricalStatsResult, groupedStats);
    }
    
    function updateStatisticsUI(numericStats, categoricalStats, groupedStats) {
        // Numeric stats
        let numericHTML = '<h3>Numeric Features</h3><div class="stats-grid">';
        Object.entries(numericStats).forEach(([feature, stats]) => {
            numericHTML += `
                <div class="stat-card">
                    <div class="stat-title">${feature}</div>
                    <div>Mean: ${stats.mean}</div>
                    <div>Median: ${stats.median}</div>
                    <div>Std Dev: ${stats.stdDev}</div>
                    <div>Range: ${stats.min} - ${stats.max}</div>
                </div>
            `;
        });
        numericHTML += '</div>';
        numericStats.innerHTML = numericHTML;
        
        // Categorical stats
        let categoricalHTML = '<h3>Categorical Features</h3>';
        Object.entries(categoricalStats).forEach(([feature, counts]) => {
            categoricalHTML += `<h4>${feature}</h4><div class="stats-grid">`;
            Object.entries(counts).forEach(([value, count]) => {
                const percentage = (count / trainData.length * 100).toFixed(1);
                categoricalHTML += `
                    <div class="stat-card">
                        <div class="stat-title">${value}</div>
                        <div class="stat-value">${count}</div>
                        <div>${percentage}%</div>
                    </div>
                `;
            });
            categoricalHTML += '</div>';
        });
        categoricalStats.innerHTML = categoricalHTML;
    }
    
    function createBarCharts() {
        if (!trainData) return;
        
        // Create bar chart for categorical features
        const features = ['Sex', 'Pclass', 'Embarked'];
        const labels = [];
        const survivedData = [];
        const notSurvivedData = [];
        
        features.forEach(feature => {
            const values = [...new Set(trainData.map(row => row[feature]).filter(v => v !== null))];
            values.forEach(value => {
                const subset = trainData.filter(row => row[feature] === value);
                const survived = subset.filter(row => row[DATASET_SCHEMA.target] === 1).length;
                const notSurvived = subset.filter(row => row[DATASET_SCHEMA.target] === 0).length;
                
                labels.push(`${feature}: ${value}`);
                survivedData.push(survived);
                notSurvivedData.push(notSurvived);
            });
        });
        
        if (charts.mainChart) charts.mainChart.destroy();
        
        charts.mainChart = new Chart(mainChartCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Survived',
                        data: survivedData,
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgba(39, 174, 96, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Not Survived',
                        data: notSurvivedData,
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgba(192, 57, 43, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Survival by Categorical Features'
                    }
                }
            }
        });
    }
    
    function createHistograms() {
        if (!trainData) return;
        
        // Create histogram for Age
        const ageData = trainData
            .map(row => row.Age)
            .filter(age => age !== null && !isNaN(age));
        
        // Create bins for histogram
        const binSize = 10;
        const maxAge = Math.ceil(Math.max(...ageData) / binSize) * binSize;
        const bins = Array(Math.floor(maxAge / binSize)).fill(0);
        
        ageData.forEach(age => {
            const binIndex = Math.floor(age / binSize);
            if (binIndex < bins.length) {
                bins[binIndex]++;
            }
        });
        
        const binLabels = Array.from({length: bins.length}, (_, i) => 
            `${i * binSize}-${(i + 1) * binSize}`
        );
        
        if (charts.mainChart) charts.mainChart.destroy();
        
        charts.mainChart = new Chart(mainChartCtx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: 'Age Distribution',
                    data: bins,
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgba(41, 128, 185, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Age Range'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Age Distribution Histogram'
                    }
                }
            }
        });
    }
    
    function createCorrelationHeatmap() {
        if (!trainData) return;
        
        // Calculate correlations between numeric features
        const numericFeatures = DATASET_SCHEMA.numericFeatures.filter(feature => 
            feature !== DATASET_SCHEMA.identifier
        );
        
        const correlationMatrix = [];
        const labels = [...numericFeatures, DATASET_SCHEMA.target];
        
        // Prepare data matrix
        const dataMatrix = [];
        trainData.forEach(row => {
            const values = [];
            labels.forEach(label => {
                values.push(row[label]);
            });
            // Only include rows with all numeric values
            if (values.every(v => v !== null && !isNaN(v))) {
                dataMatrix.push(values);
            }
        });
        
        // Calculate correlation for each pair
        labels.forEach((_, i) => {
            correlationMatrix[i] = [];
            labels.forEach((_, j) => {
                if (i === j) {
                    correlationMatrix[i][j] = 1;
                } else {
                    const x = dataMatrix.map(row => row[i]);
                    const y = dataMatrix.map(row => row[j]);
                    correlationMatrix[i][j] = pearsonCorrelation(x, y);
                }
            });
        });
        
        // Create heatmap
        if (charts.mainChart) charts.mainChart.destroy();
        
        charts.mainChart = new Chart(mainChartCtx, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Correlation Matrix',
                    data: correlationMatrix.flatMap((row, i) => 
                        row.map((value, j) => ({
                            x: j,
                            y: i,
                            v: value
                        }))
                    ),
                    backgroundColor: (ctx) => {
                        const value = ctx.dataset.data[ctx.dataIndex].v;
                        const alpha = Math.abs(value);
                        return value >= 0 
                            ? `rgba(46, 204, 113, ${alpha})`
                            : `rgba(231, 76, 60, ${alpha})`;
                    },
                    borderWidth: 1,
                    borderColor: '#fff',
                    width: ({chart}) => (chart.chartArea || {}).width / labels.length - 1,
                    height: ({chart}) => (chart.chartArea || {}).height / labels.length - 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const data = ctx.dataset.data[ctx.dataIndex];
                                return `Correlation: ${data.v.toFixed(3)}`;
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Correlation Heatmap'
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            callback: (i) => labels[i],
                            display: true
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        ticks: {
                            callback: (i) => labels[i],
                            display: true
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
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
    function exportResults() {
        if (!mergedData || !analysisResults) {
            showAlert(exportAlert, 'No data to export', 'error');
            return;
        }
        
        try {
            // Export merged data as CSV
            const csvContent = Papa.unparse(mergedData);
            downloadFile(csvContent, 'titanic_merged_data.csv', 'text/csv');
            
            // Export analysis results as JSON
            const exportData = {
                timestamp: new Date().toISOString(),
                datasetInfo: {
                    totalRows: mergedData.length,
                    trainRows: mergedData.filter(row => row[DATASET_SCHEMA.sourceCol] === 'train').length,
                    testRows: mergedData.filter(row => row[DATASET_SCHEMA.sourceCol] === 'test').length,
                    features: DATASET_SCHEMA.features
                },
                analysis: analysisResults
            };
            
            const jsonContent = JSON.stringify(exportData, null, 2);
            downloadFile(jsonContent, 'titanic_eda_summary.json', 'application/json');
            
            showAlert(exportAlert, 'Export completed successfully!', 'success');
        } catch (error) {
            showAlert(exportAlert, `Export error: ${error.message}`, 'error');
        }
    }
    
    function downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // ==================== UTILITY FUNCTIONS ====================
    function showAlert(alertElement, message, type) {
        alertElement.textContent = message;
        alertElement.className = `alert alert-${type}`;
        alertElement.style.display = 'block';
        
        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                alertElement.style.display = 'none';
            }, 5000);
        }
    }
    
    // Initialize the application
    console.log('Titanic EDA Explorer initialized');
    console.log('To analyze different datasets:');
    console.log('1. Update DATASET_SCHEMA configuration');
    console.log('2. Adjust feature lists for your dataset');
    console.log('3. Update target variable if needed');
});

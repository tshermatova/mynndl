// app.js - Titanic EDA Explorer
// To reuse this app for other datasets:
// 1. Update DATASET_SCHEMA configuration
// 2. Update feature lists for numeric/categorical
// 3. Update target variable if different

document.addEventListener('DOMContentLoaded', () => {
    // ==================== CONFIGURATION ====================
    const DATASET_SCHEMA = {
        features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        target: 'Survived',
        identifier: 'PassengerId',
        numericFeatures: ['Age', 'Fare', 'SibSp', 'Parch'],
        categoricalFeatures: ['Pclass', 'Sex', 'Embarked'],
        sourceCol: 'Dataset_Source'
    };
    
    const COLORS = {
        primary: '#3498db',
        success: '#27ae60',
        danger: '#e74c3c',
        warning: '#f39c12',
        info: '#9b59b6',
        dark: '#2c3e50',
        light: '#ecf0f1'
    };
    
    // ==================== STATE MANAGEMENT ====================
    let mergedData = null;
    let trainData = null;
    let testData = null;
    let charts = {};
    let analysisResults = {
        overview: {},
        missingValues: {},
        statistics: {},
        visualizations: {},
        correlationInsights: {}
    };
    
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
        loadBtn.disabled = true;
        
        const trainFile = trainFileInput.files[0];
        const testFile = testFileInput.files[0];
        
        if (!trainFile || !testFile) {
            showAlert(loadAlert, 'Please select both train and test CSV files', 'error');
            loadBtn.disabled = false;
            return;
        }
        
        resetApplication();
        
        Papa.parse(trainFile, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (trainResults) => {
                if (trainResults.errors.length > 0) {
                    showAlert(loadAlert, `Train CSV error: ${trainResults.errors[0].message}`, 'error');
                    loadBtn.disabled = false;
                    return;
                }
                
                if (trainResults.data.length === 0) {
                    showAlert(loadAlert, 'Train CSV file is empty', 'error');
                    loadBtn.disabled = false;
                    return;
                }
                
                trainData = trainResults.data.map(row => ({
                    ...row,
                    [DATASET_SCHEMA.sourceCol]: 'train'
                }));
                
                Papa.parse(testFile, {
                    header: true,
                    dynamicTyping: true,
                    skipEmptyLines: true,
                    complete: (testResults) => {
                        if (testResults.errors.length > 0) {
                            showAlert(loadAlert, `Test CSV error: ${testResults.errors[0].message}`, 'error');
                            loadBtn.disabled = false;
                            return;
                        }
                        
                        if (testResults.data.length === 0) {
                            showAlert(loadAlert, 'Test CSV file is empty', 'error');
                            loadBtn.disabled = false;
                            return;
                        }
                        
                        testData = testResults.data.map(row => ({
                            ...row,
                            [DATASET_SCHEMA.sourceCol]: 'test'
                        }));
                        
                        mergedData = [...trainData, ...testData];
                        
                        analysisResults.overview = {
                            totalRows: mergedData.length,
                            trainRows: trainData.length,
                            testRows: testData.length,
                            columns: Object.keys(mergedData[0]).length,
                            features: DATASET_SCHEMA.features,
                            timestamp: new Date().toISOString()
                        };
                        
                        showAlert(loadAlert, `✅ Successfully loaded ${mergedData.length} rows (${trainData.length} train + ${testData.length} test)`, 'success');
                        loadBtn.disabled = false;
                        runEDABtn.disabled = false;
                        exportBtn.disabled = false;
                        vizBarBtn.disabled = false;
                        vizHistBtn.disabled = false;
                        vizCorrBtn.disabled = false;
                        
                        updateOverview();
                    },
                    error: (error) => {
                        showAlert(loadAlert, `Test file error: ${error.message}`, 'error');
                        loadBtn.disabled = false;
                    }
                });
            },
            error: (error) => {
                showAlert(loadAlert, `Train file error: ${error.message}`, 'error');
                loadBtn.disabled = false;
            }
        });
    }
    
    // ==================== EDA FUNCTIONS ====================
    function runFullEDA() {
        if (!mergedData) {
            showAlert(loadAlert, 'Please load data first', 'error');
            return;
        }
        
        showAlert(loadAlert, 'Running full EDA analysis...', 'info');
        runEDABtn.disabled = true;
        
        destroyAllCharts();
        
        setTimeout(() => {
            analyzeMissingValues();
            calculateStatistics();
            createBarCharts();
            
            showAlert(loadAlert, '✅ EDA analysis complete! Check all sections for results.', 'success');
            runEDABtn.disabled = false;
        }, 100);
    }
    
    function updateOverview() {
        if (!mergedData || mergedData.length === 0) return;
        
        const trainRows = trainData ? trainData.length : 0;
        const testRows = testData ? testData.length : 0;
        const columns = mergedData.length > 0 ? Object.keys(mergedData[0]).length : 0;
        
        overviewStats.innerHTML = `
            <div class="stat-card">
                <div class="stat-title">Total Rows</div>
                <div class="stat-value">${mergedData.length.toLocaleString()}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Train Rows</div>
                <div class="stat-value">${trainRows.toLocaleString()}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Test Rows</div>
                <div class="stat-value">${testRows.toLocaleString()}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Columns</div>
                <div class="stat-value">${columns}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Features</div>
                <div class="stat-value">${DATASET_SCHEMA.features.length}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Target</div>
                <div class="stat-value">${DATASET_SCHEMA.target}</div>
            </div>
        `;
        
        showDataPreview();
    }
    
    function showDataPreview() {
        if (!mergedData || mergedData.length === 0) {
            previewTable.innerHTML = '<p>No data to display</p>';
            return;
        }
        
        const previewRows = mergedData.slice(0, 8);
        const headers = Object.keys(previewRows[0]).filter(h => 
            h !== DATASET_SCHEMA.identifier
        );
        
        let tableHTML = '<table><thead><tr>';
        headers.forEach(header => {
            tableHTML += `<th>${header}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';
        
        previewRows.forEach(row => {
            tableHTML += '<tr>';
            headers.forEach(header => {
                let value = row[header];
                if (value === null || value === undefined || value === '') {
                    value = '<span style="color: #e74c3c; font-style: italic;">NaN</span>';
                } else if (typeof value === 'number') {
                    value = Number.isInteger(value) ? value : value.toFixed(2);
                }
                tableHTML += `<td>${value}</td>`;
            });
            tableHTML += '</tr>';
        });
        
        tableHTML += '</tbody></table>';
        previewTable.innerHTML = tableHTML;
    }
    
    function analyzeMissingValues() {
        if (!mergedData || mergedData.length === 0) return;
        
        const columns = Object.keys(mergedData[0]).filter(col => 
            col !== DATASET_SCHEMA.sourceCol
        );
        
        const missingCounts = {};
        const missingChartData = {
            columns: [],
            counts: [],
            percentages: []
        };
        
        columns.forEach(col => {
            const missing = mergedData.filter(row => {
                const val = row[col];
                return val === null || val === undefined || val === '' || 
                       (typeof val === 'number' && isNaN(val));
            }).length;
            
            const percentage = (missing / mergedData.length * 100).toFixed(2);
            missingCounts[col] = {
                count: missing,
                percentage: parseFloat(percentage),
                hasMissing: missing > 0
            };
            
            if (missing > 0) {
                missingChartData.columns.push(col);
                missingChartData.counts.push(missing);
                missingChartData.percentages.push(parseFloat(percentage));
            }
        });
        
        analysisResults.missingValues = missingCounts;
        
        updateMissingValuesUI(missingCounts);
        createMissingChart(missingChartData);
    }
    
    function updateMissingValuesUI(missingCounts) {
        if (!missingCounts || Object.keys(missingCounts).length === 0) {
            missingValues.innerHTML = '<p>No missing values detected</p>';
            return;
        }
        
        missingValues.innerHTML = '';
        
        const sortedEntries = Object.entries(missingCounts)
            .filter(([_, data]) => data.count > 0)
            .sort((a, b) => b[1].percentage - a[1].percentage);
        
        if (sortedEntries.length === 0) {
            missingValues.innerHTML = '<p style="color: #27ae60;">✅ No missing values found!</p>';
            return;
        }
        
        sortedEntries.forEach(([col, data]) => {
            const bar = document.createElement('div');
            bar.className = 'missing-bar';
            bar.innerHTML = `
                <div class="missing-label">${col}</div>
                <div class="missing-value">${data.percentage.toFixed(1)}%</div>
                <div class="missing-visual">
                    <div class="missing-fill" style="width: ${Math.min(data.percentage, 100)}%"></div>
                </div>
                <div style="min-width: 50px; text-align: right;">(${data.count})</div>
            `;
            missingValues.appendChild(bar);
        });
    }
    
    function createMissingChart(chartData) {
        if (!chartData || chartData.columns.length === 0) {
            if (charts.missingChart) {
                charts.missingChart.destroy();
                delete charts.missingChart;
            }
            return;
        }
        
        if (charts.missingChart) {
            charts.missingChart.destroy();
        }
        
        const sortedIndices = chartData.percentages
            .map((_, idx) => idx)
            .sort((a, b) => chartData.percentages[b] - chartData.percentages[a]);
        
        const sortedColumns = sortedIndices.map(i => chartData.columns[i]);
        const sortedPercentages = sortedIndices.map(i => chartData.percentages[i]);
        
        charts.missingChart = new Chart(missingChartCtx, {
            type: 'bar',
            data: {
                labels: sortedColumns,
                datasets: [{
                    label: 'Missing Values (%)',
                    data: sortedPercentages,
                    backgroundColor: sortedPercentages.map(p => 
                        p > 50 ? 'rgba(231, 76, 60, 0.8)' : 
                        p > 20 ? 'rgba(241, 196, 15, 0.8)' : 
                        p > 5 ? 'rgba(52, 152, 219, 0.8)' : 
                        'rgba(46, 204, 113, 0.8)'
                    ),
                    borderColor: sortedPercentages.map(p => 
                        p > 50 ? 'rgba(192, 57, 43, 1)' : 
                        p > 20 ? 'rgba(243, 156, 18, 1)' : 
                        p > 5 ? 'rgba(41, 128, 185, 1)' : 
                        'rgba(39, 174, 96, 1)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Missing Values by Column (%)',
                        font: { size: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Missing: ${context.parsed.y.toFixed(2)}%`
                        }
                    },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Percentage Missing' },
                        ticks: { callback: (value) => `${value}%` }
                    },
                    x: {
                        ticks: { maxRotation: 45, minRotation: 0 }
                    }
                }
            }
        });
    }
    
    function calculateStatistics() {
        if (!mergedData || !trainData) return;
        
        const numericColumns = DATASET_SCHEMA.numericFeatures.filter(col => 
            col !== DATASET_SCHEMA.identifier
        );
        
        const categoricalColumns = DATASET_SCHEMA.categoricalFeatures.filter(col => 
            col !== DATASET_SCHEMA.identifier
        );
        
        // Numeric statistics
        const numericStatsResult = {};
        numericColumns.forEach(feature => {
            const values = trainData
                .map(row => row[feature])
                .filter(val => val !== null && !isNaN(val) && typeof val === 'number');
            
            if (values.length > 0) {
                const sum = values.reduce((a, b) => a + b, 0);
                const mean = sum / values.length;
                const sorted = [...values].sort((a, b) => a - b);
                const median = sorted.length % 2 === 0 
                    ? (sorted[sorted.length/2 - 1] + sorted[sorted.length/2]) / 2
                    : sorted[Math.floor(sorted.length / 2)];
                
                const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
                const stdDev = Math.sqrt(variance);
                
                numericStatsResult[feature] = {
                    count: values.length,
                    missing: trainData.length - values.length,
                    mean: parseFloat(mean.toFixed(4)),
                    median: parseFloat(median.toFixed(4)),
                    stdDev: parseFloat(stdDev.toFixed(4)),
                    min: parseFloat(Math.min(...values).toFixed(4)),
                    max: parseFloat(Math.max(...values).toFixed(4)),
                    range: parseFloat((Math.max(...values) - Math.min(...values)).toFixed(4)),
                    sum: parseFloat(sum.toFixed(4))
                };
            } else {
                numericStatsResult[feature] = {
                    count: 0,
                    missing: trainData.length,
                    mean: 0,
                    median: 0,
                    stdDev: 0,
                    min: 0,
                    max: 0,
                    range: 0,
                    sum: 0
                };
            }
        });
        
        // Categorical statistics
        const categoricalStatsResult = {};
        categoricalColumns.forEach(feature => {
            const valueCounts = {};
            const totalCount = trainData.length;
            let missingCount = 0;
            
            trainData.forEach(row => {
                const val = row[feature];
                if (val === null || val === undefined || val === '' || 
                    (typeof val === 'number' && isNaN(val))) {
                    missingCount++;
                } else {
                    const key = String(val);
                    valueCounts[key] = (valueCounts[key] || 0) + 1;
                }
            });
            
            const sortedCounts = Object.entries(valueCounts)
                .sort((a, b) => b[1] - a[1])
                .map(([value, count]) => ({
                    value,
                    count,
                    percentage: totalCount > 0 ? parseFloat((count / totalCount * 100).toFixed(2)) : 0
                }));
            
            categoricalStatsResult[feature] = {
                uniqueValues: Object.keys(valueCounts).length,
                missing: missingCount,
                missingPercentage: totalCount > 0 ? parseFloat((missingCount / totalCount * 100).toFixed(2)) : 0,
                valueCounts: sortedCounts,
                mostCommon: sortedCounts.length > 0 ? sortedCounts[0] : null
            };
        });
        
        // Group statistics by survival
        const groupedStats = {};
        if (trainData.some(row => row[DATASET_SCHEMA.target] !== undefined)) {
            const targetValues = [0, 1];
            
            targetValues.forEach(targetValue => {
                const subset = trainData.filter(row => row[DATASET_SCHEMA.target] === targetValue);
                const groupStats = {};
                
                numericColumns.forEach(feature => {
                    const values = subset
                        .map(row => row[feature])
                        .filter(val => val !== null && !isNaN(val) && typeof val === 'number');
                    
                    if (values.length > 0) {
                        const mean = values.reduce((a, b) => a + b, 0) / values.length;
                        groupStats[feature] = {
                            count: values.length,
                            mean: parseFloat(mean.toFixed(4)),
                            stdDev: parseFloat(Math.sqrt(
                                values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length
                            ).toFixed(4))
                        };
                    }
                });
                
                groupedStats[`${DATASET_SCHEMA.target}=${targetValue}`] = {
                    count: subset.length,
                    percentage: trainData.length > 0 ? parseFloat((subset.length / trainData.length * 100).toFixed(2)) : 0,
                    stats: groupStats
                };
            });
        }
        
        analysisResults.statistics = {
            numeric: numericStatsResult,
            categorical: categoricalStatsResult,
            grouped: groupedStats,
            summary: {
                numericFeatures: numericColumns,
                categoricalFeatures: categoricalColumns,
                totalTrainSamples: trainData.length
            }
        };
        
        updateStatisticsUI(numericStatsResult, categoricalStatsResult, groupedStats);
    }
    
    function updateStatisticsUI(numericStats, categoricalStats, groupedStats) {
        // Clear previous content
        numericStats.innerHTML = '';
        categoricalStats.innerHTML = '';
        
        // Display numeric statistics
        if (Object.keys(numericStats).length === 0) {
            numericStats.innerHTML = '<p>No numeric features found</p>';
        } else {
            let numericHTML = '<h3>📊 Numeric Features Summary</h3><div class="stats-grid">';
            Object.entries(numericStats).forEach(([feature, stats]) => {
                numericHTML += `
                    <div class="stat-card" style="border-left-color: ${COLORS.primary}">
                        <div class="stat-title">${feature}</div>
                        <div style="margin-top: 5px;">
                            <div>📈 Mean: <strong>${stats.mean}</strong></div>
                            <div>📊 Median: <strong>${stats.median}</strong></div>
                            <div>📐 Std Dev: <strong>${stats.stdDev}</strong></div>
                            <div>📉 Range: <strong>${stats.min} - ${stats.max}</strong></div>
                            <div>🔢 Count: <strong>${stats.count}</strong></div>
                            <div>❌ Missing: <strong>${stats.missing}</strong></div>
                        </div>
                    </div>
                `;
            });
            numericHTML += '</div>';
            numericStats.innerHTML = numericHTML;
        }
        
        // Display categorical statistics
        if (Object.keys(categoricalStats).length === 0) {
            categoricalStats.innerHTML = '<p>No categorical features found</p>';
        } else {
            let categoricalHTML = '<h3>📊 Categorical Features Summary</h3>';
            
            Object.entries(categoricalStats).forEach(([feature, stats]) => {
                categoricalHTML += `
                    <div style="margin-bottom: 20px;">
                        <h4>${feature} 
                            <span style="font-size: 0.8rem; color: #7f8c8d;">
                                (${stats.uniqueValues} unique values)
                            </span>
                        </h4>
                        <div class="stats-grid" style="margin-top: 10px;">
                `;
                
                // Display top 6 values
                stats.valueCounts.slice(0, 6).forEach(item => {
                    categoricalHTML += `
                        <div class="stat-card" style="border-left-color: ${COLORS.info}">
                            <div class="stat-title">${item.value}</div>
                            <div class="stat-value">${item.count}</div>
                            <div>${item.percentage}%</div>
                        </div>
                    `;
                });
                
                // Show "more" if there are additional values
                if (stats.valueCounts.length > 6) {
                    categoricalHTML += `
                        <div class="stat-card" style="border-left-color: ${COLORS.light}">
                            <div class="stat-title">+${stats.valueCounts.length - 6} more</div>
                        </div>
                    `;
                }
                
                // Show missing values
                if (stats.missing > 0) {
                    categoricalHTML += `
                        <div class="stat-card" style="border-left-color: ${COLORS.danger}">
                            <div class="stat-title">Missing</div>
                            <div class="stat-value">${stats.missing}</div>
                            <div>${stats.missingPercentage}%</div>
                        </div>
                    `;
                }
                
                categoricalHTML += '</div></div>';
            });
            
            // Display survival statistics if available
            if (Object.keys(groupedStats).length > 0) {
                const totalTrain = trainData.length;
                const survived = trainData.filter(row => row[DATASET_SCHEMA.target] === 1).length;
                const died = trainData.filter(row => row[DATASET_SCHEMA.target] === 0).length;
                const survivalRate = totalTrain > 0 ? ((survived / totalTrain) * 100).toFixed(2) : '0.00';
                const deathRate = totalTrain > 0 ? ((died / totalTrain) * 100).toFixed(2) : '0.00';
                
                categoricalHTML += `
                    <h3>🎯 Survival Statistics</h3>
                    <div style="display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
                        <div class="stat-card" style="border-left-color: ${COLORS.success}; flex: 1;">
                            <div class="stat-title">✅ Survived</div>
                            <div class="stat-value">${survived}</div>
                            <div>${survivalRate}%</div>
                        </div>
                        <div class="stat-card" style="border-left-color: ${COLORS.danger}; flex: 1;">
                            <div class="stat-title">❌ Died</div>
                            <div class="stat-value">${died}</div>
                            <div>${deathRate}%</div>
                        </div>
                    </div>
                `;
                
                // Display grouped statistics
                Object.entries(groupedStats).forEach(([groupName, groupData]) => {
                    const isSurvived = groupName.includes('1');
                    const status = isSurvived ? 'Survived' : 'Died';
                    const icon = isSurvived ? '✅' : '❌';
                    
                    categoricalHTML += `
                        <div style="margin-bottom: 15px;">
                            <h4>${icon} ${status} (${groupData.count} passengers, ${groupData.percentage}%)</h4>
                            <div class="stats-grid">
                    `;
                    
                    Object.entries(groupData.stats).forEach(([feature, stats]) => {
                        categoricalHTML += `
                            <div class="stat-card" style="border-left-color: ${isSurvived ? COLORS.success : COLORS.danger}">
                                <div class="stat-title">${feature}</div>
                                <div>Mean: ${stats.mean}</div>
                                <div>Std Dev: ${stats.stdDev}</div>
                                <div>Count: ${stats.count}</div>
                            </div>
                        `;
                    });
                    
                    categoricalHTML += '</div></div>';
                });
            }
            
            categoricalStats.innerHTML = categoricalHTML;
        }
    }
    
    function createBarCharts() {
        if (!trainData || trainData.length === 0) {
            showAlert(loadAlert, 'Please load train data first', 'error');
            return;
        }
        
        if (!trainData[0].hasOwnProperty(DATASET_SCHEMA.target)) {
            showAlert(loadAlert, `Target variable '${DATASET_SCHEMA.target}' not found in train data`, 'error');
            return;
        }
        
        const categoricalFeatures = ['Pclass', 'Sex', 'Embarked'];
        const labels = [];
        const survivedCounts = [];
        const notSurvivedCounts = [];
        
        categoricalFeatures.forEach(feature => {
            const uniqueValues = [...new Set(
                trainData.map(row => row[feature]).filter(v => v !== null && v !== undefined)
            )].sort();
            
            uniqueValues.forEach(value => {
                const subset = trainData.filter(row => row[feature] === value);
                const survived = subset.filter(row => row[DATASET_SCHEMA.target] === 1).length;
                const notSurvived = subset.filter(row => row[DATASET_SCHEMA.target] === 0).length;
                
                labels.push(`${feature}: ${value}`);
                survivedCounts.push(survived);
                notSurvivedCounts.push(notSurvived);
            });
        });
        
        if (charts.mainChart) {
            charts.mainChart.destroy();
        }
        
        charts.mainChart = new Chart(mainChartCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: '✅ Survived',
                        data: survivedCounts,
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgba(39, 174, 96, 1)',
                        borderWidth: 1
                    },
                    {
                        label: '❌ Died',
                        data: notSurvivedCounts,
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgba(192, 57, 43, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Survival Distribution by Categorical Features',
                        font: { size: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const datasetLabel = context.dataset.label;
                                const value = context.parsed.y;
                                const total = survivedCounts[context.dataIndex] + notSurvivedCounts[context.dataIndex];
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${datasetLabel}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 0,
                            font: { size: 11 }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Count' }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
        
        analysisResults.visualizations.barCharts = {
            features: categoricalFeatures,
            labels: labels,
            survivedCounts: survivedCounts,
            notSurvivedCounts: notSurvivedCounts,
            type: 'bar'
        };
    }
    
    function createHistograms() {
        if (!trainData || trainData.length === 0) {
            showAlert(loadAlert, 'Please load train data first', 'error');
            return;
        }
        
        const ageData = trainData
            .map(row => row.Age)
            .filter(age => age !== null && !isNaN(age) && typeof age === 'number');
        
        if (ageData.length === 0) {
            showAlert(loadAlert, 'No valid Age data found for histogram', 'warning');
            return;
        }
        
        const sortedAge = [...ageData].sort((a, b) => a - b);
        const iqr = sortedAge[Math.floor(sortedAge.length * 0.75)] - sortedAge[Math.floor(sortedAge.length * 0.25)];
        const binWidth = 2 * iqr * Math.pow(sortedAge.length, -1/3);
        const numBins = Math.ceil((Math.max(...ageData) - Math.min(...ageData)) / binWidth);
        const actualBins = Math.min(Math.max(numBins, 5), 20);
        
        const minAge = Math.floor(Math.min(...ageData));
        const maxAge = Math.ceil(Math.max(...ageData));
        const binSize = (maxAge - minAge) / actualBins;
        
        const bins = Array(actualBins).fill(0);
        const binLabels = Array(actualBins).fill('');
        
        ageData.forEach(age => {
            const binIndex = Math.min(Math.floor((age - minAge) / binSize), actualBins - 1);
            bins[binIndex]++;
        });
        
        for (let i = 0; i < actualBins; i++) {
            const start = minAge + i * binSize;
            const end = minAge + (i + 1) * binSize;
            binLabels[i] = `${start.toFixed(0)}-${end.toFixed(0)}`;
        }
        
        if (charts.mainChart) {
            charts.mainChart.destroy();
        }
        
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
                plugins: {
                    title: {
                        display: true,
                        text: `Age Distribution Histogram (${ageData.length} samples, ${actualBins} bins)`,
                        font: { size: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const count = context.parsed.y;
                                const percentage = ((count / ageData.length) * 100).toFixed(1);
                                return `Count: ${count} (${percentage}%)`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Frequency' }
                    },
                    x: {
                        title: { display: true, text: 'Age Range' }
                    }
                }
            }
        });
        
        analysisResults.visualizations.histograms = {
            feature: 'Age',
            data: ageData,
            bins: bins,
            binLabels: binLabels,
            statistics: {
                mean: parseFloat((ageData.reduce((a, b) => a + b, 0) / ageData.length).toFixed(2)),
                median: parseFloat(sortedAge[Math.floor(sortedAge.length / 2)].toFixed(2)),
                min: minAge,
                max: maxAge,
                count: ageData.length
            },
            type: 'histogram'
        };
    }
    
    function createCorrelationHeatmap() {
        if (!trainData || trainData.length === 0) {
            showAlert(loadAlert, 'Please load train data first', 'error');
            return;
        }
        
        // Prepare numeric features for correlation
        const numericFeatures = DATASET_SCHEMA.numericFeatures;
        const allFeatures = [...numericFeatures, DATASET_SCHEMA.target];
        
        // Collect valid data
        const dataMatrix = [];
        for (let i = 0; i < trainData.length; i++) {
            const row = trainData[i];
            const values = [];
            let valid = true;
            
            // Check if all required features are valid numbers
            for (const feature of allFeatures) {
                const value = row[feature];
                if (value === null || value === undefined || isNaN(value) || typeof value !== 'number') {
                    valid = false;
                    break;
                }
                values.push(value);
            }
            
            if (valid && values.length === allFeatures.length) {
                dataMatrix.push(values);
            }
        }
        
        if (dataMatrix.length < 10) {
            showAlert(loadAlert, `Insufficient complete data for correlation analysis (need at least 10 complete rows, got ${dataMatrix.length})`, 'warning');
            return;
        }
        
        // Calculate correlation matrix
        const n = allFeatures.length;
        const correlationMatrix = [];
        
        for (let i = 0; i < n; i++) {
            correlationMatrix[i] = [];
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    correlationMatrix[i][j] = 1.0;
                } else {
                    const x = dataMatrix.map(row => row[i]);
                    const y = dataMatrix.map(row => row[j]);
                    correlationMatrix[i][j] = calculatePearsonCorrelation(x, y);
                }
            }
        }
        
        // Store insights
        analysisResults.correlationInsights = {
            matrix: correlationMatrix,
            features: allFeatures,
            sampleSize: dataMatrix.length,
            correlationsWithTarget: {}
        };
        
        // Calculate correlations with target
        const targetIndex = allFeatures.indexOf(DATASET_SCHEMA.target);
        for (let i = 0; i < n; i++) {
            if (i !== targetIndex) {
                const corr = correlationMatrix[targetIndex][i];
                analysisResults.correlationInsights.correlationsWithTarget[allFeatures[i]] = {
                    correlation: corr,
                    strength: getCorrelationStrength(corr),
                    direction: corr > 0 ? 'positive' : 'negative'
                };
            }
        }
        
        // Prepare data for Chart.js matrix
        const dataPoints = [];
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                dataPoints.push({
                    row: i,
                    column: j,
                    value: correlationMatrix[i][j]
                });
            }
        }
        
        // Destroy previous chart
        if (charts.mainChart) {
            charts.mainChart.destroy();
        }
        
        // Create heatmap
        charts.mainChart = new Chart(mainChartCtx, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Correlation Matrix',
                    data: dataPoints.map(point => ({
                        x: point.column,
                        y: point.row,
                        v: point.value
                    })),
                    backgroundColor: (context) => {
                        const value = context.dataset.data[context.dataIndex].v;
                        if (value >= 0.7) return `rgba(46, 204, 113, ${value})`;
                        if (value >= 0.3) return `rgba(52, 152, 219, ${value})`;
                        if (value >= 0) return `rgba(149, 165, 166, ${value})`;
                        if (value >= -0.3) return `rgba(241, 196, 15, ${Math.abs(value)})`;
                        return `rgba(231, 76, 60, ${Math.abs(value)})`;
                    },
                    borderWidth: 1,
                    borderColor: '#fff',
                    width: ({chart}) => (chart.chartArea.width / n) - 1,
                    height: ({chart}) => (chart.chartArea.height / n) - 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Correlation Heatmap (${dataMatrix.length} complete samples)`,
                        font: { size: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const data = context.dataset.data[context.dataIndex];
                                const rowFeature = allFeatures[data.y];
                                const colFeature = allFeatures[data.x];
                                return `${rowFeature} ↔ ${colFeature}: ${data.v.toFixed(3)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            callback: function(value, index) {
                                return allFeatures[index];
                            },
                            maxRotation: 45
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        ticks: {
                            callback: function(value, index) {
                                return allFeatures[index];
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
        
        // Display correlation insights
        displayCorrelationInsights();
    }
    
    function calculatePearsonCorrelation(x, y) {
        const n = x.length;
        if (n < 2) return 0;
        
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        
        for (let i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        if (denominator === 0) return 0;
        
        return numerator / denominator;
    }
    
    function getCorrelationStrength(correlation) {
        const absCorr = Math.abs(correlation);
        if (absCorr >= 0.7) return 'Very Strong';
        if (absCorr >= 0.5) return 'Strong';
        if (absCorr >= 0.3) return 'Moderate';
        if (absCorr >= 0.1) return 'Weak';
        return 'Very Weak';
    }
    
    function displayCorrelationInsights() {
        const insights = analysisResults.correlationInsights;
        if (!insights || !insights.correlationsWithTarget) return;
        
        // Clear previous insights
        const existingInsights = document.querySelector('.correlation-insights');
        if (existingInsights) {
            existingInsights.remove();
        }
        
        // Sort correlations by value (most negative first)
        const correlations = Object.entries(insights.correlationsWithTarget)
            .map(([feature, data]) => ({
                feature,
                correlation: data.correlation,
                strength: data.strength,
                direction: data.direction
            }))
            .sort((a, b) => a.correlation - b.correlation);
        
        // Create insights container
        const insightsContainer = document.createElement('div');
        insightsContainer.className = 'correlation-insights';
        insightsContainer.style.cssText = `
            margin-top: 25px;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            border-left: 5px solid ${COLORS.info};
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        `;
        
        let html = `
            <h3 style="color: ${COLORS.dark}; margin-bottom: 20px; border-bottom: 2px solid ${COLORS.info}; padding-bottom: 10px;">
                🔍 CORRELATION ANALYSIS RESULTS
            </h3>
            
            <div style="margin-bottom: 25px;">
                <h4 style="color: ${COLORS.dark}; margin-bottom: 15px;">📊 Correlation with Survival</h4>
                <div style="display: grid; gap: 10px;">
        `;
        
        // Display all correlations
        correlations.forEach(item => {
            const isNegative = item.correlation < 0;
            const color = isNegative ? COLORS.danger : COLORS.success;
            const icon = isNegative ? '📉' : '📈';
            const impact = isNegative ? 'Decreases survival' : 'Increases survival';
            
            html += `
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding: 12px 15px; background: white; border-radius: 8px; 
                            border-left: 4px solid ${color};">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: ${COLORS.dark}; margin-bottom: 4px;">
                            ${item.feature}
                        </div>
                        <div style="font-size: 0.9rem; color: #666;">
                            ${impact} • ${item.strength} ${item.direction} correlation
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: ${color};">
                            ${icon} ${item.correlation.toFixed(3)}
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += `</div></div>`;
        
        // Find main death factor (most negative correlation)
        const negativeCorrelations = correlations.filter(c => c.correlation < 0);
        if (negativeCorrelations.length > 0) {
            const mainDeathFactor = negativeCorrelations[0];
            
            html += `
                <div style="margin-bottom: 25px; padding: 20px; background: ${COLORS.danger}15; 
                            border-radius: 10px; border: 2px solid ${COLORS.danger};">
                    <h4 style="color: ${COLORS.danger}; margin-bottom: 15px; display: flex; align-items: center;">
                        🎯 CONCLUSION: MAIN REASON FOR DEATH
                    </h4>
                    
                    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 15px;">
                        <div style="flex: 1;">
                            <div style="font-size: 1.3rem; font-weight: bold; color: ${COLORS.danger};">
                                ${mainDeathFactor.feature}
                            </div>
                            <div style="color: #666; margin-top: 5px;">
                                Correlation with death: ${mainDeathFactor.correlation.toFixed(3)}
                            </div>
                        </div>
                        <div style="font-size: 2rem; color: ${COLORS.danger};">
                            📉
                        </div>
                    </div>
                    
                    <div style="color: ${COLORS.dark}; line-height: 1.6;">
                        <strong>Analysis:</strong> ${getDeathFactorInterpretation(mainDeathFactor.feature, mainDeathFactor.correlation)}
                    </div>
                    
                    <div style="margin-top: 15px; padding: 12px; background: white; border-radius: 6px; border-left: 4px solid ${COLORS.warning};">
                        <strong>📈 Statistical Significance:</strong> 
                        This ${mainDeathFactor.strength.toLowerCase()} negative correlation indicates that 
                        ${mainDeathFactor.feature} has a ${Math.abs(mainDeathFactor.correlation) > 0.3 ? 'significant' : 'moderate'} 
                        relationship with mortality rates.
                    </div>
                </div>
            `;
        }
        
        // Display correlation matrix table
        html += `
            <div style="margin-bottom: 20px;">
                <h4 style="color: ${COLORS.dark}; margin-bottom: 15px;">📋 Correlation Matrix</h4>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; background: white;">
                        <thead>
                            <tr>
                                <th style="padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; text-align: center;">Feature</th>
        `;
        
        // Header row
        insights.features.forEach(feature => {
            html += `<th style="padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; text-align: center; min-width: 80px;">${feature}</th>`;
        });
        
        html += `</tr></thead><tbody>`;
        
        // Data rows
        for (let i = 0; i < insights.features.length; i++) {
            html += `<tr><td style="padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: 600;">${insights.features[i]}</td>`;
            
            for (let j = 0; j < insights.features.length; j++) {
                const corr = insights.matrix[i][j];
                const bgColor = corr >= 0.7 ? '#d4edda' : 
                              corr >= 0.3 ? '#d1ecf1' : 
                              corr >= 0 ? '#f8f9fa' : 
                              corr >= -0.3 ? '#fff3cd' : '#f8d7da';
                const textColor = corr >= 0.7 ? '#155724' : 
                                 corr >= 0.3 ? '#0c5460' : 
                                 corr >= 0 ? '#6c757d' : 
                                 corr >= -0.3 ? '#856404' : '#721c24';
                
                html += `<td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; background: ${bgColor}; color: ${textColor}; font-weight: ${Math.abs(corr) > 0.5 ? 'bold' : 'normal'};">${corr.toFixed(3)}</td>`;
            }
            
            html += `</tr>`;
        }
        
        html += `</tbody></table></div></div>`;
        
        // Legend
        html += `
            <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px; border: 1px solid #dee2e6;">
                <h5 style="color: ${COLORS.dark}; margin-bottom: 10px;">📖 Correlation Legend</h5>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: #d4edda; border: 1px solid #c3e6cb;"></div>
                        <span>Strong Positive (≥ 0.7)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: #d1ecf1; border: 1px solid #bee5eb;"></div>
                        <span>Moderate Positive (≥ 0.3)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: #fff3cd; border: 1px solid #ffeaa7;"></div>
                        <span>Weak Negative (≥ -0.3)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background: #f8d7da; border: 1px solid #f5c6cb;"></div>
                        <span>Strong Negative (< -0.3)</span>
                    </div>
                </div>
            </div>
        `;
        
        insightsContainer.innerHTML = html;
        
        // Insert after the chart container
        const chartContainer = document.querySelector('.chart-container');
        if (chartContainer && chartContainer.parentNode) {
            chartContainer.parentNode.insertBefore(insightsContainer, chartContainer.nextSibling);
        } else {
            const visualizationSection = document.querySelector('.section:nth-child(5)');
            if (visualizationSection) {
                visualizationSection.appendChild(insightsContainer);
            }
        }
    }
    
    function getDeathFactorInterpretation(feature, correlation) {
        const interpretations = {
            'Age': `Age has a ${correlation < 0 ? 'negative' : 'positive'} correlation with survival. 
                   ${correlation < 0 ? 'Younger passengers had better survival chances, possibly due to evacuation priorities and physical ability.' : 
                    'Older passengers had better survival rates, which contradicts typical disaster scenarios.'}`,
            
            'Fare': `Fare paid shows ${correlation < 0 ? 'negative' : 'positive'} correlation. 
                    ${correlation > 0 ? 'Passengers who paid higher fares (typically in 1st class) had significantly better survival rates, indicating socioeconomic status was a crucial factor.' : 
                     'Lower fare passengers had better survival, which is unusual and requires further investigation.'}`,
            
            'SibSp': `Number of siblings/spouses shows ${correlation < 0 ? 'negative' : 'positive'} correlation. 
                     ${correlation < 0 ? 'Passengers with more siblings/spouses had lower survival, possibly due to complex evacuation decisions and family priorities.' : 
                      'Having more siblings/spouses increased survival chances, possibly through mutual assistance.'}`,
            
            'Parch': `Number of parents/children shows ${correlation < 0 ? 'negative' : 'positive'} correlation. 
                     ${correlation < 0 ? 'Passengers with children or parents aboard had lower survival, likely due to prioritizing family members over self-preservation.' : 
                      'Passengers with family had better survival, possibly through coordinated evacuation efforts.'}`
        };
        
        return interpretations[feature] || 
               `The ${feature} feature shows a ${Math.abs(correlation).toFixed(3)} correlation with survival, 
               indicating it is ${Math.abs(correlation) > 0.3 ? 'a significant' : 'a moderate'} factor in determining survival outcomes.`;
    }
    
    // ==================== EXPORT FUNCTIONS ====================
    function exportResults() {
        if (!mergedData || mergedData.length === 0) {
            showAlert(exportAlert, 'No data to export', 'error');
            return;
        }
        
        exportBtn.disabled = true;
        showAlert(exportAlert, 'Exporting data...', 'info');
        
        try {
            const csvContent = Papa.unparse(mergedData);
            downloadFile(csvContent, 'titanic_merged_dataset.csv', 'text/csv;charset=utf-8;');
            
            const exportData = {
                metadata: {
                    application: 'Titanic EDA Explorer',
                    version: '1.0.0',
                    exportDate: new Date().toISOString(),
                    dataset: 'Kaggle Titanic Dataset'
                },
                datasetInfo: analysisResults.overview,
                analysis: analysisResults,
                mainConclusion: analysisResults.correlationInsights ? 
                    getExportConclusion(analysisResults.correlationInsights) :
                    'No correlation analysis performed'
            };
            
            const jsonContent = JSON.stringify(exportData, null, 2);
            downloadFile(jsonContent, 'titanic_eda_analysis.json', 'application/json');
            
            const summaryContent = generateSummaryReport();
            downloadFile(summaryContent, 'titanic_summary_report.txt', 'text/plain');
            
            showAlert(exportAlert, '✅ Export completed! 3 files downloaded.', 'success');
            
        } catch (error) {
            console.error('Export error:', error);
            showAlert(exportAlert, `Export failed: ${error.message}`, 'error');
        } finally {
            exportBtn.disabled = false;
            
            setTimeout(() => {
                exportAlert.style.display = 'none';
            }, 5000);
        }
    }
    
    function getExportConclusion(insights) {
        if (!insights.correlationsWithTarget) return 'No correlation analysis available';
        
        const correlations = Object.entries(insights.correlationsWithTarget)
            .map(([feature, data]) => ({ feature, correlation: data.correlation }))
            .sort((a, b) => a.correlation - b.correlation);
        
        if (correlations.length === 0) return 'No correlations calculated';
        
        const mainDeathFactor = correlations.find(c => c.correlation < 0);
        if (mainDeathFactor) {
            return `MAIN DEATH FACTOR: ${mainDeathFactor.feature} (correlation: ${mainDeathFactor.correlation.toFixed(3)})`;
        }
        
        return 'No negative correlations (death factors) identified';
    }
    
    function generateSummaryReport() {
        let report = '='.repeat(60) + '\n';
        report += 'TITANIC DATASET EDA SUMMARY REPORT\n';
        report += '='.repeat(60) + '\n\n';
        
        report += `Generated: ${new Date().toLocaleString()}\n`;
        report += `Application: Titanic EDA Explorer\n\n`;
        
        report += '📊 DATASET OVERVIEW\n';
        report += '-'.repeat(40) + '\n';
        report += `Total Rows: ${analysisResults.overview.totalRows || 0}\n`;
        report += `Train Rows: ${analysisResults.overview.trainRows || 0}\n`;
        report += `Test Rows: ${analysisResults.overview.testRows || 0}\n`;
        report += `Total Columns: ${analysisResults.overview.columns || 0}\n`;
        report += `Features Analyzed: ${analysisResults.overview.features?.join(', ') || 'None'}\n`;
        report += `Target Variable: ${DATASET_SCHEMA.target}\n\n`;
        
        if (analysisResults.correlationInsights && analysisResults.correlationInsights.correlationsWithTarget) {
            report += '🔍 CORRELATION ANALYSIS\n';
            report += '-'.repeat(40) + '\n';
            report += `Sample Size: ${analysisResults.correlationInsights.sampleSize || 0}\n\n`;
            
            const correlations = Object.entries(analysisResults.correlationInsights.correlationsWithTarget)
                .map(([feature, data]) => ({ feature, correlation: data.correlation, strength: data.strength }))
                .sort((a, b) => a.correlation - b.correlation);
            
            if (correlations.length > 0) {
                report += 'CORRELATIONS WITH SURVIVAL:\n';
                correlations.forEach(item => {
                    const impact = item.correlation > 0 ? 'Increases survival' : 'Decreases survival';
                    report += `  ${item.feature}: ${item.correlation.toFixed(3)} (${item.strength}, ${impact})\n`;
                });
                
                const negativeCorrelations = correlations.filter(c => c.correlation < 0);
                if (negativeCorrelations.length > 0) {
                    const mainFactor = negativeCorrelations[0];
                    report += `\n🎯 MAIN DEATH FACTOR: ${mainFactor.feature} (correlation: ${mainFactor.correlation.toFixed(3)})\n`;
                }
            }
            report += '\n';
        }
        
        report += '📈 KEY STATISTICS\n';
        report += '-'.repeat(40) + '\n';
        if (analysisResults.statistics?.numeric) {
            Object.entries(analysisResults.statistics.numeric).forEach(([feature, stats]) => {
                report += `\n${feature}:\n`;
                report += `  Count: ${stats.count}, Missing: ${stats.missing}\n`;
                report += `  Mean: ${stats.mean}, Median: ${stats.median}\n`;
                report += `  Std Dev: ${stats.stdDev}\n`;
                report += `  Range: ${stats.min} - ${stats.max}\n`;
            });
        }
        
        report += '\n' + '='.repeat(60) + '\n';
        report += 'END OF REPORT\n';
        report += '='.repeat(60);
        
        return report;
    }
    
    function downloadFile(content, filename, mimeType) {
        try {
            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.style.display = 'none';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            setTimeout(() => URL.revokeObjectURL(url), 100);
        } catch (error) {
            console.error('Download error:', error);
            throw error;
        }
    }
    
    // ==================== UTILITY FUNCTIONS ====================
    function showAlert(alertElement, message, type) {
        alertElement.textContent = message;
        alertElement.className = `alert alert-${type}`;
        alertElement.style.display = 'block';
    }
    
    function destroyAllCharts() {
        Object.values(charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        charts = {};
    }
    
    function resetApplication() {
        mergedData = null;
        trainData = null;
        testData = null;
        analysisResults = {
            overview: {},
            missingValues: {},
            statistics: {},
            visualizations: {},
            correlationInsights: {}
        };
        
        destroyAllCharts();
        
        overviewStats.innerHTML = '';
        previewTable.innerHTML = '';
        missingValues.innerHTML = '';
        numericStats.innerHTML = '';
        categoricalStats.innerHTML = '';
        
        loadAlert.style.display = 'none';
        exportAlert.style.display = 'none';
        
        // Clear correlation insights
        const existingInsights = document.querySelector('.correlation-insights');
        if (existingInsights) {
            existingInsights.remove();
        }
    }
    
    // ==================== INITIALIZATION ====================
    console.log('🚀 Titanic EDA Explorer initialized');
    console.log('📝 To analyze different datasets:');
    console.log('1. Update DATASET_SCHEMA configuration');
    console.log('2. Adjust feature lists for your dataset');
    console.log('3. Update target variable if different');
    
    showAlert(loadAlert, '📁 Please upload train.csv and test.csv files from Kaggle Titanic dataset', 'info');
});

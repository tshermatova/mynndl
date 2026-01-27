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
            }
        });
        
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
                    percentage: parseFloat((count / totalCount * 100).toFixed(2))
                }));
            
            categoricalStatsResult[feature] = {
                uniqueValues: Object.keys(valueCounts).length,
                missing: missingCount,
                missingPercentage: parseFloat((missingCount / totalCount * 100).toFixed(2)),
                valueCounts: sortedCounts,
                mostCommon: sortedCounts.length > 0 ? sortedCounts[0] : null
            };
        });
        
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
                    percentage: parseFloat((subset.length / trainData.length * 100).toFixed(2)),
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
                
                stats.valueCounts.slice(0, 6).forEach(item => {
                    categoricalHTML += `
                        <div class="stat-card" style="border-left-color: ${COLORS.info}">
                            <div class="stat-title">${item.value}</div>
                            <div class="stat-value">${item.count}</div>
                            <div>${item.percentage}%</div>
                        </div>
                    `;
                });
                
                if (stats.valueCounts.length > 6) {
                    categoricalHTML += `
                        <div class="stat-card" style="border-left-color: ${COLORS.light}">
                            <div class="stat-title">+${stats.valueCounts.length - 6} more</div>
                        </div>
                    `;
                }
                
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
            
            if (Object.keys(groupedStats).length > 0) {
                categoricalHTML += '<h3>🎯 Survival Statistics</h3>';
                
                const totalTrain = trainData.length;
                const survived = trainData.filter(row => row[DATASET_SCHEMA.target] === 1).length;
                const died = trainData.filter(row => row[DATASET_SCHEMA.target] === 0).length;
                const survivalRate = ((survived / totalTrain) * 100).toFixed(2);
                const deathRate = ((died / totalTrain) * 100).toFixed(2);
                
                categoricalHTML += `
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
        
        // Prepare data: Convert categorical features to numeric
        const features = DATASET_SCHEMA.numericFeatures.slice(); // Start with numeric features
        
        // Add encoded categorical features
        const encodedData = [];
        const validIndices = [];
        
        // First, prepare all data including encoding categorical features
        for (let i = 0; i < trainData.length; i++) {
            const row = trainData[i];
            const dataPoint = [];
            let valid = true;
            
            // Add numeric features
            DATASET_SCHEMA.numericFeatures.forEach(feature => {
                const value = row[feature];
                if (value === null || value === undefined || isNaN(value)) {
                    valid = false;
                } else {
                    dataPoint.push(value);
                }
            });
            
            // Add Fare (important for correlation)
            if (row.Fare !== null && !isNaN(row.Fare)) {
                dataPoint.push(row.Fare);
                if (!features.includes('Fare')) features.push('Fare');
            } else {
                valid = false;
            }
            
            // Add Age (important for correlation)
            if (row.Age !== null && !isNaN(row.Age)) {
                dataPoint.push(row.Age);
                if (!features.includes('Age')) features.push('Age');
            } else {
                valid = false;
            }
            
            // Add target variable
            if (row[DATASET_SCHEMA.target] !== null && row[DATASET_SCHEMA.target] !== undefined) {
                dataPoint.push(row[DATASET_SCHEMA.target]);
            } else {
                valid = false;
            }
            
            if (valid && dataPoint.length === features.length + 1) { // +1 for target
                encodedData.push(dataPoint);
                validIndices.push(i);
            }
        }
        
        // Add target to features list for labeling
        const allFeatures = [...features, DATASET_SCHEMA.target];
        
        if (validIndices.length < 10) {
            showAlert(loadAlert, `Insufficient complete data for correlation analysis (only ${validIndices.length} complete rows)`, 'warning');
            return;
        }
        
        // Calculate correlation matrix
        const n = allFeatures.length;
        const correlationMatrix = [];
        
        for (let i = 0; i < n; i++) {
            correlationMatrix[i] = [];
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    correlationMatrix[i][j] = 1;
                } else {
                    const x = encodedData.map(row => row[i]);
                    const y = encodedData.map(row => row[j]);
                    correlationMatrix[i][j] = calculatePearsonCorrelation(x, y);
                }
            }
        }
        
        // Store for analysis
        analysisResults.correlationInsights = {
            matrix: correlationMatrix,
            features: allFeatures,
            sampleSize: validIndices.length,
            correlationsWithTarget: {}
        };
        
        // Calculate correlations with target
        const targetIndex = allFeatures.indexOf(DATASET_SCHEMA.target);
        for (let i = 0; i < n; i++) {
            if (i !== targetIndex) {
                analysisResults.correlationInsights.correlationsWithTarget[allFeatures[i]] = {
                    correlation: correlationMatrix[targetIndex][i],
                    strength: getCorrelationStrength(correlationMatrix[targetIndex][i]),
                    direction: correlationMatrix[targetIndex][i] > 0 ? 'positive' : 'negative'
                };
            }
        }
        
        // Prepare data for heatmap
        const dataPoints = [];
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                dataPoints.push({
                    x: j,
                    y: i,
                    v: correlationMatrix[i][j]
                });
            }
        }
        
        // Clear previous chart
        if (charts.mainChart) {
            charts.mainChart.destroy();
        }
        
        // Create heatmap
        charts.mainChart = new Chart(mainChartCtx, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Correlation',
                    data: dataPoints,
                    backgroundColor: (ctx) => {
                        const value = ctx.dataset.data[ctx.dataIndex].v;
                        const absValue = Math.abs(value);
                        
                        // Color coding based on correlation strength
                        if (value > 0) {
                            // Positive correlation - green shades
                            const intensity = Math.min(absValue * 1.5, 1);
                            return `rgba(46, 204, 113, ${intensity})`;
                        } else if (value < 0) {
                            // Negative correlation - red shades
                            const intensity = Math.min(absValue * 1.5, 1);
                            return `rgba(231, 76, 60, ${intensity})`;
                        } else {
                            // Zero correlation
                            return 'rgba(149, 165, 166, 0.3)';
                        }
                    },
                    borderColor: 'white',
                    borderWidth: 1,
                    width: ({chart}) => {
                        const chartArea = chart.chartArea;
                        return chartArea ? (chartArea.width / n) - 1 : 20;
                    },
                    height: ({chart}) => {
                        const chartArea = chart.chartArea;
                        return chartArea ? (chartArea.height / n) - 1 : 20;
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Correlation Heatmap (Numeric Features vs Survival)',
                        font: { size: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const data = context.dataset.data[context.dataIndex];
                                const rowFeature = allFeatures[data.y];
                                const colFeature = allFeatures[data.x];
                                const correlation = data.v;
                                
                                let strength = getCorrelationStrength(correlation);
                                let direction = correlation > 0 ? 'positive' : correlation < 0 ? 'negative' : 'none';
                                let interpretation = '';
                                
                                if (rowFeature === DATASET_SCHEMA.target || colFeature === DATASET_SCHEMA.target) {
                                    const feature = rowFeature === DATASET_SCHEMA.target ? colFeature : rowFeature;
                                    const impact = correlation > 0 ? 'increases survival' : correlation < 0 ? 'decreases survival' : 'no impact';
                                    interpretation = `${feature} ${impact} (${direction} correlation)`;
                                } else {
                                    interpretation = `${direction} relationship`;
                                }
                                
                                return `${rowFeature} ↔ ${colFeature}: ${correlation.toFixed(3)} (${strength}, ${interpretation})`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            display: true,
                            callback: function(value, index) {
                                return allFeatures[index] || '';
                            },
                            font: {
                                size: 11
                            }
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        ticks: {
                            display: true,
                            callback: function(value, index) {
                                return allFeatures[index] || '';
                            },
                            font: {
                                size: 11
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
        displayCorrelationInsights(analysisResults.correlationInsights);
    }
    
    function calculatePearsonCorrelation(x, y) {
        const n = x.length;
        if (n < 2) return 0;
        
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        
        for (let i = 0; i < n; i++) {
            const xi = x[i];
            const yi = y[i];
            sumX += xi;
            sumY += yi;
            sumXY += xi * yi;
            sumX2 += xi * xi;
            sumY2 += yi * yi;
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
        return 'Very Weak/None';
    }
    
    function displayCorrelationInsights(insights) {
        // Clear previous insights
        const existingInsights = document.querySelector('.correlation-insights-container');
        if (existingInsights) {
            existingInsights.remove();
        }
        
        // Create insights container
        const insightsContainer = document.createElement('div');
        insightsContainer.className = 'correlation-insights-container';
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
                🔍 CORRELATION ANALYSIS - DEATH FACTOR INSIGHTS
            </h3>
            
            <div style="margin-bottom: 20px;">
                <h4 style="color: ${COLORS.dark}; margin-bottom: 10px;">📊 Correlation with Survival (Target Variable)</h4>
                <div style="display: grid; gap: 12px;">
        `;
        
        // Sort features by absolute correlation with target
        const correlations = [];
        for (const [feature, data] of Object.entries(insights.correlationsWithTarget)) {
            correlations.push({
                feature,
                correlation: data.correlation,
                strength: data.strength,
                direction: data.direction
            });
        }
        
        correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
        
        // Display top correlations
        correlations.forEach(item => {
            const isPositive = item.correlation > 0;
            const color = isPositive ? COLORS.success : COLORS.danger;
            const icon = isPositive ? '📈' : '📉';
            const impact = isPositive ? 'INCREASES survival chances' : 'DECREASES survival chances';
            
            html += `
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding: 10px 15px; background: white; border-radius: 8px; 
                            border-left: 4px solid ${color};">
                    <div>
                        <div style="font-weight: 600; color: ${COLORS.dark};">${item.feature}</div>
                        <div style="font-size: 0.9rem; color: #666; margin-top: 2px;">
                            ${impact}
                        </div>
                    </div>
                    <div>
                        <span style="font-weight: bold; font-size: 1.1rem; color: ${color};">
                            ${icon} ${item.correlation.toFixed(3)}
                        </span>
                        <div style="font-size: 0.8rem; color: #7f8c8d; text-align: right;">
                            ${item.strength} (${item.direction})
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += `</div></div>`;
        
        // Identify main death factors
        const negativeCorrelations = correlations.filter(c => c.correlation < 0);
        const mainDeathFactors = negativeCorrelations.slice(0, 3);
        
        if (mainDeathFactors.length > 0) {
            html += `
                <div style="margin-bottom: 20px;">
                    <h4 style="color: ${COLORS.danger}; margin-bottom: 15px;">🚨 MAIN FACTORS CONTRIBUTING TO DEATH</h4>
                    <div style="display: grid; gap: 15px;">
            `;
            
            mainDeathFactors.forEach((factor, index) => {
                const strength = factor.strength.toLowerCase();
                const riskLevel = Math.abs(factor.correlation) > 0.3 ? '🚨 HIGH RISK' : 
                                Math.abs(factor.correlation) > 0.2 ? '⚠️ MODERATE RISK' : '📊 LOW RISK';
                
                html += `
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 2px solid ${COLORS.danger};">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                            <div>
                                <span style="font-weight: bold; color: ${COLORS.danger}; font-size: 1.1rem;">
                                    ${index + 1}. ${factor.feature}
                                </span>
                                <div style="color: #666; font-size: 0.9rem; margin-top: 3px;">
                                    Correlation: ${factor.correlation.toFixed(3)} (${strength} negative)
                                </div>
                            </div>
                            <div style="background: #ffeaea; padding: 4px 10px; border-radius: 4px; font-weight: 600; color: ${COLORS.danger};">
                                ${riskLevel}
                            </div>
                        </div>
                        <div style="color: #444; font-size: 0.95rem;">
                            ${getFactorInterpretation(factor.feature, factor.correlation)}
                        </div>
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }
        
        // Positive correlations (factors that increase survival)
        const positiveCorrelations = correlations.filter(c => c.correlation > 0);
        const mainSurvivalFactors = positiveCorrelations.slice(0, 3);
        
        if (mainSurvivalFactors.length > 0) {
            html += `
                <div style="margin-bottom: 20px;">
                    <h4 style="color: ${COLORS.success}; margin-bottom: 15px;">✅ FACTORS INCREASING SURVIVAL</h4>
                    <div style="display: grid; gap: 15px;">
            `;
            
            mainSurvivalFactors.forEach((factor, index) => {
                html += `
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 2px solid ${COLORS.success};">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div>
                                <span style="font-weight: bold; color: ${COLORS.success};">
                                    ${factor.feature}
                                </span>
                                <div style="color: #666; font-size: 0.9rem; margin-top: 3px;">
                                    Correlation: +${factor.correlation.toFixed(3)} (${factor.strength.toLowerCase()} positive)
                                </div>
                            </div>
                        </div>
                        <div style="color: #444; font-size: 0.95rem; margin-top: 8px;">
                            ${getFactorInterpretation(factor.feature, factor.correlation)}
                        </div>
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }
        
        // Main conclusion
        if (mainDeathFactors.length > 0) {
            const mainFactor = mainDeathFactors[0];
            const correlationValue = Math.abs(mainFactor.correlation);
            
            html += `
                <div style="margin-top: 25px; padding: 20px; background: ${COLORS.danger}15; border-radius: 10px; border: 2px solid ${COLORS.danger};">
                    <h4 style="color: ${COLORS.danger}; margin-bottom: 10px; display: flex; align-items: center;">
                        🎯 PRIMARY CONCLUSION: MAIN DEATH FACTOR IDENTIFIED
                    </h4>
                    <div style="color: ${COLORS.dark}; font-size: 1.1rem; font-weight: 500; margin-bottom: 10px;">
                        Based on correlation analysis, the strongest predictor of death is:
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <div style="font-size: 1.3rem; font-weight: bold; color: ${COLORS.danger}; text-align: center; margin-bottom: 8px;">
                            "${mainFactor.feature}"
                        </div>
                        <div style="text-align: center; color: #666;">
                            Correlation with death: ${mainFactor.correlation.toFixed(3)} 
                            <span style="color: ${COLORS.danger};">(Negative ${getCorrelationStrength(correlationValue).toLowerCase()})</span>
                        </div>
                    </div>
                    <div style="color: #444; line-height: 1.5;">
                        <strong>Interpretation:</strong> ${getMainFactorInterpretation(mainFactor.feature, mainFactor.correlation)}
                    </div>
                    <div style="margin-top: 15px; padding: 12px; background: white; border-radius: 6px; border-left: 4px solid ${COLORS.warning};">
                        <strong>📈 Statistical Significance:</strong> 
                        ${correlationValue > 0.3 ? 'This correlation is statistically significant and indicates a strong relationship.' :
                          correlationValue > 0.2 ? 'This correlation is moderately significant.' :
                          'While this is the strongest negative correlation found, it represents a weak relationship.'}
                    </div>
                </div>
            `;
        } else {
            html += `
                <div style="margin-top: 25px; padding: 20px; background: ${COLORS.warning}15; border-radius: 10px; border: 2px solid ${COLORS.warning};">
                    <h4 style="color: ${COLORS.warning}; margin-bottom: 10px;">⚠️ NO STRONG CORRELATIONS DETECTED</h4>
                    <div style="color: #444;">
                        The correlation analysis did not reveal any strong linear relationships between 
                        numeric features and survival. This suggests that death factors may be more complex 
                        or involve categorical variables (like Sex, Pclass) that require different analysis methods.
                    </div>
                </div>
            `;
        }
        
        // Data quality note
        html += `
            <div style="margin-top: 15px; padding: 10px; background: #f0f7ff; border-radius: 6px; font-size: 0.9rem; color: #666;">
                <strong>Note:</strong> Analysis based on ${insights.sampleSize} complete data samples. 
                Correlation measures linear relationships only.
            </div>
        `;
        
        insightsContainer.innerHTML = html;
        
        // Insert after the chart container
        const chartContainer = document.querySelector('.chart-container');
        if (chartContainer && chartContainer.parentNode) {
            chartContainer.parentNode.insertBefore(insightsContainer, chartContainer.nextSibling);
        } else {
            document.querySelector('.section:nth-child(5)').appendChild(insightsContainer);
        }
    }
    
    function getFactorInterpretation(feature, correlation) {
        const interpretations = {
            'Age': correlation < 0 ? 'Younger passengers had better survival chances' : 'Older passengers had better survival chances',
            'Fare': correlation < 0 ? 'Lower fare passengers had lower survival' : 'Higher fare passengers had better survival',
            'SibSp': correlation < 0 ? 'More siblings/spouses decreased survival' : 'More siblings/spouses increased survival',
            'Parch': correlation < 0 ? 'More parents/children decreased survival' : 'More parents/children increased survival',
            'Pclass': correlation < 0 ? 'Lower class (higher number) decreased survival' : 'Higher class increased survival'
        };
        
        return interpretations[feature] || 
               `A ${correlation > 0 ? 'positive' : 'negative'} correlation indicates this feature ${correlation > 0 ? 'increases' : 'decreases'} with survival.`;
    }
    
    function getMainFactorInterpretation(feature, correlation) {
        const mainInterpretations = {
            'Age': `Age shows a ${correlation < 0 ? 'negative' : 'positive'} correlation with survival. 
                   ${correlation < 0 ? 'Younger passengers were more likely to survive, possibly due to evacuation priorities.' : 
                    'Older passengers showed higher survival rates.'}`,
            
            'Fare': `Fare paid shows a ${correlation < 0 ? 'negative' : 'positive'} correlation. 
                    ${correlation > 0 ? 'Passengers who paid higher fares (typically in better classes) had significantly better survival rates, indicating socioeconomic status was a key factor.' : 
                     'Lower fare passengers had better survival, which contradicts historical records.'}`,
            
            'SibSp': `Number of siblings/spouses shows ${correlation < 0 ? 'negative' : 'positive'} correlation. 
                     ${correlation < 0 ? 'Having more siblings/spouses aboard decreased survival chances, possibly due to family evacuation challenges.' : 
                      'Having more siblings/spouses increased survival chances, possibly through mutual assistance.'}`,
            
            'Parch': `Number of parents/children shows ${correlation < 0 ? 'negative' : 'positive'} correlation. 
                     ${correlation < 0 ? 'Passengers with more dependents had lower survival, possibly due to prioritizing others.' : 
                      'Passengers with family members had better survival, possibly through coordinated evacuation.'}`
        };
        
        return mainInterpretations[feature] || 
               `The ${feature} feature shows the strongest negative correlation (${correlation.toFixed(3)}) with survival, 
               making it the primary numeric predictor of death in this analysis.`;
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
                mainConclusion: analysisResults.correlationInsights && 
                               analysisResults.correlationInsights.correlationsWithTarget ? 
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
        if (!insights.correlationsWithTarget || Object.keys(insights.correlationsWithTarget).length === 0) {
            return 'No significant correlations found';
        }
        
        const correlations = [];
        for (const [feature, data] of Object.entries(insights.correlationsWithTarget)) {
            correlations.push({
                feature,
                correlation: data.correlation,
                direction: data.direction
            });
        }
        
        correlations.sort((a, b) => a.correlation - b.correlation); // Sort by most negative first
        
        if (correlations.length > 0 && correlations[0].correlation < 0) {
            const mainFactor = correlations[0];
            return `MAIN DEATH FACTOR: ${mainFactor.feature} (correlation: ${mainFactor.correlation.toFixed(3)})`;
        }
        
        return 'No strong negative correlations (death factors) identified';
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
            report += '🔍 CORRELATION ANALYSIS - DEATH FACTORS\n';
            report += '-'.repeat(40) + '\n';
            
            const correlations = [];
            for (const [feature, data] of Object.entries(analysisResults.correlationInsights.correlationsWithTarget)) {
                correlations.push({
                    feature,
                    correlation: data.correlation,
                    strength: data.strength,
                    direction: data.direction
                });
            }
            
            correlations.sort((a, b) => a.correlation - b.correlation); // Most negative first
            
            if (correlations.length > 0) {
                report += 'CORRELATIONS WITH SURVIVAL:\n';
                correlations.forEach(item => {
                    const impact = item.correlation > 0 ? 'Increases survival' : 'Decreases survival';
                    report += `  ${item.feature}: ${item.correlation.toFixed(3)} (${item.strength}, ${impact})\n`;
                });
                
                report += '\n';
                
                // Identify main death factors (negative correlations)
                const deathFactors = correlations.filter(c => c.correlation < 0);
                if (deathFactors.length > 0) {
                    report += 'MAIN DEATH FACTORS (Negative Correlations):\n';
                    deathFactors.slice(0, 3).forEach((factor, idx) => {
                        report += `${idx + 1}. ${factor.feature}: ${factor.correlation.toFixed(3)} (${factor.strength})\n`;
                    });
                    
                    const mainFactor = deathFactors[0];
                    report += `\n🎯 PRIMARY CONCLUSION:\n`;
                    report += `The strongest predictor of death is "${mainFactor.feature}" with a correlation of ${mainFactor.correlation.toFixed(3)}.\n`;
                    report += `This indicates a ${mainFactor.strength.toLowerCase()} negative relationship with survival.\n`;
                }
                
                // Identify survival factors (positive correlations)
                const survivalFactors = correlations.filter(c => c.correlation > 0);
                if (survivalFactors.length > 0) {
                    report += '\nMAIN SURVIVAL FACTORS (Positive Correlations):\n';
                    survivalFactors.slice(0, 3).forEach((factor, idx) => {
                        report += `${idx + 1}. ${factor.feature}: +${factor.correlation.toFixed(3)} (${factor.strength})\n`;
                    });
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
        const existingInsights = document.querySelector('.correlation-insights-container');
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

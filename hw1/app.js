// app.js - Titanic EDA Explorer
// To reuse this app for other datasets:
// 1. Update DATASET_SCHEMA configuration (lines 30-45)
// 2. Update feature lists for numeric/categorical
// 3. Update target variable if different from 'Survived'
// 4. Update death factor analysis logic for your specific dataset

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
        deathFactors: {}
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
            analyzeDeathFactors();
            
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
                categoricalHTML += '<h3>🎯 Grouped by Survival Status</h3>';
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
    
    function analyzeDeathFactors() {
        if (!trainData || trainData.length === 0) return;
        
        const deathFactors = {
            survivalRate: 0,
            deathRate: 0,
            factors: {}
        };
        
        const totalPassengers = trainData.length;
        const survived = trainData.filter(row => row[DATASET_SCHEMA.target] === 1).length;
        const died = trainData.filter(row => row[DATASET_SCHEMA.target] === 0).length;
        
        deathFactors.survivalRate = parseFloat((survived / totalPassengers * 100).toFixed(2));
        deathFactors.deathRate = parseFloat((died / totalPassengers * 100).toFixed(2));
        
        DATASET_SCHEMA.categoricalFeatures.forEach(feature => {
            const values = [...new Set(trainData.map(row => row[feature]))];
            const factorAnalysis = {};
            
            values.forEach(value => {
                const subset = trainData.filter(row => row[feature] === value);
                const total = subset.length;
                const diedInGroup = subset.filter(row => row[DATASET_SCHEMA.target] === 0).length;
                const deathRate = total > 0 ? parseFloat((diedInGroup / total * 100).toFixed(2)) : 0;
                
                factorAnalysis[value] = {
                    total,
                    died: diedInGroup,
                    deathRate,
                    survivalRate: parseFloat(((total - diedInGroup) / total * 100).toFixed(2))
                };
            });
            
            deathFactors.factors[feature] = factorAnalysis;
        });
        
        const numericAnalysis = {};
        DATASET_SCHEMA.numericFeatures.forEach(feature => {
            const diedValues = trainData
                .filter(row => row[DATASET_SCHEMA.target] === 0)
                .map(row => row[feature])
                .filter(val => val !== null && !isNaN(val));
            
            const survivedValues = trainData
                .filter(row => row[DATASET_SCHEMA.target] === 1)
                .map(row => row[feature])
                .filter(val => val !== null && !isNaN(val));
            
            if (diedValues.length > 0 && survivedValues.length > 0) {
                const diedMean = diedValues.reduce((a, b) => a + b, 0) / diedValues.length;
                const survivedMean = survivedValues.reduce((a, b) => a + b, 0) / survivedValues.length;
                const difference = parseFloat((diedMean - survivedMean).toFixed(2));
                
                numericAnalysis[feature] = {
                    diedMean: parseFloat(diedMean.toFixed(2)),
                    survivedMean: parseFloat(survivedMean.toFixed(2)),
                    difference,
                    higherInDeaths: difference > 0 ? feature : null,
                    impact: Math.abs(difference) > (Math.abs(diedMean) * 0.1) ? 'High' : 'Low'
                };
            }
        });
        
        deathFactors.numericFactors = numericAnalysis;
        
        const mainFactors = [];
        
        Object.entries(deathFactors.factors).forEach(([feature, values]) => {
            Object.entries(values).forEach(([value, stats]) => {
                if (stats.deathRate > deathFactors.deathRate * 1.5) {
                    mainFactors.push({
                        feature,
                        value,
                        deathRate: stats.deathRate,
                        impact: 'High'
                    });
                }
            });
        });
        
        Object.entries(deathFactors.numericFactors).forEach(([feature, stats]) => {
            if (stats.impact === 'High' && Math.abs(stats.difference) > 0) {
                mainFactors.push({
                    feature,
                    value: `Higher in deaths by ${Math.abs(stats.difference).toFixed(2)}`,
                    deathRate: null,
                    impact: 'High'
                });
            }
        });
        
        deathFactors.mainFactors = mainFactors.sort((a, b) => {
            const rateA = a.deathRate || 0;
            const rateB = b.deathRate || 0;
            return rateB - rateA;
        });
        
        analysisResults.deathFactors = deathFactors;
        
        displayDeathFactorsAnalysis(deathFactors);
    }
    
    function displayDeathFactorsAnalysis(deathFactors) {
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'death-factors-summary';
        summaryDiv.style.cssText = `
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            border-left: 5px solid ${COLORS.danger};
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        `;
        
        let html = `
            <h3 style="color: ${COLORS.danger}; margin-bottom: 20px; border-bottom: 2px solid ${COLORS.danger}; padding-bottom: 10px;">
                🎯 MAIN FACTORS FOR DEATH - ANALYSIS RESULTS
            </h3>
            
            <div style="margin-bottom: 25px;">
                <h4 style="color: ${COLORS.dark}; margin-bottom: 10px;">📈 Overall Survival Statistics</h4>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px;">
                        <div style="font-size: 2rem; font-weight: bold; color: ${COLORS.danger};">
                            ${deathFactors.deathRate}%
                        </div>
                        <div style="color: #666;">Death Rate</div>
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <div style="font-size: 2rem; font-weight: bold; color: ${COLORS.success};">
                            ${deathFactors.survivalRate}%
                        </div>
                        <div style="color: #666;">Survival Rate</div>
                    </div>
                </div>
            </div>
        `;
        
        if (deathFactors.mainFactors.length > 0) {
            html += `
                <div style="margin-bottom: 25px;">
                    <h4 style="color: ${COLORS.dark}; margin-bottom: 15px;">🚨 Key Risk Factors Identified</h4>
                    <div style="display: grid; gap: 15px;">
            `;
            
            deathFactors.mainFactors.slice(0, 5).forEach((factor, index) => {
                const severity = factor.deathRate > 70 ? '🚨 Critical' : 
                                factor.deathRate > 50 ? '⚠️ High' : 
                                '📊 Moderate';
                
                html += `
                    <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid ${COLORS.danger};">
                        <div style="font-weight: bold; color: ${COLORS.dark}; margin-bottom: 5px;">
                            ${index + 1}. ${factor.feature}: "${factor.value}"
                        </div>
                        <div style="color: ${COLORS.danger}; font-weight: 600;">
                            ${factor.deathRate ? `Death Rate: ${factor.deathRate}%` : `Impact: ${factor.value}`}
                        </div>
                        <div style="color: #666; font-size: 0.9rem; margin-top: 5px;">
                            Risk Level: ${severity}
                        </div>
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }
        
        html += `
            <div style="margin-bottom: 25px;">
                <h4 style="color: ${COLORS.dark}; margin-bottom: 15px;">📊 Top Death Factors by Category</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
        `;
        
        Object.entries(deathFactors.factors).forEach(([feature, values]) => {
            const sortedValues = Object.entries(values)
                .sort((a, b) => b[1].deathRate - a[1].deathRate)
                .slice(0, 3);
            
            if (sortedValues.length > 0) {
                html += `
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;">
                        <div style="font-weight: bold; color: ${COLORS.dark}; margin-bottom: 10px;">
                            ${feature}
                        </div>
                `;
                
                sortedValues.forEach(([value, stats]) => {
                    const riskColor = stats.deathRate > 70 ? COLORS.danger : 
                                     stats.deathRate > 50 ? COLORS.warning : 
                                     COLORS.primary;
                    
                    html += `
                        <div style="margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #eee;">
                            <div style="font-weight: 500;">${value}</div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                                <span style="color: ${riskColor};">Died: ${stats.deathRate}%</span>
                                <span style="color: ${COLORS.success};">Survived: ${stats.survivalRate}%</span>
                            </div>
                        </div>
                    `;
                });
                
                html += `</div>`;
            }
        });
        
        html += `</div></div>`;
        
        const conclusion = deathFactors.mainFactors.length > 0 ? 
            `<h4 style="color: ${COLORS.danger}; margin-top: 20px; padding: 15px; background: #fff5f5; border-radius: 8px;">
                🔍 CONCLUSION: The MAIN FACTOR for death is <strong>${deathFactors.mainFactors[0].feature}</strong> 
                with value "<strong>${deathFactors.mainFactors[0].value}</strong>" having 
                ${deathFactors.mainFactors[0].deathRate ? `a ${deathFactors.mainFactors[0].deathRate}% death rate` : 'significant impact'}.
                ${deathFactors.mainFactors[0].feature === 'Pclass' ? 'Lower class passengers (3rd class) had the highest mortality.' : ''}
                ${deathFactors.mainFactors[0].feature === 'Sex' ? 'Male passengers had significantly higher mortality than females.' : ''}
                ${deathFactors.mainFactors[0].feature === 'Embarked' ? 'Passengers from certain embarkation points had higher mortality.' : ''}
            </h4>` : 
            `<h4 style="color: ${COLORS.warning}; margin-top: 20px;">⚠️ No significant death factors identified above threshold.</h4>`;
        
        html += conclusion;
        
        summaryDiv.innerHTML = html;
        categoricalStats.appendChild(summaryDiv);
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
        
        const numericFeatures = ['Age', 'Fare', 'SibSp', 'Parch', DATASET_SCHEMA.target];
        const validFeatures = numericFeatures.filter(feature => {
            if (feature === DATASET_SCHEMA.target) {
                return trainData[0].hasOwnProperty(feature);
            }
            return true;
        });
        
        const featureData = validFeatures.map(feature => {
            return trainData.map(row => row[feature]);
        });
        
        const validIndices = [];
        const n = trainData.length;
        
        for (let i = 0; i < n; i++) {
            let valid = true;
            for (let j = 0; j < validFeatures.length; j++) {
                const val = featureData[j][i];
                if (val === null || val === undefined || isNaN(val)) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                validIndices.push(i);
            }
        }
        
        if (validIndices.length < 2) {
            showAlert(loadAlert, 'Insufficient complete data for correlation analysis', 'warning');
            return;
        }
        
        const dataMatrix = [];
        validIndices.forEach(idx => {
            const row = validFeatures.map((_, j) => featureData[j][idx]);
            dataMatrix.push(row);
        });
        
        const correlationMatrix = [];
        const m = validFeatures.length;
        
        for (let i = 0; i < m; i++) {
            correlationMatrix[i] = [];
            for (let j = 0; j < m; j++) {
                if (i === j) {
                    correlationMatrix[i][j] = 1;
                } else {
                    const x = dataMatrix.map(row => row[i]);
                    const y = dataMatrix.map(row => row[j]);
                    correlationMatrix[i][j] = pearsonCorrelation(x, y);
                }
            }
        }
        
        const dataPoints = [];
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < m; j++) {
                dataPoints.push({
                    x: j,
                    y: i,
                    v: correlationMatrix[i][j]
                });
            }
        }
        
        if (charts.mainChart) {
            charts.mainChart.destroy();
        }
        
        charts.mainChart = new Chart(mainChartCtx, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Correlation',
                    data: dataPoints,
                    backgroundColor: (ctx) => {
                        const value = ctx.dataset.data[ctx.dataIndex].v;
                        const alpha = Math.min(Math.abs(value) * 1.5, 1);
                        
                        if (value > 0.3) {
                            return `rgba(46, 204, 113, ${alpha})`;
                        } else if (value > 0) {
                            return `rgba(52, 152, 219, ${alpha})`;
                        } else if (value < -0.3) {
                            return `rgba(231, 76, 60, ${alpha})`;
                        } else if (value < 0) {
                            return `rgba(241, 196, 15, ${alpha})`;
                        } else {
                            return 'rgba(149, 165, 166, 0.5)';
                        }
                    },
                    borderColor: '#fff',
                    borderWidth: 1,
                    width: ({chart}) => (chart.chartArea || {}).width / m - 1,
                    height: ({chart}) => (chart.chartArea || {}).height / m - 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Correlation Heatmap (with Survival)',
                        font: { size: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const data = ctx.dataset.data[ctx.dataIndex];
                                const xLabel = validFeatures[data.x];
                                const yLabel = validFeatures[data.y];
                                const correlation = data.v;
                                let interpretation = '';
                                
                                if (correlation > 0.7) interpretation = 'Very Strong Positive';
                                else if (correlation > 0.3) interpretation = 'Positive';
                                else if (correlation > 0.1) interpretation = 'Weak Positive';
                                else if (correlation > -0.1) interpretation = 'No Correlation';
                                else if (correlation > -0.3) interpretation = 'Weak Negative';
                                else if (correlation > -0.7) interpretation = 'Negative';
                                else interpretation = 'Very Strong Negative';
                                
                                return `${yLabel} ↔ ${xLabel}: ${correlation.toFixed(3)} (${interpretation})`;
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
                            callback: (value, index) => validFeatures[index]
                        },
                        grid: { display: false }
                    },
                    y: {
                        ticks: {
                            display: true,
                            callback: (value, index) => validFeatures[index]
                        },
                        grid: { display: false }
                    }
                }
            }
        });
        
        analysisResults.visualizations.correlation = {
            features: validFeatures,
            matrix: correlationMatrix,
            completeSamples: validIndices.length,
            type: 'heatmap'
        };
        
        displayCorrelationInsights(correlationMatrix, validFeatures);
    }
    
    function displayCorrelationInsights(matrix, features) {
        const insightsDiv = document.createElement('div');
        insightsDiv.className = 'correlation-insights';
        insightsDiv.style.cssText = `
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid ${COLORS.info};
        `;
        
        let insightsHTML = '<h4 style="color: #2c3e50; margin-bottom: 15px;">📊 Correlation Insights</h4>';
        
        const targetIndex = features.indexOf(DATASET_SCHEMA.target);
        if (targetIndex !== -1) {
            insightsHTML += '<div style="margin-bottom: 10px;"><strong>Correlation with Survival:</strong></div>';
            insightsHTML += '<div style="display: grid; gap: 10px;">';
            
            for (let i = 0; i < features.length; i++) {
                if (i !== targetIndex) {
                    const correlation = matrix[targetIndex][i];
                    const absCorrelation = Math.abs(correlation);
                    let strength = '';
                    let color = '';
                    let icon = '';
                    
                    if (absCorrelation > 0.3) {
                        strength = correlation > 0 ? 'Positive' : 'Negative';
                        color = correlation > 0 ? COLORS.success : COLORS.danger;
                        icon = correlation > 0 ? '📈' : '📉';
                    } else {
                        strength = 'Weak';
                        color = COLORS.warning;
                        icon = '📊';
                    }
                    
                    insightsHTML += `
                        <div style="display: flex; justify-content: space-between; align-items: center; 
                                    padding: 8px 12px; background: white; border-radius: 6px; border-left: 3px solid ${color};">
                            <div>
                                <span style="font-weight: 500;">${features[i]}</span>
                                <span style="margin-left: 10px; font-size: 0.9em; color: #666;">
                                    ${correlation > 0 ? 'Helps Survival' : 'Reduces Survival'}
                                </span>
                            </div>
                            <div style="font-weight: bold; color: ${color};">
                                ${icon} ${correlation.toFixed(3)}
                            </div>
                        </div>
                    `;
                }
            }
            
            insightsHTML += '</div>';
        }
        
        insightsHTML += `
            <div style="margin-top: 15px; padding: 10px; background: #e8f4fc; border-radius: 6px;">
                <div style="color: #2c3e50; font-weight: 500; margin-bottom: 5px;">🎯 Key Finding:</div>
                <div>
                    ${matrix[targetIndex][features.indexOf('Fare')] > 0.2 ? 
                      'Higher fare correlates with better survival (wealthier passengers had priority).' : 
                      matrix[targetIndex][features.indexOf('Age')] < -0.1 ? 
                      'Younger passengers had slightly better survival rates.' : 
                      'No strong linear correlations found with numeric features.'}
                </div>
            </div>
        `;
        
        insightsDiv.innerHTML = insightsHTML;
        
        const chartContainer = document.querySelector('.chart-container');
        chartContainer.parentNode.insertBefore(insightsDiv, chartContainer.nextSibling);
    }
    
    function pearsonCorrelation(x, y) {
        const n = x.length;
        if (n < 2) return 0;
        
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        if (denominator === 0) return 0;
        const correlation = numerator / denominator;
        
        return Math.max(-1, Math.min(1, correlation));
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
                analysis: {
                    missingValues: analysisResults.missingValues,
                    statistics: analysisResults.statistics,
                    visualizations: analysisResults.visualizations,
                    deathFactors: analysisResults.deathFactors
                },
                schema: DATASET_SCHEMA,
                mainConclusion: analysisResults.deathFactors.mainFactors && analysisResults.deathFactors.mainFactors.length > 0 ?
                    `MAIN DEATH FACTOR: ${analysisResults.deathFactors.mainFactors[0].feature} - ${analysisResults.deathFactors.mainFactors[0].value}` :
                    'No significant death factor identified'
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
        
        if (analysisResults.deathFactors && analysisResults.deathFactors.mainFactors) {
            report += '🚨 MAIN DEATH FACTOR ANALYSIS\n';
            report += '-'.repeat(40) + '\n';
            report += `Overall Death Rate: ${analysisResults.deathFactors.deathRate || 0}%\n`;
            report += `Overall Survival Rate: ${analysisResults.deathFactors.survivalRate || 0}%\n\n`;
            
            if (analysisResults.deathFactors.mainFactors.length > 0) {
                report += 'TOP 5 DEATH FACTORS:\n';
                analysisResults.deathFactors.mainFactors.slice(0, 5).forEach((factor, idx) => {
                    report += `${idx + 1}. ${factor.feature}: ${factor.value} `;
                    if (factor.deathRate) {
                        report += `(Death Rate: ${factor.deathRate}%)\n`;
                    } else {
                        report += `(Significant Impact)\n`;
                    }
                });
                
                report += `\n🎯 PRIMARY CONCLUSION:\n`;
                const mainFactor = analysisResults.deathFactors.mainFactors[0];
                report += `The MAIN FACTOR contributing to death is "${mainFactor.feature}" `;
                report += `with value "${mainFactor.value}". `;
                if (mainFactor.deathRate) {
                    report += `This group had a ${mainFactor.deathRate}% mortality rate, `;
                    report += `which is ${mainFactor.deathRate > (analysisResults.deathFactors.deathRate || 0) ? 'above' : 'below'} average.\n`;
                }
                
                if (mainFactor.feature === 'Pclass') {
                    report += `Third class passengers had significantly lower survival chances.\n`;
                } else if (mainFactor.feature === 'Sex') {
                    report += `Male passengers had much higher mortality rates than females.\n`;
                } else if (mainFactor.feature === 'Embarked') {
                    report += `Embarkation location was a significant predictor of survival.\n`;
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
            deathFactors: {}
        };
        
        destroyAllCharts();
        
        overviewStats.innerHTML = '';
        previewTable.innerHTML = '';
        missingValues.innerHTML = '';
        numericStats.innerHTML = '';
        categoricalStats.innerHTML = '';
        
        loadAlert.style.display = 'none';
        exportAlert.style.display = 'none';
    }
    
    // ==================== INITIALIZATION ====================
    console.log('🚀 Titanic EDA Explorer initialized');
    console.log('📝 To analyze different datasets:');
    console.log('1. Update DATASET_SCHEMA configuration');
    console.log('2. Adjust feature lists for your dataset');
    console.log('3. Update target variable if different');
    console.log('4. Update death factor analysis logic');
    
    showAlert(loadAlert, '📁 Please upload train.csv and test.csv files from Kaggle Titanic dataset', 'info');
});

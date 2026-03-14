// charts.js — Chart.js setup and update logic for training visualizations.

let lossChart = null;
let energyRegChart = null;
let lrChart = null;

const MAX_POINTS = 2000;

function initCharts() {
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: {
                labels: { color: '#8b8fa8', font: { size: 11 } }
            }
        },
        scales: {
            x: {
                ticks: { color: '#8b8fa8', font: { size: 10 } },
                grid: { color: '#2e3348' }
            },
            y: {
                ticks: { color: '#8b8fa8', font: { size: 10 } },
                grid: { color: '#2e3348' }
            }
        }
    };

    // Total loss chart
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Total Loss',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                title: {
                    display: true,
                    text: 'Total Loss',
                    color: '#e4e6f0',
                    font: { size: 13 }
                }
            }
        }
    });

    // Energy vs Regularization chart
    const erCtx = document.getElementById('energy-reg-chart').getContext('2d');
    energyRegChart = new Chart(erCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Energy',
                    data: [],
                    borderColor: '#f59e0b',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'Regularization',
                    data: [],
                    borderColor: '#06b6d4',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3
                }
            ]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                title: {
                    display: true,
                    text: 'Energy vs Regularization',
                    color: '#e4e6f0',
                    font: { size: 12 }
                }
            }
        }
    });

    // Learning rate chart
    const lrCtx = document.getElementById('lr-chart').getContext('2d');
    lrChart = new Chart(lrCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Learning Rate',
                data: [],
                borderColor: '#a855f7',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.3
            }]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                title: {
                    display: true,
                    text: 'Learning Rate Schedule',
                    color: '#e4e6f0',
                    font: { size: 12 }
                }
            }
        }
    });
}

function updateCharts(metrics) {
    const step = metrics.step;

    // Total loss
    lossChart.data.labels.push(step);
    lossChart.data.datasets[0].data.push(metrics.total_loss);
    if (lossChart.data.labels.length > MAX_POINTS) {
        lossChart.data.labels.shift();
        lossChart.data.datasets[0].data.shift();
    }
    lossChart.update('none');

    // Energy + Reg
    energyRegChart.data.labels.push(step);
    energyRegChart.data.datasets[0].data.push(metrics.energy);
    energyRegChart.data.datasets[1].data.push(metrics.regularization);
    if (energyRegChart.data.labels.length > MAX_POINTS) {
        energyRegChart.data.labels.shift();
        energyRegChart.data.datasets[0].data.shift();
        energyRegChart.data.datasets[1].data.shift();
    }
    energyRegChart.update('none');

    // LR
    lrChart.data.labels.push(step);
    lrChart.data.datasets[0].data.push(metrics.learning_rate);
    if (lrChart.data.labels.length > MAX_POINTS) {
        lrChart.data.labels.shift();
        lrChart.data.datasets[0].data.shift();
    }
    lrChart.update('none');
}

function resetCharts() {
    [lossChart, energyRegChart, lrChart].forEach(chart => {
        if (!chart) return;
        chart.data.labels = [];
        chart.data.datasets.forEach(ds => { ds.data = []; });
        chart.update('none');
    });
}

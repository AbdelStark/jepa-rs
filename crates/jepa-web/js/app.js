// app.js — Main UI orchestration for the jepa-rs browser demo.

// State
let isTraining = false;
let trainingTimer = null;
let startTime = null;
let totalStepsConfig = 200;
let wasmReady = false;

// ─── Initialization ───────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initCharts();
    initDrawCanvas('draw-canvas');
    initTrainingControls();
    initInferenceControls();

    // Wait for WASM to load, then initialize.
    if (typeof init_demo === 'function') {
        onWasmReady();
    } else {
        // Trunk injects the WASM init; poll until available.
        const poll = setInterval(() => {
            if (typeof init_demo === 'function') {
                clearInterval(poll);
                onWasmReady();
            }
        }, 100);
    }
});

function onWasmReady() {
    wasmReady = true;
    init_demo();
    console.log('[app] WASM module ready');

    // Auto-create a default session.
    const config = getTrainingConfig();
    try {
        const infoJson = create_training_session(JSON.stringify(config));
        const info = JSON.parse(infoJson);
        displayModelInfo(info);
    } catch (e) {
        console.error('Failed to create training session:', e);
    }
}

// ─── Tab Navigation ───────────────────────────────────────────

function initTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => {
                t.classList.remove('active');
                t.setAttribute('aria-selected', 'false');
            });
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            tab.setAttribute('aria-selected', 'true');
            const target = tab.dataset.tab + '-tab';
            document.getElementById(target).classList.add('active');
        });
    });
}

// ─── Training Controls ────────────────────────────────────────

function initTrainingControls() {
    document.getElementById('btn-start').addEventListener('click', startTraining);
    document.getElementById('btn-pause').addEventListener('click', pauseTraining);
    document.getElementById('btn-reset').addEventListener('click', resetTraining);
}

function getTrainingConfig() {
    return {
        learning_rate: parseFloat(document.getElementById('lr').value),
        batch_size: parseInt(document.getElementById('batch-size').value, 10),
        total_steps: parseInt(document.getElementById('total-steps').value, 10),
        warmup_steps: parseInt(document.getElementById('warmup-steps').value, 10),
        ema_momentum: parseFloat(document.getElementById('ema-momentum').value),
        reg_weight: parseFloat(document.getElementById('reg-weight').value),
    };
}

function startTraining() {
    if (!wasmReady) return;

    if (!isTraining) {
        // Create fresh session if step is 0.
        const currentStep = get_current_step();
        if (currentStep === 0) {
            const config = getTrainingConfig();
            totalStepsConfig = config.total_steps;
            try {
                const infoJson = create_training_session(JSON.stringify(config));
                const info = JSON.parse(infoJson);
                displayModelInfo(info);
                resetCharts();
            } catch (e) {
                console.error('Failed to create session:', e);
                return;
            }
        }

        isTraining = true;
        startTime = startTime || performance.now();
        document.getElementById('btn-start').disabled = true;
        document.getElementById('btn-pause').disabled = false;
        document.getElementById('btn-reset').disabled = false;
        disableConfigInputs(true);

        runTrainingLoop();
    }
}

function pauseTraining() {
    isTraining = false;
    if (trainingTimer) {
        cancelAnimationFrame(trainingTimer);
        trainingTimer = null;
    }
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').textContent = 'Resume';
    document.getElementById('btn-pause').disabled = true;
}

function resetTraining() {
    pauseTraining();
    startTime = null;

    try {
        const infoJson = reset_training();
        const info = JSON.parse(infoJson);
        displayModelInfo(info);
    } catch (e) {
        console.error('Failed to reset training:', e);
    }

    resetCharts();
    updateStatusBar(0, totalStepsConfig, 0);

    document.getElementById('btn-start').textContent = 'Start';
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-reset').disabled = true;
    disableConfigInputs(false);
}

function runTrainingLoop() {
    if (!isTraining) return;

    const currentStep = get_current_step();
    if (currentStep >= totalStepsConfig) {
        pauseTraining();
        document.getElementById('btn-start').textContent = 'Start';
        disableConfigInputs(false);
        return;
    }

    try {
        const metricsJson = training_step();
        const metrics = JSON.parse(metricsJson);

        updateCharts(metrics);
        updateStatusBar(metrics.step + 1, totalStepsConfig, performance.now() - startTime);
    } catch (e) {
        console.error('Training step failed:', e);
        pauseTraining();
        return;
    }

    // Yield to browser, then continue.
    trainingTimer = requestAnimationFrame(runTrainingLoop);
}

function updateStatusBar(step, total, elapsedMs) {
    document.getElementById('step-counter').textContent = `Step: ${step} / ${total}`;

    const elapsed = (elapsedMs / 1000).toFixed(1);
    document.getElementById('elapsed-time').textContent = `Elapsed: ${elapsed}s`;

    const rate = elapsedMs > 0 ? (step / (elapsedMs / 1000)).toFixed(1) : '0';
    document.getElementById('step-rate').textContent = `${rate} steps/s`;
}

function disableConfigInputs(disabled) {
    ['lr', 'batch-size', 'total-steps', 'warmup-steps', 'ema-momentum', 'reg-weight']
        .forEach(id => {
            document.getElementById(id).disabled = disabled;
        });
}

function displayModelInfo(info) {
    const el = document.getElementById('model-info');
    if (!el) return;
    el.textContent =
        `Preset:     ${info.preset}\n` +
        `Embed dim:  ${info.embed_dim}\n` +
        `Layers:     ${info.num_layers}\n` +
        `Heads:      ${info.num_heads}\n` +
        `Patch size: ${info.patch_size[0]}x${info.patch_size[1]}\n` +
        `Image size: ${info.image_size[0]}x${info.image_size[1]}\n` +
        `Patches:    ${info.num_patches}\n` +
        `Channels:   ${info.in_channels}`;
}

// ─── Inference Controls ───────────────────────────────────────

function initInferenceControls() {
    document.getElementById('btn-run-inference').addEventListener('click', runPatternInference);
    document.getElementById('btn-clear-canvas').addEventListener('click', () => clearDrawCanvas('draw-canvas'));
    document.getElementById('btn-infer-drawing').addEventListener('click', runDrawingInference);

    document.getElementById('image-upload').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            loadImageToCanvas(file, 'input-canvas');
        }
    });
}

function runPatternInference() {
    if (!wasmReady) return;

    const pattern = document.getElementById('pattern-select').value;
    const t0 = performance.now();

    try {
        const resultJson = run_inference_on_pattern(pattern);
        const result = JSON.parse(resultJson);
        result.latency_ms = performance.now() - t0;

        displayInferenceStats('inference-stats', result);
        renderHeatmap('heatmap-canvas', result.patch_norms, result.grid_height, result.grid_width);
        document.getElementById('inference-latency').textContent =
            `Inference: ${result.latency_ms.toFixed(1)} ms`;
    } catch (e) {
        console.error('Inference failed:', e);
    }
}

function runDrawingInference() {
    if (!wasmReady) return;

    // tiny_test uses 1 channel, 8x8 images.
    const pixels = getCanvasPixelsCHW('draw-canvas', 1, 8, 8);
    const t0 = performance.now();

    try {
        const resultJson = run_inference_on_data(pixels, 1, 8, 8);
        const result = JSON.parse(resultJson);
        result.latency_ms = performance.now() - t0;

        displayInferenceStats('inference-stats', result);
        renderHeatmap('heatmap-canvas', result.patch_norms, result.grid_height, result.grid_width);
        document.getElementById('inference-latency').textContent =
            `Inference: ${result.latency_ms.toFixed(1)} ms`;
    } catch (e) {
        console.error('Drawing inference failed:', e);
    }
}

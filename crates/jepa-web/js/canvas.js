// canvas.js — Image input/output rendering and heatmap visualization.

/**
 * Render a patch-norm heatmap on a canvas element.
 *
 * @param {string} canvasId - Canvas element ID.
 * @param {number[]} patchNorms - Array of per-patch L2 norms.
 * @param {number} gridH - Number of rows in the patch grid.
 * @param {number} gridW - Number of columns in the patch grid.
 */
function renderHeatmap(canvasId, patchNorms, gridH, gridW) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cellW = canvas.width / gridW;
    const cellH = canvas.height / gridH;

    // Normalize values to [0, 1].
    const minVal = Math.min(...patchNorms);
    const maxVal = Math.max(...patchNorms);
    const range = maxVal - minVal || 1;

    for (let row = 0; row < gridH; row++) {
        for (let col = 0; col < gridW; col++) {
            const idx = row * gridW + col;
            const normalized = (patchNorms[idx] - minVal) / range;

            // Viridis-inspired colormap.
            const r = Math.round(68 + normalized * (253 - 68));
            const g = Math.round(1 + normalized * (231 - 1));
            const b = Math.round(84 + (1 - normalized) * (168 - 84));

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(
                Math.floor(col * cellW),
                Math.floor(row * cellH),
                Math.ceil(cellW),
                Math.ceil(cellH)
            );
        }
    }
}

/**
 * Display inference statistics in the stats grid.
 *
 * @param {string} containerId - Stats container element ID.
 * @param {Object} result - InferenceResult from WASM.
 */
function displayInferenceStats(containerId, result) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = [
        { label: 'Patches', value: result.num_patches },
        { label: 'Embed Dim', value: result.embed_dim },
        { label: 'Mean', value: result.mean.toFixed(6) },
        { label: 'Std Dev', value: result.std_dev.toFixed(6) },
        { label: 'Min', value: result.min.toFixed(6) },
        { label: 'Max', value: result.max.toFixed(6) },
    ].map(s =>
        `<div class="stat-card">
            <div class="label">${s.label}</div>
            <div class="value">${s.value}</div>
        </div>`
    ).join('');
}

/**
 * Set up the drawing canvas with mouse/touch event handlers.
 *
 * @param {string} canvasId - Canvas element ID.
 */
function initDrawCanvas(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let drawing = false;

    // White background
    ctx.fillStyle = '#1a1d28';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: (clientX - rect.left) * (canvas.width / rect.width),
            y: (clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    function startDraw(e) {
        e.preventDefault();
        drawing = true;
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    function draw(e) {
        if (!drawing) return;
        e.preventDefault();
        const pos = getPos(e);
        ctx.lineWidth = 6;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#e4e6f0';
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function endDraw() {
        drawing = false;
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mouseleave', endDraw);

    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', endDraw);
}

/**
 * Clear the drawing canvas.
 *
 * @param {string} canvasId - Canvas element ID.
 */
function clearDrawCanvas(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1a1d28';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

/**
 * Get pixel data from a canvas as a Float32Array in CHW format.
 *
 * @param {string} canvasId - Canvas element ID.
 * @param {number} channels - Number of output channels.
 * @param {number} targetH - Target height.
 * @param {number} targetW - Target width.
 * @returns {Float32Array} Pixel data in CHW order, normalized to [0, 1].
 */
function getCanvasPixelsCHW(canvasId, channels, targetH, targetW) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return new Float32Array(0);

    // Create temporary canvas at target size.
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = targetW;
    tmpCanvas.height = targetH;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(canvas, 0, 0, targetW, targetH);

    const imageData = tmpCtx.getImageData(0, 0, targetW, targetH);
    const pixels = imageData.data; // RGBA

    const result = new Float32Array(channels * targetH * targetW);

    for (let c = 0; c < channels; c++) {
        for (let row = 0; row < targetH; row++) {
            for (let col = 0; col < targetW; col++) {
                const rgbaIdx = (row * targetW + col) * 4;
                // For single channel, use grayscale (average of RGB).
                // For 3 channels, use R, G, B directly.
                let value;
                if (channels === 1) {
                    value = (pixels[rgbaIdx] + pixels[rgbaIdx + 1] + pixels[rgbaIdx + 2]) / (3 * 255);
                } else {
                    value = pixels[rgbaIdx + c] / 255;
                }
                result[c * targetH * targetW + row * targetW + col] = value;
            }
        }
    }

    return result;
}

/**
 * Load an image file into a canvas for preview.
 *
 * @param {File} file - Image file from file input.
 * @param {string} canvasId - Canvas element ID.
 */
function loadImageToCanvas(file, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !file) return;

    const ctx = canvas.getContext('2d');
    const reader = new FileReader();

    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // Draw image scaled to fit canvas while preserving aspect ratio.
            const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
            const w = img.width * scale;
            const h = img.height * scale;
            const x = (canvas.width - w) / 2;
            const y = (canvas.height - h) / 2;
            ctx.fillStyle = '#1a1d28';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, x, y, w, h);
        };
        img.src = e.target.result;
    };

    reader.readAsDataURL(file);
}

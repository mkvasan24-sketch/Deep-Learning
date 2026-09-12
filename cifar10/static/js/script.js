/**
 * CIFAR Vision — Frontend Client Script
 * Handles drag-and-drop, image preview, AJAX inference, and dynamic UI state.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Input Studio
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const emptyUploadState = document.getElementById('emptyUploadState');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const metaFileName = document.getElementById('metaFileName');
    const metaDimensions = document.getElementById('metaDimensions');
    const metaFileSize = document.getElementById('metaFileSize');
    const predictBtn = document.getElementById('predictBtn');
    const predictBtnText = document.getElementById('predictBtnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnIcon = document.getElementById('btnIcon');
    const clearBtn = document.getElementById('clearBtn');
    const sampleButtons = document.querySelectorAll('.btn-sample');

    // DOM Elements - Prediction Deck
    const deckEmptyState = document.getElementById('deckEmptyState');
    const deckLoadingState = document.getElementById('deckLoadingState');
    const deckResultState = document.getElementById('deckResultState');
    const winnerEmoji = document.getElementById('winnerEmoji');
    const winnerName = document.getElementById('winnerName');
    const winnerConfidence = document.getElementById('winnerConfidence');
    const winnerMeterFill = document.getElementById('winnerMeterFill');
    const topPredictionsList = document.getElementById('topPredictionsList');
    const telLatency = document.getElementById('telLatency');
    const toastContainer = document.getElementById('toastContainer');

    // State Variables
    let currentSelectedFile = null;
    let isPredicting = false;

    // Permitted file formats & max size (10 MB)
    const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    const MAX_FILE_SIZE = 10 * 1024 * 1024;

    // =========================================================================
    // File Selection & Drag-and-Drop Handlers
    // =========================================================================

    // Click dropzone to browse
    dropZone.addEventListener('click', (e) => {
        if (!isPredicting) {
            imageInput.click();
        }
    });

    // Native file input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag-over and Drag-enter
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!isPredicting) {
                dropZone.classList.add('drag-over');
            }
        });
    });

    // Drag-leave and Drop
    ['dragleave', 'dragend'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');

        if (isPredicting) return;

        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    // Keyboard accessibility for dropzone
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            imageInput.click();
        }
    });

    // =========================================================================
    // Quick Test Samples
    // =========================================================================

    sampleButtons.forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            if (isPredicting) return;

            const sampleFilename = btn.dataset.sample;
            const sampleUrl = `/static/samples/${sampleFilename}`;

            try {
                btn.classList.add('loading-sample');
                const response = await fetch(sampleUrl);
                if (!response.ok) {
                    throw new Error(`Sample ${sampleFilename} not found.`);
                }
                const blob = await response.blob();
                const file = new File([blob], sampleFilename, { type: blob.type || 'image/png' });
                
                handleFileSelection(file);
                showToast(`Loaded sample: ${btn.textContent.trim()}`, 'success');
            } catch (err) {
                showToast(`Could not load sample image: ${err.message}`, 'error');
            } finally {
                btn.classList.remove('loading-sample');
            }
        });
    });

    // =========================================================================
    // File Validation & Preview
    // =========================================================================

    function handleFileSelection(file) {
        // Validate file type
        const fileType = file.type.toLowerCase();
        const fileName = file.name.toLowerCase();
        const validExtension = /\.(jpg|jpeg|png|webp)$/i.test(fileName);

        if (!ALLOWED_TYPES.includes(fileType) && !validExtension) {
            showToast('Unsupported image format. Please upload JPG, PNG, or WEBP.', 'error');
            return;
        }

        // Validate file size
        if (file.size > MAX_FILE_SIZE) {
            showToast('File is too large. Maximum supported size is 10 MB.', 'error');
            return;
        }

        currentSelectedFile = file;

        // Populate meta
        metaFileName.textContent = file.name;
        metaFileSize.textContent = formatBytes(file.size);

        // Read and display preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.onload = () => {
                metaDimensions.textContent = `${previewImage.naturalWidth} × ${previewImage.naturalHeight} px`;
            };
            previewImage.src = e.target.result;

            // Update UI state
            emptyUploadState.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            predictBtn.disabled = false;
            clearBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // =========================================================================
    // Predict Action (AJAX inference)
    // =========================================================================

    predictBtn.addEventListener('click', async () => {
        if (!currentSelectedFile || isPredicting) {
            showToast('Please select or upload an image first.', 'error');
            return;
        }

        setLoadingState(true);

        const formData = new FormData();
        formData.append('image', currentSelectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                const errorMsg = data.error || 'Prediction failed. Please try again.';
                showToast(errorMsg, 'error');
                showEmptyDeck();
                return;
            }

            // Render prediction results
            renderPredictionResults(data);

        } catch (error) {
            console.error('Prediction request error:', error);
            showToast('Unable to communicate with the classification server. Please check your connection.', 'error');
            showEmptyDeck();
        } finally {
            setLoadingState(false);
        }
    });

    // =========================================================================
    // Result Presentation
    // =========================================================================

    function renderPredictionResults(data) {
        // Switch deck visibility
        deckEmptyState.classList.add('hidden');
        deckLoadingState.classList.add('hidden');
        deckResultState.classList.remove('hidden');

        // Winner details
        winnerEmoji.textContent = data.emoji || '🎯';
        winnerName.textContent = data.prediction.toUpperCase();
        
        // Count-up animation for confidence
        animateConfidenceCounter(winnerConfidence, data.confidence);

        // Winner meter animation
        winnerMeterFill.style.width = '0%';
        setTimeout(() => {
            winnerMeterFill.style.width = `${Math.min(data.confidence, 100)}%`;
        }, 100);

        // Render top 3 predictions
        topPredictionsList.innerHTML = '';
        if (data.top_predictions && data.top_predictions.length > 0) {
            data.top_predictions.forEach((item, index) => {
                const row = document.createElement('div');
                row.className = `prediction-row rank-${item.rank || (index + 1)}`;
                row.innerHTML = `
                    <div class="pred-row-header">
                        <div class="pred-rank-group">
                            <span class="pred-rank-badge">${item.rank || (index + 1)}</span>
                            <span class="pred-emoji">${item.emoji}</span>
                            <span class="pred-name">${item.displayName || item.class.toUpperCase()}</span>
                        </div>
                        <span class="pred-percent">${item.confidence.toFixed(2)}%</span>
                    </div>
                    <div class="pred-bar-track">
                        <div class="pred-bar-fill" style="width: 0%;"></div>
                    </div>
                `;
                topPredictionsList.appendChild(row);

                // Animate progress bar fill
                setTimeout(() => {
                    const fill = row.querySelector('.pred-bar-fill');
                    if (fill) {
                        fill.style.width = `${Math.min(item.confidence, 100)}%`;
                    }
                }, 150 + index * 100);
            });
        }

        // Telemetry
        telLatency.textContent = `${data.inference_time_ms} ms`;
    }

    // =========================================================================
    // Reset / Clear Handler
    // =========================================================================

    clearBtn.addEventListener('click', () => {
        resetAllState();
        showToast('Workspace cleared. Upload a new image to classify.', 'success');
    });

    function resetAllState() {
        currentSelectedFile = null;
        imageInput.value = '';

        // Reset upload studio
        previewImage.src = '';
        previewContainer.classList.add('hidden');
        emptyUploadState.classList.remove('hidden');
        metaFileName.textContent = '--';
        metaDimensions.textContent = '--';
        metaFileSize.textContent = '--';

        predictBtn.disabled = true;
        clearBtn.disabled = true;

        // Reset prediction deck
        showEmptyDeck();
    }

    function showEmptyDeck() {
        deckResultState.classList.add('hidden');
        deckLoadingState.classList.add('hidden');
        deckEmptyState.classList.remove('hidden');
        winnerMeterFill.style.width = '0%';
    }

    function setLoadingState(loading) {
        isPredicting = loading;

        if (loading) {
            predictBtn.disabled = true;
            clearBtn.disabled = true;
            btnSpinner.classList.remove('hidden');
            btnIcon.classList.add('hidden');
            predictBtnText.textContent = 'Analyzing...';

            deckEmptyState.classList.add('hidden');
            deckResultState.classList.add('hidden');
            deckLoadingState.classList.remove('hidden');
        } else {
            predictBtn.disabled = (currentSelectedFile === null);
            clearBtn.disabled = (currentSelectedFile === null);
            btnSpinner.classList.add('hidden');
            btnIcon.classList.remove('hidden');
            predictBtnText.textContent = 'Predict Image';
        }
    }

    // =========================================================================
    // Helper Utilities
    // =========================================================================

    function animateConfidenceCounter(element, targetValue) {
        const duration = 800; // ms
        const start = 0;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease out cubic
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            const current = (start + (targetValue - start) * easeProgress).toFixed(2);
            element.textContent = `${current}%`;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = `${targetValue.toFixed(2)}%`;
            }
        }

        requestAnimationFrame(update);
    }

    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function showToast(message, type = 'error') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icon = type === 'success' ? '✅' : (type === 'warning' ? '⚠️' : '❌');
        toast.innerHTML = `
            <span class="toast-icon">${icon}</span>
            <span class="toast-message">${message}</span>
        `;

        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }, 4500);
    }
});

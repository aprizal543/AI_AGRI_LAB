// 1. Tentukan nama kelas Anda dalam bentuk ARRAY
// Pastikan urutannya SAMA PERSIS dengan indeks (0, 1, 2, ...)
const NAMA_KELAS = [
    'Bacterial Leaf Blight', // index 0
    'Brown Spot',            // index 1
    'Healthy Rice Leaf',     // index 2
    'Leaf Blast',            // index 3
    'Leaf scald',            // index 4
    'Narrow Brown Leaf Spot',// index 5
    'Neck_Blast',            // index 6
    'Rice Hispa',            // index 7
    'Sheath Blight'          // index 8
];

const MODEL_PATH = 'model/model.json';
const EXPLAIN_API_URL = 'http://localhost:4000/explain';

let riceModel = null;

document.addEventListener('DOMContentLoaded', initializeApp);

async function initializeApp() {
    const ui = {
        uploadArea: document.getElementById('upload-area'),
        inputGambar: document.getElementById('inputGambar'),
        previewColumn: document.getElementById('preview-column'),
        imagePreview: document.getElementById('image-preview'),
        fileInfo: document.getElementById('file-info'),
        predictButton: document.getElementById('predict-button'),
        resultsSection: document.getElementById('results-section'),
        loadingSpinner: document.getElementById('loading-spinner'),
        resultsContent: document.getElementById('results-content'),
        resultImage: document.getElementById('result-image'),
        diseaseName: document.getElementById('disease-name'),
        confidenceBar: document.getElementById('confidence-bar'),
        confidenceValue: document.getElementById('confidence-value'),
        diseaseDescription: document.getElementById('disease-description'),
        diseaseTreatment: document.getElementById('disease-treatment'),
        resetButton: document.getElementById('reset-button')
    };

    attachUploadHandlers(ui);
    attachPredictHandlers(ui);

    await loadModel(ui.predictButton);
}

function attachUploadHandlers(ui) {
    if (ui.uploadArea && ui.inputGambar) {
        ui.uploadArea.addEventListener('click', () => ui.inputGambar.click());
        ui.uploadArea.addEventListener('dragover', event => {
            event.preventDefault();
            ui.uploadArea.classList.add('border-green-500');
        });
        ui.uploadArea.addEventListener('dragleave', () => {
            ui.uploadArea.classList.remove('border-green-500');
        });
        ui.uploadArea.addEventListener('drop', event => {
            event.preventDefault();
            ui.uploadArea.classList.remove('border-green-500');
            const file = event.dataTransfer?.files?.[0];
            processFile(file, ui);
        });
    }

    if (ui.inputGambar) {
        ui.inputGambar.addEventListener('change', event => {
            const file = event.target.files?.[0];
            processFile(file, ui);
        });
    }
}

function attachPredictHandlers(ui) {
    if (ui.predictButton) {
        ui.predictButton.addEventListener('click', () => handlePrediction(ui));
    }
    if (ui.resetButton) {
        ui.resetButton.addEventListener('click', () => resetInterface(ui));
    }
}

async function loadModel(predictButton) {
    if (!predictButton) {
        return;
    }
    const defaultLabel = predictButton.textContent;
    predictButton.disabled = true;
    predictButton.textContent = 'Memuat model...';

    try {
        riceModel = await tf.loadGraphModel(MODEL_PATH);
        predictButton.disabled = false;
        predictButton.textContent = defaultLabel;
    } catch (error) {
        console.error('Gagal memuat model', error);
        predictButton.textContent = 'Model gagal dimuat';
    }
}

function processFile(file, ui) {
    if (!file) {
        return;
    }
    if (!file.type.startsWith('image/')) {
        alert('Silakan pilih file gambar (PNG/JPG/WEBP).');
        return;
    }

    const reader = new FileReader();
    reader.onload = event => {
        if (!ui.imagePreview) {
            return;
        }
        ui.imagePreview.onload = () => {
            ui.imagePreview.onload = null;
            ui.previewColumn?.classList.remove('hidden');
            if (ui.fileInfo) {
                ui.fileInfo.textContent = `${file.name} (${formatBytes(file.size)})`;
            }
        };
        ui.imagePreview.src = event.target?.result || '#';
    };
    reader.readAsDataURL(file);
}

async function handlePrediction(ui) {
    if (!riceModel) {
        alert('Model belum siap. Muat ulang halaman setelah model siap.');
        return;
    }
    if (!ui.imagePreview || !ui.imagePreview.src || ui.imagePreview.src === '#') {
        alert('Silakan pilih gambar daun padi terlebih dahulu.');
        ui.inputGambar?.click();
        return;
    }

    showElement(ui.resultsSection);
    showElement(ui.loadingSpinner);
    hideElement(ui.resultsContent);

    if (ui.resultImage) {
        ui.resultImage.src = ui.imagePreview.src;
    }

    if (ui.diseaseName) {
        ui.diseaseName.textContent = 'Menganalisis gambar...';
    }
    if (ui.diseaseDescription) {
        ui.diseaseDescription.textContent = 'Menjalankan model TensorFlow.js...';
    }
    if (ui.diseaseTreatment) {
        ui.diseaseTreatment.textContent = 'Menunggu hasil prediksi...';
    }
    if (ui.confidenceBar) {
        ui.confidenceBar.style.width = '0%';
    }
    if (ui.confidenceValue) {
        ui.confidenceValue.textContent = '0%';
    }

    try {
        const { namaKelas, skorPersen } = await runPrediction(ui.imagePreview);

        if (ui.diseaseName) {
            ui.diseaseName.textContent = namaKelas;
        }
        if (ui.confidenceBar) {
            ui.confidenceBar.style.width = `${skorPersen}%`;
        }
        if (ui.confidenceValue) {
            ui.confidenceValue.textContent = `${skorPersen}%`;
        }

        hideElement(ui.loadingSpinner);
        showElement(ui.resultsContent);

        requestAIExplanation(
            namaKelas,
            Number(skorPersen),
            ui.diseaseDescription,
            ui.diseaseTreatment
        );
    } catch (error) {
        console.error('Error saat prediksi', error);
        hideElement(ui.loadingSpinner);
        showElement(ui.resultsContent);

        if (ui.diseaseName) {
            ui.diseaseName.textContent = 'Prediksi gagal';
        }
        if (ui.diseaseDescription) {
            ui.diseaseDescription.textContent = 'Terjadi kesalahan saat menjalankan model.';
        }
        if (ui.diseaseTreatment) {
            ui.diseaseTreatment.textContent = 'Periksa console untuk detail dan coba ulangi.';
        }
    }
}

async function runPrediction(imageElement) {
    if (!riceModel) {
        throw new Error('Model belum siap.');
    }

    const prediction = tf.tidy(() => {
        const inputTensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255)
            .expandDims();
        return riceModel.execute(inputTensor);
    });

    const probabilities = await prediction.data();
    const argMaxTensor = prediction.argMax(1);
    const indexPrediksi = argMaxTensor.dataSync()[0];
    argMaxTensor.dispose();
    prediction.dispose();

    const namaKelas = NAMA_KELAS[indexPrediksi] || `Label ${indexPrediksi}`;
    const skorPersen = (Math.max(0, Math.min(1, probabilities[indexPrediksi] ?? 0)) * 100).toFixed(2);
    return { namaKelas, skorPersen };
}

async function requestAIExplanation(label, confidencePercent, descriptionEl, treatmentEl) {
    if (!descriptionEl || !treatmentEl) {
        return;
    }

    descriptionEl.textContent = 'Meminta penjelasan AI dari backend...';
    treatmentEl.textContent = 'Mengambil rekomendasi penanganan...';

    try {
        const response = await fetch(EXPLAIN_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ label, confidence: confidencePercent })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const explanation = (data.explanation || '').trim();

        if (!explanation) {
            descriptionEl.textContent = 'AI belum memiliki penjelasan tambahan.';
            treatmentEl.textContent = 'Gunakan panduan agronom lokal untuk tindakan lanjutan.';
            return;
        }

        const sections = explanation.split(/\n+/).filter(Boolean);
        descriptionEl.textContent = sections.shift() || explanation;
        treatmentEl.textContent = sections.length
            ? sections.join(' ')
            : 'Ikuti rekomendasi umum pakar agronomi sesuai gejala di atas.';
    } catch (error) {
        console.error('Gagal mengambil penjelasan AI', error);
        descriptionEl.textContent = 'Tidak dapat mengambil penjelasan AI. Pastikan backend berjalan.';
        treatmentEl.textContent = 'Backend Gemini belum merespons. Coba lagi setelah koneksi stabil.';
    }
}

function resetInterface(ui) {
    if (ui.inputGambar) {
        ui.inputGambar.value = '';
    }
    if (ui.imagePreview) {
        ui.imagePreview.src = '#';
    }
    if (ui.fileInfo) {
        ui.fileInfo.textContent = '';
    }
    ui.previewColumn?.classList.add('hidden');
    hideElement(ui.resultsSection);
    hideElement(ui.loadingSpinner);
    hideElement(ui.resultsContent);
}

function showElement(element) {
    if (element) {
        element.classList.remove('hidden');
    }
}

function hideElement(element) {
    if (element) {
        element.classList.add('hidden');
    }
}

function formatBytes(bytes) {
    if (!bytes) {
        return '0 KB';
    }
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }

    return `${unitIndex === 0 ? Math.round(value) : value.toFixed(1)} ${units[unitIndex]}`;
}

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabUpload = document.getElementById('tabUpload');
    const tabCamera = document.getElementById('tabCamera');
    const contentUpload = document.getElementById('contentUpload');
    const contentCamera = document.getElementById('contentCamera');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewBar = document.getElementById('previewBar');
    const selectedPreview = document.getElementById('selectedPreview');
    const previewName = document.getElementById('previewName');
    const previewSize = document.getElementById('previewSize');
    const btnRunScreening = document.getElementById('btnRunScreening');

    const webcam = document.getElementById('webcam');
    const btnCapture = document.getElementById('btnCapture');
    const btnSwitchCamera = document.getElementById('btnSwitchCamera');
    const captureCanvas = document.getElementById('captureCanvas');

    const progressCard = document.getElementById('progressCard');
    const progressStageTitle = document.getElementById('progressStageTitle');
    const progressStageSubtitle = document.getElementById('progressStageSubtitle');
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');
    const btnCloseError = document.getElementById('btnCloseError');

    const resultsContainer = document.getElementById('resultsContainer');
    const triageBadge = document.getElementById('triageBadge');
    const triageIcon = document.getElementById('triageIcon');
    const triageTitle = document.getElementById('triageTitle');
    const triageDesc = document.getElementById('triageDesc');
    const triageRiskValue = document.getElementById('triageRiskValue');
    const cvhiScore = document.getElementById('cvhiScore');
    const cvhiCaption = document.getElementById('cvhiCaption');

    const valTVL = document.getElementById('valTVL');
    const valMBA = document.getElementById('valMBA');
    const valFD = document.getElementById('valFD');
    const valLAC = document.getElementById('valLAC');

    const activeMaskImg = document.getElementById('activeMaskImg');
    const maskTabBtns = document.querySelectorAll('.mask-tab-btn');
    const btnResetAll = document.getElementById('btnResetAll');

    let currentFile = null;
    let currentB64 = null;
    let cameraStream = null;
    let facingMode = 'environment';
    let storedVisualizations = {};

    // 1. Tab Switching
    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('active');
        tabCamera.classList.remove('active');
        contentUpload.classList.add('active');
        contentCamera.classList.remove('active');
        stopCamera();
    });

    tabCamera.addEventListener('click', () => {
        tabCamera.classList.add('active');
        tabUpload.classList.remove('active');
        contentCamera.classList.add('active');
        contentUpload.classList.remove('active');
        startCamera();
    });

    // 2. Dropzone & File Input
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--accent-blue)';
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'var(--border-color)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--border-color)';
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;
        currentB64 = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            selectedPreview.src = e.target.result;
            previewName.textContent = file.name;
            previewSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
            previewBar.style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }

    // 3. Camera Operations
    async function startCamera() {
        try {
            if (cameraStream) stopCamera();
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } },
                audio: false
            });
            webcam.srcObject = cameraStream;
        } catch (err) {
            alert("Could not access camera: " + err.message);
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
    }

    btnSwitchCamera.addEventListener('click', () => {
        facingMode = (facingMode === 'environment') ? 'user' : 'environment';
        startCamera();
    });

    btnCapture.addEventListener('click', () => {
        if (!webcam.videoWidth) return;
        captureCanvas.width = webcam.videoWidth;
        captureCanvas.height = webcam.videoHeight;
        const ctx = captureCanvas.getContext('2d');
        ctx.drawImage(webcam, 0, 0, captureCanvas.width, captureCanvas.height);

        currentB64 = captureCanvas.toDataURL('image/jpeg', 0.95);
        currentFile = null;

        selectedPreview.src = currentB64;
        previewName.textContent = "Live Camera Capture";
        previewSize.textContent = "Raw Photo";
        previewBar.style.display = 'flex';

        // Scroll to preview bar
        previewBar.scrollIntoView({ behavior: 'smooth' });
    });

    // 4. Run Screening Pipeline
    btnRunScreening.addEventListener('click', async () => {
        if (!currentFile && !currentB64) return;

        hideError();
        resultsContainer.style.display = 'none';
        progressCard.style.display = 'block';

        const stages = [
            { step: 'step1', title: 'Stage 1: Image Quality', sub: 'Checking blur, exposure, contrast, and specular glare...' },
            { step: 'step2', title: 'Stage 2: Preprocessing', sub: 'Applying LAB CLAHE and green-channel inversion...' },
            { step: 'step3', title: 'Stage 3: Sclera ROI', sub: 'Executing GhostNet-U-Net ROI segmentation...' },
            { step: 'step4', title: 'Stage 4: Vessels', sub: 'Inferring microvascular network via GhostNet...' },
            { step: 'step5', title: 'Stage 5: Biomarkers', sub: 'Extracting TVL, MBA, LAC, and FD...' },
            { step: 'step6', title: 'Stage 6: Triage', sub: 'Running XGBoost model & CVHI calculation...' },
        ];

        let stageIdx = 0;
        const stageInterval = setInterval(() => {
            if (stageIdx < stages.length) {
                const s = stages[stageIdx];
                progressStageTitle.textContent = s.title;
                progressStageSubtitle.textContent = s.sub;
                document.querySelectorAll('.step').forEach(el => el.classList.remove('active'));
                document.getElementById(s.step).classList.add('active');
                stageIdx++;
            }
        }, 500);

        try {
            let res;
            if (currentFile) {
                const formData = new FormData();
                formData.append('file', currentFile);
                res = await fetch('/api/screen', { method: 'POST', body: formData });
            } else {
                res = await fetch('/api/screen', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_base64: currentB64 })
                });
            }

            clearInterval(stageInterval);
            progressCard.style.display = 'none';

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Inference failed on server.");
            }

            const data = await res.json();

            if (!data.success) {
                showError(data.error || "Quality check failed.");
                return;
            }

            displayResults(data);

        } catch (err) {
            clearInterval(stageInterval);
            progressCard.style.display = 'none';
            showError(err.message);
        }
    });

    // 5. Display Results
    function displayResults(data) {
        const triage = data.triage;
        const bio = data.biomarkers;
        storedVisualizations = data.visualizations || {};

        const isRisk = (triage.risk_level === "DIABETES_RISK");

        if (isRisk) {
            triageBadge.className = "triage-badge risk";
            triageIcon.textContent = "!";
            triageTitle.textContent = "Diabetes Risk Detected";
            triageDesc.textContent = "Microvascular changes (capillary dropout, tortuosity shift) suggest clinical referral.";
        } else {
            triageBadge.className = "triage-badge healthy";
            triageIcon.textContent = "✓";
            triageTitle.textContent = "Healthy Vasculature";
            triageDesc.textContent = "Vascular density and branching architecture appear within normal reference limits.";
        }

        triageRiskValue.textContent = `${(triage.probability * 100).toFixed(1)}%`;
        cvhiScore.textContent = `${triage.cvhi_score.toFixed(1)}`;

        if (triage.cvhi_score >= 70) {
            cvhiCaption.textContent = "High vascular resilience";
        } else if (triage.cvhi_score >= 40) {
            cvhiCaption.textContent = "Moderate microvascular alterations";
        } else {
            cvhiCaption.textContent = "Significant microvascular rarefaction";
        }

        valTVL.textContent = `${Math.round(bio.tvl)} px`;
        valMBA.textContent = `${bio.mba.toFixed(1)}°`;
        valFD.textContent = `${bio.fd.toFixed(3)}`;
        valLAC.textContent = `${bio.lac.toFixed(2)}`;

        // Set default mask
        if (storedVisualizations.vessel_overlay) {
            activeMaskImg.src = storedVisualizations.vessel_overlay;
        }

        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // 6. Mask Tabs
    maskTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            maskTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const target = btn.getAttribute('data-target');
            if (storedVisualizations[target]) {
                activeMaskImg.src = storedVisualizations[target];
            }
        });
    });

    // 7. Reset All
    btnResetAll.addEventListener('click', () => {
        resultsContainer.style.display = 'none';
        previewBar.style.display = 'none';
        currentFile = null;
        currentB64 = null;
        fileInput.value = '';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Error Banner Helpers
    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.style.display = 'flex';
        errorBanner.scrollIntoView({ behavior: 'smooth' });
    }

    function hideError() {
        errorBanner.style.display = 'none';
    }

    btnCloseError.addEventListener('click', hideError);
});

(() => {
    'use strict';

    const byId = id => document.getElementById(id);
    const setText = (id, value) => {
        const element = byId(id);
        if (element) element.textContent = String(value);
    };

    const openCamBtn = byId('openCam');
    const closeCamBtn = byId('closeCam');
    const startLiveBtn = byId('startLiveBtn');
    const stopLiveBtn = byId('stopLiveBtn');
    const video = byId('video');
    const canvas = byId('canvas');
    const faceOverlay = byId('faceOverlay');
    const cameraStage = byId('cameraStage');
    const cameraBadge = byId('cameraBadge');
    const cameraPlaceholder = byId('cameraPlaceholder');
    const detectionOverlay = byId('detectionOverlay');
    const detectionLog = byId('detectionLog');
    const refreshHistoryBtn = byId('refreshHistory');
    const menuBtn = byId('menuBtn');
    const sidebar = byId('sidebar');
    const sidebarBackdrop = byId('sidebarBackdrop');
    const detectionIntervalSelect = byId('detectionInterval');
    const cameraQualitySelect = byId('cameraQuality');
    const showFaceBoxesInput = byId('showFaceBoxes');
    const guestNameInput = byId('guestName');
    const saveGuestResultsInput = byId('saveGuestResults');
    const deleteGuestResultsBtn = byId('deleteGuestResults');

    const isGuest = document.body.dataset.guestMode === 'true';
    let currentUserName = document.body.dataset.userName?.trim() || (isGuest ? 'Guest' : 'User');
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';
    const allowedEmotions = new Set(['Happy', 'Angry', 'Sad', 'Neutral', 'Surprise']);
    const storageKey = isGuest ? 'emotion-recognition:guest:session' : `emotion-recognition:session:${currentUserName}`;
    const guestHistoryKey = 'emotion-recognition:guest:history';
    const guestNameKey = 'emotion-recognition:guest:display-name';
    const settingsKey = 'emotion-recognition:camera-settings';

    const sectionDetails = {
        dashboard: ['Dashboard', 'Overview of your emotion-detection session'],
        'live-detection': ['Live Detection', 'Real-time facial emotion recognition using CNN'],
        history: ['History', 'Review your emotion-detection records'],
        analytics: ['Analytics', 'View simple detection performance metrics'],
        settings: ['Settings', 'Adjust live camera and detection preferences']
    };

    let stream = null;
    let liveTimer = null;
    let liveActive = false;
    let liveBusy = false;
    let activeRequest = null;
    let sessionRecords = [];
    let storedRecords = [];
    let historyLoaded = false;
    let historyLoading = false;
    let analyticsLoading = false;
    let historyDirty = true;
    let analyticsDirty = true;
    let latestFaceResults = [];

    function requestHeaders(initialHeaders = {}) {
        const headers = new Headers(initialHeaders);
        if (csrfToken) headers.set('X-CSRF-Token', csrfToken);
        return headers;
    }

    function setResultMessage(message, state = '') {
        const resultMeta = byId('resultMeta');
        if (!resultMeta) return;
        resultMeta.textContent = message;
        resultMeta.classList.toggle('is-error', state === 'error');
        resultMeta.classList.toggle('is-working', state === 'working');
    }

    function updateDateTime() {
        const chip = byId('dateTimeChip');
        if (!chip) return;
        const now = new Date();
        chip.dateTime = now.toISOString();
        chip.textContent = new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
        }).format(now);
    }

    function validSectionId(candidate) {
        return Object.prototype.hasOwnProperty.call(sectionDetails, candidate) ? candidate : 'dashboard';
    }

    function sectionFromLocation() {
        try {
            return validSectionId(decodeURIComponent(window.location.hash.replace(/^#/, '')));
        } catch {
            return 'dashboard';
        }
    }

    function closeSidebar({ restoreFocus = false } = {}) {
        if (!sidebar || !menuBtn || !sidebarBackdrop) return;
        sidebar.classList.remove('open');
        sidebarBackdrop.hidden = true;
        document.body.classList.remove('menu-open');
        menuBtn.setAttribute('aria-expanded', 'false');
        menuBtn.setAttribute('aria-label', 'Open navigation menu');
        if (restoreFocus) menuBtn.focus();
    }

    function openSidebar() {
        if (!sidebar || !menuBtn || !sidebarBackdrop) return;
        sidebar.classList.add('open');
        sidebarBackdrop.hidden = false;
        document.body.classList.add('menu-open');
        menuBtn.setAttribute('aria-expanded', 'true');
        menuBtn.setAttribute('aria-label', 'Close navigation menu');
        sidebar.querySelector('.sidebar-link')?.focus();
    }

    function showSection(sectionId, { updateLocation = true } = {}) {
        const targetId = validSectionId(sectionId);

        document.querySelectorAll('.page-section').forEach(section => {
            const isTarget = section.id === targetId;
            section.classList.toggle('active', isTarget);
            section.hidden = !isTarget;
        });

        document.querySelectorAll('.sidebar-link').forEach(item => {
            const isTarget = item.dataset.section === targetId;
            item.classList.toggle('active', isTarget);
            if (isTarget) {
                item.setAttribute('aria-current', 'page');
            } else {
                item.removeAttribute('aria-current');
            }
        });

        const [title, subtitle] = sectionDetails[targetId];
        setText('pageTitle', title);
        setText('pageSubtitle', subtitle);
        document.title = `${title} — Emotion Recognition`;

        if (targetId !== 'live-detection' && liveActive) {
            stopDetection('Live detection paused when you left the camera page.');
        }

        if (targetId === 'history' && (!historyLoaded || historyDirty)) {
            loadHistory();
        }
        if (targetId === 'analytics' && analyticsDirty) {
            loadAnalytics();
        }

        if (updateLocation && window.location.hash !== `#${targetId}`) {
            window.history.pushState({ section: targetId }, '', `#${targetId}`);
        }

        closeSidebar();
    }

    document.querySelectorAll('.sidebar-link[data-section]').forEach(item => {
        item.addEventListener('click', () => showSection(item.dataset.section));
    });

    document.querySelectorAll('[data-open-section]').forEach(item => {
        item.addEventListener('click', () => showSection(item.dataset.openSection));
    });

    window.addEventListener('popstate', () => showSection(sectionFromLocation(), { updateLocation: false }));

    function canonicalEmotion(value) {
        const input = String(value || '').trim().toLowerCase();
        const emotion = [...allowedEmotions].find(label => label.toLowerCase() === input);
        return emotion || 'Unknown';
    }

    function sentenceForEmotion(emotion) {
        const label = canonicalEmotion(emotion);
        if (label === 'Unknown') return `${currentUserName}'s expression could not be classified`;
        const adjective = label === 'Surprise' ? 'Surprised' : label;
        return `${currentUserName} is ${adjective}`;
    }

    function confidencePercent(rawValue) {
        const numericValue = Number(rawValue);
        if (!Number.isFinite(numericValue)) return 0;
        const percentage = numericValue <= 1 ? numericValue * 100 : numericValue;
        return Math.max(0, Math.min(100, percentage));
    }

    function normalizedFaceCount(rawValue) {
        const numericValue = Number(rawValue);
        return Number.isFinite(numericValue) ? Math.max(0, Math.trunc(numericValue)) : 0;
    }

    function formatDetectionTime(value = new Date()) {
        const date = value instanceof Date ? value : new Date(value);
        const safeDate = Number.isNaN(date.getTime()) ? new Date() : date;
        return safeDate.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    function formatHistoryDateTime(value) {
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value || '—');
        return new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
        }).format(date);
    }

    function guestResultsArePersistent() {
        return isGuest && Boolean(saveGuestResultsInput?.checked);
    }

    function resultStorage() {
        return guestResultsArePersistent() ? localStorage : sessionStorage;
    }

    function resultStorageKey() {
        return guestResultsArePersistent() ? guestHistoryKey : storageKey;
    }

    function restoreSessionRecords() {
        try {
            const value = JSON.parse(resultStorage().getItem(resultStorageKey()) || '[]');
            if (!Array.isArray(value)) return [];
            return value
                .slice(0, 50)
                .filter(item => item && typeof item === 'object')
                .map(item => ({
                    emotion: canonicalEmotion(item.emotion),
                    confidence: confidencePercent(item.confidence),
                    faces: normalizedFaceCount(item.faces),
                    time: String(item.time || formatDetectionTime(item.detectedAt)),
                    detectedAt: item.detectedAt || ''
                }));
        } catch {
            return [];
        }
    }

    function persistSessionRecords() {
        try {
            resultStorage().setItem(resultStorageKey(), JSON.stringify(sessionRecords.slice(0, 50)));
        } catch {
            // The dashboard remains usable when browser storage is unavailable.
        }
    }

    function createCell(value, { strong = false } = {}) {
        const cell = document.createElement('td');
        if (strong) {
            const emphasis = document.createElement('strong');
            emphasis.textContent = String(value);
            cell.appendChild(emphasis);
        } else {
            cell.textContent = String(value);
        }
        return cell;
    }

    function createHistoryRow(record) {
        const row = document.createElement('tr');
        row.append(
            createCell(record.time),
            createCell(currentUserName),
            createCell(sentenceForEmotion(record.emotion), { strong: true }),
            createCell(`${record.confidence.toFixed(1)}%`),
            createCell(record.faces)
        );
        return row;
    }

    function renderHistory() {
        if (!detectionLog) return;
        detectionLog.replaceChildren();
        const recordsToRender = historyLoaded ? storedRecords : sessionRecords;

        if (!recordsToRender.length) {
            const row = document.createElement('tr');
            row.id = 'emptyLog';
            const cell = createCell(
                historyLoaded
                    ? 'No saved detection records yet.'
                    : 'No detections in this browser session yet.'
            );
            cell.colSpan = 5;
            cell.className = 'empty-cell';
            row.appendChild(cell);
            detectionLog.appendChild(row);
            return;
        }

        const fragment = document.createDocumentFragment();
        recordsToRender.slice(0, 50).forEach(record => fragment.appendChild(createHistoryRow(record)));
        detectionLog.appendChild(fragment);
    }

    function updateSessionCards() {
        const total = sessionRecords.length;
        const latest = sessionRecords[0];
        const confidenceAverage = total
            ? sessionRecords.reduce((sum, record) => sum + record.confidence, 0) / total
            : 0;

        setText('dashboardCount', total);

        if (!latest) {
            setText('dashboardEmotion', 'Waiting');
            setText('dashboardConfidence', '—');
            setText('dashboardSentence', 'No expression detected yet');
            setText('dashboardResultHint', 'Your latest result will appear here after live detection begins.');
            updateFallbackAnalytics(confidenceAverage);
            return;
        }

        setText('dashboardEmotion', latest.emotion);
        setText('dashboardConfidence', `${latest.confidence.toFixed(1)}%`);
        setText('dashboardSentence', sentenceForEmotion(latest.emotion));
        setText('dashboardResultHint', `Detected with ${latest.confidence.toFixed(1)}% confidence.`);
        updateFallbackAnalytics(confidenceAverage);
    }

    function updateFallbackAnalytics(confidenceAverage = 0) {
        const latest = sessionRecords[0];
        setText('analyticsTotal', sessionRecords.length);
        setText('analyticsAverage', sessionRecords.length ? `${confidenceAverage.toFixed(1)}%` : '—');
        setText('analyticsLatest', latest?.emotion || 'Waiting');
        setText('analyticsFaces', latest?.faces || 0);
    }

    function normalizeStoredRecord(item) {
        const detectedAt = item?.detected_at || '';
        return {
            emotion: canonicalEmotion(item?.emotion),
            confidence: confidencePercent(item?.confidence),
            faces: normalizedFaceCount(item?.faces_detected),
            time: formatHistoryDateTime(detectedAt),
            detectedAt
        };
    }

    async function loadHistory() {
        if (isGuest) {
            storedRecords = [...sessionRecords];
            historyLoaded = true;
            historyDirty = false;
            renderHistory();
            setText(
                'historyDescription',
                sessionRecords.length
                    ? `Showing ${sessionRecords.length} guest result${sessionRecords.length === 1 ? '' : 's'} from this browser.`
                    : 'No guest detection results are available in this browser yet.'
            );
            return;
        }
        if (historyLoading) return;
        historyLoading = true;
        if (refreshHistoryBtn) refreshHistoryBtn.disabled = true;
        setText('historyDescription', 'Loading your saved detection records…');

        try {
            const response = await fetch('/api/history', {
                headers: requestHeaders(),
                cache: 'no-store'
            });
            const payload = await parseJsonResponse(response);
            if (!response.ok || !payload?.ok || !Array.isArray(payload.records)) {
                throw new Error(payload?.error || 'Detection history could not be loaded.');
            }

            storedRecords = payload.records.map(normalizeStoredRecord);
            historyLoaded = true;
            historyDirty = false;
            renderHistory();
            setText(
                'historyDescription',
                storedRecords.length
                    ? `Showing ${storedRecords.length} saved detection record${storedRecords.length === 1 ? '' : 's'} for your account.`
                    : 'No saved detection records are available for your account yet.'
            );
        } catch (error) {
            console.error('History loading error:', error);
            renderHistory();
            setText(
                'historyDescription',
                `${error?.message || 'Detection history could not be loaded.'} Showing this browser session instead.`
            );
        } finally {
            historyLoading = false;
            if (refreshHistoryBtn) refreshHistoryBtn.disabled = false;
        }
    }

    async function loadAnalytics() {
        if (isGuest) {
            const total = sessionRecords.length;
            const average = total
                ? sessionRecords.reduce((sum, record) => sum + record.confidence, 0) / total
                : 0;
            updateFallbackAnalytics(average);
            setText(
                'analyticsDescription',
                guestResultsArePersistent()
                    ? 'These metrics summarize guest results saved only in this browser.'
                    : 'These metrics summarize guest results from this browser session only.'
            );
            analyticsDirty = false;
            return;
        }
        if (analyticsLoading) return;
        analyticsLoading = true;
        setText('analyticsDescription', 'Loading analytics from your saved detection records…');

        try {
            const response = await fetch('/api/analytics', {
                headers: requestHeaders(),
                cache: 'no-store'
            });
            const payload = await parseJsonResponse(response);
            if (!response.ok || !payload?.ok) {
                throw new Error(payload?.error || 'Analytics could not be loaded.');
            }

            const total = Math.max(0, Number(payload.total) || 0);
            const latest = payload.latest || null;
            setText('analyticsTotal', total);
            setText(
                'analyticsAverage',
                total ? `${confidencePercent(payload.average_confidence).toFixed(1)}%` : '—'
            );
            setText('analyticsLatest', latest ? canonicalEmotion(latest.emotion) : 'Waiting');
            setText('analyticsFaces', latest ? normalizedFaceCount(latest.faces_detected) : 0);
            setText(
                'analyticsDescription',
                total
                    ? `These metrics summarize ${total} saved detection record${total === 1 ? '' : 's'} for your account.`
                    : 'Your saved detection analytics will appear after your first successful detection.'
            );
            analyticsDirty = false;
        } catch (error) {
            console.error('Analytics loading error:', error);
            const total = sessionRecords.length;
            const average = total
                ? sessionRecords.reduce((sum, record) => sum + record.confidence, 0) / total
                : 0;
            updateFallbackAnalytics(average);
            setText(
                'analyticsDescription',
                `${error?.message || 'Saved analytics could not be loaded.'} Showing this browser session instead.`
            );
        } finally {
            analyticsLoading = false;
        }
    }

    function addDetectionRecord(record) {
        sessionRecords.unshift(record);
        sessionRecords = sessionRecords.slice(0, 50);
        historyDirty = true;
        analyticsDirty = true;
        persistSessionRecords();
        if (isGuest) {
            storedRecords = [...sessionRecords];
            historyLoaded = true;
            renderHistory();
        } else if (!historyLoaded) {
            renderHistory();
        }
        updateSessionCards();
    }

    function updateResult(data) {
        const emotion = canonicalEmotion(data?.emotion);
        const confidence = confidencePercent(data?.confidence);
        const faces = normalizedFaceCount(data?.faces_detected);
        const detectedAt = data?.detected_at || new Date().toISOString();
        const time = formatDetectionTime(detectedAt);

        setText('resultLabel', emotion);
        setText('confidenceValue', `${confidence.toFixed(1)}%`);
        setText('resultFaces', faces);
        setText('resultTime', time);
        setText('namedEmotionResult', sentenceForEmotion(emotion));
        drawFaceOverlay(data?.faces);

        const framesUsed = normalizedFaceCount(data?.frames_used);
        setResultMessage(
            framesUsed
                ? `Detection completed using ${framesUsed} captured frame${framesUsed === 1 ? '' : 's'}.`
                : 'Detection completed successfully.'
        );

        addDetectionRecord({ emotion, confidence, faces, time, detectedAt });
    }

    function setGuestName(value, { persist = true } = {}) {
        if (!isGuest) return;
        const sanitized = String(value || '').trim().replace(/\s+/g, ' ').slice(0, 100);
        currentUserName = sanitized || 'Guest';
        document.body.dataset.userName = currentUserName;
        setText('userChipName', currentUserName);
        setText('greetingName', `Hello, ${currentUserName}`);
        setText('userAvatar', currentUserName.charAt(0).toUpperCase());
        if (guestNameInput && guestNameInput.value !== sanitized) guestNameInput.value = sanitized;
        if (persist) {
            try {
                if (sanitized) sessionStorage.setItem(guestNameKey, sanitized);
                else sessionStorage.removeItem(guestNameKey);
            } catch {
                // The dashboard remains usable when browser storage is unavailable.
            }
        }
    }

    function clearFaceOverlay() {
        latestFaceResults = [];
        if (!faceOverlay) return;
        const context = faceOverlay.getContext('2d');
        context?.clearRect(0, 0, faceOverlay.width, faceOverlay.height);
        faceOverlay.hidden = true;
    }

    function drawFaceOverlay(faces) {
        latestFaceResults = Array.isArray(faces) ? faces : [];
        if (!faceOverlay || !video || !cameraStage || !showFaceBoxesInput?.checked || !latestFaceResults.length) {
            if (faceOverlay) {
                const context = faceOverlay.getContext('2d');
                context?.clearRect(0, 0, faceOverlay.width, faceOverlay.height);
                faceOverlay.hidden = true;
            }
            return;
        }

        const sourceWidth = video.videoWidth;
        const sourceHeight = video.videoHeight;
        const stageRect = cameraStage.getBoundingClientRect();
        const stageWidth = Math.round(stageRect.width);
        const stageHeight = Math.round(stageRect.height);
        if (!sourceWidth || !sourceHeight || !stageWidth || !stageHeight) return;

        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        faceOverlay.width = Math.round(stageWidth * pixelRatio);
        faceOverlay.height = Math.round(stageHeight * pixelRatio);
        faceOverlay.style.width = `${stageWidth}px`;
        faceOverlay.style.height = `${stageHeight}px`;
        faceOverlay.hidden = false;

        const context = faceOverlay.getContext('2d');
        if (!context) return;
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        context.clearRect(0, 0, stageWidth, stageHeight);

        // The preview uses object-fit: cover and is mirrored, so map the server's
        // source-frame coordinates to the visible camera stage before drawing.
        const scale = Math.max(stageWidth / sourceWidth, stageHeight / sourceHeight);
        const renderedWidth = sourceWidth * scale;
        const renderedHeight = sourceHeight * scale;
        const offsetX = (stageWidth - renderedWidth) / 2;
        const offsetY = (stageHeight - renderedHeight) / 2;

        context.lineWidth = 2;
        context.font = '600 12px ui-sans-serif, system-ui, sans-serif';
        context.textBaseline = 'middle';

        latestFaceResults.forEach(face => {
            const box = face?.box || {};
            const x = Number(box.x);
            const y = Number(box.y);
            const width = Number(box.w);
            const height = Number(box.h);
            if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return;

            const displayWidth = width * scale;
            const displayHeight = height * scale;
            const displayX = stageWidth - (offsetX + (x + width) * scale);
            const displayY = offsetY + y * scale;
            if (displayX + displayWidth < 0 || displayX > stageWidth || displayY + displayHeight < 0 || displayY > stageHeight) return;

            const emotion = canonicalEmotion(face?.emotion);
            const confidence = confidencePercent(face?.confidence);
            const label = `${emotion} ${confidence.toFixed(1)}%`;
            const labelWidth = Math.ceil(context.measureText(label).width) + 14;
            const labelHeight = 23;
            const labelX = Math.max(0, Math.min(displayX, stageWidth - labelWidth));
            const labelY = displayY >= labelHeight + 4 ? displayY - labelHeight - 4 : Math.min(stageHeight - labelHeight, displayY + displayHeight + 4);

            context.strokeStyle = '#34d399';
            context.strokeRect(displayX, displayY, displayWidth, displayHeight);
            context.fillStyle = 'rgba(5, 29, 44, 0.92)';
            context.fillRect(labelX, labelY, labelWidth, labelHeight);
            context.fillStyle = '#f8fafc';
            context.fillText(label, labelX + 7, labelY + (labelHeight / 2));
        });
    }

    function selectedCameraConstraints() {
        const quality = cameraQualitySelect?.value || '1280x720';
        const [width, height] = quality.split('x').map(Number);
        return {
            facingMode: 'user',
            width: { ideal: width || 1280 },
            height: { ideal: height || 720 }
        };
    }

    function abortableDelay(milliseconds, signal) {
        return new Promise((resolve, reject) => {
            const timer = window.setTimeout(resolve, milliseconds);
            signal.addEventListener('abort', () => {
                window.clearTimeout(timer);
                reject(new DOMException('Request cancelled', 'AbortError'));
            }, { once: true });
        });
    }

    async function captureFrames(signal) {
        if (!video?.videoWidth || !video?.videoHeight) {
            throw new Error('The camera is still starting. Please try again.');
        }

        const context = canvas?.getContext('2d');
        if (!canvas || !context) throw new Error('Camera capture is not supported by this browser.');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const form = new FormData();

        for (let index = 0; index < 5; index += 1) {
            if (signal.aborted) throw new DOMException('Request cancelled', 'AbortError');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.86));
            if (blob) form.append('file', blob, `live_frame_${index}.jpg`);
            if (index < 4) await abortableDelay(70, signal);
        }

        if (!form.has('file')) throw new Error('The browser could not capture a camera frame.');
        return form;
    }

    function setDetectionBusy(isBusy) {
        liveBusy = isBusy;
        if (cameraStage) cameraStage.setAttribute('aria-busy', String(isBusy));
        if (detectionOverlay) detectionOverlay.hidden = !isBusy;
    }

    async function parseJsonResponse(response) {
        const contentType = response.headers.get('content-type') || '';
        if (!contentType.includes('application/json')) {
            if (response.redirected || response.status === 401 || response.status === 403) {
                throw new Error('Your session has expired. Please log in again.');
            }
            throw new Error(`The server returned an unexpected response (${response.status}).`);
        }
        return response.json();
    }

    async function runLiveDetectionOnce() {
        if (!stream || !liveActive || liveBusy) return;

        const controller = new AbortController();
        activeRequest = controller;
        setDetectionBusy(true);
        setResultMessage('Analyzing your facial expression…', 'working');

        try {
            const form = await captureFrames(controller.signal);
            const response = await fetch('/predict', {
                method: 'POST',
                headers: requestHeaders(),
                body: form,
                signal: controller.signal,
                cache: 'no-store'
            });
            const data = await parseJsonResponse(response);
            if (!response.ok) throw new Error(data?.error || `Detection failed (${response.status}).`);
            if (liveActive) updateResult(data);
        } catch (error) {
            if (error?.name !== 'AbortError') {
                console.error('Live detection error:', error);
                setResultMessage(error?.message || 'Could not complete detection.', 'error');
            }
        } finally {
            if (activeRequest === controller) {
                activeRequest = null;
                setDetectionBusy(false);
            }
        }
    }

    async function detectionLoop() {
        if (!liveActive) return;
        await runLiveDetectionOnce();
        if (!liveActive) return;
        const delay = Math.max(1000, Number(detectionIntervalSelect?.value) || 1500);
        liveTimer = window.setTimeout(detectionLoop, delay);
    }

    function cameraErrorMessage(error) {
        switch (error?.name) {
            case 'NotAllowedError':
                return 'Camera access was denied. Allow camera permission in your browser and try again.';
            case 'NotFoundError':
                return 'No camera was found on this device.';
            case 'NotReadableError':
                return 'The camera is already in use by another application.';
            case 'OverconstrainedError':
                return 'The selected camera quality is not supported. Choose Standard quality and try again.';
            default:
                return 'The camera could not be opened. Check browser permissions and try again.';
        }
    }

    async function openCamera() {
        if (!navigator.mediaDevices?.getUserMedia) {
            setResultMessage('Camera access is not supported in this browser.', 'error');
            return;
        }

        if (openCamBtn) openCamBtn.disabled = true;
        setResultMessage('Requesting camera access…', 'working');

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: selectedCameraConstraints(),
                audio: false
            });
            video.srcObject = stream;
            await video.play();
            video.style.display = 'block';
            clearFaceOverlay();
            if (cameraPlaceholder) cameraPlaceholder.hidden = true;
            cameraBadge?.classList.remove('is-live');
            setText('cameraBadge', 'Camera Ready');
            setText('dashboardCamera', 'Ready');
            if (closeCamBtn) closeCamBtn.disabled = false;
            if (startLiveBtn) startLiveBtn.disabled = false;
            setResultMessage('Camera ready. Position your face clearly inside the frame.');
            setText('namedEmotionResult', `${currentUserName} is ready for detection`);
            startDetection();
        } catch (error) {
            console.error('Camera error:', error);
            stream?.getTracks().forEach(track => track.stop());
            stream = null;
            setResultMessage(cameraErrorMessage(error), 'error');
            if (openCamBtn) openCamBtn.disabled = false;
        }
    }

    function startDetection() {
        if (!stream || liveActive) return;
        liveActive = true;
        if (startLiveBtn) startLiveBtn.disabled = true;
        if (stopLiveBtn) stopLiveBtn.disabled = false;
        cameraBadge?.classList.add('is-live');
        setText('cameraBadge', 'Live Detection');
        setText('dashboardCamera', 'Detecting');
        setResultMessage('Live detection started. Analyzing your facial expression…', 'working');
        detectionLoop();
    }

    function stopDetection(message = 'Live detection stopped. Your camera remains open.') {
        liveActive = false;
        if (liveTimer) window.clearTimeout(liveTimer);
        liveTimer = null;
        activeRequest?.abort();
        activeRequest = null;
        setDetectionBusy(false);

        if (stream) {
            if (startLiveBtn) startLiveBtn.disabled = false;
            if (stopLiveBtn) stopLiveBtn.disabled = true;
            cameraBadge?.classList.remove('is-live');
            setText('cameraBadge', 'Camera Ready');
            setText('dashboardCamera', 'Paused');
            setResultMessage(message);
        }
    }

    function closeCamera() {
        if (!stream) return;
        stopDetection();
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        if (video) {
            video.srcObject = null;
            video.style.display = 'none';
        }
        clearFaceOverlay();
        if (cameraPlaceholder) cameraPlaceholder.hidden = false;
        cameraBadge?.classList.remove('is-live');
        setText('cameraBadge', 'Camera Off');
        setText('dashboardCamera', 'Inactive');
        if (openCamBtn) openCamBtn.disabled = false;
        if (closeCamBtn) closeCamBtn.disabled = true;
        if (startLiveBtn) startLiveBtn.disabled = true;
        if (stopLiveBtn) stopLiveBtn.disabled = true;
        setResultMessage('Open the camera to begin live detection.');
        setText('namedEmotionResult', `${currentUserName} is waiting for detection`);
    }

    function restoreSettings() {
        try {
            const settings = JSON.parse(localStorage.getItem(settingsKey) || '{}');
            const intervals = [...(detectionIntervalSelect?.options || [])].map(option => option.value);
            const qualities = [...(cameraQualitySelect?.options || [])].map(option => option.value);
            if (intervals.includes(String(settings.interval))) detectionIntervalSelect.value = String(settings.interval);
            if (qualities.includes(String(settings.quality))) cameraQualitySelect.value = String(settings.quality);
            if (typeof settings.showFaceBoxes === 'boolean' && showFaceBoxesInput) {
                showFaceBoxesInput.checked = settings.showFaceBoxes;
            }
            if (isGuest && typeof settings.saveGuestResults === 'boolean' && saveGuestResultsInput) {
                saveGuestResultsInput.checked = settings.saveGuestResults;
            }
            if (isGuest) {
                if (document.body.dataset.guestFresh === 'true') {
                    sessionStorage.removeItem(guestNameKey);
                }
                setGuestName(sessionStorage.getItem(guestNameKey) || '', { persist: false });
            }
        } catch {
            // Default settings remain selected.
        }
    }

    function saveSettings() {
        try {
            localStorage.setItem(settingsKey, JSON.stringify({
                interval: detectionIntervalSelect?.value,
                quality: cameraQualitySelect?.value,
                showFaceBoxes: Boolean(showFaceBoxesInput?.checked),
                saveGuestResults: Boolean(saveGuestResultsInput?.checked)
            }));
        } catch {
            // The setting still applies until the page is refreshed.
        }
    }

    openCamBtn?.addEventListener('click', openCamera);
    startLiveBtn?.addEventListener('click', startDetection);
    stopLiveBtn?.addEventListener('click', () => stopDetection());
    closeCamBtn?.addEventListener('click', closeCamera);

    refreshHistoryBtn?.addEventListener('click', () => {
        historyDirty = true;
        loadHistory();
    });

    menuBtn?.addEventListener('click', () => {
        if (sidebar?.classList.contains('open')) closeSidebar({ restoreFocus: true });
        else openSidebar();
    });

    sidebarBackdrop?.addEventListener('click', () => closeSidebar({ restoreFocus: true }));

    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && sidebar?.classList.contains('open')) {
            closeSidebar({ restoreFocus: true });
        }
    });

    detectionIntervalSelect?.addEventListener('change', () => {
        saveSettings();
        setText('settingsStatus', 'Detection interval saved. It applies to the next detection request.');
    });

    cameraQualitySelect?.addEventListener('change', () => {
        saveSettings();
        setText(
            'settingsStatus',
            stream
                ? 'Camera quality saved. Close and reopen the camera to apply it.'
                : 'Camera quality saved. It will apply when the camera opens.'
        );
    });

    showFaceBoxesInput?.addEventListener('change', () => {
        saveSettings();
        drawFaceOverlay(latestFaceResults);
        setText(
            'settingsStatus',
            showFaceBoxesInput.checked
                ? 'Face boxes and labels are enabled for this browser. This preview does not change saved records.'
                : 'Face boxes and labels are hidden. This preview does not change saved records.'
        );
    });

    guestNameInput?.addEventListener('input', () => setGuestName(guestNameInput.value));

    saveGuestResultsInput?.addEventListener('change', () => {
        saveSettings();
        if (saveGuestResultsInput.checked) {
            persistSessionRecords();
            setText('settingsStatus', 'Guest results will be saved in this browser for your next visit. They are not stored in the system database.');
        } else {
            sessionStorage.setItem(storageKey, JSON.stringify(sessionRecords.slice(0, 50)));
            setText('settingsStatus', 'Guest results will remain only for this browser session. Use the delete button to remove any previously saved guest results.');
        }
    });

    deleteGuestResultsBtn?.addEventListener('click', () => {
        if (!window.confirm('Delete all guest detection results saved in this browser? This cannot be undone.')) return;
        try {
            localStorage.removeItem(guestHistoryKey);
            sessionStorage.removeItem(storageKey);
        } catch {
            // The in-memory records are still cleared below.
        }
        sessionRecords = [];
        storedRecords = [];
        historyLoaded = true;
        historyDirty = false;
        analyticsDirty = true;
        renderHistory();
        updateSessionCards();
        setText('settingsStatus', 'All guest detection results were deleted from this browser.');
    });

    window.addEventListener('resize', () => drawFaceOverlay(latestFaceResults));

    document.addEventListener('visibilitychange', () => {
        if (document.hidden && liveActive) {
            stopDetection('Live detection paused because this browser tab is no longer visible.');
        }
    });

    window.addEventListener('beforeunload', () => {
        activeRequest?.abort();
        stream?.getTracks().forEach(track => track.stop());
    });

    restoreSettings();
    sessionRecords = restoreSessionRecords();
    renderHistory();
    updateSessionCards();
    updateDateTime();
    window.setInterval(updateDateTime, 30000);
    showSection(sectionFromLocation(), { updateLocation: false });
    if (!window.location.hash || sectionFromLocation() !== window.location.hash.slice(1)) {
        window.history.replaceState({ section: 'dashboard' }, '', '#dashboard');
    }
})();

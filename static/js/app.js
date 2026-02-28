const state = {
    selectedImages: [],
    selectedModel: 'fusion',
    selectedType: 'general',
    isProcessing: false,
    chatHistory: [],
    currentImage: null  
};

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const startBtn = document.getElementById('startBtn');
const previewSection = document.getElementById('previewSection');
const previewGrid = document.getElementById('previewGrid');
const chatSection = document.getElementById('chatSection');
const chatContainer = document.getElementById('chatContainer');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsList = document.getElementById('resultsList');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingSpinner = document.getElementById('loadingSpinner');
const loadingProgressContainer = document.getElementById('loadingProgressContainer');
const loadingProgressBar = document.getElementById('loadingProgressBar');
const loadingText = document.getElementById('loadingText');

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUploadArea();
    initModelSelection();
    initTypeSelection();
    initStartButton();
    initChat();
    initSettings();
});

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageName = link.dataset.page;
            switchPage(pageName);

            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

function switchPage(pageName) {
    
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });

    const targetPage = document.getElementById(`page-${pageName}`);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    if (pageName === 'history') {
        loadHistory();
    }
}

function initUploadArea() {
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
}

function handleFiles(files) {
    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
        alert('请选择图片文件');
        return;
    }

    imageFiles.forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
            state.selectedImages.push({
                file: file,
                preview: e.target.result,
                id: Date.now() + Math.random()
            });
            updatePreview();
            updateStartButton();
        };
        reader.readAsDataURL(file);
    });
}

function updatePreview() {
    if (state.selectedImages.length === 0) {
        previewSection.style.display = 'none';
        return;
    }

    previewSection.style.display = 'block';
    previewGrid.innerHTML = state.selectedImages.map(img => `
        <div class="preview-item" data-id="${img.id}">
            <img src="${img.preview}" alt="预览">
            <button class="preview-remove" onclick="removeImage(${img.id})">×</button>
        </div>
    `).join('');
}

function removeImage(id) {
    state.selectedImages = state.selectedImages.filter(img => img.id !== id);
    updatePreview();
    updateStartButton();

    if (state.selectedImages.length === 0) {
        chatSection.style.display = 'none';
        disableChat();
    }
}

function initModelSelection() {
    const modelCards = document.querySelectorAll('.model-card-h');
    
    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            modelCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            state.selectedModel = card.dataset.model;
        });
    });
}

function initTypeSelection() {
    const typeCards = document.querySelectorAll('.type-card');
    
    typeCards.forEach(card => {
        card.addEventListener('click', () => {
            typeCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            state.selectedType = card.dataset.type;
        });
    });
}

function initStartButton() {
    startBtn.addEventListener('click', startRecognition);
}

function updateStartButton() {
    startBtn.disabled = state.selectedImages.length === 0 || state.isProcessing;
}

async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-status');
        const data = await response.json();
        return data.loaded;
    } catch (error) {
        console.error('检查模型状态失败:', error);
        return false;
    }
}

function showModelLoadingStatus() {
    loadingSpinner.style.display = 'block';
    loadingProgressContainer.style.display = 'block';
    loadingText.innerHTML = '正在加载模型权重...<br><span style="font-size: 14px; color: #64748b;">首次加载需要较长的时间（配置较低的机型大概需要等待10~15分钟），请耐心等待</span>';

    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90; 
        loadingProgressBar.style.width = progress + '%';
    }, 1000);
    
    return progressInterval;
}

function showRecognitionStatus() {
    loadingSpinner.style.display = 'block';
    loadingProgressContainer.style.display = 'none';
    loadingText.textContent = '正在识别中...';
    loadingProgressBar.style.width = '0%';
}

function showModelLoadedStatus() {
    loadingProgressBar.style.width = '100%';
    loadingText.textContent = '模型已在本地加载完成';
    loadingSpinner.style.display = 'none';
}

async function startRecognition() {
    if (state.selectedImages.length === 0) return;
    
    state.isProcessing = true;
    updateStartButton();
    loadingOverlay.style.display = 'flex';

    const isModelLoaded = await checkModelStatus();
    let progressInterval = null;
    
    if (!isModelLoaded) {
        
        progressInterval = showModelLoadingStatus();
    } else {
        
        showRecognitionStatus();
    }

    const results = [];

    try {
        for (const image of state.selectedImages) {
            try {
                const result = await recognizeImage(image);
                results.push({
                    image: image.preview,
                    ...result
                });
            } catch (error) {
                console.error('识别失败:', error);
                results.push({
                    image: image.preview,
                    model: state.selectedModel,
                    text: '识别失败: ' + error.message
                });
            }
        }

        if (progressInterval) {
            clearInterval(progressInterval);
        }

        if (!isModelLoaded) {
            showModelLoadedStatus();
            
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        console.log('准备显示结果，结果数量:', results.length);
        await displayResults(results);

        console.log('显示对话窗口');
        chatSection.style.display = 'block';
        console.log('chatSection display 设置为 block');
        enableChat();

        if (results.length > 0 && results[0].image) {
            state.currentImage = results[0].image;
            console.log('已保存当前图片用于对话');
        }

        const firstResult = results[0];
        console.log('第一个结果:', firstResult);
        if (firstResult && firstResult.text) {
            console.log('添加 AI 消息:', firstResult.text.substring(0, 50) + '...');
            addAIMessage(firstResult.text);
        } else {
            console.log('没有有效的第一个结果文本');
        }

        console.log('滚动到结果区域');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        console.error('处理结果时出错:', error);
    } finally {
        state.isProcessing = false;
        updateStartButton();
        loadingOverlay.style.display = 'none';
        loadingProgressBar.style.width = '0%';
        if (progressInterval) {
            clearInterval(progressInterval);
        }
    }
}

async function recognizeImage(image) {
    const formData = new FormData();
    formData.append('image', image.file);
    formData.append('model', state.selectedModel);
    formData.append('type', state.selectedType);

    const response = await fetch('/api/recognize', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || '识别失败');
    }

    return await response.json();
}

async function displayResults(results) {
    console.log('显示结果:', results);
    resultsSection.style.display = 'block';
    
    const modelNames = {
        'fusion': '融合模型',
        '300vglm': '300VGLM',
        '5000vglm': '5000VGLM'
    };

    const typeNames = {
        'general': '通用描述',
        'background': '背景分析',
        'detailed': '详细描述',
        'english': '英文描述'
    };

    resultsList.innerHTML = results.map((result, index) => `
        <div class="result-item">
            <img src="${result.image}" alt="结果${index + 1}" class="result-image">
            <div class="result-content">
                <div class="result-model">${modelNames[state.selectedModel] || state.selectedModel} · ${typeNames[state.selectedType] || state.selectedType}</div>
                <div class="result-text">${result.text || result.description || '无识别结果'}</div>
            </div>
        </div>
    `).join('');

    console.log('开始保存到历史记录，结果数量:', results.length);
    for (let i = 0; i < results.length; i++) {
        const result = results[i];
        console.log(`处理第 ${i + 1} 个结果:`, result);
        if (result.text) {
            console.log(`第 ${i + 1} 个结果有文本，调用 saveToHistory`);
            await saveToHistory(result);
        } else {
            console.log(`第 ${i + 1} 个结果没有文本，跳过`);
        }
    }
    console.log('历史记录保存完成');
}

function initChat() {
    chatSendBtn.addEventListener('click', sendChatMessage);
    
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

function enableChat() {
    chatInput.disabled = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
}

function disableChat() {
    chatInput.disabled = true;
    chatSendBtn.disabled = true;
}

async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addUserMessage(message);
    chatInput.value = '';

    chatSendBtn.disabled = true;

    const typingIndicator = addTypingIndicator();
    
    try {
        const requestBody = {
            message: message,
            model: state.selectedModel,
            history: state.chatHistory
        };

        if (state.currentImage) {
            requestBody.image = state.currentImage;
            console.log('发送对话请求，包含图片，图片长度:', state.currentImage.length);
        } else {
            console.log('发送对话请求，没有图片');
        }
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        removeTypingIndicator(typingIndicator);
        
        if (response.ok) {
            const data = await response.json();
            addAIMessage(data.response);

            state.chatHistory.push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.response }
            );
        } else {
            addAIMessage('抱歉，我暂时无法回答这个问题。');
        }
    } catch (error) {
        removeTypingIndicator(typingIndicator);
        console.error('对话失败:', error);
        addAIMessage('抱歉，发生了错误，请稍后再试。');
    }
    
    chatSendBtn.disabled = false;
}

function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message ai-message typing-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <circle cx="15.5" cy="8.5" r="1.5"/>
                <path d="M9 16c.5.3 1.2.5 2 .5s1.5-.2 2-.5"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function removeTypingIndicator(typingElement) {
    if (typingElement && typingElement.parentNode) {
        typingElement.parentNode.removeChild(typingElement);
    }
}

function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message user-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                <circle cx="12" cy="7" r="4"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(text)}</div>
            <div class="message-time">${getCurrentTime()}</div>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addAIMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message ai-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <circle cx="15.5" cy="8.5" r="1.5"/>
                <path d="M9 16c.5.3 1.2.5 2 .5s1.5-.2 2-.5"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(text)}</div>
            <div class="message-time">${getCurrentTime()}</div>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('vglm_history') || '[]');
    const historyEmpty = document.getElementById('historyEmpty');
    const historyList = document.getElementById('historyList');
    
    if (history.length === 0) {
        historyEmpty.style.display = 'block';
        historyList.style.display = 'none';
    } else {
        historyEmpty.style.display = 'none';
        historyList.style.display = 'flex';
        
        historyList.innerHTML = history.map((item, index) => `
            <div class="history-item">
                <img src="${item.image}" alt="历史记录" class="history-image">
                <div class="history-content">
                    <div class="history-meta">${item.model} · ${item.type} · ${item.time}</div>
                    <div class="history-text">${escapeHtml(item.text)}</div>
                </div>
                <button class="history-delete" onclick="deleteHistory(${index})">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 6L6 18M6 6l12 12"/>
                    </svg>
                </button>
            </div>
        `).join('');
    }
}

function deleteHistory(index) {
    let history = JSON.parse(localStorage.getItem('vglm_history') || '[]');
    history.splice(index, 1);
    localStorage.setItem('vglm_history', JSON.stringify(history));
    loadHistory();
}

function compressImage(base64Image, maxWidth = 200) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const scale = maxWidth / img.width;
            canvas.width = maxWidth;
            canvas.height = img.height * scale;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            resolve(canvas.toDataURL('image/jpeg', 0.6));
        };
        img.src = base64Image;
    });
}

async function saveToHistory(result) {
    console.log('保存到历史记录:', result);

    if (!isAutoSaveEnabled()) {
        console.log('自动保存已关闭，跳过保存');
        return;
    }
    
    if (!result.text) {
        console.log('结果没有文本，跳过保存');
        return;
    }
    
    const history = JSON.parse(localStorage.getItem('vglm_history') || '[]');
    const modelNames = {
        'fusion': '融合模型',
        '300vglm': '300VGLM',
        '5000vglm': '5000VGLM'
    };
    const typeNames = {
        'general': '通用描述',
        'background': '背景分析',
        'detailed': '详细描述',
        'english': '英文描述'
    };

    let compressedImage = result.image;
    if (result.image && result.image.length > 50000) { 
        try {
            compressedImage = await compressImage(result.image);
            console.log('图片已压缩，原大小:', result.image.length, '压缩后:', compressedImage.length);
        } catch (e) {
            console.log('图片压缩失败，使用原图:', e);
        }
    }
    
    history.unshift({
        image: compressedImage,
        text: result.text,
        model: modelNames[state.selectedModel] || state.selectedModel,
        type: typeNames[state.selectedType] || state.selectedType,
        time: new Date().toLocaleString('zh-CN')
    });

    while (history.length > 10) {
        history.pop();
    }
    
    try {
        localStorage.setItem('vglm_history', JSON.stringify(history));
        console.log('历史记录已保存，当前记录数:', history.length);
    } catch (e) {
        console.error('保存历史记录失败（存储空间不足）:', e);

        while (history.length > 5) {
            history.pop();
        }
        try {
            localStorage.setItem('vglm_history', JSON.stringify(history));
            console.log('历史记录已保存（精简版），当前记录数:', history.length);
        } catch (e2) {
            console.error('仍然无法保存历史记录:', e2);
        }
    }
}

function initSettings() {
    console.log('初始化设置...');

    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        const savedDarkMode = localStorage.getItem('vglm_dark_mode');
        const isDarkMode = savedDarkMode !== null ? savedDarkMode === 'true' : true;
        
        console.log('当前暗色模式设置:', isDarkMode, 'localStorage值:', savedDarkMode);
        
        darkModeToggle.checked = isDarkMode;
        applyDarkMode(isDarkMode);
        
        darkModeToggle.addEventListener('change', (e) => {
            const isDark = e.target.checked;
            console.log('暗色模式开关变化:', isDark);
            applyDarkMode(isDark);
            localStorage.setItem('vglm_dark_mode', isDark);
            console.log('暗色模式:', isDark ? '开启' : '关闭');
        });
    }

    const autoSaveToggle = document.getElementById('autoSaveToggle');
    if (autoSaveToggle) {
        const savedAutoSave = localStorage.getItem('vglm_auto_save');
        const isAutoSave = savedAutoSave !== null ? savedAutoSave === 'true' : true;
        
        autoSaveToggle.checked = isAutoSave;
        
        autoSaveToggle.addEventListener('change', (e) => {
            const isAuto = e.target.checked;
            localStorage.setItem('vglm_auto_save', isAuto);
            console.log('自动保存历史记录:', isAuto ? '开启' : '关闭');
        });
    }

    const browseModelPathBtn = document.getElementById('browseModelPathBtn');
    if (browseModelPathBtn) {
        browseModelPathBtn.addEventListener('click', () => {

            alert('模型路径设置功能需要在后端实现文件浏览器。\n当前模型路径: ./visualglm');
        });
    }

    const clearCacheBtn = document.getElementById('clearCacheBtn');
    if (clearCacheBtn) {
        clearCacheBtn.addEventListener('click', () => {
            if (confirm('确定要清除所有缓存和历史记录吗？')) {
                localStorage.removeItem('vglm_history');
                localStorage.removeItem('vglm_dark_mode');
                localStorage.removeItem('vglm_auto_save');
                alert('缓存已清除，页面将刷新');
                location.reload();
            }
        });
    }
}

function applyDarkMode(isDark) {
    console.log('应用暗色模式:', isDark);
    if (isDark) {
        document.body.classList.remove('light-mode');
        console.log('已移除 light-mode class');
    } else {
        document.body.classList.add('light-mode');
        console.log('已添加 light-mode class, body classes:', document.body.className);
    }
}

function isAutoSaveEnabled() {
    const saved = localStorage.getItem('vglm_auto_save');
    return saved !== null ? saved === 'true' : true;
}

/**
 * DICOM GFR 计算工具 - 前端 JavaScript
 */

// ====================================================================
// 全局变量
// ====================================================================

let currentFile = {
    dynamic: null,
    ct: null
};
// CT 多结果切换：当前展示的切片索引与主结果（用于李氏深度等）
let ctResultsList = [];
let ctResultIndex = 0;
let ctMainResult = null;

// ====================================================================
// 页面初始化
// ====================================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeFileUploads();
    initializeButtons();
    initializeLogResizer();
    getStatus();
});

// ====================================================================
// 导航功能
// ====================================================================

function initializeNavigation() {
    // 导航按钮点击事件
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const page = this.getAttribute('data-page');
            showPage(page);
        });
    });

    // 功能卡片点击事件
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('click', function() {
            const page = this.getAttribute('data-page');
            if (page === 'manual') {
                showManualDialog();
            } else {
                showPage(page);
            }
        });
    });
}

function showPage(pageName) {
    // 隐藏所有页面
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });

    // 显示目标页面
    const targetPage = document.getElementById(`page-${pageName}`);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    // 更新导航按钮状态
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-page') === pageName) {
            btn.classList.add('active');
        }
    });

    // 特殊处理：首页按钮
    if (pageName === 'home') {
        const homeBtn = document.querySelector('.nav-btn[data-page="home"]');
        if (homeBtn) homeBtn.classList.add('active');
    }
    // CT 页：隐藏或显示左右切换箭头
    const al = document.getElementById('ct-arrow-left');
    const ar = document.getElementById('ct-arrow-right');
    if (pageName !== 'ct') {
        if (al) al.style.display = 'none';
        if (ar) ar.style.display = 'none';
    } else if (ctResultsList.length > 1) {
        if (al) { al.style.display = 'flex'; al.disabled = ctResultIndex <= 0; }
        if (ar) { ar.style.display = 'flex'; ar.disabled = ctResultIndex >= ctResultsList.length - 1; }
    }

    logMessage(`[系统提示] 已切换到: ${pageName}`, 'info');
}

// ====================================================================
// 文件上传功能
// ====================================================================

function initializeFileUploads() {
    // 肾动态显像上传
    const dynamicUploadBox = document.getElementById('dynamic-upload-box');
    const dynamicFileInput = document.getElementById('dynamic-file-input');

    dynamicUploadBox.addEventListener('click', () => dynamicFileInput.click());
    dynamicUploadBox.addEventListener('dragover', handleDragOver);
    dynamicUploadBox.addEventListener('drop', (e) => handleDrop(e, 'dynamic'));
    dynamicFileInput.addEventListener('change', (e) => handleFileSelect(e, 'dynamic'));

    // CT 深度上传：同一上传区，两个入口——选择文件（可进文件夹选单/多文件）、选择文件夹
    const ctUploadBox = document.getElementById('ct-upload-box');
    const ctFileInput = document.getElementById('ct-file-input');
    const ctFolderInput = document.getElementById('ct-folder-input');
    const ctBtnFile = document.getElementById('ct-btn-file');
    const ctBtnFolder = document.getElementById('ct-btn-folder');

    ctUploadBox.addEventListener('dragover', handleDragOver);
    ctUploadBox.addEventListener('drop', (e) => handleDrop(e, 'ct'));
    ctFileInput.addEventListener('change', (e) => handleFileSelectCtFile(e));
    ctFolderInput.addEventListener('change', (e) => handleFileSelectCtFolder(e));
    if (ctBtnFile) ctBtnFile.addEventListener('click', (e) => { e.stopPropagation(); ctFileInput.click(); });
    if (ctBtnFolder) ctBtnFolder.addEventListener('click', (e) => { e.stopPropagation(); ctFolderInput.click(); });
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.style.borderColor = '#0078D7';
    e.currentTarget.style.background = 'rgba(0, 120, 215, 0.1)';
}

function handleDrop(e, type) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.style.borderColor = '';
    e.currentTarget.style.background = '';

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    if (type === 'ct' && files.length > 1) {
        var allowed = files.filter(f => /\.(dcm|dicom)$/i.test(f.name || ''));
        if (allowed.length > 0) handleFiles(allowed, 'ct');
        else showError('请拖拽 .dcm 或 .dicom 文件');
    } else {
        handleFile(files[0], type);
    }
}

function handleFileSelect(e, type) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    if (type === 'dynamic') {
        handleFile(files[0], 'dynamic');
    }
}

function handleFileSelectCtFile(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    var list = Array.from(files).filter(f => /\.(dcm|dicom)$/i.test(f.name || ''));
    if (list.length === 0) {
        showError('请选择 .dcm 或 .dicom 文件');
        return;
    }
    if (list.length === 1) handleFile(list[0], 'ct');
    else handleFiles(list, 'ct');
}

function handleFileSelectCtFolder(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    var list = Array.from(files).filter(f => /\.(dcm|dicom)$/i.test(f.name || ''));
    if (list.length > 0) handleFiles(list, 'ct');
    else showError('该文件夹内未找到 .dcm 或 .dicom 文件');
}

function handleFile(file, type) {
    const allowedExtensions = ['.dcm', '.dicom'];
    const fileExtension = '.' + (file.name || '').split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
        showError('不支持的文件格式，请上传 .dcm 或 .dicom 文件');
        return;
    }
    currentFile[type] = file;
    displayFileInfo(file, type);
    const processBtn = document.getElementById(`btn-process-${type}`);
    if (processBtn) processBtn.disabled = false;
    logMessage(`[文件选择] 已选择文件: ${file.name}`, 'success');
}

function handleFiles(files, type) {
    if (type !== 'ct' || !files.length) return;
    currentFile[type] = files;
    displayFileInfo(files, type);
    const processBtn = document.getElementById('btn-process-ct');
    if (processBtn) processBtn.disabled = false;
    logMessage(`[文件选择] 已选择文件夹，共 ${files.length} 个 DICOM 文件`, 'success');
}

function displayFileInfo(file, type) {
    const uploadBox = document.getElementById(`${type}-upload-box`);
    const fileInfo = document.getElementById(`${type}-file-info`);
    const fileName = document.getElementById(`${type}-file-name`);
    if (!uploadBox || !fileInfo || !fileName) return;
    uploadBox.style.display = 'none';
    fileInfo.style.display = 'flex';
    fileName.textContent = Array.isArray(file)
        ? (file.length === 1 ? '已选择 1 个文件' : `已选择 ${file.length} 个文件（来自文件夹）`)
        : (file.name || '');
}

function clearFile(type) {
    currentFile[type] = null;
    const uploadBox = document.getElementById(`${type}-upload-box`);
    const fileInfo = document.getElementById(`${type}-file-info`);
    const fileInput = document.getElementById(`${type}-file-input`);
    const folderInput = type === 'ct' ? document.getElementById('ct-folder-input') : null;
    if (uploadBox) uploadBox.style.display = 'block';
    if (fileInfo) fileInfo.style.display = 'none';
    if (fileInput) fileInput.value = '';
    if (folderInput) folderInput.value = '';
    const processBtn = document.getElementById(`btn-process-${type}`);
    if (processBtn) processBtn.disabled = true;
    logMessage(`[文件清除] 已清除 ${type} 选择`, 'info');
}

// ====================================================================
// 按钮功能
// ====================================================================

function initializeButtons() {
    // 处理肾动态显像
    document.getElementById('btn-process-dynamic').addEventListener('click', () => {
        processDynamicStudy();
    });

    // 处理 CT 深度
    document.getElementById('btn-process-ct').addEventListener('click', () => {
        processCTDepth();
    });

    // 计算 GFR
    document.getElementById('btn-calculate-gfr').addEventListener('click', () => {
        calculateGFR();
    });
}

// ====================================================================
// API 调用
// ====================================================================

async function processDynamicStudy() {
    if (!currentFile.dynamic) {
        showError('请先选择文件');
        return;
    }

    showLoading(true);
    logMessage('[处理开始] 开始处理肾动态显像...', 'info');

    try {
        const formData = new FormData();
        formData.append('file', currentFile.dynamic);

        const response = await fetch('/api/upload/dynamic', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            logMessage('[处理成功] 肾动态显像处理完成', 'success');
            displayDynamicResults(result);
        } else {
            showError(result.message || '处理失败');
            logMessage(`[处理失败] ${result.message}`, 'error');
        }
    } catch (error) {
        showError('网络错误: ' + error.message);
        logMessage(`[错误] ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function processCTDepth() {
    if (!currentFile.ct) {
        showError('请先选择单个文件或整个文件夹');
        return;
    }

    showLoading(true);
    logMessage('[处理开始] 开始处理 CT 深度...', 'info');

    try {
        const formData = new FormData();
        if (Array.isArray(currentFile.ct)) {
            currentFile.ct.forEach(f => formData.append('files', f));
        } else {
            formData.append('file', currentFile.ct);
        }

        const response = await fetch('/api/upload/ct', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            logMessage('[处理成功] CT 深度处理完成', 'success');
            ctMainResult = result;
            ctResultsList = result.allSliceResults && result.allSliceResults.length > 0
                ? result.allSliceResults
                : [{
                    originalPngPath: (result.imageUrls && result.imageUrls.originalPngPath) || result.originalPngPath,
                    overlayPngPath: (result.imageUrls && result.imageUrls.overlayPngPath) || result.overlayPngPath,
                    leftDepth: result.modelLeftDepth,
                    rightDepth: result.modelRightDepth,
                    sliceName: result.deepestSliceName || ''
                }];
            ctResultIndex = 0;
            displayCTResults();
        } else {
            showError(result.message || '处理失败');
            logMessage(`[处理失败] ${result.message}`, 'error');
        }
    } catch (error) {
        showError('网络错误: ' + error.message);
        logMessage(`[错误] ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function calculateGFR() {
    showLoading(true);
    logMessage('[计算开始] 开始计算 GFR...', 'info');

    try {
        const response = await fetch('/api/calculate/gfr', {
            method: 'POST'
        });

        const result = await response.json();

        if (result.success) {
            logMessage('[计算成功] GFR 计算完成', 'success');
            displayGFRResults(result.gfr || {});
        } else {
            // 显示详细的错误信息
            let errorMsg = result.message || '计算失败';
            if (result.missing_data && result.missing_data.length > 0) {
                errorMsg += '\n\n缺少的数据：' + result.missing_data.join('、');
                errorMsg += '\n\n请先完成以下步骤：';
                if (result.missing_data.some(d => d.includes('厂商') || d.includes('计数'))) {
                    errorMsg += '\n1. 处理肾动态显像（获取厂商信息和肾脏计数）';
                }
                if (result.missing_data.some(d => d.includes('深度'))) {
                    errorMsg += '\n2. 处理 CT 深度或手动上传深度信息';
                }
            }
            showError(errorMsg);
            logMessage(`[计算失败] ${result.message}`, 'error');
        }
    } catch (error) {
        showError('网络错误: ' + error.message);
        logMessage(`[错误] ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function uploadManualDepth() {
    const form = document.getElementById('manual-form');
    const formData = {
        leftDepth: document.getElementById('left-depth').value || null,
        rightDepth: document.getElementById('right-depth').value || null,
        height: document.getElementById('height').value,
        weight: document.getElementById('weight').value,
        age: document.getElementById('age').value,
        sex: document.getElementById('sex').value
    };

    // 验证必填字段
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    showLoading(true);
    logMessage('[上传开始] 正在上传深度和患者信息...', 'info');

    try {
        const response = await fetch('/api/upload/depth-manual', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (result.success) {
            logMessage('[上传成功] 深度信息已上传，李氏深度计算完成', 'success');
            // 显示计算后的肾脏深度数据（与源程序一致）
            const depths = result.kidney_depths || {};
            logMessage('[肾脏深度 kidney_depths] leftDepth: ' + (depths.leftDepth != null ? depths.leftDepth + ' mm' : 'N/A') +
                ', rightDepth: ' + (depths.rightDepth != null ? depths.rightDepth + ' mm' : 'N/A') +
                ', LiLeftKidneyDepth: ' + (depths.LiLeftKidneyDepth != null ? depths.LiLeftKidneyDepth.toFixed(2) + ' mm' : 'N/A') +
                ', LiRightKidneyDepth: ' + (depths.LiRightKidneyDepth != null ? depths.LiRightKidneyDepth.toFixed(2) + ' mm' : 'N/A'), 'success');
            closeManualDialog();
            getStatus();
        } else {
            showError(result.message || '上传失败');
            logMessage(`[上传失败] ${result.message}`, 'error');
        }
    } catch (error) {
        showError('网络错误: ' + error.message);
        logMessage(`[错误] ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function getStatus() {
    try {
        const response = await fetch('/api/status');
        const result = await response.json();

        if (result.success) {
            logMessage('[状态] 处理器状态已更新', 'info');
            // 可以在这里更新状态显示
        }
    } catch (error) {
        logMessage(`[错误] 获取状态失败: ${error.message}`, 'error');
    }
}

async function resetProcessor() {
    if (!confirm('确定要重置处理器状态吗？')) {
        return;
    }

    try {
        const response = await fetch('/api/reset', {
            method: 'POST'
        });

        const result = await response.json();

        if (result.success) {
            logMessage('[重置成功] 处理器状态已重置', 'success');
            getStatus();
        } else {
            showError(result.message || '重置失败');
        }
    } catch (error) {
        showError('网络错误: ' + error.message);
    }
}

// ====================================================================
// 结果显示
// ====================================================================

function displayDynamicResults(result) {
    const resultsSection = document.getElementById('dynamic-results');
    if (resultsSection) resultsSection.style.display = 'block';

    // 显示图像
    const imageUrls = result.imageUrls || {};
    displayImage('dynamic-original', imageUrls.imageUrl);
    displayImage('dynamic-curve', imageUrls.countsTimeUrl);
    displayImage('dynamic-roi', imageUrls.overlayUrl);
    displayImage('dynamic-halfcurve', imageUrls.halfCurveUrl);

    // 显示计数数据
    const counts = result.kidneyCounts || {};
    updateTableCell('left-kidney-count', counts.leftKidneyCount);
    updateTableCell('right-kidney-count', counts.rightKidneyCount);
    updateTableCell('left-bg-count', counts.leftBackgroundCount);
    updateTableCell('right-bg-count', counts.rightBackgroundCount);

    // 显示半排指标
    const halfMetrics = result.halfMetrics || {};
    if (halfMetrics.left) {
        updateTableCell('left-half-time', formatValue(halfMetrics.left.t_half, ' min'));
        updateTableCell('left-half-rate', formatValue(halfMetrics.left.half_rate, ' 1/min'));
        updateTableCell('left-half-reached', halfMetrics.left.reached_half ? '是' : '否');
    }
    if (halfMetrics.right) {
        updateTableCell('right-half-time', formatValue(halfMetrics.right.t_half, ' min'));
        updateTableCell('right-half-rate', formatValue(halfMetrics.right.half_rate, ' 1/min'));
        updateTableCell('right-half-reached', halfMetrics.right.reached_half ? '是' : '否');
    }
}

function displayCTResults() {
    const resultsSection = document.getElementById('ct-results');
    const navWrap = document.getElementById('ct-result-nav');
    if (!resultsSection) return;
    resultsSection.style.display = 'block';

    if (ctResultsList.length === 0) return;
    const item = ctResultsList[ctResultIndex];
    const main = ctMainResult || {};

    // 显示当前切片图像（allSliceResults 里已是 URL）
    displayImage('ct-original', item.originalPngPath);
    displayImage('ct-overlay', item.overlayPngPath);

    // 显示当前切片深度
    updateTableCell('model-left-depth', formatValue(item.leftDepth, ' mm'));
    updateTableCell('model-right-depth', formatValue(item.rightDepth, ' mm'));
    updateTableCell('li-left-depth', formatValue(main.LiLeftDepth, ' mm'));
    updateTableCell('li-right-depth', formatValue(main.LiRightDepth, ' mm'));

    // 多张时显示页面左右箭头与计数
    const showNav = ctResultsList.length > 1;
    const arrowLeft = document.getElementById('ct-arrow-left');
    const arrowRight = document.getElementById('ct-arrow-right');
    if (arrowLeft) {
        arrowLeft.style.display = showNav ? 'flex' : 'none';
        arrowLeft.disabled = ctResultIndex <= 0;
    }
    if (arrowRight) {
        arrowRight.style.display = showNav ? 'flex' : 'none';
        arrowRight.disabled = ctResultIndex >= ctResultsList.length - 1;
    }
    if (navWrap) {
        navWrap.style.display = showNav ? 'flex' : 'none';
        const countText = document.getElementById('ct-result-count');
        if (countText) countText.textContent = `第 ${ctResultIndex + 1} / ${ctResultsList.length} 张`;
    }
}

function ctResultPrev() {
    if (ctResultIndex > 0) {
        ctResultIndex--;
        displayCTResults();
    }
}

function ctResultNext() {
    if (ctResultIndex < ctResultsList.length - 1) {
        ctResultIndex++;
        displayCTResults();
    }
}

function displayGFRResults(gfrData) {
    const resultsSection = document.getElementById('gfr-results');
    if (resultsSection) resultsSection.style.display = 'block';

    // 更新 GFR 表格
    updateTableCell('left-gfr', formatValue(gfrData.leftGFR));
    updateTableCell('right-gfr', formatValue(gfrData.rightGFR));
    updateTableCell('total-gfr', formatValue(gfrData.totalGFR));
    updateTableCell('li-left-gfr', formatValue(gfrData.LiLeftGFR));
    updateTableCell('li-right-gfr', formatValue(gfrData.LiRightGFR));
    updateTableCell('li-total-gfr', formatValue(gfrData.LiTotalGFR));
}

function displayImage(elementId, imageUrl) {
    const imgElement = document.getElementById(elementId);
    const placeholder = imgElement?.nextElementSibling;

    if (imgElement && imageUrl) {
        imgElement.src = imageUrl;
        imgElement.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';
    } else if (imgElement) {
        imgElement.style.display = 'none';
        if (placeholder) placeholder.style.display = 'block';
    }
}

function updateTableCell(elementId, value) {
    const cell = document.getElementById(elementId);
    if (cell) {
        cell.textContent = value !== null && value !== undefined ? value : 'N/A';
    }
}

function formatValue(value, unit = '') {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') {
        return value.toFixed(2) + unit;
    }
    return value + unit;
}

// ====================================================================
// 对话框功能
// ====================================================================

function showManualDialog() {
    const dialog = document.getElementById('manual-dialog');
    if (dialog) {
        dialog.style.display = 'flex';
    }
}

function closeManualDialog() {
    const dialog = document.getElementById('manual-dialog');
    if (dialog) {
        dialog.style.display = 'none';
        // 清空表单
        document.getElementById('manual-form').reset();
    }
}

function submitManualForm() {
    uploadManualDepth();
}

// 点击对话框外部关闭
document.getElementById('manual-dialog')?.addEventListener('click', function(e) {
    if (e.target === this) {
        closeManualDialog();
    }
});

// ====================================================================
// 工具函数
// ====================================================================

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

function showError(message) {
    // 创建自定义错误对话框
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-dialog';
    errorDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #fff;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 2rem;
        max-width: 500px;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 4000;
    `;
    
    errorDiv.innerHTML = `
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <i class="fas fa-exclamation-triangle" style="color: #dc3545; font-size: 2rem; margin-right: 1rem;"></i>
            <h3 style="margin: 0; color: #dc3545;">错误提示</h3>
        </div>
        <div style="color: #333; white-space: pre-line; line-height: 1.6; margin-bottom: 1.5rem;">${message}</div>
        <button onclick="this.parentElement.remove()" style="
            background: #dc3545;
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            width: 100%;
        ">确定</button>
    `;
    
    document.body.appendChild(errorDiv);
    
    // 点击背景关闭
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 3999;
    `;
    overlay.onclick = () => {
        errorDiv.remove();
        overlay.remove();
    };
    document.body.appendChild(overlay);
}

function logMessage(message, type = 'info') {
    const logContent = document.getElementById('log-content');
    if (!logContent) return;

    const timestamp = new Date().toLocaleTimeString();
    const logClass = `log-${type}`;
    const logEntry = document.createElement('div');
    logEntry.className = logClass;
    logEntry.textContent = `[${timestamp}] ${message}`;
    
    logContent.appendChild(logEntry);
    logContent.scrollTop = logContent.scrollHeight;

    // 限制日志条数
    while (logContent.children.length > 100) {
        logContent.removeChild(logContent.firstChild);
    }
}

function clearLog() {
    const logContent = document.getElementById('log-content');
    if (logContent) {
        logContent.innerHTML = '';
        logMessage('日志已清空', 'info');
    }
}

// ====================================================================
// 日志面板高度调整功能
// ====================================================================

function initializeLogResizer() {
    const resizer = document.getElementById('log-resizer');
    const logPanel = document.getElementById('log-panel');
    const logHeader = logPanel.querySelector('.log-header');
    
    if (!resizer || !logPanel) return;
    
    let isResizing = false;
    let startY = 0;
    let startHeight = 0;
    const headerHeight = logHeader.offsetHeight;
    const minHeight = headerHeight + 60; // 最小高度：标题高度 + 60px内容区域
    const maxHeight = window.innerHeight - 100; // 最大高度：视窗高度 - 100px
    
    resizer.addEventListener('mousedown', function(e) {
        isResizing = true;
        startY = e.clientY;
        startHeight = logPanel.offsetHeight;
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', function(e) {
        if (!isResizing) return;
        
        const deltaY = startY - e.clientY; // 向上拖拽是正数
        let newHeight = startHeight + deltaY;
        
        // 限制高度范围
        if (newHeight < minHeight) {
            newHeight = minHeight;
        } else if (newHeight > maxHeight) {
            newHeight = maxHeight;
        }
        
        // 更新面板高度
        logPanel.style.height = newHeight + 'px';
        document.documentElement.style.setProperty('--log-panel-height', newHeight + 'px');
        
        // 更新主容器的底部 padding
        const mainContainer = document.querySelector('.main-container');
        if (mainContainer) {
            mainContainer.style.paddingBottom = (newHeight + 32) + 'px';
        }
    });
    
    document.addEventListener('mouseup', function() {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            
            // 保存高度到 localStorage
            const currentHeight = logPanel.offsetHeight;
            localStorage.setItem('logPanelHeight', currentHeight);
        }
    });
    
    // 从 localStorage 恢复高度
    const savedHeight = localStorage.getItem('logPanelHeight');
    if (savedHeight) {
        const height = parseInt(savedHeight);
        if (height >= minHeight && height <= maxHeight) {
            logPanel.style.height = height + 'px';
            document.documentElement.style.setProperty('--log-panel-height', height + 'px');
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer) {
                mainContainer.style.paddingBottom = (height + 32) + 'px';
            }
        }
    }
    
    // 窗口大小改变时调整最大高度
    window.addEventListener('resize', function() {
        const currentHeight = logPanel.offsetHeight;
        const newMaxHeight = window.innerHeight - 100;
        
        if (currentHeight > newMaxHeight) {
            logPanel.style.height = newMaxHeight + 'px';
            document.documentElement.style.setProperty('--log-panel-height', newMaxHeight + 'px');
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer) {
                mainContainer.style.paddingBottom = (newMaxHeight + 32) + 'px';
            }
        }
    });
}


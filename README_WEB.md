# DICOM GFR 计算工具 - Web 版本

## 📋 项目说明

本项目已从 Tkinter GUI 重构为基于 Flask 的 Web 应用，提供现代化的 HTML/CSS/JavaScript 前端界面。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Web 应用

```bash
cd GFR-main
python -m back.app_flask
```

或者：

```bash
cd GFR-main/back
python app_flask.py
```

### 3. 访问应用

打开浏览器访问：`http://127.0.0.1:5000`

## 📁 项目结构

```
GFR-main/
├── back/
│   ├── app_flask.py          # Flask Web 后端（新）
│   ├── app.py                # 原 Tkinter GUI（保留）
│   ├── local_dicom_process.py # 核心业务逻辑
│   ├── templates/            # HTML 模板
│   │   └── index.html
│   └── static/               # 静态资源
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── main.js
├── requirements.txt          # Python 依赖
└── README_WEB.md            # 本文档
```

## 🎯 功能特性

### Web 版本优势

1. **现代化界面**：基于 HTML5/CSS3 的响应式设计
2. **跨平台**：可在任何有浏览器的设备上使用
3. **易于部署**：可部署到服务器供多人使用
4. **更好的用户体验**：拖拽上传、实时日志、加载动画等

### 主要功能

1. **处理肾动态显像**：上传 DICOM 文件，自动计算肾脏计数和生成曲线图
2. **处理 CT 深度**：上传 CT 图像，计算肾脏深度
3. **手动上传信息**：手动输入深度和患者信息
4. **计算 GFR**：基于处理结果计算肾小球滤过率

## 🔧 API 接口

### 获取状态
```
GET /api/status
```

### 重置处理器
```
POST /api/reset
```

### 上传肾动态显像
```
POST /api/upload/dynamic
Content-Type: multipart/form-data
Body: file (DICOM文件)
```

### 上传 CT 深度
```
POST /api/upload/ct
Content-Type: multipart/form-data
Body: file (DICOM文件)
```

### 手动上传深度信息
```
POST /api/upload/depth-manual
Content-Type: application/json
Body: {
    "leftDepth": float (可选),
    "rightDepth": float (可选),
    "height": float (必填),
    "weight": float (必填),
    "age": int (必填),
    "sex": "男" | "女" (必填)
}
```

### 计算 GFR
```
POST /api/calculate/gfr
```

## 📝 使用说明

### 处理肾动态显像

1. 点击导航栏的"肾动态显像"或首页的功能卡片
2. 点击上传区域或拖拽 DICOM 文件到上传区域
3. 点击"开始处理"按钮
4. 等待处理完成，查看结果图像和数据表格

### 处理 CT 深度

1. 点击导航栏的"CT 深度计算"
2. 上传 CT DICOM 文件或文件夹
3. 点击"开始处理"按钮
4. 查看深度计算结果和标注图像

### 计算 GFR

1. 确保已完成肾动态显像处理和深度计算
2. 点击导航栏的"GFR 计算"
3. 点击"开始计算 GFR"按钮
4. 查看 GFR 计算结果表格

## 🛠️ 开发说明

### 前端技术栈
- HTML5
- CSS3 (使用 CSS 变量和 Flexbox/Grid)
- JavaScript (原生 ES6+)
- Font Awesome 图标库

### 后端技术栈
- Flask 3.0
- Flask-CORS (跨域支持)
- 原有业务逻辑模块（DicomProcessor）

### 自定义样式

修改 `back/static/css/style.css` 中的 CSS 变量：

```css
:root {
    --primary-color: #0078D7;  /* 主色调 */
    --bg-color: #f5f7fa;        /* 背景色 */
    /* ... 更多变量 */
}
```

## ⚠️ 注意事项

1. **权重文件**：确保 `back/weights/` 目录下有必需的模型权重文件
2. **文件大小**：默认最大上传 500MB，可在 `app_flask.py` 中修改
3. **输出目录**：处理结果保存在 `output/` 目录下
4. **浏览器兼容**：推荐使用 Chrome、Firefox、Edge 等现代浏览器

## 🔄 从 Tkinter 版本迁移

原有的 Tkinter GUI (`app.py`) 仍然保留，可以继续使用。Web 版本 (`app_flask.py`) 是新的实现，两者共享相同的业务逻辑 (`local_dicom_process.py`)。

## 📞 问题反馈

如遇到问题，请检查：
1. Python 版本（推荐 3.8+）
2. 所有依赖是否已安装
3. 权重文件是否存在
4. 浏览器控制台是否有错误信息

## 📄 许可证

与原项目保持一致。


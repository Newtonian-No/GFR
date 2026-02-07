#!/bin/bash

echo "========================================"
echo "DICOM GFR 计算工具 - Web 版本"
echo "========================================"
echo ""
echo "正在启动 Web 服务器..."
echo ""

cd "$(dirname "$0")"
python3 -m back.app_flask


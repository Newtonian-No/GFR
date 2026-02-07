"""
Flask Web 应用 - DICOM GFR 计算工具
将原有的 Tkinter GUI 重构为 Web 应用
"""
import os
import json
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import numpy as np

from back.local_dicom_process import DicomProcessor
from back.constants import ALL_OUTPUT_DIRS, OUTPUT_DIR

# 初始化 Flask 应用
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)  # 允许跨域请求

# 配置
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 最大上传 500MB
app.config['UPLOAD_FOLDER'] = str(OUTPUT_DIR / 'uploads')
app.config['SECRET_KEY'] = 'your-secret-key-here'

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
for dir_path in ALL_OUTPUT_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)

# 全局处理器实例（线程安全）
processor = DicomProcessor()

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'dcm', 'dicom'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_numpy_types(obj):
    """
    递归地将 NumPy 类型转换为 Python 原生类型，以便 JSON 序列化
    支持: np.integer, np.floating, np.ndarray, np.bool_, np.complex 等
    """
    # NumPy 整数类型 (包括 uint8, uint16, uint32, uint64, int8, int16, int32, int64 等)
    if isinstance(obj, (np.integer, np.uint8, np.uint16, np.uint32, np.uint64,
                       np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    # NumPy 浮点类型 (包括 float16, float32, float64 等)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    # NumPy 布尔类型
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # NumPy 数组
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # NumPy 复数类型
    elif isinstance(obj, np.complexfloating):
        return complex(obj)
    # 字典类型 - 递归处理
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    # 列表和元组类型 - 递归处理
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    # Python 原生类型 - 直接返回
    elif isinstance(obj, (bool, type(None), str, int, float)):
        return obj
    else:
        # 对于其他类型，尝试转换为字符串
        try:
            # 检查是否是 NumPy 类型（通过检查是否有 item() 方法）
            if hasattr(obj, 'item'):
                return obj.item()
            return str(obj)
        except:
            return obj

# ====================================================================
# 路由定义
# ====================================================================

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取处理器状态"""
    try:
        status = {
            'manufacturer': processor.last_manufacturer,
            'patient_name': processor.last_patient_name,
            'patient_info': processor.last_patient_info,
            'kidney_counts': processor.last_kidney_counts,
            'kidney_depths': processor.kidney_depths
        }
        # 转换 NumPy 类型为 Python 原生类型
        status = convert_numpy_types(status)
        return jsonify({'success': True, 'data': status})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_processor():
    """重置处理器状态"""
    try:
        processor.reset_state()
        return jsonify({'success': True, 'message': '状态已重置'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/upload/dynamic', methods=['POST'])
def upload_dynamic_study():
    """上传并处理肾动态显像 DICOM 文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名为空'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': '不支持的文件格式'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理文件
        result = processor.process_dynamic_study_dicom(filepath)
        
        # 转换路径为 URL 可访问的路径
        if result.get('success'):
            # 转换输出路径为相对路径
            output_paths = {}
            for key in ['imageUrl', 'overlayUrl', 'countsTimeUrl', 'halfCurveUrl']:
                if key in result and result[key]:
                    try:
                        path = Path(result[key])
                        if path.exists():
                            # 转换为相对于 output 目录的路径
                            try:
                                rel_path = path.relative_to(OUTPUT_DIR)
                                output_paths[key] = f'/output/{rel_path.as_posix()}'
                            except ValueError:
                                # 如果路径不在 OUTPUT_DIR 下，使用绝对路径的最后一个部分
                                output_paths[key] = f'/output/{path.name}'
                    except Exception as e:
                        print(f"路径转换错误 {key}: {e}")
            
            result['imageUrls'] = output_paths
        
        # 转换 NumPy 类型为 Python 原生类型
        result = convert_numpy_types(result)
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'处理失败: {str(e)}'}), 500

@app.route('/api/upload/ct', methods=['POST'])
def upload_ct_depth():
    """上传并处理 CT 深度 DICOM 文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名为空'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': '不支持的文件格式'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理文件
        result = processor.process_depth_and_li_depth(filepath)
        
        # 转换路径为 URL 可访问的路径
        if result.get('success'):
            output_paths = {}
            for key in ['originalPngPath', 'overlayPngPath']:
                if key in result and result[key]:
                    try:
                        path = Path(result[key])
                        if path.exists():
                            try:
                                rel_path = path.relative_to(OUTPUT_DIR)
                                output_paths[key] = f'/output/{rel_path.as_posix()}'
                            except ValueError:
                                # 如果路径不在 OUTPUT_DIR 下，使用绝对路径的最后一个部分
                                output_paths[key] = f'/output/{path.name}'
                    except Exception as e:
                        print(f"路径转换错误 {key}: {e}")
            
            result['imageUrls'] = output_paths
        
        # 转换 NumPy 类型为 Python 原生类型
        result = convert_numpy_types(result)
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'处理失败: {str(e)}'}), 500

@app.route('/api/upload/depth-manual', methods=['POST'])
def upload_depth_manual():
    """手动上传深度和患者信息"""
    try:
        data = request.get_json()
        
        left_depth = data.get('leftDepth')
        right_depth = data.get('rightDepth')
        height = data.get('height')
        weight = data.get('weight')
        age = data.get('age')
        sex = data.get('sex')
        
        # 验证必填字段
        if not all([height, weight, age, sex]):
            return jsonify({'success': False, 'message': '请填写所有必填字段'}), 400
        
        if sex not in ['男', '女']:
            return jsonify({'success': False, 'message': '性别必须是"男"或"女"'}), 400
        
        # 调用处理器
        result = processor.manual_upload_depth_and_calculate_li(
            left_depth=float(left_depth) if left_depth else None,
            right_depth=float(right_depth) if right_depth else None,
            height_m=float(height),
            weight_kg=float(weight),
            age_y=int(age),
            sex_cn=sex
        )
        
        # 返回完整肾脏深度
        if result.get('success'):
            result['kidney_depths'] = convert_numpy_types(processor.kidney_depths)
        
        # 转换 NumPy 类型为 Python 原生类型
        result = convert_numpy_types(result)
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'处理失败: {str(e)}'}), 500

@app.route('/api/calculate/gfr', methods=['POST'])
def calculate_gfr():
    """计算 GFR"""
    try:
        result = processor.calculate_gfr()
        # 转换 NumPy 类型为 Python 原生类型
        result = convert_numpy_types(result)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'计算失败: {str(e)}'}), 500

# ====================================================================
# 静态文件服务
# ====================================================================

@app.route('/output/<path:filename>')
def output_files(filename):
    """提供输出文件的访问"""
    return send_from_directory(str(OUTPUT_DIR), filename)

# ====================================================================
# 主程序入口
# ====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DICOM GFR 计算工具 - Web 版本")
    print("=" * 60)
    print(f"访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)


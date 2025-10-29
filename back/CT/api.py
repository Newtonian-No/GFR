# api.py
import shutil
from pathlib import Path
from typing import Dict, Any, Union
import cv2
import pydicom

from ..constants import ORIGINAL_CT_DIR , ORIGINAL_PNG_DIR , OVERLAY_PNG_DIR , YOLO_LABELS_DIR , ANALYSIS_RESULTS_DIR , DETECT_MODEL_WEIGHTS_PATH
from .detect import detect_single_dcm, detect_dcm_folder
from .depth_core import measure_kidney_depth, find_deepest_slice, load_dicom_image


def _dicom_to_png(dcm_path: Path, output_path: Path):
    """将 DICOM 图像转换为 PNG 格式，用于生成原始图片的 PNG 路径。"""
    try:
        # load_dicom_image 返回 (HU image, display_image (uint8))
        _, display_image = load_dicom_image(str(dcm_path)) 
        cv2.imwrite(str(output_path), display_image)
    except Exception as e:
        print(f"无法将 DICOM 转换为 PNG: {dcm_path.name}. 错误: {e}")
        output_path.touch() # 创建空文件占位

def _setup_and_copy_input(input_path: Path) -> Path:
    """复制输入文件/文件夹到标准备份目录，并返回备份路径。"""
    if input_path.is_file():
        dest_dir = ORIGINAL_CT_DIR
        dest_path = dest_dir / input_path.name
        shutil.copy2(input_path, dest_path)
        return dest_path
        
    elif input_path.is_dir():
        # 为文件夹创建一个子目录备份
        dest_path = ORIGINAL_CT_DIR / input_path.name 
        
        if dest_path.exists():
             shutil.rmtree(dest_path)
             
        shutil.copytree(input_path, dest_path)
        return dest_path
        
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

# ----------------------------------------------------------------------
# 核心 API
# ----------------------------------------------------------------------

def process_single_dcm(dcm_path: Path) -> Dict[str, Any]:
    """处理单个 DICOM 文件，返回深度和路径信息。"""
    dcm_stem = dcm_path.stem
    
    # 1. YOLO 检测 (标签保存到 YOLO_LABELS_DIR)
    txt_path = detect_single_dcm(dcm_path)
    
    # 2. 深度计算和可视化 (结果和叠加图保存到 ANALYSIS_RESULTS_DIR 和 OVERLAY_PNG_DIR)
    results = measure_kidney_depth(dcm_path, txt_path, is_deepest=True)
    
    # 3. 原始 DICOM 转换为 PNG (保存到 ORIGINAL_PNG_DIR)
    png_path = ORIGINAL_PNG_DIR / f"{dcm_stem}_original.png"
    _dicom_to_png(dcm_path, png_path)
    
    # 4. 构造输出路径
    overlay_path = OVERLAY_PNG_DIR / f"{dcm_stem}_overlay.png"

    return {
        'success': True,
        'leftDepth': results.get('L', 'N/A'),
        'rightDepth': results.get('R', 'N/A'),
        'originalPngPath': str(png_path),         # 项目根目录/output/original_png/
        'overlayPngPath': str(overlay_path),      # 项目根目录/output/overlay_png/
        'originalDicomPath': str(dcm_path),       # 项目根目录/output/original_dcm_backup/
    }


def process_dcm_series(folder_path: Path) -> Dict[str, Any]:
    """处理 DICOM 文件夹系列，返回最深切片的深度和路径信息。"""
    
    # 1. YOLO 批量检测 (所有标签保存到 YOLO_LABELS_DIR)
    yolo_labels_folder = detect_dcm_folder(folder_path)
    
    # 2. 查找最深切片 (在过程中会生成所有切片的 analysis_results)
    final_result = find_deepest_slice(folder_path, yolo_labels_folder)
    
    deepest_slice_name = final_result.get('deepest_slice')
    if not deepest_slice_name:
        return {'success': False, 'message': '未在系列中找到有效的肾脏深度切片'}
    
    # print(f"DEBUG:最深切片名称: {deepest_slice_name}")
    deepest_dcm_path = folder_path / deepest_slice_name
    dcm_stem = Path(deepest_slice_name).stem
    
    # 3. 原始 DICOM 转换为 PNG (保存最深切片到 ORIGINAL_PNG_DIR)
    png_path = ORIGINAL_PNG_DIR / f"{dcm_stem}_deepest_original.png"
    _dicom_to_png(deepest_dcm_path, png_path)

    # 4. 构造输出路径 (最深切片的叠加图已在 find_deepest_slice 中生成)
    overlay_path = OVERLAY_PNG_DIR / f"{dcm_stem}_deepest_overlay.png"
    # print(f"DEBUG:最深切片PNG路径: {png_path}")
    # print(f"DEBUG:最深切片叠加图路径: {overlay_path}")

    return {
        'success': True,
        'leftDepth': final_result.get('L', 'N/A'),
        'rightDepth': final_result.get('R', 'N/A'),
        'deepestSliceName': deepest_slice_name,
        'maxOverallDepthMm': final_result.get('max_overall_depth_mm', 0.0),
        'originalPngPath': str(png_path),             # 项目根目录/output/original_png/
        'overlayPngPath': str(overlay_path),          # 项目根目录/output/overlay_png/
        'originalDicomPath': str(deepest_dcm_path),   # 项目根目录/output/original_dcm_backup/
    }


def process_ct_input(input_path: Union[str, Path]) -> Dict[str, Any]:
    """健壮、统一地处理 CT DICOM 文件或文件夹，计算深度并输出可视化文件。"""
    input_path = Path(input_path)

    if not input_path.exists():
        return {'success': False, 'message': f"输入路径不存在: {input_path}"}
        
    try:
        # 1. 文件/文件夹复制到备份目录
        copied_path = _setup_and_copy_input(input_path) 
    except Exception as e:
        return {'success': False, 'message': f"文件/文件夹准备失败: {e}"}

    try:
        # 2. 按输入类型分派处理逻辑
        if copied_path.is_file():
            return process_single_dcm(copied_path)
        else:
            return process_dcm_series(copied_path)
            
    except Exception as e:
        import traceback
        print(f"致命错误：{e}")
        return {'success': False, 'message': f"数据处理失败: {e}\n{traceback.format_exc()}"}
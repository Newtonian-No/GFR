"""
重构后的主要功能：
1. 处理肾动态显像DICOM文件，提取信息、计算肾脏和背景的放射性计数。
2. 处理CT图像DICOM文件，使用模型和李氏公式计算肾脏深度。
3. 支持手动输入肾脏深度和人体测量数据来计算李氏深度。
4. GFR计算，结合肾脏放射性计数和深度数据，应用衰减公式计算 GFR。

注意：此版本移除了所有Flask和Web相关的依赖，并通过参数/返回值进行数据交互。
依赖模块 (假设已安装/存在): pydicom, numpy, math, CT, preprocess, predict, ROI, overlay, graph
"""
import os
import sys
import pydicom
import numpy as np
import re
import math
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
# 假设这些模块已存在且能被导入
from back.CT import CT, find_deepest_slice_in_series
from back.preprocess import extract_frame, resample_dicom
from back.predict import segment_kidney
from back.ROI import add_background_roi
from back.overlay import display_dicom_with_roi_overlay
from back.graph import graph
from back.constants import (
    LOCAL_DICOM_MODEL_PATH, 
    OUTPUT_DIR, 
    CONVERTED_OUTPUT_DIR, 
    SEGMENTED_OUTPUT_DIR, 
    GRAPH_OUTPUT_DIR,
    UPLOAD_DIR,
    DEPTH_OUTPUT_DIR,
    ORIGINAL_CT_FILE_DIR,
    ORIGINAL_CT_DIR,
    ALL_OUTPUT_DIRS # 用于创建目录列表
)
# MODEL_PATH_RELATIVE = 'weights/best_epoch_weights.pth'
# DICOM_MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE
# --- 模拟外部依赖模块 (用于使代码独立运行/测试) ---
# 在实际应用中，请替换为您的实际实现
# def CT(dicom_path: str) -> Tuple[Dict[str, Any], str]:
#     """模拟 CT 模块的肾脏深度计算"""
#     print(f"--- 模拟 CT 深度计算: {dicom_path}")
#     # 模拟结果：深度单位为 mm
#     return {'L': 75.2, 'R': 70.8}, "output/depth_visualization.png"

# def extract_frame(dicom_path: str, frame_number: int) -> Tuple[str, Any]:
#     """模拟从多帧DICOM中提取单帧"""
#     print(f"--- 模拟 extract_frame: 提取帧 {frame_number}")
#     # 实际应返回保存的单帧DCM路径和数据
#     return dicom_path, None

# def resample_dicom(original_dcm_path: str, order: int) -> Tuple[str, Any, str]:
#     """模拟重采样DICOM和生成PNG"""
#     print(f"--- 模拟 resample_dicom: 重采样和生成PNG")
#     # 实际应返回重采样后的DCM路径、数据和PNG路径
#     return "output/resampled.dcm", None, "output/converted/image.png"

# def segment_kidney(resampled_dcm_path: str, model_path: str, img_size: int, num_classes: int) -> Tuple[str, Any]:
#     """模拟肾脏分割"""
#     print(f"--- 模拟 segment_kidney: 进行肾脏分割")
#     # 实际应返回分割结果路径和数据
#     return "output/segmentation/result.dcm", None

# def add_background_roi(segmentation_path: str) -> Tuple[str, Dict[str, Any]]:
#     """模拟添加背景ROI"""
#     print(f"--- 模拟 add_background_roi: 添加本底ROI")
#     # 模拟 roi_data: 包含左右肾、左右背景的 ROI 掩码信息
#     roi_data = {'leftKidneyROI': None, 'rightKidneyROI': None, 'leftBackgroundROI': None, 'rightBackgroundROI': None}
#     return "output/ROI/roi_image.png", roi_data

# def display_dicom_with_roi_overlay(resampled_dcm_path: str, roi_data: Dict[str, Any]) -> str:
#     """模拟生成带ROI叠加的图像"""
#     print(f"--- 模拟 display_dicom_with_roi_overlay: 生成叠加图")
#     return "output/ROI/overlay_image.png"

# def graph(dicom_path: str, roi_data: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
#     """模拟计算肾脏计数和生成时间-计数曲线"""
#     print(f"--- 模拟 graph: 计算计数和生成曲线")
#     # 模拟计数结果 (假设为整数)
#     left_k_count = 150000
#     right_k_count = 140000
#     left_b_count = 5000
#     right_b_count = 4500
#     return left_k_count, right_k_count, left_b_count, right_b_count, "output/counts_time_graph.png"
# ----------------------------------------------------------------------


class DicomProcessor:
    """
    封装DICOM处理、深度计算和GFR计算的类，管理内部状态。
    """
    
    def __init__(self):
        # 路径配置
        self.UPLOAD_FOLDER = str(UPLOAD_DIR) 
        self.CONVERTED_FOLDER_NAME = 'converted' # 仅保留名称 (可选)
        self.SEGMENTED_FOLDER_NAME = 'ROI'
        self.ALLOWED_EXTENSIONS = {'dcm'}
        
        # 全局状态变量
        self.kidney_depths: Dict[str, Optional[float]] = {
            'leftDepth': None,
            'rightDepth': None,
            'LiLeftKidneyDepth': None,
            'LiRightKidneyDepth': None
        }
        self.last_patient_name: Optional[str] = None
        self.last_manufacturer: Optional[str] = None # 上一个病人的设备厂商
        self.last_kidney_counts: Dict[str, int] = {} # 上一个病人的肾脏计数信息
        self.last_patient_info: Dict[str, Any] = {}  # 存储病人信息

        self._create_required_directories()
        
    def _allowed_file(self, filename: str) -> bool:
        """检查文件扩展名"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def _create_required_directories(self):
        """创建处理DICOM文件所需的所有目录"""
        # 使用 ALL_OUTPUT_DIRS 常量列表
        for directory_path in ALL_OUTPUT_DIRS:
            try:
                # 优先使用 Pathlib
                directory_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                # 兜底处理
                os.makedirs(str(directory_path), exist_ok=True)

    

    def extract_patient_info(self, dicom_path: str) -> Dict[str, Any]:
        """从DICOM文件中提取患者信息"""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            
            # 处理性别信息
            sex = dicom_data.get('PatientSex', 'N/A')
            if sex == 'M':
                sex = '男'
            elif sex == 'F':
                sex = '女'
            
            # 处理年龄信息
            age = dicom_data.get('PatientAge', 'N/A')
            if age and isinstance(age, str):
                age = re.sub(r'\D', '', age).lstrip('0')  # 移除非数字字符
            
            # 处理患者姓名，移除分隔符
            patient_name = 'N/A'
            if 'PatientName' in dicom_data:
                raw_name = str(dicom_data.PatientName)
                patient_name = raw_name.replace('^', '')
            
            # 构建患者信息字典
            return {
                'name': patient_name,
                'sex': sex,
                'age': age,
                'height': dicom_data.get('PatientSize', 'N/A'),
                'weight': dicom_data.get('PatientWeight', 'N/A'),
                'manufacturer': dicom_data.get('Manufacturer', 'N/A'),
                'date': dicom_data.get('StudyDate', 'N/A')
            }
        except Exception as e:
            print(f"提取患者信息时出错: {e}")
            return {
                'name': 'N/A', 'sex': 'N/A', 'age': 'N/A',
                'height': 'N/A', 'weight': 'N/A', 
                'manufacturer': 'N/A', 'date': 'N/A'
            }

    def _process_kidney_dicom(self, dicom_path: str, patient_name: str , frame_number: int = 61, model_path: str = LOCAL_DICOM_MODEL_PATH) -> Dict[str, Any]:
        """
        处理肾动态扫描DICOM图像，进行肾脏分割和分析
        
        返回: 包含处理结果路径和肾脏计数信息的字典
        """
        # 注意: 依赖于外部导入的函数 (preprocess, predict, ROI, overlay, graph)
        # 实际代码中需要确保这些模块可用
        
        print(f"步骤1: 检查DICOM类型并处理: {dicom_path}")
        dicom_data = pydicom.dcmread(dicom_path)
        is_multiframe = dicom_data.get('NumberOfFrames', 1) > 1
        
        if is_multiframe:
            print(f"检测到多帧DICOM，提取第{frame_number}帧...")
            original_dcm_path, _ = extract_frame(dicom_path, frame_number)
            # original_dcm_path = dicom_path # 模拟，实际应为提取后的单帧路径
        else:
            print("检测到单帧DICOM，直接处理...")
            original_dcm_path = dicom_path
        
        # 使用 CONVERTED_OUTPUT_DIR
        png_filename = f"{patient_name}_{frame_number}_resampled.png" 
        png_path_abs = str(CONVERTED_OUTPUT_DIR / png_filename) # 完整的绝对路径字符串

        print("步骤2: 重采样DICOM...")
        resampled_dcm_path, _, png_path = resample_dicom(original_dcm_path, output_png_path=png_path_abs, order=1)
        # resampled_dcm_path = "output/resampled.dcm" # 模拟结果
        # png_path = os.path.join(self.CONVERTED_FOLDER, "image.png") # 模拟结果
        
        print("步骤3-4: 进行肾脏分割和ROI添加...")
        overlay_filename = f"{patient_name}_roi_overlay.png"
        overlay_path_abs = str(SEGMENTED_OUTPUT_DIR / overlay_filename)
        segmentation_path, _ = segment_kidney(resampled_dcm_path, model_path, img_size=224, num_classes=3)
        roi_path, roi_data = add_background_roi(segmentation_path)
        # segmentation_path = "output/segmentation/result.dcm" # 模拟结果
        # _, roi_data = add_background_roi(segmentation_path) # 模拟调用
        
        overlay_path = display_dicom_with_roi_overlay(resampled_dcm_path, roi_data, output_path=overlay_path_abs)
        # overlay_path = os.path.join(self.SEGMENTED_FOLDER, "overlay_image.png") # 模拟结果
        
        print("步骤5: 计算肾脏计数和生成曲线...")
        graph_filename = f"{patient_name}_counts_curve.png"
        graph_path_abs = str(GRAPH_OUTPUT_DIR / graph_filename)
        left_kidney_count, right_kidney_count, left_background_count, right_background_count, graph_path = graph(
            dicom_path, 
            roi_data,
            output_path=graph_path_abs # 传入绝对路径
        )
        # left_kidney_count, right_kidney_count, left_background_count, right_background_count, graph_path = graph(dicom_path, roi_data) # 模拟调用
        
        print(f"左肾计数: {left_kidney_count}, 右肾计数: {right_kidney_count}")
        
        kidney_counts = {
            'leftKidneyCount': int(left_kidney_count),
            'rightKidneyCount': int(right_kidney_count),
            'leftBackgroundCount': int(left_background_count),
            'rightBackgroundCount': int(right_background_count)
        }
        
        return {
            'png_path': png_path,
            'overlay_path': overlay_path,
            'graph_path': graph_path, # 返回绝对路径
            'kidney_counts': kidney_counts
        }

    def process_dynamic_study_dicom(self, dicom_path: str) -> Dict[str, Any]:
        """
        处理肾动态显像的DICOM文件，包括信息提取和图像处理。
        
        参数:
            dicom_path: 肾动态显像DICOM文件路径。
            
        返回:
            Dict: 包含患者信息、肾脏计数和图像路径的字典。
        """
        if not os.path.exists(dicom_path) or not self._allowed_file(dicom_path):
            return {'success': False, 'message': '文件不存在或格式不支持'}

        try:
            # 1. 提取病人信息
            patient_info = self.extract_patient_info(dicom_path)
            
            # 2. 更新全局变量
            self.last_patient_info = patient_info
            self.last_manufacturer = patient_info['manufacturer']
            
            # 3. 处理DICOM序列并进行肾脏分析
            result = self._process_kidney_dicom(dicom_path, patient_info['name'])
            
            # 4. 更新全局肾脏计数变量
            self.last_kidney_counts = result['kidney_counts']
            
            # 5. 构建响应数据
            return {
                'success': True,
                'imageUrl': result['png_path'],
                'overlayUrl': result['overlay_path'],
                'countsTimeUrl': result['graph_path'],
                'patientInfo': patient_info,
                'kidneyCounts': result['kidney_counts']
            }
                
        except Exception as e:
            print(f"处理DICOM文件时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'message': f'处理DICOM文件时发生错误: {str(e)}'}

    def _calculate_li_depth(self, height_cm: float, weight_kg: float, age_y: int, sex: str) -> Tuple[Optional[float], Optional[float]]:
        """
        计算李氏肾脏深度。
        参数单位: height_cm (cm), weight_kg (kg), age_y (岁), sex ('M'/'F')
        返回单位: (mm)
        """
        LiKidney_L, LiKidney_R = None, None
        
        if sex == 'F':  # 女性
            LiKidney_L = 0.013 * age_y - 0.044 * height_cm + 0.087 * weight_kg + 7.951
            LiKidney_R = 0.005 * age_y - 0.035 * height_cm + 0.082 * weight_kg + 7.266
        elif sex == 'M':  # 男性
            LiKidney_L = 0.013 * age_y + 0.117 - 0.044 * height_cm + 0.087 * weight_kg + 7.951
            LiKidney_R = 0.005 * age_y + 0.013 - 0.035 * height_cm + 0.082 * weight_kg + 7.266

        if LiKidney_L is not None and LiKidney_R is not None:
            return round(LiKidney_L * 10, 2), round(LiKidney_R * 10, 2)  # 转换为mm
        else:
            return None, None

    def _get_patient_anthropometric_data(self, dicom_data: pydicom.Dataset) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[str], bool]:
        """
        从CT DICOM或动态显像历史数据中获取计算李氏公式所需的患者人体测量数据。
        返回: (身高_cm, 体重_kg, 年龄_y, 性别_M_F, 是否使用历史数据)
        """
        ct_info = {
            'height': dicom_data.get('PatientSize', 'N/A'),
            'weight': dicom_data.get('PatientWeight', 'N/A'),
            'age': dicom_data.get("PatientAge", 'N/A'),
            'sex': dicom_data.get("PatientSex", 'N/A')
        }
        
        # 尝试使用CT信息，不足时使用历史信息补充
        final_info = {}
        using_previous_info = False

        keys = ['height', 'weight', 'age', 'sex']
        for key in keys:
            ct_value = ct_info[key]
            # 优先使用CT信息
            if ct_value != 'N/A' and ct_value is not None:
                final_info[key] = ct_value
            # 尝试使用历史信息补充
            elif self.last_patient_info.get(key) not in ['N/A', None]:
                final_info[key] = self.last_patient_info[key]
                using_previous_info = True

        # 格式化和转换数据
        height_value = final_info.get('height')
        weight_value = final_info.get('weight')
        patient_age_str = final_info.get('age')
        patient_sex = final_info.get('sex')
        
        patient_height_cm: Optional[float] = None
        patient_weight_kg: Optional[float] = None
        patient_age_y: Optional[int] = None
        patient_sex_M_F: Optional[str] = None
        
        # 1. 处理身高 (转为cm)
        try:
            if height_value is not None:
                height_val = float(height_value) if isinstance(height_value, str) and height_value.isdigit() else float(height_value)
                patient_height_cm = height_val * 100 if height_val < 10 else height_val
        except (ValueError, TypeError):
            pass
            
        # 2. 处理体重 (kg)
        try:
            if weight_value is not None:
                patient_weight_kg = float(weight_value) if isinstance(weight_value, str) and weight_value.replace('.', '', 1).isdigit() else float(weight_value)
        except (ValueError, TypeError):
            pass
            
        # 3. 处理年龄 (岁)
        try:
            if patient_age_str is not None:
                if isinstance(patient_age_str, str):
                    match = re.search(r'\d+', patient_age_str)
                    if match: patient_age_y = int(match.group())
                else:
                    patient_age_y = int(patient_age_str)
        except (ValueError, TypeError):
            pass
            
        # 4. 处理性别 (M/F)
        if patient_sex in ['M', '男']:
            patient_sex_M_F = 'M'
        elif patient_sex in ['F', '女']:
            patient_sex_M_F = 'F'

        return patient_height_cm, patient_weight_kg, patient_age_y, patient_sex_M_F, using_previous_info


    def process_depth_dicom(self, dicom_path: str) -> Dict[str, Any]:
        """
        处理单个CT DICOM文件，进行模型深度计算和李氏公式深度计算。
        
        参数:
            dicom_path: CT DICOM文件路径。
            
        返回:
            Dict: 包含深度计算结果和图像路径的字典。
        """
        if not os.path.exists(dicom_path):
            return {'success': False, 'message': f"路径不存在: {dicom_path}"}
        
        # final_dcm_path = dicom_path

        # # 1. 检查输入是文件夹还是文件
        # if os.path.isdir(dicom_path):
        #     # 如果是文件夹，调用 find_deepest_kidney_slice
        #     print(f"输入是文件夹，开始寻找最深切片...")
            
            
        #     # 找到最深切片的 DICOM 文件路径
        #     # 注意：find_deepest_kidney_slice 可能返回 (文件路径, 结果)
        #     deepest_slice_result = find_deepest_kidney_slice(dicom_path, MODEL_PATH)
            
        #     # 假设 find_deepest_kidney_slice 返回第一个元素是最佳切片的完整路径
        #     if deepest_slice_result and deepest_slice_result[1]:
        #         dicom_path = os.path.join(dicom_path, deepest_slice_result[1])
        #         print(f"找到最深切片: {final_dcm_path}")
        #     else:
        #         return {'success': False, 'message': f"从文件夹 {dicom_path} 中未找到包含肾脏的最深切片。"}
            
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            # dicom_data.pixel_array # 原代码中读取但未使用的部分，此处省略
            
            # 1. 检查和更新患者姓名，决定是否清空历史深度
            current_patient_name = str(dicom_data.get('PatientName', 'N/A')).upper().replace('^', '')
            last_name_normalized = str(self.last_patient_name).upper().replace('^', '') if self.last_patient_name else None
            
            if last_name_normalized != current_patient_name:
                print(f"检测到新患者: {current_patient_name}，重置肾脏深度信息")
                self.kidney_depths = {k: None for k in self.kidney_depths}
                self.last_patient_name = str(dicom_data.get('PatientName'))
                
            # 2. 获取人体测量数据并计算李氏深度
            (height, weight, age, sex, using_previous_info) = self._get_patient_anthropometric_data(dicom_data)
            
            self.kidney_depths['LiLeftKidneyDepth'] = None
            self.kidney_depths['LiRightKidneyDepth'] = None
            
            if all([height, weight, age, sex]):
                Li_L, Li_R = self._calculate_li_depth(height, weight, age, sex)
                self.kidney_depths['LiLeftKidneyDepth'] = Li_L
                self.kidney_depths['LiRightKidneyDepth'] = Li_R
                print(f"李氏深度计算完成 (使用动态显像数据: {using_previous_info})")
            else:
                missing = [k for k, v in zip(['身高', '体重', '年龄', '性别'], [height, weight, age, sex]) if v is None]
                print(f"缺少计算李氏肾脏深度所需的患者信息: {', '.join(missing)}")
                
            # 3. 模型计算肾脏深度 (单位mm)
            patient_name_safe = current_patient_name if current_patient_name != 'N/A' else 'UNKNOWN'
            viz_filename = f"{patient_name_safe}_depth_visualization.png"
            viz_path_abs = str(DEPTH_OUTPUT_DIR / viz_filename) # 使用常量

            result, viz_path = CT(dicom_path, output_path=viz_path_abs) # 假设 CT 函数返回 mm 单位的深度
            left_depth_model = result.get('L')
            right_depth_model = result.get('R')

            # 更新全局变量中的肾脏深度信息
            if left_depth_model is not None and left_depth_model != 'N/A':
                self.kidney_depths['leftDepth'] = round(float(left_depth_model), 2)
            if right_depth_model is not None and right_depth_model != 'N/A':
                self.kidney_depths['rightDepth'] = round(float(right_depth_model), 2)

            # 4. 构建返回结果
            kidneyDepth = {k: self.kidney_depths[k] for k in self.kidney_depths}
            
            if kidneyDepth['leftDepth'] is None and kidneyDepth['rightDepth'] is None:
                return {'success': False, 'message': '未检测到肾脏深度，请检查CT图像。'}
            else:
                message_parts = []
                if kidneyDepth['leftDepth'] is not None and kidneyDepth['rightDepth'] is not None:
                    message_parts.append("已检测到双侧肾脏深度")
                elif kidneyDepth['leftDepth'] is not None:
                    message_parts.append("只检测到左肾深度")
                elif kidneyDepth['rightDepth'] is not None:
                    message_parts.append("只检测到右肾深度")
                    
                if kidneyDepth['LiLeftKidneyDepth'] is not None and kidneyDepth['LiRightKidneyDepth'] is not None:
                    source = "肾动态显像" if using_previous_info else "当前CT图像"
                    message_parts.append(f"已使用{source}数据计算李氏深度")
                
                return {
                    'success': True, 
                    'depthImageUrl': os.path.join(self.BASE_URL, viz_path), 
                    'message': "，".join(message_parts) + "。",
                    'kidneyDepth': kidneyDepth,
                    'usingDynamicPatientInfo': using_previous_info
                }

        except Exception as e:
            print(f"处理CT DICOM文件时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'message': f'处理CT DICOM文件时发生错误: {str(e)}'}

    def process_depth_dicomfile(self, dcm_folder_path: str) -> Dict[str, Any]:
        """
        处理整个CT DICOM文件夹，找到肾脏深度最大的切片，并保存深度值。
        
        Args:
            dcm_folder_path (str): 包含CT DICOM文件的文件夹路径。

        Returns:
            Dict: 包含处理结果和最深切片深度信息的字典。
        """
        print(f"开始处理 CT 序列文件夹: {dcm_folder_path}")
        
        try:
            # 调用 CT.py 中封装的函数来遍历文件夹并找到最深切片
            # 所有的中间结果（YOLO标签、深度计算）将保存在一个临时目录
            output_dir = "depth_series_analysis" # 或使用一个动态的路径
            
            # 替换 find_deepest_kidney_slice 的功能
            deepest_result = find_deepest_slice_in_series(dcm_folder_path, output_dir)
            
            deepest_slice_filename = deepest_result.get('deepest_slice')
            
            # 调用 process_depth_dicom 处理最深切片
            if deepest_slice_filename:
                # ------------------- 关键修改 -------------------
                # 拼接待处理文件的完整路径
                deepest_slice_full_path = os.path.join(dcm_folder_path, deepest_slice_filename)
                
                # 调用 process_depth_dicom 处理完整路径
                result = self.process_depth_dicom(deepest_slice_full_path) 
                # ------------------------------------------------
            else:
                return {
                    'success': False,
                    'message': '未找到有效的肾脏切片或深度信息。',
                    'kidneyDepth': self.kidney_depths
                }
            
            # 返回 process_depth_dicom 的结果
            return result

        except Exception as e:
            print(f"处理 CT 序列时发生错误: {e}")
            return {
                'success': False,
                'message': f'处理 CT 序列时发生错误: {e}',
                'kidneyDepth': self.kidney_depths
            }
        
    def upload_depth_and_calculate_li(self, left_depth: Optional[float], right_depth: Optional[float], 
                                     height_m: float, weight_kg: float, age_y: int, sex_cn: str) -> Dict[str, Any]:
        """
        手动上传肾脏深度和患者信息，并计算李氏肾脏深度。
        
        参数:
            left_depth: 左肾深度 (mm)。
            right_depth: 右肾深度 (mm)。
            height_m: 身高 (m)。
            weight_kg: 体重 (kg)。
            age_y: 年龄 (岁)。
            sex_cn: 性别 ('男' 或 '女')。
            
        返回:
            Dict: 包含李氏深度计算结果的字典。
        """
        print("开始上传肾脏深度并计算李氏深度...")
        
        # 1. 更新模型/手动深度
        if left_depth is not None:
            self.kidney_depths['leftDepth'] = float(left_depth)
        if right_depth is not None:
            self.kidney_depths['rightDepth'] = float(right_depth)
        
        # 2. 检查和转换病人信息
        patient_height_cm = height_m * 100 
        patient_weight = float(weight_kg)
        patient_age = int(age_y)
        patient_sex_M_F = 'F' if sex_cn == '女' else ('M' if sex_cn == '男' else None)

        if not patient_height_cm or not patient_weight or not patient_age or not patient_sex_M_F:
            return {'success': False, 'message': '缺少计算李氏深度所需的病人信息'}

        # 3. 计算李氏肾脏深度
        Li_L, Li_R = self._calculate_li_depth(patient_height_cm, patient_weight, patient_age, patient_sex_M_F)
        
        if Li_L is not None and Li_R is not None:
            self.kidney_depths['LiLeftKidneyDepth'] = Li_L
            self.kidney_depths['LiRightKidneyDepth'] = Li_R
            
            return {
                'success': True,
                'message': '肾脏深度和李氏深度计算成功',
                'LiLeftKidneyDepth': Li_L,
                'LiRightKidneyDepth': Li_R
            }
        else:
            return {'success': False, 'message': '无法计算李氏肾脏深度'}

    def calculate_gfr(self, depth_method: str = 'model') -> Dict[str, Any]:
        """
        计算 GFR。
        
        参数:
            depth_method: 'model' (模型/手动输入的深度) 或 'li' (李氏公式深度)。
            
        返回:
            Dict: 包含两种方法计算的 GFR 结果。
        """
        print("开始计算 GFR...")
        
        left_depth_model = self.kidney_depths.get('leftDepth')
        right_depth_model = self.kidney_depths.get('rightDepth')
        Li_left_depth = self.kidney_depths.get('LiLeftKidneyDepth')
        Li_right_depth = self.kidney_depths.get('LiRightKidneyDepth')

        if not self.last_manufacturer or not self.last_kidney_counts or not all([left_depth_model, right_depth_model, Li_left_depth, Li_right_depth]):
            return {'success': False, 'message': '缺少计算 GFR 所需的数据（厂商信息、肾脏计数或所有深度数据）。'}
        
        # 根据 manufacturer 判断 injection_count
        manufacturer = self.last_manufacturer
        if 'GE MEDICAL SYSTEMS' in manufacturer:
            injection_count = 183916
        elif 'SIEMENS NM' in manufacturer:
            injection_count = 139857
        else:
            print("厂商信息未知，使用 GE 默认值。")
            injection_count = 183916

        print(f"使用厂商: {manufacturer}, 注入计数: {injection_count}")

        left_count = self.last_kidney_counts.get('leftKidneyCount', 0) - self.last_kidney_counts.get('leftBackgroundCount', 0)
        right_count = self.last_kidney_counts.get('rightKidneyCount', 0) - self.last_kidney_counts.get('rightBackgroundCount', 0)

        if left_count <= 0 or right_count <= 0:
            return {'success': False, 'message': '肾脏净计数小于或等于零，无法计算 GFR。'}
            
        u = 0.153
        
        # --- 模型/手动深度计算 GFR ---
        eL = math.exp(-u * left_depth_model * 0.1)
        eR = math.exp(-u * right_depth_model * 0.1)

        # GFR 公式: ((计数 * 0.001) / 衰减因子 / (注入计数 * 0.001 * 6)) * 100 * 9.81270 - 6.82519
        GFR_left = ((left_count * 0.001) / eL / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
        GFR_right = ((right_count * 0.001) / eR / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
        GFR_total = GFR_left + GFR_right
        
        # --- 李氏公式深度计算 GFR ---
        eL_Li = math.exp(-u * Li_left_depth * 0.1)
        eR_Li = math.exp(-u * Li_right_depth * 0.1)

        Li_GFR_left = ((left_count * 0.001) / eL_Li / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
        Li_GFR_right = ((right_count * 0.001) / eR_Li / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
        Li_GFR_total = Li_GFR_left + Li_GFR_right

        print("GFR计算完成。")
        
        return {
            'success': True,
            'gfr': {
                'leftGFR': round(GFR_left, 2),
                'rightGFR': round(GFR_right, 2),
                'totalGFR': round(GFR_total, 2),
                'LiLeftGFR': round(Li_GFR_left, 2),
                'LiRightGFR': round(Li_GFR_right, 2),
                'LiTotalGFR': round(Li_GFR_total, 2),
            }
        }

# --- 示例用法 (作为模块被调用时) ---
if __name__ == '__main__':
    # 假设的 DICOM 文件路径 (需要替换为实际文件)
    DYNAMIC_DICOM_PATH = "path/to/kidney_dynamic_study.dcm"
    CT_DICOM_PATH = "/home/kevin/Documents/GFR/CT_all/CT11"
    
    # 1. 初始化处理器
    processor = DicomProcessor(upload_folder='temp_uploads', base_url='/static')
    
    print("\n--- 1. 处理肾动态显像 DICOM ---")
    # 模拟文件存在
    if not os.path.exists(DYNAMIC_DICOM_PATH):
        print(f"注意: 动态显像文件 '{DYNAMIC_DICOM_PATH}' 不存在，使用模拟数据。")
        # 模拟设置状态以进行后续 GFR 计算
        processor.last_manufacturer = 'GE MEDICAL SYSTEMS'
        processor.last_kidney_counts = {'leftKidneyCount': 150000, 'rightKidneyCount': 140000, 
                                        'leftBackgroundCount': 5000, 'rightBackgroundCount': 4500}
        processor.last_patient_info = {'name': 'TESTPATIENT', 'sex': '男', 'age': '45', 'height': 1.75, 'weight': 70.5, 'manufacturer': 'GE MEDICAL SYSTEMS', 'date': '20250101'}
        dynamic_result = {'success': True, 'message': '模拟动态显像处理成功'}
    else:
        dynamic_result = processor.process_dynamic_study_dicom(DYNAMIC_DICOM_PATH)
    
    print("动态显像处理结果:", dynamic_result)
    
    print("\n--- 2. 处理 CT 深度 DICOM ---")
    # 模拟文件存在
    if not os.path.exists(CT_DICOM_PATH):
        print(f"注意: CT 文件 '{CT_DICOM_PATH}' 不存在，使用模拟深度数据。")
        # 模拟设置状态以进行后续 GFR 计算
        processor.kidney_depths['leftDepth'] = 75.2
        processor.kidney_depths['rightDepth'] = 70.8
        
        # 模拟李氏深度计算 (需依赖 patient_info)
        height_cm = 175.0
        weight_kg = 70.5
        age_y = 45
        sex_M_F = 'M'
        Li_L, Li_R = processor._calculate_li_depth(height_cm, weight_kg, age_y, sex_M_F)
        processor.kidney_depths['LiLeftKidneyDepth'] = Li_L
        processor.kidney_depths['LiRightKidneyDepth'] = Li_R
        depth_result = {'success': True, 'message': '模拟深度计算成功', 'kidneyDepth': processor.kidney_depths}
    else:
        depth_result = processor.process_depth_dicom(CT_DICOM_PATH)
    
    print("CT 深度处理结果:", depth_result)

    print("\n--- 3. 计算 GFR (使用模型/CT深度) ---")
    if processor.kidney_depths['leftDepth'] and processor.last_kidney_counts:
        gfr_result = processor.calculate_gfr(depth_method='model')
        print("GFR 计算结果:", gfr_result)
    else:
        print("GFR 计算条件不足。")
        
    print("\n--- 4. 演示手动上传深度和计算李氏深度 ---")
    manual_depth_result = processor.upload_depth_and_calculate_li(
        left_depth=72.0, right_depth=68.0, 
        height_m=1.80, weight_kg=80.0, age_y=50, sex_cn='男'
    )
    print("手动上传结果:", manual_depth_result)
    
    print("\n--- 5. 计算 GFR (使用手动/李氏深度) ---")
    if processor.kidney_depths['leftDepth'] and processor.last_kidney_counts:
        gfr_result_manual = processor.calculate_gfr(depth_method='li')
        print("GFR (基于手动/新李氏) 计算结果:", gfr_result_manual)
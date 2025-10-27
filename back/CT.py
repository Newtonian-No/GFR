"""
从CT图像中检测肾脏并测量肾脏深度
"""
from pathlib import Path
from typing import Dict, Any
import json
from back.detect import detect , detect_folder
import os
from back.anatomical_segmentation import kidneydepth ,find_deepest_slice

def CT(file_path, output_path: str = None):

    if output_path is None:
        # 如果未传入，使用原有的相对路径作为备用（但不推荐）
        output_dir = os.path.join("converted", f"{patient_name}_{frame_number}_resampled.png")
    else:
        # 使用上层传入的绝对路径
        output_dir = output_path 

    # result_paths = detect(source=file_path) 
                              
    # Debugging output
    print(f"Result paths: {result_paths}")

    txt_path = result_paths
    txt_file_path = os.path.join('static/results/exp/labels', os.path.basename(txt_path))

    # Debugging output
    print(f"TXT file path: {txt_file_path}")

    dicom_file_path = file_path
    print("测量肾脏深度")
    result_data, viz_path= kidneydepth(dicom_file_path, txt_file_path, output_dir=output_dir)
    print("测量完毕")

    # 处理结果数据
    depth_left = result_data.get('L', 0)  # 默认左侧深度为0
    depth_right = result_data.get('R', 0)  # 默认右侧深度为0

    # 如果结果数据中只包含一边的深度，将另一边深度设置为0
    if 'L' not in result_data:
        depth_left = "N/A"
    if 'R' not in result_data:
        depth_right = "N/A"

    # 对深度数据取绝对值（如果是数字）
    if isinstance(depth_left, (int, float)):
        depth_left = abs(depth_left)
    if isinstance(depth_right, (int, float)):
        depth_right = abs(depth_right)

    # 将处理后的深度数据返回
    result_data = {'L': depth_left, 'R': depth_right}

    return result_data, viz_path

def find_deepest_slice_in_series(dicom_folder_path, output_base_dir="depth_series_analysis") -> Dict[str, Any]:
    """
    处理整个DICOM文件夹，找到肾脏深度最大的切片。

    Args:
        dicom_folder_path (str): 包含DICOM文件的文件夹路径。
        output_base_dir (str): 用于保存检测结果和深度分析的基目录。

    Returns:
        Dict[str, Any]: 包含最深切片文件名和左右深度信息的字典。
    """
    
    # 1. 设置 YOLO 输出路径
    # YOLO 标签将保存在 output_base_dir/exp_series/labels
    yolo_project_name = "YOLO_series_results" # 为 YOLO 结果定义一个名称
    
    # 2. 调用 detect.py 中的批量检测函数，生成所有切片的YOLO标签
    yolo_labels_folder = detect_folder(
        source_folder=dicom_folder_path, 
        project=output_base_dir, 
        name=yolo_project_name
    )
    
    if not yolo_labels_folder:
        print("Error: YOLO 标签生成失败。")
        return {'deepest_slice': None, 'max_depth_mm': 0.0, 'L': "N/A", 'R': "N/A"}

    # 3. 调用 anatomical_segmentation.py 中的深度查找函数
    # output_base_dir 传入作为深度分析的总输出路径
    final_result = find_deepest_slice(
        dicom_folder=dicom_folder_path, 
        yolo_labels_folder=str(yolo_labels_folder), 
        output_base_dir=output_base_dir
    )
    
    # 4. 从 find_deepest_slice 的结果中获取最深切片的左右肾深度
    deepest_slice = final_result.get('deepest_slice')
    max_depth_mm = final_result.get('max_depth_mm')

    L_depth = "N/A"
    R_depth = "N/A"

    if deepest_slice:
        # find_deepest_slice 内部将最深切片的结果保存在了 output_base_dir/base_name/base_name_results.json
        base_name = Path(deepest_slice).stem
        results_json_path = Path(output_base_dir) / base_name / f"{base_name}_results.json"
        
        if results_json_path.exists():
            try:
                with open(results_json_path, 'r') as f:
                    deepest_slice_results = json.load(f)
                    # 假设 JSON 包含 'L' 和 'R' 键
                    L_depth = deepest_slice_results.get('L', "N/A")
                    R_depth = deepest_slice_results.get('R', "N/A")
            except Exception as e:
                print(f"Error reading deepest slice JSON: {e}")
    print("find_deepest_slice_in_series结束")
    # 返回最终结果，包含最深切片名和左右肾深度
    return {
        'deepest_slice': deepest_slice,
        'max_overall_depth_mm': max_depth_mm,
        'L': L_depth,
        'R': R_depth
    }


#CT('FZL-ImageFileName59.dcm')
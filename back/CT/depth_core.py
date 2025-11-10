# depth_core.py
import os
import pydicom
import cv2
import json
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from ..constants import ORIGINAL_CT_DIR , ORIGINAL_PNG_DIR , OVERLAY_PNG_DIR , YOLO_LABELS_DIR , ANALYSIS_RESULTS_DIR , DETECT_MODEL_WEIGHTS_PATH

def load_dicom_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    image = dicom_data.pixel_array
    slope = getattr(dicom_data, 'RescaleSlope', 1)
    intercept = getattr(dicom_data, 'RescaleIntercept', 0)
    image = image * slope + intercept
    display_image = image.copy()
    display_image = np.clip(display_image, -200, 400)
    display_image = ((display_image + 200) / 600 * 255).astype('uint8')
    return image, display_image

def segment_body(image):
    body_mask = (image > -200).astype(np.uint8) * 255
    original_mask = body_mask.copy()
    kernel = np.ones((5, 5), np.uint8)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
    eroded_mask = cv2.erode(body_mask, np.ones((7, 7), np.uint8), iterations=3)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded_mask, connectivity=8)
    if num_labels > 1:
        max_size = 0
        max_label = 0
        for i in range(1, num_labels):
            size = stats[i, cv2.CC_STAT_AREA]
            if size > max_size:
                max_size = size
                max_label = i
        body_mask = (labels == max_label).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        dilated_body = cv2.dilate(body_mask, kernel, iterations=4)
        body_mask = cv2.bitwise_and(dilated_body, original_mask)
    return body_mask

def extract_body_contour(body_mask):
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    body_contour = max(contours, key=cv2.contourArea)
    contour_image = np.zeros_like(body_mask)
    cv2.drawContours(contour_image, [body_contour], 0, 255, 1)
    return contour_image

def get_coordinates(txt_path, image_width, image_height):
    coordinates = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    obj_class, x_center, y_center, _, _, *_ = map(float, parts[:6])
                    x_pixel = round(x_center * image_width)
                    y_pixel = round(y_center * image_height)
                    coordinates.append((int(obj_class), x_pixel, y_pixel))
                except ValueError as e:
                    print(f"Error: Could not convert parts to float: {e}")
                    continue
            else:
                print(f"Error: Line does not have enough parts: {len(parts)} parts found.")
    return coordinates

def count_pixel(file_path):
    file = sitk.ReadImage(file_path)
    return file.GetSpacing()[1]

def find_posterior_boundary(contour_image, kidney_x, kidney_y):
    height, width = contour_image.shape
    if kidney_x < 0 or kidney_x >= width or kidney_y < 0 or kidney_y >= height:
        return None
    
    sample_width = 10
    boundary_points = []
    
    for offset in range(-sample_width, sample_width + 1):
        sample_x = kidney_x + offset
        
        if 0 <= sample_x < width:
            for y in range(kidney_y, height):
                if contour_image[y, sample_x] > 0:
                    boundary_points.append(y)
                    break
    
    if len(boundary_points) < 5:
        return None
    
    boundary_y = int(np.median(boundary_points))
    return boundary_y


def measure_kidney_depth(dicom_path: Path, txt_path: Path, is_deepest: bool = False) -> dict:
    """
    测量单个切片的肾脏深度，并生成可视化叠加图和 JSON 结果。

    Args:
        dicom_path: 原始 DICOM 文件的路径。
        txt_path: 对应的 YOLO 标签文件路径。
        is_deepest: 是否为系列中的最深切片（影响最终可视化路径）。

    Returns:
        dict: 包含左右肾深度（mm）的结果字典。
    """
    file_base_name = dicom_path.stem
    
    # 加载DICOM图像
    hu_image, display_image = load_dicom_image(str(dicom_path))
    
    # 分割身体 & 提取轮廓
    body_mask = segment_body(hu_image)
    posterior_contour = extract_body_contour(body_mask)
    
    # 获取肾脏坐标
    coordinates = get_coordinates(str(txt_path), display_image.shape[1], display_image.shape[0])
    
    # 计算结果
    results = {}
    pixel_spacing = count_pixel(str(dicom_path))
    
    visualization = cv2.cvtColor(display_image.copy(), cv2.COLOR_GRAY2BGR)
    colored_contour = np.zeros_like(visualization)
    if posterior_contour is not None:
        colored_contour[posterior_contour > 0] = [0, 255, 255]
        visualization = cv2.addWeighted(visualization, 1, colored_contour, 0.3, 0)
    
    for obj_class, x_coordinate, y_coordinate in coordinates:
        if posterior_contour is not None:
            posterior_boundary = find_posterior_boundary(posterior_contour, x_coordinate, y_coordinate)
            
            label = 'L' if obj_class == 0 else 'R'
            
            if posterior_boundary is not None:
                difference = (posterior_boundary - y_coordinate) * pixel_spacing
                results[label] = round(abs(difference), 2)
                
                # 可视化绘制
                cv2.circle(visualization, (x_coordinate, y_coordinate), 5, (0, 255, 0), -1)
                cv2.circle(visualization, (x_coordinate, posterior_boundary), 5, (0, 0, 255), -1)
                cv2.line(visualization, (x_coordinate, y_coordinate), (x_coordinate, posterior_boundary), (255, 0, 0), 2)
                cv2.putText(visualization, 
                            f"{results[label]:.1f}mm", 
                            (x_coordinate + 10, (y_coordinate + posterior_boundary)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                results[label] = "N/A"
        else:
             results[label] = "N/A"

    # 保存 JSON 结果 (放在 ANALYSIS_RESULTS_DIR)
    json_path = ANALYSIS_RESULTS_DIR / f"{file_base_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # 保存可视化 PNG (放在 OVERLAY_PNG_DIR)
    if is_deepest:
        viz_path = OVERLAY_PNG_DIR / f"{file_base_name}_deepest_overlay.png"
    else:
        viz_path = OVERLAY_PNG_DIR / f"{file_base_name}_overlay.png"
        
    cv2.imwrite(str(viz_path), visualization)
    
    return results

def find_deepest_slice(dicom_folder: Path, yolo_labels_folder: Path) -> dict:
    """
    遍历文件夹中的所有切片，找到肾脏深度测量值最大的切片。

    Returns:
        dict: 包含最深切片名、最大深度以及该切片的左右深度。
    """
    dicom_files = list(dicom_folder.glob('*.dcm')) + list(dicom_folder.glob('*.dicom'))

    max_depth_mm = -1.0
    deepest_slice_name = None
    deepest_slice_results = {}

    for dicom_file_path in dicom_files:
        file_base_name = dicom_file_path.stem
        txt_file_path = yolo_labels_folder / f"{file_base_name}.txt"
        
        if not txt_file_path.is_file():
            continue

        try:
            # 测量当前切片深度 (使用 is_deepest=False, 结果会保存到 analysis_results)
            results = measure_kidney_depth(dicom_file_path, txt_file_path, is_deepest=False)
            
            current_max_depth = 0.0
            for depth in results.values():
                if isinstance(depth, (int, float)):
                    current_max_depth = max(current_max_depth, depth)

            if current_max_depth > max_depth_mm:
                max_depth_mm = current_max_depth
                deepest_slice_name = dicom_file_path.name
                deepest_slice_results = results

        except Exception as e:
            print(f"Error processing {dicom_file_path.name}: {e}")
            continue

    # 重新调用 measure_kidney_depth 以生成带有 "_deepest_overlay.png" 后缀的可视化图
    if deepest_slice_name:
        deepest_dcm_path = dicom_folder / deepest_slice_name
        deepest_txt_path = yolo_labels_folder / f"{Path(deepest_slice_name).stem}.txt"
        measure_kidney_depth(deepest_dcm_path, deepest_txt_path, is_deepest=True) 
        # print("DEBUG:最深切片路径：", deepest_dcm_path)

    return {
        'deepest_slice': deepest_slice_name,
        'max_overall_depth_mm': max_depth_mm,
        'L': deepest_slice_results.get('L', 'N/A'),
        'R': deepest_slice_results.get('R', 'N/A'),
    }
"""
肾脏深度测量函数
输入: DICOM文件和YOLOv5的.txt文件
输出: 包含身体掩码、轮廓、测量结果的JSON文件和可视化图像
"""
import os
import pydicom
import cv2
import json
import SimpleITK as sitk
import numpy as np
from skimage import filters, measure, morphology
import glob
from pathlib import Path

# 加载DICOM图片
def load_dicom_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    image = dicom_data.pixel_array
    # 保留原始HU值以便更好地分割不同组织
    slope = getattr(dicom_data, 'RescaleSlope', 1)
    intercept = getattr(dicom_data, 'RescaleIntercept', 0)
    image = image * slope + intercept
    
    # 归一化用于显示
    display_image = image.copy()
    display_image = np.clip(display_image, -200, 400)  # 软组织窗口
    display_image = ((display_image + 200) / 600 * 255).astype('uint8')  # 归一化到0-255
    
    return image, display_image

# 基于HU值的身体分割 - 改进版
def segment_body(image):
    """使用HU值阈值分割身体，并通过形态学操作分离身体与设备"""
    # 空气通常<-1000HU, 软组织通常>-200HU
    body_mask = (image > -200).astype(np.uint8) * 255
    
    # 创建副本用于分析
    original_mask = body_mask.copy()
    
    # 1. 形态学操作填充小孔
    kernel = np.ones((5, 5), np.uint8)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
    
    # 2. 尝试通过腐蚀操作分离身体与设备
    # 先腐蚀，再重建，这可以分离弱连接
    eroded_mask = cv2.erode(body_mask, np.ones((7, 7), np.uint8), iterations=3)
    
    # 3. 保留最大连通区域（身体）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded_mask, connectivity=8)
    
    if num_labels > 1:  # 至少有背景和一个前景
        # 排除背景（通常是第一个标签）
        max_size = 0
        max_label = 0
        for i in range(1, num_labels):
            size = stats[i, cv2.CC_STAT_AREA]
            if size > max_size:
                max_size = size
                max_label = i
        
        # 提取最大连通区域
        body_mask = (labels == max_label).astype(np.uint8) * 255
        
        # 4. 重建身体轮廓，但不扩张到设备区域
        kernel = np.ones((5, 5), np.uint8)
        dilated_body = cv2.dilate(body_mask, kernel, iterations=4)
        
        # 只保留原始掩码中有值的部分
        body_mask = cv2.bitwise_and(dilated_body, original_mask)
    
    return body_mask

# 优化的轮廓提取函数 - 直接从body_mask获取完整轮廓
def extract_body_contour(body_mask):
    """直接从身体掩码中提取完整轮廓"""
    # 提取轮廓
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # 提取最大轮廓
    body_contour = max(contours, key=cv2.contourArea)
    
    # 创建轮廓图像
    contour_image = np.zeros_like(body_mask)
    cv2.drawContours(contour_image, [body_contour], 0, 255, 1)
    
    return contour_image

# 从YOLOv5的.txt文件中获取中心坐标
def get_coordinates(txt_path, image_width, image_height):
    coordinates = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    obj_class, x_center, y_center, _, _, *_ = map(float, parts[:6])  # 只解包前六个
                    x_pixel = round(x_center * image_width)
                    y_pixel = round(y_center * image_height)
                    coordinates.append((int(obj_class), x_pixel, y_pixel))
                except ValueError as e:
                    print(f"Error: Could not convert parts to float: {e}")
                    continue
            else:
                print(f"Error: Line does not have enough parts: {len(parts)} parts found.")
    return coordinates

# 读取pixel信息
def count_pixel(file_path):
    file = sitk.ReadImage(file_path)
    return file.GetSpacing()[1]

# 改进的后方边界查找，使用多点采样和统计
def find_posterior_boundary(contour_image, kidney_x, kidney_y):
    """找到肾脏后方的身体边界，使用多点采样策略避免单点错误"""
    height, width = contour_image.shape
    if kidney_x < 0 or kidney_x >= width or kidney_y < 0 or kidney_y >= height:
        return None
    
    # 在肾脏x坐标附近多点采样
    sample_width = 10  # 采样宽度
    boundary_points = []
    
    for offset in range(-sample_width, sample_width + 1):
        sample_x = kidney_x + offset
        
        # 确保采样点在图像范围内
        if 0 <= sample_x < width:
            # 从肾脏向后扫描
            for y in range(kidney_y, height):
                if contour_image[y, sample_x] > 0:  # 找到边界点
                    boundary_points.append(y)
                    break
    
    # 如果没有找到足够的边界点，返回None
    if len(boundary_points) < 5:  # 至少需要5个点才能做可靠的统计
        return None
    
    # 使用中位数而不是平均值，避免异常值的影响
    boundary_y = int(np.median(boundary_points))
    
    return boundary_y

def kidneydepth(dicom_file_path, txt_file_path, output_dir=None):
    """处理单个DICOM和对应的txt文件"""
    if not os.path.isfile(dicom_file_path):
        print(f"Error: DICOM file not found at {dicom_file_path}")
        return
    
    if not os.path.isfile(txt_file_path):
        print(f"Error: TXT file not found at {txt_file_path}")
        return
    
    # 提取文件名（不含扩展名）作为输出文件前缀
    file_base_name = os.path.splitext(os.path.basename(dicom_file_path))[0]
    
    # 如果没有指定输出目录，使用当前目录
    if output_dir is None:
        output_dir = os.getcwd()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载DICOM图像
    hu_image, display_image = load_dicom_image(dicom_file_path)
    
    # 分割身体
    body_mask = segment_body(hu_image)
    
    # 找到后部轮廓
    posterior_contour = extract_body_contour(body_mask)
    
    # 保存结果
    body_mask_path = os.path.join(output_dir, f"{file_base_name}_body_mask.png")
    cv2.imwrite(body_mask_path, body_mask)
    print(f"身体掩码已保存至: {body_mask_path}")
    
    contour_path = None
    if posterior_contour is not None:
        contour_path = os.path.join(output_dir, f"{file_base_name}_contour.png")
        cv2.imwrite(contour_path, posterior_contour)
        print(f"轮廓已保存至: {contour_path}")
    
    # 获取肾脏坐标
    coordinates = get_coordinates(txt_file_path, display_image.shape[1], display_image.shape[0])
    
    # 计算结果
    results = {}
    pixel_spacing = count_pixel(dicom_file_path)
    
    for obj_class, x_coordinate, y_coordinate in coordinates:
        if x_coordinate is not None and y_coordinate is not None and posterior_contour is not None:
            posterior_boundary = find_posterior_boundary(posterior_contour, x_coordinate, y_coordinate)
            
            if posterior_boundary is not None:
                difference = (posterior_boundary - y_coordinate) * pixel_spacing
                label = 'L' if obj_class == 0 else 'R'
                results[label] = difference
            else:
                label = 'L' if obj_class == 0 else 'R'
                results[label] = "undefined"
    
    # 打印结果
    print(f"{file_base_name} 的测量结果: {json.dumps(results)}")
    
    # 保存结果为JSON文件
    results_path = os.path.join(output_dir, f"{file_base_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"结果已保存至: {results_path}")
    
    # 可视化
    if posterior_contour is not None:
        visualization = cv2.cvtColor(display_image.copy(), cv2.COLOR_GRAY2BGR)

        # 叠加半透明身体轮廓
        colored_contour = np.zeros_like(visualization)
        colored_contour[posterior_contour > 0] = [0, 255, 255]  # 黄色轮廓
        visualization = cv2.addWeighted(visualization, 1, colored_contour, 0.3, 0)

        for obj_class, x_coordinate, y_coordinate in coordinates:
            posterior_boundary = find_posterior_boundary(posterior_contour, x_coordinate, y_coordinate)
            if posterior_boundary is not None:
                # 绘制肾脏中心点
                cv2.circle(visualization, (x_coordinate, y_coordinate), 5, (0, 255, 0), -1)
                # 绘制后方边界点
                cv2.circle(visualization, (x_coordinate, posterior_boundary), 5, (0, 0, 255), -1)
                # 绘制连接线
                cv2.line(visualization, (x_coordinate, y_coordinate), (x_coordinate, posterior_boundary), (255, 0, 0), 2)
                # 显示测量信息
                distance = (posterior_boundary - y_coordinate) * pixel_spacing
                cv2.putText(visualization, 
                            f"{distance:.1f}mm", 
                            (x_coordinate + 10, (y_coordinate + posterior_boundary)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        viz_path = os.path.join(output_dir, f"{file_base_name}_visualization.png")
        cv2.imwrite(viz_path, visualization)
        print(f"可视化结果已保存至: {viz_path}")
    
    return results, viz_path

def find_deepest_slice(dicom_folder, yolo_labels_folder, output_base_dir):
    """
    遍历文件夹中的所有DICOM切片，找到肾脏深度测量值最大的切片（即“最深切片”）。

    Args:
        dicom_folder (str): 包含DICOM文件的文件夹路径。
        yolo_labels_folder (str): 包含对应YOLO检测结果.txt文件的文件夹路径。
        output_base_dir (str): 用于保存所有中间和最终结果的基目录。

    Returns:
        dict: 包含最深切片文件名和最大深度的信息，例如:
              {'deepest_slice': 'ImageFileName100.dcm', 'max_depth_mm': 65.2}
    """
    dicom_path = Path(dicom_folder)
    labels_path = Path(yolo_labels_folder)
    
    # 用于存储所有切片的深度结果
    all_slice_results = {}
    
    max_depth_mm = -1.0
    deepest_slice_name = None

    # 1. 查找所有 DICOM 文件
    # 查找 .dcm 或 .dicom 文件，可根据实际情况调整后缀
    dicom_files = list(dicom_path.glob('*.dcm')) + list(dicom_path.glob('*.dicom'))

    if not dicom_files:
        print(f"Error: No DICOM files found in {dicom_folder}")
        return {'deepest_slice': None, 'max_depth_mm': 0.0}

    print(f"找到 {len(dicom_files)} 个 DICOM 文件，开始处理...")

    # 2. 遍历每个 DICOM 文件
    for dicom_file_path in dicom_files:
        file_base_name = dicom_file_path.stem  # 不含后缀的文件名
        txt_file_path = labels_path / f"{file_base_name}.txt"
        
        # 检查对应的 YOLO .txt 文件是否存在
        if not txt_file_path.is_file():
            print(f"Warning: YOLO label file not found for {dicom_file_path.name}")
            continue

        # 为当前切片创建一个特定的输出目录
        slice_output_dir = Path(output_base_dir) / file_base_name
        slice_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"-> Processing {dicom_file_path.name}...")

        # 3. 调用 kidneydepth 函数进行测量
        try:
            # kidneydepth 返回的是结果字典和可视化路径
            results, _ = kidneydepth(str(dicom_file_path), str(txt_file_path), str(slice_output_dir))
            
            # 4. 提取最大深度
            current_max_depth = 0.0
            for label, depth in results.items():
                if isinstance(depth, (int, float)):
                    current_max_depth = max(current_max_depth, depth)
            
            all_slice_results[dicom_file_path.name] = current_max_depth

            # 5. 更新最深切片
            if current_max_depth > max_depth_mm:
                max_depth_mm = current_max_depth
                deepest_slice_name = dicom_file_path.name
                print(f"   -- New deepest slice found: {deepest_slice_name} with {max_depth_mm:.2f} mm")

        except Exception as e:
            print(f"Error processing {dicom_file_path.name}: {e}")
            continue

    # 6. 保存所有切片的深度结果
    all_results_path = Path(output_base_dir) / "all_slices_depth_summary.json"
    with open(all_results_path, 'w') as f:
        json.dump(all_slice_results, f, indent=4)
    print(f"\n所有切片深度汇总已保存至: {all_results_path}")

    final_result = {
        'deepest_slice': deepest_slice_name,
        'max_depth_mm': max_depth_mm
    }
    
    # 7. 打印并返回最终结果
    print("\n==============================================")
    print(f"遍历完成。最深切片是: {deepest_slice_name}")
    print(f"最大肾脏深度为: {max_depth_mm:.2f} mm")
    print("==============================================")
    
    return final_result


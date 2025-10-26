"""
YOLO11 detect
"""
import os
import sys
import argparse
from pathlib import Path
import time
import numpy as np
import cv2
import torch
import SimpleITK as sitk
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from back.constants import DETECT_MODEL_WEIGHTS_PATH
# MODEL_PATH_RELATIVE = 'weights/best.pt'
# MODEL_WEIGHTS_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE 
# 示例: Path('/home/kevin/Code/ROI') / 'weights/best.pt' 得到 
#       /home/kevin/Code/ROI/weights/best.pt

def get_first_int_from_metadata(image, key):
    """从DICOM元数据中获取第一个整数值"""
    try:
        metadata_value = image.GetMetaData(key)
        values = list(map(int, metadata_value.split('\\')))
        return values[0]
    except:
        return None

def adjust_window_level(img_array, window_center, window_width):
    """调整图像的窗位和窗宽"""
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    img_array = np.clip(img_array, lower_bound, upper_bound)
    img_array = (img_array - lower_bound) / (upper_bound - lower_bound) * 255.0
    return img_array.astype(np.uint8)

def load_dicom_image(file_path):
    """加载DICOM图像并返回适合YOLO处理的格式"""
    try:
        # 读取DICOM图像
        image = sitk.ReadImage(file_path)
        
        # 获取窗位窗宽
        window_center = get_first_int_from_metadata(image, '0028|1050')
        window_width = get_first_int_from_metadata(image, '0028|1051')
        
        # 如果无法获取窗位窗宽，使用默认值
        window_center = 40 if window_center is None else window_center
        window_width = 400 if window_width is None else window_width
        
        # 获取像素数据
        img_array = sitk.GetArrayFromImage(image).astype(np.int16)
        
        # 处理多维数组 (如果是3D DICOM，取第一个切片)
        if img_array.ndim > 2:
            img_array = img_array[0]
        
        # 调整窗位窗宽
        img_array = adjust_window_level(img_array, window_center, window_width)
        
        # 确保图像是三通道的
        if len(img_array.shape) == 2:  # 如果是灰度图
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        return img_array, image.GetSpacing()
    
    except Exception as e:
        print(f"处理DICOM文件时出错: {e}")
        return None, None

def save_txt_file(txt_path, results, names):
    """保存检测结果到txt文件中 (YOLOv5格式)"""
    with open(txt_path, 'w') as f:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()  # 归一化的xywh
                line = f"{cls} {x} {y} {w} {h} {conf}\n"  # YOLOv5格式
                f.write(line)

def detect(source, weights=DETECT_MODEL_WEIGHTS_PATH, img_size=640, conf_thres=0.25, 
           iou_thres=0.45, device='', view_img=False, save_txt=True, 
           save_conf=False, save_img=True, classes=None, agnostic_nms=False, 
           augment=False, project='static/results', name='exp', exist_ok=False):
    """使用YOLOv11模型检测图像，支持DICOM格式"""
    model = YOLO(weights)
    
    # 设置保存路径
    save_dir = Path(Path(project) / name)
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_txt:
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 确定输入源类型
    source = Path(source)
    is_dicom = source.suffix.lower() in ['.dcm', '.dicom']
    
    # 处理单个文件
    if source.is_file():
        if is_dicom:
            # 处理DICOM文件
            img, spacing = load_dicom_image(source)
            if img is None:
                print(f"无法读取DICOM图像: {source}")
                return
            
            # 创建临时JPG文件
            temp_jpg = save_dir / "temp_dicom.jpg"
            cv2.imwrite(str(temp_jpg), img)
            
            # 运行YOLO预测
            results = model.predict(
                source=str(temp_jpg),
                save=False,  # 不保存，我们将手动保存
                conf=conf_thres,
                iou=iou_thres,
                device=device,
                imgsz=img_size,
                classes=classes,
                agnostic_nms=agnostic_nms,
                augment=augment
            )
            
            # 创建输出图像
            result = results[0]
            names = model.names
            
            # 在图像上绘制检测框
            annotated_img = img.copy()
            annotator = Annotator(annotated_img)
            
            if len(result.boxes) > 0:
                for box in result.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f'{names[cls]} {conf:.2f}'
                    annotator.box_label(b, label, color=colors(cls))
            
            # 保存带有检测结果的图像
            save_path = str(save_dir / f"{source.stem}.jpg")
            cv2.imwrite(save_path, annotated_img)
            
            # 保存txt结果
            if save_txt:
                txt_path = str(save_dir / 'labels' / f"{source.stem}.txt")
                save_txt_file(txt_path, results, names)
                
            # 删除临时文件
            os.unlink(temp_jpg)
            
            if view_img:
                cv2.imshow(str(source), annotated_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        else:
            # 处理普通图像文件
            results = model.predict(
                source=str(source),
                save=save_img,
                save_txt=save_txt,
                save_conf=save_conf,
                project=project,
                name=name,
                exist_ok=True,
                conf=conf_thres,
                iou=iou_thres,
                device=device,
                imgsz=img_size,
                classes=classes,
                agnostic_nms=agnostic_nms,
                augment=augment,
                show=view_img
            )
    
    if view_img:
        cv2.destroyAllWindows()
    
    print(f"结果已保存到 {save_dir}")
    return txt_path if save_txt else None # 返回txt坐标路径

    # --- 添加到 detect.py ---

def detect_folder(source_folder, weights=DETECT_MODEL_WEIGHTS_PATH, img_size=640, conf_thres=0.25, 
                  iou_thres=0.45, device='', save_txt=True, 
                  classes=None, agnostic_nms=False, augment=False, 
                  project='static/results', name='exp_series', exist_ok=False):
    """
    使用YOLOv11模型批量检测文件夹中的所有DICOM切片。
    
    Args:
        source_folder (str): 包含DICOM文件的文件夹路径。
        ... (其他YOLO参数) ...
        
    Returns:
        Path: 保存 YOLO 标签 (.txt 文件) 的文件夹路径。
    """
    model = YOLO(weights)
    source_path = Path(source_folder)
    
    # 查找所有DICOM文件
    dicom_files = list(source_path.glob('*.dcm')) + list(source_path.glob('*.dicom'))
    if not dicom_files:
        print(f"Error: No DICOM files found in {source_folder}")
        return None
        
    # 设置保存路径
    save_dir = Path(project) / name
    labels_dir = save_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始批量检测 {len(dicom_files)} 个切片...")

    # 遍历并处理每个DICOM文件
    for dicom_file_path in dicom_files:
        
        # A. 图像预处理 (与 detect 函数内逻辑相同)
        img, spacing = load_dicom_image(dicom_file_path)
        if img is None:
            continue
        
        # 创建临时JPG文件
        temp_jpg = save_dir / "temp_dicom.jpg"
        cv2.imwrite(str(temp_jpg), img)
        
        # B. 运行YOLO预测
        results = model.predict(
            source=str(temp_jpg),
            save=False, 
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            imgsz=img_size,
            classes=classes,
            agnostic_nms=agnostic_nms,
            augment=augment
        )
        
        # C. 保存txt结果
        if save_txt:
            result = results[0]
            names = model.names
            txt_path = str(labels_dir / f"{dicom_file_path.stem}.txt")
            save_txt_file(txt_path, [result], names) # [result] 是为了兼容 save_txt_file 的预期输入
            
        # D. 保存带有检测结果的可视化图像 (可选，这里仅保存标签)
        # 如果需要保存可视化图像，请添加相关逻辑

        # E. 删除临时文件
        os.unlink(temp_jpg)

    print(f"所有切片的YOLO标签已保存到 {labels_dir}")
    return labels_dir


if __name__ == '__main__':
    detect(source='/home/kevin/Documents/GFR/CT_all/CT1/ImageFileName053.dcm')
# detect.py
import os
from pathlib import Path
import numpy as np
import cv2
import SimpleITK as sitk
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ..constants import ORIGINAL_CT_DIR , ORIGINAL_PNG_DIR , OVERLAY_PNG_DIR , YOLO_LABELS_DIR , ANALYSIS_RESULTS_DIR , DETECT_MODEL_WEIGHTS_PATH

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

def load_dicom_for_yolo(file_path: Path):
    """加载DICOM图像并返回适合YOLO处理的BGR格式图像和间距"""
    try:
        # 读取DICOM图像
        image = sitk.ReadImage(str(file_path))
        
        # 获取窗位窗宽
        window_center = get_first_int_from_metadata(image, '0028|1050')
        window_width = get_first_int_from_metadata(image, '0028|1051')
        window_center = 40 if window_center is None else window_center
        window_width = 400 if window_width is None else window_width
        
        # 获取像素数据
        img_array = sitk.GetArrayFromImage(image).astype(np.int16)
        
        if img_array.ndim > 2:
            img_array = img_array[0]
        
        # 调整窗位窗宽
        img_array = adjust_window_level(img_array, window_center, window_width)
        
        # 确保图像是三通道的
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        return img_array, image.GetSpacing()
    
    except Exception as e:
        print(f"处理DICOM文件时出错: {e}")
        return None, None

def save_txt_file(txt_path: Path, results, names):
    """保存检测结果到txt文件中 (YOLOv5格式)"""
    with open(txt_path, 'w') as f:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()  # 归一化的xywh
                line = f"{cls} {x} {y} {w} {h} {conf:.4f}\n" 
                f.write(line)

# --- 核心函数 ---
def detect_single_dcm(dcm_path: Path, weights=DETECT_MODEL_WEIGHTS_PATH, **kwargs) -> Path:
    """对单个 DICOM 文件进行检测，并将其 YOLO 标签保存到 YOLO_LABELS_DIR。"""
    
    model = YOLO(str(weights))
    img_array, _ = load_dicom_for_yolo(dcm_path)
    if img_array is None:
        raise ValueError(f"无法读取 DICOM 图像: {dcm_path.name}")
    
    # 使用临时文件进行预测
    temp_jpg = dcm_path.parent / f"temp_{dcm_path.stem}.jpg"
    cv2.imwrite(str(temp_jpg), img_array)

    # 运行YOLO预测
    results = model.predict(
        source=str(temp_jpg),
        save=False,
        conf=kwargs.get('conf_thres', 0.25),
        iou=kwargs.get('iou_thres', 0.45),
        device=kwargs.get('device', ''),
        imgsz=kwargs.get('img_size', 640),
        classes=kwargs.get('classes', None)
    )
    
    # 保存txt结果到标准路径
    txt_path = YOLO_LABELS_DIR / f"{dcm_path.stem}.txt"
    save_txt_file(txt_path, results, model.names)
    
    os.unlink(temp_jpg) # 删除临时文件
    
    return txt_path

def detect_dcm_folder(folder_path: Path, weights=DETECT_MODEL_WEIGHTS_PATH, **kwargs) -> Path:
    """批量检测 DICOM 文件夹，将所有 YOLO 标签保存到 YOLO_LABELS_DIR。"""
    
    model = YOLO(str(weights))
    
    dicom_files = list(folder_path.glob('*.dcm')) + list(folder_path.glob('*.dicom'))
    if not dicom_files:
        raise FileNotFoundError(f"文件夹中未找到 DICOM 文件: {folder_path.name}")
        
    print(f"开始批量检测 {len(dicom_files)} 个切片...")
    
    # 临时目录 (在文件夹下创建，用于存储临时JPG)
    temp_dir = folder_path / "temp_yolo"
    temp_dir.mkdir(exist_ok=True)
    
    for dcm_path in dicom_files:
        img_array, _ = load_dicom_for_yolo(dcm_path)
        if img_array is None:
            continue
            
        temp_jpg = temp_dir / f"{dcm_path.stem}.jpg"
        cv2.imwrite(str(temp_jpg), img_array)
        
    # YOLO 批量预测
    results_list = model.predict(
        source=str(temp_dir),
        save=False,
        conf=kwargs.get('conf_thres', 0.25),
        iou=kwargs.get('iou_thres', 0.45),
        device=kwargs.get('device', ''),
        imgsz=kwargs.get('img_size', 640),
        classes=kwargs.get('classes', None)
    )
    
    # 保存txt结果
    for result in results_list:
        # result.path 是临时 JPG 文件的路径
        temp_jpg_path = Path(result.path) 
        dcm_stem = temp_jpg_path.stem # 使用临时文件名作为 DICOM 的 stem
        
        txt_path = YOLO_LABELS_DIR / f"{dcm_stem}.txt"
        save_txt_file(txt_path, [result], model.names) 
        
    import shutil
    shutil.rmtree(temp_dir) # 删除整个临时目录
    
    return YOLO_LABELS_DIR # 返回标签文件夹路径
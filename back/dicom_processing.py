"""
主要功能：
1. 处理肾动态显像DICOM文件
   - 提取患者基本信息
   - 分割左右肾脏区域
   - 添加背景ROI区域
   - 计算肾脏和背景区域的放射性计数
   - 生成时间-计数曲线

2. 处理CT图像DICOM文件
   - 使用深度学习模型测量肾脏深度
   - 基于患者人体测量数据计算李氏肾脏深度
   - 处理不同来源的患者信息

3. 处理前端上传的肾脏深度

4. GFR计算
   - 结合肾脏放射性计数和深度数据
   - 应用衰减公式计算GFR
   - 支持模型测量深度和李氏公式深度两种方法

"""
import os
import pydicom
import numpy as np
from PIL import Image
import re
from CT import CT
from PIL import ImageEnhance
from detect import detect
from anatomical_segmentation import kidneydepth
from CT_find import find_deepest_kidney_slice

# =============================================================
# 局部配置 (替代 app.config)
# =============================================================

# 替代 Flask app.config，用于定义文件夹路径
LOCAL_CONFIG = {
    'CONVERTED_FOLDER': 'converted',
    'SEGMENTED_FOLDER': 'ROI',
    'DEPTH_FOLDER': 'depth',
    # 假设您的其他输出文件夹也需要定义
    'OUTPUT_ORIGINAL_DCM': 'output/original_dcm',
    'OUTPUT_RESAMPLED_DCM': 'output/resampled_dcm',
    'OUTPUT_ROI': 'output/ROI',
    'OUTPUT_SEGMENTATION': 'output/segmentation'
}

# 确保文件夹存在
for folder in LOCAL_CONFIG.values():
    if isinstance(folder, str) and folder not in ['BASE_URL']:
        os.makedirs(folder, exist_ok=True)
        
UPLOAD_FOLDER = 'uploads'
CONVERTED_FOLDER = 'converted'
SEGMENTED_FOLDER = 'ROI'
DEPTH_FOLDER = 'depth'
ALLOWED_EXTENSIONS = {'dcm'}

# =============================================================
# 全局变量定义和初始化
# =============================================================
kidney_depths = {
    'leftDepth': None,
    'rightDepth': None,
    'LiLeftKidneyDepth': None,
    'LiRightKidneyDepth': None
}
last_patient_name = None

last_manufacturer = None # 上一个病人的设备厂商
last_kidney_counts = {} # 上一个病人的肾脏计数信息
last_patient_info = {}  # 新增全局变量，用于存储病人信息

def full_calculate(dicomfile_path):
    """结合肾动态显像和CT图像进行GFR计算的完整流程"""
    #1.找到肾脏深度最大的切片
    MODEL_PATH = '/home/kevin/Code/ROI/final_kidney_segmenter.pth'
    result = find_deepest_kidney_slice(dicomfile_path, MODEL_PATH)
    if result:
        idx, filename, depth_pixels, mask = result
        print(f"Y轴跨度最大（最深）切片信息:")
        print(f"  索引 (D轴): {idx}")
        print(f"  原始DICOM文件名: {filename}")
        print(f"  肾脏Y轴跨度 (256x256像素): {depth_pixels}") 
        print(f"  分割图尺寸: {mask.shape}")
        dicom_path = os.path.join(dicomfile_path, filename)
    else:
        print("未找到肾脏切片")
        return None
    # 2. 提取病人信息
    patient_info = extract_patient_info(dicom_path)
        
    # 3. 更新全局变量
    last_patient_info = patient_info
    last_manufacturer = patient_info['manufacturer']

    create_required_directories_local()
    
    # 4. 处理肾动态显像DICOM文件
    result = process_kidney_dicom_local(dicom_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_required_directories_local():
    """创建处理DICOM文件所需的所有目录"""
    # 直接使用本地配置字典中的值
    directories = [
        LOCAL_CONFIG['CONVERTED_FOLDER'],
        LOCAL_CONFIG['SEGMENTED_FOLDER'],
        LOCAL_CONFIG['DEPTH_FOLDER'],
        LOCAL_CONFIG['OUTPUT_ORIGINAL_DCM'], 
        LOCAL_CONFIG['OUTPUT_RESAMPLED_DCM'], 
        LOCAL_CONFIG['OUTPUT_ROI'], 
        LOCAL_CONFIG['OUTPUT_SEGMENTATION']
    ]
    for directory in directories:
        # 使用 os.path.join 确保跨操作系统的兼容性
        os.makedirs(os.path.join(os.getcwd(), directory), exist_ok=True)
        print(f"创建目录: {directory}")


def convert_dicom_local(dicom_path):
    """本地处理肾动态显像的DICOM文件"""
    global last_manufacturer, last_kidney_counts, last_patient_info

    if not os.path.exists(dicom_path) or not allowed_file(dicom_path):
        print('未找到文件或文件格式不支持')
        return None

    try:
        patient_info = extract_patient_info(dicom_path)
        last_patient_info = patient_info
        last_manufacturer = patient_info['manufacturer']
        create_required_directories_local()
        result = process_kidney_dicom_local(dicom_path)
        last_kidney_counts = result['kidney_counts']
        print("处理完成，相关文件路径如下：")
        print("PNG图像：", result["png_path"])
        print("叠加图像：", result["overlay_path"])
        print("计数曲线：", result["graph_path"])
        print("患者信息：", patient_info)
        print("肾脏计数：", result['kidney_counts'])
        return result
    except Exception as e:
        print(f"处理DICOM文件时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_patient_info(dicom_path):
    """从DICOM文件中提取患者信息"""
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        sex = dicom_data.PatientSex if 'PatientSex' in dicom_data else 'N/A'
        if sex == 'M':
            sex = '男'
        elif sex == 'F':
            sex = '女'
        age = dicom_data.PatientAge if 'PatientAge' in dicom_data else 'N/A'
        if age and isinstance(age, str):
            age = re.sub(r'\D', '', age).lstrip('0')
        patient_name = 'N/A'
        if 'PatientName' in dicom_data:
            raw_name = str(dicom_data.PatientName)
            patient_name = raw_name.replace('^', '')
        return {
            'name': patient_name,
            'sex': sex,
            'age': age,
            'height': dicom_data.PatientSize if 'PatientSize' in dicom_data else 'N/A',
            'weight': dicom_data.PatientWeight if 'PatientWeight' in dicom_data else 'N/A',
            'manufacturer': dicom_data.Manufacturer if 'Manufacturer' in dicom_data else 'N/A',
            'date': dicom_data.StudyDate if 'StudyDate' in dicom_data else 'N/A'
        }
    except Exception as e:
        print(f"提取患者信息时出错: {e}")
        return {
            'name': 'N/A', 'sex': 'N/A', 'age': 'N/A',
            'height': 'N/A', 'weight': 'N/A',
            'manufacturer': 'N/A', 'date': 'N/A'
        }

def create_required_directories_local():
    """本地创建处理DICOM文件所需的所有目录"""
    directories = [
        CONVERTED_FOLDER,
        SEGMENTED_FOLDER,
        DEPTH_FOLDER,
        "output/original_dcm",
        "output/resampled_dcm",
        "output/ROI",
        "output/segmentation"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def process_kidney_dicom_local(dicom_path, frame_number=61, model_path="weights/best_epoch_weights.pth"):
    """本地处理肾动态扫描DICOM图像，进行肾脏分割和分析"""
    from preprocess import extract_frame, resample_dicom
    from predict import segment_kidney
    from ROI import add_background_roi
    from overlay import display_dicom_with_roi_overlay
    from graph import graph

    print(f"步骤1: 检查DICOM类型并处理...")
    dicom_data = pydicom.dcmread(dicom_path)
    is_multiframe = 'NumberOfFrames' in dicom_data and int(dicom_data.NumberOfFrames) > 1

    if is_multiframe:
        print(f"检测到多帧DICOM，共{dicom_data.NumberOfFrames}帧，提取第{frame_number}帧...")
        original_dcm_path, _ = extract_frame(dicom_path, frame_number)
    else:
        print("检测到单帧DICOM，直接处理...")
        original_dcm_path = dicom_path

    print("步骤2: 重采样DICOM...")
    resampled_dcm_path, _, png_path = resample_dicom(original_dcm_path, order=1)

    print("步骤3-4: 进行肾脏分割和ROI添加...")
    segmentation_path, _ = segment_kidney(resampled_dcm_path, model_path, img_size=224, num_classes=3)
    roi_path, roi_data = add_background_roi(segmentation_path)
    overlay_path = display_dicom_with_roi_overlay(resampled_dcm_path, roi_data)

    print("步骤5: 计算肾脏计数和生成曲线...")
    left_kidney_count, right_kidney_count, left_background_count, right_background_count, graph_path = graph(dicom_path, roi_data)

    print(f"左肾的像素值之和: {left_kidney_count}")
    print(f"右肾的像素值之和: {right_kidney_count}")
    print(f"左本底的像素值之和: {left_background_count}")
    print(f"右本底的像素值之和: {right_background_count}")

    kidney_counts = {
        'leftKidneyCount': int(left_kidney_count),
        'rightKidneyCount': int(right_kidney_count),
        'leftBackgroundCount': int(left_background_count),
        'rightBackgroundCount': int(right_background_count)
    }

    return {
        'png_path': png_path,
        'overlay_path': overlay_path,
        'graph_path': graph_path,
        'kidney_counts': kidney_counts
    }

def convert_depth_dicom_local(dicom_path):
    """本地处理肾脏深度计算部分的 DICOM 文件"""
    global kidney_depths, last_patient_name, last_patient_info

    if not os.path.exists(dicom_path) or not allowed_file(dicom_path):
        print('未找到文件或文件格式不支持')
        return None

    dicom_data = pydicom.dcmread(dicom_path)
    pixel_array = dicom_data.pixel_array.astype(float)
    current_patient_name = str(dicom_data.PatientName).upper().replace('^', '')
    last_name_normalized = last_patient_name.upper().replace('^', '') if last_patient_name else None

    if last_name_normalized != current_patient_name:
        print(f"检测到新患者: {current_patient_name}，重置肾脏深度信息")
        kidney_depths = {
            'leftDepth': None,
            'rightDepth': None,
            'LiLeftKidneyDepth': None,
            'LiRightKidneyDepth': None
        }
        last_patient_name = str(dicom_data.PatientName)

    ct_patient_info = {
        'height': dicom_data.get('PatientSize', None),
        'weight': dicom_data.get('PatientWeight', None),
        'age': dicom_data.get("PatientAge", None),
        'sex': dicom_data.get("PatientSex", None)
    }

    ct_info_complete = all(value is not None and value != 'N/A' for value in ct_patient_info.values())
    has_previous_info = all(last_patient_info.get(key, 'N/A') != 'N/A' for key in ['height', 'weight', 'age', 'sex'])
    using_previous_info = False

    if not ct_info_complete and has_previous_info:
        using_previous_info = True
        print(f"CT图像缺少病人信息，使用之前肾动态显像的病人信息计算李氏公式")
        height_value = last_patient_info.get('height', 'N/A')
        weight_value = last_patient_info.get('weight', 'N/A')
        patient_age_str = last_patient_info.get('age', 'N/A')
        patient_sex = last_patient_info.get('sex', 'N/A')
        if patient_sex == '男':
            patient_sex = 'M'
        elif patient_sex == '女':
            patient_sex = 'F'
    else:
        height_value = ct_patient_info['height']
        weight_value = ct_patient_info['weight']
        patient_age_str = ct_patient_info['age']
        patient_sex = ct_patient_info['sex']
        if height_value is None or height_value == 'N/A':
            height_value = last_patient_info.get('height', 'N/A')
            if height_value != 'N/A' and using_previous_info == False:
                using_previous_info = True
                print("使用肾动态显像提供的身高数据")
        if weight_value is None or weight_value == 'N/A':
            weight_value = last_patient_info.get('weight', 'N/A')
            if weight_value != 'N/A' and using_previous_info == False:
                using_previous_info = True
                print("使用肾动态显像提供的体重数据")
        if patient_age_str is None or patient_age_str == 'N/A':
            patient_age_str = last_patient_info.get('age', 'N/A')
            if patient_age_str != 'N/A' and using_previous_info == False:
                using_previous_info = True
                print("使用肾动态显像提供的年龄数据")
        if patient_sex is None or patient_sex == 'N/A':
            patient_sex = last_patient_info.get('sex', 'N/A')
            if patient_sex != 'N/A' and using_previous_info == False:
                using_previous_info = True
                print("使用肾动态显像提供的性别数据")
                if patient_sex == '男':
                    patient_sex = 'M'
                elif patient_sex == '女':
                    patient_sex = 'F'

    if height_value != 'N/A' and weight_value != 'N/A' and patient_age_str != 'N/A' and patient_sex != 'N/A':
        try:
            if isinstance(height_value, str):
                try:
                    height_value = float(height_value)
                    if height_value < 10:
                        patient_height = height_value * 100
                    else:
                        patient_height = height_value
                except ValueError:
                    print(f"无法转换身高值: {height_value}")
                    patient_height = None
            else:
                if height_value < 10:
                    patient_height = float(height_value) * 100
                else:
                    patient_height = float(height_value)
            if isinstance(weight_value, str):
                try:
                    patient_weight = float(weight_value)
                except ValueError:
                    print(f"无法转换体重值: {weight_value}")
                    patient_weight = None
            else:
                patient_weight = float(weight_value)
            if isinstance(patient_age_str, str):
                if patient_age_str.endswith('Y'):
                    patient_age = int(patient_age_str[:-1])
                else:
                    try:
                        patient_age = int(patient_age_str)
                    except ValueError:
                        import re
                        num_match = re.search(r'\d+', patient_age_str)
                        if num_match:
                            patient_age = int(num_match.group())
                        else:
                            print(f"无法解析年龄字符串: {patient_age_str}")
                            patient_age = None
            else:
                try:
                    patient_age = int(patient_age_str)
                except (ValueError, TypeError):
                    patient_age = None
            if patient_height is not None and patient_weight is not None and patient_age is not None:
                if patient_sex == 'F':
                    LiKidney_L = 0.013 * patient_age - 0.044 * patient_height + 0.087 * patient_weight + 7.951
                    LiKidney_R = 0.005 * patient_age - 0.035 * patient_height + 0.082 * patient_weight + 7.266
                elif patient_sex == 'M':
                    LiKidney_L = 0.013 * patient_age + 0.117 - 0.044 * patient_height + 0.087 * patient_weight + 7.951
                    LiKidney_R = 0.005 * patient_age + 0.013 - 0.035 * patient_height + 0.082 * patient_weight + 7.266
                else:
                    LiKidney_L = None
                    LiKidney_R = None
                if LiKidney_L is not None and LiKidney_R is not None:
                    kidney_depths['LiLeftKidneyDepth'] = round(LiKidney_L * 10, 2)
                    kidney_depths['LiRightKidneyDepth'] = round(LiKidney_R * 10, 2)
                    data_source = "肾动态显像" if using_previous_info else "当前CT图像"
                    print(f"李氏深度计算完成 (使用{data_source}数据)：")
                    print(f"- 使用值: 身高={patient_height}cm, 体重={patient_weight}kg, 年龄={patient_age}, 性别={patient_sex}")
                    print(f"- 结果: 左肾={kidney_depths['LiLeftKidneyDepth']}mm, 右肾={kidney_depths['LiRightKidneyDepth']}mm")
            else:
                print("转换后的数据无效，无法计算李氏肾脏深度")
        except (ValueError, TypeError) as e:
            print(f"计算李氏肾脏深度时出错: {e}")
            print(f"使用的值: 身高={height_value}, 体重={weight_value}, 年龄={patient_age_str}, 性别={patient_sex}")
    else:
        missing_data = []
        if height_value == 'N/A': missing_data.append("身高")
        if weight_value == 'N/A': missing_data.append("体重")
        if patient_age_str == 'N/A': missing_data.append("年龄")
        if patient_sex == 'N/A': missing_data.append("性别")
        print(f"缺少计算李氏肾脏深度所需的患者信息: {', '.join(missing_data)}")
        if not has_previous_info:
            print("提示: 先上传肾动态显像可以获取患者信息")

    result, viz_path = CT(dicom_path)
    print("CT 计算结果:", result)
    left_depth = result['L']
    right_depth = result['R']

    if left_depth != 'N/A':
        kidney_depths['leftDepth'] = round(left_depth, 2)
        print(f"记录到左肾深度: {kidney_depths['leftDepth']}mm")
    if right_depth != 'N/A':
        kidney_depths['rightDepth'] = round(right_depth, 2)
        print(f"记录到右肾深度: {kidney_depths['rightDepth']}mm")

    kidneyDepth = {
        'leftDepth': kidney_depths['leftDepth'],
        'rightDepth': kidney_depths['rightDepth'],
        'LiLeftKidneyDepth': kidney_depths['LiLeftKidneyDepth'],
        'LiRightKidneyDepth': kidney_depths['LiRightKidneyDepth']
    }

    if left_depth == 'N/A' and right_depth == 'N/A':
        print('未检测到肾脏深度，请重新上传CT图像。')
        return None
    else:
        print("肾脏深度信息：", kidneyDepth)
        print("深度可视化图片路径：", viz_path)
        return kidneyDepth

def upload_depth_local(left_depth, right_depth, height, weight, age, sex):
    """本地上传肾脏深度计算结果"""
    global kidney_depths

    if sex == '女':
        sex = 'F'
    elif sex == '男':
        sex = 'M'

    if left_depth is not None:
        kidney_depths['leftDepth'] = float(left_depth)
    if right_depth is not None:
        kidney_depths['rightDepth'] = float(right_depth)

    patient_height = height * 100
    patient_weight = weight
    patient_age = int(age)
    patient_sex = sex

    if not patient_height or not patient_weight or not patient_age or not patient_sex:
        print('缺少病人信息')
        return None

    LiKidney_L = None
    LiKidney_R = None

    if patient_sex == 'F':
        LiKidney_L = 0.013 * patient_age - 0.044 * patient_height + 0.087 * patient_weight + 7.951
        LiKidney_R = 0.005 * patient_age - 0.035 * patient_height + 0.082 * patient_weight + 7.266
    elif patient_sex == 'M':
        LiKidney_L = 0.013 * patient_age + 0.117 - 0.044 * patient_height + 0.087 * patient_weight + 7.951
        LiKidney_R = 0.005 * patient_age + 0.013 - 0.035 * patient_height + 0.082 * patient_weight + 7.266

    if LiKidney_L is not None and LiKidney_R is not None:
        kidney_depths['LiLeftKidneyDepth'] = round(LiKidney_L * 10, 2)
        kidney_depths['LiRightKidneyDepth'] = round(LiKidney_R * 10, 2)
        print('肾脏深度上传成功')
        print('李氏左肾深度:', kidney_depths['LiLeftKidneyDepth'])
        print('李氏右肾深度:', kidney_depths['LiRightKidneyDepth'])
        return {
            'LiLeftKidneyDepth': kidney_depths['LiLeftKidneyDepth'],
            'LiRightKidneyDepth': kidney_depths['LiRightKidneyDepth']
        }
    else:
        print('无法计算李氏肾脏深度')
        return None

def calculate_gfr_local():
    """本地计算 GFR"""
    global kidney_depths, last_manufacturer, last_kidney_counts

    print("last_manufacturer:", last_manufacturer)
    print("last_kidney_counts:", last_kidney_counts)
    print("kidney_depths:", kidney_depths)

    if not last_manufacturer or not last_kidney_counts or not kidney_depths['leftDepth'] or not kidney_depths['rightDepth']:
        print('缺少计算 GFR 所需的数据')
        return None

    if 'GE MEDICAL SYSTEMS' in last_manufacturer:
        injection_count = 183916
    elif 'SIEMENS NM' in last_manufacturer:
        injection_count = 139857
    else:
        injection_count = 150000  # 默认值

    print("injection_count:", injection_count)

    left_count = last_kidney_counts['leftKidneyCount'] - last_kidney_counts['leftBackgroundCount']
    right_count = last_kidney_counts['rightKidneyCount'] - last_kidney_counts['rightBackgroundCount']
    left_depth = kidney_depths['leftDepth']
    right_depth = kidney_depths['rightDepth']

    u = 0.153
    import math
    eL = math.exp(-u * left_depth * 0.1)
    eR = math.exp(-u * right_depth * 0.1)

    GFR_left = ((left_count * 0.001) / eL / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
    GFR_right = ((right_count * 0.001) / eR / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
    GFR_total = GFR_left + GFR_right

    print("GFR计算结果：")
    print(" GFR_left:", round(GFR_left, 2))
    print(" GFR_right:", round(GFR_right, 2))
    print(" GFR_total:", round(GFR_total, 2))

    Li_left_depth = kidney_depths['LiLeftKidneyDepth']
    Li_right_depth = kidney_depths['LiRightKidneyDepth']
    eL_Li = math.exp(-u * Li_left_depth * 0.1)
    eR_Li = math.exp(-u * Li_right_depth * 0.1)

    Li_GFR_left = ((left_count * 0.001) / eL_Li / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
    Li_GFR_right = ((right_count * 0.001) / eR_Li / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
    Li_GFR_total = Li_GFR_left + Li_GFR_right

    print("李氏公式计算结果：")
    print(" Li_GFR_left:", round(Li_GFR_left, 2))
    print(" Li_GFR_right:", round(Li_GFR_right, 2))
    print(" Li_GFR_total:", round(Li_GFR_total, 2))

    return {
        'leftGFR': round(GFR_left, 2),
        'rightGFR': round(GFR_right, 2),
        'totalGFR': round(GFR_total, 2),
        'LiLeftGFR': round(Li_GFR_left, 2),
        'LiRightGFR': round(Li_GFR_right, 2),
        'LiTotalGFR': round(Li_GFR_total, 2),
    }


#--------------------------------------------------------------------------


def find_largest_kidney_dcm(folder_path, weights='weights/best.pt', img_size=640):
    """
    遍历文件夹下所有dcm文件，对每一张进行肾脏区域识别（调用YOLO检测），返回肾脏像素最多的一张文件路径和像素数
    """
    max_pixels = 0
    max_file = None
    max_box = None

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                dcm_path = os.path.join(root, file)
                try:
                    # 调用YOLO检测，返回txt结果路径
                    txt_path = detect(
                        source=dcm_path,
                        weights=weights,
                        img_size=img_size,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        device='cpu',
                        save_txt=True,
                        save_img=False,
                        view_img=False,
                        project='static/results',
                        name='exp',
                        exist_ok=True
                    )
                    # 解析txt文件，统计最大检测框面积
                    if txt_path and os.path.exists(txt_path):
                        with open(txt_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    # YOLO格式: class x_center y_center width height [conf]
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    area = w * h
                                    if area > max_pixels:
                                        max_pixels = area
                                        max_file = dcm_path
                                        max_box = parts
                except Exception as e:
                    print(f"处理 {dcm_path} 时出错: {e}")

    print(f"肾脏检测框面积最大的文件: {max_file}, 面积: {max_pixels}")
    return max_file, max_pixels, max_box
    # dicom转为png在前端显示，并读取病人信息，处理肾动态显像图
def convert_dicom(app):
    """
    处理肾动态显像的DICOM文件上传并进行处理
    
    参数:
        app: Flask应用实例，用于获取配置信息
        
    返回:
        JSON响应: 包含处理结果和相关图像URL
    """
    global last_manufacturer, last_kidney_counts, last_patient_info

    # 检查文件是否存在
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '未收到文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400

    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': '不支持的文件格式'}), 400

    try:
        # 1. 保存并读取DICOM文件
        filename = secure_filename(file.filename)
        dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dicom_path)
        
        # 2. 提取病人信息
        patient_info = extract_patient_info(dicom_path)
        
        # 3. 更新全局变量
        last_patient_info = patient_info
        last_manufacturer = patient_info['manufacturer']
        
        # 4. 创建处理所需的目录结构
        create_required_directories(app)
        
        # 5. 处理DICOM序列并进行肾脏分析
        result = process_kidney_dicom(dicom_path, app)
        
        # 6. 更新全局肾脏计数变量
        last_kidney_counts = result['kidney_counts']
        
        # 7. 构建前端所需的响应数据
        # 注意：确保使用一致的基础URL，根据app配置
        base_url = app.config['BASE_URL']
        return jsonify({
            'success': True,
            'imageUrl': f'{base_url}/{result["png_path"]}',
            'overlayUrl': f'{base_url}/{result["overlay_path"]}',
            'countsTimeUrl': f'{base_url}/{result["graph_path"]}',
            'patientInfo': patient_info,
            'kidneyCounts': result['kidney_counts']
        })
            
    except Exception as e:
        print(f"处理DICOM文件时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return jsonify({'success': False, 'message': f'处理DICOM文件时发生错误: {str(e)}'}), 500

# 辅助函数：提取患者信息
def extract_patient_info(dicom_path):
    """从DICOM文件中提取患者信息"""
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        
        # 处理性别信息
        sex = dicom_data.PatientSex if 'PatientSex' in dicom_data else 'N/A'
        if sex == 'M':
            sex = '男'
        elif sex == 'F':
            sex = '女'
        
        # 处理年龄信息
        age = dicom_data.PatientAge if 'PatientAge' in dicom_data else 'N/A'
        if age and isinstance(age, str):
            age = re.sub(r'\D', '', age).lstrip('0')  # 移除非数字字符

        # 处理患者姓名，移除分隔符
        patient_name = 'N/A'
        if 'PatientName' in dicom_data:
            raw_name = str(dicom_data.PatientName)
            # 移除姓名中的分隔符，如 ^ 符号
            patient_name = raw_name.replace('^', '')
        
        # 构建患者信息字典
        return {
            'name': patient_name,
            'sex': sex,
            'age': age,
            'height': dicom_data.PatientSize if 'PatientSize' in dicom_data else 'N/A',
            'weight': dicom_data.PatientWeight if 'PatientWeight' in dicom_data else 'N/A',
            'manufacturer': dicom_data.Manufacturer if 'Manufacturer' in dicom_data else 'N/A',
            'date': dicom_data.StudyDate if 'StudyDate' in dicom_data else 'N/A'
        }
    except Exception as e:
        print(f"提取患者信息时出错: {e}")
        # 返回默认值
        return {
            'name': 'N/A', 'sex': 'N/A', 'age': 'N/A',
            'height': 'N/A', 'weight': 'N/A', 
            'manufacturer': 'N/A', 'date': 'N/A'
        }

# 辅助函数：创建所需目录
def create_required_directories(app):
    """创建处理DICOM文件所需的所有目录"""
    directories = [
        app.config['CONVERTED_FOLDER'],
        app.config['SEGMENTED_FOLDER'],
        app.config['DEPTH_FOLDER'],
        "output/original_dcm", 
        "output/resampled_dcm", 
        "output/ROI", 
        "output/segmentation"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# 辅助函数：处理肾脏DICOM图像
def process_kidney_dicom(dicom_path, app, frame_number=61, model_path="weights/best_epoch_weights.pth"):
    """
    处理肾动态扫描DICOM图像，进行肾脏分割和分析
    
    参数:
        dicom_path: DICOM文件路径
        frame_number: 要处理的帧号
        model_path: 肾脏分割模型的路径
        
    返回:
        dict: 包含处理结果路径和肾脏计数信息的字典
    """
    # 按需导入模块，避免循环导入
    from preprocess import extract_frame, resample_dicom
    from predict import segment_kidney
    from ROI import add_background_roi
    from overlay import display_dicom_with_roi_overlay
    from graph import graph

    print(f"步骤1: 检查DICOM类型并处理...")
    # 首先检查是单帧还是多帧DICOM
    dicom_data = pydicom.dcmread(dicom_path)

    # 检查是否为多帧DICOM - 根据NumberOfFrames属性判断
    is_multiframe = 'NumberOfFrames' in dicom_data and int(dicom_data.NumberOfFrames) > 1
    
    if is_multiframe:
        print(f"检测到多帧DICOM，共{dicom_data.NumberOfFrames}帧，提取第{frame_number}帧...")
        # 提取和重采样DICOM帧
        original_dcm_path, _ = extract_frame(dicom_path, frame_number)
    else:
        print("检测到单帧DICOM，直接处理...")
        # 单帧DICOM直接使用原始路径
        original_dcm_path = dicom_path
    
    print("步骤2: 重采样DICOM...")
    # 提取和重采样DICOM帧
    #original_dcm_path, _ = extract_frame(dicom_path, frame_number)
    resampled_dcm_path, _, png_path = resample_dicom(original_dcm_path, order=1)
    
    print("步骤3-4: 进行肾脏分割和ROI添加...")
    # 执行肾脏分割
    segmentation_path, _ = segment_kidney(resampled_dcm_path, model_path, img_size=224, num_classes=3)
    
    # 添加本底区域ROI
    roi_path, roi_data = add_background_roi(segmentation_path)
    
    # 生成叠加显示图像
    overlay_path = display_dicom_with_roi_overlay(resampled_dcm_path, roi_data)
    
    print("步骤5: 计算肾脏计数和生成曲线...")
    # 计算肾脏计数并生成曲线
    left_kidney_count, right_kidney_count, left_background_count, right_background_count, graph_path = graph(dicom_path, roi_data)
    
    # 输出肾脏计数
    print(f"左肾的像素值之和: {left_kidney_count}")
    print(f"右肾的像素值之和: {right_kidney_count}")
    print(f"左本底的像素值之和: {left_background_count}")
    print(f"右本底的像素值之和: {right_background_count}")
    
    # 肾脏计数表格
    kidney_counts = {
        'leftKidneyCount': int(left_kidney_count),
        'rightKidneyCount': int(right_kidney_count),
        'leftBackgroundCount': int(left_background_count),
        'rightBackgroundCount': int(right_background_count)
    }
    
    return {
        'png_path': png_path,
        'overlay_path': overlay_path,
        'graph_path': graph_path,
        'kidney_counts': kidney_counts
    }

# 处理肾脏深度计算部分的 DICOM 文件上传
def convert_depth_dicom(app):
    print("开始处理肾脏深度计算部分的 DICOM 文件上传...")
    global kidney_depths, last_patient_name, last_patient_info

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dicom_path)

        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array.astype(float)

        # 获取当前病人的姓名并标准化为大写
        current_patient_name = str(dicom_data.PatientName).upper().replace('^', '')
        
        # 如果上一个姓名存在，也标准化为大写进行比较
        last_name_normalized = last_patient_name.upper().replace('^', '') if last_patient_name else None
        
        # 如果当前病人的姓名与上一个病人的姓名不一样，就清空肾脏深度的信息
        if last_name_normalized != current_patient_name:
            print(f"检测到新患者: {current_patient_name}，重置肾脏深度信息")
            kidney_depths = {
                'leftDepth': None,
                'rightDepth': None,
                'LiLeftKidneyDepth': None,
                'LiRightKidneyDepth': None
            }
            last_patient_name = str(dicom_data.PatientName)

        # 从CT DICOM获取患者信息
        ct_patient_info = {
            'height': dicom_data.get('PatientSize', None),
            'weight': dicom_data.get('PatientWeight', None),
            'age': dicom_data.get("PatientAge", None),
            'sex': dicom_data.get("PatientSex", None)
        }
        
        # 检查CT DICOM信息是否完整
        ct_info_complete = all(value is not None and value != 'N/A' 
                             for value in ct_patient_info.values())
        
        # 检查是否有之前肾动态显像的信息
        has_previous_info = all(last_patient_info.get(key, 'N/A') != 'N/A' 
                              for key in ['height', 'weight', 'age', 'sex'])
        
        # 确定使用哪个信息源
        using_previous_info = False
        if not ct_info_complete and has_previous_info:
            using_previous_info = True
            print(f"CT图像缺少病人信息，使用之前肾动态显像的病人信息计算李氏公式")
            height_value = last_patient_info.get('height', 'N/A')
            weight_value = last_patient_info.get('weight', 'N/A')
            patient_age_str = last_patient_info.get('age', 'N/A')
            patient_sex = last_patient_info.get('sex', 'N/A')
            
            # 处理性别格式转换
            if patient_sex == '男':
                patient_sex = 'M'
            elif patient_sex == '女':
                patient_sex = 'F'
        else:
            # 使用CT图像中的信息
            height_value = ct_patient_info['height']
            weight_value = ct_patient_info['weight']
            patient_age_str = ct_patient_info['age']
            patient_sex = ct_patient_info['sex']
            
            # 如果CT图像信息不完整，则尝试补充
            if height_value is None or height_value == 'N/A':
                height_value = last_patient_info.get('height', 'N/A')
                if height_value != 'N/A' and using_previous_info == False:
                    using_previous_info = True
                    print("使用肾动态显像提供的身高数据")
                    
            if weight_value is None or weight_value == 'N/A':
                weight_value = last_patient_info.get('weight', 'N/A')
                if weight_value != 'N/A' and using_previous_info == False:
                    using_previous_info = True
                    print("使用肾动态显像提供的体重数据")
                    
            if patient_age_str is None or patient_age_str == 'N/A':
                patient_age_str = last_patient_info.get('age', 'N/A')
                if patient_age_str != 'N/A' and using_previous_info == False:
                    using_previous_info = True
                    print("使用肾动态显像提供的年龄数据")
                    
            if patient_sex is None or patient_sex == 'N/A':
                patient_sex = last_patient_info.get('sex', 'N/A')
                if patient_sex != 'N/A' and using_previous_info == False:
                    using_previous_info = True
                    print("使用肾动态显像提供的性别数据")
                    # 处理性别格式转换
                    if patient_sex == '男':
                        patient_sex = 'M'
                    elif patient_sex == '女':
                        patient_sex = 'F'
    
        # 检查值是否可用于计算
        if height_value != 'N/A' and weight_value != 'N/A' and patient_age_str != 'N/A' and patient_sex != 'N/A':
            try:
                # 处理数据类型
                if isinstance(height_value, str):
                    try:
                        # 尝试将身高值转换为数字类型
                        height_value = float(height_value)
                        # 判断单位 - 如果小于10，可能是米单位
                        if height_value < 10:
                            patient_height = height_value * 100  # 转为厘米
                        else:
                            patient_height = height_value  # 已经是厘米单位
                    except ValueError:
                        print(f"无法转换身高值: {height_value}")
                        patient_height = None
                else:
                    # 如果已经是数字类型
                    if height_value < 10:
                        patient_height = float(height_value) * 100
                    else:
                        patient_height = float(height_value)
                
                # 处理体重
                if isinstance(weight_value, str):
                    try:
                        patient_weight = float(weight_value)
                    except ValueError:
                        print(f"无法转换体重值: {weight_value}")
                        patient_weight = None
                else:
                    patient_weight = float(weight_value)
                
                # 解析病人年龄字符串
                if isinstance(patient_age_str, str):
                    if patient_age_str.endswith('Y'):
                        patient_age = int(patient_age_str[:-1])
                    else:
                        try:
                            patient_age = int(patient_age_str)
                        except ValueError:
                            # 尝试提取数字部分
                            import re
                            num_match = re.search(r'\d+', patient_age_str)
                            if num_match:
                                patient_age = int(num_match.group())
                            else:
                                print(f"无法解析年龄字符串: {patient_age_str}")
                                patient_age = None
                else:
                    try:
                        patient_age = int(patient_age_str)
                    except (ValueError, TypeError):
                        patient_age = None
                
                # 检查所有转换后的值是否有效
                if patient_height is not None and patient_weight is not None and patient_age is not None:
                    # 计算李氏肾脏深度，单位cm
                    if patient_sex == 'F':  # 女性
                        LiKidney_L = 0.013 * patient_age - 0.044 * patient_height + 0.087 * patient_weight + 7.951
                        LiKidney_R = 0.005 * patient_age - 0.035 * patient_height + 0.082 * patient_weight + 7.266
                    elif patient_sex == 'M':  # 男性
                        LiKidney_L = 0.013 * patient_age + 0.117 - 0.044 * patient_height + 0.087 * patient_weight + 7.951
                        LiKidney_R = 0.005 * patient_age + 0.013 - 0.035 * patient_height + 0.082 * patient_weight + 7.266
                    else:
                        LiKidney_L = None
                        LiKidney_R = None
                    
                    if LiKidney_L is not None and LiKidney_R is not None:
                        kidney_depths['LiLeftKidneyDepth'] = round(LiKidney_L * 10, 2)  # 转换为mm
                        kidney_depths['LiRightKidneyDepth'] = round(LiKidney_R * 10, 2)  # 转换为mm
                        data_source = "肾动态显像" if using_previous_info else "当前CT图像"
                        print(f"李氏深度计算完成 (使用{data_source}数据)：")
                        print(f"- 使用值: 身高={patient_height}cm, 体重={patient_weight}kg, 年龄={patient_age}, 性别={patient_sex}")
                        print(f"- 结果: 左肾={kidney_depths['LiLeftKidneyDepth']}mm, 右肾={kidney_depths['LiRightKidneyDepth']}mm")
                else:
                    print("转换后的数据无效，无法计算李氏肾脏深度")
            except (ValueError, TypeError) as e:
                print(f"计算李氏肾脏深度时出错: {e}")
                print(f"使用的值: 身高={height_value}, 体重={weight_value}, 年龄={patient_age_str}, 性别={patient_sex}")
        else:
            missing_data = []
            if height_value == 'N/A': missing_data.append("身高")
            if weight_value == 'N/A': missing_data.append("体重")
            if patient_age_str == 'N/A': missing_data.append("年龄")
            if patient_sex == 'N/A': missing_data.append("性别")
            
            print(f"缺少计算李氏肾脏深度所需的患者信息: {', '.join(missing_data)}")
            if not has_previous_info:
                print("提示: 先上传肾动态显像可以获取患者信息")

        # 模型计算肾脏深度
        result, viz_path= CT(dicom_path)
        print("CT 计算结果:", result)  # 调试输出
        left_depth = result['L']
        right_depth = result['R']

        # 更新全局变量中的肾脏深度信息
        if left_depth != 'N/A':
            kidney_depths['leftDepth'] = round(left_depth, 2)
            print(f"记录到左肾深度: {kidney_depths['leftDepth']}mm")
        if right_depth != 'N/A':
            kidney_depths['rightDepth'] = round(right_depth, 2)
            print(f"记录到右肾深度: {kidney_depths['rightDepth']}mm")

        kidneyDepth = {
            'leftDepth': kidney_depths['leftDepth'],
            'rightDepth': kidney_depths['rightDepth'],
            'LiLeftKidneyDepth': kidney_depths['LiLeftKidneyDepth'],
            'LiRightKidneyDepth': kidney_depths['LiRightKidneyDepth']
        }

        base_url = app.config['BASE_URL']
       
        if left_depth == 'N/A' and right_depth == 'N/A':
            return jsonify({'success': False, 'message': '未检测到肾脏深度，请重新上传CT图像。'}), 400
        else:
            # 构建返回消息
            message_parts = []
            
            # CT测量深度信息
            if left_depth != 'N/A' and right_depth != 'N/A':
                message_parts.append("已检测到双侧肾脏深度")
            elif left_depth != 'N/A':
                message_parts.append("只检测到左肾深度")
            elif right_depth != 'N/A':
                message_parts.append("只检测到右肾深度")
                
            # 李氏深度信息
            if kidney_depths['LiLeftKidneyDepth'] is not None and kidney_depths['LiRightKidneyDepth'] is not None:
                data_source = "肾动态显像" if using_previous_info else "当前CT图像"
                message_parts.append(f"已使用{data_source}数据计算李氏深度")
            
            message = "，".join(message_parts) + "。"
            
            return jsonify({
                'success': True, 
                'depthImageUrl': f'{base_url}/{viz_path}', 
                'message': message,
                'kidneyDepth': kidneyDepth,
                'usingDynamicPatientInfo': using_previous_info
            })

    return jsonify({'success': False, 'message': 'Invalid file format'}), 400

# 上传肾脏深度计算结果
def upload_depth():
    global kidney_depths

    data = request.get_json()
    print("开始上传肾脏深度...")
    print("从前端接收到的数据：", data)
    left_depth = data.get('leftKidney')
    right_depth = data.get('rightKidney')
    patient_height = data.get('height') * 100 # 身高单位cm
    patient_weight = data.get('weight') # 体重单位kg
    patient_age = int(data.get('age'))
    patient_sex = data.get('sex')

    if patient_sex == '女':
        patient_sex = 'F'
    elif patient_sex == '男':
        patient_sex = 'M'

    if left_depth is not None:
        kidney_depths['leftDepth'] = float(left_depth)
    if right_depth is not None:
        kidney_depths['rightDepth'] = float(right_depth)
    
    # 检查病人信息是否有效
    if not patient_height or not patient_weight or not patient_age or not patient_sex:
        return jsonify({'success': False, 'message': '缺少病人信息'}), 400

    # 计算李氏肾脏深度
    LiKidney_L = None
    LiKidney_R = None

    if patient_sex == 'F':  # 女性
        LiKidney_L = 0.013 * patient_age - 0.044 * patient_height + 0.087 * patient_weight + 7.951
        LiKidney_R = 0.005 * patient_age - 0.035 * patient_height + 0.082 * patient_weight + 7.266
    elif patient_sex == 'M':  # 男性
        LiKidney_L = 0.013 * patient_age + 0.117 - 0.044 * patient_height + 0.087 * patient_weight + 7.951
        LiKidney_R = 0.005 * patient_age + 0.013 - 0.035 * patient_height + 0.082 * patient_weight + 7.266

    if LiKidney_L is not None and LiKidney_R is not None:
        kidney_depths['LiLeftKidneyDepth'] = round(LiKidney_L * 10, 2)
        kidney_depths['LiRightKidneyDepth'] = round(LiKidney_R * 10, 2)
    else:
        return jsonify({'success': False, 'message': '无法计算李氏肾脏深度'}), 400

    return jsonify({
        'success': True,
        'message': '肾脏深度上传成功',
        'LiLeftKidneyDepth': kidney_depths['LiLeftKidneyDepth'],
        'LiRightKidneyDepth': kidney_depths['LiRightKidneyDepth']
    })

def calculate_gfr():
    print("开始计算 GFR...")
    global kidney_depths, last_manufacturer, last_kidney_counts
    
    # 打印调试信息
    print("last_manufacturer:", last_manufacturer)
    print("last_kidney_counts:", last_kidney_counts)
    print("kidney_depths:", kidney_depths)

    if not last_manufacturer or not last_kidney_counts or not kidney_depths['leftDepth'] or not kidney_depths['rightDepth']:
        return jsonify({'success': False, 'message': '缺少计算 GFR 所需的数据'}), 400
    
    # 根据 manufacturer 判断 injection_count
    if 'GE MEDICAL SYSTEMS' in last_manufacturer:
        injection_count = 183916
    elif 'SIEMENS NM' in last_manufacturer:
        injection_count = 139857

    print("injection_count:", injection_count)

    # 分别计算左右肾 GFR
    left_count = last_kidney_counts['leftKidneyCount'] - last_kidney_counts['leftBackgroundCount']
    right_count = last_kidney_counts['rightKidneyCount'] - last_kidney_counts['rightBackgroundCount']
    left_depth = kidney_depths['leftDepth'] 
    right_depth = kidney_depths['rightDepth'] 

    u = 0.153
    import math
    eL = math.exp(-u * left_depth * 0.1)
    eR = math.exp(-u * right_depth * 0.1)

    GFR_left = ((left_count * 0.001) / eL / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
    GFR_right = ((right_count * 0.001) / eR / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
    GFR_total = GFR_left + GFR_right

    # 打印计算结果
    print("GFR计算结果：")
    print(" GFR_left:", round(GFR_left, 2))
    print(" GFR_right:", round(GFR_right, 2))
    print(" GFR_total:", round(GFR_total, 2))

    # 使用李氏公式计算GFR
    Li_left_depth = kidney_depths['LiLeftKidneyDepth']
    Li_right_depth = kidney_depths['LiRightKidneyDepth']
    eL_Li = math.exp(-u * Li_left_depth * 0.1)
    eR_Li = math.exp(-u * Li_right_depth * 0.1)

    Li_GFR_left = ((left_count * 0.001) / eL_Li / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
    Li_GFR_right = ((right_count * 0.001) / eR_Li / (injection_count * 0.001 * 6)) * 100 * 9.81270 - 6.82519
    Li_GFR_total = Li_GFR_left + Li_GFR_right

    print("李氏公式计算结果：")
    print(" Li_GFR_left:", round(Li_GFR_left, 2))
    print(" Li_GFR_right:", round(Li_GFR_right, 2))
    print(" Li_GFR_total:", round(Li_GFR_total, 2))

    return jsonify({
        'success': True,
        'gfr': {
            'leftGFR': round(GFR_left, 2),
            'rightGFR': round(GFR_right, 2),
            'totalGFR': round(GFR_total, 2),
            'LiLeftGFR': round(Li_GFR_left, 2),
            'LiRightGFR': round(Li_GFR_right, 2),
            'LiTotalGFR': round(Li_GFR_total, 2),
        }
    })
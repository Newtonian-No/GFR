# 肾动态显像预处理
import pydicom
import numpy as np
from scipy.ndimage import zoom
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import re



# 获取有效的文件名（移除非法字符）
def get_valid_filename(s):
    """
    将字符串转换为有效的文件名，移除非法字符
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

# 步骤1: 从DICOM序列中提取第62帧
def extract_frame(file_path, frame_number):
    """提取指定帧的DICOM数据并保存"""
    print(f"步骤1: 从{file_path}中提取第{frame_number+1}帧...")
    
    # 读取DICOM序列
    image = sitk.ReadImage(file_path)
    image_data = sitk.GetArrayFromImage(image)
    
    # 读取DICOM标签获取病人姓名
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    reader.ReadImageInformation()
    
    # 尝试获取病人姓名，如果不存在则使用默认名称
    try:
        patient_name = reader.GetMetaData("0010|0010").strip()
        patient_name = get_valid_filename(patient_name)
        if not patient_name:
            patient_name = "unknown"
    except:
        print("警告: 无法读取病人姓名，使用默认名称")
        patient_name = "unknown"
    
    if 0 <= frame_number < image_data.shape[0]:
        # 获取指定帧的数据
        frame_data = image_data[frame_number]
        
        # 将NumPy数组转换回SimpleITK图像
        frame_image = sitk.GetImageFromArray(frame_data)
        frame_image.SetOrigin(image.GetOrigin())
        frame_image.SetSpacing(image.GetSpacing())
        
        # 复制元数据
        meta_keys = image.GetMetaDataKeys()
        for key in meta_keys:
            frame_image.SetMetaData(key, image.GetMetaData(key))
        
        # 构建输出文件路径，使用病人姓名
        output_file_name = f"{patient_name}_frame{frame_number+1}.dcm"
        output_file_path = os.path.join("output/original_dcm", output_file_name)
        
        # 保存DICOM图像
        sitk.WriteImage(frame_image, output_file_path)
        print(f"已保存第{frame_number+1}帧到: {output_file_path}")
        
        return output_file_path, frame_data
    else:
        raise ValueError(f"帧索引{frame_number}超出范围[0, {image_data.shape[0]-1}]")

# 步骤2: 对DICOM图像进行重采样
def resample_dicom(input_path, order=1):
    """对DICOM图像进行重采样以获得等间距像素"""
    print(f"步骤2: 对{input_path}进行重采样...")
    
    # 读取DICOM文件
    ds = pydicom.dcmread(input_path, force=True)
    
    # 获取病人姓名
    try:
        patient_name = str(ds.PatientName).strip()
        patient_name = get_valid_filename(patient_name)
        if not patient_name:
            patient_name = "unknown"
    except:
        print("警告: 无法读取病人姓名，使用默认名称")
        patient_name = "unknown"
    
    # 从文件名获取帧号
    frame_number = os.path.basename(input_path).split('_')[-1].split('.')[0]
    if not frame_number.startswith('frame'):
        frame_number = "frameX"  # 默认帧号
        
    # 提取像素数据
    arr = ds.pixel_array
    original_shape = arr.shape
    
    # 获取并验证像素间距
    if 'PixelSpacing' not in ds:
        print(f"警告: 文件缺少PixelSpacing标签，使用默认值[1.0, 1.0]")
        original_spacing = [1.0, 1.0]
    else:
        original_spacing = [float(ps) for ps in ds.PixelSpacing]
    
    # 计算缩放因子（原始间距越大，放大倍数越大）
    zoom_factor = (original_spacing[0], original_spacing[1])
    
    # 执行重采样
    resampled_arr = zoom(arr, zoom_factor, order=order)
    
    # 转换回原始数据类型
    resampled_arr = resampled_arr.astype(arr.dtype)
    
    # 更新DICOM头信息
    ds.PixelSpacing = [1.0, 1.0]
    ds.Rows, ds.Columns = resampled_arr.shape
    
    # 更新像素数据
    ds.PixelData = resampled_arr.tobytes()
    
    # 保存文件，使用病人姓名
    output_file_name = f"{patient_name}_{frame_number}_resampled.dcm"
    output_path = os.path.join("output/resampled_dcm", output_file_name)
    ds.save_as(output_path)
    print(f"已保存重采样DICOM到: {output_path}")

    # 处理像素数据并归一化
    pixel_array = resampled_arr.astype(float)
    
    # 获取窗宽窗位（如果存在）
    window_center = getattr(ds, 'WindowCenter', None)
    window_width = getattr(ds, 'WindowWidth', None)
    
    # 如果是列表，取第一个元素
    if isinstance(window_center, (list, tuple)):
        window_center = window_center[0]
    if isinstance(window_width, (list, tuple)):
        window_width = window_width[0]
    if hasattr(window_center, '__getitem__'):
        window_center = window_center[0]
    if hasattr(window_width, '__getitem__'):
        window_width = window_width[0]

    # 如果存在窗宽窗位，按照窗宽窗位调整像素值
    if window_center is not None and window_width is not None:
        min_value = window_center - window_width / 2
        max_value = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, min_value, max_value)
        pixel_array = ((pixel_array - min_value) / (max_value - min_value)) * 255
    else:
        # 如果没有窗宽窗位，进行简单归一化
        if pixel_array.max() > pixel_array.min():
            pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255
    
    # 转换为uint8
    pixel_array = pixel_array.astype(np.uint8)
    
    # 保存为PNG - 使用PIL
    image = Image.fromarray(pixel_array)
    
    # 增强对比度（可选）
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)  # 增强因子1.2
    
    # 保存高质量PNG
    png_path = os.path.join("converted", f"{patient_name}_{frame_number}_resampled.png")
    image.save(png_path, compress_level=0)
    print(f"已保存重采样PNG图像到: {png_path}")
    
    return output_path, resampled_arr, png_path

def convert_and_display_dicom(dicom_path, custom_wc=None, custom_ww=None, enhance_contrast=True):
    """
    将单个DICOM文件转换为PNG格式的图像数组，并展示出来，不保存到本地。
    支持手动指定窗宽窗位以获得最佳显示效果。
    
    :param dicom_path: 单个DICOM文件的完整路径。
    :param custom_wc: 手动指定的窗位 (Window Center)。如果为 None，则尝试读取DICOM头。
    :param custom_ww: 手动指定的窗宽 (Window Width)。如果为 None，则尝试读取DICOM头。
    :param enhance_contrast: 是否应用对比度增强（默认为True）。
    :return: 转换并处理后的图像数组 (np.uint8)，如果失败则返回 None。
    """
    print(f"开始处理 DICOM 文件: {dicom_path}")

    
    try:
        # 1. 读取DICOM文件
        ds = pydicom.dcmread(dicom_path, force=True)
        pixel_array = ds.pixel_array.astype(float)
        
        # 2. 确定使用的窗宽窗位
        if custom_wc is not None and custom_ww is not None and float(custom_ww) > 0: # 优先使用用户手动指定的参数
            window_center = float(custom_wc)
            window_width = float(custom_ww)
            print(f"使用手动设置的 WC={window_center}, WW={window_width}")
        else:
            # 尝试从 DICOM 头读取
            window_center = getattr(ds, 'WindowCenter', None)
            window_width = getattr(ds, 'WindowWidth', None)
            
            # 检查是否为多值数据（列表、元组或类似对象）
            if isinstance(window_center, (list, tuple)) or (window_center is not None and hasattr(window_center, '__getitem__') and not isinstance(window_center, (str, bytes))):
                window_center = window_center[0]
                
            if isinstance(window_width, (list, tuple)) or (window_width is not None and hasattr(window_width, '__getitem__') and not isinstance(window_width, (str, bytes))):
                window_width = window_width[0]

            # 确保它们是浮点数或 None，以便后续进行数学运算
            if window_center is not None:
                 window_center = float(window_center)
            if window_width is not None:
                 window_width = float(window_width)
                 
            print(f"使用DICOM头信息 WC={window_center}, WW={window_width}")

        # 3. 应用窗宽窗位或简单归一化
        if window_center is not None and window_width is not None and window_width > 0:
            # 窗宽窗位调整
            min_value = window_center - window_width / 2
            max_value = window_center + window_width / 2
            
            # 应用裁剪和归一化
            pixel_array = np.clip(pixel_array, min_value, max_value)
            # 归一化到 0-255
            if max_value > min_value:
                pixel_array = ((pixel_array - min_value) / (max_value - min_value)) * 255
            else: # 避免除以零
                 pixel_array = np.zeros_like(pixel_array) 
        else:
            # 简单归一化 (min-max normalization)
            print("警告: 未使用有效的窗宽窗位调整，执行 Min-Max 归一化。这可能导致曝光过度或不足。")
            if pixel_array.max() > pixel_array.min():
                pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255
            else:
                pixel_array = np.zeros_like(pixel_array) # 单一值图像

        # 转换为uint8
        pixel_array = pixel_array.astype(np.uint8)

        # 4. 对比度增强（可选，使用 PIL 逻辑）
        if enhance_contrast:
            image = Image.fromarray(pixel_array)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # 增强因子1.2
            final_array = np.array(image)
        else:
            final_array = pixel_array

        # 5. 展示图像 (使用 matplotlib)
        plt.figure(figsize=(8, 8))
        
        # 使用 'gray' colormap 来正确显示灰度 DICOM 图像
        plt.imshow(final_array, cmap='gray')
        
        # 从文件名中提取标题，并显示使用的WC/WW
        title = os.path.basename(dicom_path)
        # 格式化显示的窗宽窗位信息
        wc_info = f"WC={window_center:.1f}" if window_center is not None else "WC=N/A"
        ww_info = f"WW={window_width:.1f}" if window_width is not None else "WW=N/A"
        plt.title(f"DICOM Display: {title} ({wc_info}, {ww_info})")
        
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        
        print("图像已成功转换并展示。")
        return final_array

    except FileNotFoundError:
        print(f"错误：文件未找到 at {dicom_path}")
        return None
    except Exception as e:
        print(f"处理DICOM文件 {dicom_path} 时发生错误: {e}")
        return None

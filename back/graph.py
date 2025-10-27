import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
import os
from scipy.ndimage.interpolation import zoom
from typing import Optional, Tuple, List

def calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label):
    # 获取指定肾脏区域的二值掩膜
    kidney_mask = segmentation_mask == kidney_label
    # 对 DICOM 图像应用掩膜
    kidney_image = dicom_image * kidney_mask
    # 计算肾脏区域的像素值之和
    sum_pixel_values = np.sum(kidney_image)
    return sum_pixel_values

def graph(dicom_path, ROI_data, output_path: Optional[str] = None) -> Tuple[float, float, float, float, str]:
    print(f"步骤6: 肾脏计数变化曲线...")
    # 读取原始dicom文件
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_images = dicom_data.pixel_array  # 获取所有帧的图像数据

    # 获取DICOM图像的实际尺寸
    dicom_shape = dicom_images[0].shape if len(dicom_images.shape) > 2 else dicom_images.shape
    print(f"DICOM图像尺寸: {dicom_shape}")

    ROI_data = np.squeeze(ROI_data)
    zoom_factors = (dicom_shape[0] / ROI_data.shape[0], dicom_shape[1] / ROI_data.shape[1])
    ROI_data = zoom(ROI_data, zoom_factors, order=0)

    segmentation_mask = ROI_data
    
    # 计算每一帧的左右肾像素值
    left_kidney_sums = []
    right_kidney_sums = []
    left_background_sums = []
    right_background_sums = []

    if len(dicom_images.shape) > 2:  # 多帧DICOM

        for i in range(dicom_images.shape[0]):
            dicom_image = dicom_images[i]
            left_kidney_sum = calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label=1) 
            right_kidney_sum = calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label=2) 
            left_background_sum = calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label=3) 
            right_background_sum = calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label=4) 
            left_kidney_sums.append(left_kidney_sum)
            right_kidney_sums.append(right_kidney_sum)
            left_background_sums.append(left_background_sum)
            right_background_sums.append(right_background_sum)
        
        # 绘制图像
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 定义横坐标
        x1 = np.linspace(0, 1, 60)  # 前60个值在0到1分钟之间
        x2 = [1, 2, 3, 4, 5]  # 第61到65个值的位置（1分钟到5分钟）

        # 纵坐标（计数/秒）
        left_y = left_kidney_sums[:60] + [left_kidney_sums[60] / 60, left_kidney_sums[61] / 60, left_kidney_sums[62] / 60, 
                                        left_kidney_sums[63] / 60, left_kidney_sums[64] / 60]
        right_y = right_kidney_sums[:60] + [right_kidney_sums[60] / 60, right_kidney_sums[61] / 60, right_kidney_sums[62] / 60, 
                                        right_kidney_sums[63] / 60, right_kidney_sums[64] / 60]

        # 合并横坐标和纵坐标
        x = list(x1) + x2

        # 绘制图像
        plt.figure(figsize=(10, 6), facecolor='lavender')  # 设置背景颜色
        plt.plot(x, left_y, color='red', label='左肾')  # 左肾曲线
        plt.plot(x, right_y, color='green', label='右肾')  # 右肾曲线

        # 设置横纵坐标范围和刻度
        plt.xlim(0, 5)  # 横坐标范围
        plt.ylim(0, max(max(left_y), max(right_y)) * 1.1)  # 纵坐标范围，根据最大值动态调整
        plt.xticks(np.arange(0, 5.5, 0.5))  # 横坐标刻度：0, 0.5, 1, ..., 5
        plt.yticks(np.arange(0, max(max(left_y), max(right_y)) * 1.1, 20))  # 纵坐标刻度，每20为一档

        # 返回第62帧（勾画的那一帧）的数据
        kidney_index = 61  # 通常是第62帧（索引从0开始，所以是61）

    else:  # 单帧DICOM
        print("检测到单帧DICOM，计算静态肾脏计数...")
        # 计算单帧图像的肾脏区域像素值总和
        left_kidney_sum = calculate_kidney_pixel_sum(dicom_images, segmentation_mask, kidney_label=1)
        right_kidney_sum = calculate_kidney_pixel_sum(dicom_images, segmentation_mask, kidney_label=2)
        left_background_sum = calculate_kidney_pixel_sum(dicom_images, segmentation_mask, kidney_label=3)
        right_background_sum = calculate_kidney_pixel_sum(dicom_images, segmentation_mask, kidney_label=4)
        
        left_kidney_sums = [left_kidney_sum]
        right_kidney_sums = [right_kidney_sum]
        left_background_sums = [left_background_sum]
        right_background_sums = [right_background_sum]
        
        # 创建一个简单的条形图来展示单帧的肾脏计数
        plt.figure(figsize=(8, 5), facecolor='lavender')
        bars = plt.bar(['左肾', '右肾'], [left_kidney_sum, right_kidney_sum], color=['red', 'green'])
        
        # 在条形上方显示计数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('肾脏区域像素值总和', fontsize=16)
        plt.ylabel('像素值总和', fontsize=12)
        
        # 对于单帧，返回第一个（唯一的）元素
        kidney_index = 0

    # 添加标题和坐标标签
    if len(dicom_images.shape) > 2:
        plt.title('Kidney Time-Activity Curve', fontsize=16, color='black')
        plt.xlabel('Minutes', fontsize=12)
        plt.ylabel('Counts/Second', fontsize=12)
        plt.legend()

    if output_path is None:
        # 如果没有提供 output_path，生成一个默认的绝对路径 (基于调用者预期的结构)
        # 注意：在 DicomProcessor 中我们会传入绝对路径，这里的 None 分支只是作为保障
        base_name = os.path.basename(dicom_path)
        name, ext = os.path.splitext(base_name)
        # 这里的相对路径 'ROI' 可能导致问题，最好在调用者端保证路径是绝对的
        output_file = os.path.join('ROI', f"counts_time_{name}.png") 
        print(f"[警告] graph 函数未提供 output_path，使用默认相对路径: {output_file}")
    else:
        # 使用传入的绝对路径
        output_file = output_path

    plt.savefig(output_file, facecolor='lavender')  # 保存图像
    plt.close()  # 关闭图像以释放内存

# 返回第62帧（勾画的那一帧）的左右肾像素值和背景区域像素值
    #return left_kidney_sums[61], right_kidney_sums[61], left_background_sums[61], right_background_sums[61], output_file

# 返回肾脏区域和背景区域的像素值之和
    return (left_kidney_sums[kidney_index], right_kidney_sums[kidney_index], 
            left_background_sums[kidney_index], right_background_sums[kidney_index], 
            output_file)

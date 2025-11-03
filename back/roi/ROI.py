# 单个nii处理 （可以多个）勾画ROI本底区域
import os
import numpy as np
from scipy.ndimage import find_objects, center_of_mass
import SimpleITK as sitk
import matplotlib.pyplot as plt

def add_background_roi(nii_file_path):
    """在肾脏分割结果下方添加背景ROI椭圆"""
    print(f"步骤4: 为{nii_file_path}添加肾脏本底区域...")
    
    # 读取原始的NIfTI文件
    label_image = sitk.ReadImage(nii_file_path)
    label_data = sitk.GetArrayFromImage(label_image)

    print(f"原始标签数据形状: {label_data.shape}")
    
    # 检查图像维度，确保是2D图像
    if len(label_data.shape) > 2:
        # 如果是3D图像，取中间切片
        label_data = label_data[label_data.shape[0]//2, :, :]
    
    # 定义标签和参数
    left_kidney_label = 1
    right_kidney_label = 2
    ellipse_label_left = 3  # 左肾背景标签
    ellipse_label_right = 4  # 右肾背景标签
    
    # 找到左右肾脏的位置
    left_kidney_mask = (label_data == left_kidney_label)
    right_kidney_mask = (label_data == right_kidney_label)
    
    # 检查左右肾脏是否存在
    left_kidney_exists = np.any(left_kidney_mask)
    right_kidney_exists = np.any(right_kidney_mask)
    
    print(f"左肾存在: {left_kidney_exists}")
    print(f"右肾存在: {right_kidney_exists}")
    
    # 参数设置
    ellipse_width = 20  # 椭圆的宽度
    ellipse_height = 10  # 椭圆的高度
    ellipse_offset = 10  # 椭圆与肾脏底部的距离
    ellipse_angle = 30  # 椭圆的倾斜角度
    
    # 绘制左肾椭圆
    if left_kidney_exists:
        try:
            from scipy.ndimage import find_objects, center_of_mass
            left_kidney_objects = find_objects(left_kidney_mask)
            if left_kidney_objects:
                left_kidney_bottom = left_kidney_objects[-1][0].stop - 1
                left_kidney_center = np.array(center_of_mass(left_kidney_mask))
                
                # 绘制倾斜椭圆
                draw_tilted_ellipse(
                    label_data, 
                    left_kidney_center, 
                    left_kidney_bottom, 
                    ellipse_width, 
                    ellipse_height, 
                    ellipse_offset, 
                    ellipse_angle, 
                    ellipse_label_left, 
                    x_shift=-15
                )
                print("已添加左肾背景ROI")
            else:
                print("无法确定左肾位置，跳过添加左肾背景ROI")
        except Exception as e:
            print(f"添加左肾背景ROI时出错: {e}")
    
    # 绘制右肾椭圆
    if right_kidney_exists:
        try:
            from scipy.ndimage import find_objects, center_of_mass
            right_kidney_objects = find_objects(right_kidney_mask)
            if right_kidney_objects:
                right_kidney_bottom = right_kidney_objects[-1][0].stop - 1
                right_kidney_center = np.array(center_of_mass(right_kidney_mask))
                
                # 绘制倾斜椭圆
                draw_tilted_ellipse(
                    label_data, 
                    right_kidney_center, 
                    right_kidney_bottom, 
                    ellipse_width, 
                    ellipse_height, 
                    ellipse_offset, 
                    -ellipse_angle, 
                    ellipse_label_right, 
                    x_shift=15
                )
                print("已添加右肾背景ROI")
            else:
                print("无法确定右肾位置，跳过添加右肾背景ROI")
        except Exception as e:
            print(f"添加右肾背景ROI时出错: {e}")
    
    # 生成输出文件路径
    _, filename = os.path.split(nii_file_path)
    output_path = os.path.join("output/ROI", f"ROI_{filename}")
    
    # 保存修改后的标签
    new_label_image = sitk.GetImageFromArray(label_data)
    new_label_image.CopyInformation(label_image)
    sitk.WriteImage(new_label_image, output_path)
    print(f"已保存ROI掩码到: {output_path}")
    
    return output_path, label_data

def draw_tilted_ellipse(label_data, center, bottom, width, height, offset, angle, label_value, x_shift=0):
    """绘制倾斜的椭圆"""
    # 椭圆中心位置
    ellipse_center_y = int(bottom) + offset
    ellipse_center_x = int(center[1]) + x_shift
    
    # 创建网格
    y, x = np.ogrid[:label_data.shape[0], :label_data.shape[1]]
    y = y - ellipse_center_y
    x = x - ellipse_center_x
    
    # 旋转坐标
    theta = np.radians(angle)
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    
    # 椭圆方程
    ellipse_mask = (x_rot**2 / (width / 2)**2 + y_rot**2 / (height / 2)**2) <= 1
    
    # 边界检查，防止越界
    valid_mask = (
        (ellipse_mask) & 
        (0 <= y + ellipse_center_y) & (y + ellipse_center_y < label_data.shape[0]) &
        (0 <= x + ellipse_center_x) & (x + ellipse_center_x < label_data.shape[1])
    )
    
    # 应用掩膜
    label_data[valid_mask] = label_value
    
    return label_data

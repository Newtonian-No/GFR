import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 添加一个新函数用于显示DICOM和ROI掩码叠加图像
def display_dicom_with_roi_overlay(dicom_image_path, segmentation_mask, output_path=None):
    """
    显示DICOM图像与ROI掩码的叠加效果
    
    参数:
        dicom_image: DICOM图像数组 重采样后的
        segmentation_mask: 分割掩码数组
        output_path: 如果提供，保存图像到该路径
    """
    print(f"步骤5: 生成ROI叠加结果...")
    
    # 读取DICOM图像
    ds = pydicom.dcmread(dicom_image_path, force=True)
    dicom_image = ds.pixel_array

    # 压缩维度（nii为3维时需要）
    segmentation_mask = np.squeeze(segmentation_mask)
 
    # 确保DICOM图像和掩码具有相同的尺寸
    if dicom_image.shape != segmentation_mask.shape:
        # 调整DICOM图像尺寸以匹配掩码
        zoom_factors = (segmentation_mask.shape[0] / dicom_image.shape[0], 
                        segmentation_mask.shape[1] / dicom_image.shape[1])
        dicom_image = zoom(dicom_image, zoom_factors, order=1)
    
    # 创建图形，只显示叠加后的图像
    plt.figure(figsize=(10, 8))
    
    # 显示基础DICOM图像
    plt.imshow(dicom_image, cmap="gray")
    
    # 为每个标签创建叠加层
    colors = [(1, 0, 0, 0.5), (0, 0, 1, 0.5), (0, 1, 0, 0.5), (1, 1, 0, 0.5)]
    labels = [1, 2, 3, 4]
    
    for label, color in zip(labels, colors):
        mask = segmentation_mask == label
        if np.any(mask):  # 仅当掩码非空时才叠加
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = color
            plt.imshow(colored_mask)
    
    plt.axis('off')
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=(1, 0, 0, 0.5), label='左肾'),
        plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 1, 0.5), label='右肾'),
        plt.Rectangle((0, 0), 1, 1, fc=(0, 1, 0, 0.5), label='左本底'),
        plt.Rectangle((0, 0), 1, 1, fc=(1, 1, 0, 0.5), label='右本底')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # 保存图像
    if output_path is None:
        output_path = os.path.join("ROI", f"overlay_{os.path.basename(dicom_image_path).split('.')[0]}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像，不显示
    
    print(f"已保存叠加图像到: {output_path}")
    return output_path

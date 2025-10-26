"""
从选定的文件夹中读取CT图像并找到其中肾脏区域像素占比最多的切片
"""
import os
import glob
import torch
import numpy as np
import pydicom
from skimage.transform import resize
from scipy import ndimage
import torch.nn.functional as F
from pathlib import Path
import sys
import matplotlib.pyplot as plt
# 导入 vit_seg_configs 模块
from back.networks import vit_seg_configs

# 导入 vit_seg_modeling_mamba 模块中的 VisionTransformer 类
from back.networks.vit_seg_modeling_mamba import VisionTransformer
from back.constants import CTFIND_MODEL_PATH
# --- 全局/常量参数 (来自您的 train_kidney_segmentor.py) ---
TARGET_SIZE = 256
NUM_CLASSES = 3  # 背景(0), 左肾(1), 右肾(2)
GLOBAL_MEAN = -60.0  # 示例值
GLOBAL_STD = 300.0   # 示例值




def load_ct_slices(dcm_folder_path):
    """
    加载文件夹中所有的 DICOM 文件，并按位置（Slice Location）排序重建 3D 图像。
    注意：此函数仅为演示，实际生产代码需要更健壮的 DICOM 处理。
    """
    dcm_files = glob.glob(os.path.join(dcm_folder_path, '*.dcm'))
    if not dcm_files:
        raise FileNotFoundError(f"未找到 DICOM 文件: {dcm_folder_path}")

    slices = [pydicom.dcmread(f) for f in dcm_files]
    
    # 尝试根据 SliceLocation 排序，如果缺失则根据 InstanceNumber
    try:
        slices.sort(key=lambda x: x.SliceLocation)
    except AttributeError:
        # 如果没有 SliceLocation，则使用 InstanceNumber
        slices.sort(key=lambda x: x.InstanceNumber)

    # 提取像素数据和 Hounsfield Unit (HU) 转换参数
    # HU = pixel_value * RescaleSlope + RescaleIntercept
    image_3d = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float32)

    # 应用 HU 转换
    for i, s in enumerate(slices):
        slope = s.RescaleSlope if 'RescaleSlope' in s else 1
        intercept = s.RescaleIntercept if 'RescaleIntercept' in s else 0
        image_3d[..., i] = image_3d[..., i] * slope + intercept

    # 将轴顺序从 [H, W, D] 转换为模型期望的 [D, H, W]
    image_3d = np.transpose(image_3d, (2, 0, 1))
    
    return image_3d, slices # 返回 3D 数组和切片列表


def find_deepest_kidney_slice(dcm_folder_path, model_weights_path='/weights/final_kidney_segmenter.pth'):
    """
    传入一个 DICOM 文件夹，使用模型找到其中左右肾在 Y 轴（前后）跨度之和最大的切片并返回。

    Args:
        dcm_folder_path (str): 包含病人所有 .dcm 文件的文件夹路径。
        model_weights_path (str): 训练好的模型权重 .pth 文件路径。

    Returns:
        tuple: (最大总深度切片的索引, 最大总深度切片的DICOM文件名, 左右肾Y轴最大总跨度, 预测的分割图)
               如果加载失败则返回 None
    """
    print("调用 find_deepest_kidney_slice 函数")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    config = vit_seg_configs.get_r50_l16_config()
    config.n_classes = NUM_CLASSES 
    config.n_channels = 1
    
    try:
        model = VisionTransformer(config, img_size=TARGET_SIZE, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"加载模型或权重时出错: {e}")
        return None

    # 2. 加载和预处理 3D CT 图像
    try:
        # slices 列表是已经根据位置排序的
        image_3d, slices = load_ct_slices(dcm_folder_path)
    except Exception as e:
        print(f"加载 DICOM 文件时出错: {e}")
        return None

    D, H, W = image_3d.shape
    
    # 追踪最大总Y轴跨度（总深度）
    max_total_depth_pixels = -1 
    deepest_slice_idx = -1
    best_segmentation_map = None

    # 3. 遍历所有切片进行预测
    with torch.no_grad():
        for i in range(D):
            image_slice = image_3d[i, :, :]
            
            # 3.1 强度归一化 (Z-Score)
            normalized_slice = (image_slice - GLOBAL_MEAN) / GLOBAL_STD
            
            # 3.2 强制 Resize 到 TARGET_SIZE (256x256)
            zoom_factors = (TARGET_SIZE / H, TARGET_SIZE / W)
            resized_slice = resize(normalized_slice, 
                                   (TARGET_SIZE, TARGET_SIZE), 
                                   order=1, # 线性插值
                                   mode='constant',
                                   preserve_range=True).astype(np.float32)

            # 3.3 转换为 Tensor 并送入模型
            image_tensor = torch.from_numpy(resized_slice).unsqueeze(0).unsqueeze(0).to(device)

            # 3.4 模型推理
            output = model(image_tensor)
            
            # 3.5 后处理：获取分割图 (Softmax -> Argmax)
            probabilities = F.softmax(output, dim=1)
            predicted_mask = torch.argmax(probabilities, dim=1).cpu().squeeze(0).numpy() # (H, W)

            # 4. 分别计算左肾(1)和右肾(2)在 Y 轴（前后）的最大跨度
            
            # predicted_mask 是 (H, W)，对应于 (Y, X) 轴
            
            # --- 左肾 (类别 1) ---
            left_kidney_pixels = np.where(predicted_mask == 1)
            left_y_span = 0
            if left_kidney_pixels[0].size > 0:
                # Y 轴对应于行索引 (H 维度)
                left_y_min = left_kidney_pixels[0].min()
                left_y_max = left_kidney_pixels[0].max()
                # 跨度 = 最大索引 - 最小索引 + 1 (像素计数)
                left_y_span = left_y_max - left_y_min + 1
                
            # --- 右肾 (类别 2) ---
            right_kidney_pixels = np.where(predicted_mask == 2)
            right_y_span = 0
            if right_kidney_pixels[0].size > 0:
                # Y 轴对应于行索引 (H 维度)
                right_y_min = right_kidney_pixels[0].min()
                right_y_max = right_kidney_pixels[0].max()
                # 跨度 = 最大索引 - 最小索引 + 1 (像素计数)
                right_y_span = right_y_max - right_y_min + 1
                
            # --- 计算总深度 (左肾 Y 跨度 + 右肾 Y 跨度) ---
            current_total_depth_pixels = left_y_span + right_y_span

            # 5. 更新最大总深度
            if current_total_depth_pixels > max_total_depth_pixels:
                max_total_depth_pixels = current_total_depth_pixels
                deepest_slice_idx = i
                # 记录该切片的左右肾单独深度
                best_left_depth_pixels = left_y_span
                best_right_depth_pixels = right_y_span
                
                # 将分割图 Resize 回原始切片尺寸 (H, W)
                original_mask = ndimage.zoom(predicted_mask.astype(np.uint8), 
                                             (H / TARGET_SIZE, W / TARGET_SIZE), 
                                             order=0, 
                                             mode='nearest')
                best_segmentation_map = original_mask
                
    # 6. 确定最大切片的原始 DICOM 文件名
    if deepest_slice_idx != -1:
        # slices 列表是排序后的，deepest_slice_idx 是该列表中的索引
        deepest_slice_filename = os.path.basename(slices[deepest_slice_idx].filename)
        # 注意: 此时 max_total_depth_pixels 是在 TARGET_SIZE (256x256) 图像上的像素数
        # 如果需要真实的物理距离 (mm)，需要结合切片的 Pixel Spacing 参数
    else:
        deepest_slice_filename = "N/A"
        
    return (deepest_slice_idx, deepest_slice_filename, max_total_depth_pixels, best_segmentation_map , best_left_depth_pixels, best_right_depth_pixels)

# def find_deepest_kidney_slice(dcm_folder_path, model_weights_path='final_kidney_segmenter.pth'):
#     """
#     传入一个 DICOM 文件夹，使用模型找到其中肾脏在 Y 轴（前后）跨度最大的切片并返回。

#     Args:
#         dcm_folder_path (str): 包含病人所有 .dcm 文件的文件夹路径。
#         model_weights_path (str): 训练好的模型权重 .pth 文件路径。

#     Returns:
#         tuple: (最大深度切片的索引, 最大深度切片的DICOM文件名, 肾脏Y轴最大跨度, 预测的分割图)
#                如果加载失败则返回 None
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 1. 加载模型
#     config = vit_seg_configs.get_r50_l16_config()
#     config.n_classes = NUM_CLASSES 
#     config.n_channels = 1
    
#     try:
#         model = VisionTransformer(config, img_size=TARGET_SIZE, num_classes=NUM_CLASSES)
#         model.load_state_dict(torch.load(model_weights_path, map_location=device))
#         model.to(device)
#         model.eval()
#     except Exception as e:
#         print(f"加载模型或权重时出错: {e}")
#         return None

#     # 2. 加载和预处理 3D CT 图像
#     try:
#         # slices 列表是已经根据位置排序的
#         image_3d, slices = load_ct_slices(dcm_folder_path)
#     except Exception as e:
#         print(f"加载 DICOM 文件时出错: {e}")
#         return None

#     D, H, W = image_3d.shape
    
#     # 追踪最大Y轴跨度（深度）
#     max_y_span_pixels = -1 
#     deepest_slice_idx = -1
#     best_segmentation_map = None

#     # 3. 遍历所有切片进行预测
#     with torch.no_grad():
#         for i in range(D):
#             image_slice = image_3d[i, :, :]
            
#             # 3.1 强度归一化 (Z-Score)
#             normalized_slice = (image_slice - GLOBAL_MEAN) / GLOBAL_STD
            
#             # 3.2 强制 Resize 到 TARGET_SIZE (256x256)
#             zoom_factors = (TARGET_SIZE / H, TARGET_SIZE / W)
#             resized_slice = resize(normalized_slice, 
#                                    (TARGET_SIZE, TARGET_SIZE), 
#                                    order=1, # 线性插值
#                                    mode='constant',
#                                    preserve_range=True).astype(np.float32)

#             # 3.3 转换为 Tensor 并送入模型
#             image_tensor = torch.from_numpy(resized_slice).unsqueeze(0).unsqueeze(0).to(device)

#             # 3.4 模型推理
#             output = model(image_tensor)
            
#             # 3.5 后处理：获取分割图 (Softmax -> Argmax)
#             probabilities = F.softmax(output, dim=1)
#             predicted_mask = torch.argmax(probabilities, dim=1).cpu().squeeze(0).numpy() # (H, W)

#             # 4. 计算肾脏在 Y 轴（前后）的最大跨度
            
#             # predicted_mask 是 (H, W)，对应于 (Y, X) 轴
#             kidney_pixels = np.where(predicted_mask != 0)
            
#             current_y_span = 0
#             if kidney_pixels[0].size > 0:
#                 # Y 轴对应于行索引 (H 维度)
#                 y_min = kidney_pixels[0].min()
#                 y_max = kidney_pixels[0].max()
#                 # 跨度 = 最大索引 - 最小索引 + 1 (像素计数)
#                 current_y_span = y_max - y_min + 1 

#             # 5. 更新最大深度
#             if current_y_span > max_y_span_pixels:
#                 max_y_span_pixels = current_y_span
#                 deepest_slice_idx = i
                
#                 # 将分割图 Resize 回原始切片尺寸 (H, W)
#                 original_mask = ndimage.zoom(predicted_mask.astype(np.uint8), 
#                                              (H / TARGET_SIZE, W / TARGET_SIZE), 
#                                              order=0, 
#                                              mode='nearest')
#                 best_segmentation_map = original_mask
                
#     # 6. 确定最大切片的原始 DICOM 文件名
#     if deepest_slice_idx != -1:
#         # slices 列表是排序后的，deepest_slice_idx 是该列表中的索引
#         deepest_slice_filename = os.path.basename(slices[deepest_slice_idx].filename)
#         # 注意: 此时 max_y_span_pixels 是在 TARGET_SIZE (256x256) 图像上的像素数
#         # 如果需要真实的物理距离 (mm)，需要结合切片的 Pixel Spacing 参数
#     else:
#         deepest_slice_filename = "N/A"
        
#     return (deepest_slice_idx, deepest_slice_filename, max_y_span_pixels, best_segmentation_map)


# --- 示例用法 (假设您的模型和数据路径存在) ---
if __name__ == '__main__':
    PATIENT_DCM_PATH = '/home/kevin/Documents/GFR/肾动态显像+CT/WANGAIFANG/CT'  # 替换为实际路径
    # OUTPUT_SAVE_DIR = 'test_output_deepest_slice'
    
    # 确保保存目录存在
    # os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True) 
    
    result = find_deepest_kidney_slice(PATIENT_DCM_PATH, CTFIND_MODEL_PATH)
    
    if result:
        idx, filename, depth_pixels, mask , left_depth_pixels, right_depth_pixels = result
        print(f"\n--- Y轴跨度最大（最深）切片信息 ---")
        print(f"  索引 (D轴): {idx}")
        print(f"  原始DICOM文件名: {filename}")
        print(f"  肾脏Y轴跨度 (256x256像素): {depth_pixels}") 
        print(f"  分割图尺寸: {mask.shape}")
        print(f"  左肾Y轴跨度 (256x256像素): {left_depth_pixels}")
        print(f"  右肾Y轴跨度 (256x256像素): {right_depth_pixels}")

        # --- 新增功能：展示 DICOM 图并保存 ---
        full_dcm_path = os.path.join(PATIENT_DCM_PATH, filename)
        # output_png_path = os.path.join(OUTPUT_SAVE_DIR, f"deepest_slice_{idx}_{filename.replace('.dcm', '.png')}")

        try:
            print("开始展示")
            # 1. 读取 DICOM 文件
            ds = pydicom.dcmread(full_dcm_path)
            image_data = ds.pixel_array

            # 2. 绘制图像和分割掩模
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # 显示 DICOM 图像 (使用灰度图)
            ax.imshow(image_data, cmap=plt.cm.gray)
            
            # 叠加分割掩模 (使用不同的颜色和透明度)
            # 创建一个只包含掩模中非零部分的 NumPy 掩码数组
            masked_image = np.ma.masked_where(mask == 0, mask)
            ax.imshow(masked_image, cmap='jet', alpha=0.5)
            
            title_text = f"最深切片: D 轴 {idx} ({filename})"
            

            ax.axis('off')

            # 3. 保存图像
            # plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
            # print(f"  [保存成功] 最深切片图像已保存至: {output_png_path}")

            # 4. 显示图像（仅在本地测试时使用）
            plt.show() 
            
        except Exception as e:
            print(f"  [错误] 绘制或保存图像失败: {e}")

    else:
        print("未找到最深切片信息。")
    


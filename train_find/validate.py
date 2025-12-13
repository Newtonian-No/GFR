# --- START OF FILE validation.py ---
import argparse
import logging
import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torchvision import transforms

# 导入核心依赖
import pydicom
import nibabel as nib 
import SimpleITK as sitk
from typing import List, Tuple, Dict, Any, Optional

# 导入用户提供的模块
# 假设这些模块都在同一个包结构下，以相对导入方式引用
from .networks.vit_seg_configs import get_r50_l16_config
from .networks.vit_seg_modeling_bimambaattention import VisionTransformer as TransUnet
from .dataloader import GFRDataset # 仅作为参考，但我们将定义自己的推理数据集
from .roi.ROI import add_background_roi
from .roi.graph import graph

# ==============================================================================
# 1. 推理专用的 DICOM 数据加载器 (提取特定帧并预处理)
# ==============================================================================

class ValidationDicomDataset(Dataset):
    """
    用于验证/推理的多帧 DICOM 数据集加载器。
    它提取特定的关键帧 (默认为第62帧，即索引61) 进行预处理，以匹配训练输入。
    同时返回原始多帧 DICOM 路径，供 graph.py 使用。
    """
    def __init__(self, data_dir: str, img_size: int = 224, target_frame_index: int = 61):
        self.data_dir = data_dir
        self.img_size = img_size
        self.target_frame_index = target_frame_index # 默认第62帧，索引 61
        self.dcm_files = self._get_dcm_files()
        
        # 预处理流程必须与 dataloader.py::GFRDataset 保持一致
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 将 PIL Image 转换为 (1, H, W) Tensor
        ])
        
    def _get_dcm_files(self) -> List[str]:
        """查找目录中所有扩展名为 .dcm 的文件。"""
        dcm_files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.dcm'):
                dcm_files.append(os.path.join(self.data_dir, filename))
        return sorted(dcm_files)

    def __len__(self):
        return len(self.dcm_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        full_dicom_path = self.dcm_files[idx]
        
        # 1. 读取多帧 DICOM 数据
        try:
            dcm_data = pydicom.dcmread(full_dicom_path, force=True)
            image_array_full = dcm_data.pixel_array.astype(np.float32)
            
            # 从 DICOM 中提取病人姓名 (用于输出文件命名)
            patient_name = str(getattr(dcm_data, 'PatientName', f'Patient_{idx}')).strip()
            # 简单处理病人姓名，移除非法字符
            patient_name = "".join(c for c in patient_name if c.isalnum() or c in ('_', '-'))
            if not patient_name:
                patient_name = f'Patient_{idx}'
            
        except Exception as e:
            logging.warning(f"无法读取 DICOM 文件 {full_dicom_path}: {e}. 跳过此文件.")
            return {'image': torch.zeros(1, self.img_size, self.img_size), 
                    'patient_name': os.path.basename(full_dicom_path), 
                    'full_dicom_path': full_dicom_path,
                    'valid': False}

        # 2. 提取目标帧 (用于推理)
        if image_array_full.ndim == 3 and image_array_full.shape[0] > self.target_frame_index:
            image_frame = image_array_full[self.target_frame_index, :, :]
        elif image_array_full.ndim == 2:
            # 可能是单帧 DICOM，直接使用
            image_frame = image_array_full
        else:
            logging.warning(f"DICOM 尺寸 {image_array_full.shape} 异常或目标帧 {self.target_frame_index} 不存在. 跳过.")
            return {'image': torch.zeros(1, self.img_size, self.img_size), 
                    'patient_name': patient_name, 
                    'full_dicom_path': full_dicom_path,
                    'valid': False}
        
        # --- 3. 归一化 (与 dataloader.py 一致) ---
        # Min-Max 归一化到 0-1 
        image_normalized = (image_frame - image_frame.min()) / (image_frame.max() - image_frame.min() + 1e-8)

        # --- 4. 调整大小 (Resize) (与 dataloader.py 一致) ---
        img_pil = Image.fromarray((image_normalized * 255).astype(np.uint8))
        # Image.BICUBIC 对应于 GFRDataset 中的 Image.BICUBIC
        img_pil = img_pil.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        
        # --- 5. 转换为 Tensor ---
        image_tensor = self.transform(img_pil) # 变为 (1, H, W)

        sample = {'image': image_tensor, 
                  'patient_name': patient_name, 
                  'full_dicom_path': full_dicom_path,
                  'valid': True}
        return sample

# ==============================================================================
# 2. 推理主函数
# ==============================================================================

def inference(args):
    # --- 1. 设备设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 2. 路径设置和临时目录创建 ---
    os.makedirs(args.output_dir, exist_ok=True)
    temp_nii_dir = os.path.join(args.output_dir, 'temp_nii_masks')
    os.makedirs(temp_nii_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ROI'), exist_ok=True) # graph.py/ROI.py 可能会用到
    os.makedirs(os.path.join(args.output_dir, 'output/ROI'), exist_ok=True) # ROI.py 可能会用到
    
    log_file = os.path.join(args.output_dir, f'validation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Validation started. Logs saved to {log_file}")
    logging.info(f"Args: {args}")

    # --- 3. 模型配置与加载 ---
    config = get_r50_l16_config()
    config.n_classes = args.num_classes
    config.n_skip = 3
    
    # 应用之前讨论的 Bug 修复：确保 config.patches.grid 正确
    # 224 / 16 = 14
    if args.img_size % 16 != 0:
        logging.error(f"Img_size {args.img_size} 必须是 16 的倍数以匹配 ResNet/ViT 结构。")
        shutil.rmtree(temp_nii_dir)
        return

    grid_size = int(args.img_size / 16)
    config.patches.grid = (grid_size, grid_size)
    logging.info(f"Applying config fix: config.patches.grid set to ({grid_size}, {grid_size})")

    model = TransUnet(config, img_size=args.img_size, num_classes=args.num_classes)
    
    # 加载权重
    try:
        # 加载 DDP 模式下保存的权重
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Successfully loaded model weights from {args.model_path}")
    except Exception as e:
        logging.error(f"Failed to load model weights from {args.model_path}: {e}")
        shutil.rmtree(temp_nii_dir)
        return

    model.to(device)
    model.eval() 

    # --- 4. 数据加载 ---
    dataset = ValidationDicomDataset(
        data_dir=args.data_dir, 
        img_size=args.img_size,
        target_frame_index=args.target_frame_index # 默认 61
    )
    if len(dataset) == 0:
        logging.warning("在指定目录中没有找到 .dcm 文件。")
        shutil.rmtree(temp_nii_dir)
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True
    )
    logging.info(f"找到 {len(dataset)} 个文件进行验证/分析。")

    # --- 5. 推理、后处理与分析循环 ---
    analysis_results = []
    
    output_path = os.path.join(args.output_dir, f'gfr_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    with torch.no_grad():
        for sampled_batch in tqdm(dataloader, desc="Inference & Analysis"):
            
            valid_mask = sampled_batch['valid'] # 布尔 Tensor
            
            # --- 修正 Bug: 过滤非 Tensor 数据 (patient_name, dicom_path) ---
            
            # 1. 提取完整的 Python 列表数据
            patient_names_full = sampled_batch['patient_name']
            dicom_paths_full = sampled_batch['full_dicom_path']
            
            # 2. 将布尔 Tensor 转换为 NumPy 数组，用于索引 Python 列表
            valid_indices = valid_mask.cpu().numpy().astype(bool)
            
            # 3. 过滤出有效的样本列表
            valid_patient_names = np.array(patient_names_full)[valid_indices].tolist()
            valid_dicom_paths = np.array(dicom_paths_full)[valid_indices].tolist()
            # ----------------------------------------------------

            image_batch = sampled_batch['image'][valid_mask].to(device)
            
            # 过滤无效数据并记录 (使用原始索引 i)
            for i, mask in enumerate(valid_mask):
                if not mask:
                    # 记录无效文件，使用原始列表的索引
                    analysis_results.append({
                        'patient_name': patient_names_full[i],
                        'left_kidney_count': 'N/A',
                        'right_kidney_count': 'N/A',
                        # ... 省略其他 N/A 字段
                        'status': 'DATA_ERROR'
                    })

            if image_batch.numel() == 0:
                continue
                
            # 模型推理
            # outputs: [B_valid, NumClasses, H, W]
            outputs = model(image_batch) 
            
            # 预测类别：[B_valid, H, W]
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # 逐个样本进行后处理和分析 (循环次数为 valid 样本数)
            for i in range(predictions.shape[0]): 
                # 现在可以直接通过 i 索引已经过滤好的有效样本列表
                patient_name = valid_patient_names[i]
                full_dicom_path = valid_dicom_paths[i]
                pred_mask = predictions[i] # [H, W], 标签 0, 1, 2
                
                logging.info(f"-> Processing Patient: {patient_name}")
                
                # --- 5a. 临时保存 NIfTI 文件 (模拟模型输出) ---
                # 必须创建一个 NIfTI 文件，因为 ROI.py::add_background_roi 接收的是文件路径
                temp_nii_name = f"{patient_name}_pred_mask.nii.gz"
                temp_nii_path = os.path.join(temp_nii_dir, temp_nii_name)
                
                # 使用 SimpleITK 或 nibabel 创建 NIfTI 文件
                # 由于 ROI.py 使用 SimpleITK 读取，这里也用它创建
                try:
                    # 将预测的 mask (尺寸: args.img_size x args.img_size) 转换为 SimpleITK Image
                    pred_itk = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
                    sitk.WriteImage(pred_itk, temp_nii_path)
                    
                    # --- 5b. 调用 ROI.py 勾画背景区域 ---
                    # 返回的 ROI_data 是包含标签 1, 2, 3, 4 的 NumPy 数组
                    roi_nii_path, roi_data_array = add_background_roi(temp_nii_path)
                    
                    # --- 5c. 调用 graph.py 进行计数分析和曲线绘制 ---
                    # output_dir 用于 graph.py 保存曲线图，我们使用主输出目录
                    count_output_path = os.path.join(args.output_dir, f'counts_{patient_name}.png')
                    half_output_path = os.path.join(args.output_dir, f'half_curve_{patient_name}.png')

                    (lk_count, rk_count, lbg_count, rbg_count, 
                     curve_path, half_metrics, half_curve_path_out) = graph(
                        dicom_path=full_dicom_path,
                        ROI_data=roi_data_array, # 使用从 ROI.py 返回的 NumPy 数组
                        output_path=count_output_path,
                        half_output_path=half_output_path
                    )
                    
                    # 提取半排时间
                    l_thalf = half_metrics['left']['t_half'] if half_metrics else 'N/A'
                    r_thalf = half_metrics['right']['t_half'] if half_metrics else 'N/A'
                    
                    analysis_results.append({
                        'patient_name': patient_name,
                        'left_kidney_count': f'{lk_count:.2f}',
                        'right_kidney_count': f'{rk_count:.2f}',
                        'left_background_count': f'{lbg_count:.2f}',
                        'right_background_count': f'{rbg_count:.2f}',
                        'left_thalf': f'{l_thalf}' if l_thalf is not None else 'N/A',
                        'right_thalf': f'{r_thalf}' if r_thalf is not None else 'N/A',
                        'status': 'SUCCESS'
                    })
                    logging.info(f"-> Analysis SUCCESS. LK:{lk_count:.0f}, RK:{rk_count:.0f}, LT½:{l_thalf}, RT½:{r_thalf}")

                except Exception as e:
                    logging.error(f"Error processing {patient_name} in post-analysis chain: {e}")
                    analysis_results.append({
                        'patient_name': patient_name,
                        'left_kidney_count': 'N/A',
                        'right_kidney_count': 'N/A',
                        'left_background_count': 'N/A',
                        'right_background_count': 'N/A',
                        'left_thalf': 'N/A',
                        'right_thalf': 'N/A',
                        'status': f'ANALYSIS_ERROR: {e}'
                    })
    
    # --- 6. 结果写入 TXT 文件 ---
    
    header = "{:<20} | {:<15} | {:<15} | {:<15} | {:<15} | {:<8} | {:<8} | {}\n".format(
        "Patient Name", "LK Count", "RK Count", "LBG Count", "RBG Count", "L T1/2", "R T1/2", "Status"
    )

    with open(output_path, 'w') as f:
        f.write("Validation and GFR Analysis Results\n")
        f.write("-" * 125 + "\n")
        f.write(header)
        f.write("-" * 125 + "\n")
        
        for res in analysis_results:
            line = "{:<20} | {:<15} | {:<15} | {:<15} | {:<15} | {:<8} | {:<8} | {}\n".format(
                res['patient_name'], 
                res['left_kidney_count'], 
                res['right_kidney_count'], 
                res['left_background_count'], 
                res['right_background_count'], 
                res['left_thalf'], 
                res['right_thalf'], 
                res['status']
            )
            f.write(line)
            
    # --- 7. 清理 ---
    shutil.rmtree(temp_nii_dir)
    logging.info(f"Inference and analysis complete. Temporary files removed.")
    logging.info(f"Final results saved to {output_path}")

# ==============================================================================
# 3. 主函数与参数解析
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and GFR Analysis using TransUNet with BiMamba")
    
    # 路径参数
    parser.add_argument('--data_dir', type=str, default='/home/kevin/Documents/GFR/200例像素值-放射性计数拟合数据/200kidneycounts_RESAMPLE/images/', help='Directory containing the FULL MULTI-FRAME .dcm files for analysis')
    parser.add_argument('--model_path', type=str, default='/home/kevin/Code/ROI/back/weights/results_single_gpu/epoch_last.pth', help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='validation_analysis_results/', help='Output directory for logs, results, and plots')
    
    # 推理参数 (必须与训练时一致)
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size for inference')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input (must match training)')
    parser.add_argument('--num_classes', type=int, default=3, help='output channel of network (e.g., 3 for BG, Left, Right)')
    parser.add_argument('--target_frame_index', type=int, default=61, help='The frame index (0-based) to extract from the multi-frame DICOM for segmentation. Default 61 (62nd frame).')

    args = parser.parse_args()
    inference(args)
# --- END OF FILE validation.py ---

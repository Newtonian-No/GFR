# --- START OF FILE validation.py ---
import argparse
import logging
import os
import shutil
import numpy as np
import torch
import csv  # 1. 导入 csv 库
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
from .networks.vit_seg_configs import get_r50_b16_config
from .networks.vit_seg_modeling import VisionTransformer as TransUnet
from .dataloader import GFRDataset 
from .roi.ROI import add_background_roi
from .roi.graph import graph

# ==============================================================================
# 1. 推理专用的 DICOM 数据加载器 (保持不变)
# ==============================================================================

class ValidationDicomDataset(Dataset):
    def __init__(self, data_dir: str, img_size: int = 224, target_frame_index: int = 61):
        self.data_dir = data_dir
        self.img_size = img_size
        self.target_frame_index = target_frame_index 
        self.dcm_files = self._get_dcm_files()
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
        ])
        
    def _get_dcm_files(self) -> List[str]:
        dcm_files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.dcm'):
                dcm_files.append(os.path.join(self.data_dir, filename))
        return sorted(dcm_files)

    def __len__(self):
        return len(self.dcm_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        full_dicom_path = self.dcm_files[idx]
        try:
            dcm_data = pydicom.dcmread(full_dicom_path, force=True)
            image_array_full = dcm_data.pixel_array.astype(np.float32)
            patient_name = str(getattr(dcm_data, 'PatientName', f'Patient_{idx}')).strip()
            patient_name = "".join(c for c in patient_name if c.isalnum() or c in ('_', '-'))
            if not patient_name:
                patient_name = f'Patient_{idx}'
        except Exception as e:
            return {'image': torch.zeros(1, self.img_size, self.img_size), 
                    'patient_name': os.path.basename(full_dicom_path), 
                    'full_dicom_path': full_dicom_path,
                    'valid': False}

        if image_array_full.ndim == 3 and image_array_full.shape[0] > self.target_frame_index:
            image_frame = image_array_full[self.target_frame_index, :, :]
        elif image_array_full.ndim == 2:
            image_frame = image_array_full
        else:
            return {'image': torch.zeros(1, self.img_size, self.img_size), 
                    'patient_name': patient_name, 
                    'full_dicom_path': full_dicom_path,
                    'valid': False}
        
        image_normalized = (image_frame - image_frame.min()) / (image_frame.max() - image_frame.min() + 1e-8)
        img_pil = Image.fromarray((image_normalized * 255).astype(np.uint8))
        img_pil = img_pil.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        image_tensor = self.transform(img_pil) 

        return {'image': image_tensor, 'patient_name': patient_name, 'full_dicom_path': full_dicom_path, 'valid': True}

# ==============================================================================
# 2. 推理主函数
# ==============================================================================

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('output/ROI', exist_ok=True)
    temp_nii_dir = os.path.join(args.output_dir, 'temp_nii_masks')
    os.makedirs(temp_nii_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, f'validation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    console = logging.StreamHandler(); console.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s'))
    logging.getLogger('').addHandler(console)
    
    # 模型配置与加载
    config = get_r50_b16_config()
    config.n_classes, config.n_skip = args.num_classes, 3
    grid_size = int(args.img_size / 16)
    config.patches.grid = (grid_size, grid_size)

    model = TransUnet(config, img_size=args.img_size, num_classes=args.num_classes)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Loaded weights from {args.model_path}")
    except Exception as e:
        logging.error(f"Load failed: {e}"); shutil.rmtree(temp_nii_dir); return

    model.to(device).eval() 

    # 数据加载
    dataset = ValidationDicomDataset(args.data_dir, args.img_size, args.target_frame_index)
    if len(dataset) == 0:
        logging.warning("No .dcm files found."); shutil.rmtree(temp_nii_dir); return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    analysis_results = []
    # 2. 修改输出路径后缀为 .csv
    output_path = os.path.join(args.output_dir, f'gfr_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

    with torch.no_grad():
        for sampled_batch in tqdm(dataloader, desc="Inference & Analysis"):
            valid_mask = sampled_batch['valid']
            patient_names_full = sampled_batch['patient_name']
            dicom_paths_full = sampled_batch['full_dicom_path']
            valid_indices = valid_mask.cpu().numpy().astype(bool)
            
            valid_patient_names = np.array(patient_names_full)[valid_indices].tolist()
            valid_dicom_paths = np.array(dicom_paths_full)[valid_indices].tolist()

            # 处理无效样本
            for i, mask in enumerate(valid_mask):
                if not mask:
                    analysis_results.append({
                        'patient_name': patient_names_full[i],
                        'left_kidney_count': 'N/A', 'right_kidney_count': 'N/A',
                        'left_background_count': 'N/A', 'right_background_count': 'N/A',
                        'left_thalf': 'N/A', 'right_thalf': 'N/A', 'status': 'DATA_ERROR'
                    })

            image_batch = sampled_batch['image'][valid_mask].to(device)
            if image_batch.numel() == 0: continue
                
            outputs = model(image_batch) 
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for i in range(predictions.shape[0]): 
                patient_name, full_dicom_path, pred_mask = valid_patient_names[i], valid_dicom_paths[i], predictions[i]
                
                try:
                    temp_nii_path = os.path.join(temp_nii_dir, f"{patient_name}_pred_mask.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(pred_mask.astype(np.uint8)), temp_nii_path)
                    
                    roi_nii_path, roi_data_array = add_background_roi(temp_nii_path)
                    
                    count_out = os.path.join(args.output_dir, f'counts_{patient_name}.png')
                    half_out = os.path.join(args.output_dir, f'half_curve_{patient_name}.png')

                    (lk, rk, lbg, rbg, _, half_metrics, _) = graph(
                        dicom_path=full_dicom_path, ROI_data=roi_data_array,
                        output_path=count_out, half_output_path=half_out
                    )
                    
                    l_thalf = half_metrics['left']['t_half'] if half_metrics else 'N/A'
                    r_thalf = half_metrics['right']['t_half'] if half_metrics else 'N/A'
                    
                    analysis_results.append({
                        'patient_name': patient_name,
                        'left_kidney_count': f'{lk:.2f}', 'right_kidney_count': f'{rk:.2f}',
                        'left_background_count': f'{lbg:.2f}', 'right_background_count': f'{rbg:.2f}',
                        'left_thalf': f'{l_thalf}' if l_thalf is not None else 'N/A',
                        'right_thalf': f'{r_thalf}' if r_thalf is not None else 'N/A',
                        'status': 'SUCCESS'
                    })
                except Exception as e:
                    analysis_results.append({
                        'patient_name': patient_name,
                        'left_kidney_count': 'N/A', 'right_kidney_count': 'N/A',
                        'left_background_count': 'N/A', 'right_background_count': 'N/A',
                        'left_thalf': 'N/A', 'right_thalf': 'N/A', 'status': f'ERROR: {e}'
                    })
    
    # --- 3. 结果写入 CSV 文件 ---
    # 定义与原代码相同的列名
    csv_header = ["Patient Name", "LK Count", "RK Count", "LBG Count", "RBG Count", "L T1/2", "R T1/2", "Status"]
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(csv_header)
            
            # 写入数据行
            for res in analysis_results:
                writer.writerow([
                    res['patient_name'], 
                    res['left_kidney_count'], 
                    res['right_kidney_count'], 
                    res['left_background_count'], 
                    res['right_background_count'], 
                    res['left_thalf'], 
                    res['right_thalf'], 
                    res['status']
                ])
        logging.info(f"Final results saved to CSV: {output_path}")
    except Exception as e:
        logging.error(f"Failed to write CSV: {e}")
            
    shutil.rmtree(temp_nii_dir)
    logging.info(f"Inference and analysis complete.")

# ==============================================================================
# 3. 主函数与参数解析 (保持不变)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and GFR Analysis using TransUNet with BiMamba")
    parser.add_argument('--data_dir', type=str, default='/home/kevin/Documents/GFR/200例像素值-放射性计数拟合数据/200kidneycounts_RESAMPLE/images/')
    parser.add_argument('--model_path', type=str, default='/home/kevin/Code/ROI/back/weights/best_epoch_weights.pth')
    parser.add_argument('--output_dir', type=str, default='validation_results/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--target_frame_index', type=int, default=61)

    args = parser.parse_args()
    inference(args)
# --- END OF FILE validation.py ---
import argparse
import logging
import os
import shutil
import csv
import numpy as np
import torch  # 修复之前的 NameError
import pydicom
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# --- 导入软件原有的预处理和分析模块 ---
# 确保这些模块在你的 PYTHONPATH 中
from back.roi.preprocess import extract_frame, resample_dicom 
from back.roi.predict import segment_kidney 
from back.roi.ROI import add_background_roi
from back.roi.graph import graph

def run_validation(args):
    # 1. 环境准备与设备检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 关键修复：预先创建软件逻辑依赖的输出目录 ---
    # 软件中的 resample_dicom 似乎硬编码或依赖于这些路径
    required_dirs = [
        'output',
        'output/resampled_dcm',
        'output/converted_png',
        'output/segmentation',     # <--- 修复你当前报错的关键
        'output/segmented_nii',    # 预防下一个可能的报错
        'output/graph_results',
        'output/ROI',
        args.output_dir
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
        print(f"目录已就绪: {d}")

    output_base = Path(args.output_dir)
    
    # 2. 获取 DICOM 文件列表
    dcm_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.dcm')])
    if not dcm_files:
        print(f"错误: 在 {args.data_dir} 中未找到 .dcm 文件")
        return

    analysis_results = []
    
    # 3. 循环处理
    for filename in tqdm(dcm_files, desc="批量验证进度"):
        dicom_path = os.path.join(args.data_dir, filename)
        patient_name = Path(filename).stem
        
        try:
            # 步骤 1: 检查并提取多帧
            dcm_data = pydicom.dcmread(dicom_path)
            is_multiframe = dcm_data.get('NumberOfFrames', 1) > 1
            
            if is_multiframe:
                # 调用软件的 extract_frame
                original_dcm_working, _ = extract_frame(dicom_path, args.target_frame_index)
            else:
                original_dcm_working = dicom_path

            # 步骤 2: 重采样 (这里会自动保存到 output/resampled_dcm/)
            # 软件逻辑会读取保存后的结果，所以我们必须确保路径存在
            png_output_path = f"output/converted_png/{patient_name}_resampled.png"
            
            # 调用软件函数
            resampled_dcm_path, _, _ = resample_dicom(
                original_dcm_working, 
                output_png_path=png_output_path, 
                order=1
            )

            # 步骤 3 & 4: 分割与 ROI 提取
            # 直接调用 predict.py 里的函数，保证预处理和模型加载完全一致
            segmentation_path, _ = segment_kidney(
                resampled_dcm_path, 
                args.model_path, 
                img_size=args.img_size, 
                num_classes=args.num_classes
            )
            
            roi_path, roi_data = add_background_roi(segmentation_path)

            # 步骤 5: 绘图与分析 (graph 函数)
            graph_out = str(output_base / f"{patient_name}_counts.png")
            half_out = str(output_base / f"{patient_name}_half.png")

            res_tuple = graph(
                dicom_path, # 曲线计算通常使用原始多帧序列
                roi_data,
                output_path=graph_out,
                half_output_path=half_out
            )
            
            lk, rk, lbg, rbg, _, half_metrics, _ = res_tuple

            # --- 安全提取半衰期数据 (防止 NoneType 报错) ---
            l_t = "N/A"
            r_t = "N/A"
            if half_metrics and isinstance(half_metrics, dict):
                left_info = half_metrics.get('left')
                right_info = half_metrics.get('right')
                if left_info: l_t = f"{left_info.get('t_half', 'N/A')}"
                if right_info: r_t = f"{right_info.get('t_half', 'N/A')}"

            analysis_results.append([
                patient_name, f"{lk:.2f}", f"{rk:.2f}", 
                f"{lbg:.2f}", f"{rbg:.2f}", l_t, r_t, "SUCCESS"
            ])

        except Exception as e:
            logging.error(f"处理失败 {patient_name}: {str(e)}")
            analysis_results.append([patient_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", f"ERROR: {e}"])

    # 4. 生成 CSV 报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_report = output_base / f"validation_report_{timestamp}.csv"
    
    with open(csv_report, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Patient Name", "LK Count", "RK Count", "LBG Count", "RBG Count", "L T1/2", "R T1/2", "Status"])
        writer.writerows(analysis_results)

    print(f"\n验证完成！CSV 结果保存在: {csv_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/kevin/Documents/GFR/200例像素值-放射性计数拟合数据/200kidneycounts_RESAMPLE/images/')
    parser.add_argument('--model_path', type=str, default='/home/kevin/Code/ROI/back/weights/best_epoch_weights.pth')
    parser.add_argument('--output_dir', type=str, default='validation_results')
    parser.add_argument('--target_frame_index', type=int, default=61)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=3)
    
    args = parser.parse_args()
    run_validation(args)



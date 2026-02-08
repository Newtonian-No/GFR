import os
import csv
import traceback
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# 导入你的处理类和工具函数
from back.local_dicom_process import DicomProcessor
from back.utils import calculate_metric_percase  # 确保路径正确

def load_nifti_as_array(file_path):
    """读取 nii.gz 文件并返回 numpy 数组"""
    img = sitk.ReadImage(str(file_path))
    return sitk.GetArrayFromImage(img)

def batch_validate_with_metrics(test_dir: str, output_csv: str):
    """
    批量处理 test 文件夹，包含 images 和 labels 子文件夹
    计算计数信息以及分割指标 (Dice, HD95 等)
    """
    processor = DicomProcessor()
    
    test_path = Path(test_dir)
    images_dir = test_path / "images"
    labels_dir = test_path / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"错误: 确保 {test_dir} 下存在 'images' 和 'labels' 文件夹")
        return

    # 准备 CSV 表头
    # 假设 utils.calculate_metric_percase 返回的是 [Dice, HD95]
    headers = [
        "Patient Name", "LK Count", "RK Count", "LBG Count", "RBG Count", 
        "Dice_Mean", "HD95_Mean", "Status", "File Name"
    ]
    
    results = []
    dicom_files = list(images_dir.glob('*.dcm'))
    print(f"找到 {len(dicom_files)} 个测试病例，开始验证...")

    for dcm_path in dicom_files:
        case_id = dcm_path.stem  # 获取文件名（不带扩展名）作为匹配 ID
        print(f"正在处理: {case_id}")
        
        # 1. 寻找对应的金标准标签文件 (假设是 .nii.gz 格式)
        possible_label_names = [
            f"{case_id}.nii",
            f"{case_id}.nii.gz",
            f"{case_id}_label.nii",
            f"{case_id}_label.nii.gz"
        ]
        
        label_path = None
        for name in possible_label_names:
            p = labels_dir / name
            if p.exists():
                label_path = p
                break
        
        row = {
            "Patient Name": "N/A", "LK Count": 0, "RK Count": 0, 
            "LBG Count": 0, "RBG Count": 0, "Dice_Mean": 0, 
            "HD95_Mean": 0, "Status": "Label Missing", "File Name": dcm_path.name
        }

        if not label_path.exists():
            print(f"警告: 未找到 {case_id} 对应的标签文件，跳过指标计算")
            results.append(row)
            continue

        try:
            # 2. 调用 DicomProcessor 运行模型推理并获取计数
            # process_dynamic_study_dicom 会生成预测文件到 SEGMENTED_OUTPUT_DIR
            res = processor.process_dynamic_study_dicom(str(dcm_path))
            
            if res.get('success'):
                p_info = res.get('patientInfo', {})
                k_counts = res.get('kidneyCounts', {})
                
                # 更新基础信息
                row.update({
                    "Patient Name": p_info.get('name', 'Unknown'),
                    "LK Count": k_counts.get('leftKidneyCount', 0),
                    "RK Count": k_counts.get('rightKidneyCount', 0),
                    "LBG Count": k_counts.get('leftBackgroundCount', 0),
                    "RBG Count": k_counts.get('rightBackgroundCount', 0),
                })

                # 3. 加载预测结果和金标准进行指标对比
                # 预测结果路径通常在你的 constants 定义的 SEGMENTED_OUTPUT_DIR 下
                pred_file_path = Path(res.get('segmentedFilePath', "")) # 确保 processor 返回了这个路径
                
                if pred_file_path.exists():
                    pred_arr = load_nifti_as_array(pred_file_path)
                    gt_arr = load_nifti_as_array(label_path)
                    
                    # 确保数据是二值的 (0 或 1)
                    pred_arr = (pred_arr > 0).astype(np.uint8)
                    gt_arr = (gt_arr > 0).astype(np.uint8)

                    # 调用 utils 中的函数计算指标
                    # 注意：如果你的模型是多分类（左肾、右肾），需要分别计算
                    metrics = calculate_metric_percase(pred_arr, gt_arr)
                    
                    row["Dice_Mean"] = round(metrics[0], 4)
                    row["HD95_Mean"] = round(metrics[1], 4)
                    row["Status"] = "Success"
                else:
                    row["Status"] = "Prediction File Not Found"
            else:
                row["Status"] = f"Processing Failed: {res.get('message')}"

        except Exception as e:
            traceback.print_exc()
            row["Status"] = f"Error: {str(e)}"
            
        results.append(row)

    # 4. 写入 CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n验证完成！结果已保存至: {output_csv}")

if __name__ == "__main__":
    # 这里的 test 文件夹结构应为:
    # test/
    #   ├── images/ (存放 .dcm)
    #   └── labels/ (存放 .nii.gz)
    TEST_DATA_DIR = "/media/kevin/3167F095163AC0C3/GFR/整理的数据集/肾动态显像/原始数据/GFR/test/" 
    OUTPUT_REPORT = "model_validation_report.csv"
    
    batch_validate_with_metrics(TEST_DATA_DIR, OUTPUT_REPORT)
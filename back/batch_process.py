import os
import csv
import traceback
from pathlib import Path
# 导入你提供的处理类
from back.local_dicom_process import DicomProcessor

def batch_process_to_csv(input_dir: str, output_csv: str):
    """
    批量处理文件夹下的DICOM文件，并将结果保存到CSV
    """
    # 1. 初始化处理器
    processor = DicomProcessor()
    
    # 2. 准备 CSV 表头
    headers = ["Patient Name", "LK Count", "RK Count", "LBG Count", "RBG Count", "File Path", "Status"]
    
    # 确保输入路径存在
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return

    results = []
    
    # 3. 遍历目录下的所有 .dcm 文件
    dicom_files = [f for f in input_path.glob('**/*.dcm')]
    print(f"找到 {len(dicom_files)} 个 DICOM 文件，开始处理...")

    for idx, dcm_path in enumerate(dicom_files):
        print(f"[{idx+1}/{len(dicom_files)}] 正在处理: {dcm_path.name}")
        
        try:
            # 调用核心处理逻辑
            # 注意：process_dynamic_study_dicom 会内部触发分割、ROI生成和计数计算
            res = processor.process_dynamic_study_dicom(str(dcm_path))
            
            if res.get('success'):
                pred_path_str = res.get('segmentedFilePath')
                print(f"DEBUG: 模型输出的预测文件路径为: {pred_path_str}")
                
                p_info = res.get('patientInfo', {})
                k_counts = res.get('kidneyCounts', {})
                
                row = {
                    "Patient Name": p_info.get('name', 'Unknown'),
                    "LK Count": k_counts.get('leftKidneyCount', 0),
                    "RK Count": k_counts.get('rightKidneyCount', 0),
                    "LBG Count": k_counts.get('leftBackgroundCount', 0),
                    "RBG Count": k_counts.get('rightBackgroundCount', 0),
                    "File Path": dcm_path.name,
                    "Status": "Success"
                }
            else:
                row = {
                    "Patient Name": "N/A",
                    "LK Count": 0, "RK Count": 0, "LBG Count": 0, "RBG Count": 0,
                    "File Path": dcm_path.name,
                    "Status": f"Failed: {res.get('message')}"
                }
                
        except Exception as e:
            print(f"处理文件 {dcm_path.name} 时发生异常")
            traceback.print_exc()
            row = {
                "Patient Name": "Error",
                "LK Count": 0, "RK Count": 0, "LBG Count": 0, "RBG Count": 0,
                "File Path": dcm_path.name,
                "Status": f"Exception: {str(e)}"
            }
            
        results.append(row)

    # 4. 写入 CSV 文件
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n处理完成！结果已保存至: {output_csv}")
    except Exception as e:
        print(f"保存 CSV 失败: {e}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 存放肾动态显像DCM文件的文件夹路径
    INPUT_DICOM_FOLDER = "/media/kevin/3167F095163AC0C3/GFR/200例像素值-放射性计数拟合数据/200kidneycounts/images/" 
    # 输出CSV的路径
    OUTPUT_CSV_PATH = "validation_results.csv"
    # ----------------
    
    batch_process_to_csv(INPUT_DICOM_FOLDER, OUTPUT_CSV_PATH)
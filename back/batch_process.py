import os
import csv
from back.local_dicom_process import DicomProcessor
# 1. 基础配置
input_folder = '/home/kevin/Documents/GFR/200例像素值-放射性计数拟合数据/200kidneycounts_RESAMPLE/images/'  # 替换为你的DICOM文件夹路径
output_csv = 'batch_results.csv'        # 输出结果文件名

# 2. 实例化你的类 (假设类名为 KidneyProcessor)
processor = DicomProcessor() 

results = []
print("开始处理...")

# 3. 循环遍历文件夹
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.dcm'):
        file_path = os.path.join(input_folder, filename)
        
        # 调用你的函数
        res = processor.process_dynamic_study_dicom(file_path) # 如果在类内部测试用self，外部用processor
        
        # 提取核心数据
        if res['success']:
            row = {
                '文件名': filename,
                '姓名': res['patientInfo'].get('name', '未知'),
                '状态': '成功',
                '左肾计数': res['kidneyCounts'].get('left', 0),
                '右肾计数': res['kidneyCounts'].get('right', 0),
                '消息': ''
            }
        else:
            row = {
                '文件名': filename,
                '姓名': 'N/A',
                '状态': '失败',
                '左肾计数': 0,
                '右肾计数': 0,
                '消息': res.get('message', '未知错误')
            }
        
        results.append(row)
        print(f"已处理: {filename} -> {row['姓名']}")

# 4. 写入 CSV
if results:
    keys = results[0].keys()
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

print(f"\n验证完成！结果已写入 {output_csv}")
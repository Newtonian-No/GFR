import sys
from pathlib import Path
import os

if getattr(sys, 'frozen', False):
    # 打包后：使用 PyInstaller 临时目录作为根目录
    PROJECT_ROOT = Path(sys._MEIPASS) 
else:
    # 正常运行时：使用脚本文件所在的目录作为根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path(os.getcwd()) / 'output'

MODEL_PATH_RELATIVE1 = 'back/weights/final_kidney_segmenter.pth'
CTFIND_MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE1
MODEL_PATH_RELATIVE2 = 'back/weights/best.pt'
DETECT_MODEL_WEIGHTS_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE2
# MODEL_PATH_RELATIVE3 = 'back/weights/best_epoch_weights.pth'
# LOCAL_DICOM_MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE3
MODEL_PATH_RELATIVE4 = 'back/weights/best_epoch_weights.pth'
LOCAL_DICOM_MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE4

MODEL_PATH_RELATIVE4 = 'back/weights/best.pt'  #yolo13，这里还在用旧的
DETECT_MODEL_WEIGHTS_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE4
# 新增：定义所有关键的输出目录路径 (绝对路径)
OUTPUT_DIR = OUTPUT_ROOT

# 动态显像相关输出目录
CONVERTED_OUTPUT_DIR = OUTPUT_DIR / 'converted' # 用于原始图像
SEGMENTED_OUTPUT_DIR = OUTPUT_DIR / 'ROI'       # 用于叠加图
GRAPH_OUTPUT_DIR = OUTPUT_DIR  / 'curve'        # 用于曲线图

# CT/深度相关输出目录
DEPTH_OUTPUT_DIR = OUTPUT_DIR / 'depth'     
SEG_DEPTH_OUTPUT_DIR = OUTPUT_DIR / 'seg_depth'   
ORIGINAL_DCM_DIR = OUTPUT_DIR / 'original_dcm'
RESAMPLED_DCM_DIR = OUTPUT_DIR / 'resampled_dcm'
SEGMENTATION_DIR = OUTPUT_DIR / 'segmentation'
# 1. 原始 DICOM 文件存储 (备份/归档)
ORIGINAL_CT_DIR = OUTPUT_DIR / 'original_dcm_backup'
# 2. 原始图片 PNG 路径 (用于前端展示原始切片)
ORIGINAL_PNG_DIR = OUTPUT_DIR / 'original_png' 
# 3. 深度分析可视化叠加图路径 (最终结果图)
OVERLAY_PNG_DIR = OUTPUT_DIR / 'overlay_png' 
# 4. YOLO 标签文件存储路径 (YOLO TXT)
YOLO_LABELS_DIR = OUTPUT_DIR / 'yolo_labels'
# 5. 中间结果和 JSON 汇总 (每个切片的详细数据)
ANALYSIS_RESULTS_DIR = OUTPUT_DIR / 'analysis_results'

# 将所有需要创建的目录路径放在一个列表中
ALL_OUTPUT_DIRS = [
    OUTPUT_DIR, CONVERTED_OUTPUT_DIR, SEGMENTED_OUTPUT_DIR, GRAPH_OUTPUT_DIR,
    DEPTH_OUTPUT_DIR, ORIGINAL_DCM_DIR, RESAMPLED_DCM_DIR, SEGMENTATION_DIR,
    ORIGINAL_CT_DIR, SEG_DEPTH_OUTPUT_DIR, ORIGINAL_PNG_DIR, OVERLAY_PNG_DIR,
    YOLO_LABELS_DIR, ANALYSIS_RESULTS_DIR
]
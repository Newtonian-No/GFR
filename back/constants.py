import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    # 打包后：使用 PyInstaller 临时目录作为根目录
    PROJECT_ROOT = Path(sys._MEIPASS) 
else:
    # 正常运行时：使用脚本文件所在的目录作为根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH_RELATIVE1 = 'back/weights/final_kidney_segmenter.pth'
CTFIND_MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE1
MODEL_PATH_RELATIVE2 = 'back/weights/best.pt'
DETECT_MODEL_WEIGHTS_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE2
MODEL_PATH_RELATIVE3 = 'back/weights/best_epoch_weights.pth'
LOCAL_DICOM_MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE3

# 新增：定义所有关键的输出目录路径 (绝对路径)
OUTPUT_DIR = PROJECT_ROOT / 'output'

# 动态显像相关输出目录
CONVERTED_OUTPUT_DIR = OUTPUT_DIR / 'converted' # 用于原始图像
SEGMENTED_OUTPUT_DIR = OUTPUT_DIR / 'ROI'       # 用于叠加图
GRAPH_OUTPUT_DIR = OUTPUT_DIR  / 'curve'        # 用于曲线图

# CT/深度相关输出目录
DEPTH_OUTPUT_DIR = OUTPUT_DIR / 'depth'        
ORIGINAL_DCM_DIR = OUTPUT_DIR / 'original_dcm'
ORIGINAL_CT_FILE_DIR = OUTPUT_DIR / 'original_ct_file' # 当传入CT文件夹时 用于保存原始CT文件
ORIGINAL_CT_DIR = OUTPUT_DIR / 'original_ct' # 当传入CT文件时 用于保存原始CT文件    
RESAMPLED_DCM_DIR = OUTPUT_DIR / 'resampled_dcm'
SEGMENTATION_DIR = OUTPUT_DIR / 'segmentation'

# 上传目录
UPLOAD_DIR = PROJECT_ROOT / 'uploads'

# 将所有需要创建的目录路径放在一个列表中
ALL_OUTPUT_DIRS = [
    OUTPUT_DIR, CONVERTED_OUTPUT_DIR, SEGMENTED_OUTPUT_DIR, GRAPH_OUTPUT_DIR,
    DEPTH_OUTPUT_DIR, ORIGINAL_DCM_DIR, RESAMPLED_DCM_DIR, SEGMENTATION_DIR,
    UPLOAD_DIR, ORIGINAL_CT_FILE_DIR, ORIGINAL_CT_DIR
]
import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    # 打包后：使用 PyInstaller 临时目录作为根目录
    PROJECT_ROOT = Path(sys._MEIPASS) 
else:
    # 正常运行时：使用脚本文件所在的目录作为根目录
    PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH_RELATIVE1 = 'weights/final_kidney_segmenter.pth'
CTFIND_MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE1
MODEL_PATH_RELATIVE2 = 'weights/best.pt'
DETECT_MODEL_WEIGHTS_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE2
MODEL_PATH_RELATIVE3 = 'weights/best_epoch_weights.pth'
LOCAL_DICOM_MODEL_PATH = PROJECT_ROOT / MODEL_PATH_RELATIVE3
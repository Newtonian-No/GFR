import nibabel as nib
import numpy as np
import os

def check_nii_labels(nii_file_path):
    """
    加载一个NIfTI文件，并返回其中所有唯一的非零像素值（labels）。

    Args:
        nii_file_path (str): .nii 或 .nii.gz 文件的完整路径。

    Returns:
        set: 包含所有唯一非零label值的集合。
             如果文件不存在或无法加载，则返回空集合。
    """
    if not os.path.exists(nii_file_path):
        print(f"错误：文件不存在于路径：{nii_file_path}")
        return set()

    try:

        img = nib.load(nii_file_path)

        data = img.get_fdata()

        non_zero_data = data[data != 0]

        unique_labels = np.unique(non_zero_data)

        return set(unique_labels.astype(int)) 

    except nib.filebased.FileOrUrlNotFound:
        print(f"错误：无法找到或加载文件：{nii_file_path}")
        return set()
    except Exception as e:
        print(f"发生错误：{e}")
        return set()

if __name__ == '__main__':
    your_nii_file = '/media/kevin/3167F095163AC0C3/GFR/整理的数据集/肾动态显像/原始数据/GFR/test/labels/G-CHAO DA MING_frame_61.nii'
    print(f"正在检查文件: {your_nii_file}")
    labels = check_nii_labels(your_nii_file)
    print(f"发现的唯一 Label 集合: {labels}")
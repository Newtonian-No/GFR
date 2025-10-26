'''交互式DICOM转png，允许实时调整窗宽窗位。'''
'''窗位871,窗宽701   这个值附近不错'''
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider # 新增导入！
from PIL import Image, ImageEnhance # （用于对比度增强，但在这里简化，直接用np数组） 

def apply_window_level(pixel_array, wc, ww):
    """根据窗位窗宽调整像素数组，并归一化到 0-255"""
    
    # 确保窗宽有效
    if ww <= 0:
        return np.zeros_like(pixel_array, dtype=np.uint8)

    min_value = wc - ww / 2
    max_value = wc + ww / 2
    
    # 应用裁剪
    img_clipped = np.clip(pixel_array, min_value, max_value)
    
    # 归一化到 0-255
    if max_value > min_value:
        img_normalized = ((img_clipped - min_value) / (max_value - min_value)) * 255
    else:
        img_normalized = np.zeros_like(pixel_array)

    # 转换为 uint8
    return img_normalized.astype(np.uint8)


def interactive_dicom_viewer_and_get_params(dicom_path, initial_wc=40, initial_ww=400):
    """
    加载DICOM文件，并提供交互式滑块实时调整窗宽窗位。
    当窗口关闭时，返回最终的 WC 和 WW 值。
    """
    print(f"加载 DICOM 文件: {dicom_path} 进行交互式调整...")

    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        
        # 转换为 HU 值（假设 Rescale Slope/Intercept 存在）
        raw_pixel_array = ds.pixel_array.astype(float)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu_array = raw_pixel_array * slope + intercept
        
        data_min = hu_array.min()
        data_max = hu_array.max()
        data_range = data_max - data_min

    except Exception as e:
        print(f"错误: 无法加载或处理DICOM文件 {dicom_path}: {e}")
        return None, None
    
    # 用于存储最终 WC 和 WW 的可变对象
    final_params = {'wc': initial_wc, 'ww': initial_ww}

    # --- Matplotlib 设置 ---
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    initial_img = apply_window_level(hu_array, initial_wc, initial_ww)
    image_display = ax.imshow(initial_img, cmap='gray')
    ax.set_title(f"DICOM Viewer: {os.path.basename(dicom_path)}\nWC={initial_wc:.1f}, WW={initial_ww:.1f}\n(Close window to proceed with YOLO)")
    ax.axis('off')

    axcolor = 'lightgoldenrodyellow'
    ax_wc = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=axcolor)
    wc_slider = Slider(ax=ax_wc, label='Window Center (WC)', valmin=data_min, valmax=data_max, valinit=initial_wc, valstep=1.0)
    ax_ww = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)
    ww_slider = Slider(ax=ax_ww, label='Window Width (WW)', valmin=1.0, valmax=data_range, valinit=initial_ww, valstep=5.0)
    
    def update_image(val):
        wc = wc_slider.val
        ww = ww_slider.val
        
        new_img = apply_window_level(hu_array, wc, ww)
        image_display.set_data(new_img)
        
        ax.set_title(f"DICOM Viewer: {os.path.basename(dicom_path)}\nWC={wc:.1f}, WW={ww:.1f}\n(Close window to proceed with YOLO)")
        
        # 实时更新最终值
        final_params['wc'] = wc
        final_params['ww'] = ww
        
        fig.canvas.draw_idle()

    wc_slider.on_changed(update_image)
    ww_slider.on_changed(update_image)
    
    # 阻塞程序，直到用户关闭窗口
    plt.show()

    # 返回窗口关闭时的最终值
    print(f"交互式调整完成。最终 WC={final_params['wc']:.1f}, WW={final_params['ww']:.1f}")
    return final_params['wc'], final_params['ww'], hu_array




# def main():
#     test_dicom_path = "/home/kevin/Documents/GFR/整理的数据集/CT数据/testsets-dicom/CLZ-ImageFileName63.dcm"
#     if os.path.exists(test_dicom_path):
#         interactive_dicom_viewer(test_dicom_path, initial_wc=40, initial_ww=400)
#     else:
#         print(f"请替换 {test_dicom_path} 为有效的 DICOM 文件路径。")

# if __name__ == "__main__":
#     main()

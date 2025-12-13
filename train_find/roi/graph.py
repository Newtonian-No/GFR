import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
import os
from scipy.ndimage.interpolation import zoom
from typing import Optional, Tuple, List, Dict

def calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label):
    # 获取指定肾脏区域的二值掩膜
    kidney_mask = segmentation_mask == kidney_label
    # 对 DICOM 图像应用掩膜
    kidney_image = dicom_image * kidney_mask
    # 计算肾脏区域的像素值之和
    sum_pixel_values = np.sum(kidney_image)
    return sum_pixel_values

def _build_time_axis() -> List[float]:
    """构建与时间-计数曲线对应的时间轴 (分钟)。"""
    first_minute = np.linspace(0, 1, 60)  # 前 60 帧
    remaining_minutes = [1, 2, 3, 4, 5]   # 1-5 分钟
    return list(first_minute) + remaining_minutes


def _compute_half_metrics(values: List[float], times: List[float]) -> Dict[str, Optional[float]]:
    """计算半排时间/半排率等指标。"""
    if not values or not times or len(values) != len(times):
        return {
            't_half': None,
            'half_rate': None,
            'peak_value': None,
            'target_value': None,
            'reached_half': False
        }

    peak_value = max(values)
    if peak_value <= 0:
        return {
            't_half': None,
            'half_rate': None,
            'peak_value': peak_value,
            'target_value': None,
            'reached_half': False
        }

    peak_index = values.index(peak_value)
    target_value = peak_value * 0.5
    t_half = None

    # 调试信息：打印峰值和目标值
    print(f"[半排计算] 峰值: {peak_value:.2f}, 目标值(50%): {target_value:.2f}, 峰值时间: {times[peak_index]:.2f} min")
    
    # 检查峰值后的最小值，用于判断是否有下降趋势
    if peak_index < len(values) - 1:
        post_peak_values = values[peak_index:]
        min_post_peak = min(post_peak_values)
        print(f"[半排计算] 峰值后最小值: {min_post_peak:.2f}, 是否可能达到50%: {min_post_peak <= target_value}")

    for idx in range(peak_index, len(values)):
        current_value = values[idx]
        if current_value <= target_value:
            if idx == peak_index:
                t_half = times[idx]
            else:
                prev_value = values[idx - 1]
                prev_time = times[idx - 1]
                curr_time = times[idx]
                if current_value == prev_value:
                    t_half = curr_time
                else:
                    # 线性插值计算精确半排时间
                    slope = (current_value - prev_value) / (curr_time - prev_time)
                    if slope == 0:
                        t_half = curr_time
                    else:
                        t_half = prev_time + (target_value - prev_value) / slope
            print(f"[半排计算] 找到半排时间: {t_half:.2f} min (在时间点 {times[idx]:.2f} 达到目标值)")
            break

    reached_half = t_half is not None
    if not reached_half:
        # 如果未达到50%，检查观察窗口内的最低值
        if peak_index < len(values) - 1:
            final_value = values[-1]
            final_time = times[-1]
            print(f"[半排计算] 警告: 在观察窗口内({final_time:.2f} min)未达到50%峰值，最终值: {final_value:.2f} (目标: {target_value:.2f})")
    
    half_rate = 0.5 / t_half if reached_half and t_half > 0 else None

    return {
        't_half': round(t_half, 2) if t_half is not None else None,
        'half_rate': round(half_rate, 4) if half_rate is not None else None,
        'peak_value': peak_value,
        'target_value': target_value,
        'reached_half': reached_half
    }


def _plot_half_curve(x: List[float],
                     left_y: List[float],
                     right_y: List[float],
                     left_metrics: Dict[str, Optional[float]],
                     right_metrics: Dict[str, Optional[float]],
                     output_path: str) -> None:
    """绘制带半排标记的曲线图。"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.plot(x, left_y, color='red', label='左肾', linewidth=2)
    plt.plot(x, right_y, color='green', label='右肾', linewidth=2)

    # 左肾标记
    left_peak_value = left_metrics.get('peak_value')
    left_target_value = left_metrics.get('target_value')
    left_reached = left_metrics.get('reached_half', False)
    
    if left_peak_value and left_target_value:
        # 标记峰值点
        peak_idx = left_y.index(max(left_y))
        plt.scatter(x[peak_idx], left_peak_value, color='red', s=100, marker='o', 
                   zorder=5, label='左肾峰值')
        plt.text(x[peak_idx], left_peak_value, f'峰值\n{left_peak_value:.0f}', 
                color='red', fontsize=8, ha='center', va='bottom')
        
        # 50%目标值水平线（始终显示）
        plt.axhline(left_target_value, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        plt.text(4.5, left_target_value, f'50%目标\n{left_target_value:.0f}', 
                color='red', fontsize=8, ha='right', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 如果达到50%，显示半排时间标记
        if left_reached and left_metrics.get('t_half'):
            plt.axvline(left_metrics['t_half'], color='red', linestyle='--', 
                       linewidth=2, label='左肾T½')
            plt.scatter(left_metrics['t_half'], left_target_value, color='red', 
                       s=150, marker='*', zorder=5)
            plt.text(left_metrics['t_half'], left_target_value,
                    f"T½={left_metrics['t_half']:.2f}min", color='red', fontsize=10, 
                    ha='left', va='bottom', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
        else:
            # 未达到50%，显示提示信息
            final_value = left_y[-1] if left_y else None
            if final_value:
                plt.text(2.5, left_target_value * 0.7, 
                        f'左肾未达到50%\n最终值: {final_value:.0f}', 
                        color='red', fontsize=9, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # 右肾标记
    right_peak_value = right_metrics.get('peak_value')
    right_target_value = right_metrics.get('target_value')
    right_reached = right_metrics.get('reached_half', False)
    
    if right_peak_value and right_target_value:
        # 标记峰值点
        peak_idx = right_y.index(max(right_y))
        plt.scatter(x[peak_idx], right_peak_value, color='green', s=100, marker='o', 
                   zorder=5, label='右肾峰值')
        plt.text(x[peak_idx], right_peak_value, f'峰值\n{right_peak_value:.0f}', 
                color='green', fontsize=8, ha='center', va='bottom')
        
        # 50%目标值水平线（始终显示）
        plt.axhline(right_target_value, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        plt.text(4.5, right_target_value, f'50%目标\n{right_target_value:.0f}', 
                color='green', fontsize=8, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 如果达到50%，显示半排时间标记
        if right_reached and right_metrics.get('t_half'):
            plt.axvline(right_metrics['t_half'], color='green', linestyle='--', 
                       linewidth=2, label='右肾T½')
            plt.scatter(right_metrics['t_half'], right_target_value, color='green', 
                       s=150, marker='*', zorder=5)
            plt.text(right_metrics['t_half'], right_target_value,
                    f"T½={right_metrics['t_half']:.2f}min", color='green', fontsize=10, 
                    ha='left', va='bottom', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        else:
            # 未达到50%，显示提示信息
            final_value = right_y[-1] if right_y else None
            if final_value:
                plt.text(2.5, right_target_value * 0.5, 
                        f'右肾未达到50%\n最终值: {final_value:.0f}', 
                        color='green', fontsize=9, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    plt.title('半排率/半排时间标记曲线', fontsize=16, weight='bold')
    plt.xlabel('时间 (分钟)', fontsize=12)
    plt.ylabel('计数/秒 (Counts/Second)', fontsize=12)
    plt.xlim(0, 5)
    max_y = max(max(left_y), max(right_y)) if left_y and right_y else 1
    plt.ylim(0, max_y * 1.1)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, facecolor='white', dpi=150)
    plt.close()


def graph(dicom_path,
          ROI_data,
          output_path: Optional[str] = None,
          half_output_path: Optional[str] = None) -> Tuple[float, float, float, float, str, Optional[Dict[str, Dict[str, Optional[float]]]], Optional[str]]:
    print(f"步骤6: 肾脏计数变化曲线...")
    # 读取原始dicom文件
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_images = dicom_data.pixel_array  # 获取所有帧的图像数据

    # 获取DICOM图像的实际尺寸
    dicom_shape = dicom_images[0].shape if len(dicom_images.shape) > 2 else dicom_images.shape
    print(f"DICOM图像尺寸: {dicom_shape}")

    ROI_data = np.squeeze(ROI_data)
    zoom_factors = (dicom_shape[0] / ROI_data.shape[0], dicom_shape[1] / ROI_data.shape[1])
    ROI_data = zoom(ROI_data, zoom_factors, order=0)

    segmentation_mask = ROI_data
    
    # 计算每一帧的左右肾像素值
    left_kidney_sums = []
    right_kidney_sums = []
    left_background_sums = []
    right_background_sums = []

    if len(dicom_images.shape) > 2:  # 多帧DICOM

        for i in range(dicom_images.shape[0]):
            dicom_image = dicom_images[i]
            left_kidney_sum = calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label=1) 
            right_kidney_sum = calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label=2) 
            left_background_sum = calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label=3) 
            right_background_sum = calculate_kidney_pixel_sum(dicom_image, segmentation_mask, kidney_label=4) 
            left_kidney_sums.append(left_kidney_sum)
            right_kidney_sums.append(right_kidney_sum)
            left_background_sums.append(left_background_sum)
            right_background_sums.append(right_background_sum)
        
        # 绘制图像
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 定义横坐标
        x = _build_time_axis()

        # 纵坐标（计数/秒）
        left_y = left_kidney_sums[:60] + [left_kidney_sums[60] / 60, left_kidney_sums[61] / 60, left_kidney_sums[62] / 60,
                                        left_kidney_sums[63] / 60, left_kidney_sums[64] / 60]
        right_y = right_kidney_sums[:60] + [right_kidney_sums[60] / 60, right_kidney_sums[61] / 60, right_kidney_sums[62] / 60,
                                        right_kidney_sums[63] / 60, right_kidney_sums[64] / 60]

        # 绘制图像
        plt.figure(figsize=(10, 6), facecolor='lavender')  # 设置背景颜色
        plt.plot(x, left_y, color='red', label='左肾')  # 左肾曲线
        plt.plot(x, right_y, color='green', label='右肾')  # 右肾曲线

        # 设置横纵坐标范围和刻度
        plt.xlim(0, 5)  # 横坐标范围
        plt.ylim(0, max(max(left_y), max(right_y)) * 1.1)  # 纵坐标范围，根据最大值动态调整
        plt.xticks(np.arange(0, 5.5, 0.5))  # 横坐标刻度：0, 0.5, 1, ..., 5
        plt.yticks(np.arange(0, max(max(left_y), max(right_y)) * 1.1, 20))  # 纵坐标刻度，每20为一档

        # 返回第62帧（勾画的那一帧）的数据
        kidney_index = 61  # 通常是第62帧（索引从0开始，所以是61）

        # 计算半排指标并绘制半排曲线
        left_metrics = _compute_half_metrics(left_y, x)
        right_metrics = _compute_half_metrics(right_y, x)
        half_metrics = {'left': left_metrics, 'right': right_metrics}

        half_curve_file = None
        if half_output_path is None:
            base_name = os.path.basename(dicom_path)
            half_curve_file = os.path.join('ROI', f"half_curve_{base_name}.png")
            print(f"[警告] 未提供半排曲线输出路径，使用默认相对路径: {half_curve_file}")
        else:
            half_curve_file = half_output_path

        _plot_half_curve(x, left_y, right_y, left_metrics, right_metrics, half_curve_file)

    else:  # 单帧DICOM
        print("检测到单帧DICOM，计算静态肾脏计数...")
        # 计算单帧图像的肾脏区域像素值总和
        left_kidney_sum = calculate_kidney_pixel_sum(dicom_images, segmentation_mask, kidney_label=1)
        right_kidney_sum = calculate_kidney_pixel_sum(dicom_images, segmentation_mask, kidney_label=2)
        left_background_sum = calculate_kidney_pixel_sum(dicom_images, segmentation_mask, kidney_label=3)
        right_background_sum = calculate_kidney_pixel_sum(dicom_images, segmentation_mask, kidney_label=4)
        
        left_kidney_sums = [left_kidney_sum]
        right_kidney_sums = [right_kidney_sum]
        left_background_sums = [left_background_sum]
        right_background_sums = [right_background_sum]
        
        # 创建一个简单的条形图来展示单帧的肾脏计数
        plt.figure(figsize=(8, 5), facecolor='lavender')
        bars = plt.bar(['左肾', '右肾'], [left_kidney_sum, right_kidney_sum], color=['red', 'green'])
        
        # 在条形上方显示计数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('肾脏区域像素值总和', fontsize=16)
        plt.ylabel('像素值总和', fontsize=12)
        
        # 对于单帧，返回第一个（唯一的）元素
        kidney_index = 0
        half_metrics = None
        half_curve_file = None

    # 添加标题和坐标标签
    if len(dicom_images.shape) > 2:
        plt.title('Kidney Time-Activity Curve', fontsize=16, color='black')
        plt.xlabel('Minutes', fontsize=12)
        plt.ylabel('Counts/Second', fontsize=12)
        plt.legend()

    if output_path is None:
        # 如果没有提供 output_path，生成一个默认的绝对路径 (基于调用者预期的结构)
        # 注意：在 DicomProcessor 中我们会传入绝对路径，这里的 None 分支只是作为保障
        base_name = os.path.basename(dicom_path)
        name, ext = os.path.splitext(base_name)
        # 这里的相对路径 'ROI' 可能导致问题，最好在调用者端保证路径是绝对的
        output_file = os.path.join('ROI', f"counts_time_{name}.png") 
        print(f"[警告] graph 函数未提供 output_path，使用默认相对路径: {output_file}")
    else:
        # 使用传入的绝对路径
        output_file = output_path

    plt.savefig(output_file, facecolor='lavender')  # 保存图像
    plt.close()  # 关闭图像以释放内存

# 返回第62帧（勾画的那一帧）的左右肾像素值和背景区域像素值
    #return left_kidney_sums[61], right_kidney_sums[61], left_background_sums[61], right_background_sums[61], output_file

# 返回肾脏区域和背景区域的像素值之和
    return (left_kidney_sums[kidney_index], right_kidney_sums[kidney_index], 
            left_background_sums[kidney_index], right_background_sums[kidney_index], 
            output_file, half_metrics, half_curve_file)

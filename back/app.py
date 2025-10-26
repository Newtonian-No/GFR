import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
import json
import traceback
import threading
import os
import platform 
from pathlib import Path
import sys

# ----------------------------------------------------------------------
# 导入 DicomProcessor 类
# 确保 local_dicom_process.py 文件位于此脚本的同一目录下
# ----------------------------------------------------------------------
try:
    from back.local_dicom_process import DicomProcessor
except ImportError:
    messagebox.showerror("导入错误", "无法导入 DicomProcessor 类。请确保 'local_dicom_process.py' 文件在当前目录中。")
    # 如果导入失败，仍然定义一个空的桩类，确保 GUI 框架能运行
    class DicomProcessor:
        def __init__(self, *args, **kwargs):
            messagebox.showwarning("警告", "DicomProcessor 类未找到，功能将无法执行。")
            self.kidney_depths = {}
            self.last_patient_info = {}
            self.last_kidney_counts = {}
            self.last_manufacturer = None
        def process_dynamic_study_dicom(self, path): return {'success': False, 'message': '处理器未初始化'}
        def process_depth_dicom(self, path): return {'success': False, 'message': '处理器未初始化'}
        def upload_depth_and_calculate_li(self, *args): return {'success': False, 'message': '处理器未初始化'}
        def calculate_gfr(self, *args): return {'success': False, 'message': '处理器未初始化'}
        def reset_state(self):
            self.kidney_depths = {}
            self.last_patient_info = {}
            self.last_kidney_counts = {}
            self.last_manufacturer = None
            return True

# ====================================================================
# Tkinter GUI 实现
# ====================================================================

class DicomGFRApp:
    def __init__(self, master):
        self.master = master
        master.title("DICOM GFR 计算工具")

        # 实例化核心处理器类
        self.processor = DicomProcessor()
        
        # --- 新增逻辑: 字体设置 ---
        default_font = ('Arial', 14)
        
        # 检查操作系统
        if platform.system() == 'Linux':
            self.app_font = ('newspaper', 14) 
        else:
            self.app_font = default_font
            # self.os_message(f"[系统提示] 当前系统: {platform.system()}，应用默认字体。")
        # ---------------------------

        # -------------------------------------------------------------
        # GUI 元素定义 (使用 self.app_font)
        # -------------------------------------------------------------
        
        # 结果显示区 (Scrolled Text for Output)
        tk.Label(master, text="操作日志与状态结果:", font=self.app_font).pack(pady=(10, 0))
        # 通常日志区会使用等宽字体，但这里根据您的需求使用 self.app_font
        self.output_text = scrolledtext.ScrolledText(master, width=90, height=18, font=self.app_font)
        self.output_text.pack(padx=10)
        # self.log_message(self.os_message)

        # 文件路径输入区
        self.path_frame = tk.LabelFrame(master, text="DICOM 文件/文件夹路径选择", padx=5, pady=5, font=self.app_font)
        self.path_frame.pack(padx=10, pady=5, fill="x")
        
        self.path_entry = tk.Entry(self.path_frame, width=50, font=self.app_font)
        self.path_entry.insert(0, "请点击 '选择' 按钮")
        self.path_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0, 5))
        
        tk.Button(self.path_frame, 
                  text="选择文件", 
                  command=self.select_file, 
                  font=self.app_font).pack(side=tk.LEFT, padx=(5, 2))
        
        tk.Button(self.path_frame, 
                  text="选择文件夹", 
                  command=self.select_folder, 
                  font=self.app_font).pack(side=tk.LEFT, padx=(2, 5))

        # 按钮区
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=10, padx=10)

        # 1. 肾动态按钮
        tk.Button(self.button_frame, text="1. 处理肾动态 (DCM)", 
                  command=lambda: self.run_threaded_task(self.handle_convert_dicom), font=self.app_font).pack(side=tk.LEFT, padx=5)
        # 2. CT 深度按钮
        tk.Button(self.button_frame, text="2. 处理 CT 深度 (DCM)", 
                  command=lambda: self.run_threaded_task(self.handle_convert_depth_dicom), font=self.app_font).pack(side=tk.LEFT, padx=5)
        # 3. 手动上传深度按钮
        tk.Button(self.button_frame, text="3. 手动上传深度", 
                  command=self.show_upload_depth_window, font=self.app_font).pack(side=tk.LEFT, padx=5)
        # 4. GFR 计算按钮
        tk.Button(self.button_frame, text="4. 计算 GFR", 
                  command=self.handle_calculate_gfr, font=self.app_font).pack(side=tk.LEFT, padx=5)
        
        # 状态区
        self.status_button_frame = tk.Frame(master)
        self.status_button_frame.pack(pady=(0, 10), padx=10)
        tk.Button(self.status_button_frame, text="查看当前处理器状态", command=self.display_status, font=self.app_font).pack(side=tk.LEFT, padx=5)
        tk.Button(self.status_button_frame, text="重置处理器状态", command=self.clear_globals, font=self.app_font).pack(side=tk.LEFT, padx=5)
        
        # 初始状态显示
        self.display_status()

    # -------------------------------------------------------------
    # 通用工具函数
    # -------------------------------------------------------------

    def log_message(self, message):
        """将消息记录到输出文本框"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END) # 自动滚动到末尾

    def select_file(self):
        """打开文件对话框选择 DICOM 文件"""
        filepath = filedialog.askopenfilename(
            title="选择 DICOM 文件",
            filetypes=[("DICOM Files", "*.dcm;*.dicom"), ("All Files", "*.*")]
        )
        if filepath:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, filepath)
            self.log_message(f"[系统提示] 已选择文件: {filepath}")

    def select_folder(self):
        """打开文件夹对话框选择 DICOM 文件夹 (用于 CT 序列)"""
        folderpath = filedialog.askdirectory(
            title="选择 DICOM 文件夹 (用于 CT 序列)"
        )
        if folderpath:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, folderpath)
            self.log_message(f"[系统提示] 已选择文件夹: {folderpath}")

    def run_threaded_task(self, target_func):
        """将耗时任务放入单独线程执行"""
        self.log_message(f"\n[系统提示] 开始执行任务: {target_func.__name__}...")
        self.master.config(cursor="watch") # 更改光标为等待状态
        thread = threading.Thread(target=target_func, daemon=True)
        thread.start()

    def after_thread_completion(self, result, task_name):
        """线程完成后在主线程中更新 UI"""
        self.master.config(cursor="") # 恢复光标
        
        if result.get('success'):
            self.log_message(f"[成功] {task_name} 完成。")
            if 'message' in result:
                self.log_message(f"  > 消息: {result['message']}")
            if 'patientInfo' in result:
                self.log_message("  > 患者信息已更新。")
            if 'kidneyCounts' in result:
                self.log_message("  > 肾脏计数已更新。")
            if 'kidneyDepth' in result:
                self.log_message("  > 肾脏深度信息已更新。")
        else:
            self.log_message(f"[失败] {task_name} 失败。")
            self.log_message(f"  > 错误信息: {result.get('message', '未知错误')}")
        
        # 无论成功失败，都刷新状态显示
        self.display_status()
    
    # -------------------------------------------------------------
    # 状态管理函数
    # -------------------------------------------------------------
    
    def display_status(self):
        """显示当前的处理器内部状态"""
        processor = self.processor
        status = "\n" + "="*50 + "\n"
        status += "         DicomProcessor 内部状态           \n"
        status += "="*50 + "\n"
        status += f"厂商信息 (last_manufacturer): {processor.last_manufacturer}\n"
        status += "\n[患者信息 last_patient_info]:\n"
        status += json.dumps(processor.last_patient_info, indent=4, ensure_ascii=False) + "\n"
        status += "\n[肾脏计数 last_kidney_counts]:\n"
        status += json.dumps(processor.last_kidney_counts, indent=4, ensure_ascii=False) + "\n"
        status += "\n[肾脏深度 kidney_depths]:\n"
        status += json.dumps(processor.kidney_depths, indent=4, ensure_ascii=False) + "\n"
        status += "-"*50 + "\n"
        self.log_message(status)

    def clear_globals(self):
        """重置处理器状态"""
        # DicomProcessor 类中没有 reset_state 方法，我们在此处直接重置或重新实例化
        # 如果 DicomProcessor 有 reset_state 方法，应该调用它。
        # 这里我们假设 DicomProcessor 是单例且可以通过重新实例化来重置（但这不是好的实践，应在类中添加 reset_state 方法）。
        # 暂且直接修改内部状态：
        try:
            self.processor.kidney_depths = {k: None for k in self.processor.kidney_depths}
            self.processor.last_patient_info.clear()
            self.processor.last_kidney_counts.clear()
            self.processor.last_manufacturer = None
            self.processor.last_patient_name = None
            self.log_message("\n[系统提示] DicomProcessor 内部状态已重置。")
            self.display_status()
        except Exception as e:
            self.log_message(f"[错误] 重置状态失败: {e}")

    # -------------------------------------------------------------
    # 核心操作函数 (在单独线程中执行)
    # -------------------------------------------------------------
    
    def handle_convert_dicom(self):
        """处理肾动态显像 DICOM 文件 (线程目标函数)"""
        dicom_path = self.path_entry.get().strip()
        if not dicom_path or dicom_path == "请点击 '选择文件' 按钮":
            result = {'success': False, 'message': '文件路径不能为空'}
        else:
            try:
                # 调用 DicomProcessor 的方法
                result = self.processor.process_dynamic_study_dicom(dicom_path)
            except Exception as e:
                result = {'success': False, 'message': f'处理时发生未捕获的异常: {e}'}
                traceback.print_exc()

        # 使用 after 在主线程中更新 UI
        self.master.after(0, self.after_thread_completion, result, "肾动态显像处理")

    def handle_convert_depth_dicom(self):
        """处理 CT 图像 DICOM 深度计算 (线程目标函数)"""
        dicom_path = self.path_entry.get().strip()
        if not dicom_path or dicom_path == "请点击 '选择文件' 按钮":
            result = {'success': False, 'message': '文件路径不能为空'}
        else:
            if os.path.isdir(dicom_path):
                # 如果是文件夹，调用 find_deepest_kidney_slice
                print(f"输入是文件夹，开始寻找最深切片...")
                try:
                    # 调用 DicomProcessor 的方法
                    result = self.processor.process_depth_dicomfile(dicom_path)
                except Exception as e:
                    result = {'success': False, 'message': f'处理时发生未捕获的异常: {e}'}
                    traceback.print_exc()
            else:
                print(f"输入是单个文件，开始处理深度 DICOM...")
                try:
                    # 调用 DicomProcessor 的方法
                    result = self.processor.process_depth_dicom(dicom_path)
                except Exception as e:
                    result = {'success': False, 'message': f'处理时发生未捕获的异常: {e}'}
                    traceback.print_exc()

        # 使用 after 在主线程中更新 UI
        self.master.after(0, self.after_thread_completion, result, "CT 深度计算")


    # -------------------------------------------------------------
    # 手动上传深度窗口和逻辑 (需要确保子窗口的控件也使用新字体)
    # -------------------------------------------------------------
    
    def show_upload_depth_window(self):
        """创建手动上传深度信息的子窗口"""
        
        if hasattr(self, 'upload_window') and self.upload_window.winfo_exists():
            self.upload_window.focus()
            return
            
        self.upload_window = tk.Toplevel(self.master)
        self.upload_window.title("手动上传肾脏深度及患者信息")
        
        # 定义输入字段和标签
        fields = [
            ("左肾深度 (mm) (可选):", "leftDepth", self.processor.kidney_depths.get('leftDepth')),
            ("右肾深度 (mm) (可选):", "rightDepth", self.processor.kidney_depths.get('rightDepth')),
            ("身高 (米, e.g. 1.75):", "height"),
            ("体重 (公斤, e.g. 70.0):", "weight"),
            ("年龄 (岁, e.g. 50):", "age"),
            ("性别 ('男'/'女'):", "sex")
        ]
        self.depth_entries = {}
        
        for i, (label_text, key, default_val) in enumerate(fields):
            row = tk.Frame(self.upload_window)
            # 使用 self.app_font 设置子窗口标签字体
            label = tk.Label(row, width=22, text=label_text, anchor='w', font=self.app_font)
            # 使用 self.app_font 设置子窗口输入框字体
            entry = tk.Entry(row, width=30, font=self.app_font)
            
            # ... (填充默认值逻辑保持不变)

            # ... (填充默认值逻辑保持不变)
            if key == "sex":
                entry.insert(0, self.processor.last_patient_info.get('sex', '男'))
            elif key == "height":
                h = self.processor.last_patient_info.get('height')
                if h:
                    entry.insert(0, str(round(float(h), 2)))
            elif key == "weight":
                w = self.processor.last_patient_info.get('weight')
                if w:
                    entry.insert(0, str(round(float(w), 2)))
            elif key == "age":
                a = self.processor.last_patient_info.get('age')
                if a:
                    entry.insert(0, str(a))
            elif default_val is not None:
                entry.insert(0, str(default_val))

            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            label.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.depth_entries[key] = entry
            
        # 使用 self.app_font 设置子窗口按钮字体
        tk.Button(self.upload_window, text="确认上传并计算李氏深度", 
                  command=self.handle_upload_depth, font=self.app_font).pack(pady=10)
        
    def handle_upload_depth(self):
        """处理手动上传深度和计算李氏深度"""
        self.log_message("\n--- 3. 手动上传肾脏深度 ---")

        try:
            left_depth_str = self.depth_entries['leftDepth'].get().strip()
            right_depth_str = self.depth_entries['rightDepth'].get().strip()
            height = self.depth_entries['height'].get().strip()
            weight = self.depth_entries['weight'].get().strip()
            age = self.depth_entries['age'].get().strip()
            sex = self.depth_entries['sex'].get().strip()
            
            # 数据转换和校验
            left_depth = float(left_depth_str) if left_depth_str else None
            right_depth = float(right_depth_str) if right_depth_str else None
            
            if not all([height, weight, age, sex]) or sex not in ['男', '女']:
                 messagebox.showerror("输入错误", "请检查身高、体重、年龄、性别是否都已输入且性别为 ('男'/'女')。")
                 return
            
            # 调用 DicomProcessor 的方法
            result = self.processor.upload_depth_and_calculate_li(
                left_depth=left_depth, right_depth=right_depth, 
                height_m=float(height), weight_kg=float(weight), age_y=int(age), sex_cn=sex
            )
            
            if result.get('success'):
                self.log_message("[成功] 手动上传深度和李氏深度计算完成。")
                self.log_message("李氏深度结果:\n" + json.dumps({'LiLeftKidneyDepth': result.get('LiLeftKidneyDepth'), 'LiRightKidneyDepth': result.get('LiRightKidneyDepth')}, indent=4, ensure_ascii=False))
                self.upload_window.destroy()
                self.display_status()
            else:
                messagebox.showerror("操作失败", result.get('message', "计算失败"))
                self.log_message(f"[失败] 手动上传深度失败: {result.get('message')}")
                
        except ValueError:
            messagebox.showerror("输入错误", "深度、身高、体重、年龄必须是有效的数字。")
        except Exception as e:
            self.log_message(f"[致命错误] 处理失败: {e}")
            traceback.print_exc()

    # -------------------------------------------------------------
    # GFR 计算函数
    # -------------------------------------------------------------

    def handle_calculate_gfr(self):
        """计算 GFR"""
        self.log_message("\n--- 4. 计算 GFR ---")

        try:
            # 调用 DicomProcessor 的方法
            result = self.processor.calculate_gfr()

            if result.get('success'):
                self.log_message("[成功] GFR 计算完成。")
                self.log_message("计算结果:\n" + json.dumps(result.get('gfr', {}), indent=4, ensure_ascii=False))
            else:
                messagebox.showerror("计算失败", result.get('message', "GFR 计算失败"))
                self.log_message(f"[失败] GFR 计算失败: {result.get('message')}")
        except Exception as e:
            self.log_message(f"[致命错误] 处理失败: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    # 创建 'output' 目录，防止 DicomProcessor 实例化时出错
    os.makedirs("output", exist_ok=True)
    
    root = tk.Tk()
    app = DicomGFRApp(root)
    root.mainloop()
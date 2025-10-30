import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog, ttk # 导入 ttk
import json
import traceback
import threading
import os
import platform 
from pathlib import Path
import sys
from PIL import Image, ImageTk
from typing import Dict, Any, Optional
# ----------------------------------------------------------------------
# 导入 DicomProcessor 类
# 确保 local_dicom_process.py 文件位于此脚本的同一目录下
# ----------------------------------------------------------------------
try:
    # 假设 'back.local_dicom_process' 位于正确的 Python 路径上
    from back.local_dicom_process import DicomProcessor
except ImportError:
    messagebox.showerror("导入错误", "无法导入 DicomProcessor 类。请确保 'local_dicom_process.py' 文件在正确路径中。")
    # 如果导入失败，仍然定义一个空的桩类，确保 GUI 框架能运行
    class DicomProcessor:
        def __init__(self, *args, **kwargs):
            messagebox.showwarning("警告", "DicomProcessor 类未找到，功能将无法执行。")
            self.kidney_depths = {}
            self.last_patient_info = {}
            self.last_kidney_counts = {}
            self.last_manufacturer = None
            self.last_patient_name = None # 保持与您原代码中的重置逻辑一致
        def process_dynamic_study_dicom(self, path): return {'success': False, 'message': '处理器未初始化'}
        def process_depth_dicom(self, path): return {'success': False, 'message': '处理器未初始化'}
        def process_depth_dicomfile(self, path): return {'success': False, 'message': '处理器未初始化'} # 确保文件夹处理方法存在
        def upload_depth_and_calculate_li(self, *args): return {'success': False, 'message': '处理器未初始化'}
        def calculate_gfr(self, *args): return {'success': False, 'message': '处理器未初始化'}
        def reset_state(self):
            self.kidney_depths = {}
            self.last_patient_info = {}
            self.last_kidney_counts = {}
            self.last_manufacturer = None
            self.last_patient_name = None
            return True

# ====================================================================
# Tkinter GUI 实现 (重写为基于 Frame 的切换结构)
# ====================================================================

class DicomGFRApp:
    def __init__(self, master):
        self.master = master
        master.title("DICOM GFR 计算工具")
        master.geometry("1000x1200") # 设置一个更宽敞的初始窗口大小

        # 实例化核心处理器类
        self.processor = DicomProcessor()
        
        # --- 字体设置 (使用 ttk 统一风格) ---
        default_font_name = 'Arial'
        
        if platform.system() == 'Linux':
            # 保持用户确认有效的 'newspaper' 字体
            self.app_font = ('newspaper', 12) 
        else:
            self.app_font = (default_font_name, 12)
            
        self.title_font = (self.app_font[0], 16, 'bold')
        self.button_font = (self.app_font[0], 12, 'bold')
        self.log_font = (self.app_font[0], 10)
        
        # 应用 ttk 样式
        self.style = ttk.Style()
        self.style.configure('.', font=self.app_font)
        self.style.configure('TButton', font=self.button_font, padding=10)
        self.style.configure('TLabel', font=self.app_font)
        self.style.configure('Accent.TButton', foreground='white', background='#0078D7') # 突出显示主要操作
        # ------------------------------------

        # --- 界面切换容器 ---
        self.main_container = ttk.Frame(master)
        self.main_container.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 0))
        
        self.frames = {}
        
        # --- 实例化所有功能界面 ---
        self.frames["MainMenu"] = MainMenuFrame(self.main_container, self)
        self.frames["Feature1"] = DynamicStudyFrame(self.main_container, self)
        self.frames["Feature2"] = CTDepthFrame(self.main_container, self)
        # Note: Feature3 (手动上传) 被移除，改为对话框
        self.frames["Feature4"] = GFRCalculationFrame(self.main_container, self)
        
        # 将所有 Frame 放到同一位置，初始只显示 MainMenu
        for name, frame in self.frames.items():
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # 底部日志和状态区 (保持在主窗口底部)
        self.create_log_status_area()

        # 显示主菜单
        self.show_frame("MainMenu")
        
        # 初始状态显示
        self.display_status()

    # --- 界面切换逻辑 ---
    def show_frame(self, page_name):
        """显示指定名称的 Frame"""
        frame = self.frames[page_name]
        frame.tkraise() # 提升到顶部显示
        self.log_message(f"\n[系统提示] 已切换到: {page_name}")
        
    # --- 统一的日志和状态区 ---
    def create_log_status_area(self):
        """创建通用的日志和状态显示区"""
        log_frame = ttk.LabelFrame(self.master, text="操作日志与状态结果:", padding=(10, 5))
        log_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.output_text = scrolledtext.ScrolledText(
            log_frame, width=90, height=12, font=self.log_font, wrap=tk.WORD
        )
        self.output_text.pack(fill="both", expand=True)
        
        # 添加查看和重置状态按钮
        status_button_frame = ttk.Frame(log_frame)
        status_button_frame.pack(fill="x", pady=5)
        
        ttk.Button(status_button_frame, text="查看当前处理器状态", 
                   command=self.display_status).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(status_button_frame, text="重置处理器状态", 
                   command=self.clear_globals).pack(side=tk.LEFT, padx=5, expand=True)


    # -------------------------------------------------------------
    # 通用工具函数 (从原类复制并保留)
    # -------------------------------------------------------------
    
    def log_message(self, message):
        """将消息记录到输出文本框"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END) # 自动滚动到末尾

    def run_threaded_task(self, target_func, *args, **kwargs):
        """将耗时任务放入单独线程执行"""
        self.log_message(f"\n[系统提示] 开始执行任务: {target_func.__name__}...")
        self.master.config(cursor="watch") # 更改光标为等待状态
        thread = threading.Thread(target=target_func, args=args, kwargs=kwargs, daemon=True)
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
            if 'leftDepth' in result or 'rightDepth' in result:
                self.log_message("  > 肾脏深度信息已更新。")

            # --- 动态显像结果更新 (针对 DynamicStudyFrame) ---
            if task_name == "肾动态显像处理" and result.get('kidneyCounts'):
                dynamic_frame = self.frames["Feature1"]
                # 检查是否是正确的类型以调用更新方法
                if isinstance(dynamic_frame, DynamicStudyFrame):
                    # 调用 DynamicStudyFrame 中新的更新方法
                    dynamic_frame.update_results(result) 
            # ------------------------------------
            # +++ CT 深度结果更新 (针对 CTDepthFrame) +++
            if task_name == "CT 深度计算" and (result.get('leftDepth') or result.get('rightDepth') or result.get('deepestSliceName') or result.get('originalPngPath')):
                ct_depth_frame = self.frames["Feature2"]
                if isinstance(ct_depth_frame, CTDepthFrame):
                    # 调用 CTDepthFrame 中新的更新方法
                    ct_depth_frame.update_results(result)
            # +++ ---------------------------------- +++
            # --- GFR 计算结果更新 ---
            if task_name == "GFR 计算" and result.get('gfr'):
                #  self.log_message("计算结果:\n" + json.dumps(result.get('gfr', {}), indent=4, ensure_ascii=False))
                pass
            # ------------------------
            
        else:
            self.log_message(f"[失败] {task_name} 失败。")
            self.log_message(f"  > 错误信息: {result.get('message', '未知错误')}")
        
        self.display_status()
    
    # -------------------------------------------------------------
    # 状态管理函数 (从原类复制并保留)
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
        try:
            # 调用 DicomProcessor 的 reset_state 方法
            self.processor.reset_state()
            self.log_message("\n[系统提示] DicomProcessor 内部状态已重置。")
            self.display_status()
        except Exception as e:
            self.log_message(f"[错误] 重置状态失败: {e}")



    # -------------------------------------------------------------
    # 3. 手动上传深度模态对话框
    # -------------------------------------------------------------
    
    def show_upload_depth_dialog(self):
        """创建手动上传深度信息的模态对话框"""
        
        # 检查是否已打开，防止重复创建
        if hasattr(self, 'upload_dialog') and self.upload_dialog.winfo_exists():
            self.upload_dialog.focus()
            return
            
        self.upload_dialog = tk.Toplevel(self.master)
        self.upload_dialog.title("手动上传肾脏深度及患者信息")
        self.upload_dialog.attributes('-topmost', True) # 保持在最前
        self.upload_dialog.grab_set() # 设置为模态（阻止操作主窗口）
        
        # 样式和字体
        input_frame = ttk.LabelFrame(self.upload_dialog, text="请输入以下信息", padding=10)
        input_frame.pack(padx=20, pady=20, fill='x')

        # 定义输入字段和标签
        fields = [
            ("左肾深度 (mm) (可选):", "leftDepth", self.processor.kidney_depths.get('leftDepth')),
            ("右肾深度 (mm) (可选):", "rightDepth", self.processor.kidney_depths.get('rightDepth')),
            ("身高 (米, e.g. 1.75):", "height", None),
            ("体重 (公斤, e.g. 70.0):", "weight", None),
            ("年龄 (岁, e.g. 50):", "age", None),
            ("性别 ('男'/'女'):", "sex", None)
        ]
        self.depth_entries = {} 
        
        for i, (label_text, key, default_val) in enumerate(fields):
            row = ttk.Frame(input_frame)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            
            ttk.Label(row, width=22, text=label_text, anchor='w').pack(side=tk.LEFT, padx=(5, 10))
            entry = ttk.Entry(row, width=30)
            
            # 填充默认值逻辑 (使用 processor 的当前状态)
            if key == "sex":
                entry.insert(0, self.processor.last_patient_info.get('sex', '男'))
            elif key == "height":
                h = self.processor.last_patient_info.get('height')
                if h is not None:
                    entry.insert(0, str(round(float(h), 2)))
            elif key == "weight":
                w = self.processor.last_patient_info.get('weight')
                if w is not None:
                    entry.insert(0, str(round(float(w), 2)))
            elif key == "age":
                a = self.processor.last_patient_info.get('age')
                if a is not None:
                    entry.insert(0, str(a))
            elif default_val is not None:
                entry.insert(0, str(default_val))
                
            entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X, padx=5)
            self.depth_entries[key] = entry
            
        # 确认按钮
        ttk.Button(self.upload_dialog, text="确认上传并计算李氏深度", 
                  command=self.handle_upload_depth,
                  style='Accent.TButton'
                  ).pack(pady=15, ipadx=20, ipady=5)
        
        self.upload_dialog.protocol("WM_DELETE_WINDOW", self.upload_dialog.destroy) 
        self.master.wait_window(self.upload_dialog) 

    def handle_upload_depth(self):
        """处理手动上传深度和计算李氏深度 (在主线程中执行)"""
        self.log_message("\n--- 手动上传肾脏深度 ---")
        
        try:
            # 使用 self.depth_entries 获取 Toplevel 窗口中的值
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
            
            result = self.processor.manual_upload_depth_and_calculate_li(
                left_depth=left_depth, right_depth=right_depth, 
                height_m=float(height), weight_kg=float(weight), age_y=int(age), sex_cn=sex
            )
            
            if result.get('success'):
                self.log_message("[成功] 手动上传深度和李氏深度计算完成。")
                self.log_message("李氏深度结果:\n" + json.dumps({'LiLeftKidneyDepth': result.get('LiLeftKidneyDepth'), 'LiRightKidneyDepth': result.get('LiRightKidneyDepth')}, indent=4, ensure_ascii=False))
                self.upload_dialog.destroy() # 成功后关闭对话框
                self.display_status()
            else:
                messagebox.showerror("操作失败", result.get('message', "计算失败"))
                self.log_message(f"[失败] 手动上传深度失败: {result.get('message')}")
                
        except ValueError:
            messagebox.showerror("输入错误", "深度、身高、体重、年龄必须是有效的数字。")
        except Exception as e:
            self.log_message(f"[致命错误] 处理失败: {e}")
            traceback.print_exc()

# ====================================================================
# 主菜单 Frame
# ====================================================================

class MainMenuFrame(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        
        # 标题
        ttk.Label(self, text="DICOM GFR 计算工具 - 主菜单", font=controller.title_font).pack(pady=40, padx=10)
        
        # 按钮容器
        button_container = ttk.Frame(self)
        button_container.pack(pady=20)
        
        # 按钮定义 
        buttons = [
            ("1. 处理肾动态 (DCM)", "Feature1"),
            ("2. 处理 CT 深度 (DCM)", "Feature2"),
            # 3. 按钮直接调用对话框
            ("3. 手动上传深度/患者信息", "DIALOG"), 
            ("4. 计算 GFR", "Feature4"),
        ]
        
        for text, target_frame in buttons:
            if target_frame == "DIALOG":
                 command_func = controller.show_upload_depth_dialog
            else:
                 command_func = lambda tf=target_frame: controller.show_frame(tf)
                 
            ttk.Button(button_container, 
                       text=text, 
                       command=command_func, 
                       width=30).pack(pady=15, padx=20, fill='x')

# ====================================================================
# 功能子界面 Frame 基类
# ====================================================================

class FeatureFrameBase(ttk.Frame):
    """所有功能子界面的基类"""
    def __init__(self, parent, controller, title):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.processor = controller.processor

        # 顶部框架，用于放置返回按钮和标题
        top_bar = ttk.Frame(self)
        top_bar.pack(fill='x', pady=10, padx=10)
        
        ttk.Button(top_bar, text="< 返回主菜单", 
                   command=lambda: controller.show_frame("MainMenu"), 
                   width=15).pack(side=tk.LEFT)
                   
        ttk.Label(top_bar, text=title, font=controller.title_font).pack(side=tk.LEFT, padx=20)
        
        # 放置核心内容的容器
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 路径选择入口
        self.path_entry = None

        self.create_widgets()
        
    def create_widgets(self):
        """子类必须实现此方法来创建特定的控件"""
        pass # 占位，由子类重写

    def create_path_selection(self, container, is_folder_only=False):
        """创建统一的路径选择控件"""
        
        path_frame = ttk.LabelFrame(container, text="DICOM 文件/文件夹路径选择", padding=(10, 5))
        path_frame.pack(padx=20, pady=10, fill="x")
        
        self.path_entry = ttk.Entry(path_frame, width=60)
        self.path_entry.insert(0, "请点击 '选择' 按钮")
        self.path_entry.pack(side=tk.LEFT, expand=True, fill="x", padx=(0, 5))
        
        if not is_folder_only:
             ttk.Button(path_frame, text="选择文件", 
                        command=self.select_file).pack(side=tk.LEFT, padx=(5, 2))
        
        ttk.Button(path_frame, text="选择文件夹", 
                   command=self.select_folder).pack(side=tk.LEFT, padx=(2, 5))
                   
        return path_frame

    def select_file(self):
        """打开文件对话框选择 DICOM 文件"""
        filepath = filedialog.askopenfilename(
            title="选择 DICOM 文件",
            # filetypes=[("DICOM Files", "*.dcm;*.dicom"), ("All Files", "*.*")]
        )
        if filepath and self.path_entry:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, filepath)
            self.controller.log_message(f"[系统提示] 已选择文件: {filepath}")

    def select_folder(self):
        """打开文件夹对话框选择 DICOM 文件夹"""
        folderpath = filedialog.askdirectory(
            title="选择 DICOM 文件夹"
        )
        if folderpath and self.path_entry:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, folderpath)
            self.controller.log_message(f"[系统提示] 已选择文件夹: {folderpath}")
            
# -------------------------------------------------------------
# 1. 肾动态显像处理界面
# -------------------------------------------------------------
class DynamicStudyFrame(FeatureFrameBase):
    def __init__(self, parent, controller):
        self.image_labels = {} 
        self.count_labels = {} 
        self.image_tk_references = {} 
        self.image_display_size = (300, 300)
        super().__init__(parent, controller, "1. 处理肾动态显像 DICOM")

    def create_widgets(self):
        
        # 路径选择
        self.create_path_selection(self.content_frame)

        # 功能按钮区
        action_frame = ttk.Frame(self.content_frame)
        action_frame.pack(pady=30)
        
        # 主要功能按钮
        ttk.Button(action_frame, 
                   text="开始处理肾动态 (DCM)", 
                   command=lambda: self.controller.run_threaded_task(self.handle_convert_dicom),
                   style='Accent.TButton'
                   ).pack(side=tk.LEFT, padx=10, ipadx=20, ipady=10)
                   
        # 辅助功能按钮 (调用对话框)
        ttk.Button(action_frame, 
                   text="手动上传深度/患者信息", 
                   command=self.controller.show_upload_depth_dialog,
                   ).pack(side=tk.LEFT, padx=10, ipadx=10, ipady=10)

        # --- 结果显示区 ---
        result_frame = ttk.Frame(self.content_frame)
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 顶部两栏
        top_row = ttk.Frame(result_frame)
        top_row.pack(fill='both', expand=True) # 使用 expand=True 使图像区域能拉伸
        
        # 底部两栏
        bottom_row = ttk.Frame(result_frame)
        bottom_row.pack(fill='both', expand=True)

        # 1. 原始肾动态显像 (左上)
        self.original_image_frame = self._create_result_panel(top_row, "原始肾动态显像", "original")
        self.original_image_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)

        # 2. 肾脏计数变化曲线 (右上)
        self.curve_image_frame = self._create_result_panel(top_row, "肾脏计数变化曲线", "curve")
        self.curve_image_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)

        # 3. ROI 勾画结果 (左下)
        self.roi_image_frame = self._create_result_panel(bottom_row, "ROI 勾画结果", "roi")
        self.roi_image_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)
        
        # 4. 肾脏放射性计数表 (右下)
        self.count_table_frame = self._create_count_table_panel(bottom_row)
        self.count_table_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)

    # 核心处理逻辑
    def handle_convert_dicom(self):
        """处理肾动态显像 DICOM 文件 (线程目标函数)"""
        dicom_path = self.path_entry.get().strip()
        if not dicom_path or dicom_path == "请点击 '选择' 按钮":
            result = {'success': False, 'message': '文件路径不能为空'}
        else:
            try:
                result = self.processor.process_dynamic_study_dicom(dicom_path)
            except Exception as e:
                result = {'success': False, 'message': f'处理时发生未捕获的异常: {e}'}
                traceback.print_exc()

        self.controller.master.after(0, self.controller.after_thread_completion, result, "肾动态显像处理")
        
    def _create_result_panel(self, parent, title, key):
        """创建用于显示图像的LabelFrame容器"""
        frame = ttk.LabelFrame(parent, text=title, padding=5) 
        
        # LabelFrame 内部使用 Grid 布局，并让其可扩展
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        # 图像显示区域 (Label) - 移除 width/height 属性
        image_label = ttk.Label(frame, text="暂无图像\n(等待加载...)", 
                                                anchor="center", 
                                                justify=tk.CENTER,
                                                background='#E0E0E0', 
                                                relief=tk.SUNKEN)
        
        # 使用 grid 布局，并设置为 sticky='nsew' 填满 LabelFrame
        image_label.grid(row=0, column=0, sticky='nsew', padx=5, pady=5) 
        
        self.image_labels[key] = image_label
        return frame

    def _create_count_table_panel(self, parent):
        """创建用于显示计数表的LabelFrame容器"""
        frame = ttk.LabelFrame(parent, text="肾脏放射性计数表(counts)", padding=10)
        
        # 使用 Grid 布局创建表头
        ttk.Label(frame, text="指标", font=self.controller.button_font).grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        ttk.Label(frame, text="左肾", font=self.controller.button_font).grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        ttk.Label(frame, text="右肾", font=self.controller.button_font).grid(row=0, column=2, padx=10, pady=5, sticky='ew')
        
        # 第一行：肾脏放射性计数
        ttk.Label(frame, text="肾脏放射性计数").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.count_labels['leftKidneyCount'] = ttk.Label(frame, text="N/A")
        self.count_labels['leftKidneyCount'].grid(row=1, column=1, padx=10, pady=5, sticky='e')
        self.count_labels['rightKidneyCount'] = ttk.Label(frame, text="N/A")
        self.count_labels['rightKidneyCount'].grid(row=1, column=2, padx=10, pady=5, sticky='e')

        # 第二行：背景放射性计数
        ttk.Label(frame, text="背景放射性计数").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.count_labels['leftBackgroundCount'] = ttk.Label(frame, text="N/A")
        self.count_labels['leftBackgroundCount'].grid(row=2, column=1, padx=10, pady=5, sticky='e')
        self.count_labels['rightBackgroundCount'] = ttk.Label(frame, text="N/A")
        self.count_labels['rightBackgroundCount'].grid(row=2, column=2, padx=10, pady=5, sticky='e')
        
        # 确保列可扩展
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)
        
        return frame
        
    def update_results(self, result: Dict[str, Any]):
        """在处理完成后，在主线程中更新图像占位符和计数表"""
        
        # --- 1. 更新图像占位符文本为文件路径 ---
        image_map = {
            'original': result.get('imageUrl'),
            'roi': result.get('overlayUrl'),
            'curve': result.get('countsTimeUrl')
        }
        
        # 关键修改：调用 _load_and_display_image 方法来加载和显示图片
        self._load_and_display_image(self.image_labels['original'], image_map['original'], 'original')
        self._load_and_display_image(self.image_labels['roi'], image_map['roi'], 'roi')
        self._load_and_display_image(self.image_labels['curve'], image_map['curve'], 'curve')
        
        # --- 2. 更新计数表 ---
        counts = result.get('kidneyCounts', {})
        for key, label in self.count_labels.items():
            value = counts.get(key, "N/A")
            # 格式化显示，例如保留两位小数
            if isinstance(value, (int, float)):
                display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            else:
                display_value = str(value)

            label.config(text=display_value)
        
        self.controller.log_message("[系统提示] 图像路径和计数表已更新。")

    def _load_and_display_image(self, label: ttk.Label, image_path: Optional[str], key: str):
        """
        加载图像文件，调整大小，并在指定的 Label 控件中显示。
        依赖于 PIL/Pillow 库。
        
        参数:
            label (ttk.Label): 目标 Tkinter Label 控件。
            image_path (Optional[str]): 图像文件的绝对路径。
            key (str): 用于在 self.image_tk_references 中存储引用的键名。
        """
        # 1. 检查文件是否存在
        if not image_path or not os.path.exists(image_path):
            label.config(image='', text="暂无图像\n(文件未找到)", background='#F2DEDE')
            # 移除旧的引用，释放内存
            self.image_tk_references.pop(key, None) 
            return

        try:
            # 1. 打开图像
            img = Image.open(image_path)
            
            # 2. 调整图像尺寸以适应显示区域
            
            # 确保获取最新的尺寸
            label.update_idletasks() 
            # 获取 Label 控件当前的实际大小 (至少200x200作为最小尺寸保障)
            width = max(label.winfo_width(), 200) 
            height = max(label.winfo_height(), 200)
            
            # 使用 Image.resize 确保填满 Label，或者 Image.thumbnail 保持比例
            # 推荐使用 thumbnail 保持比例，避免拉伸变形。
            # 如果希望图片完全填充控件，牺牲长宽比，则使用 resize
            
            # 方案 1 (推荐): 保持比例缩放，并居中显示
            img_w, img_h = img.size
            
            # 计算缩放因子，使图片能适应 Label 尺寸
            ratio_w = width / img_w
            ratio_h = height / img_h
            
            # 取较小的比例因子，确保图片完全可见
            scale_factor = min(ratio_w, ratio_h) 
            
            new_w = int(img_w * scale_factor)
            new_h = int(img_h * scale_factor)

            # 缩放图片
            img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # 3. 创建 Tkinter PhotoImage 对象
            tk_img = ImageTk.PhotoImage(img)
            
            # 4. 更新 Label 控件
            # 确保 Label 对齐方式正确 (anchor='center' 通常就够了)
            label.config(image=tk_img, text='', anchor='center')
            
            # 6. 存储引用以防止垃圾回收
            self.image_tk_references[key] = tk_img
            label.config(background='white') # 成功加载后，背景色设为白色
            
        except Exception as e:
            # 7. 处理加载失败的情况
            self.controller.log_message(f"[图像错误] 无法加载 {image_path}: {e}")
            label.config(image='', text=f"图像加载失败:\n{Path(image_path).name}", background='#F2DEDE')

# -------------------------------------------------------------
# 2. CT 深度计算界面
# -------------------------------------------------------------
class CTDepthFrame(FeatureFrameBase):
    def __init__(self, parent, controller):
        self.image_labels = {}         # 必须先初始化，供 _create_result_panel 使用
        self.image_label = None        # 用于显示 CT 图像的 Label (引用 self.image_labels['ct_slice'])
        self.depth_labels = {}         # 用于显示深度的 Label
        self.image_tk_references = {}  # 存储 PhotoImage 引用
        self.image_display_size = (300, 300)
        super().__init__(parent, controller, "2. 处理 CT 深度计算")

    def _create_result_panel(self, parent, title, key):
        """创建用于显示图像的LabelFrame容器"""
        frame = ttk.LabelFrame(parent, text=title, padding=5) 
        
        # LabelFrame 内部使用 Grid 布局，并让其可扩展
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        # 图像显示区域 (Label) - 移除 width/height 属性
        image_label = ttk.Label(frame, text="暂无图像\n(等待加载...)", 
                                                anchor="center", 
                                                justify=tk.CENTER,
                                                background='#E0E0E0', 
                                                relief=tk.SUNKEN)
        
        # 使用 grid 布局，并设置为 sticky='nsew' 填满 LabelFrame
        image_label.grid(row=0, column=0, sticky='nsew', padx=5, pady=5) 
        
        self.image_labels[key] = image_label
        return frame

    def create_widgets(self):
        
        # 路径选择 
        self.create_path_selection(self.content_frame) # <-- 沿用父类的 path_entry

        ttk.Label(self.content_frame, text="提示: 请选择 CT 序列所在的文件夹，或者单个包含深度信息的 DICOM 文件。", 
                  font=self.controller.log_font).pack(pady=(0, 20))

        # 功能按钮区
        action_frame = ttk.Frame(self.content_frame)
        action_frame.pack(pady=30)

        # 主要功能按钮
        ttk.Button(action_frame, 
                   text="开始处理 CT 深度", 
                   command=lambda: self.controller.run_threaded_task(self.handle_convert_depth_dicom), # <--- 关键修改
                   style='Accent.TButton'
                   ).pack(side=tk.LEFT, padx=10, ipadx=20, ipady=10)
                   
        # 辅助功能按钮 (调用对话框)
        ttk.Button(action_frame, 
                   text="手动上传深度/患者信息", 
                   command=self.controller.show_upload_depth_dialog,
                   ).pack(side=tk.LEFT, padx=10, ipadx=10, ipady=10)

        # --- 结果显示区 (2x2 Grid) ---
        result_grid = ttk.Frame(self.content_frame)
        result_grid.pack(fill='both', expand=True, padx=5, pady=5) 
        
        # 配置 Grid 权重，让所有区域都可扩展
        result_grid.grid_rowconfigure(0, weight=1)
        result_grid.grid_rowconfigure(1, weight=1)
        result_grid.grid_columnconfigure(0, weight=1)
        result_grid.grid_columnconfigure(1, weight=1)

        # =================================================================
        # 1. 左上: 肾脏 CT/最深切片 (原图像)
        # =================================================================
        self.original_image_frame = self._create_result_panel(result_grid, "肾脏 CT/最深切片 (原图)", 'original')
        self.original_image_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5) 
        
        # =================================================================
        # 2. 右上: 叠加识别结果 (标注图像)
        # =================================================================
        # 假设处理器返回的标注图像键名为 'depthOverlayUrl' (需要 DicomProcessor 支持)
        self.overlay_image_frame = self._create_result_panel(result_grid, "深度识别叠加结果 (标注图)", 'overlay')
        self.overlay_image_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5) 

        # =================================================================
        # 3. 左下: 深度测量结果面板 (文本)
        # =================================================================
        self.count_table_frame = self._create_count_table_panel(result_grid)
        self.count_table_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5) 

        # =================================================================
        # 4. 右下: 空白区域（留待扩展）
        # =================================================================
        empty_frame = ttk.LabelFrame(result_grid, text="其他信息/扩展模块", padding="10")
        empty_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5) 
        ttk.Label(empty_frame, text="此区域暂时留空，可用于后续添加图像或数据展示。").pack(padx=10, pady=10)
        
        # 更新 self.image_label 的引用（现在它指向原图）
        self.image_label = self.image_labels.get('ct_original')

    def handle_convert_depth_dicom(self):
        """处理 CT 深度 DICOM 文件/文件夹 (线程目标函数)"""
        dicom_path = self.path_entry.get().strip()
        if not dicom_path or dicom_path == "请点击 '选择' 按钮":
            result = {'success': False, 'message': '文件路径不能为空'}
        else:
            try:
                result = self.processor.process_depth_and_li_depth(dicom_path)
            except Exception as e:
                result = {'success': False, 'message': f'处理时发生未捕获的异常: {e}'}
                traceback.print_exc()

        self.controller.master.after(0, self.controller.after_thread_completion, result, "CT 深度计算")        


    def update_results(self, result: Dict[str, Any]):
        """在处理完成后，在主线程中更新 CT 图像和深度表"""

        print(result)
        # --- 1. 更新图像占位符文本为文件路径 ---
        image_map = {
            'original': result.get('originalPngPath'),
            'overlay': result.get('overlayPngPath')
        }

        self._load_and_display_image(self.image_labels['original'], image_map['original'], 'original')
        self._load_and_display_image(self.image_labels['overlay'], image_map['overlay'], 'overlay')
        
        # 2. 更新深度表
        result_map = {
            'modelLeftDepth': '模型左肾深度',
            'modelRightDepth': '模型右肾深度',
            'LiLeftDepth': '李氏左肾深度',
            'LiRightDepth': '李氏右肾深度'
        }

        for key, label_widget in self.depth_labels.items():
            value = result.get(key, "N/A")

            # 格式化显示 (保留两位小数，单位 mm)
            if isinstance(value, (int, float)):
                # 确保格式化为两位小数
                display_value = f"{value:.2f} mm"
            else:
                # 如果是 "N/A" 或其他非数字值，直接显示
                display_value = str(value)

            label_widget.config(text=display_value)
        
        self.controller.log_message("[系统提示] CT 图像和深度表已更新。")

    def _load_and_display_image(self, label: ttk.Label, image_path: Optional[str], key: str):
        """
        加载图像文件，调整大小，并在指定的 Label 控件中显示。
        依赖于 PIL/Pillow 库。
        
        参数:
            label (ttk.Label): 目标 Tkinter Label 控件。
            image_path (Optional[str]): 图像文件的绝对路径。
            key (str): 用于在 self.image_tk_references 中存储引用的键名。
        """
        # 1. 检查文件是否存在
        if not image_path or not os.path.exists(image_path):
            label.config(image='', text="暂无图像\n(文件未找到)", background='#F2DEDE')
            # 移除旧的引用，释放内存
            self.image_tk_references.pop(key, None) 
            return

        try:
            # 1. 打开图像
            img = Image.open(image_path)
            
            # 2. 调整图像尺寸以适应显示区域
            
            # 确保获取最新的尺寸
            label.update_idletasks() 
            # 获取 Label 控件当前的实际大小 (至少200x200作为最小尺寸保障)
            width = max(label.winfo_width(), 200) 
            height = max(label.winfo_height(), 200)
            
            # 使用 Image.resize 确保填满 Label，或者 Image.thumbnail 保持比例
            # 推荐使用 thumbnail 保持比例，避免拉伸变形。
            # 如果希望图片完全填充控件，牺牲长宽比，则使用 resize
            
            # 方案 1 (推荐): 保持比例缩放，并居中显示
            img_w, img_h = img.size
            
            # 计算缩放因子，使图片能适应 Label 尺寸
            ratio_w = width / img_w
            ratio_h = height / img_h
            
            # 取较小的比例因子，确保图片完全可见
            scale_factor = min(ratio_w, ratio_h) 
            
            new_w = int(img_w * scale_factor)
            new_h = int(img_h * scale_factor)

            # 缩放图片
            img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # 3. 创建 Tkinter PhotoImage 对象
            tk_img = ImageTk.PhotoImage(img)
            
            # 4. 更新 Label 控件
            # 确保 Label 对齐方式正确 (anchor='center' 通常就够了)
            label.config(image=tk_img, text='', anchor='center')
            
            # 6. 存储引用以防止垃圾回收
            self.image_tk_references[key] = tk_img
            label.config(background='white') # 成功加载后，背景色设为白色
            
        except Exception as e:
            # 7. 处理加载失败的情况
            self.controller.log_message(f"[图像错误] 无法加载 {image_path}: {e}")
            label.config(image='', text=f"图像加载失败:\n{Path(image_path).name}", background='#F2DEDE')


    def _create_count_table_panel(self, parent):
        """创建用于显示计数表的LabelFrame容器"""
        frame = ttk.LabelFrame(parent, text="肾脏深度计算表", padding=10)
        
        # 使用 Grid 布局创建表头
        ttk.Label(frame, text="指标", font=self.controller.button_font).grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        ttk.Label(frame, text="左肾", font=self.controller.button_font).grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        ttk.Label(frame, text="右肾", font=self.controller.button_font).grid(row=0, column=2, padx=10, pady=5, sticky='ew')
        
        # 第一行：视觉肾脏深度计数
        ttk.Label(frame, text="肾脏深度计算").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.depth_labels['modelLeftDepth'] = ttk.Label(frame, text="N/A")
        self.depth_labels['modelLeftDepth'].grid(row=1, column=1, padx=10, pady=5, sticky='e')
        self.depth_labels['modelRightDepth'] = ttk.Label(frame, text="N/A")
        self.depth_labels['modelRightDepth'].grid(row=1, column=2, padx=10, pady=5, sticky='e')

        # 第二行：李氏计数
        ttk.Label(frame, text="李氏肾脏深度计算").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.depth_labels['LiLeftDepth'] = ttk.Label(frame, text="N/A") 
        self.depth_labels['LiLeftDepth'].grid(row=2, column=1, padx=10, pady=5, sticky='e')
        self.depth_labels['LiRightDepth'] = ttk.Label(frame, text="N/A") 
        self.depth_labels['LiRightDepth'].grid(row=2, column=2, padx=10, pady=5, sticky='e')
        
        # 确保列可扩展
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)
        
        return frame
# -------------------------------------------------------------
# 4. GFR 计算界面
# -------------------------------------------------------------
class GFRCalculationFrame(FeatureFrameBase):
    def __init__(self, parent, controller):
        self.gfr_labels = {}
        super().__init__(parent, controller, "4. 计算 GFR")

    def create_widgets(self):
        
        # 提示信息
        ttk.Label(self.content_frame, text="请确保肾动态数据、深度信息（通过CT或手动上传）已准备就绪。", 
                  font=self.controller.app_font).pack(pady=20)
                  
        # 功能按钮区
        action_frame = ttk.Frame(self.content_frame)
        action_frame.pack(pady=30)
                  
        # 主要功能按钮
        ttk.Button(action_frame, 
                   text="开始计算 GFR", 
                   command=lambda: self.controller.run_threaded_task(self.handle_calculate_gfr_threaded),
                   style='Accent.TButton'
                   ).pack(side=tk.LEFT, padx=10, ipadx=20, ipady=10)
                   
        # 辅助功能按钮 (调用对话框)
        ttk.Button(action_frame, 
                   text="手动上传深度/患者信息", 
                   command=self.controller.show_upload_depth_dialog,
                   ).pack(side=tk.LEFT, padx=10, ipadx=10, ipady=10)
        
        # 2. GFR 结果表格容器
        self.gfr_table_frame = self._create_gfr_table_panel(self.content_frame)
        self.gfr_table_frame.pack(pady=30, padx=20, fill='x')
                   
    # 核心处理逻辑 (在线程中执行)
    def handle_calculate_gfr_threaded(self):
        """实际进行 GFR 计算的线程目标函数"""
        self.controller.log_message("\n--- 4. 计算 GFR ---")

        try:
            result = self.processor.calculate_gfr()

            if result.get('success'):
                # 成功处理
                self.controller.master.after(0, self._gfr_completion_success, result)
            else:
                # 失败处理
                self.controller.master.after(0, self._gfr_completion_failure, result)
        except Exception as e:
            result = {'success': False, 'message': f'处理时发生未捕获的异常: {e}'}
            traceback.print_exc()
            self.controller.master.after(0, self._gfr_completion_failure, result)

    def _create_gfr_table_panel(self, parent):
        """创建用于显示 GFR 结果的 LabelFrame 容器"""
        frame = ttk.LabelFrame(parent, text="肾小球滤过率 (GFR) 计算结果 (ml/min)", padding=10)
        
        # 表格数据结构
        # 键: (行号, 描述, GFR-Key)
        data_rows = [
            (1, "基于模型深度 GFR", 'leftGFR', 'rightGFR', 'totalGFR'),
            (2, "基于李氏深度 GFR", 'LiLeftGFR', 'LiRightGFR', 'LiTotalGFR')
        ]
        
        # 表头
        ttk.Label(frame, text="指标", font=self.controller.button_font).grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        ttk.Label(frame, text="左肾 GFR", font=self.controller.button_font).grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        ttk.Label(frame, text="右肾 GFR", font=self.controller.button_font).grid(row=0, column=2, padx=10, pady=5, sticky='ew')
        ttk.Label(frame, text="总 GFR", font=self.controller.button_font, foreground='blue').grid(row=0, column=3, padx=10, pady=5, sticky='ew')
        
        # 填充数据行
        for row_num, desc, left_key, right_key, total_key in data_rows:
            ttk.Label(frame, text=desc).grid(row=row_num, column=0, padx=10, pady=5, sticky='w')
            
            # 左肾
            self.gfr_labels[left_key] = ttk.Label(frame, text="N/A")
            self.gfr_labels[left_key].grid(row=row_num, column=1, padx=10, pady=5, sticky='e')
            
            # 右肾
            self.gfr_labels[right_key] = ttk.Label(frame, text="N/A")
            self.gfr_labels[right_key].grid(row=row_num, column=2, padx=10, pady=5, sticky='e')
            
            # 总 GFR (加粗显示)
            self.gfr_labels[total_key] = ttk.Label(frame, text="N/A", font=self.controller.button_font, foreground='blue')
            self.gfr_labels[total_key].grid(row=row_num, column=3, padx=10, pady=5, sticky='e')

        # 确保列可扩展
        for c in range(1, 4):
            frame.grid_columnconfigure(c, weight=1)
            
        return frame
    
    def update_gfr_results(self, gfr_data: Dict[str, float]):
        """根据 GFR 结果更新表格内容"""
        
        for key, label_widget in self.gfr_labels.items():
            value = gfr_data.get(key)
            
            if isinstance(value, (int, float)):
                # 格式化为两位小数
                display_value = f"{value:.2f}"
            else:
                display_value = "N/A"
                
            label_widget.config(text=display_value)
            
        self.controller.log_message("[系统提示] GFR 计算结果表格已更新。")

    def _gfr_completion_success(self, result):
        """GFR 计算成功后在主线程中更新 UI"""
        self.controller.master.config(cursor="") 
        self.controller.log_message("[成功] GFR 计算完成。")
        self.update_gfr_results(result.get('gfr', {}))
        # self.controller.log_message("计算结果:\n" + json.dumps(result.get('gfr', {}), indent=4, ensure_ascii=False))
        self.controller.display_status()

    def _gfr_completion_failure(self, result):
        """GFR 计算失败后在主线程中更新 UI"""
        self.controller.master.config(cursor="")
        messagebox.showerror("计算失败", result.get('message', "GFR 计算失败"))
        self.controller.log_message(f"[失败] GFR 计算失败: {result.get('message')}")
        self.controller.display_status()


# ====================================================================
# 主程序入口
# ====================================================================

if __name__ == "__main__":
    # 创建 'output' 目录，防止 DicomProcessor 实例化时出错
    os.makedirs("output", exist_ok=True)
    
    root = tk.Tk()
    
    # 尝试设置 ttk 主题
    try:
        style = ttk.Style()
        # 尝试使用现代主题，例如 'clam' 或 'alt'
        style.theme_use('clam') 
    except:
        print("Failed to set ttk theme.")
        pass # 无法设置主题，使用默认样式
        
    app = DicomGFRApp(root)
    root.mainloop()
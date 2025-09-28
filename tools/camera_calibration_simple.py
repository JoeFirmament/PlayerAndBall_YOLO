#!/usr/bin/env python3
"""
精简版相机标定工具 - Camera Calibration Studio (Simplified)
只保留核心功能：相机标定、地面标定、基础相机控制
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import json
import threading
from datetime import datetime
import atexit

class SimpleCalibrationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Camera Calibration Studio (Simplified)")
        self.root.geometry("1200x700")

        # 基础数据结构
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_points = []
        self.object_points = []

        # Ground calibration数据
        self.ground_images_paths = []
        self.ground_homography_matrix = None
        self.ground_calibration_results = {}

        # 相机控制
        self.capture_cap = None
        self.preview_active = False

        # UI变量
        self.capture_path_var = tk.StringVar(value="/home/orangepi/Qworkspace/yolov8_pose_basketball/tools")
        self.capture_prefix_var = tk.StringVar(value="calib")
        self.board_w_var = tk.StringVar(value="9")
        self.board_h_var = tk.StringVar(value="6")
        self.square_size_var = tk.StringVar(value="25")

        # Ground calibration变量
        self.ground_folder_path_var = tk.StringVar()
        self.ground_board_w_var = tk.StringVar(value="7")
        self.ground_board_h_var = tk.StringVar(value="6")
        self.ground_square_size_var = tk.StringVar(value="50")

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.ground_progress_var = tk.DoubleVar()

        self.setup_ui()
        self.setup_bindings()

        atexit.register(self.cleanup_on_exit)

    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 创建选项卡
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)

        # 相机标定标签页
        calib_tab = tk.Frame(self.notebook)
        self.notebook.add(calib_tab, text='Camera Calibration')
        self.setup_calibration_tab(calib_tab)

        # 地面标定标签页
        ground_tab = tk.Frame(self.notebook)
        self.notebook.add(ground_tab, text='Ground Calibration')
        self.setup_ground_tab(ground_tab)

        # 相机控制标签页
        camera_tab = tk.Frame(self.notebook)
        self.notebook.add(camera_tab, text='Camera Control')
        self.setup_camera_tab(camera_tab)

        # 状态栏
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_calibration_tab(self, parent):
        """设置相机标定标签页"""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # 左侧：控制面板
        left_frame = tk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        # 图像文件夹选择
        folder_frame = tk.Frame(left_frame)
        folder_frame.pack(fill='x', pady=(0, 10))

        tk.Label(folder_frame, text="Calibration Images:").pack(anchor='w')
        folder_entry = tk.Entry(folder_frame, textvariable=self.capture_path_var)
        folder_entry.pack(fill='x', pady=(5, 0))

        tk.Button(folder_frame, text="Browse", command=self.select_folder).pack(anchor='e', pady=(5, 0))

        # 棋盘格参数
        param_frame = tk.Frame(left_frame)
        param_frame.pack(fill='x', pady=(10, 10))

        tk.Label(param_frame, text="Chessboard Parameters:").pack(anchor='w')

        # 宽度
        w_frame = tk.Frame(param_frame)
        w_frame.pack(fill='x', pady=(5, 0))
        tk.Label(w_frame, text="Width:").pack(side='left')
        tk.Entry(w_frame, textvariable=self.board_w_var, width=5).pack(side='right')

        # 高度
        h_frame = tk.Frame(param_frame)
        h_frame.pack(fill='x', pady=(5, 0))
        tk.Label(h_frame, text="Height:").pack(side='left')
        tk.Entry(h_frame, textvariable=self.board_h_var, width=5).pack(side='right')

        # 方格大小
        size_frame = tk.Frame(param_frame)
        size_frame.pack(fill='x', pady=(5, 0))
        tk.Label(size_frame, text="Square Size (mm):").pack(side='left')
        tk.Entry(size_frame, textvariable=self.square_size_var, width=5).pack(side='right')

        # 控制按钮
        button_frame = tk.Frame(left_frame)
        button_frame.pack(fill='x', pady=(20, 0))

        tk.Button(button_frame, text="Run Calibration", command=self.run_calibration,
                 bg='#4CAF50', fg='white', height=2).pack(fill='x', pady=(0, 10))

        tk.Button(button_frame, text="Save Results", command=self.save_results).pack(fill='x', pady=(0, 10))
        tk.Button(button_frame, text="Load Results", command=self.load_results).pack(fill='x')

        # 右侧：结果显示
        right_frame = tk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky='nsew')

        # 结果文本框
        result_frame = tk.Frame(right_frame)
        result_frame.pack(fill='both', expand=True)

        tk.Label(result_frame, text="Calibration Results:").pack(anchor='w')

        self.result_text = tk.Text(result_frame, height=20, wrap='word')
        scrollbar = tk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def setup_ground_tab(self, parent):
        """设置地面标定标签页"""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # 左侧：控制面板
        left_frame = tk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        # 文件夹选择
        folder_frame = tk.Frame(left_frame)
        folder_frame.pack(fill='x', pady=(0, 10))

        tk.Label(folder_frame, text="Ground Images:").pack(anchor='w')
        ground_entry = tk.Entry(folder_frame, textvariable=self.ground_folder_path_var)
        ground_entry.pack(fill='x', pady=(5, 0))

        tk.Button(folder_frame, text="Browse", command=self.select_ground_folder).pack(anchor='e', pady=(5, 0))

        # 棋盘格参数
        param_frame = tk.Frame(left_frame)
        param_frame.pack(fill='x', pady=(10, 10))

        tk.Label(param_frame, text="Chessboard Parameters:").pack(anchor='w')

        # 宽度
        w_frame = tk.Frame(param_frame)
        w_frame.pack(fill='x', pady=(5, 0))
        tk.Label(w_frame, text="Width:").pack(side='left')
        tk.Entry(w_frame, textvariable=self.ground_board_w_var, width=5).pack(side='right')

        # 高度
        h_frame = tk.Frame(param_frame)
        h_frame.pack(fill='x', pady=(5, 0))
        tk.Label(h_frame, text="Height:").pack(side='left')
        tk.Entry(h_frame, textvariable=self.ground_board_h_var, width=5).pack(side='right')

        # 方格大小
        size_frame = tk.Frame(param_frame)
        size_frame.pack(fill='x', pady=(5, 0))
        tk.Label(size_frame, text="Square Size (mm):").pack(side='left')
        tk.Entry(size_frame, textvariable=self.ground_square_size_var, width=5).pack(side='right')

        # 控制按钮
        button_frame = tk.Frame(left_frame)
        button_frame.pack(fill='x', pady=(20, 0))

        tk.Button(button_frame, text="Load Images", command=self.load_ground_images_from_folder,
                 bg='#2196F3', fg='white', height=2).pack(fill='x', pady=(0, 10))

        tk.Button(button_frame, text="Run Ground Calibration", command=self.start_ground_calibration,
                 bg='#4CAF50', fg='white', height=2).pack(fill='x', pady=(0, 10))

        tk.Button(button_frame, text="Save Ground Results", command=self.save_ground_results).pack(fill='x')

        # 进度条
        progress_frame = tk.Frame(left_frame)
        progress_frame.pack(fill='x', pady=(10, 0))

        self.ground_progress_bar = ttk.Progressbar(progress_frame, variable=self.ground_progress_var)
        self.ground_progress_bar.pack(fill='x')

        self.ground_progress_label = tk.Label(progress_frame, text="Ready")
        self.ground_progress_label.pack(anchor='w')

        # 右侧：结果显示
        right_frame = tk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky='nsew')

        result_frame = tk.Frame(right_frame)
        result_frame.pack(fill='both', expand=True)

        tk.Label(result_frame, text="Ground Calibration Results:").pack(anchor='w')

        self.ground_results_text = tk.Text(result_frame, height=20, wrap='word')
        scrollbar = tk.Scrollbar(result_frame, command=self.ground_results_text.yview)
        self.ground_results_text.configure(yscrollcommand=scrollbar.set)

        self.ground_results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def setup_camera_tab(self, parent):
        """设置相机控制标签页"""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # 左侧：控制面板
        left_frame = tk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        # 相机控制
        control_frame = tk.Frame(left_frame)
        control_frame.pack(fill='x', pady=(0, 10))

        tk.Button(control_frame, text="Connect Camera", command=self.connect_camera,
                 bg='#4CAF50', fg='white').pack(fill='x', pady=(0, 5))

        tk.Button(control_frame, text="Disconnect Camera", command=self.disconnect_camera).pack(fill='x', pady=(0, 5))

        tk.Button(control_frame, text="Start Preview", command=self.start_preview).pack(fill='x', pady=(0, 10))

        tk.Button(control_frame, text="Stop Preview", command=self.stop_preview).pack(fill='x', pady=(0, 5))

        tk.Button(control_frame, text="Take Photo", command=self.capture_single_image).pack(fill='x')

        # 保存设置
        save_frame = tk.Frame(left_frame)
        save_frame.pack(fill='x', pady=(20, 0))

        tk.Label(save_frame, text="Save Settings:").pack(anchor='w')

        tk.Label(save_frame, text="Path:").pack(anchor='w', pady=(5, 0))
        path_entry = tk.Entry(save_frame, textvariable=self.capture_path_var)
        path_entry.pack(fill='x', pady=(0, 5))

        tk.Label(save_frame, text="Prefix:").pack(anchor='w')
        prefix_entry = tk.Entry(save_frame, textvariable=self.capture_prefix_var)
        prefix_entry.pack(fill='x')

        # 右侧：预览区域
        right_frame = tk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky='nsew')

        preview_frame = tk.Frame(right_frame)
        preview_frame.pack(fill='both', expand=True)

        tk.Label(preview_frame, text="Camera Preview:").pack(anchor='w')

        # 创建画布用于显示预览
        self.preview_canvas = tk.Canvas(preview_frame, bg='black')
        self.preview_canvas.pack(fill='both', expand=True)

        self.preview_status_label = tk.Label(preview_frame, text="Camera not connected")
        self.preview_status_label.pack(anchor='w', pady=(5, 0))

    def setup_bindings(self):
        """设置键盘绑定"""
        # 空格键快速拍摄
        self.root.bind('<space>', lambda e: self.quick_capture())
        # 回车键拍摄
        self.root.bind('<Return>', lambda e: self.capture_single_image())

    # ==================== 相机标定功能 ====================

    def select_folder(self):
        """选择标定图像文件夹"""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.capture_path_var.set(folder_selected)

    def run_calibration(self):
        """运行相机标定"""
        try:
            # 获取参数
            board_w = int(self.board_w_var.get())
            board_h = int(self.board_h_var.get())
            square_size = float(self.square_size_var.get())

            # 查找图像
            folder_path = self.capture_path_var.get()
            if not os.path.exists(folder_path):
                messagebox.showerror("Error", f"Folder does not exist: {folder_path}")
                return

            # 查找jpg文件
            image_files = [f for f in os.listdir(folder_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                messagebox.showerror("Error", "No image files found in the selected folder")
                return

            # 处理图像
            objpoints = []  # 3D世界坐标
            imgpoints = []  # 2D图像坐标

            for filename in image_files:
                filepath = os.path.join(folder_path, filename)
                img = cv2.imread(filepath)
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 检测棋盘格角点
                ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

                if ret:
                    # 细化角点
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    # 准备对象点
                    objp = np.zeros((board_w * board_h, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * square_size

                    objpoints.append(objp)
                    imgpoints.append(corners_refined)

            if len(objpoints) == 0:
                messagebox.showerror("Error", "No chessboard corners found in any image")
                return

            # 标定相机
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            # 计算重投影误差
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                                    self.camera_matrix, self.dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
                total_error += error

            mean_error = total_error / len(objpoints)

            # 显示结果
            result_text = f"""Camera Calibration Results:

Calibration Summary:
• Total images processed: {len(image_files)}
• Successful detections: {len(objpoints)}
• Success rate: {(len(objpoints) / len(image_files) * 100):.1f}%

Calibration Parameters:
• Chessboard size: {board_w}×{board_h}
• Square size: {square_size} mm

Camera Matrix:
{self.camera_matrix}

Distortion Coefficients:
{self.dist_coeffs.flatten()}

Accuracy Metrics:
• Mean reprojection error: {mean_error:.3f} pixels
• Expected coordinate accuracy: ±{mean_error * 5:.1f}mm

Calibration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            self.result_text.delete('1.0', 'end')
            self.result_text.insert('1.0', result_text)

            messagebox.showinfo("Success", f"Camera calibration completed!\nMean error: {mean_error:.3f} pixels")

        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed: {e}")

    def save_results(self):
        """保存标定结果"""
        if self.camera_matrix is None:
            messagebox.showerror("Error", "No calibration results to save")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_calibration.json"

            data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                'calibration_date': datetime.now().isoformat(),
                'board_size': [int(self.board_w_var.get()), int(self.board_h_var.get())],
                'square_size': float(self.square_size_var.get())
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            messagebox.showinfo("Success", f"Results saved to: {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")

    def load_results(self):
        """加载标定结果"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if not filename:
                return

            with open(filename, 'r') as f:
                data = json.load(f)

            self.camera_matrix = np.array(data['camera_matrix'])
            self.dist_coeffs = np.array(data['dist_coeffs'])

            # 更新UI
            if 'board_size' in data:
                self.board_w_var.set(str(data['board_size'][0]))
                self.board_h_var.set(str(data['board_size'][1]))
            if 'square_size' in data:
                self.square_size_var.set(str(data['square_size']))

            messagebox.showinfo("Success", "Calibration results loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {e}")

    # ==================== 地面标定功能 ====================

    def select_ground_folder(self):
        """选择地面标定图像文件夹"""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.ground_folder_path_var.set(folder_selected)

    def load_ground_images_from_folder(self):
        """从文件夹加载地面标定图像"""
        folder_path = self.ground_folder_path_var.get()
        if not folder_path or not os.path.exists(folder_path):
            messagebox.showerror("Error", "Please select a valid folder")
            return

        # 查找图像文件
        image_files = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            messagebox.showerror("Error", "No image files found in the selected folder")
            return

        # 加载图像路径
        self.ground_images_paths = [os.path.join(folder_path, f) for f in image_files]

        messagebox.showinfo("Success", f"Loaded {len(self.ground_images_paths)} ground calibration images")

    def start_ground_calibration(self):
        """开始地面标定"""
        if not self.ground_images_paths:
            messagebox.showerror("Error", "Please load ground calibration images first")
            return

        # 在后台线程中运行
        threading.Thread(target=self.run_ground_calibration, daemon=True).start()

    def run_ground_calibration(self):
        """运行地面标定"""
        try:
            # 获取参数
            board_w = int(self.ground_board_w_var.get())
            board_h = int(self.ground_board_h_var.get())
            square_size = float(self.ground_square_size_var.get())

            # 更新进度
            self.root.after(0, lambda: self.ground_progress_var.set(0))
            self.root.after(0, lambda: self.ground_progress_label.config(text="Initializing..."))

            # 初始化3D坐标
            objp = np.zeros((board_w * board_h, 3), np.float32)
            objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * square_size

            objpoints = []  # 3D世界坐标
            imgpoints = []  # 2D图像坐标

            total_images = len(self.ground_images_paths)

            for i, image_path in enumerate(self.ground_images_paths, 1):
                # 更新进度
                progress = (i - 1) / total_images * 100
                self.root.after(0, lambda p=progress: self.ground_progress_var.set(p))
                self.root.after(0, lambda: self.ground_progress_label.config(
                    text=f"Processing image {i}/{total_images}"))

                img = cv2.imread(image_path)
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 使用用户输入的参数检测棋盘格
                ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

                if ret:
                    # 细化角点
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    objpoints.append(objp)
                    imgpoints.append(corners_refined)

            # 计算单应矩阵
            if len(objpoints) >= 1:
                # 使用第一个成功的图像
                src_points = objpoints[0][:, :2]  # 世界坐标 (只取x,y)
                dst_points = imgpoints[0][:, :2]  # 图像坐标

                # 计算单应矩阵
                H, status = cv2.findHomography(src_points, dst_points)

                if H is not None:
                    self.ground_homography_matrix = H

                    # 计算重投影误差
                    projected = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H)
                    projected = projected.reshape(-1, 2)
                    error = np.mean(np.linalg.norm(projected - dst_points, axis=1))

                    # 显示结果
                    result_text = f"""Ground Calibration Results:

Calibration Summary:
• Total images processed: {total_images}
• Successful detections: {len(objpoints)}
• Success rate: {(len(objpoints) / total_images * 100):.1f}%

Calibration Parameters:
• Chessboard size: {board_w}×{board_h}
• Square size: {square_size} mm

Homography Matrix:
{H}

Accuracy Metrics:
• Reprojection error: {error:.3f} pixels
• Expected coordinate accuracy: ±{error * 5:.1f}mm

Calibration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

                    self.root.after(0, lambda: self.ground_results_text.delete('1.0', 'end'))
                    self.root.after(0, lambda: self.ground_results_text.insert('1.0', result_text))

                    # 完成
                    self.root.after(0, lambda: self.ground_progress_var.set(100))
                    self.root.after(0, lambda: self.ground_progress_label.config(text="Ground calibration completed!"))

                    self.root.after(0, lambda: messagebox.showinfo("Success",
                        f"Ground calibration completed!\nReprojection error: {error:.3f} pixels"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to compute homography matrix"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "No chessboard corners detected"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Ground calibration failed: {e}"))

    def save_ground_results(self):
        """保存地面标定结果"""
        if self.ground_homography_matrix is None:
            messagebox.showerror("Error", "No ground calibration results to save")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_ground_calibration.json"

            data = {
                'homography_matrix': self.ground_homography_matrix.tolist(),
                'calibration_date': datetime.now().isoformat(),
                'board_size': [int(self.ground_board_w_var.get()), int(self.ground_board_h_var.get())],
                'square_size': float(self.ground_square_size_var.get()),
                'image_count': len(self.ground_images_paths)
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            messagebox.showinfo("Success", f"Ground calibration results saved to: {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save ground results: {e}")

    # ==================== 相机控制功能 ====================

    def connect_camera(self):
        """连接相机"""
        try:
            self.capture_cap = cv2.VideoCapture(0)
            if self.capture_cap.isOpened():
                self.status_bar.config(text="Camera connected successfully")
                messagebox.showinfo("Success", "Camera connected successfully")
            else:
                self.capture_cap = None
                messagebox.showerror("Error", "Failed to connect to camera")
        except Exception as e:
            messagebox.showerror("Error", f"Camera connection failed: {e}")

    def disconnect_camera(self):
        """断开相机连接"""
        if hasattr(self, 'capture_cap') and self.capture_cap is not None:
            if self.preview_active:
                self.stop_preview()
            self.capture_cap.release()
            self.capture_cap = None
            self.status_bar.config(text="Camera disconnected")

    def start_preview(self):
        """开始预览"""
        if not hasattr(self, 'capture_cap') or self.capture_cap is None or not self.capture_cap.isOpened():
            messagebox.showerror("Error", "Camera not connected")
            return

        self.preview_active = True
        self.preview_status_label.config(text="Preview active")
        self.status_bar.config(text="Camera preview started")

        # 启动预览线程
        threading.Thread(target=self.update_preview, daemon=True).start()

    def stop_preview(self):
        """停止预览"""
        self.preview_active = False
        self.preview_status_label.config(text="Preview stopped")
        self.status_bar.config(text="Camera preview stopped")

    def update_preview(self):
        """更新预览"""
        while self.preview_active and hasattr(self, 'capture_cap') and self.capture_cap.isOpened():
            ret, frame = self.capture_cap.read()
            if ret:
                # 转换为RGB格式用于显示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 在Tkinter中显示
                self.root.after(0, lambda f=frame_rgb: self.display_preview_frame(f))

            # 短暂延迟
            cv2.waitKey(30)

    def display_preview_frame(self, frame):
        """在画布上显示预览帧"""
        try:
            # 获取画布尺寸
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # 调整图像大小以适应画布
                frame_height, frame_width = frame.shape[:2]
                scale = min(canvas_width / frame_width, canvas_height / frame_height)

                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)

                frame_resized = cv2.resize(frame, (new_width, new_height))

                # 转换为PhotoImage
                from PIL import Image, ImageTk
                img = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image=img)

                # 在画布上显示
                self.preview_canvas.create_image(
                    canvas_width // 2, canvas_height // 2,
                    image=photo, anchor=tk.CENTER
                )

                # 保持引用以防止垃圾回收
                self.preview_canvas.photo = photo

        except Exception as e:
            print(f"Preview display error: {e}")

    def capture_single_image(self):
        """拍摄单张图像"""
        if not hasattr(self, 'capture_cap') or self.capture_cap is None or not self.capture_cap.isOpened():
            messagebox.showerror("Error", "Camera not connected")
            return

        try:
            ret, frame = self.capture_cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image")
                return

            # 保存图像
            save_path = self.capture_path_var.get()
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            prefix = self.capture_prefix_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(save_path, filename)

            cv2.imwrite(filepath, frame)

            self.status_bar.config(text=f"Image saved: {filename}")
            messagebox.showinfo("Success", f"Image saved: {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

    def quick_capture(self):
        """快速拍摄"""
        self.capture_single_image()

    def cleanup_on_exit(self):
        """退出时清理"""
        if hasattr(self, 'capture_cap') and self.capture_cap is not None:
            self.capture_cap.release()

    def run(self):
        """运行应用程序"""
        self.root.mainloop()

def main():
    app = SimpleCalibrationGUI()
    app.run()

if __name__ == "__main__":
    main()



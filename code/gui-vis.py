#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单球体查看器
使用 omegapy 加载 pkl 数据，在 tkinter 中使用 matplotlib 3D 显示 3D 球体
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import cubeio as cio

import omegapy.omega_data as od
OMEGAPY_AVAILABLE = True


class SpectrumWindow:
    """光谱显示窗口，管理0.8-2.4um波段反射率折线图的显示"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.window = None  # Toplevel 窗口
        self.fig = None     # matplotlib Figure
        self.ax = None      # matplotlib Axes
        self.canvas = None  # FigureCanvasTkAgg
        self.lines = []     # 存储多个光谱线 (line, label)
        self.legend = None  # 图例
        self.spectrum_count = 0  # 光谱计数器
        self.max_spectra = 10     # 最大光谱线数量
        
    def create_window(self):
        """创建或显示光谱窗口"""
        if self.window is None or not tk.Toplevel.winfo_exists(self.window):
            # 创建新窗口
            self.window = tk.Toplevel(self.parent)
            self.window.title("Spectrum Viewer (0.8-2.4 μm)")
            self.window.geometry("800x600")
            
            # 创建matplotlib画布
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("Reflectance Spectrum (0.8-2.4 μm)")
            self.ax.set_xlabel("Wavelength (μm)")
            self.ax.set_ylabel("Reflectance")
            self.ax.grid(True, alpha=0.3)
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 添加控制按钮
            control_frame = tk.Frame(self.window)
            control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
            
            clear_btn = tk.Button(control_frame, text="Clear All", command=self.clear_spectra)
            clear_btn.pack(side=tk.LEFT, padx=5)
            
            save_btn = tk.Button(control_frame, text="Save Figure", command=self.save_figure)
            save_btn.pack(side=tk.LEFT, padx=5)
            
            # 状态标签显示当前光谱数量
            self.status_label = tk.Label(control_frame, text="Spectra: 0/3")
            self.status_label.pack(side=tk.RIGHT, padx=5)
        else:
            # 窗口已存在，将其提到前面
            self.window.lift()
            self.window.focus_force()
    
    def add_spectrum(self, wavelengths, reflectance, label=None):
        """添加一条光谱线到窗口
        
        Args:
            wavelengths: 波长数组 (μm)
            reflectance: 反射率数组
            label: 光谱线标签，如果为None则自动生成
        """
        # 筛选0.8-2.4um波段（只显示存在的部分）
        mask = (wavelengths >= 0.8) & (wavelengths <= 2.4)
        if not mask.any():
            print("警告：没有0.8-2.4um范围内的数据")
            return
        
        # 确保窗口已创建
        self.create_window()
        
        # 如果超过最大数量，移除最早的光谱线
        if len(self.lines) >= self.max_spectra:
            oldest_line, oldest_label = self.lines.pop(0)
            oldest_line.remove()
        
        # 绘制新光谱线
        filtered_wl = wavelengths[mask]
        filtered_rf = reflectance[mask]
        # filtered_wl, filtered_rf = wavelengths, reflectance
        
        # 生成标签：编号 + 经纬度 + 文件名
        self.spectrum_count += 1
        if label is None:
            label = f"#{self.spectrum_count}"
        
        # 绘制光谱线
        line, = self.ax.plot(filtered_wl, filtered_rf, marker='o', markersize=3,
                            linewidth=1.5, alpha=0.8)
        self.lines.append((line, label))
        
        # 更新图例
        self.update_legend()
        
        # 更新状态标签
        self.status_label.config(text=f"Spectra: {len(self.lines)}/{self.max_spectra}")
        
        # 更新画布
        self.canvas.draw()
    
    def update_legend(self):
        """更新图例"""
        if self.legend is not None:
            self.legend.remove()
        
        if self.lines:
            lines = [line for line, _ in self.lines]
            labels = [label if label else f"Spectrum {i+1}" for i, (_, label) in enumerate(self.lines)]
            self.legend = self.ax.legend(lines, labels, loc='upper right', fontsize=9)
    
    def clear_spectra(self):
        """清除所有光谱线"""
        for line, _ in self.lines:
            line.remove()
        self.lines.clear()
        self.spectrum_count = 0
        
        if self.legend is not None:
            self.legend.remove()
            self.legend = None
        
        # 更新状态标签
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Spectra: 0/{self.max_spectra}")
        
        # 更新画布
        if self.canvas is not None:
            self.canvas.draw()
    
    def save_figure(self):
        """保存图形到文件"""
        if self.fig is not None:
            filepath = filedialog.asksaveasfilename(
                title="Save Spectrum Figure",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if filepath:
                try:
                    self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"光谱图已保存到: {filepath}")
                except Exception as e:
                    print(f"保存失败: {e}")

class SimpleSphereViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Sphere Viewer")
        self.root.geometry("1000x600")  # 增加宽度以容纳左侧栏
        
        self.data = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.colorbar = None  # 存储颜色条的引用
        
        # 新增属性用于点击交互
        self.spectrum_window = SpectrumWindow(parent=self.root)
        self.click_cid = None  # 点击事件连接ID
        self.current_surface = None  # 当前投影表面
        self.data_coords = None  # 数据点3D坐标缓存 (x, y, z)
        self.current_lat = None  # 当前纬度数据
        self.current_lon = None  # 当前经度数据
        self.current_reflectance = None  # 当前反射率数据（2D投影）
        self.current_im = None  # 当前图像对象（2D投影）
        
        # 左侧栏相关属性
        self.file_listbox = None
        self.processed_files = []  # 存储检测到的文件列表
        cio.cubeio = None  # CubeIO实例
        
        # 创建界面
        self.create_widgets()
        
        # 初始化CubeIO并扫描processed目录
        self.init_cubeio_and_scan()
    def init_cubeio_and_scan(self):
        """初始化CubeIO并扫描processed目录"""
        try:
            # 导入config和cubeio
            from config import config
            from cubeio import CubeIO
            
            # 创建CubeIO实例
            cio.cubeio = CubeIO()
            
            # 扫描processed目录
            self.scan_processed_files()
            
        except ImportError as e:
            print(f"初始化CubeIO失败: {e}")
            self.status_label.config(text="初始化失败: 无法导入模块")
        except Exception as e:
            print(f"扫描processed目录失败: {e}")
            self.status_label.config(text=f"扫描失败: {str(e)}")
    
    def scan_processed_files(self):
        """扫描processed目录中的文件"""
        try:
            from config import config
            import os
            import glob
            
            processed_dir = os.path.join(config.py_path, 'processed')
            if not os.path.exists(processed_dir):
                print(f"processed目录不存在: {processed_dir}")
                self.status_label.config(text=f"目录不存在: {processed_dir}")
                return
            
            # 查找所有.pkl文件
            pattern = os.path.join(processed_dir, '*_processed.pkl')
            files = glob.glob(pattern)
            
            self.processed_files = []
            for filepath in files:
                filename = os.path.basename(filepath)
                # 提取cube名称，例如: "0982_3_processed.pkl" -> "0982_3"
                if filename.endswith('_processed.pkl'):
                    cube_name = filename[:-len('_processed.pkl')]
                    file_size = os.path.getsize(filepath)
                    file_size_mb = file_size / (1024 * 1024)
                    
                    self.processed_files.append({
                        'path': filepath,
                        'filename': filename,
                        'cube_name': cube_name,
                        'size_mb': file_size_mb
                    })
            
            # 更新文件列表显示
            self.update_file_listbox()
            
            print(f"扫描到 {len(self.processed_files)} 个processed文件")
            self.status_label.config(text=f"找到 {len(self.processed_files)} 个数据文件")
            
        except Exception as e:
            print(f"扫描文件失败: {e}")
            self.status_label.config(text=f"扫描失败: {str(e)}")
    
    def update_file_listbox(self):
        """更新文件列表框显示"""
        if self.file_listbox:
            self.file_listbox.delete(0, tk.END)
            for file_info in self.processed_files:
                display_text = f"{file_info['cube_name']} ({file_info['size_mb']:.1f} MB)"
                self.file_listbox.insert(tk.END, display_text)
    
    def create_widgets(self):
        """创建界面控件"""
        # 主容器：使用PanedWindow实现可调整大小的左右分割
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧栏：文件列表
        left_frame = tk.Frame(main_paned, width=250, relief=tk.RAISED, borderwidth=2)
        main_paned.add(left_frame, minsize=200, stretch="always")
        
        # 左侧栏标题
        left_title = tk.Label(left_frame, text="Processed Data Files", font=("Arial", 12, "bold"))
        left_title.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # 刷新按钮
        refresh_btn = tk.Button(left_frame, text="Refresh List", command=self.scan_processed_files)
        refresh_btn.pack(side=tk.TOP, padx=10, pady=(0, 10))
        
        # 文件列表框
        list_frame = tk.Frame(left_frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # 添加滚动条
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            font=("Courier", 10)
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # 绑定双击事件
        self.file_listbox.bind('<Double-Button-1>', self.on_file_double_click)
        
        # 右侧区域：控制面板和画布
        right_frame = tk.Frame(main_paned)
        main_paned.add(right_frame, minsize=600, stretch="always")
        
        # 控制面板
        control_frame = tk.Frame(right_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # 加载按钮（保留但功能将修改）
        self.load_btn = tk.Button(control_frame, text="Load Selected File", command=self.load_selected_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建球体按钮
        self.sphere_btn = tk.Button(control_frame, text="Create Sphere", command=self.create_sphere)
        self.sphere_btn.pack(side=tk.LEFT, padx=5)
        
        # 投影数据按钮
        self.project_btn = tk.Button(control_frame, text="Project Data", command=self.project_data)
        self.project_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        self.status_label = tk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # 画布区域
        self.canvas_frame = tk.Frame(right_frame, bg="gray")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 初始化 matplotlib 画布
        self.init_matplotlib_canvas()
        
    def on_file_double_click(self, event):
        """处理文件列表框的双击事件"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_file_by_index(index)
    
    def load_selected_file(self):
        """加载选中的文件"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_file_by_index(index)
        else:
            self.status_label.config(text="请先选择一个文件")
    
    def load_file_by_index(self, index):
        """根据索引加载文件"""
        if index < 0 or index >= len(self.processed_files):
            self.status_label.config(text="无效的文件索引")
            return
        
        file_info = self.processed_files[index]
        cube_name = file_info['cube_name']
        filepath = file_info['path']
        
        self.status_label.config(text=f"正在加载: {cube_name}...")
        self.root.update()
        
        try:
            
            
            # 加载processed类型的数据
            self.data = cio.cubeio.load(cube_name=cube_name, type='processed')
            
            # 更新状态
            self.status_label.config(text=f"已加载: {cube_name}")
            print(f"成功加载文件: {cube_name}")
            
            # 自动创建球体并投影数据
            self.create_sphere()
            self.project_data()
            
        except Exception as e:
            self.status_label.config(text=f"加载失败: {str(e)}")
            print(f"加载文件失败: {e}")
            import traceback
            traceback.print_exc()
        
    def init_matplotlib_canvas(self):
        """初始化 matplotlib 画布"""
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("3D Sphere View")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_sphere(self):
        """使用 matplotlib 3D 创建球体"""
        self.status_label.config(text="Creating sphere...")
        self.root.update()
        
        try:
            # 清除之前的绘图
            self.ax.clear()
            
            # 创建球体参数
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            
            # 球体参数方程
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # 绘制球体线框
            self.ax.plot_wireframe(x, y, z, color='gray', alpha=0.3, linewidth=0.5)
            
            # 设置坐标轴属性
            self.ax.set_title("3D Sphere")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.set_aspect('equal')
            
            # 更新画布
            self.canvas.draw()
            
            self.status_label.config(text="Sphere created successfully")
            print("create_sphere: Sphere created")
            
        except Exception as e:
            self.status_label.config(text=f"Creation failed: {e}")
            print(f"create_sphere error: {e}")
            
    def project_data(self):
        """使用多边形投影和双视图布局将数据投影到球体上"""
        self.status_label.config(text="Projecting data with dual view...")
        self.root.update()
        
        try:
            if self.data is None:
                self.status_label.config(text="Error: Please load data first")
                return
            
            # 清除之前的绘图和颜色条
            self.fig.clear()
            
            # 如果存在旧的颜色条则移除
            if self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except:
                    pass  # 颜色条可能已被移除
                self.colorbar = None
            
            # 创建双视图布局：左=球体概览（3D），右=数据投影（2D）
            ax1 = self.fig.add_subplot(121, projection='3d')  # 左：球体概览（3D）
            ax2 = self.fig.add_subplot(122)                   # 右：数据投影（2D）
            
            # 获取数据
            lat = self.data.lat
            lon = self.data.lon
            cube_rf = self.data.cube_rf
            
            if lat is None or lon is None or cube_rf is None:
                raise ValueError("Simulated data missing required fields")
            
            reflectance = cube_rf[:, :, 0]
            
            # 获取形状以进行性能优化
            ny, nx = lat.shape
            
            # 为着色归一化反射率
            reflectance_min = reflectance.min()
            reflectance_max = reflectance.max()
            if reflectance_max > reflectance_min:
                reflectance_norm = (reflectance - reflectance_min) / (reflectance_max - reflectance_min)
            else:
                reflectance_norm = np.zeros_like(reflectance)
            
            # 创建颜色映射
            cmap = cm.get_cmap('viridis')
            
            # 将每个像素作为多边形投影到球体上
            radius = 1.0
            
            # 转换为弧度
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            
            # 转换为 3D 坐标（仅用于左图）
            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)
            
            # --- 左图：带有覆盖区域的球体概览（3D）---
            # 创建球体线框
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # 绘制球体线框
            ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3, linewidth=0.5)
            
            # 计算覆盖区域边界
            lat_min, lat_max = lat.min(), lat.max()
            lon_min, lon_max = lon.min(), lon.max()
            
            # 创建覆盖区域多边形（球面圆弧边界）
            # 为每个边采样多个点以形成球面圆弧
            
            # 采样点数
            n_points = 20
            
            # 初始化边界点列表
            boundary_lats = []
            boundary_lons = []
            
            # 底部边：纬度固定为 lat_min，经度从 lon_min 到 lon_max
            lons_bottom = np.linspace(lon_min, lon_max, n_points)
            boundary_lats.extend([lat_min] * n_points)
            boundary_lons.extend(lons_bottom)
            
            # 右边：经度固定为 lon_max，纬度从 lat_min 到 lat_max
            lats_right = np.linspace(lat_min, lat_max, n_points)
            boundary_lats.extend(lats_right)
            boundary_lons.extend([lon_max] * n_points)
            
            # 顶部边：纬度固定为 lat_max，经度从 lon_max 到 lon_min
            lons_top = np.linspace(lon_max, lon_min, n_points)
            boundary_lats.extend([lat_max] * n_points)
            boundary_lons.extend(lons_top)
            
            # 左边：经度固定为 lon_min，纬度从 lat_max 到 lat_min
            lats_left = np.linspace(lat_max, lat_min, n_points)
            boundary_lats.extend(lats_left)
            boundary_lons.extend([lon_min] * n_points)
            
            # 转换为 numpy 数组
            coverage_lats = np.array(boundary_lats)
            coverage_lons = np.array(boundary_lons)
            
            coverage_lat_rad = np.radians(coverage_lats)
            coverage_lon_rad = np.radians(coverage_lons)
            
            coverage_x = radius * np.cos(coverage_lat_rad) * np.cos(coverage_lon_rad)
            coverage_y = radius * np.cos(coverage_lat_rad) * np.sin(coverage_lon_rad)
            coverage_z = radius * np.sin(coverage_lat_rad)
            
            # 将覆盖区域绘制为半透明灰色表面
            # 为覆盖区域创建网格
            lat_grid = np.linspace(lat_min, lat_max, 10)
            lon_grid = np.linspace(lon_min, lon_max, 10)
            lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
            
            lat_mesh_rad = np.radians(lat_mesh)
            lon_mesh_rad = np.radians(lon_mesh)
            
            coverage_x_mesh = radius * np.cos(lat_mesh_rad) * np.cos(lon_mesh_rad)
            coverage_y_mesh = radius * np.cos(lat_mesh_rad) * np.sin(lon_mesh_rad)
            coverage_z_mesh = radius * np.sin(lat_mesh_rad)
            
            # 绘制覆盖区域
            ax1.plot_surface(coverage_x_mesh, coverage_y_mesh, coverage_z_mesh,
                           color='gray', alpha=0.3, shade=False)
            
            # 绘制覆盖边界
            ax1.plot(coverage_x, coverage_y, coverage_z, color='red', linewidth=2, alpha=0.7)
            
            # 设置左图属性
            ax1.set_title("Sphere Overview with Coverage Area")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            ax1.set_aspect('equal')
            ax1.view_init(elev=20, azim=45)
            
            # --- 右图：数据投影（2D）---
            # 创建2D投影：使用经纬度网格显示反射率
            # 使用pcolormesh进行2D投影
            im = ax2.pcolormesh(lon, lat, reflectance, cmap=cmap, shading='auto', vmin=-0.02, vmax = 0.07)
            
            # 为右图添加颜色条
            self.colorbar = self.fig.colorbar(im, ax=ax2, label='Reflectance')
            
            # 设置右图属性
            ax2.set_title("OMEGA Data Projected (2D)")
            ax2.set_xlabel("Longitude (°)")
            ax2.set_ylabel("Latitude (°)")
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            
            # 保存数据用于点击交互（2D版本）
            self.current_lat = lat
            self.current_lon = lon
            self.current_reflectance = reflectance
            self.current_im = im  # 保存图像对象用于坐标转换
            
            # 移除旧的点击事件
            if self.click_cid is not None:
                self.fig.canvas.mpl_disconnect(self.click_cid)
            
            # 添加新的点击事件
            self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            
            # 更新画布
            self.canvas.draw()
            
            self.status_label.config(text="Data projected with dual view (click to show spectrum)")
            print(f"project_data: Dual view created (coverage: lat[{lat_min:.1f},{lat_max:.1f}], lon[{lon_min:.1f},{lon_max:.1f}])")
            
        except Exception as e:
            self.status_label.config(text=f"Projection failed: {e}")
            print(f"project_data error: {e}")
            import traceback
            traceback.print_exc()
    
    def find_nearest_pixel_2d(self, click_lon, click_lat):
        """在2D投影中找到点击位置最近的数据像素
        
        Args:
            click_lon: 点击的经度坐标
            click_lat: 点击的纬度坐标
            
        Returns:
            (i, j): 最近像素的索引
            (lat, lon): 最近像素的经纬度
        """
        if self.current_lat is None or self.current_lon is None:
            return None, None
        
        lat = self.current_lat
        lon = self.current_lon
        
        # 计算点击点与所有数据点的经纬度距离（简化欧氏距离）
        # 注意：这不是精确的大圆距离，但对于小范围数据足够
        distances = np.sqrt((lon - click_lon)**2 + (lat - click_lat)**2)
        
        # 找到最小距离的索引
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        i, j = min_idx
        
        # 获取对应的经纬度
        lat_val = lat[i, j]
        lon_val = lon[i, j]
        
        return (i, j), (lat_val, lon_val)
    
    def on_click(self, event):
        """处理点击事件（2D版本）"""
        if event.inaxes is None:
            return  # 点击在画布外
        
        # 检查是否点击在右图（数据投影图）
        # 右图是第二个子图（索引1）
        if len(self.fig.axes) < 2 or event.inaxes != self.fig.axes[1]:
            return  # 不是点击在右图
        
        # 获取点击的2D坐标（经纬度）
        click_lon, click_lat = event.xdata, event.ydata
        
        # 找到最近的数据像素
        pixel_idx, (lat, lon) = self.find_nearest_pixel_2d(click_lon, click_lat)
        
        if pixel_idx is None:
            print("点击事件：无法找到最近像素")
            return
        
        i, j = pixel_idx
        
        # 提取光谱数据
        try:
            if self.data is None:
                print("点击事件：数据未加载")
                return
            
            # 获取波长和反射率数据
            if OMEGAPY_AVAILABLE and not isinstance(self.data, dict):
                # OMEGAdata 对象
                omega = self.data
                if hasattr(omega, 'lam'):
                    wavelengths = omega.lam
                else:
                    wavelengths = None
                
                if hasattr(omega, 'cube_rf'):
                    reflectance = omega.cube_rf[i, j, :]
                else:
                    reflectance = None
            else:
                pass
            
            if wavelengths is None or reflectance is None:
                print("点击事件：无法获取波长或反射率数据")
                return
            
            # 生成标签
            filename = self.data.name
            
            label = f"#{self.spectrum_window.spectrum_count + 1}: ({lat:.1f}°, {lon:.1f}°) [{filename}]"
            
            # 添加到光谱窗口
            self.spectrum_window.add_spectrum(wavelengths, reflectance, label=label)
            
            # 更新状态
            self.status_label.config(text=f"Spectrum extracted at ({lat:.1f}°, {lon:.1f}°)")
            print(f"点击事件：提取光谱 (i={i}, j={j}, lat={lat:.1f}°, lon={lon:.1f}°)")
            
        except Exception as e:
            print(f"点击事件处理错误: {e}")
            import traceback
            traceback.print_exc()
            
    def run(self):
        """运行主循环"""
        self.root.mainloop()

if __name__ == "__main__":
    viewer = SimpleSphereViewer()
    viewer.run()
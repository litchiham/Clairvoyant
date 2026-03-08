import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from pathlib import Path
import os
import glob
import threading
import queue
import time
import subprocess
import sys
import process
import predict
import cubeio as cio
from config import *
class TkinterGui():
    def __init__(self,root:tk.Tk):
        self.root=root
        self.root.title('gui_v0.2')
        self.root.geometry('700x500')

        # 设置样式
        self.setup_styles()

        # 创建主框架
        self.create_main_frame()
                      
        # 创建标签页（Notebook）容器
        self.create_notebook()
        
        # 创建路径栏
        self.bin_path=tk.StringVar(value=config.bin_path)
        self.py_path=tk.StringVar(value=config.py_path)
        self.buffer_path=tk.StringVar(value=config.buffer_path)
        self.dust_path=tk.StringVar(value=config.dust_path)
        self.create_pathbar()

        # 创建队列
        self.q0=queue.Queue()
        self.q1=queue.Queue()
        self.poll()

    def setup_styles(self):
        '''设置控件样式'''
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('微软雅黑', 16, 'bold'))
        style.configure('Heading.TLabel', font=('微软雅黑', 12, 'bold'))
        style.configure('Status.TLabel', font=('微软雅黑', 10))

    def create_main_frame(self):
        """创建主框架"""
        # 使用Frame作为主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置根窗口的网格权重，使其随窗口缩放
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 配置主框架的网格权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题标签
        title_label = ttk.Label(
            main_frame, 
            text="TkinterGui程序", 
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # 保存主框架引用
        self.main_frame = main_frame

    def create_notebook(self):
        """创建标签页容器"""
        # 创建Notebook（标签页控件）
        notebook = ttk.Notebook(self.main_frame)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N),pady=(0,10))
        
        # 创建各个标签页
        self.create_preprocessing_tab(notebook)
        self.create_predicting_tab(notebook)
        self.create_visualizing_tab(notebook)

        # 初始化IO路径
        self.folderI=[]
        self.folderO=[]
        for i in range(3):
            self.folderI.append(tk.StringVar())
            self.folderO.append(tk.StringVar())

        # 初始化cube_names
        self.cube_names=[]
        for i in range(3):
            self.cube_names.append([])

        # 保存notebook引用
        self.notebook = notebook

    def create_preprocessing_tab(self,notebook):
        """创建预处理标签页"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="预处理")
        
        # 配置网格
        for i in range(6):
            frame.columnconfigure(i, weight=1)

        # 显示当前文件路径
        self.file_label_preprocessing = ttk.Label(frame, text="未选择Cube", foreground='gray', style='Status.TLabel')
        self.file_label_preprocessing.grid(row=1,column=0,sticky=tk.W,padx=5)
        # 开始按钮
        self.button_process0 = ttk.Button(frame,text="开始处理",command=self.process0,state=tk.NORMAL)
        self.button_process0.grid(row=0,column=1,sticky=tk.W,padx=5)   
        # 显示按钮
        button_show = ttk.Button(frame, text="显示",command=self.show0,state=tk.NORMAL)
        button_show.grid(row=2, column=1, sticky=tk.W, padx=5,pady=5)

        # 进度条
        self.progress_var = tk.IntVar(value=0)
        self.progressbar0 = ttk.Progressbar(frame, variable=self.progress_var,length=300)
        self.progressbar0.grid(row=0,column=2,sticky=tk.W)    
        self.progressbar0_label=ttk.Label(frame,text='',style='Status.TLabel')  
        self.progressbar0_label.grid(row=0,column=3)

    def create_predicting_tab(self,notebook):
        """创建预测标签页"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="预测")
        
        # 配置网格
        for i in range(6):
            frame.columnconfigure(i, weight=1)

        # 显示当前文件路径
        self.file_label_predicting = ttk.Label(frame, text="未选择Cube", foreground='gray', style='Status.TLabel')
        self.file_label_predicting.grid(row=1,column=0,sticky=tk.W,padx=5)
        # 处理按钮
        self.button_process1 = ttk.Button(frame,text="开始预测",command=self.process1)
        self.button_process1.grid(row=0,column=1,sticky=tk.W,padx=5)  
        # 显示按钮
        button_show = ttk.Button(frame, text="显示",command=self.show1)
        button_show.grid(row=2, column=1, sticky=tk.W, padx=5,pady=5)  

        # 进度条
        self.predict_var = tk.IntVar(value=0)
        self.progressbar1 = ttk.Progressbar(frame, variable=self.predict_var,length=300)
        self.progressbar1.grid(row=0,column=2,sticky=tk.W)                

    def create_visualizing_tab(self,notebook):
        """创建可视化标签页"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="可视化")
        
        # 配置网格
        for i in range(6):
            frame.columnconfigure(i, weight=1)

        # 显示按钮
        self.button_show = ttk.Button(frame, text="显示",command=self.show_visualization)
        self.button_show.grid(row=1, column=1, sticky=tk.W, padx=5,pady=5)  

    def create_pathbar(self):
        pathbar = ttk.LabelFrame(self.main_frame, text="工作路径")
        pathbar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 100))

        pathbar.columnconfigure(0, weight=1)

        frame = ttk.Frame(pathbar, padding=10)
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        for i in range(6):
            frame.columnconfigure(i, weight=1)

        bin_path_label = ttk.Label(frame, text="bin_path", style='Status.TLabel')
        bin_path_label.grid(row=0, column=0, sticky=tk.W,padx=10)
        self.bin_path_bar = tk.Label(frame,textvariable=self.bin_path,width=50,relief="sunken",bg="white",anchor="w",padx=5)
        self.bin_path_bar.grid(row=0,column=1,sticky=tk.W)
        bin_path_btn = ttk.Button(frame,text='选择路径',command=self.select_path0)
        bin_path_btn.grid(row=0,column=2)

        py_path_label = ttk.Label(frame, text="py_path", style='Status.TLabel')
        py_path_label.grid(row=1, column=0, sticky=tk.W,padx=10)
        self.py_path_bar = tk.Label(frame,textvariable=self.py_path,width=50,relief="sunken",bg="white",anchor="w",padx=5)
        self.py_path_bar.grid(row=1,column=1,sticky=tk.W)
        py_path_btn = ttk.Button(frame,text='选择路径',command=self.select_path1)
        py_path_btn.grid(row=1,column=2)

        buffer_path_label = ttk.Label(frame, text="buffer_path", style='Status.TLabel')
        buffer_path_label.grid(row=2, column=0, sticky=tk.W,padx=10)
        self.buffer_path_bar = tk.Label(frame,textvariable=self.buffer_path,width=50,relief="sunken",bg="white",anchor="w",padx=5)
        self.buffer_path_bar.grid(row=2,column=1,sticky=tk.W)
        buffer_path_btn = ttk.Button(frame,text='选择路径',command=self.select_path2)
        buffer_path_btn.grid(row=2,column=2)

        dust_path_label = ttk.Label(frame, text="dust_path", style='Status.TLabel')
        dust_path_label.grid(row=3, column=0, sticky=tk.W,padx=10)
        self.dust_path_bar = tk.Label(frame,textvariable=self.dust_path,width=50,relief="sunken",bg="white",anchor="w",padx=5)
        self.dust_path_bar.grid(row=3,column=1,sticky=tk.W)
        dust_path_btn = ttk.Button(frame,text='选择路径',command=self.select_path3)
        dust_path_btn.grid(row=3,column=2)
    # 事件处理方法
    def button_clicked_example(self):
        """按钮点击事件处理"""
        messagebox.showinfo("按钮点击", "点了")
    
    def select_import0(self):
        folder=filedialog.askdirectory(title='请选择数据文件夹')
        if folder:
            try:
                self.cube_names[0]=self.extract_cube_names(folder)
                self.folderI[0]=folder
            except:
                messagebox.showerror(title='错误',message='请重新选择')
        if len(self.cube_names[0]) > 0:
            self.file_label_preprocessing.configure(text=f'已选择{len(self.cube_names[0])}个Cube',foreground='black')
        else :
            self.file_label_preprocessing.configure(text="未选择Cube", foreground='gray')

    def select_export0(self):
        pass
    def process0(self):   
        if getattr(self, '_running0', False):
            return 
        self.cube_names[0]=self.extract_cube_names(config.bin_path)
        def f(callback=None):
            p = process.Process()
            p.import_cubes(self.cube_names[0])
            p.process_cubes(callback=callback)
            cio.log('Process', 'Processing completed.', 'INFO')            
        self.progress_var.set(0)
        self.button_process0.config(state='disabled')
        self.progressbar0.config(maximum=len(self.cube_names[0]))
        self._running0 = True
        threading.Thread(target=self._worker, args=(f,self.q0), daemon=True).start()
    def show0(self):
        pass
    def show_visualization(self):
        """启动可视化窗口"""
        try:
            subprocess.Popen([sys.executable, 'code/gui-vis.py'])
        except Exception as e:
            messagebox.showerror("错误", f"无法启动可视化窗口: {e}")
    def extract_cube_names(self,folder):
        '''获取当前目录下所有 ORBxxxx_x_DATA 文件夹中的 xxxx_x 部分，并返回列表'''
        result = []
        # 匹配所有以 ORB 开头、以 _DATA 结尾的文件夹
        for name in glob.glob(str(Path(folder)/'ORB*_DATA')):
            middle = str(name)[-11:-5]
            result.append(middle)
        return result
    def extract_processed_cube_names(self, folder):
        '''获取当前目录下所有 *_processed.pkl 文件中的 cube_name 部分，并返回列表'''
        result = []
        # 匹配所有以 _processed.pkl 结尾的文件
        for name in glob.glob(str(Path(folder) / '*_processed.pkl')):
            filename = os.path.basename(name)
            if filename.endswith('_processed.pkl'):
                cube_name = filename[:-len('_processed.pkl')]
                result.append(cube_name)
        return result

    def select_export1(self):
        pass
    def process1(self):
        if getattr(self, '_running1', False):
            return
        if not self.cube_names[1]:
            processed_dir = os.path.join(config.py_path, 'processed')
            if os.path.exists(processed_dir):
                self.cube_names[1] = self.extract_processed_cube_names(processed_dir)
                self.file_label_predicting.configure(text=f'已选择{len(self.cube_names[1])}个Processed Cube', foreground='black')
            else:
                messagebox.showerror("错误", "processed目录不存在，请先处理数据")
                return
        if not self.cube_names[1]:
            messagebox.showerror("错误", "没有找到processed cube")
            return

        # prepare progress tracking
        self.predict_total = len(self.cube_names[1])
        self.predict_count = 0
        self.predict_var.set(0)
        self.progressbar1.config(maximum=100, mode='determinate')
        self.progressbar1['value'] = 0

        def f():
            p = predict.Predict()
            p.predict_cubes(self.cube_names[1], max_workers=1, callback=self._predict_callback)
            cio.log('Predict', 'Predicting completed.', 'INFO')

        self.button_process1.config(state='disabled')
        self._running1 = True
        threading.Thread(target=self._worker_simple, args=(f, self.q1), daemon=True).start()
    def show1(self):
        predicted_dir = os.path.join(config.py_path, 'predicted')
        if os.path.exists(predicted_dir):
            files = os.listdir(predicted_dir)
            if files:
                file_list = '\n'.join(files)
                messagebox.showinfo("预测结果", f"预测结果文件：\n{file_list}")
            else:
                messagebox.showinfo("预测结果", "predicted目录为空")
        else:
            messagebox.showerror("错误", "predicted目录不存在")

    def select_import2(self):
        pass 
    def select_export2(self):
        pass
    def show2(self):
        pass
    
##########
    def select_path0(self):
        folder=filedialog.askdirectory(title='请选择路径')
        if folder:
            config.bin_path=folder
            self.bin_path.set(folder)
    def select_path1(self):
        folder=filedialog.askdirectory(title='请选择路径')
        if folder:
            config.py_path=folder
            self.py_path.set(folder)
    def select_path2(self):
        folder=filedialog.askdirectory(title='请选择路径')
        if folder:
            config.buffer_path=folder
            self.buffer_path.set(folder)
    def select_path3(self):
        folder=filedialog.askdirectory(title='请选择路径')
        if folder:
            config.dust_path=folder
            self.dust_path.set(folder)

    def poll(self):
        self.poll0()
        self.poll1()
    def poll0(self):
        try:
            while True:
                item = self.q0.get_nowait()
                if item == 'done':
                    self._running0 = False
                    self.button_process0.config(state='normal')
                elif isinstance(item, tuple) and item[0] == 'err':
                    messagebox.showerror('Error', item[1])
                    self._running0 = False
                    self.button_process0.config(state='normal')
                else:
                    self.progress_var.set(min(len(self.cube_names[0]), self.progress_var.get() + int(item)))
                    self.progressbar0_label.config(text=f'{self.progress_var.get()}/{len(self.cube_names[0])}')
        except queue.Empty:
            pass
        self.root.after(100, self.poll0)
    def poll1(self):
        try:
            while True:
                item = self.q1.get_nowait()
                if item == 'done':
                    self._running1 = False
                    self.button_process1.config(state='normal')
                elif isinstance(item, tuple) and item[0] == 'err':
                    messagebox.showerror('Error', item[1])
                    self._running1 = False
                    self.button_process1.config(state='normal')
                elif isinstance(item, int):
                    # progress increment
                    self.predict_count += item
                    percent = int(self.predict_count / self.predict_total * 100)
                    self.predict_var.set(percent)
                    self.progressbar1['value'] = percent
        except queue.Empty:
            pass
        self.root.after(100, self.poll1)   

    def _worker(self,f,q):
        def cb(_result):
            q.put(1)

        try:
            f(callback=cb)
        except Exception as e:
            q.put(('err', str(e)))
        finally:
            q.put('done')
    def _worker_simple(self, f, q):
        try:
            f()
        except Exception as e:
            q.put(('err', str(e)))
        finally:
            q.put('done')

    def _predict_callback(self, result):
        # called from predict_cubes after each cube finishes
        self.q1.put(1) 

def main():
    """主函数"""
    root=tk.Tk()
    app = TkinterGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
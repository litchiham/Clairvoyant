import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from pathlib import Path
import os
import glob
import threading
class TkinterGui():
    def __init__(self):
        self.root=tk.Tk()
        self.root.title('gui_v0.1')
        self.root.geometry('700x500')

        # 设置样式
        self.setup_styles()

        # 创建主框架
        self.create_main_frame()
                      
        # 创建标签页（Notebook）容器
        self.create_notebook()
        
        # 创建状态栏
        #self.create_statusbar()

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
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
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

        # 导入按钮
        button_import = ttk.Button(frame, text="选择Cube",command=self.select_import0)
        button_import.grid(row=0, column=0, sticky=tk.W, padx=5)
        # 显示当前文件路径
        self.file_label_preprocessing = ttk.Label(frame, text="未选择Cube", foreground='gray', style='Status.TLabel')
        self.file_label_preprocessing.grid(row=1,column=0,sticky=tk.W,padx=5)
        # 开始按钮
        button_process = ttk.Button(frame,text="开始处理",command=self.process0,state=tk.DISABLED)
        button_process.grid(row=0,column=1,sticky=tk.W,padx=5)   
        # 导出按钮
        button_export = ttk.Button(frame, text="导出为...",command=self.select_export0)
        button_export.grid(row=2, column=0, sticky=tk.W, padx=5,pady=5)
        # 显示按钮
        button_show = ttk.Button(frame, text="显示",command=self.show0,state=tk.DISABLED)
        button_show.grid(row=2, column=1, sticky=tk.W, padx=5,pady=5)

        # 进度条
        self.progress_var = tk.DoubleVar()
        progressbar = ttk.Progressbar(frame, variable=self.progress_var,length=300)
        progressbar.grid(row=0,column=2,sticky=tk.W)      

    def create_predicting_tab(self,notebook):
        """创建预测标签页"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="预测")
        
        # 配置网格
        for i in range(6):
            frame.columnconfigure(i, weight=1)

        # 导入按钮
        button_import = ttk.Button(frame, text="选择Cube",command=self.button_clicked_example)
        button_import.grid(row=0, column=0, sticky=tk.W, padx=5)
        # 显示当前文件路径
        self.file_label_predicting = ttk.Label(frame, text="未选择Cube", foreground='gray', style='Status.TLabel')
        self.file_label_predicting.grid(row=1,column=0,sticky=tk.W,padx=5)

        # 处理按钮
        button_process = ttk.Button(frame,text="开始预测",command=self.button_clicked_example)
        button_process.grid(row=0,column=1,sticky=tk.W,padx=5)  
        # 导出按钮
        button_export = ttk.Button(frame, text="导出为...",command=self.button_clicked_example)
        button_export.grid(row=2, column=0, sticky=tk.W, padx=5,pady=5)  
        # 显示按钮
        button_show = ttk.Button(frame, text="显示",command=self.button_clicked_example)
        button_show.grid(row=2, column=1, sticky=tk.W, padx=5,pady=5)  

        # 进度条
        self.progress_var = tk.DoubleVar()
        progressbar = ttk.Progressbar(frame, variable=self.progress_var,length=300)
        progressbar.grid(row=0,column=2,sticky=tk.W)                

    def create_visualizing_tab(self,notebook):
        """创建可视化标签页"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="可视化")
        
        # 配置网格
        for i in range(6):
            frame.columnconfigure(i, weight=1)

        # 导入按钮
        button_import = ttk.Button(frame, text="导入结果",command=self.button_clicked_example)
        button_import.grid(row=0, column=0, sticky=tk.W, padx=5)  
        # 显示当前文件路径
        self.file_label_visualizing = ttk.Label(frame, text="未选择Cube", foreground='gray', style='Status.TLabel')
        self.file_label_visualizing.grid(row=0,column=1,sticky=tk.W,padx=5)

        # 导出按钮
        button_export = ttk.Button(frame, text="导出为...",command=self.button_clicked_example)
        button_export.grid(row=1, column=0, sticky=tk.W, padx=5,pady=5)  
        # 显示按钮
        button_show = ttk.Button(frame, text="显示",command=self.button_clicked_example)
        button_show.grid(row=1, column=1, sticky=tk.W, padx=5,pady=5)  

        # 进度条
        self.progress_var = tk.DoubleVar()
        progressbar = ttk.Progressbar(frame, variable=self.progress_var,length=300)
        progressbar.grid(row=0,column=2,sticky=tk.W) 

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
        pass
    def show0(self):
        pass
    def extract_cube_names(self,folder):
        '''获取当前目录下所有 ORBxxxx_x_DATA 文件夹中的 xxxx_x 部分，并返回列表'''
        result = []
        # 匹配所有以 ORB 开头、以 _DATA 结尾的文件夹
        for name in glob.glob(str(Path(folder)/'ORB*_DATA')):
            middle = str(name)[-11:-5]
            result.append(middle)
        return result

    def select_import1(self):
        pass
    def select_export1(self):
        pass
    def process1(self):
        pass
    def show1(self):
        pass

    def select_import2(self):
        pass 
    def select_export2(self):
        pass
    def show2(self):
        pass


def main():
    """主函数"""
    app = TkinterGui()
    app.root.mainloop()

if __name__ == "__main__":
    main()
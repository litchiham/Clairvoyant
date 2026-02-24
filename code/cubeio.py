# 导入必要的模块
# omegapy.omega_data: OMEGA数据处理核心模块，用于加载和操作OMEGA数据
# omegapy.omega_plots: OMEGA数据可视化模块，提供各种绘图功能
# os: 操作系统接口模块，用于文件路径操作和目录管理
# time: 时间处理模块，用于生成时间戳
# tkinter.messagebox: 图形用户界面消息框模块，用于显示通知对话框
# predict: 预测模块，用于处理预测结果
# typing.Literal: 类型提示模块，用于指定字符串字面量类型，提高代码可读性和类型安全性

from config import *
import omegapy.omega_data as od
import omegapy.omega_plots as op
import os
import time
import tkinter.messagebox as messagebox
import numpy as np
from typing import Literal



# 日志级别在config.py中
_log_levels={
    'DEBUG':0,
    'INFO':1,
    'WARNING':2,
    'ERROR':3
}
# 日志记录函数
def log(source:str,message:str, type: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'], show_in_window=False, flush = False):
    if(_log_levels[type] >= _log_levels[config.log_level]):
        time_stamp = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
        log_str = f"{time_stamp} [{type}] @{source}: {message}"
        if(flush):
            print(log_str,end = '\r', flush=True)
        else:
            print(log_str)
        if(show_in_window):
            #此处有问题，messagebox可能阻塞线程
            messagebox.showinfo(title=type, message=message)

class Predicted:
    '''存储和一个cube的预测结果'''
    
    def __init__(self, points_array=None):
        """
        初始化预测结果对象
        
        参数:
        points_array: numpy数组，形状为(n, 5)，每行代表一个点
        [lon, lat, class_x1, class_x2, reg]
        可能更改
        """
        if points_array is None:
            self.points = np.empty((0, 5))  # 创建空的5列数组
        else:
            self.points = np.array(points_array, dtype=float)


    def get(self, key):
        if len(self.points) == 0:
            return np.array([])
            
        key_map = {
            'lon': 0,
            'lat': 1, 
            'class_x1': 2,
            'class_x2': 3,
            'reg': 4
        }
        
        if key not in key_map:
            raise ValueError(f"不支持的key: {key}")
            
        col_idx = key_map[key]
        return self.points[:, col_idx]
    
    # 一些辅助方法
    def append(self, point):
        self.points = np.append(self.points, point, axis=0)
    def to_array(self):
        return self.points
    
def save_Predicted(predicted_cube, filepath, format='npz'):
    if format == 'npz':
        np.savez(filepath, points=predicted_cube.points)
    elif format == 'pkl':
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(predicted_cube, f)


def load_Predicted(filepath, format='npz'):
    if format == 'npz':
        data = np.load(filepath)
        return Predicted(points_array=data['points'])
    

# CubeIO 类：OMEGA数据输入输出管理器
# 功能：统一管理OMEGA数据的加载、存储和路径配置
# 该类提供对原始二进制数据、处理后的Python数据和预测结果的统一访问接口

class CubeIO:
    def __init__(self):
        
        # 直接使用 config 中的路径配置
        if(os.path.exists(config.buffer_path) == False):
            os.mkdir(config.buffer_path)
        od.set_omega_bin_path(config.bin_path)
        pkl_path = os.path.join(config.py_path, 'processed')
        od.set_omega_py_path(pkl_path)  #此处存在语义不清
    # 获取指定类型的数据路径
    # 参数：
    #   type: 路径类型，必须是 'bin'、'py'或 'buffer'之一
    # 返回值：对应类型的路径字符串
    def get_path(self, type: Literal['bin', 'py', 'buffer']):
        if(type == 'bin'):
            return config.bin_path
        elif(type == 'py'):
            return config.py_path
        elif(type == 'buffer'):
            return config.buffer_path
        else:
            raise ValueError('Invalid type')
    
    # 设置指定类型的数据路径
    # 参数：
    #   type: 路径类型，必须是 'bin'、'py' 或 'buffer' 之一
    #   path: 新的路径字符串
    def set_path(self, type: Literal['bin', 'py', 'buffer'], path:str):
        if(type == 'bin'):
            config.bin_path = path
            od.set_omega_bin_path(path)
        elif(type == 'py'):
            config.py_path = path
            pkl_path = os.path.join(path, 'processed')
            od.set_omega_py_path(pkl_path)  #此处存在语义不清
        elif(type == 'buffer'):
            config.buffer_path = path
        else:
            raise ValueError('Invalid type')
    # 主要数据加载方法：根据指定类型加载OMEGA立方体数据
    # 参数：
    #   cube_name: 字符串，立方体名称（如 '0982_3'）
    #   type: 数据类型，必须是 'raw'（原始二进制数据）、'processed'（处理后的Python数据）或 'predicted'（预测结果）之一
    # 返回值：加载的数据对象（具体类型取决于type参数）
    def load(self, cube_name:str, type: Literal['raw', 'processed', 'predicted']):
        import shutil
        import os
        
        # 记录开始
        log('CubeIO.load', f'开始加载立方体 {cube_name}，类型 {type}', 'INFO')
        
        try:
            # 1. 确定源文件路径（具体名称稍后处理）
            # 这里需要根据 type 确定源路径，暂时使用简单逻辑
            if type == 'raw':
                source_path = os.path.join(config.bin_path, f'ORB{cube_name}_DATA')
            elif type == 'processed':
                source_path = os.path.join(config.py_path,'processed', f'{cube_name}_processed.pkl')
            elif type == 'predicted':
                source_path = os.path.join(config.py_path,'predicted', f'{cube_name}_predicted.npz')
            else:
                raise ValueError(f'无效的类型: {type}')
            
            # 检查源文件是否存在
            if not os.path.exists(source_path):
                raise FileNotFoundError(f'源文件不存在: {source_path}')
            
            # 2. 复制到缓冲区
            buffer_target = os.path.join(config.buffer_path, os.path.basename(source_path))
            log('CubeIO.load', f'复制 {source_path} 到 {buffer_target}', 'INFO')
            
            if os.path.isdir(source_path):
                shutil.copytree(source_path, buffer_target, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, buffer_target)
            
            # 3. 调用相应的加载方法
            result = None
            if type == 'raw':
                result = self._load_raw(cube_name=cube_name) #因为omegapy的加载生数据参数不同
            elif type == 'processed':
                result = self._load_processed(buffer_target)
            elif type == 'predicted':
                result = self._load_predicted(buffer_target)
            
            # 4. 加载完成后删除缓冲区文件
            log('CubeIO.load', f'删除缓冲区文件: {buffer_target}', 'INFO')
            if os.path.isdir(buffer_target):
                shutil.rmtree(buffer_target)
            else:
                os.remove(buffer_target)
            
            log('CubeIO.load', f'成功加载立方体 {cube_name}', 'INFO')
            return result
            
        except Exception as e:
            log('CubeIO.load', f'加载失败: {str(e)}', 'ERROR')
            raise
    # 私有方法：加载原始OMEGA二进制数据
    def _load_raw(self,cube_name:str):
        log('CubeIO._load_raw', f'开始加载立方体 {cube_name}，类型 raw', 'DEBUG')
        ret_omega_data=od.OMEGAdata(cube_name,disp=False)
        return ret_omega_data
    
    # 私有方法：加载处理后的Python数据
    def _load_processed(self,buffer_target:str):
        log('CubeIO._load_processed', f'开始加载立方体 {buffer_target}，类型 processed', 'DEBUG')
        ret_omega_data=od.load_omega(buffer_target)
        return ret_omega_data
    
    # 私有方法：加载预测结果数据
    def _load_predicted(self, buffer_target:str):
        log('CubeIO._load_predicted', f'开始加载立方体 {buffer_target}，类型 predicted', 'DEBUG')
        return load_Predicted(buffer_target)
cubeio=CubeIO()
if __name__ == '__main__':
    # 设置日志级别为DEBUG，显示详细调试信息
    config.log_level = 'DEBUG'
    # 创建CubeIO实例，配置数据路径
    res = cubeio.load(cube_name='0982_3', type='raw')
    op.show_omega_interactif_v2(res)
    # 等待用户按回车键继续，防止控制台窗口关闭
    input("Press Enter to continue...")
    
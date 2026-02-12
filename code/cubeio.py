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
import predict
from typing import Literal



# 日志级别配置
# log_level: 当前日志输出级别，控制哪些日志消息会被显示
# _log_levels: 日志级别映射字典，将日志级别名称映射为数字优先级（数值越小，优先级越高）
#   'DEBUG':0 - 调试信息，最详细
#   'INFO':1 - 一般信息，程序运行状态
#   'WARNING':2 - 警告信息，潜在问题
#   'ERROR':3 - 错误信息，程序异常

log_level = config.log_level
_log_levels={
    'DEBUG':0,
    'INFO':1,
    'WARNING':2,
    'ERROR':3
}
# 日志记录函数
# 功能：统一的日志记录接口，支持不同严重级别的日志输出
# 参数：
#   source: 字符串，标识日志来源（模块/函数名）
#   message: 字符串，日志消息内容
#   type: 日志级别，必须是 'DEBUG'、'INFO'、'WARNING' 或 'ERROR' 之一
#   show_in_window: 布尔值，是否在GUI窗口中显示消息框
# 返回值：无
# 注意：messagebox.showinfo 可能在GUI应用中阻塞主线程，需谨慎使用
def log(source:str,message:str, type: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'], show_in_window=False):
    if(_log_levels[type] >= _log_levels[log_level]):
        time_stamp = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
        log_str = f"{time_stamp} [{type}] @{source}: {message}"
        print(log_str)
        if(show_in_window):
            #此处有问题，messagebox可能阻塞线程
            messagebox.showinfo(title=type, message=message)

# CubeIO 类：OMEGA数据输入输出管理器
# 功能：统一管理OMEGA数据的加载、存储和路径配置
# 该类提供对原始二进制数据、处理后的Python数据和预测结果的统一访问接口

class CubeIO:
    # 类变量：存储基础路径配置
    # _base_bin_path: 原始OMEGA二进制数据文件的根目录路径
    # _base_py_path: 处理后的Python数据文件（pkl）的根目录路径
    # _base_buffer_path: 临时缓冲区路径，用于数据加载过程中的中间存储
    _base_bin_path = ''
    _base_py_path = ''
    _base_buffer_path = '~/buffer'
    
    # 构造函数：初始化CubeIO实例
    # 参数：
    #   bin_path: 原始二进制数据目录路径
    #   py_path: Python处理数据目录路径
    #   buffer_path: 临时缓冲区路径，默认为'~/buffer'
    # 功能：
    #   1. 创建缓冲区目录（如果不存在）
    #   2. 初始化内部路径变量
    #   3. 配置omegapy模块的路径设置
    # 注意：
    #   - omegapy.set_omega_bin_path() 设置原始数据路径
    #   - omegapy.set_omega_py_path() 设置处理后数据路径（指向'processed'子目录）
    def __init__(self):
        
        
        self._base_bin_path = config.bin_path
        self._base_py_path = config.py_path
        self._base_buffer_path = config.buffer_path
        if(os.path.exists(self._base_buffer_path) == False):
            os.mkdir(self._base_buffer_path)
        od.set_omega_bin_path(self._base_bin_path)
        pkl_path = os.path.join(self._base_py_path, 'processed')
        od.set_omega_py_path(pkl_path)  #此处存在语义不清
    # 获取指定类型的数据路径
    # 参数：
    #   type: 路径类型，必须是 'bin'（原始二进制数据）、'py'（Python处理数据）或 'buffer'（临时缓冲区）之一
    # 返回值：对应类型的路径字符串
    # 异常：如果type参数无效，则抛出ValueError异常
    def get_path(self, type: Literal['bin', 'py', 'buffer']):
        if(type == 'bin'):
            return self._base_bin_path
        elif(type == 'py'):
            return self._base_py_path
        elif(type == 'buffer'):
            return self._base_buffer_path
        else:
            raise ValueError('Invalid type')
    
    # 设置指定类型的数据路径
    # 参数：
    #   type: 路径类型，必须是 'bin'、'py' 或 'buffer' 之一
    #   path: 新的路径字符串
    # 功能：
    #   - 更新内部路径变量
    #   - 同步更新omegapy模块的相应路径配置
    # 注意：
    #   - 对于'py'类型，omegapy路径设置指向'processed'子目录
    #   - 对于'bin'类型，直接设置为指定路径
    # 异常：如果type参数无效，则抛出ValueError异常
    def set_path(self, type: Literal['bin', 'py', 'buffer'], path:str):
        if(type == 'bin'):
            self._base_bin_path = path
            od.set_omega_bin_path(path)
        elif(type == 'py'):
            self._base_py_path = path  
            pkl_path = os.path.join(path, 'processed')
            od.set_omega_py_path(pkl_path)  #此处存在语义不清
        elif(type == 'buffer'):
            self._base_buffer_path = path
        else:
            raise ValueError('Invalid type')
    # 主要数据加载方法：根据指定类型加载OMEGA立方体数据
    # 参数：
    #   cube_name: 字符串，立方体名称（如 '0982_3'）
    #   type: 数据类型，必须是 'raw'（原始二进制数据）、'processed'（处理后的Python数据）或 'predicted'（预测结果）之一
    # 返回值：加载的数据对象（具体类型取决于type参数）
    # 功能流程：
    #   1. 根据type参数确定源文件路径
    #   2. 验证源文件存在性
    #   3. 将源文件复制到临时缓冲区
    #   4. 调用对应的私有加载方法处理数据
    #   5. 清理临时缓冲区文件
    # 异常处理：
    #   - 如果type参数无效，抛出ValueError
    #   - 如果源文件不存在，抛出FileNotFoundError
    #   - 其他任何异常都会被捕获并重新抛出，同时记录错误日志
    def load(self, cube_name:str, type: Literal['raw', 'processed', 'predicted']):
        import shutil
        import os
        
        # 记录开始
        log('CubeIO.load', f'开始加载立方体 {cube_name}，类型 {type}', 'INFO')
        
        try:
            # 1. 确定源文件路径（具体名称稍后处理）
            # 这里需要根据 type 确定源路径，暂时使用简单逻辑
            if type == 'raw':
                source_path = os.path.join(self._base_bin_path, f'ORB{cube_name}_DATA')
            elif type == 'processed':
                source_path = os.path.join(self._base_py_path,'processed', f'{cube_name}_processed.pkl')
            elif type == 'predicted':
                source_path = os.path.join(self._base_py_path,'predicted', f'{cube_name}_predicted.pkz')
            else:
                raise ValueError(f'无效的类型: {type}')
            
            # 检查源文件是否存在
            if not os.path.exists(source_path):
                raise FileNotFoundError(f'源文件不存在: {source_path}')
            
            # 2. 复制到缓冲区
            buffer_target = os.path.join(self._base_buffer_path, os.path.basename(source_path))
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
    # 参数：
    #   cube_name: 字符串，立方体名称（如 '0982_3'）
    # 返回值：OMEGAdata对象，包含原始OMEGA数据
    # 功能：调用omegapy模块的OMEGAdata类创建数据对象
    def _load_raw(self,cube_name:str):
        log('CubeIO._load_raw', f'开始加载立方体 {cube_name}，类型 raw', 'DEBUG')
        ret_omega_data=od.OMEGAdata(cube_name,disp=False)
        return ret_omega_data
    
    # 私有方法：加载处理后的Python数据
    # 参数：
    #   buffer_target: 字符串，缓冲区中处理后数据文件的完整路径
    # 返回值：OMEGAdata对象，包含处理后的OMEGA数据
    # 功能：调用omegapy模块的load_omega函数加载pkl格式的数据
    def _load_processed(self,buffer_target:str):
        log('CubeIO._load_processed', f'开始加载立方体 {buffer_target}，类型 processed', 'DEBUG')
        ret_omega_data=od.load_omega(buffer_target)
        return ret_omega_data
    
    # 私有方法：加载预测结果数据
    # 参数：
    #   buffer_target: 字符串，缓冲区中预测结果文件的完整路径
    # 返回值：Predicted对象，包含预测结果数据
    # 功能：创建Predicted类实例（当前为占位实现，需根据实际预测模块完善）
    # 注意：此方法当前为临时实现，需要根据实际的predict模块进行完善
    def _load_predicted(self, buffer_target:str):
        log('CubeIO._load_predicted', f'开始加载立方体 {buffer_target}，类型 predicted', 'DEBUG')
        ret4test=predict.Predicted() ##暂时留空
        return ret4test
# 主程序入口：用于测试CubeIO类功能的示例代码
# 当直接运行此文件时执行，不作为模块导入时跳过
if __name__ == '__main__':
    # 设置日志级别为DEBUG，显示详细调试信息
    log_level = 'DEBUG'
    # 创建CubeIO实例，配置数据路径
    # 参数：
    #   bin_path: 原始二进制数据目录（D:\Project\Clairvoyant-data\bin）
    #   py_path: Python处理数据目录（D:\Project\Clairvoyant-data\py）
    #   buffer_path: 临时缓冲区路径（D:\Project\Clairvoyant-buffer）
    cubeio=CubeIO()
    # 加载指定名称的原始立方体数据
    # cube_name: '0982_3' - 示例立方体名称
    # type: 'raw' - 加载原始二进制数据
    res = cubeio.load(cube_name='0982_3', type='raw')
    # 使用omegapy的交互式可视化函数显示加载的数据
    op.show_omega_interactif_v2(res)
    # 等待用户按回车键继续，防止控制台窗口关闭
    input("Press Enter to continue...")
    
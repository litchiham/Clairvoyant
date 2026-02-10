import omegapy.omega_data as od
import omegapy.omega_plots as op
import os
import time
import tkinter.messagebox as messagebox
import predict
from typing import Literal



log_level = 'DEBUG'
_log_levels={
    'DEBUG':0,
    'INFO':1,
    'WARNING':2,
    'ERROR':3
}
def log(source:str,message:str, type: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'], show_in_window=False):
    if(_log_levels[type] >= _log_levels[log_level]):
        time_stamp = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime())
        log_str = f"{time_stamp} [{type}] @{source}: {message}"
        print(log_str)
        if(show_in_window):
            #此处有问题，messagebox可能阻塞线程
            messagebox.showinfo(title=type, message=message)

class CubeIO:
    _base_bin_path = ''
    _base_py_path = ''
    _base_buffer_path = '~/buffer'
    def __init__(self, bin_path:str, py_path:str, buffer_path='~/buffer'):
        if(os.path.exists(buffer_path) == False):
            os.mkdir(buffer_path)
        
        self._base_bin_path = bin_path
        self._base_py_path = py_path
        self._base_buffer_path = buffer_path
        od.set_omega_bin_path(bin_path)
        pkl_path = os.path.join(py_path, 'processed')
        od.set_omega_py_path(pkl_path)  #此处存在语义不清
    def get_path(self, type: Literal['bin', 'py', 'buffer']):
        if(type == 'bin'):
            return self._base_bin_path
        elif(type == 'py'):
            return self._base_py_path
        elif(type == 'buffer'):
            return self._base_buffer_path
        else:
            raise ValueError('Invalid type')
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
    def _load_raw(self,cube_name:str):
        log('CubeIO._load_raw', f'开始加载立方体 {cube_name}，类型 raw', 'DEBUG')
        ret_omega_data=od.OMEGAdata(cube_name)
        return ret_omega_data
    def _load_processed(self,buffer_target:str):
        log('CubeIO._load_processed', f'开始加载立方体 {buffer_target}，类型 processed', 'DEBUG')
        ret_omega_data=od.load_omega(buffer_target)
        return ret_omega_data
    def _load_predicted(self, buffer_target:str):
        log('CubeIO._load_predicted', f'开始加载立方体 {buffer_target}，类型 predicted', 'DEBUG')
        ret4test=predict.Predicted() ##暂时留空
        return ret4test
if __name__ == '__main__':
    log_level = 'DEBUG'
    cubeio=CubeIO(r"D:\Project\Clairvoyant-data\bin", r"D:\Project\Clairvoyant-data\py", r"D:\Project\Clairvoyant-buffer")
    res = cubeio.load(cube_name='0982_3', type='raw')
    op.show_omega_interactif_v2(res)
    input("Press Enter to continue...")
    
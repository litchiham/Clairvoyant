# Clairvoyant-Omega
## 功能
1. 预处理->预测
2. 可视化
## 数据输入
1. 原始cube
2. 预处理后的*.pkl
3. 预测后的*.npz
## 数据输出
1. 预处理后的*.pkl
2. 预测后的*.npz/其他格式
## 结构
引用关系
GUI->Process平行Predict->CubeIO

```python
    #cubeio.py
    log_level = 'DEBUG'
    def log(source:str, type=['DEBUG','INFO','WARN','ERROR'],message):
        '''按格式打印日志'''
    class CubeIO:
        '''用于读取和保存数据，避免内存占用过大'''
        base_bin_path = ''  #raw data
        base_py_path = '' #*.pkl/*.npz
        '''
        base_bin_path
          -ORBXXXX
            -*.QUB
            -*.NAV
        base_py_path
          -processed
            -ORBXXXX_processed.pkl
          -predicted
            -ORBXXXX_predicted.npz
        '''
        def set_base_py_path(self, path):
        def get_base_bin_path(self):
        def load_raw(self, cube_name)->OmegaData:
        def load_processed(self, cube_name)->OmegaData:
        def load_predicted(self, cube_name)->Predicted:
        def save_processed(self, cube_name, cube:OmegaData):
        def save_predicted(self, cube_name, predicted:Predicted):    
    #process.py
    class Process:
        raw_data_paths = [] #cube_names
        def import_cubes(self, ['filepath'])->bool:
        '''向raw_data_list中导入路径'''
        def process_cubes(self, save=True)->OMEGAData:
        '''使用CubeIO接口'''
        '''注意与GUI协调多线程,信号/槽实现进度条'''
        '''执行预处理逻辑'''
    #predict.py
    class Predicted:
        '''存储和处理一个cube的预测结果'''
        points=np.array([{'lon','lat','?'}])
    class Predict:
        Processed_data_paths = []
        def import_cubes(self, ['filepath'])->bool:
        '''向raw_data_list中导入路径'''
        def predict_cubes(self, save=True, save_as=['npz'])->bool:
        '''使用CubeIO接口'''
        '''注意与GUI协调多线程,信号/槽实现进度条'''
        '''执行预处理逻辑'''
    #gui.py
    class GUI_main:
        '''主界面GUI'''
        '''需要商榷'''
    '''建议GUI_preprocess和GUI_predict继承某个类，避免重复代码'''
    class GUI_preprocess:
        '''预处理结果可视化界面GUI'''
        '''需要商榷'''
    class GUI_predict:
        '''预测结果可视化界面GUI'''
        '''需要商榷'''
```
## 错误处理:
### 错误通知:
`messagebox.showerror()`同时使用`log(source, type='ERROR',message)`

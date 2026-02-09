import os
os.environ["OMEGA_BIN_PATH"] = r"D:\PlanetaryDataSystem\OMEGA\bin_test"
os.environ["OMEGA_PY_PATH"] = r"D:\PlanetaryDataSystem\OMEGA\py_test" #这个似乎没用

import time
import glob
import numpy as np
import xarray as xr
import omegapy
import omegapy.omega_data as od
import math
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegapy.omega_plots import show_omega_interactif_v2
import traceback

# ====== 配置路径 ======
DATA_ROOT = r"D:\PlanetaryDataSystem\OMEGA\bin_test"   # 每个子文件夹都有 .nav/.qub 和 reference_files
DUST_PATH = r"D:\PlanetaryDataSystem\krigedcdod_v2\*.nc"  # 所有 dust nc 文件
OUTPUT_DIR_out =r"D:\PlanetaryDataSystem\OMEGA\py_test"  
LOG_FILE = os.path.join(OUTPUT_DIR_out, "process_errors.log")
os.makedirs(OUTPUT_DIR_out, exist_ok=True)
H = 11.0
MAX_LAM=2.4
MIN_LAM=0.85

total_cubes=0 #记录总个数用于测试

# ====== 错误日志 ======
def log_error(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()}: {msg}\n")
# ====== 预加载 dust 数据，并统一坐标 ======
dust_files = sorted(glob.glob(DUST_PATH))
dust_datasets = []

for f in dust_files:
    try:
        ds = xr.open_dataset(f)

        # 统一坐标名
        ds = ds.rename({'Time': 'sol', 'latitude': 'lat', 'longitude': 'lon'})
        dust_datasets.append(ds)

    except Exception as e:
        log_error(f"Failed to load {f}: {e}")

print(f"Loaded {len(dust_datasets)} dust datasets.")


def slice_cube(cube):
    try:
        mask = (cube.lam >= MIN_LAM) & (cube.lam <= MAX_LAM) #创建布尔掩码
        indices = np.where(mask)[0] #获取整数索引
        print("原始lam数目", len(cube.lam))
        print("切片后数目",len(indices))
        if len(indices) == 0:
            return None # 如果没有落在这个范围内的波段，返回None

        # 更新波长轴
        cube.lam = cube.lam[indices]
        
        # 更新数据轴 (通常是 [x, y, lambda])
        if hasattr(cube, 'cube_rf') and cube.cube_rf is not None:
            cube.cube_rf = cube.cube_rf[:, :, indices]
        
        if hasattr(cube, 'cube_i') and cube.cube_i is not None:
            cube.cube_i = cube.cube_i[:, :, indices]
            
        # 如果有其他关联的光谱维度属性（如不确定度等），也需在此切片
        return cube
    except Exception as e:
        log_error(f"Error during spectral slicing: {e}")
        return cube


# ====== 单个 cube 处理 ======
def process_cube(qub_path):
    try:
        # 从路径中提取观测名称
        base_name = os.path.basename(qub_path)
        name = base_name.split('.')[0][-6:]
        
        print(f"Processing {name}")
        


        # 加载OMEGA数据
        cube = od.OMEGAdata(name)
        # slice_cube(cube=cube)
        print(f"Cube RF Shape: {cube.cube_rf.shape}")
        print(f"Wavelengths Length: {len(cube.lam)}")


        cube_corr_atm = od.corr_atm(cube)
        

        show_omega_interactif_v2(cube_corr_atm)
        input()
        cube_corr_therm_atm=od.corr_therm_atm(cube, npool=2)
        show_omega_interactif_v2(cube_corr_therm_atm)
        input()
        # #cube_corr=cube

        # #计算残差
        # mask = (cube.lam >= 0.85) & (cube.lam <= 2.4)
        # wvl_target = cube.lam[mask]
        # data_thermal = cube_corr_therm_atm.cube_rf[:, :, mask]
        # data_simple = cube_corr_atm.cube_rf[:, :, mask]
        # diff=(data_thermal-data_simple)
        # mean_diff = np.nanmean(diff, axis=(0, 1))  # 随波长变化的平均误差

        # #可视化
        # plt.figure(figsize=(10, 5))
        # plt.plot(wvl_target, mean_diff)
        # plt.title("Difference between Thermal and Simple Correction (0.85-2.4um)")
        # plt.xlabel("Wavelength (um)")
        # plt.ylabel("Absolute Reflectance Difference")
        # plt.grid(True)
        # plt.show()

        # print(f"Max diff 在指定波长区间为: {np.nanmax(mean_diff):.5f}")
        # input()
        # print(np.nanmean(cube.cube_rf,(0,1)))

        out_name = os.path.join(OUTPUT_DIR_out, f"{name}_processed.pkl")
        # 使用omegapy的保存函数
        od.save_omega(cube_corr_atm, savname = out_name)
        
        print(f"{name} is saved")
        
        return out_name
    except Exception as e:
        error_msg = f"Error processing {qub_path}: {str(e)}"
        print(error_msg)
        log_error(error_msg)
        traceback.print_exc()
        return None

# ====== 主流程 ======
def process_all_parallel(max_workers=1):
    folders = [f for f in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(f)]
    print(f"Found {len(folders)} OMEGA folders")

    all_cubes = []

    for folder in folders:
        # print(f"Processing folder: {folder}")
        qub_files = glob.glob(os.path.join(folder, "*.QUB"))
        
        if not qub_files:
            continue

        for qub in qub_files:
            all_cubes.append(qub)

    global total_cubes
    total_cubes=len(all_cubes) #记录总数

    for cube in all_cubes:
        process_cube(cube)


time_list=[]
ave_list=[]
worker_list=[]
if __name__ == "__main__":
    process_all_parallel(max_workers=1) #设置并行数
        



'''
    更新日志：
    v2: 修复多线程，修复
    v3_test:研究并行数关系，disabled判定，只校准。
'''


'''
    测试日志：
    1. 小规模测试：失败。多线程导致广播错误，解决方案：改用多进程，尝试
    2. 小规模测试：成功。26轨道，并行数4，总用时23:35，平均0.90min/个

    3.(v3_test)：10个轨道，用于测试4-10的平均用时。全部校准omitfyomijbnuy
    结果：（进程数，平均用时（秒），总用时）
        4     64.92670373916626     649.2670373916626
        5     78.1880782365799     781.880782365799
        6     69.7449359178543     697.4493591785431
        7     65.08731553554534     650.8731553554535
        8     59.78385808467865     597.8385808467865
        9     59.11768312454224     591.1768312454224
'''
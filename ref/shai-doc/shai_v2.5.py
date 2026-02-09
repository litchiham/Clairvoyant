import os
os.environ["OMEGA_BIN_PATH"] = r"G:\Omega0"
os.environ["OMEGA_PY_PATH"] = r"G:\Omega0-py" #这个似乎没用

import glob
import numpy as np
import xarray as xr
import omegapy
import omegapy.omega_data as od
import math
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback

# ====== 配置路径 ======
DATA_ROOT = r"G:\Omega0"   # 每个子文件夹都有 .nav/.qub 和 reference_files
DUST_PATH = r"G:\krigedcdod_v2\*.nc"  # 所有 dust nc 文件
OUTPUT_DIR_No = r"G:\Omega0-py\Omega_after_no"    #不需要处理个别像素的文件
OUTPUT_DIR_clean_cube =r"G:\Omega0-py\Omega_after_yes"    #已经经过处理个别像素的文件
OUTPUT_DIR_rejected=r"G:\Omega0-py\Omega_after_rejected"  
OUTPUT_DIR_atm=r"G:\Omega0-py\Omega_after_atm"  
LOG_FILE = os.path.join(OUTPUT_DIR_No, "process_errors.log")
os.makedirs(OUTPUT_DIR_No, exist_ok=True)
os.makedirs(OUTPUT_DIR_clean_cube, exist_ok=True)
os.makedirs(OUTPUT_DIR_rejected, exist_ok=True)
os.makedirs(OUTPUT_DIR_atm, exist_ok=True)
H = 11.0

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

# ====== UTC -> Sol ======
def utc_to_sol(utc_time):
    """
    参数:
    utc_time: datetime对象，表示UTC时间
    
    返回:
    字典，包含:
      - mars_year: 火星年
      - mars_sol: 火星日(从0开始)
      - fractional_sol: 火星日的小数部分(表示时间)
    """
    # 火星公元纪元开始时间 (Mars Year 1, Sol 0)
    mars_epoch = datetime(1955, 4, 11, 11, 4, 0)
    
    # 计算时间差(以天为单位)
    time_difference = utc_time - mars_epoch
    earth_days = time_difference.total_seconds() / 86400.0
    
    # 火星日与地球日的比率
    mars_sol_ratio = 1.02749125
    
    # 计算总火星日数
    total_mars_sols = earth_days / mars_sol_ratio
    
    # 火星年平均长度(火星日)
    mars_year_length = 668.5991
    
    # 计算火星年
    mars_year = total_mars_sols / mars_year_length
    mars_year_int = int(mars_year) + 1  # 火星年从1开始
    
    # 计算当前火星年中的火星日
    sol_in_year = (mars_year - int(mars_year)) * mars_year_length
    sol_int = int(sol_in_year)
    fractional_sol = sol_in_year - sol_int
    
    return {
        "mars_year": mars_year_int,
        "mars_sol": sol_int,
        "fractional_sol": fractional_sol
    }

# ====== dust 插值 ======
def get_tau(sol, lat, lon, varname='cdodtot'):
    for i in range(0, len(dust_datasets)):
        if str(sol['mars_year']) in dust_files[i]:
            ds = dust_datasets[i]
            if sol['mars_sol'] >= ds.sol.min() and sol['mars_sol'] <= ds.sol.max():
                try:
                    val = ds[varname].interp(sol=sol['mars_sol'], lat=lat, lon=lon)
                    return float(val)
                except Exception as e:
                    log_error(f"Failed interpolation for sol={sol}, lat={lat}, lon={lon}: {e}")
    return None

# ====== cube 筛选 ======
def cube_passes_filters(cube, name):
    try:
        # 获取平均几何条件
        inc = np.nanmean(cube.inci)
        emg = np.nanmean(cube.emer)
        alt = np.nanmean(cube.alt)
        lat = np.nanmean(cube.lat)
        lon = np.nanmean(cube.lon)
        
        # 获取UTC时间字符串
        utc_str = cube.utc.strftime("%Y-%m-%d %H:%M:%S")
        utc_time = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S")
        sol = utc_to_sol(utc_time)

        if emg >= 15 or inc >= 75:
            print(f"{name} is passed by first")
            add_rejected_record(name=name, reason="first")
            return False

        tau = get_tau(sol, lat, lon)
        if tau is None:
            print(f"{name} is passed notau")
            add_rejected_record(name=name, reason="notau")
            return False

        dust_crit = tau * np.exp(alt / H) * (1 + 1 / np.cos(np.radians(inc)))
        if dust_crit >= 2.0:
            print(f"{name} is passed by second")
            add_rejected_record(name=name, reason="second")
            return False
        
        

        return True
    except Exception as e:
        log_error(f"cube_passes_filters error: {e}")
        return False

# ====== 对于被剔除的文件记录在案，避免重复操作
def add_rejected_record(name, reason):
    expected_rejected = os.path.join(OUTPUT_DIR_rejected, f"{name}_rejected.txt")
    with open(expected_rejected, "w") as f:
                f.write(f"Timestamp: {datetime.now()}\nReason: {reason}\n")
                print(f"{name} rejected: {reason}")
                return expected_rejected

# ====== 清理 cube ======
def clean_cube(cube):
    try:
        # 创建一个掩码来标识有效像素
        valid_mask = np.ones(cube.cube_rf.shape[:2], dtype=bool)
        
        key = 0
        # 检查全零或全一的帧
        for i in range(cube.cube_rf.shape[0]):
            for j in range(cube.cube_rf.shape[1]):
                spectrum = cube.cube_rf[i, j]
                if np.all(spectrum == 0) or np.all(spectrum == 1):
                    valid_mask[i, j] = False
                    key = 1
        
        # 应用掩码
        for attr in ['cube_rf', 'cube_i', 'lat', 'lon', 'alt', 'inci', 'emer']:
            if hasattr(cube, attr):
                data = getattr(cube, attr)
                if data is not None:
                    # 对于3D数据（光谱立方体）
                    if len(data.shape) == 3:
                        for k in range(data.shape[2]):
                            data_slice = data[:, :, k]
                            data_slice[~valid_mask] = np.nan
                    # 对于2D数据（几何数据）
                    elif len(data.shape) == 2:
                        data[~valid_mask] = np.nan
                    setattr(cube, attr, data)

        return cube, key
    except Exception as e:
        log_error(f"clean_cube error: {e}")
        return cube

# ====== 单个 cube 处理 ======
def process_cube(qub_path):
    try:
        # 从路径中提取观测名称
        base_name = os.path.basename(qub_path)
        name = base_name.split('.')[0][-6:]
        
        print(f"Processing {name}")
        
        # ====== 新增：检测是否已经校准/处理过 ======
        expected_out_no = os.path.join(OUTPUT_DIR_No, f"{name}_processed.pkl")
        expected_out_clean = os.path.join(OUTPUT_DIR_clean_cube, f"{name}_processed.pkl")
        expected_out_rejected = os.path.join(OUTPUT_DIR_rejected, f"{name}_rejected.txt")
        
        if os.path.exists(expected_out_no):
            print(f"Skipping {name}: Already processed (found in OUTPUT_DIR_No)")
            return expected_out_no
        
        if os.path.exists(expected_out_clean):
            print(f"Skipping {name}: Already processed (found in OUTPUT_DIR_clean_cube)")
            return expected_out_clean
        
        if os.path.exists(expected_out_rejected):
            print(f"Skipping {name}: Already processed (found in OUTPUT_DIR_Rejected)")
            return expected_out_rejected
        


        # 加载OMEGA数据
        cube = od.OMEGAdata(name)
        
        if not cube_passes_filters(cube, name):
            return None

        # 检查是否还有有效数据
        if np.all(np.isnan(cube.cube_rf)):
            print("All NAN")
            add_rejected_record(name=name, reason='all-nan')
            return None
            
        
        cube_corr = od.corr_therm_atm(cube, npool=1)
        # cube_corr=cube #测试

        cube_corr, key = clean_cube(cube_corr)

        if np.all(np.isnan(cube_corr.cube_rf)):
            print("All NAN after therm_atm")
            cube_corr = od.corr_atm(cube)
            cube_corr, _ = clean_cube(cube_corr)
            if np.all(np.isnan(cube_corr.cube_rf)):
                print("Still all NAN after atm")
                return None
            key = 2

        if key == 1:
            out_name = os.path.join(OUTPUT_DIR_clean_cube, f"{name}_processed.pkl")
        elif key == 2:
            out_name = os.path.join(OUTPUT_DIR_atm, f"{name}_processed.pkl")
        else:
            out_name = os.path.join(OUTPUT_DIR_No, f"{name}_processed.pkl")
        
        

        # 使用omegapy的保存函数
        od.save_omega(cube_corr, savname = out_name)
        
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_cube, qub): qub for qub in all_cubes}

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing cubes"):
            result = f.result()
            if result:
                print(f"Successfully processed: {result}")

if __name__ == "__main__":
    process_all_parallel(max_workers=8) #设置并行数



'''
    更新日志：
    v2: 修复多线程，添加中断机制
'''


'''
    测试日志：
    1. 小规模测试：失败。多线程导致广播错误，解决方案：改用多进程，尝试
    2. 小规模测试：成功。26轨道，并行数4，总用时23:35，平均0.90min/个
'''
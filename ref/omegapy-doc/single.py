import os
os.environ["OMEGA_BIN_PATH"] = r"H:\Omega1"
os.environ["OMEGA_PY_PATH"] = r"G:\ANACONDA\envs\general\lib\site-packages\omegapy"

import glob
import numpy as np
import xarray as xr
import omegapy
import omegapy.omega_data as od
import math
from datetime import datetime, timedelta
import traceback

# ====== 配置路径 ======
DATA_ROOT = "H:/Omega1"   # 每个子文件夹都有 .nav/.qub 和 reference_files
DUST_PATH = "H:/krigedcdod_v2/*.nc"  # 所有 dust nc 文件
OUTPUT_DIR = "S:/Omega_after"
LOG_FILE = os.path.join(OUTPUT_DIR, "process_errors.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            print(sol['mars_year'])
            ds = dust_datasets[i]
            print(ds.sol.min(), ds.sol.max())
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
            return False

        tau = get_tau(sol, lat, lon)
        if tau is None:
            print(f"{name} is passed notau")
            return False

        dust_crit = tau * np.exp(alt / H) * (1 + 1 / np.cos(np.radians(inc)))
        if dust_crit >= 2.0:
            print(f"{name} is passed by second")
            return False

        return True
    except Exception as e:
        log_error(f"cube_passes_filters error: {e}")
        return False

# ====== 清理 cube ======
def clean_cube(cube):
    try:
        # 创建一个掩码来标识有效像素
        valid_mask = np.ones(cube.cube_rf.shape[:2], dtype=bool)
        
        # 检查全零或全一的帧
        for i in range(cube.cube_rf.shape[0]):
            for j in range(cube.cube_rf.shape[1]):
                spectrum = cube.cube_rf[i, j]
                if np.all(spectrum == 0) or np.all(spectrum == 1):
                    valid_mask[i, j] = False
        
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
        
        return cube
    except Exception as e:
        log_error(f"clean_cube error: {e}")
        return cube

# ====== 单个 cube 处理 ======
def process_cube(qub_path):
    try:
        # 从路径中提取观测名称
        name = qub_path
        
        print(f"Processing {name}")
        
        # 加载OMEGA数据
        cube = od.OMEGAdata(name)
        
        if not cube_passes_filters(cube, name):
            return None

        cube = clean_cube(cube)
        
        # 检查是否还有有效数据
        if np.all(np.isnan(cube.cube_rf)):
            return None

        out_name = os.path.join(OUTPUT_DIR, f"{name}_processed.pkl")
        
        # 使用omegapy的保存函数
        od.save_omega(cube, savname = out_name)
        
        print(f"{name} is saved")
        
        return out_name
    except Exception as e:
        error_msg = f"Error processing {qub_path}: {str(e)}"
        print(error_msg)
        log_error(error_msg)
        traceback.print_exc()
        return None

# ====== 主流程 ======
def process_all_single_thread():
    
    result = process_cube("0041_2")
    if result:
        print(f"Successfully processed: {result}")

if __name__ == "__main__":
    # 处理数据
    process_all_single_thread()

import os
# 初始路径设置 (主进程)
os.environ["OMEGA_BIN_PATH"] = r"G:\Omega4"
os.environ["OMEGA_PY_PATH"] = r"G:\Omega4-py"

import glob
import numpy as np
import xarray as xr
import omegapy
import omegapy.omega_data as od
import math
import shutil
import time
import subprocess
import multiprocessing
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import traceback

# ====== 配置路径 (根据您的指令更新) ======
DATA_ROOT = r"G:\Omega4"              # 源数据根目录 (HDD)
BUFFER_DIR = r"D:\Omega4-buffer"      # 文件缓冲区地址 (SSD)
DUST_PATH = r"G:\krigedcdod_v2\*.nc"  # 所有 dust nc 文件   建议改为考到自己的盘上，速度快一点
OUTPUT_DIR_No = r"G:\Omega2-py\Omega_after_no"
OUTPUT_DIR_clean_cube = r"G:\Omega2-py\Omega_after_yes"
OUTPUT_DIR_rejected = r"G:\Omega2-py\Omega_after_rejected"
LOG_FILE = os.path.join(OUTPUT_DIR_No, "process_errors.log") #错误日志
PROCESSED_LOG = os.path.join(BUFFER_DIR, "processed_filename_list.log") #已处理列表日志

# 确保目录存在
os.makedirs(OUTPUT_DIR_No, exist_ok=True)
os.makedirs(OUTPUT_DIR_clean_cube, exist_ok=True)
os.makedirs(OUTPUT_DIR_rejected, exist_ok=True)
os.makedirs(BUFFER_DIR, exist_ok=True)

H = 11.0


# ====== 日志管理函数 ======
def get_processed_folders():
    """读取已处理成功的轨道列表"""
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_folder_as_processed(folder_name):
    """记录处理成功的轨道名"""
    with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{folder_name}\n")



# ====== 错误日志与核心逻辑======
def log_error(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()}: {msg}\n")

# 预加载 dust 数据 (由各子进程独立加载或主进程传递)
dust_files = sorted(glob.glob(DUST_PATH))
dust_datasets = []
for f in dust_files:
    try:
        ds = xr.open_dataset(f)
        ds = ds.rename({'Time': 'sol', 'latitude': 'lat', 'longitude': 'lon'})
        dust_datasets.append(ds)
    except Exception as e:
        log_error(f"Failed to load {f}: {e}")

def utc_to_sol(utc_time):
    mars_epoch = datetime(1955, 4, 11, 11, 4, 0)
    time_difference = utc_time - mars_epoch
    earth_days = time_difference.total_seconds() / 86400.0
    mars_sol_ratio = 1.02749125
    total_mars_sols = earth_days / mars_sol_ratio
    mars_year_length = 668.5991
    mars_year = total_mars_sols / mars_year_length
    mars_year_int = int(mars_year) + 1
    sol_in_year = (mars_year - int(mars_year)) * mars_year_length
    sol_int = int(sol_in_year)
    fractional_sol = sol_in_year - sol_int
    return {"mars_year": mars_year_int, "mars_sol": sol_int, "fractional_sol": fractional_sol}

def get_tau(sol, lat, lon, varname='cdodtot'):
    for i in range(len(dust_datasets)):
        if str(sol['mars_year']) in dust_files[i]:
            ds = dust_datasets[i]
            if ds.sol.min() <= sol['mars_sol'] <= ds.sol.max():
                try:
                    val = ds[varname].interp(sol=sol['mars_sol'], lat=lat, lon=lon)
                    return float(val)
                except: return None
    return None

def cube_passes_filters(cube, name):
    try:
        inc, emg, alt = np.nanmean(cube.inci), np.nanmean(cube.emer), np.nanmean(cube.alt)
        lat, lon = np.nanmean(cube.lat), np.nanmean(cube.lon)
        utc_time = datetime.strptime(cube.utc.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
        sol = utc_to_sol(utc_time)
        if emg >= 15 or inc >= 75:
            add_rejected_record(name, "first"); return False
        tau = get_tau(sol, lat, lon)
        if tau is None:
            add_rejected_record(name, "notau"); return False
        if (tau * np.exp(alt / H) * (1 + 1 / np.cos(np.radians(inc)))) >= 2.0:
            add_rejected_record(name, "second"); return False
        return True
    except: return False

def add_rejected_record(name, reason):
    expected_rejected = os.path.join(OUTPUT_DIR_rejected, f"{name}_rejected.txt")
    with open(expected_rejected, "w") as f:
        f.write(f"Timestamp: {datetime.now()}\nReason: {reason}\n")

def clean_cube(cube):
    valid_mask = np.ones(cube.cube_rf.shape[:2], dtype=bool)
    key = 0
    for i in range(cube.cube_rf.shape[0]):
        for j in range(cube.cube_rf.shape[1]):
            spectrum = cube.cube_rf[i, j]
            if np.all(spectrum == 0) or np.all(spectrum == 1):
                valid_mask[i, j] = False; key = 1
    for attr in ['cube_rf', 'cube_i', 'lat', 'lon', 'alt', 'inci', 'emer']:
        if hasattr(cube, attr):
            data = getattr(cube, attr)
            if data is not None:
                if len(data.shape) == 3:
                    for k in range(data.shape[2]):
                        data_slice = data[:, :, k]
                        data_slice[~valid_mask] = np.nan
                elif len(data.shape) == 2:
                    data[~valid_mask] = np.nan
                setattr(cube, attr, data)
    return cube, key

# ====== 缓冲区管理进程 (Windows高速拷贝) ======
def buffer_manager_process(folders_to_copy, max_workers):
    """
    独立进程：监控缓冲区文件夹数量。
    如果数量 < max_workers，则从 F 盘搬运下一个文件夹到 D 盘缓冲区。
    """
    copied_count = 0
    total = len(folders_to_copy)
    
    while copied_count < total:
        # 统计缓冲区当前文件夹数量
        current_buffers = [d for d in os.listdir(BUFFER_DIR) if os.path.isdir(os.path.join(BUFFER_DIR, d))]
        
        if len(current_buffers) < max_workers + 2:
            src = folders_to_copy[copied_count]
            folder_name = os.path.basename(src)
            dst = os.path.join(BUFFER_DIR, folder_name)
            
            if not os.path.exists(dst):
                # 使用 Windows robocopy 高速拷贝
                subprocess.run(["robocopy", src, dst, "/E", "/MT:16", "/NFL", "/NDL", "/NJH", "/NJS"], shell=True)
            
            copied_count += 1
        else:
            time.sleep(2) # 缓冲区满了，歇会儿

# ====== 计算任务进程 ======
def process_folder_task(folder_name):
    """
    在子进程中执行。folder_name 仅为文件夹名。
    从 BUFFER_DIR 读取。
    """
    try:
        # 修改子进程环境变量指向缓冲区
        os.environ["OMEGA_BIN_PATH"] = BUFFER_DIR
        buffer_path = os.path.join(BUFFER_DIR, folder_name)
        
        qub_files = glob.glob(os.path.join(buffer_path, "*.QUB"))
        for qub_path in qub_files:
            name = os.path.basename(qub_path).split('.')[0][-6:]
            
            # 检测是否已存在
            if os.path.exists(os.path.join(OUTPUT_DIR_No, f"{name}_processed.pkl")) or \
               os.path.exists(os.path.join(OUTPUT_DIR_clean_cube, f"{name}_processed.pkl")) or \
               os.path.exists(os.path.join(OUTPUT_DIR_rejected, f"{name}_rejected.txt")):
                print(f"DADA {folder_name} passed because 它被处理过了")
                continue

            # 使用 omegapy 加载 (此时会从 BUFFER_DIR 找)
            cube = od.OMEGAdata(name)
            if not cube_passes_filters(cube, name):
                mark_folder_as_processed(folder_name)   #mark已经处理过了
                continue

            cube_corr = od.corr_atm(cube) ###
            cube_final, key = clean_cube(cube_corr)
            
            if np.all(np.isnan(cube_final.cube_rf)): continue

            out_dir = OUTPUT_DIR_clean_cube if key == 1 else OUTPUT_DIR_No
            od.save_omega(cube_final, savname=os.path.join(out_dir, f"{name}_processed.pkl"))
            mark_folder_as_processed(folder_name)   #mark已经处理过了
        
        return folder_name
    except Exception as e:
        log_error(f"Process Error {folder_name}: {traceback.format_exc()}")
        return None
    finally:
        # 处理完毕，立即删除缓冲区文件夹
        target = os.path.join(BUFFER_DIR, folder_name)
        if os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True)

# ====== 主流程 ======
def process_all_parallel(max_workers=4):
    all_folders = [f for f in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(f)]
    print(f"Total Folders: {len(all_folders)}")

    # 1. 开启缓冲区管理进程
    bg_buffer = multiprocessing.Process(target=buffer_manager_process, args=(all_folders, max_workers))
    bg_buffer.daemon = True
    bg_buffer.start()

    # 2. 调度计算进程
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        pending_folders = all_folders.copy()
        
        pbar = tqdm(total=len(all_folders), desc="Overall Progress")
        
        while pending_folders or futures:
            # 检查是否有就绪的缓冲区文件夹
            if pending_folders and len(futures) < max_workers:
                next_f = pending_folders[0]
                folder_name = os.path.basename(next_f)
                buf_path = os.path.join(BUFFER_DIR, folder_name)
                
                # 只有缓冲区拷贝完成了，才提交任务
                if os.path.exists(buf_path):
                    processed_list = get_processed_folders()
                    print(f"处理过的:{processed_list}")
                    if folder_name in processed_list:
                        print(f"跳过 {folder_name} 因为处理过了")
                        pending_folders.pop(0) # 关键：必须从列表中移除！
                        # 如果缓冲区已经拷贝了一半，也顺手清理掉
                        if os.path.exists(buf_path):
                            shutil.rmtree(buf_path, ignore_errors=True)
                        continue
                    pending_folders.pop(0)
                    fut = executor.submit(process_folder_task, folder_name)
                    futures[fut] = folder_name
                    
                    # 新建进程后休息 20 秒，防止趋同
                    if len(pending_folders) > 0:
                        print(f"\n[Main] Worker started for {folder_name}. Cooling down 20s...")
                        time.sleep(20)
                else:
                    # 等待缓冲区拷贝
                    time.sleep(2)

            # 检查完成的任务
            done_futures = [f for f in futures if f.done()]
            for f in done_futures:
                res = f.result()
                del futures[f]
                pbar.update(1)

    bg_buffer.terminate()

if __name__ == "__main__":
    # 根据您的机器性能设置 max_workers
    process_all_parallel(max_workers=8)
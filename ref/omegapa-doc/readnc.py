import xarray as xr

nc_file = r"H:/krigedcdod_v2/dustscenario_MY24_v2-0.nc"
ds = xr.open_dataset(nc_file)

# 查看数据集的基本信息
print(ds)

# 查看所有变量名
print("Variables:", list(ds.data_vars))

# 查看所有坐标名
print("Coordinates:", list(ds.coords))

import xarray as xr
import numpy as np
import os
import time
import netCDF4 as nc
import pandas as pd


download_folder = "./combined_lsm"
years = range(1950, 2013 + 1)
months = [f'{i:02d}' for i in range(1, 13)]  # 1月到12月

# 用來存儲所有年份和月份的數據
all_data = []

for year in years:
    for month in months:
        file_path = os.path.join(download_folder, f"combined_lsm_{year}_{month}.nc")

        if os.path.exists(file_path):
            try:
                # 開啟數據集並提取 SST 變數
                data = xr.open_dataset(file_path)
                variable_name = '__xarray_dataarray_variable__'  # 更新為 SST 變數名稱
                sst_data = data[variable_name]
                

                # 確認時間維度並重新命名
                if 'time' not in sst_data.coords and 'date' in data.variables:
                    data = data.rename({'date': 'time'})
                    data = data.assign_coords(time=pd.to_datetime(data['time'].values, format='%Y%m%d'))
                    sst_data = data[variable_name]

                # 將此文件的數據加入列表中
                all_data.append(sst_data)

            except Exception as e:
                print(f"處理 {file_path} 時出現錯誤: {e}")

# 合併所有數據到一個 DataArray 中
combined_data = xr.concat(all_data, dim="time")

# 按月分組並計算每月平均氣候值
monthly_climatology = combined_data.groupby("time.month").mean(dim="time")

# 儲存每月平均氣候值至 NetCDF 文件
monthly_climatology.to_netcdf("training_data_monthly_climatology.nc")
print("已將training_data的每月平均氣候值儲存為 training_data_monthly_climatology.nc")


"""
file_path_mean = "C:/Users/1/processed_data/training_data_monthly_climatology.nc"
ds_mean = xr.open_dataset(file_path_mean)
print(ds_mean)
# training_data_mean = np.load("C:/Users/1/processed_data/monthly_climatology_30years.npy")
training_data_mean = ds_mean['__xarray_dataarray_variable__']


data_avg = training_data_mean.coarsen(latitude=4, longitude=4, boundary='trim').mean()
lat_new = np.linspace(-64, 62, 64)
lon_new = np.linspace(0, 360, 128)
data_interp = data_avg.interp(latitude=lat_new, longitude=lon_new).fillna(0)
# data_numpy = data_interp.values
data_interp.to_netcdf("training_data_monthly_climatology.nc")
"""









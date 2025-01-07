import netCDF4 as nc
import numpy as np
import xarray as xr
import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import time


# 創建一個處理單個文件的函數
def process_file(file_path, year, month):
    
    # 打開 NetCDF 文件
    data = xr.open_dataset(file_path)
    ds = nc.Dataset(file_path)


    # 讀取數據
    sst_data = data['sst']
    t2m_data = data['t2m']
    lon = data['longitude']
    lat = data['latitude']
    land_sea_mask = data['lsm']

    # 合併SST和T2M數據
    combined_temperature = xr.where(land_sea_mask == 0, sst_data, t2m_data)

    # 設定存儲路徑
    output_folder = "./combined_data"
    os.makedirs(output_folder, exist_ok=True)  # 如果資料夾不存在，則創建它
    combined_filename = os.path.join(output_folder, f"combined_{year}_{month}.nc")
    
    # 保存合併的數據到NetCDF
    combined_temperature.to_netcdf(combined_filename)
    print(f"Saved combined data to {combined_filename}")



# 遍歷所有下載的文件
download_folder = "./"  # 假設你所有的 .nc 文件都在當前目錄下
years = range(2021, 2021+1)  # 例如從1950年到2021年
months = [f'{i:02d}' for i in range(1, 13)]  # '01', '02', ..., '12'

for year in years:
    for month in months:
        file_path = os.path.join(download_folder, f"downloaded_{year}_{month}.nc")
        if os.path.exists(file_path):
            try:
                print(f"Processing {file_path}")
                process_file(file_path, year, month)
                time.sleep(1)  # 可選，讓程式每次處理完等待 1 秒
            except Exception as e:
                print(f"Error processing {file_path}: {e}")



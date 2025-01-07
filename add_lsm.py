import xarray as xr
import numpy as np
import netCDF4 as nc
import os
from global_land_mask import globe
import time

# 1. 定義處理單個文件的函數
def process_nc_file(file_path, output_dir, year, month):
    # 加載 NetCDF 文件
    dataset = xr.open_dataset(file_path)

    # 提取變量
    latitudes = dataset['latitude']  # 緯度
    longitudes = dataset['longitude']  # 經度
    sst = dataset['__xarray_dataarray_variable__']  # SST 數據 (或您需要的變量)

    lon_adjust = longitudes - 180

    # 建立經緯度網格
    lon_grid, lat_grid = np.meshgrid(lon_adjust, latitudes)

    # 使用 global-land-mask 檢測陸地
    land_mask = globe.is_land(lat_grid, lon_grid)  # 返回布林陣列 (True 表示陸地)

    # 交換經度遮罩以匹配資料順序
    temp = land_mask[:, 0:(len(longitudes) // 2)].copy()
    land_mask[:, 0:(len(longitudes) // 2)] = land_mask[:, (len(longitudes) // 2):]
    land_mask[:, (len(longitudes) // 2):] = temp

    # 應用遮罩
    sst_with_mask = xr.where(land_mask == 0, sst, 0)    # 將陸地上的值設為 0
    combined_temperature = sst_with_mask - 273.15

    # 保存合併的數據到NetCDF
    combined_filename = f"climatology_lsm_{month}.nc"
    output_file = os.path.join(output_dir, combined_filename)
    combined_temperature.to_netcdf(output_file)
    print(f"Saved climatology data with land-sea mask to {combined_filename}")



# 設置輸出目錄
output_dir = "./process_data/"  # 替換為保存處理後文件的目錄

# 遍歷所有下載的文件
combined_folder = "./combined_data/"  # 假設你所有的 .nc 文件都在當前目錄下
years = range(1950, 2021 + 1)  # 例如從1950年到2021年
months = [f'{i:02d}' for i in range(1, 13)]  # '01', '02', ..., '12'


for year in years:
    for month in months:
        file_path = os.path.join(combined_folder, f"combined_{year}_{month}.nc")
        if os.path.exists(file_path):
            try:
                print(f"Processing {file_path}")
                process_nc_file(file_path,output_dir, year, month)
                time.sleep(1)  # 可選，讓程式每次處理完等待 1 秒
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
             

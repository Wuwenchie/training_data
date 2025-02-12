import xarray as xr
import os
from netCDF4 import Dataset
import netCDF4 as nc
import numpy as np

file_paths = r'./data/traing_data/sos'
#定義一個空列表來儲存該文件里的.nc數據路徑
file_path = [] 

for file_name in os.listdir(file_paths):
    file_path.append(r'D:/NOAA_file2023/01/'+file_name)
file_path ##输出文件路徑

cmorph_new = [] ##建立一个空列表，存储逐日降雨数据
for i in range(len(file_path)):
    cmorph = xr.open_dataset(file_path[i])['cmorph']
    cmorph_new.append(cmorph) ##存储每日的降雨数据
da = xr.concat(cmorph_new,dim='time') ##将数据以时间维度来进行拼接
#print(da)
path_new = r"./data/training_data" ##设置新路径
#print(da.variable)
da.to_netcdf(path_new + 'CMORPH_6.nc') ##对拼接好的文件进行存储
da.close() ##关闭文件

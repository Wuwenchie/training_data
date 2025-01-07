import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, UpSampling2D, Input
from tensorflow.keras.layers import MaxPooling2D, concatenate, TimeDistributed, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import h5py
import pickle



# 導入文件
file_path = "C:/Users/1/processed_data/test_data.nc"
file_path_mean = "C:/Users/1/processed_data/training_data_monthly_climatology.nc"
file_actual = "C:/Users/1/processed_data/test_visual_data.nc"
future_predictions = np.load("C:/Users/1/processed_data/long_term_predictions.npy")

ds = xr.open_dataset(file_path)
ds_actual = xr.open_dataset(file_actual)
sample_index = 2
time_index = 1

x_test = ds['x_test'].isel(samples=3, time_steps=7)
y_test = ds['y_test'].isel(samples=sample_index, future_steps=time_index)
lat = y_test.coords['latitude']
lon = y_test.coords['longitude']

actual = ds_actual['x_test'].isel(samples=3, time_steps=1)

ds_mean = xr.open_dataset(file_path_mean)
training_data_mean = ds_mean['__xarray_dataarray_variable__']

mean = training_data_mean.isel(month=5)
mean = mean.values
training_min = mean.min()
training_max = mean.max()





plt.figure(figsize=(10, 6))


plt.subplot(2, 2, 1) 
# 設定經度和緯度範圍
lon2, lat2 = np.meshgrid(lon, lat)

m = Basemap(projection='cyl',
            llcrnrlat=-64,
            urcrnrlat=62,
            llcrnrlon=0,
            urcrnrlon=360,
            resolution='c')

# 繪製地圖
cx, cy = m(lon2, lat2)

actual_sst = actual.squeeze(dim='channels')
# actual_sst = actual_sst * (training_max-training_min) + training_min
# cs = m.pcolormesh(cx,cy,np.squeeze(actual_sst[:,:]), cmap='jet',vmin=-2, vmax=30, shading='auto')
cs = plt.contourf(cx,cy,np.squeeze(actual_sst[:,:]), np.arange(-3,33,1), extend='both', cmap=cm.jet)

# 畫海岸線
m.drawcoastlines()
# 添加 colorbar
cbar = m.colorbar(cs,"bottom", pad="10%")

cbar.set_label('Temperature (K)')
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Actual SST')




plt.subplot(2, 2, 2)
# 從 xarray.DataArray 轉為 NumPy 並增加批次維度
pre_1 = np.squeeze(future_predictions)
# print("first sample of autogressive prediction shape:", pre_1.shape)

future_sample = 4
future_index = 1
pre_1 = pre_1[future_sample]
pre_1 = pre_1[future_index]
# print("first sample of autogressive prediction shape of 0:", pre_1.shape)


predicted_sst = pre_1 * (training_max-training_min) + training_min

cs = plt.contourf(cx,cy,np.squeeze(predicted_sst[:,:]), np.arange(-3,33,1), extend='both', cmap=cm.jet)

m.drawcoastlines()
cbar = m.colorbar(cs, "bottom", pad="10%")
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Predict SST')


plt.subplot(2, 2, 3)
difference = predicted_sst - actual_sst   

difference_array = xr.DataArray(difference, dims=["latitude", "longitude"], coords={"latitude": lat, "longitude": lon})

# cs = m.pcolormesh(cx,cy,np.squeeze(difference[:,:]), cmap='jet', vmin=-3, vmax=3, shading='auto')
cs = plt.contourf(cx,cy,np.squeeze(difference[:,:]), np.arange(-3,3,0.1), extend='both', cmap=cm.jet)

m.drawcoastlines()
cbar = m.colorbar(cs, "bottom", pad="10%")
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Difference SST')



plt.subplot(2, 2, 4)

climatology_array = xr.DataArray(mean, dims=["latitude", "longitude"], coords={"latitude": lat, "longitude": lon})
anomaly_sst = predicted_sst - mean
# cs = m.pcolormesh(cx,cy,np.squeeze(anomaly_sst[:,:]), cmap='jet', vmin=-5, vmax=5, shading='auto')
cs = plt.contourf(cx,cy,np.squeeze(anomaly_sst[:,:]), np.arange(-5,5,0.1), extend='both', cmap=cm.jet)

m.drawcoastlines()
cbar = m.colorbar(cs, "bottom", pad="10%")
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Model Anomaly SST')

plt.suptitle('June 2017', fontsize=16, fontweight='normal', ha='center')

plt.tight_layout()
plt.savefig(f"June 2017.png")

plt.show()



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
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import matplotlib.cm as cm
import h5py
import pickle


# y_pred = model.predict(x_test)
model = load_model("unet_lstm_nc.h5", custom_objects={'Functional': tf.keras.Model, "L2":tf.keras.regularizers.l2})
# print(model.summary())


file_path = "C:/Users/1/processed_data/test_data.nc"
ds = xr.open_dataset(file_path)
# print(ds)

sample_index = 2
time_index = 1

x_test = ds['x_test'].isel(samples=sample_index)
y_test = ds['y_test'].isel(samples=sample_index, future_steps=time_index)

# y_test_visual = np.load("C:/Users/1/processed_data/y_test_visual.npy")
file_path_mean = "C:/Users/1/processed_data/training_data_monthly_climatology.nc"
ds_mean = xr.open_dataset(file_path_mean)
# training_data_mean = np.load("C:/Users/1/processed_data/monthly_climatology_30years.npy")
training_data_mean = ds_mean['__xarray_dataarray_variable__']
lat = y_test.coords['latitude']
lon = y_test.coords['longitude']

# 2016_June
  
# actual_sst = np.squeeze(y_test_visual[sample_index])  # shape: (64, 128)
# actual_sst = actual_sst[time_index]
# data_min = actual_sst.min()
# data_max = actual_sst.max()
June_mean = training_data_mean.isel(month=6)
# June_mean = np.squeeze(training_data_mean[5])

June_mean = June_mean.values

training_min = June_mean.min()
training_max = June_mean.max()




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

actual_sst = y_test.squeeze(dim='channels')
actual_sst = actual_sst * (training_max-training_min) + training_min
# cs = m.pcolormesh(cx,cy,np.squeeze(actual_sst[:,:]), cmap='jet',vmin=-2, vmax=30, shading='auto')
cs = plt.contourf(cx,cy,np.squeeze(actual_sst[:,:]), np.arange(-3,33,2), extend='both', cmap=cm.jet)

# 畫海岸線
m.drawcoastlines()
# 添加 colorbar
cbar = m.colorbar(cs,"bottom", pad="10%")

cbar.set_label('Temperature (K)')
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Actual SST')




plt.subplot(2, 2, 2)
# 從 xarray.DataArray 轉為 NumPy 並增加批次維度
x_test = np.expand_dims(x_test.values, axis=0)
print(f"x_test shape after expand_dims: {x_test.shape}")  # 應輸出 (1, 12, 64, 128, 1)

predicted_sst = model.predict(x_test) # shape: (1, 12, 64, 128, 1)
predicted_sst = np.squeeze(predicted_sst)  # shape: (64, 128)
predicted_sst_1 = predicted_sst[time_index]
    

predicted_sst_1 = predicted_sst_1 * (training_max-training_min) + training_min

# cs = m.pcolormesh(cx,cy,np.squeeze(predicted_sst_1[:,:]), cmap='jet', vmin=-2, vmax=30, shading='auto')
cs = plt.contourf(cx,cy,np.squeeze(predicted_sst_1[:,:]), np.arange(-3,33,1), extend='both', cmap=cm.jet)

m.drawcoastlines()
cbar = m.colorbar(cs, "bottom", pad="10%")
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Predict SST')

plt.subplot(2, 2, 3)
difference = predicted_sst_1 - actual_sst   

difference_array = xr.DataArray(difference, dims=["latitude", "longitude"], coords={"latitude": lat, "longitude": lon})

# cs = m.pcolormesh(cx,cy,np.squeeze(difference[:,:]), cmap='jet', vmin=-3, vmax=3, shading='auto')
cs = plt.contourf(cx,cy,np.squeeze(difference[:,:]), np.arange(-5,5,1), extend='both', cmap=cm.jet)

m.drawcoastlines()
cbar = m.colorbar(cs, "bottom", pad="10%")
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Difference SST')


plt.subplot(2, 2, 4)
# climatology_sst = np.squeeze(climatology[4])  # shape: (64, 128)

climatology_array = xr.DataArray(June_mean, dims=["latitude", "longitude"], coords={"latitude": lat, "longitude": lon})
anomaly_sst = predicted_sst_1 - June_mean
# cs = m.pcolormesh(cx,cy,np.squeeze(anomaly_sst[:,:]), cmap='jet', vmin=-5, vmax=5, shading='auto')
cs = plt.contourf(cx,cy,np.squeeze(anomaly_sst[:,:]), np.arange(-5,5,1), extend='both', cmap=cm.jet)

m.drawcoastlines()
cbar = m.colorbar(cs, "bottom", pad="10%")
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Model Anomaly SST')

plt.suptitle('June 2016', fontsize=16, fontweight='normal', ha='center')

plt.tight_layout()
plt.savefig(f"June 2016 map.png")

plt.show()


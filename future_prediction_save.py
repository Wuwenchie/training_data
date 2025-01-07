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

"""
model = load_model("unet_lstm_nc.h5", custom_objects={'Functional': tf.keras.Model, "L2":tf.keras.regularizers.l2})

file_path = "C:/Users/1/processed_data/test_data.nc"
ds = xr.open_dataset(file_path)


sample_index = 2
time_index = 1

x_test = ds['x_test'].isel(samples=sample_index)
y_test = ds['y_test'].isel(samples=sample_index, future_steps=time_index)
lat = y_test.coords['latitude']
lon = y_test.coords['longitude']

x_test = np.expand_dims(x_test.values, axis=0)
print(f"x_test shape after expand_dims: {x_test.shape}")  # 應輸出 (1, 12, 64, 128, 1)

file_path_mean = "C:/Users/1/processed_data/training_data_monthly_climatology.nc"
ds_mean = xr.open_dataset(file_path_mean)
training_data_mean = ds_mean['__xarray_dataarray_variable__']

June_mean = training_data_mean.isel(month=6)
June_mean = June_mean.values
training_min = June_mean.min()
training_max = June_mean.max()


def autoregressive_prediction(model, initial_input, steps=12):
    predicted_sst = []
    current_input = initial_input

    for step in range(steps):
        prediction = model.predict(current_input)
        predicted_sst.append(prediction)
        # 更新輸入數據，將預測結果作為下一次的輸入
        # 丟掉最早的數據，加入新預測的數據
        # print("Prediction Shape:", prediction.shape)
        current_input = np.concatenate([current_input[:, 2:, :, :, :], prediction], axis=1)

    # 將所有預測結果轉換為數組
    predicted_sst = np.array(predicted_sst)
    return np.array(predicted_sst)

# 使用初始的12個月數據進行24個月的自迴歸預測
future_predictions = autoregressive_prediction(model, x_test)
print("future prediction shape:", future_predictions.shape)    

# 儲存成 .npy 文件
np.save("long_term_predictions.npy", future_predictions)
print("save lond_term_predictions.npy")
"""

future_predictions = np.load("C:/Users/1/processed_data/long_term_predictions.npy")
print(future_predictions.shape)
latitudes = np.linspace(-64, 62, future_predictions.shape[2])  # 替換為正確的經緯度範圍
longitudes = np.linspace(0, 360, future_predictions.shape[3])  # 替換為正確的經緯度範圍

# 建立 xarray.DataArray
future_predictions_da = xr.DataArray(
    future_predictions,
    dims=["samples", "batch", "future_steps", "latitude", "longitude", "channels"],
    coords={
        "samples": np.arange(future_predictions.shape[0]),
        "future_steps": np.arange(future_predictions.shape[1]),
        "latitude": latitudes,
        "longitude": longitudes
    },
    name="future_predictions"
)

# 將 DataArray 儲存為 NetCDF 檔案
output_nc_path = "C:/Users/1/processed_data/long_term_predictions.nc"
future_predictions_da.to_netcdf(output_nc_path)

print(f"Future predictions saved to {output_nc_path}")

"""
pre_1 = future_predictions[0,:,:,:,:,:]
print("first sample of autogressive prediction shape:", pre_1.shape)
"""

"""
predicted_sst = predicted_sst * (training_max-training_min) + training_min

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
cs = plt.contourf(cx,cy,np.squeeze(predicted_sst[:,:]), np.arange(-3,33,1), extend='both', cmap=cm.jet)

m.drawcoastlines()
cbar = m.colorbar(cs, "bottom", pad="10%")
cbar.set_label('Temperature ($^\circ$C)')
plt.title(f'Predict SST')
"""

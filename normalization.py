import xarray as xr
import numpy as np
import os

def process_data(file_path):
    data = xr.open_dataset(file_path)['__xarray_dataarray_variable__']
    data_avg = data.coarsen(latitude=4, longitude=4, boundary='trim').mean()
    lat_new = np.linspace(-64, 62, 64)
    lon_new = np.linspace(0, 360, 128)
    data_interp = data_avg.interp(latitude=lat_new, longitude=lon_new).fillna(0)

    # 標準化
    data_min = data_interp.min()
    data_max = data_interp.max()
    data_range = data_max - data_min
    data_normalized = xr.where(data_range == 0, 0, (data_interp - data_min) / data_range)

    return data_normalized.values

def save_to_nc(x_test, y_test, output_folder):
    dataset = xr.Dataset(
        {
            "x_test": (["samples", "time_steps", "latitude", "longitude", "channels"], x_test),
            "y_test": (["samples", "future_steps", "latitude", "longitude", "channels"], y_test),
        },
        coords={
            "samples": np.arange(x_test.shape[0]),
            "time_steps": np.arange(x_test.shape[1]),
            "future_steps": np.arange(y_test.shape[1]),
            "latitude": np.linspace(-64, 62, x_test.shape[2]),
            "longitude": np.linspace(0, 360, x_test.shape[3]),
        },
    )
    output_path = os.path.join(output_folder, "test_data.nc")
    dataset.to_netcdf(output_path)
    print(f"Testing data saved to {output_path}")

def process_all_files(input_folder, time_steps, future_steps, output_folder):
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".nc")]
    processed_data_list = []

    for file_path in file_paths:
        print(f"Processing {file_path}...")
        processed_data = process_data(file_path)
        processed_data_list.append(processed_data)

    processed_data_array = np.stack(processed_data_list, axis=0)
    processed_data_array = np.expand_dims(processed_data_array, axis=-1)

    n_samples = len(processed_data_array) - time_steps - future_steps + 1
    x_test = []
    y_test = []

    for i in range(n_samples):
        x_test.append(processed_data_array[i:i + time_steps])
        y_test.append(processed_data_array[i + time_steps:i + time_steps + future_steps])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    save_to_nc(x_test, y_test, output_folder)

    print(f"Processed data saved to {output_folder}.")


# 使用範例
input_folder = './testing_data'
output_folder = './processed_data'

# 設定參數
time_steps = 12
future_steps = 2

# 確保輸出文件夾存在
os.makedirs(output_folder, exist_ok=True)

# 處理數據
process_all_files(input_folder, time_steps, future_steps, output_folder)

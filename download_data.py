import cdsapi
import time

# 初始化 CDS API 客戶端
client = cdsapi.Client()

# 設定要下載的年份範圍和月份
years = list(range(1950, 2021 + 1))  # 1950 年至 2021 年
months = [f'{i:02d}' for i in range(1, 13)]  # '01', '02', ..., '12'

# 定義資料集
dataset = "reanalysis-era5-single-levels-monthly-means"

# 遍歷所有年份和月份
for year in years:
    for month in months:
        # 構建請求的參數
        request = {
            'product_type': ['monthly_averaged_reanalysis'],
            'variable': ['2m_temperature', 'sea_surface_temperature', 'land_sea_mask'],
            'year': [str(year)],
            'month': [month],
            'time': ['00:00'],
            'data_format': 'netcdf',
            'area': [90, 0, -90, 360]  # 全球範圍
        }

        # 文件名
        filename = f'downloaded_{year}_{month}.nc'

        try:
            # 下載數據
            print(f"Downloading data for {year}-{month}...")
            client.retrieve(dataset, request).download(filename)
            print(f"Downloaded: {filename}")

            # 加入延遲，避免過多請求
            time.sleep(10)  # 等待 10 秒

        except Exception as e:
            print(f"Failed to download data for {year}-{month}: {e}")

time.sleep(10)  # 每次下載完等待 10 秒再進行下一次請求

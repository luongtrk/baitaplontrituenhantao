"""
Giai đoạn 1: Thu thập dữ liệu khí tượng & chất lượng không khí theo giờ
từ Open-Meteo Archive API cho khu vực Hải Phòng (Cảng Lạch Huyện / Trung tâm Hải Phòng).

6 nhóm biến khí tượng theo Đề cương N04.14:
  (1) Nhiệt độ không khí (temperature_2m - °C)
  (2) Độ ẩm tương đối (relative_humidity_2m - %)
  (3) Áp suất khí quyển bề mặt (surface_pressure - hPa)
  (4) Tốc độ gió & Hướng gió (wind_speed_10m - km/h, wind_direction_10m - °, wind_gusts_10m - km/h)
  (5) Lượng mưa tích lũy (precipitation - mm) + weather_code (WMO)
  (6) Chất lượng không khí (pm2_5, pm10, us_aqi)
"""

import os
import time
import requests
import pandas as pd

# Tọa độ trạm quan trắc chính khu vực Hải Phòng (Cảng Lạch Huyện / Cát Hải)
HAI_PHONG_LAT = 20.8524
HAI_PHONG_LON = 106.9152

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

RAW_DATA_PATH = os.path.join("data", "raw_weather.csv")


def fetch_weather_archive(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "precipitation",
            "cloud_cover",
            "weather_code",
        ]),
        "timezone": "Asia/Bangkok",
    }
    print(f"[1/2] Dang tai du lieu khi tuong Open-Meteo ({start_date} -> {end_date})...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    print(f"      -> Da tai {len(df):,} ban ghi khi tuong theo gio.")
    return df


def fetch_air_quality_archive(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10,us_aqi",
        "timezone": "Asia/Bangkok",
    }
    print(f"[2/2] Dang tai du lieu chat luong khong khi Open-Meteo ({start_date} -> {end_date})...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    print(f"      -> Da tai {len(df):,} ban ghi AQI theo gio.")
    return df


def main():
    os.makedirs("data", exist_ok=True)
    t0 = time.time()

    df_weather = fetch_weather_archive(HAI_PHONG_LAT, HAI_PHONG_LON, START_DATE, END_DATE)
    try:
        df_aqi = fetch_air_quality_archive(HAI_PHONG_LAT, HAI_PHONG_LON, START_DATE, END_DATE)
        df = pd.merge(df_weather, df_aqi, on="time", how="left")
    except Exception as exc:
        print(f"[CANH BAO] Khong tai duoc AQI ({exc}), tao cot mac dinh.")
        df = df_weather.copy()
        df["pm2_5"] = 20.0
        df["pm10"] = 35.0
        df["us_aqi"] = 55.0

    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n[HOAN TAT] Da luu {len(df):,} dong x {df.shape[1]} cot vao '{RAW_DATA_PATH}' ({elapsed:.1f}s).")
    print("Cac cot:", list(df.columns))
    print("So gia tri thieu (NaN) ban dau:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()

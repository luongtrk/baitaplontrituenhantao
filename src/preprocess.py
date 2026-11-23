"""
Giai đoạn 2: Tiền xử lý dữ liệu, nội suy tuyến tính, xử lý ngoại lai (Outliers),
tạo đặc trưng (Feature Engineering) và kỹ thuật Cửa sổ trượt (Sliding Window 168 giờ = 7 ngày).
"""

import os
import numpy as np
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw_weather.csv")
PROCESSED_DATA_PATH = os.path.join("data", "data_processed.csv")

# Các đặc trưng đầu vào cho mạng LSTM (cửa sổ trượt 168 giờ)
LSTM_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "pm2_5",
    "us_aqi",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "wind_u",
    "wind_v",
]

# 3 biến mục tiêu dự báo hồi quy liên tục (Bài toán 1)
TARGET_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
]

# Đặc trưng đầu vào cho bài toán phân loại (Bài toán 2 - không dùng weather_code để tránh rò rỉ dữ liệu)
CLF_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    "pm2_5",
    "pm10",
    "us_aqi",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "temp_diff_3h",
    "pressure_diff_3h",
    "humidity_diff_3h",
    "precip_sum_6h",
    "wind_mean_6h",
]

LOOKBACK_HOURS = 168  # 7 ngày liên tiếp theo giờ
HORIZON_HOURS = 24    # Dự báo 24 giờ tiếp theo


def map_weather_class(row: pd.Series) -> int:
    """
    Ánh xạ trạng thái thời tiết thành 4 nhãn chuẩn theo Đề cương N04.14:
      0: Nắng quang mây (WMO 0, 1, 2 và không mưa/gió mạnh)
      1: Âm u / Sương mù (WMO 3, 45, 48)
      2: Mưa rào / Mưa phùn (WMO 51-67, 80-82 hoặc lượng mưa 0.2 - 7.5 mm/h)
      3: Giông bão / Gió mạnh nguy hiểm (WMO >= 95 hoặc mưa > 7.5 mm/h hoặc gió >= 35 km/h)
    """
    code = int(row["weather_code"]) if not pd.isna(row["weather_code"]) else 0
    precip = float(row["precipitation"])
    wind = float(row["wind_speed_10m"])

    if code >= 95 or precip >= 7.5 or wind >= 35.0:
        return 3
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82) or precip >= 0.2:
        return 2
    if code in (3, 45, 48):
        return 1
    return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng tuần hoàn thời gian, vector gió và biến động khí tượng."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Biến thời gian tuần hoàn (Cyclical Encoding)
    hour = df["time"].dt.hour
    month = df["time"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Vector gió (u, v) từ tốc độ & hướng gió
    rad = np.deg2rad(df["wind_direction_10m"].fillna(0.0))
    df["wind_u"] = df["wind_speed_10m"] * np.cos(rad)
    df["wind_v"] = df["wind_speed_10m"] * np.sin(rad)

    # Biến động ngắn hạn (hữu ích cho phân loại giông bão / áp thấp)
    df["temp_diff_3h"] = df["temperature_2m"].diff(3).fillna(0.0)
    df["pressure_diff_3h"] = df["surface_pressure"].diff(3).fillna(0.0)
    df["humidity_diff_3h"] = df["relative_humidity_2m"].diff(3).fillna(0.0)
    df["precip_sum_6h"] = df["precipitation"].rolling(6, min_periods=1).sum()
    df["wind_mean_6h"] = df["wind_speed_10m"].rolling(6, min_periods=1).mean()

    return df


def preprocess_raw_data(raw_path: str = RAW_DATA_PATH, out_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    print(f"[1/4] Dang doc du lieu tho tu '{raw_path}'...")
    df = pd.read_csv(raw_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    # Báo cáo missing values trước khi xử lý
    missing_before = int(df.isna().sum().sum())
    print(f"      -> Tong so o bi thieu (NaN) truoc xu ly: {missing_before}")

    # 1. Nội suy tuyến tính (Linear Interpolation) cho các cột số liên tục
    numeric_cols = [
        "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "precipitation", "cloud_cover", "pm2_5", "pm10", "us_aqi",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear").bfill().ffill()

    if "weather_code" in df.columns:
        df["weather_code"] = pd.to_numeric(df["weather_code"], errors="coerce").ffill().bfill().fillna(0).astype(int)

    # 2. Xử lý ngoại lai (Outliers) theo ngưỡng vật lý khí tượng Hải Phòng
    print("[2/4] Dang xu ly ngoai lai (Outliers) theo nguong vat ly...")
    bounds = {
        "temperature_2m": (2.0, 45.0),
        "relative_humidity_2m": (10.0, 100.0),
        "surface_pressure": (940.0, 1050.0),
        "wind_speed_10m": (0.0, 150.0),
        "wind_gusts_10m": (0.0, 200.0),
        "precipitation": (0.0, 150.0),
        "pm2_5": (0.0, 400.0),
        "pm10": (0.0, 500.0),
        "us_aqi": (0.0, 500.0),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # 3. Tạo đặc trưng
    print("[3/4] Dang tao dac trung tuan hoan & bien dong khi tuong...")
    df = engineer_features(df)

    # 4. Tạo nhãn phân loại 4 trạng thái thời tiết
    df["weather_label"] = df.apply(map_weather_class, axis=1).astype(int)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    missing_after = int(df.isna().sum().sum())
    print(f"[4/4] Hoan tat! Da luu '{out_path}' ({len(df):,} dong x {df.shape[1]} cot, NaN con lai: {missing_after}).")
    print("Phan bo nhan phan loai (weather_label):")
    print(df["weather_label"].value_counts().sort_index())
    return df


def create_sliding_windows(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    lookback: int = LOOKBACK_HOURS,
    horizon: int = HORIZON_HOURS,
    stride: int = 1,
):
    """
    Tạo ma trận Tensor 3D cho LSTM:
      X_seq: (N, lookback=168, n_features)
      y_seq: (N, horizon=24, n_targets=3)
    """
    X_list, y_list = [], []
    max_idx = len(X_scaled) - lookback - horizon + 1
    for i in range(0, max_idx, stride):
        X_list.append(X_scaled[i : i + lookback])
        y_list.append(y_scaled[i + lookback : i + lookback + horizon])
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


if __name__ == "__main__":
    preprocess_raw_data()

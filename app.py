"""
Giai đoạn 5: Backend FastAPI phục vụ suy luận AI thời gian thực (LSTM + XGBoost)
và cung cấp Web Dashboard Dự báo Thời tiết & An toàn Hàng hải Hải Phòng.

Chạy ứng dụng:
    .venv\\Scripts\\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
Sau đó mở trình duyệt tại:
    http://127.0.0.1:8000
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tensorflow import keras

from src.beaufort import (
    WEATHER_LABELS,
    degrees_to_compass_vi,
    wind_speed_to_beaufort,
)
from src.preprocess import (
    CLF_FEATURES,
    LOOKBACK_HOURS,
    LSTM_FEATURES,
    PROCESSED_DATA_PATH,
    TARGET_COLS,
    engineer_features,
)

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
DASHBOARD_HTML = "ai_weather_dashboard.html"

STATIONS: Dict[str, Dict[str, Any]] = {
    "lach_huyen": {
        "name": "Cảng Lạch Huyện, Hải Phòng",
        "short_area": "Luồng tàu Lạch Huyện - Nam Triệu",
        "lat": 20.8524,
        "lon": 106.9152,
        "gradient": "linear-gradient(135deg, #72edf2 10%, #5151e5 100%)",
    },
    "hai_phong": {
        "name": "Trung tâm TP. Hải Phòng",
        "short_area": "Nội thành & sông Cấm",
        "lat": 20.8449,
        "lon": 106.6881,
        "gradient": "linear-gradient(135deg, #ffd166 10%, #5151e5 100%)",
    },
    "do_son": {
        "name": "Bán đảo Đồ Sơn, Hải Phòng",
        "short_area": "Vùng biển ven bờ Đồ Sơn",
        "lat": 20.7093,
        "lon": 106.7937,
        "gradient": "linear-gradient(135deg, #48cae4 10%, #023e8a 100%)",
    },
    "cat_ba": {
        "name": "Quần đảo Cát Bà, Hải Phòng",
        "short_area": "Vịnh Lan Hạ & Quần đảo Cát Bà",
        "lat": 20.7275,
        "lon": 107.0503,
        "gradient": "linear-gradient(135deg, #60a5fa 10%, #1e3a8a 100%)",
    },
    "dinh_vu": {
        "name": "Bến Cảng Đình Vũ, Hải Phòng",
        "short_area": "Cụm Cảng Đình Vũ - sông Bạch Đằng",
        "lat": 20.8351,
        "lon": 106.7583,
        "gradient": "linear-gradient(135deg, #38bdf8 10%, #312e81 100%)",
    },
}

WEEKDAY_VI = [
    "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"
]

app = FastAPI(
    title="AI Weather Forecasting & Maritime Safety API (Hai Phong)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

FEATURE_LABELS_VI = {
    "temperature_2m": "Nhiệt độ không khí (°C)",
    "relative_humidity_2m": "Độ ẩm tương đối (%)",
    "surface_pressure": "Áp suất bề mặt (hPa)",
    "wind_speed_10m": "Tốc độ gió (km/h)",
    "wind_gusts_10m": "Gió giật tối đa (km/h)",
    "precipitation": "Lượng mưa hiện tại (mm/h)",
    "cloud_cover": "Độ che phủ mây (%)",
    "pm2_5": "Bụi mịn PM2.5",
    "pm10": "Bụi mịn PM10",
    "us_aqi": "Chỉ số không khí AQI",
    "hour_sin": "Chu kỳ giờ (sin)",
    "hour_cos": "Chu kỳ giờ (cos)",
    "month_sin": "Chu kỳ mùa (sin)",
    "month_cos": "Chu kỳ mùa (cos)",
    "temp_diff_3h": "Biến động nhiệt độ 3h",
    "pressure_diff_3h": "Biến động áp suất 3h",
    "humidity_diff_3h": "Biến động độ ẩm 3h",
    "precip_sum_6h": "Tổng lượng mưa 6h",
    "wind_mean_6h": "Gió trung bình 6h",
}


class SimulationInput(BaseModel):
    temperature_2m: float = 28.0
    relative_humidity_2m: float = 78.0
    surface_pressure: float = 1012.0
    wind_speed_10m: float = 24.0
    wind_gusts_10m: float = 34.0
    precipitation: float = 0.0
    cloud_cover: float = 40.0
    pressure_diff_3h: float = 0.0
    us_aqi: float = 50.0

# Bộ nhớ đệm mô hình AI
MODEL_CACHE: Dict[str, Any] = {}


def load_models():
    """Nạp các mô hình AI đã huấn luyện vào bộ nhớ."""
    if MODEL_CACHE.get("loaded"):
        return
    lstm_path = os.path.join(MODELS_DIR, "lstm_weather.keras")
    xgb_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    scaler_x_path = os.path.join(MODELS_DIR, "scaler_X.pkl")
    scaler_y_path = os.path.join(MODELS_DIR, "scaler_y.pkl")

    if not all(os.path.exists(p) for p in [lstm_path, xgb_path, scaler_x_path, scaler_y_path]):
        raise RuntimeError(
            "Chua tim thay file mo hinh trong thu muc 'models/'. "
            "Hay chay 'python -m src.train_models' truoc!"
        )

    MODEL_CACHE["lstm"] = keras.models.load_model(lstm_path)
    MODEL_CACHE["xgb"] = joblib.load(xgb_path)
    MODEL_CACHE["scaler_X"] = joblib.load(scaler_x_path)
    MODEL_CACHE["scaler_y"] = joblib.load(scaler_y_path)
    MODEL_CACHE["loaded"] = True
    print("[FastAPI] Da nap thanh cong mo hinh LSTM (.keras) & XGBoost (.pkl).")


@app.on_event("startup")
def startup_event():
    try:
        load_models()
    except Exception as exc:
        print(f"[CANH BAO] Chua the nap mo hinh luc khoi dong: {exc}")


def fetch_realtime_open_meteo(lat: float, lon: float) -> pd.DataFrame:
    """Lấy 7 ngày quá khứ (168h cho LSTM) + 7 ngày dự báo theo giờ từ Open-Meteo."""
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": 7,
        "forecast_days": 7,
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
    w_resp = requests.get(w_url, params=w_params, timeout=15)
    w_resp.raise_for_status()
    df_w = pd.DataFrame(w_resp.json()["hourly"])
    df_w["time"] = pd.to_datetime(df_w["time"])

    # Lấy thêm chỉ số chất lượng không khí
    try:
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": lat,
            "longitude": lon,
            "past_days": 7,
            "forecast_days": 5,
            "hourly": "pm2_5,pm10,us_aqi",
            "timezone": "Asia/Bangkok",
        }
        aq_resp = requests.get(aq_url, params=aq_params, timeout=12)
        aq_resp.raise_for_status()
        df_aq = pd.DataFrame(aq_resp.json()["hourly"])
        df_aq["time"] = pd.to_datetime(df_aq["time"])
        df = pd.merge(df_w, df_aq, on="time", how="left")
    except Exception:
        df = df_w.copy()
        df["pm2_5"] = 22.0
        df["pm10"] = 38.0
        df["us_aqi"] = 52.0

    for col in [
        "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "precipitation", "cloud_cover", "pm2_5", "pm10", "us_aqi",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(method="linear").bfill().ffill()

    return engineer_features(df)


def load_offline_fallback() -> pd.DataFrame:
    """Dùng 14 ngày cuối của tập dữ liệu lịch sử nếu máy demo mất kết nối Internet."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    return df.tail(14 * 24).reset_index(drop=True)


def aqi_status_vi(aqi: float) -> str:
    if aqi <= 50:
        return "PM2.5: Trong lành (Xanh)"
    if aqi <= 100:
        return "PM2.5: Trung bình (Vàng)"
    if aqi <= 150:
        return "PM2.5: Kém cho nhóm nhạy cảm"
    return "PM2.5: Ô nhiễm (Đỏ)"


def build_forecast_response(station_key: str) -> Dict[str, Any]:
    load_models()
    station = STATIONS.get(station_key, STATIONS["lach_huyen"])

    t0 = time.time()
    source_mode = "Open-Meteo Live + LSTM/XGBoost AI"
    try:
        df = fetch_realtime_open_meteo(station["lat"], station["lon"])
        # Mốc hiện tại là khoảng giờ thứ 7*24 trong chuỗi (sau 7 ngày quá khứ)
        now_local = pd.Timestamp.now().floor("h")
        past_mask = df["time"] <= now_local
        if past_mask.sum() >= LOOKBACK_HOURS:
            curr_idx = int(np.where(past_mask)[0][-1])
        else:
            curr_idx = LOOKBACK_HOURS
    except Exception as exc:
        source_mode = f"Offline Dataset + LSTM/XGBoost AI ({type(exc).__name__})"
        df = load_offline_fallback()
        curr_idx = LOOKBACK_HOURS

    # 1. Chạy suy luận LSTM 24 giờ tới từ đúng cửa sổ trượt 168 giờ gần nhất
    window_df = df.iloc[curr_idx - LOOKBACK_HOURS + 1 : curr_idx + 1]
    X_win = MODEL_CACHE["scaler_X"].transform(window_df[LSTM_FEATURES].values)
    X_tensor = np.expand_dims(X_win, axis=0).astype(np.float32)  # (1, 168, n_features)

    y_pred_sc = MODEL_CACHE["lstm"].predict(X_tensor, verbose=0)[0]  # (24, 3)
    y_pred_inv = MODEL_CACHE["scaler_y"].inverse_transform(y_pred_sc)  # °C, %, km/h

    lstm_temp_24h = [round(float(v), 1) for v in y_pred_inv[:, 0]]
    lstm_hum_24h = [round(float(np.clip(v, 10, 100)), 1) for v in y_pred_inv[:, 1]]
    lstm_wind_24h = [round(float(max(0.0, v)), 1) for v in y_pred_inv[:, 2]]

    # 2. Chạy suy luận XGBoost cho 7 ngày (từ hôm nay đến +6 ngày)
    days_payload: List[Dict[str, Any]] = []
    start_date = df.iloc[curr_idx]["time"].normalize()

    for d_offset in range(7):
        day_start_dt = start_date + pd.Timedelta(days=d_offset)
        day_end_dt = day_start_dt + pd.Timedelta(hours=23)
        day_slice = df[(df["time"] >= day_start_dt) & (df["time"] <= day_end_dt)]
        if len(day_slice) == 0:
            day_slice = df.iloc[max(0, curr_idx - 23) : curr_idx + 1]

        # Chọn mẫu đại diện (giờ hiện tại cho ngày 0, hoặc giữa ngày cho các ngày sau)
        rep_row = df.iloc[curr_idx] if d_offset == 0 else day_slice.iloc[len(day_slice) // 2]

        # XGBoost Classifier suy luận trạng thái thời tiết
        X_clf = day_slice[CLF_FEATURES].values
        probs = MODEL_CACHE["xgb"].predict_proba(X_clf)
        mean_probs = probs.mean(axis=0)
        # Ưu tiên cảnh báo nếu có giờ trong ngày đạt xác suất giông bão cao
        if probs[:, 3].max() >= 0.45:
            pred_label = 3
            confidence = round(float(probs[:, 3].max()) * 100.0, 1)
        else:
            pred_label = int(np.argmax(mean_probs))
            confidence = round(float(mean_probs[pred_label]) * 100.0, 1)

        label_info = WEATHER_LABELS[pred_label]

        # Số liệu khí tượng trong ngày
        if d_offset == 0:
            temp_val = round(float(rep_row["temperature_2m"]), 1)
            hum_val = int(round(float(rep_row["relative_humidity_2m"])))
            wind_val = round(float(rep_row["wind_speed_10m"]), 1)
            temp_hourly_12 = [lstm_temp_24h[i] for i in range(0, 24, 2)]
            wind_hourly_12 = [lstm_wind_24h[i] for i in range(0, 24, 2)]
            hourly_labels = [
                (df.iloc[curr_idx]["time"] + pd.Timedelta(hours=i + 1)).strftime("%H:00")
                for i in range(0, 24, 2)
            ]
        else:
            temp_val = round(float(day_slice["temperature_2m"].mean()), 1)
            hum_val = int(round(float(day_slice["relative_humidity_2m"].mean())))
            wind_val = round(float(day_slice["wind_speed_10m"].mean()), 1)
            step = max(1, len(day_slice) // 12)
            sub = day_slice.iloc[::step].head(12)
            temp_hourly_12 = [round(float(v), 1) for v in sub["temperature_2m"].values]
            wind_hourly_12 = [round(float(v), 1) for v in sub["wind_speed_10m"].values]
            hourly_labels = [t.strftime("%H:00") for t in sub["time"]]

        gust_val = round(float(day_slice["wind_gusts_10m"].max()), 1)
        rain_sum = round(float(day_slice["precipitation"].sum()), 1)
        pressure_val = int(round(float(rep_row["surface_pressure"])))
        aqi_val = int(round(float(rep_row["us_aqi"])))
        wind_dir_str = degrees_to_compass_vi(float(rep_row["wind_direction_10m"]))

        beaufort = wind_speed_to_beaufort(max(wind_val, gust_val * 0.75))

        day_name = "Hôm nay" if d_offset == 0 else WEEKDAY_VI[day_start_dt.weekday()]
        full_date = f"{day_start_dt.day:02d} Tháng {day_start_dt.month:02d}, {day_start_dt.year}"

        days_payload.append({
            "dayName": day_name,
            "fullDate": full_date,
            "temp": f"{temp_val}°C",
            "desc": label_info["name"],
            "icon": label_info["icon"],
            "iconColor": label_info["color"],
            "windTag": f"Gió {int(round(wind_val))} km/h",
            "humidity": f"{hum_val}%",
            "subHumidity": "Độ ẩm cao" if hum_val >= 85 else ("Khô ráo" if hum_val <= 65 else "Độ ẩm tương đối"),
            "wind": f"{wind_val} km/h",
            "windDir": wind_dir_str,
            "gust": f"{gust_val} km/h",
            "subGust": f"Cấp {beaufort['level']} ({beaufort['name_vi']})",
            "rain": f"{rain_sum} mm",
            "subRain": "Mưa lớn" if rain_sum >= 20 else ("Có mưa rào" if rain_sum >= 1 else "Ít khả năng mưa"),
            "pressure": f"{pressure_val} hPa",
            "subPressure": "Áp thấp cần theo dõi" if pressure_val < 1008 else "Khí áp ổn định",
            "aqi": f"{aqi_val} AQI",
            "subAqi": aqi_status_vi(aqi_val),
            "seaTitle": beaufort["safety_title"],
            "seaDesc": f"Gió {wind_val} km/h • {beaufort['wave_desc']}.",
            "barBadge": (
                f'<i class="fa-solid fa-triangle-exclamation"></i> Cấp {beaufort["level"]} Beaufort'
                if beaufort["is_warning"]
                else f'<i class="fa-solid fa-wind"></i> Cấp {beaufort["level"]} Beaufort'
            ),
            "barDanger": beaufort["is_warning"],
            "barText": f"{station['short_area']}: {beaufort['advisory']}",
            "aiLabel": label_info["name"],
            "aiConfidence": confidence,
            "isLstm24h": d_offset == 0,
            "hourlyLabels": hourly_labels,
            "tempHourly": temp_hourly_12,
            "windHourly": wind_hourly_12,
        })

    inference_ms = round((time.time() - t0) * 1000.0, 1)

    return {
        "stationKey": station_key,
        "stationName": station["name"],
        "gradient": station["gradient"],
        "source": source_mode,
        "updatedAt": datetime.now().strftime("%H:%M %d/%m/%Y"),
        "inferenceMs": inference_ms,
        "days": days_payload,
    }


@app.get("/")
def serve_dashboard():
    return FileResponse(DASHBOARD_HTML)


@app.get("/api/stations")
def get_stations():
    return {
        k: {"name": v["name"], "lat": v["lat"], "lon": v["lon"]}
        for k, v in STATIONS.items()
    }


@app.get("/api/weather")
def get_weather(station: str = Query(default="lach_huyen")):
    try:
        payload = build_forecast_response(station)
        return JSONResponse(content=payload)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/metrics")
def get_metrics():
    metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        return JSONResponse(status_code=404, content={"error": "Chua co file model_metrics.json"})
    with open(metrics_path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.get("/api/explain")
def get_explainability():
    """Giải thích mô hình AI (Explainable AI) dựa trên Feature Importance của XGBoost."""
    try:
        load_models()
        xgb = MODEL_CACHE["xgb"]
        importances = xgb.feature_importances_
        total = float(np.sum(importances)) or 1.0
        items = []
        for feat, imp in zip(CLF_FEATURES, importances):
            items.append({
                "feature": feat,
                "label": FEATURE_LABELS_VI.get(feat, feat),
                "importance": round(float(imp) / total * 100.0, 2),
            })
        items.sort(key=lambda x: x["importance"], reverse=True)
        return JSONResponse(content={
            "model": "XGBoost Classifier (4 lớp trạng thái)",
            "top_features": items[:8],
            "all_features": items,
        })
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/simulate")
def simulate_scenario(inp: SimulationInput):
    """Bộ mô phỏng What-If: chạy suy luận trực tiếp trên XGBoost & Thang Beaufort."""
    try:
        load_models()
        now = datetime.now()
        hour_sin = float(np.sin(2 * np.pi * now.hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * now.hour / 24.0))
        month_sin = float(np.sin(2 * np.pi * now.month / 12.0))
        month_cos = float(np.cos(2 * np.pi * now.month / 12.0))

        row = {
            "temperature_2m": inp.temperature_2m,
            "relative_humidity_2m": inp.relative_humidity_2m,
            "surface_pressure": inp.surface_pressure,
            "wind_speed_10m": inp.wind_speed_10m,
            "wind_gusts_10m": max(inp.wind_gusts_10m, inp.wind_speed_10m * 1.25),
            "precipitation": inp.precipitation,
            "cloud_cover": inp.cloud_cover,
            "pm2_5": max(5.0, inp.us_aqi * 0.4),
            "pm10": max(10.0, inp.us_aqi * 0.65),
            "us_aqi": inp.us_aqi,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "temp_diff_3h": -2.5 if inp.precipitation >= 10 else 0.2,
            "pressure_diff_3h": inp.pressure_diff_3h,
            "humidity_diff_3h": 8.0 if inp.precipitation >= 5 else 0.0,
            "precip_sum_6h": inp.precipitation * 3.0,
            "wind_mean_6h": inp.wind_speed_10m * 0.9,
        }

        X_sim = np.array([[row[c] for c in CLF_FEATURES]], dtype=np.float32)
        probs = MODEL_CACHE["xgb"].predict_proba(X_sim)[0]
        pred_idx = int(np.argmax(probs))
        label_info = WEATHER_LABELS[pred_idx]
        beaufort = wind_speed_to_beaufort(max(inp.wind_speed_10m, inp.wind_gusts_10m * 0.75))

        return JSONResponse(content={
            "predicted_class": pred_idx,
            "label": label_info["name"],
            "icon": label_info["icon"],
            "color": label_info["color"],
            "risk": label_info["risk"],
            "confidence": round(float(probs[pred_idx]) * 100.0, 1),
            "probabilities": [
                {
                    "class_id": i,
                    "name": WEATHER_LABELS[i]["name"],
                    "color": WEATHER_LABELS[i]["color"],
                    "prob": round(float(probs[i]) * 100.0, 1),
                }
                for i in range(4)
            ],
            "beaufort": beaufort,
        })
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

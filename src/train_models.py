"""
Giai đoạn 3 & 4: Huấn luyện, đánh giá và đối sánh các mô hình Trí tuệ Nhân tạo:
  - Bài toán 1 (Hồi quy chuỗi đa biến 24h):
      + Baseline: Linear Regression (Ridge trên đặc trưng tóm tắt cửa sổ 168h)
      + Deep Learning: Mạng LSTM 2 tầng (TensorFlow/Keras) -> models/lstm_weather.keras
  - Bài toán 2 (Phân loại 4 trạng thái thời tiết):
      + XGBoost Classifier -> models/xgboost.pkl
      + Random Forest Classifier -> models/random_forest.pkl

Tuân thủ chặt chẽ nguyên tắc ML Best Practices:
  - Chia Train (70%) / Val (15%) / Test (15%) THEO MỐC THỜI GIAN (Chronological Split)
  - Fit MinMaxScaler CHỈ trên tập Train rồi mới transform Val & Test (chống Data Leakage)
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocess import (
    PROCESSED_DATA_PATH,
    LSTM_FEATURES,
    TARGET_COLS,
    CLF_FEATURES,
    LOOKBACK_HOURS,
    HORIZON_HOURS,
    create_sliding_windows,
)

# Cố định seed để tái lập kết quả
np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def extract_baseline_features(X_seq: np.ndarray) -> np.ndarray:
    """
    Trích xuất đặc trưng tóm tắt từ cửa sổ 168h cho mô hình Baseline Linear Regression:
    - Giá trị tại bước cuối (t), t-1, t-3, t-6, t-12, t-24, t-48, t-168
    - Trung bình 24h gần nhất và 168h
    """
    idx = [-1, -2, -4, -7, -13, -24, -48, 0]
    sampled = X_seq[:, idx, :].reshape(len(X_seq), -1)
    mean_24 = X_seq[:, -24:, :].mean(axis=1)
    mean_168 = X_seq.mean(axis=1)
    return np.hstack([sampled, mean_24, mean_168])


def build_lstm_model(lookback: int, n_features: int, horizon: int, n_targets: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(horizon * n_targets),
        layers.Reshape((horizon, n_targets)),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def evaluate_regression_predictions(y_true_inv: np.ndarray, y_pred_inv: np.ndarray) -> dict:
    """Tính MAE, RMSE, R2 cho từng biến mục tiêu trên toàn bộ 24 bước dự báo."""
    metrics = {}
    for idx, col in enumerate(TARGET_COLS):
        yt = y_true_inv[:, :, idx].ravel()
        yp = y_pred_inv[:, :, idx].ravel()
        metrics[col] = {
            "mae": round(float(mean_absolute_error(yt, yp)), 4),
            "rmse": round(float(root_mean_squared_error(yt, yp)), 4),
            "r2": round(float(r2_score(yt, yp)), 4),
        }
    # Đánh giá riêng bước đầu tiên (t+1h) và trung bình 24h
    yt_1h = y_true_inv[:, 0, 0]
    yp_1h = y_pred_inv[:, 0, 0]
    metrics["temp_t1h"] = {
        "mae": round(float(mean_absolute_error(yt_1h, yp_1h)), 4),
        "rmse": round(float(root_mean_squared_error(yt_1h, yp_1h)), 4),
        "r2": round(float(r2_score(yt_1h, yp_1h)), 4),
    }
    return metrics


def plot_eda_correlation(df: pd.DataFrame):
    cols = [
        "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "wind_speed_10m", "wind_gusts_10m", "precipitation", "pm2_5", "us_aqi",
    ]
    corr = df[cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Ma tran tuong quan cac bien khi tuong Hai Phong", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "eda_correlation.png"), dpi=150)
    plt.close()


def plot_lstm_vs_actual(y_true_inv: np.ndarray, y_pred_inv: np.ndarray):
    """Vẽ biểu đồ so khớp Thực tế vs Dự báo LSTM cho 200 giờ đầu tập Test (bước t+1h và t+24h)."""
    n_plot = min(200, len(y_true_inv))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    titles = [
        ("Nhiet do (Temperature - °C)", 0, "#06b6d4"),
        ("Do am tuong doi (Humidity - %)", 1, "#3b82f6"),
        ("Toc do gio (Wind Speed - km/h)", 2, "#f59e0b"),
    ]
    for ax, (title, idx, color) in zip(axes, titles):
        ax.plot(y_true_inv[:n_plot, 0, idx], label="Thuc te (Actual)", color="#10b981", linewidth=1.8)
        ax.plot(y_pred_inv[:n_plot, 0, idx], label="Du bao LSTM (t+1h)", color=color, linestyle="--", linewidth=1.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Buoc thoi gian tren Tap Kiem thu (Gio)")
    plt.suptitle("Doi soat Thuc te vs Du bao Mang LSTM tren Test Set (Hai Phong)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "lstm_actual_vs_pred.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    labels = ["Nang quang", "Am u", "Mua rao", "Giong bao"]
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Nhan du doan (Predicted)")
    plt.ylabel("Nhan thuc te (Actual)")
    plt.title("Ma tran nham lan (Confusion Matrix) - XGBoost Classifier", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix_xgb.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("============================================================")
    print("  HUAN LUYEN MO HINH AI DU BAO THOI TIET HAI PHONG (N04.14) ")
    print("============================================================")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    print(f"Tong du lieu sach: {len(df):,} gio ({df['time'].min()} -> {df['time'].max()})")

    # Xuất biểu đồ EDA tương quan
    plot_eda_correlation(df)

    # ------------------------------------------------------------------
    # CHIA TAP CHRONOLOGICAL SPLIT: 70% Train / 15% Val / 15% Test
    # ------------------------------------------------------------------
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:n_val].reset_index(drop=True)
    df_test = df.iloc[n_val:].reset_index(drop=True)
    print(f"Chronological Split -> Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

    # ------------------------------------------------------------------
    # BAI TOAN 1: HOI QUY CHUOI DA BIEN (BASELINE LINEAR + LSTM)
    # ------------------------------------------------------------------
    print("\n--- BAI TOAN 1: HOI QUY CHUOI THOI GIAN (SLIDING WINDOW 168h -> 24h) ---")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # STRICT FEATURIZATION ORDERING: chỉ fit trên Train!
    X_train_sc = scaler_X.fit_transform(df_train[LSTM_FEATURES].values)
    y_train_sc = scaler_y.fit_transform(df_train[TARGET_COLS].values)

    X_val_sc = scaler_X.transform(df_val[LSTM_FEATURES].values)
    y_val_sc = scaler_y.transform(df_val[TARGET_COLS].values)

    X_test_sc = scaler_X.transform(df_test[LSTM_FEATURES].values)
    y_test_sc = scaler_y.transform(df_test[TARGET_COLS].values)

    joblib.dump(scaler_X, os.path.join(MODELS_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, "scaler_y.pkl"))

    # Sliding Window (stride=2 trên train để huấn luyện nhanh trên CPU mà vẫn phủ toàn bộ mẫu)
    X_tr_seq, y_tr_seq = create_sliding_windows(X_train_sc, y_train_sc, LOOKBACK_HOURS, HORIZON_HOURS, stride=2)
    X_va_seq, y_va_seq = create_sliding_windows(X_val_sc, y_val_sc, LOOKBACK_HOURS, HORIZON_HOURS, stride=2)
    X_te_seq, y_te_seq = create_sliding_windows(X_test_sc, y_test_sc, LOOKBACK_HOURS, HORIZON_HOURS, stride=1)
    print(f"Tensor 3D shapes -> Train: {X_tr_seq.shape} | Val: {X_va_seq.shape} | Test: {X_te_seq.shape}")

    # Inverse transform y_test gốc để đánh giá chính xác theo đơn vị thực (°C, %, km/h)
    y_te_inv = scaler_y.inverse_transform(y_te_seq.reshape(-1, len(TARGET_COLS))).reshape(y_te_seq.shape)

    # 1A. Baseline Model: Linear Regression (Ridge)
    print("[1/2] Dang huan luyen Baseline Linear Regression (Ridge)...")
    t0 = time.time()
    X_tr_base = extract_baseline_features(X_tr_seq)
    X_te_base = extract_baseline_features(X_te_seq)
    baseline_lr = Ridge(alpha=1.0)
    baseline_lr.fit(X_tr_base, y_tr_seq.reshape(len(y_tr_seq), -1))
    y_pred_base_sc = baseline_lr.predict(X_te_base).reshape(y_te_seq.shape)
    y_pred_base_inv = scaler_y.inverse_transform(y_pred_base_sc.reshape(-1, len(TARGET_COLS))).reshape(y_te_seq.shape)
    baseline_metrics = evaluate_regression_predictions(y_te_inv, y_pred_base_inv)
    joblib.dump(baseline_lr, os.path.join(MODELS_DIR, "baseline_lr.pkl"))
    print(f"      -> Baseline xong ({time.time()-t0:.1f}s) | Temp 24h MAE={baseline_metrics['temperature_2m']['mae']}°C, R2={baseline_metrics['temperature_2m']['r2']}")

    # 1B. Deep Learning: Mạng LSTM (TensorFlow/Keras)
    print("[2/2] Dang huan luyen Mang LSTM (TensorFlow/Keras)...")
    t0 = time.time()
    lstm_model = build_lstm_model(LOOKBACK_HOURS, len(LSTM_FEATURES), HORIZON_HOURS, len(TARGET_COLS))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-4),
    ]
    history = lstm_model.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_va_seq, y_va_seq),
        epochs=18,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )
    lstm_train_time = round(time.time() - t0, 1)

    t_inf = time.time()
    y_pred_lstm_sc = lstm_model.predict(X_te_seq, batch_size=128, verbose=0)
    inf_ms = round(((time.time() - t_inf) / len(X_te_seq)) * 1000.0, 2)

    y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm_sc.reshape(-1, len(TARGET_COLS))).reshape(y_te_seq.shape)
    lstm_metrics = evaluate_regression_predictions(y_te_inv, y_pred_lstm_inv)

    lstm_path = os.path.join(MODELS_DIR, "lstm_weather.keras")
    lstm_model.save(lstm_path)
    plot_lstm_vs_actual(y_te_inv, y_pred_lstm_inv)
    print(f"      -> LSTM xong ({lstm_train_time}s) | Temp 24h MAE={lstm_metrics['temperature_2m']['mae']}°C, RMSE={lstm_metrics['temperature_2m']['rmse']}°C, R2={lstm_metrics['temperature_2m']['r2']}")
    print(f"      -> LSTM (t+1h) Temp MAE={lstm_metrics['temp_t1h']['mae']}°C, R2={lstm_metrics['temp_t1h']['r2']}")

    # ------------------------------------------------------------------
    # BAI TOAN 2: PHAN LOAI TRANG THAI THOI TIET (XGBOOST + RANDOM FOREST)
    # ------------------------------------------------------------------
    print("\n--- BAI TOAN 2: PHAN LOAI 4 TRANG THAI THOI TIET (XGBOOST & RANDOM FOREST) ---")
    X_clf_train = df_train[CLF_FEATURES].values
    y_clf_train = df_train["weather_label"].values
    X_clf_val = df_val[CLF_FEATURES].values
    y_clf_val = df_val["weather_label"].values
    X_clf_test = df_test[CLF_FEATURES].values
    y_clf_test = df_test["weather_label"].values

    # 2A. XGBoost Classifier
    print("[1/2] Dang huan luyen XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_clf_train, y_clf_train, eval_set=[(X_clf_val, y_clf_val)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_clf_test)
    xgb_acc = round(float(accuracy_score(y_clf_test, y_pred_xgb)) * 100.0, 2)
    xgb_f1_weighted = round(float(f1_score(y_clf_test, y_pred_xgb, average="weighted")) * 100.0, 2)
    xgb_f1_macro = round(float(f1_score(y_clf_test, y_pred_xgb, average="macro")) * 100.0, 2)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))
    plot_confusion_matrix(y_clf_test, y_pred_xgb)
    print(f"      -> XGBoost Test Accuracy={xgb_acc}%, Weighted F1={xgb_f1_weighted}%, Macro F1={xgb_f1_macro}%")
    print(classification_report(y_clf_test, y_pred_xgb, zero_division=0))

    # 2B. Random Forest Classifier
    print("[2/2] Dang huan luyen Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_clf_train, y_clf_train)
    y_pred_rf = rf_model.predict(X_clf_test)
    rf_acc = round(float(accuracy_score(y_clf_test, y_pred_rf)) * 100.0, 2)
    rf_f1_weighted = round(float(f1_score(y_clf_test, y_pred_rf, average="weighted")) * 100.0, 2)
    rf_f1_macro = round(float(f1_score(y_clf_test, y_pred_rf, average="macro")) * 100.0, 2)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))
    print(f"      -> Random Forest Test Accuracy={rf_acc}%, Weighted F1={rf_f1_weighted}%, Macro F1={rf_f1_macro}%")

    # ------------------------------------------------------------------
    # LUU METRICS & CONFIG
    # ------------------------------------------------------------------
    summary = {
        "dataset": {
            "total_rows": int(len(df)),
            "train_rows": int(len(df_train)),
            "val_rows": int(len(df_val)),
            "test_rows": int(len(df_test)),
            "start_time": str(df["time"].min()),
            "end_time": str(df["time"].max()),
            "lookback_hours": LOOKBACK_HOURS,
            "horizon_hours": HORIZON_HOURS,
        },
        "regression": {
            "baseline_linear": baseline_metrics,
            "lstm": lstm_metrics,
            "lstm_train_seconds": lstm_train_time,
            "lstm_inference_ms": inf_ms,
            "epochs_trained": len(history.history["loss"]),
        },
        "classification": {
            "xgboost": {
                "accuracy": xgb_acc,
                "f1_weighted": xgb_f1_weighted,
                "f1_macro": xgb_f1_macro,
            },
            "random_forest": {
                "accuracy": rf_acc,
                "f1_weighted": rf_f1_weighted,
                "f1_macro": rf_f1_macro,
            },
        },
    }

    with open(os.path.join(OUTPUTS_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "lstm_features": LSTM_FEATURES,
            "target_cols": TARGET_COLS,
            "clf_features": CLF_FEATURES,
            "lookback_hours": LOOKBACK_HOURS,
            "horizon_hours": HORIZON_HOURS,
        }, f, ensure_ascii=False, indent=2)

    print("\n[HOAN TAT] Da luu toan bo mo hinh vao 'models/' va ket qua vao 'outputs/model_metrics.json'.")


if __name__ == "__main__":
    main()

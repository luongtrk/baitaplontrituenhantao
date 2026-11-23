# Hệ Thống Dự Báo Thời Tiết & Cảnh Báo An Toàn Hàng Hải Hải Phòng (N04.14)

Hệ thống dự báo khí tượng thông minh khu vực Hải Phòng ứng dụng Trí tuệ Nhân tạo (**LSTM** + **XGBoost** / **Random Forest**) kết hợp quy đổi **Thang cấp gió Beaufort** nhằm cảnh báo sớm an toàn luồng tàu và hoạt động cảng biển.

## Cấu trúc dự án

```text
├── app.py                        # Backend FastAPI suy luận AI thời gian thực & serve Web Dashboard
├── ai_weather_dashboard.html     # Giao diện Web Dashboard trực quan (Chart.js + Beaufort)
├── run_demo.bat                  # File chạy nhanh 1-click trên Windows để trình chiếu / demo
├── requirements.txt              # Danh sách thư viện Python (cài trong .venv)
├── src/
│   ├── collect_data.py           # Giai đoạn 1: Thu thập dữ liệu lịch sử 2 năm từ Open-Meteo API
│   ├── preprocess.py             # Giai đoạn 2: Làm sạch, nội suy tuyến tính, Sliding Window 168h
│   ├── train_models.py           # Giai đoạn 3 & 4: Huấn luyện & đối sánh LSTM, Linear, XGBoost, RF
│   ├── beaufort.py               # Quy đổi tốc độ gió sang cấp gió Beaufort & cảnh báo hàng hải
│   └── build_notebook.py         # Tạo Jupyter Notebook phân tích EDA & kết quả đánh giá
├── data/
│   ├── raw_weather.csv           # Dữ liệu thô 17.544 giờ (2024–2025)
│   └── data_processed.csv        # Dữ liệu sạch đã trích xuất đặc trưng & gán nhãn
├── models/
│   ├── lstm_weather.keras        # Trọng số mạng nơ-ron sâu LSTM dự báo đa biến 24h
│   ├── xgboost.pkl               # Mô hình XGBoost phân loại 4 trạng thái thời tiết
│   ├── random_forest.pkl         # Mô hình Random Forest đối sánh
│   ├── baseline_lr.pkl           # Mô hình Baseline Linear Regression (Ridge)
│   ├── scaler_X.pkl              # MinMaxScaler đầu vào (fit riêng trên Train Set)
│   └── scaler_y.pkl              # MinMaxScaler đầu ra
├── outputs/
│   ├── model_metrics.json        # Bảng chỉ số đánh giá MAE, RMSE, R², Accuracy, F1-Score
│   ├── eda_correlation.png       # Biểu đồ ma trận tương quan các biến khí tượng
│   ├── lstm_actual_vs_pred.png   # Biểu đồ đối soát Thực tế vs Dự báo LSTM
│   └── confusion_matrix_xgb.png  # Ma trận nhầm lẫn XGBoost Classifier
└── notebooks/
    └── EDA_and_Training.ipynb    # Notebook thực nghiệm EDA & đánh giá mô hình
```

## Hướng dẫn chạy nhanh (Demo)

### Cách 1: Chạy 1-click trên Windows
Nhấp đúp vào file **`run_demo.bat`**, sau đó mở trình duyệt tại:
**http://127.0.0.1:8000**

### Cách 2: Chạy bằng dòng lệnh (PowerShell)
```powershell
# 1. Chạy Web Dashboard & API suy luận AI thời gian thực
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### Cách chạy lại toàn bộ pipeline huấn luyện (nếu cần)
```powershell
.\.venv\Scripts\python.exe -m src.collect_data
.\.venv\Scripts\python.exe -m src.preprocess
.\.venv\Scripts\python.exe -m src.train_models
.\.venv\Scripts\python.exe -m src.build_notebook
```

## API Endpoints chính
- `GET /` — Giao diện Web Dashboard dự báo 24 giờ & 7 ngày tới cho 5 trạm Hải Phòng.
- `GET /api/weather?station=lach_huyen` — Lấy dữ liệu thời gian thực từ Open-Meteo, chạy suy luận LSTM (24h) + XGBoost (phân loại) + quy đổi cấp gió Beaufort.
- `GET /api/metrics` — Xem bảng đối chuẩn hiệu năng thực nghiệm các mô hình trên tập kiểm thử (Test Set).
- `GET /docs` — Tài liệu Swagger API tự động của FastAPI.

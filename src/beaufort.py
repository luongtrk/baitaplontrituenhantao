"""
Quy đổi tốc độ gió (km/h) sang Thang cấp gió Beaufort (0 - 12)
và sinh cảnh báo an toàn luồng tàu / cảng biển khu vực Hải Phòng.
"""

from typing import Dict, Any

BEAUFORT_SCALE = [
    (0, 1.0, "Gió lặng", "Calm", "Mặt biển phẳng lặng như gương", False),
    (1, 5.5, "Gió rất nhẹ", "Light Air", "Sóng lăn tăn nhẹ, tàu thuyền hoạt động hoàn hảo", False),
    (2, 11.9, "Gió nhẹ", "Light Breeze", "Sóng nhỏ, luồng tàu êm dịu", False),
    (3, 19.7, "Gió yếu", "Gentle Breeze", "Sóng cao 0.5 - 1.0m, tàu thuyền lưu thông tối ưu", False),
    (4, 28.7, "Gió vừa", "Moderate Breeze", "Sóng 1.0 - 1.5m, tàu thuyền cập bến bình thường", False),
    (5, 38.8, "Gió tươi", "Fresh Breeze", "Sóng 1.5 - 2.0m, tàu nhỏ và sà lan cần giữ khoảng cách an toàn", True),
    (6, 49.9, "Gió mạnh", "Strong Breeze", "Sóng 2.0 - 3.0m, cảnh báo cẩu bờ và hạn chế tàu nhỏ xuất bến", True),
    (7, 61.8, "Gió gần bão", "Near Gale", "Sóng 3.0 - 4.0m, biển động mạnh, tạm dừng phà và cấm tàu nhỏ", True),
    (8, 74.9, "Gió bão", "Gale", "Sóng 4.0 - 5.5m, biển động rất mạnh, cấm xuất bến", True),
    (9, 88.1, "Gió bão mạnh", "Strong Gale", "Sóng 5.5 - 7.0m, nguy hiểm đặc biệt, neo đậu khẩn cấp", True),
    (10, 102.4, "Bão", "Storm", "Sóng 7.0 - 9.0m, đóng cửa cảng khẩn cấp", True),
    (11, 117.4, "Bão dữ dội", "Violent Storm", "Sóng trên 9.0m, rủi ro thiên tai cấp độ cao", True),
    (12, 999.0, "Siêu bão", "Hurricane", "Sức tàn phá cực lớn, toàn bộ tàu thuyền trú ẩn tuyệt đối", True),
]

WEATHER_LABELS = {
    0: {"name": "Nắng quang mây", "icon": "fa-sun", "color": "#fbbf24", "risk": "An toàn"},
    1: {"name": "Âm u / Nhiều mây", "icon": "fa-cloud", "color": "#cbd5e1", "risk": "Bình thường"},
    2: {"name": "Mưa rào", "icon": "fa-cloud-rain", "color": "#60a5fa", "risk": "Cần lưu ý"},
    3: {"name": "Giông bão / Gió mạnh", "icon": "fa-cloud-bolt", "color": "#f87171", "risk": "Nguy hiểm"},
}


def wind_speed_to_beaufort(wind_kmh: float) -> Dict[str, Any]:
    """Chuyển đổi tốc độ gió (km/h) sang thông tin cấp gió Beaufort & cảnh báo hàng hải."""
    speed = max(0.0, float(wind_kmh))
    for level, max_speed, name_vi, name_en, wave_desc, is_warning in BEAUFORT_SCALE:
        if speed <= max_speed:
            if level <= 3:
                safety_title = f"An toàn hàng hải: Cấp {level}"
                advisory = "Luồng tàu thông suốt, mọi hoạt động bốc dỡ và tàu hoa tiêu hoạt động bình thường."
            elif level == 4:
                safety_title = f"An toàn hàng hải: Cấp {level}"
                advisory = "Điều kiện luồng ổn định, tàu container và sà lan ra vào bến bình thường."
            elif level == 5:
                safety_title = f"Cảnh báo luồng tàu: Cấp {level}"
                advisory = "Gió tươi giật mạnh, tàu lai dắt tăng cường hỗ trợ tàu lớn; tàu nhỏ chú ý giữ khoảng cách."
            elif level == 6:
                safety_title = f"Cảnh báo nguy hiểm: Cấp {level}"
                advisory = "Biển động, giảm tốc độ cẩu bờ STS và hạn chế sà lan kéo tự do rời bến."
            else:
                safety_title = f"BÁO ĐỘNG HÀNG HẢI: Cấp {level}"
                advisory = "Gió rất mạnh, biển động dữ dội. Tạm dừng xuất bến, toàn bộ phương tiện vào âu neo đậu tránh trú."

            return {
                "level": level,
                "name_vi": name_vi,
                "name_en": name_en,
                "wave_desc": wave_desc,
                "is_warning": is_warning,
                "is_danger": level >= 6,
                "safety_title": safety_title,
                "advisory": advisory,
            }
    return wind_speed_to_beaufort(120.0)


def degrees_to_compass_vi(deg: float) -> str:
    """Chuyển góc hướng gió (0-360°) sang tên hướng tiếng Việt."""
    directions = [
        "Bắc", "Đông Bắc", "Đông", "Đông Nam",
        "Nam", "Tây Nam", "Tây", "Tây Bắc",
    ]
    idx = int(((float(deg) % 360) + 22.5) // 45) % 8
    return f"{directions[idx]} ({int(round(deg)) % 360:03d}°)"

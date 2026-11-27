# Kết quả MobileNetV3Large trên BRACOL (4 lớp: healthy, miner, phoma, rust — 5-Fold CV + 8-View TTA)

- Dữ liệu BRACOL thô: 1,664 dòng CSV (loại 60 ảnh đồng nhiễm 2 bệnh) → 1,604 ảnh đơn nhãn → lọc trùng xoay/lật 4× bằng pHash còn **395 lá Arabica độc lập** (tỷ lệ mất cân bằng 1.49:1).
- **5-Fold CV (Single-View, mean ± std):** Accuracy `0.9468 ± 0.0373` | Macro-F1 `0.9479 ± 0.0375`.
- **5-Fold CV (8-View TTA, mean ± std):** Accuracy `0.9595 ± 0.0315` | Macro-F1 `0.9605 ± 0.0304` | Weighted-F1 `0.9595 ± 0.0315`.
- **Out-Of-Fold (OOF, toàn bộ 395 lá độc lập):** Single-View Acc `0.9468` / Macro-F1 `0.9475` → **8-View TTA Acc `0.9595` / Macro-F1 `0.9607`**.
- Thời gian huấn luyện 5 fold: 8.73 phút trên Tesla T4.
- Lớp OOF F1 thấp nhất: `phoma` (`0.9483`, support=118).
- Nhầm nhiều nhất (OOF + TTA): phoma → healthy (6 lá).
- Suy luận CPU độc lập và TorchScript (Fold 0) đã kiểm tra khớp đầu ra.

## Hạn chế và hướng phát triển
- BRACOL chụp lá trên nền trắng trong điều kiện kiểm soát; cần kiểm thử thêm trên ảnh chụp ngoài thực địa tự nhiên trước khi triển khai thực tế.
- 60 ảnh đồng nhiễm 2 bệnh đã được tách khỏi bài toán đơn nhãn 4 lớp; hướng mở rộng tự nhiên là bài toán đa nhãn (multi-label BCEWithLogitsLoss).
- Chưa đo latency trên thiết bị di động.
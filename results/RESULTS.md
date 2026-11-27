# Kết quả MobileNetV3Large trên JMuBEN (4 lớp bệnh thuần, 0% Domain Shift, 5-Fold CV + 8-View TTA — v6)

- Dữ liệu JMuBEN thô: 58,549 ảnh (loại 18,983 ảnh Healthy vì chỉ có 2 lá gốc) → 1,347 sau pHash → **212 nhóm lá bệnh độc lập** (0 nhóm xung đột nhãn).
- **5-Fold CV (Single-View, mean ± std):** Accuracy `0.9392 ± 0.0358` | Macro-F1 `0.9262 ± 0.0478`.
- **5-Fold CV (8-View TTA, mean ± std):** Accuracy `0.9490 ± 0.0293` | Macro-F1 `0.9343 ± 0.0490` | Weighted-F1 `0.9498 ± 0.0286`.
- **Out-Of-Fold (OOF, toàn bộ 212 nhóm độc lập):** Single-View Acc `0.9434` / Macro-F1 `0.9390` → **8-View TTA Acc `0.9528` / Macro-F1 `0.9469`**.
- Kiểm chứng chống leak xuyên fold: max cosine = `0.8499` (< 0.85), mean frozen 1-NN = `0.9125`.
- Thời gian huấn luyện 5 fold: 8.06 phút trên Tesla T4.
- Lớp OOF F1 thấp nhất: `miner` (`0.9184`, support=50).
- Nhầm nhiều nhất (OOF + TTA): miner → phoma (4 ảnh).
- Suy luận CPU độc lập và TorchScript (Fold 0) đã kiểm tra khớp đầu ra.

## Hạn chế và lưu ý khoa học
- Bài toán chỉ phân loại 4 loại bệnh trong phạm vi JMuBEN (không phân biệt lá khỏe vì `JMuBEN/Healthy` chỉ có 2 lá gốc — xem `healthy_2leaf_evidence.png`).
- Số nhóm độc lập của `cercospora` còn nhỏ (21 nhóm); ảnh JMuBEN là ảnh cắt cận 128×128 nên chưa đại diện ảnh thực địa ngoài vườn.
- Chưa đo latency trên thiết bị di động.
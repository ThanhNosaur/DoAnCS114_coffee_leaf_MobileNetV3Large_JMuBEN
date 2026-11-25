# Kết quả MobileNetV3Large trên JMuBEN (+ RoCoLe Healthy) — 5-Fold CV theo nhóm

- Dữ liệu thô: 59,340 ảnh (JMuBEN + RoCoLe coffee___healthy) → 2,106 sau pHash → 569 nhóm.
- **5-Fold CV (5 lớp, mean ± std):** Accuracy `0.9616 ± 0.0083` | Macro-F1 `0.9124 ± 0.0174` | Weighted-F1 `0.9621 ± 0.0085`.
- **5-Fold CV (ảnh bệnh JMuBEN, mô hình 5 lớp):** Accuracy `0.8968 ± 0.0212` | Macro-F1 `0.8905 ± 0.0217`.
- **Out-Of-Fold (OOF, toàn bộ 569 nhóm):** Accuracy `0.9613` | Macro-F1 `0.9084` | Macro-F1 trên ảnh bệnh JMuBEN `0.8855`.
- Kiểm chứng chống leak xuyên fold JMuBEN: max cosine = `0.8499` (< 0.85), mean frozen 1-NN = `0.9140`.
- Thời gian huấn luyện 5 fold: 15.09 phút trên Tesla T4.
- Lớp OOF F1 thấp nhất: `cercospora` (`0.8444`, support=21).
- Nhầm nhiều nhất (OOF): miner → phoma (7 ảnh).
- Suy luận CPU độc lập và TorchScript (Fold 0) đã kiểm tra khớp đầu ra.

## Hạn chế và lưu ý khoa học
- **Domain shift lớp `healthy`:** `JMuBEN/Healthy` chỉ có 2 lá gốc nên thay bằng RoCoLe `coffee___healthy` (ảnh thực địa Robusta). Cần trình bày song song kết quả 4 lớp bệnh thuần JMuBEN.
- Kiểm tra nhóm dùng pHash, embedding và mã lá RoCoLe. Nguồn gốc lá JMuBEN được suy ra từ ảnh. Các ngưỡng tương đồng không xác nhận được danh tính của mọi lá.
- Chưa đo latency trên thiết bị di động và chưa đánh giá trên ảnh bệnh ngoài thực địa.
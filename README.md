# Chẩn đoán Bệnh Lá Cà Phê Arabica — MobileNetV3Large trên BRACOL

Huấn luyện trên Kaggle (Tesla T4) bằng [kaggle/mobilenetv3large_bracol.ipynb](kaggle/mobilenetv3large_bracol.ipynb), fine-tune **MobileNetV3Large** (ImageNet1K V2), **5-Fold StratifiedGroupKFold Cross-Validation + 8-View Dihedral TTA**, seed 42, ảnh 224×224.

## 1. Dữ liệu & Quy trình khoa học (BRACOL — 0% Domain Shift, 0% Data Leakage)
1. **Bộ dữ liệu BRACOL (`badasstechie/coffee-leaf-diseases` — Brazilian Arabica Coffee Leaf):**
   - Toàn bộ 1,664 dòng trong `train_classes.csv` + `test_classes.csv` gồm ảnh lá cà phê Arabica chụp thật trên nền trắng đồng nhất (0% domain shift).
   - Loại 60 ảnh đồng nhiễm 2 bệnh cùng lúc để giữ bài toán phân loại đơn nhãn sạch **4 lớp**: `healthy`, `miner`, `phoma`, `rust` (`1,604` ảnh đơn nhãn).
2. **Chống Data Leakage bằng Dihedral 64-bit DCT pHash (Hamming ≤ 4):**
   - Kiểm toán phát hiện tập ảnh gốc chứa các bản sao xoay/lật 4× của cùng một chiếc lá (xem [results/duplicate_evidence.png](results/duplicate_evidence.png)).
   - Gộp nhóm liên thông theo pHash 8 hướng, cách ly 2 nhóm xung đột nhãn → còn **395 lá Arabica độc lập** rất cân bằng (tỷ lệ mất cân bằng chỉ **1.49:1**).
   - Chia **5-Fold StratifiedGroupKFold** theo nhóm độc lập (0 giao nhóm, 0 giao canonical pHash giữa các fold; xem kiểm chứng trực quan [results/cross_split_nearest_pairs.png](results/cross_split_nearest_pairs.png)).
3. **Huấn luyện 5-Fold CV + Kiểm soát Overfitting + 8-View Dihedral TTA:**
   - Mỗi fold tách 1 fold làm `test`, 1 fold làm `val` (giữ đúng **1 ảnh đại diện / lá độc lập**), 3 fold làm `train`.
   - Fine-tune 2 giai đoạn (`warm-up` → `fine-tune`), `CosineAnnealingLR` (`1e-4 → 1e-6`), `label_smoothing = 0.05`, `weight_decay = 1e-3`, `RandomResizedCrop`, `RandomRotation(20)`, `ColorJitter`, `RandomErasing(0.15)`. Xem [results/learning_curves.png](results/learning_curves.png).
   - Đối chứng **Single-View vs 8-View Dihedral TTA** (trung bình xác suất trên 4 góc xoay × 2 phép lật).

## 2. Kết quả 5-Fold CV & Out-Of-Fold (395 lá độc lập, Tesla T4, 8.73 phút)

### Đối chứng Single-View vs 8-View Dihedral TTA
| Phương pháp suy luận | 5-Fold CV Accuracy (`mean ± std`) | 5-Fold CV Macro-F1 (`mean ± std`) | OOF Accuracy (395 lá) | OOF Macro-F1 (395 lá) |
| --- | --- | --- | --- | --- |
| Single-View | `0.9468 ± 0.0373` | `0.9479 ± 0.0375` | `0.9468` | `0.9475` |
| **8-View Dihedral TTA** | **`0.9595 ± 0.0315`** | **`0.9605 ± 0.0304`** | **`0.9595`** | **`0.9607`** |

### Kết quả từng lớp (Out-Of-Fold + 8-View TTA, toàn bộ 395 lá độc lập)
| Lớp | Precision | Recall | F1 | Số lá (Support) |
| --- | --- | --- | --- | --- |
| `healthy` | 0.9252 | 1.0000 | 0.9612 | 99 |
| `miner` | 0.9789 | 0.9394 | 0.9588 | 99 |
| `phoma` | 0.9649 | 0.9322 | 0.9483 | 118 |
| `rust` | 0.9747 | 0.9747 | 0.9747 | 79 |
| **Macro avg** | **0.9609** | **0.9616** | **0.9607** | **395** |
| **Weighted avg** | **0.9604** | **0.9595** | **0.9594** | **395** |

- Chi tiết từng fold xem tại [results/cv_fold_results.csv](results/cv_fold_results.csv).
- Lớp OOF F1 thấp nhất: `phoma` (`0.9483`); nhầm nhiều nhất: `phoma → healthy` (6 lá).
- Giải thích mô hình [results/gradcam.png](results/gradcam.png) cho thấy mạng tập trung trực tiếp vào vùng tổn thương bệnh trên phiến lá, không dựa vào nền trắng.
- Suy luận CPU độc lập và TorchScript (Fold 0) đã kiểm tra khớp đầu ra (`inference_verified: true`, `torchscript_verified: true`).

## 3. Hạn chế và hướng phát triển
- BRACOL chụp lá trên nền trắng trong điều kiện kiểm soát; cần kiểm thử thêm trên ảnh chụp ngoài thực địa tự nhiên trước khi triển khai thực tế.
- 60 ảnh đồng nhiễm 2 bệnh đã được tách khỏi bài toán đơn nhãn 4 lớp; hướng mở rộng tự nhiên là bài toán đa nhãn (`BCEWithLogitsLoss`).
- Chưa đo latency trên thiết bị di động.

## 4. Nội dung repo
- `kaggle/`: notebook ([mobilenetv3large_bracol.ipynb](kaggle/mobilenetv3large_bracol.ipynb)) và `kernel-metadata.json`.
- `results/`: [RESULTS.md](results/RESULTS.md), `cv_fold_results.csv`, `metrics.json`, `data_audit.json`, `classification_report.csv`, `duplicate_evidence.png`, `cross_split_nearest_pairs.png`, `confusion_matrix.png`, `learning_curves.png`, `gradcam.png`, `misclassified.png`, `kaggle_run.log`.
- Checkpoint `best.pt`, TorchScript `.ts` và `report_assets.zip` không đưa lên repo (file lớn); lấy từ output của Kaggle kernel `ngtunthanh/mobilenetv3large-bracol`.

# Chẩn đoán 4 bệnh lá cà phê Arabica — MobileNetV3Large trên JMuBEN (v6: Thuần JMuBEN, 5-Fold CV + 8-View TTA)

Huấn luyện trên Kaggle (Tesla T4) bằng [kaggle/mobilenetv3large_jmuben.ipynb](kaggle/mobilenetv3large_jmuben.ipynb), fine-tune MobileNetV3Large (ImageNet1K V2), 5-Fold StratifiedGroupKFold Cross-Validation + 8-View Dihedral TTA, seed 42, ảnh 224x224.

## 1. Quy trình khoa học & kỹ thuật (v6)
1. **Thuần 1 nguồn dữ liệu JMuBEN (0% Domain Shift):**
   - Kiểm toán toàn bộ 58,549 ảnh thô phát hiện **18,983 ảnh `JMuBEN/Healthy`** sau lọc pHash chỉ còn 15 ảnh và thực chất sinh ra từ **đúng 2 lá gốc duy nhất** (xem bằng chứng trực quan [results/healthy_2leaf_evidence.png](results/healthy_2leaf_evidence.png)).
   - Loại `Healthy` khỏi benchmark (không trộn nguồn ảnh thực địa bên ngoài như RoCoLe để tránh shortcut learning / 100% ảo do khác nền ảnh), tập trung đánh giá trung thực bài toán **phân loại 4 bệnh lá cà phê**: `cercospora`, `leaf_rust`, `miner`, `phoma`.
2. **Chống Data Leakage 2 tầng:**
   - Tầng 1: Dihedral 64-bit DCT pHash (Hamming ≤ 4) lọc 39,566 ảnh 4 bệnh xuống 1,347 ảnh `canonical`.
   - Tầng 2: Connected components trên frozen ImageNet MobileNetV3Large embedding 8 hướng (`cosine ≥ 0.85`) gom thành **212 nhóm lá độc lập** (0 nhóm xung đột nhãn).
   - Kiểm chứng trên toàn bộ 5 fold: 0 giao nhóm, 0 giao canonical pHash, **max cross-fold cosine = `0.8499` (< 0.85)**. Xem [results/cross_split_nearest_pairs.png](results/cross_split_nearest_pairs.png).
3. **5-Fold StratifiedGroupKFold CV + Regularization + 8-View Dihedral TTA:**
   - Mỗi fold tách 1 fold làm `test`, 1 fold làm `val` (1 ảnh đại diện / nhóm độc lập), 3 fold làm `train`.
   - Kiểm soát overfitting bằng `CosineAnnealingLR` (`1e-4 → 1e-6`), `label_smoothing = 0.05`, `weight_decay = 1e-3`, `RandomResizedCrop`, `RandomRotation(20)`, `ColorJitter`, `RandomErasing(0.15)`. Xem [results/learning_curves.png](results/learning_curves.png).
   - So sánh đối chứng **Single-View vs 8-View Dihedral TTA** (trung bình xác suất trên 4 góc xoay × 2 phép lật).

## 2. Kết quả 5-Fold CV & Out-Of-Fold (212 nhóm lá độc lập, Tesla T4, 8.06 phút)

### Đối chứng Single-View vs 8-View TTA
| Phương pháp suy luận | 5-Fold CV Accuracy (`mean ± std`) | 5-Fold CV Macro-F1 (`mean ± std`) | OOF Accuracy (212 nhóm) | OOF Macro-F1 (212 nhóm) |
| --- | --- | --- | --- | --- |
| Single-View | `0.9392 ± 0.0358` | `0.9262 ± 0.0478` | `0.9434` | `0.9390` |
| **8-View Dihedral TTA** | **`0.9490 ± 0.0293`** | **`0.9343 ± 0.0490`** | **`0.9528`** | **`0.9469`** |

### Kết quả từng lớp (Out-Of-Fold + 8-View TTA, toàn bộ 212 nhóm lá độc lập)
| Lớp bệnh | Precision | Recall | F1 | Support (nhóm) |
| --- | --- | --- | --- | --- |
| cercospora | 0.8750 | 1.0000 | 0.9333 | 21 |
| leaf_rust | 1.0000 | 0.9859 | 0.9929 | 71 |
| miner | 0.9375 | 0.9000 | 0.9184 | 50 |
| phoma | 0.9429 | 0.9429 | 0.9429 | 70 |
| **Macro avg** | **0.9388** | **0.9572** | **0.9469** | **212** |
| **Weighted avg** | **0.9540** | **0.9528** | **0.9529** | **212** |

- Chi tiết từng fold xem tại [results/cv_fold_results.csv](results/cv_fold_results.csv).
- Lớp OOF F1 thấp nhất: `miner` (`0.9184`); nhầm nhiều nhất: `miner → phoma` (4 ảnh) và `phoma → miner` (3 ảnh).
- Suy luận CPU độc lập và TorchScript (Fold 0) đã kiểm tra khớp đầu ra (`inference_verified: true`, `torchscript_verified: true`).

## 3. Hạn chế và lưu ý khoa học
- Bài toán tập trung phân loại 4 loại bệnh trong phạm vi JMuBEN (không phân biệt lá khỏe vì `JMuBEN/Healthy` chỉ có 2 lá gốc — xem [results/healthy_2leaf_evidence.png](results/healthy_2leaf_evidence.png)).
- Số nhóm độc lập của `cercospora` còn nhỏ (21 nhóm); ảnh JMuBEN là ảnh cắt cận 128×128 nên chưa đại diện ảnh chụp thực địa ngoài vườn.
- Chưa đo latency trên thiết bị di động.

## 4. Nội dung repo
- `kaggle/`: notebook v6 ([mobilenetv3large_jmuben.ipynb](kaggle/mobilenetv3large_jmuben.ipynb)) và `kernel-metadata.json`.
- `results/`: [RESULTS.md](results/RESULTS.md), `cv_fold_results.csv`, `metrics.json`, `data_audit.json`, `classification_report.csv`, `healthy_2leaf_evidence.png`, `cross_split_nearest_pairs.png`, `confusion_matrix.png`, `learning_curves.png`, `gradcam.png`, `misclassified.png`, `kaggle_run.log`.
- Checkpoint `best.pt`, TorchScript `.ts` và `report_assets.zip` không đưa lên repo (file lớn); lấy từ output của Kaggle kernel `ngtunthanh/mobilenetv3large-jmuben` (version 7).

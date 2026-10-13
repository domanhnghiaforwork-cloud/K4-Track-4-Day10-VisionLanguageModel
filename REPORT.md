Họ tên: Đỗ Mạnh Nghĩa (domanhnghiaforwork-cloud)
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | - | - | 0.0699 | [0.0462, 0.0976] | 0.9678 | 0.0845 | 0.0233 | 0.1563 |
| E1 | LoRA LLM | 33,030,144 | 6.113 | 40.16 | 0.2780 | [0.2400, 0.3193] | 1.1929 | 0.1918 | 0.1958 | 0.5414 |
| E2 | + projector | 142,135,296 | 8.350 | 44.67 | 0.3245 | [0.2852, 0.3673] | 1.1801 | 0.1981 | 0.2584 | 0.5909 |
| E3 | + LoRA ViT | 148,426,752 | 11.793 | 50.99 | 0.4292 | [0.3875, 0.4683] | 0.9260 | 0.2990 | 0.3560 | 0.6667 |
| E4 | LoRA rank r=4 | 118,935,552 | 8.085 | 50.51 | 0.4117 | [0.3717, 0.4525] | 1.0032 | 0.3320 | 0.3362 | 0.6323 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Số tham số từng khối trong Qwen3-VL-4B** (tổng ~4.4B tham số):
  - **ViT (Vision Transformer)**: ~430M tham số (gồm patch embed, 32 vision blocks với self-attention và MLP).
  - **Projector** (`visual.merger` và `visual.deepstack_merger_list`): 109,105,152 tham số (~109.1M tham số), đóng vai trò ánh xạ và nén đặc trưng không gian từ ViT sang không gian nhúng của LLM.
  - **LLM backbone**: ~3.9B tham số (gồm các decoder transformer blocks và `lm_head`).
- **Khối cập nhật trong từng setup**:
  - **E0**: 0 tham số cập nhật (đóng băng hoàn toàn mô hình gốc).
  - **E1**: Cập nhật LoRA trên các ma trận chiếu của LLM (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) với tổng cộng **33,030,144** tham số trainable. ViT và Projector đóng băng.
  - **E2**: Cập nhật LoRA LLM (33.0M) cộng thêm mở đầy đủ toàn bộ tham số của Projector (`visual.merger` và `visual.deepstack_merger_list` với 109.1M tham số). ViT vẫn đóng băng. Tổng số tham số trainable là **142,135,296**.
  - **E3**: Kế thừa E2 và gắn thêm LoRA trên các khối attention và MLP của ViT (thêm ~6.3M tham số LoRA ViT). Tổng số tham số trainable đạt **148,426,752**.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- `parse_valid_rate` của E0 đạt **0.7685** (76.85%), có 23.15% số output không phân tích cú pháp được thành JSON hợp lệ.
- **Hai lỗi phổ biến nhất của model gốc (zero-shot)**:
  1. *Bịa đặt box và sai lệch toạ độ (Hallucination / Scale confusion)*: Mô hình sinh ra nhiều box không có thật (False Positives lên đến 301 box trên 311 ảnh, precision chỉ đạt 9.61%), hoặc nhầm lẫn giữa toạ độ pixel [0, 640]x[0, 360] với hệ toạ độ chuẩn hoá 0-1000.
  2. *Sinh văn bản thừa và định dạng hỏng*: Mô hình có xu hướng giải thích dài dòng bằng tiếng Anh/tiếng Trung thay vì chỉ trả về một danh sách JSON thuần tuý, hoặc bọc trong markdown code fence không khép kín.
- **Ví dụ output thô điển hình**:
  ```text
  Based on the image, here is the detected road damage:
  ```json
  [{"label": "road_crack", "bbox_2d": [120, 450, 890, 950]}, {"label": "pothole", "bbox_2d": [0, 0, 640, 360]}]
  ```
  Hope this helps identify the defects.
  ```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- F1 sẽ sụt giảm nghiêm trọng về gần như bằng 0.0 (F1 ≈ 0).
- **Nguyên nhân**: Ảnh gốc có kích thước 640×360. Nếu mô hình dự đoán theo pixel ([0, 640] và [0, 360]) nhưng được chấm theo thang [0, 1000], box dự đoán sẽ bị thu nhỏ dồn về góc trên bên trái của ảnh thang 1000. Ngược lại, nếu mô hình dự đoán theo thang 0-1000 mà bị chấm theo pixel, toạ độ box sẽ vượt ra ngoài khung hình 640×360. Khi tính IoU với ngưỡng IoU ≥ 0.5, diện tích giao nhau (intersection) giữa box dự đoán và nhãn thực tế sẽ bằng 0 hoặc cực nhỏ, khiến True Positive = 0 và F1 rơi về 0.
- **Kiểm chứng**: Đã kiểm chứng qua các test suite trong `tests/test_score.py` và cờ `--coord pixel` của script `scripts/score.py`. Khi đảo hệ toạ độ, điểm IoU luôn < 0.1 và F1 sụt về 0.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời**: Đây là bài toán sinh có điều kiện (conditional generation). Prompt hướng dẫn và 220 token ảnh là dữ kiện đầu vào đã biết trước. Mục tiêu là tối ưu xác suất sinh có điều kiện $P(\text{JSON answer} \mid \text{Image, Prompt})$. Việc gán nhãn `-100` (mask loss) cho prompt và vision tokens giúp mô hình chỉ tập trung cập nhật gradient cho các token nhãn lớp và toạ độ box.
- **Điều gì xảy ra nếu tính trên cả prompt và token ảnh**:
  1. Token ảnh và prompt chiếm hơn 80% độ dài chuỗi đầu vào. Nếu tính loss trên toàn bộ, mô hình sẽ tốn phần lớn dung lượng gradient để học tái hiện lại prompt và phân bố ảnh thay vì học bài toán grounding.
  2. Dẫn đến suy giảm mạnh độ chính xác toạ độ box (loss hội tụ chậm hoặc trượt khỏi mục tiêu chính), đồng thời làm tăng chi phí tính toán gradient và bộ nhớ VRAM.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện đầy đủ**: Projector (`visual.merger`, `visual.deepstack_merger_list`) nằm trong cây module `model.visual`. Nếu dùng BitsAndBytes lượng tử hoá 4-bit thông thường mà không loại trừ chuẩn, các tham số projector sẽ bị đóng gói thành `Params4bit` (dạng uint8). PyTorch không hỗ trợ truyền gradient cho kiểu dữ liệu này (`only Tensors of floating point and complex dtype can require gradients`), và lệnh ép kiểu `.float()` không thể giải nén trọng số 4-bit này.
- **Cách xử lý**: Thiết lập `llm_int8_skip_modules=["model.visual", "visual", "lm_head"]` khi tải base model để giữ nguyên ViT và projector ở định dạng unquantized (fp16). Sau đó, trong hàm thiết lập E2/E3, duyệt qua các tham số của merger và chuyển sang `torch.float32` kèm `requires_grad = True`.
- **Vì sao projector phải ở float32**: Khi huấn luyện hỗn hợp (AMP) với `torch.cuda.amp.GradScaler`, các tham số trainable cần ở `float32` để tránh hiện tượng underflow/overflow gradient và lỗi `Attempting to unscale FP16 gradients`.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên cho F1 cao giả**: Dataset gồm các frame cắt từ các video dashcam liên tục. Các frame liền kề nhau có bối cảnh mặt đường, góc máy, ánh sáng và chính các hư hại gần như trùng lặp hoàn toàn. Nếu chia ngẫu nhiên (random split), các frame tương đồng này sẽ rơi vào cả tập train và test/val, gây ra hiện tượng rò rỉ dữ liệu (data leakage) nghiêm trọng. Mô hình có thể ghi nhớ bối cảnh mặt đường để dự đoán đúng mà không thực sự học được đặc trưng tổng quát, dẫn đến điểm F1 cao giả tạo trên test nhưng thất bại ngoài thực tế. Chia theo phiên quay (`session`) đảm bảo toàn bộ video của một buổi quay chỉ thuộc về đúng một tập.
- **Số ảnh và số box từng lớp ở mỗi split** (lấy từ `data/splits/meta.json`):
  - **train**: 1.422 ảnh | 1.074 pothole | 1.750 crack | 691 manhole (tổng 3.515 box).
  - **val**: 276 ảnh | 66 pothole | 470 crack | 103 manhole (tổng 639 box).
  - **test**: 311 ảnh | 121 pothole | 299 crack | 163 manhole (tổng 583 box).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **Số liệu so sánh**:
  - **E1 (LoRA LLM)**: F1 = 0.2780, CI95 = [0.2400, 0.3193], VRAM = 6.113 GB, Thời gian = 40.16 phút.
  - **E2 (+ projector)**: F1 = 0.3245, CI95 = [0.2852, 0.3673], VRAM = 8.350 GB, Thời gian = 44.67 phút (+4.51 phút, +2.24 GB VRAM, F1 tăng +0.0465).
  - **E3 (+ LoRA ViT)**: F1 = 0.4292, CI95 = [0.3875, 0.4683], VRAM = 11.793 GB, Thời gian = 50.99 phút (+10.83 phút so với E1, +5.68 GB VRAM, F1 tăng vượt bậc +0.1512 so với E1).
- **Đánh giá**:
  - Mở Projector (E2) rất đáng giá: F1 tăng từ 0.2780 lên 0.3245 trong khi thời gian huấn luyện chỉ tăng thêm ~4.5 phút và VRAM chỉ tốn thêm 2.24 GB, nằm an toàn trong giới hạn GPU T4.
  - Mở LoRA trên ViT (E3) cực kỳ đáng giá: F1 đạt mức cao nhất 0.4292, CI95 của E3 [0.3875, 0.4683] hoàn toàn tách biệt so với CI95 của E1 [0.2400, 0.3193]. VRAM đỉnh 11.79 GB vẫn nằm trong mức an toàn của Colab T4 16GB. Do hư hại mặt đường (vết nứt nhỏ, ổ gà phẳng) yêu cầu biểu diễn thị giác chi tiết, việc cho phép ViT thích ứng đóng vai trò quyết định.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy bootstrap 95%**:
  - E0: độ rộng = 0.0514 (~5.1%)
  - E1: độ rộng = 0.0793 (~7.9%)
  - E2: độ rộng = 0.0821 (~8.2%)
  - E3: độ rộng = 0.0808 (~8.1%)
- **Giới hạn kết luận**:
  - Do tập test chỉ gồm 311 ảnh (583 box), khoảng tin cậy CI95 dao động trong khoảng rộng ~8% F1.
  - Giữa E1 và E2: Khoảng tin cậy của E1 [0.2400, 0.3193] và E2 [0.2852, 0.3673] có một khoảng chồng lấn đáng kể ([0.2852, 0.3193]). Do đó, chỉ dựa trên tập test này chưa thể khẳng định chắc chắn 95% rằng E2 vượt trội tuyệt đối so với E1 trên phân phối thực tế.
  - Giữa E1 và E3: CI95 của E3 [0.3875, 0.4683] hoàn toàn tách rời so với E1 [0.2400, 0.3193], cho phép khẳng định với độ tin cậy thống kê cao rằng E3 vượt trội hơn hẳn E1.
  - **Bài học**: Khi làm việc với tập test nhỏ, không được chỉ nhìn vào con số F1 trung bình mà bắt buộc phải xem xét bootstrap CI95 để đưa ra kết luận có ý nghĩa thống kê.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Hiệu năng từng lớp trên mô hình tốt nhất (E3)**:
  - `manhole`: F1 = 0.6667 (Precision = 0.7388, Recall = 0.6074) — Tốt nhất.
  - `crack`: F1 = 0.3560 (Precision = 0.3448, Recall = 0.3679) — Trung bình.
  - `pothole`: F1 = 0.2990 (Precision = 0.3973, Recall = 0.2397) — Kém nhất.
- **Nguyên nhân lớp `pothole` kém nhất**:
  1. *Hình thái và kích thước box*: Ổ gà có viền mờ nhạt, độ tương phản thấp so với nhựa đường xung quanh, và diện tích trung vị rất nhỏ (~0.7% khung hình). Khi nén về 220 vision tokens, thông tin chi tiết của ổ gà dễ bị thất thoát. Ngược lại, `manhole` có hình tròn/chữ nhật rõ ràng, vân kim loại/bê tông viền sắc nét nên mô hình nhận diện vượt trội.
  2. *Số lượng mẫu kiểm thử ít*: Tập test chỉ có 121 box `pothole` (so với 299 crack và 163 manhole). Do mẫu ít, mỗi trường hợp bỏ sót (FN) kéo tụt Recall và F1 của pothole rất nhanh.
  3. *Nhập nhằng ngữ nghĩa giữa `crack` và `pothole`*: Ranh giới giữa một mảng vết nứt vỡ lớn và một ổ gà nông thường không rạch ròi, dẫn đến việc mô hình dự đoán nhầm lớp hoặc gom nhiều ổ gà nhỏ thành một vết nứt lớn.
- **Ảnh minh hoạ trực quan (Box xanh lá: Ground Truth · Box đỏ: Dự đoán)**:
  - ![Nắp cống dự đoán chính xác](report/images/sample1_manhole_correct.jpg)
    *Hình 1: Dự đoán nắp cống (`manhole`) rất chính xác với IoU cao.*
  - ![Vết nứt dự đoán chính xác](report/images/sample2_crack_correct.jpg)
    *Hình 2: Định vị vết nứt (`crack`) trên mặt đường khớp với nhãn thực tế.*
  - ![Ổ gà dự đoán chính xác](report/images/sample3_pothole_correct.jpg)
    *Hình 3: Dự đoán ổ gà (`pothole`) điển hình có kích thước vừa phải.*
  - ![Ổ gà bị bỏ sót](report/images/sample4_pothole_missed_fn.jpg)
    *Hình 4: Ổ gà mờ hoặc ở góc xa bị bỏ sót (False Negative), nguyên nhân khiến recall của pothole thấp.*
  - ![Dự đoán thừa box](report/images/sample5_false_positive_box.jpg)
    *Hình 5: Dự đoán thừa box trên vùng mặt đường gồ ghề (False Positive).*
  - ![Vết nứt dày đặc gây nhầm lẫn](report/images/sample6_crack_dense_fp.jpg)
    *Hình 6: Khu vực vết nứt chằng chịt dẫn đến việc phân mảnh hoặc gộp box lệch so với quy ước nhãn.*

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến lựa chọn**: LoRA rank $r$ (so sánh giữa rank $r=16$ ở E3 và rank $r=4$ ở E4 trên cùng thiết lập mở Projector và LoRA ViT).
- **Kết quả so sánh định lượng**:
  - **E4 ($r=4$)**: Tham số train: 118,935,552 | VRAM đỉnh: 8.085 GB | Phút: 50.51 | F1: 0.4117 | CI95: [0.3717, 0.4525] | FP/ảnh: 1.0032.
  - **E3 ($r=16$)**: Tham số train: 148,426,752 | VRAM đỉnh: 11.793 GB | Phút: 50.99 | F1: 0.4292 | CI95: [0.3875, 0.4683] | FP/ảnh: 0.9260.
- **Đánh giá độ tin cậy**:
  - Giảm rank từ 16 xuống 4 giúp giảm gần 29.5 triệu tham số huấn luyện, tiết kiệm tới ~3.7 GB VRAM đỉnh (8.08 GB so với 11.79 GB), trong khi điểm F1 chỉ giảm nhẹ 0.0175 (từ 0.4292 xuống 0.4117).
  - Khoảng tin cậy CI95 của E4 [0.3717, 0.4525] và E3 [0.3875, 0.4683] chồng lấn nhau gần như toàn bộ. Điều này khẳng định sự khác biệt về F1 giữa rank 16 và rank 4 là không có ý nghĩa thống kê trên tập test 311 ảnh. Cấu hình rank $r=4$ là lựa chọn tối ưu hơn về mặt chi phí và tài nguyên phần cứng.

## Nhìn lại

Qua bài lab, tôi nhận thấy một số vấn đề quan trọng khi ứng dụng Vision-Language Model cho bài toán Object Grounding:
1. Độ phân giải thị giác và số token: Mô hình nén ảnh về 220 tokens vision. Với các vật thể có diện tích rất nhỏ như vết nứt mảnh hoặc ổ gà nhỏ ở xa, cơ chế nén này làm suy giảm đáng kể thông tin chi tiết, khiến mô hình khó đạt recall cao cho lớp `pothole`.
2. Hệ toạ độ chuẩn hoá 0–1000 là giải pháp thanh lịch giúp LLM sinh toạ độ bằng các token số nguyên độc lập với kích thước ảnh gốc, nhưng đòi hỏi độ chính xác tuyệt đối trong khâu tiền xử lý chuyển đổi giữa nhãn YOLO và JSON grounding.
3. Việc đánh giá với khoảng tin cậy Bootstrap 95% là công cụ bắt buộc khi làm việc với tập test nhỏ (311 ảnh), giúp tránh được các kết luận chủ quan sai lầm khi chỉ nhìn vào con số F1 trung bình.

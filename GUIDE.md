# Hướng dẫn thực hiện Lab Day 10

> Đọc [`README.md`](README.md) trước. Mỗi bước gồm **Mục tiêu · Pseudo code · Kiểm tra · Câu hỏi** (điền vào [`REPORT.md`](REPORT.md)). Pseudo code **không chạy được ngay**: bạn tự viết mã thật trong notebook (hoặc file `.py` của bạn).
> Quy ước: tên module lấy từ cây model Qwen3-VL-4B thật. Hãy tự in `[n for n, _ in model.named_modules()]` để xác nhận.

| Phút | Nội dung |
|---|---|
| 0–30 | Bước 0–1: môi trường, dữ liệu |
| 30–60 | Bước 2–3: nhãn JSON, E0 zero-shot |
| 60–90 | Bước 4–5: dữ liệu huấn luyện, setup LoRA |
| 90–200 | Bước 6–7: huấn luyện và chấm E1, E2, E3 |
| 200–230 | Bước 8: E4 |
| 230+ | Bước 9 (tuỳ chọn) và báo cáo |

---

## Bước 0 · Môi trường

```bash
pip install -r requirements-colab.txt
python -m pytest -q
```

Lab dùng bitsandbytes, không cần backend torchao. Nếu Colab có sẵn `torchao==0.10.0`, gỡ bằng `%pip uninstall -y torchao` trước khi import PEFT, rồi chọn **Runtime → Restart session** nếu đã import thư viện. PEFT 0.21.2 yêu cầu torchao >= 0.16.0 khi torchao được cài; E3 có thể chạm kiểm tra này khi gắn LoRA lên các lớp ViT. Không cần nâng cấp torch/torchvision để sửa lỗi này. Ô môi trường trong notebook đã có lệnh gỡ torchao.

Nếu gặp `ImportError ... requires the Torchvision library`: cài `torchvision` cùng bản torch của Colab.

## Bước 1 · Dữ liệu

```bash
python scripts/prepare_data.py --kaggle
```

Sinh `data/images/`, `data/splits/{train,val,test}.jsonl` và `data/splits/meta.json`. Mỗi dòng `.jsonl`:

```json
{"image_id": "...", "image_path": "images/....jpg", "width": 640, "height": 360,
 "session": "vlc-2025-02-18-23", "objects": [{"label": "pothole", "yolo": [xc, yc, w, h]}]}
```

**Kiểm tra:** in số ảnh/box từng split; vẽ 6 ảnh có box lên để chắc bạn hiểu nhãn. Xác nhận không `session` nào xuất hiện ở hai split.

**Câu hỏi:** vì sao chia theo `session` mà không chia ngẫu nhiên? Điều gì xảy ra với F1 nếu chia ngẫu nhiên?

## Bước 2 · Nhãn YOLO → JSON thang 0–1000

```
hàm yolo_to_box(label, yolo):
    xc, yc, w, h = yolo                          # chuẩn hoá [0,1], tâm + kích thước
    x1 = (xc - w/2) * 1000;   y1 = (yc - h/2) * 1000
    x2 = (xc + w/2) * 1000;   y2 = (yc + h/2) * 1000
    làm tròn thành số nguyên, kẹp vào [0, 1000]
    trả {"label": label, "bbox_2d": [x1, y1, x2, y2]}

hàm objects_to_answer(objects):
    boxes = [yolo_to_box(...) cho mỗi object]
    sắp boxes theo (y1, x1)                      # trên→dưới, trái→phải: thứ tự ổn định cho model học
    trả json.dumps(boxes, separators=(",", ":")) # "[]" nếu không có object
```

Viết **prompt** yêu cầu model chỉ trả JSON list, nêu 3 lớp và nói rõ "thang 0–1000".

**Kiểm tra:** với một ảnh, chuyển nhãn sang JSON rồi quy ngược về pixel và vẽ lại; box phải trùng box gốc.

**Câu hỏi:** vì sao Qwen3-VL dùng thang 0–1000 chứ không dùng pixel tuyệt đối? (Slide 04, "đầu ra grounding").

## Bước 3 · E0: zero-shot

```
quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=float16, bnb_4bit_use_double_quant=True,
                  llm_int8_skip_modules=["model.visual", "visual", "lm_head"])
model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", quantization_config=quantization,
            dtype=float16, device_map={"": 0})
processor = AutoProcessor.from_pretrained(...);  processor.tokenizer.padding_side = "left"

với mỗi batch ảnh test (batch 4):
    messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text",  "text": PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", padding=True)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = decode(out[:, độ_dài_prompt:], skip_special_tokens=True)
    ghi {"image_id": ..., "raw": raw} vào work/e0.jsonl      # mỗi dòng một ảnh
```

Chấm:

```bash
python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json
```

> Với Transformers 5.19, dùng `llm_int8_skip_modules=["model.visual","visual","lm_head"]` để giữ ViT (kể cả projector) ở fp16. Tên thực tế bắt đầu bằng `model.visual`; chỉ dùng `"visual"` không loại trừ các lớp con trong phiên bản này. Nếu ViT bị lượng tử hoá 4-bit, trọng số của merger là trọng số lượng tử và **không thể huấn luyện đầy đủ** ở E2/E3. Notebook kiểm tra điều này ngay sau khi nạp base model và trước khi mở gradient.

**Kiểm tra:** `parse_valid_rate` là bao nhiêu? In 5 output thô để xem model trả gì (có bọc ```json không? có chữ thừa không?).

**Câu hỏi:** hai lỗi phổ biến nhất của E0 là gì (sai thang, bịa box, sai lớp, JSON hỏng)? Kèm một ví dụ output.

## Bước 4 · Ví dụ huấn luyện và mask loss

```
hàm encode_example(image, prompt, answer):
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True,
                     tokenize=True, return_dict=True, return_tensors="pt")
    full_enc   = processor.apply_chat_template([user, assistant(answer)], add_generation_prompt=False, ...)
    n = độ dài prompt_enc.input_ids
    KIỂM TRA full_enc.input_ids[0, :n] == prompt_enc.input_ids[0]    # prompt là tiền tố của chuỗi đầy đủ
    labels = bản sao của full_enc.input_ids;  labels[:, :n] = -100   # chỉ học phần trả lời
    trả full_enc kèm labels

hàm collate(batch):
    pad input_ids / attention_mask / mm_token_type_ids / labels đến cùng độ dài (labels đệm = -100)
    pixel_values, image_grid_thw: nối (cat) theo chiều 0, KHÔNG pad
```

**Kiểm tra:** với một ví dụ, in số token ảnh (phải là **220** cho ảnh 640×360), và giải mã riêng các token có `labels != -100`: kết quả phải đúng là JSON đáp án (cộng `<|im_end|>`).

**Câu hỏi:** vì sao không tính loss trên prompt và token ảnh? Điều gì xảy ra nếu tính?

## Bước 5 · Setup LoRA / đóng băng (E1, E2, E3)

```
hàm apply_setup(model, setup):
    mọi tham số: requires_grad = False
    model.enable_input_require_grads()            # cần cho gradient checkpointing
    mục tiêu LoRA (regex trên tên module):
        LLM (E1, E2, E3): model.language_model.layers.N.self_attn.{q,k,v,o}_proj
                          model.language_model.layers.N.mlp.{gate,up,down}_proj      # 36 × 7 = 252 module
        ViT (chỉ E3):     model.visual.blocks.N.attn.{qkv,proj}
                          model.visual.blocks.N.mlp.{linear_fc1,linear_fc2}          # 24 × 4 = 96 module
    peft_model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                  target_modules=<regex>))
    nếu setup là E2 hoặc E3:
        với mọi tham số có tên chứa "visual.merger" hoặc "visual.deepstack_merger_list":
            đưa về float32;  requires_grad = True
    với mọi tham số requires_grad (cả LoRA và projector): đưa về float32
    kiểm tra dtype float32 trước khi tạo optimizer
    in tổng số tham số huấn luyện
```

Gợi ý quy mô (để bạn đối chiếu): LLM + embedding ≈ 4,0B; ViT ≈ 415M; trong ViT, `visual.merger` ≈ 27M và `visual.deepstack_merger_list` ≈ 82M (DeepStack lấy đặc trưng ở lớp 5, 11, 17 của ViT). Tổng tham số Qwen3-VL-4B ≈ 4,44B.

**Kiểm tra:** `len(matched_modules)` phải là 252 (E1, E2) và 348 (E3). Số tham số huấn luyện của E2 phải lớn hơn E1 khoảng 109M.

**Câu hỏi:** vì sao merger phải huấn luyện ở float32 trong khi các trọng số còn lại ở fp16? Vì sao không thể huấn luyện đầy đủ merger nếu ViT bị lượng tử hoá 4-bit?

## Bước 6 · Vòng huấn luyện

```
tối ưu = AdamW(các tham số requires_grad, lr=2e-4, weight_decay=0)
lịch lr: warmup 5% bước đầu, sau đó cosine về 0
scaler = GradScaler("cuda")                                   # fp16
bật gradient checkpointing (use_reentrant=False); model.config.use_cache = False
lặp đến khi đủ max_steps = 300 bước tối ưu:
    lặp 4 lần (tích luỹ gradient), mỗi lần một batch 2 ảnh:
        với autocast(fp16):  loss = model(**batch).loss / 4
        nếu loss không hữu hạn: DỪNG và báo (nguy cơ tràn số fp16)
        scaler.scale(loss).backward()
    scaler.unscale_(opt); clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); opt.zero_grad()
    chỉ tăng step và cập nhật scheduler nếu GradScaler không bỏ qua bước do tràn gradient
    mỗi 100 bước: lưu adapter  VÀ  lưu riêng các tham số projector đã huấn luyện (E2, E3)
ghi run.json: trainable_params, peak_vram_gb (torch.cuda.max_memory_allocated), train_minutes
```

> `model.save_pretrained()` của peft **chỉ lưu adapter LoRA**. Projector (E2/E3) được huấn luyện đầy đủ, **không nằm trong adapter**: nếu quên lưu riêng, lúc suy luận bạn sẽ nạp lại projector gốc và kết quả E2/E3 trông giống E1.
> Chép adapter sang Google Drive sau mỗi lần chạy: session miễn phí có thể bị ngắt.

**Kiểm tra:** loss giảm đều và hữu hạn; ghi lại VRAM đỉnh và thời gian.

**Câu hỏi:** setup nào dùng nhiều VRAM nhất và vì sao (gợi ý: gradient qua ViT, tham số float32 của projector)?

## Bước 7 · Suy luận và chấm E1–E3

```
nạp model gốc (cùng cấu hình 4-bit như bước 3) → nạp adapter LoRA → nạp lại tham số projector (nếu có)
suy luận trên test như bước 3 → work/e1.jsonl
```

```bash
python scripts/score.py --pred work/e1.jsonl --name E1 --out report/evidence/e1.json \
    --meta trainable_params=<số> --meta peak_vram_gb=<số> --meta train_minutes=<số>
```

Mẹo: gỡ lỗi trên 40 ảnh của `val` (`--split val`); chỉ chấm `test` một lần cho mỗi setup.

Notebook lưu projector ở float32 và dùng `autocast(float16)` cả lúc suy luận để tương thích với ViT float16. Thiếu adapter, projector hoặc `run.json` thì dừng đánh giá; không điền số liệu tài nguyên thay thế. Checkpoint mới có `completed_steps` để phát hiện lần train chưa hoàn tất.

**Kiểm tra:** E1 phải hơn hẳn E0. Nếu không, kiểm tra mask loss (bước 4) và thang toạ độ (bước 2).

**Câu hỏi:** lập bảng E0–E3 trong `REPORT.md` (F1, CI95, FP/ảnh, F1 từng lớp, VRAM, thời gian). Setup nào đáng đồng tiền nhất? Chênh lệch có vượt khoảng tin cậy không?

## Bước 8 · E4: ablation (chọn một)

Đổi **đúng một biến** so với setup tốt nhất của E1–E3 (xem `configs/e4.yaml`) và chạy cả hai giá trị (một trong hai có thể chính là setup gốc):

| Biến | Cách đổi |
|---|---|
| `rank` | r = 4 và r = 64 (đặt alpha = 2·r) |
| `data_size` | 10% và 100% ảnh train |
| `coord_scale` | `norm1000` và `pixel`: đổi **cả** nhãn, prompt, và chấm với `--coord pixel` |

Lưu: `report/evidence/e4_<biến>_<giá trị>.json`.

Notebook chọn setup bằng F1 trên `val`, chạy cả rank 4 và 64, và lưu vào `work/e4_rank_4/`, `work/e4_rank_64/`. Không dùng lại thư mục checkpoint E1–E3. Nếu cần đổi batch để vừa VRAM, giữ cùng batch và grad accumulation cho mọi thí nghiệm so sánh.

**Câu hỏi:** kết luận của bạn có đáng tin không khi test chỉ có 311 ảnh? (Nhìn CI95.)

## Bước 9 · E5 (tuỳ chọn, +10 điểm): GRPO với reward IoU

```
hàm reward(completion, gt_boxes):
    pred = parse_output(completion)             # dùng được scripts.score.parse_output
    nếu pred là None: trả 0.0                   # JSON hỏng
    nếu không có gt và không có pred: trả 1.0
    ghép đôi tham lam theo IoU giảm dần, cùng lớp, chỉ nhận IoU ≥ 0.5
    tổng_iou = tổng IoU các cặp khớp
    trả 0.2 + 0.8 * tổng_iou / max(số_gt, số_pred, 1)    # box bịa bị phạt qua mẫu số
```

Viết **test cho reward trước khi chạy GRPO** (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng). Hàm reward là Python thuần nên test được mà không cần GPU.

GRPO: khởi tạo từ adapter tốt nhất, `num_generations=4`, `max_completion_length=128`, khoảng 60 bước, `use_vllm=False`. Dùng `trl.GRPOTrainer` (đã ghim `trl==1.9.2`; hãy đọc docs của đúng phiên bản này).

**Câu hỏi:** có dấu hiệu reward hacking không (model trả `[]` hoặc một box khổng lồ)? So với SFT, F1 thay đổi thế nào và có vượt CI95 không?

## Khi gặp sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `CUDA out of memory` | batch quá lớn, quên gradient checkpointing | `batch_size=1`, `grad_accum=8`, bật gradient checkpointing |
| `Attempting to unscale FP16 gradients` | còn trọng số trainable float16; Colab giữ định nghĩa hàm cũ | đưa toàn bộ LoRA/projector trainable về float32 trước optimizer; chạy lại các ô định nghĩa setup/train/evaluate |
| `only Tensors of floating point and complex dtype can require gradients` | projector vẫn là `Params4bit`/uint8 vì skip_modules không khớp tên `model.visual` | thêm `"model.visual"` vào skip_modules và nạp lại base; `.float()` không giải lượng tử trọng số đóng gói |
| `Found an incompatible version of torchao ... 0.10.0` | backend torchao cũ có sẵn trong Colab bị PEFT kiểm tra khi gắn LoRA | `%pip uninstall -y torchao`, restart session rồi nạp lại các imports/hàm; lab dùng bitsandbytes nên không cần torchao |
| Loss `nan`/`inf` | tràn số fp16 | giảm lr; thử `batch_size=1`; kiểm tra bạn đã mask đúng |
| F1 E2/E3 ≈ E1 chính xác | quên nạp lại projector lúc suy luận | lưu và nạp `extra_trainable` (bước 6) |
| `parse_valid_rate` thấp | model lặp vô hạn / bị cắt | tăng `max_new_tokens`, kiểm tra token kết thúc `<|im_end|>` trong nhãn |
| F1 ≈ 0 mà JSON hợp lệ | sai thang toạ độ (pixel thay vì 0–1000) | xem lại bước 2 và `--coord` |
| Mất session giữa chừng | Colab free ngắt | chép adapter lên Drive sau mỗi lần |

Sau khi cập nhật notebook, chạy lại ô imports/processor cùng các hàm `make_quantization_config`, `load_base_model` ở mục 3, rồi các ô định nghĩa setup/train/evaluate trước khi chạy riêng ô E2. Ô tải model E0 đã tách riêng để có thể bỏ qua. Nếu lần E2 trước đã lưu checkpoint dở, chạy `meta_e2 = train_experiment('E2', run_name='e2_retry')`, rồi `evaluate_experiment('E2', meta_e2, evidence_filename='e2.json')`. Nếu traceback cũ giữ VRAM, restart session và nạp lại dữ liệu, processor, imports và các hàm. E0/E1 chạy bằng config cũ có thể đã lượng tử hoá cả ViT: chạy lại với config mới để so sánh nhất quán, dùng run_name mới để giữ checkpoint cũ.

Cách khớp module của bản đã ghim: [Transformers 5.19 `should_convert_module`](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/quantizers/quantizers_utils.py).

Hướng dẫn dtype chính thức: [PEFT troubleshooting](https://huggingface.co/docs/peft/developer_guides/troubleshooting#valueerror-attempting-to-unscale-fp16-gradients).

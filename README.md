# AIC Video Retrieval Pipeline

Pipeline tìm kiếm và xác minh khoảnh khắc trong video cho AIC, hỗ trợ ba loại truy vấn:

- **KIS**: tìm video và frame phù hợp với mô tả.
- **QA**: tìm bằng chứng hình ảnh và trả lời câu hỏi.
- **TRAKE**: tìm chuỗi sự kiện theo đúng thứ tự thời gian.

Hệ thống phân tích truy vấn bằng LLM, truy hồi bằng CLIP + FAISS, kết hợp metadata object và cụm thời gian, sau đó dùng Qwen3-VL để xác minh trước khi tạo file nộp. SỬ DỤNG VLM LÀ QWEN 8B ĐỂ TRỰC TIẾP XEM VIDEO. ƯU ĐIỂM TIẾT KIỆM CHI PHÍ, TƯƠNG ĐỐI CHÍNH XÁC NHƯNG CŨNG CẦN CẢI THIỆN THÊM VÌ MODEL 8B CÓ GIỚI HẠN NÊN PHẦN RETRIEVAL PHẢI LÀM TỐT HƠN. NHƯỢC ĐIỂM LÀ LÂU, VLM YẾU SO VỚI API CỦA GPT HAY GEMINI, YÊU CẦU VRAM CAO (GG COLAB PRO A100) ĐỂ CHẠY ỔN ĐỊNH.

## Cấu trúc

```text
.
├── main.py                 # Entry point, chạy query và tạo submission.zip
├── config.py               # Đường dẫn dữ liệu và tham số pipeline
├── main_AIC                # Notebook Colab (JSON, có thể đổi đuôi thành .ipynb)
├── phantichquery/          # Phân loại và phân tích truy vấn bằng LLM
├── retrieval/              # CLIP, FAISS, object boost, temporal fusion, TRAKE
└── video_pipeline/         # Giải nén video, cắt clip và xác minh bằng VLM
```

## Yêu cầu

- Python 3.10+
- GPU CUDA được khuyến nghị mạnh cho Qwen3-VL
- `ffmpeg` và `ffprobe`
- OpenAI API key và Hugging Face token (nếu model yêu cầu)
- Các tài nguyên: FAISS index, metadata pickle, object ZIP và các file `Videos_*.zip`

Cài đặt theo môi trường Colab hiện tại của dự án:

```bash
apt-get update && apt-get install -y ffmpeg
pip install -U accelerate faiss-cpu huggingface_hub qwen-vl-utils av decord openai
pip install -U git+https://github.com/openai/CLIP.git
pip install -U git+https://github.com/huggingface/transformers.git
```

## Cấu hình

1. Sửa các đường dẫn trong `config.py`:
   - `FAISS_INDEX_PATH`, `METADATA_PATH`
   - `OBJECT_ZIP_PATH`, `VIDEOS_ROOT`
   - `QUERY_PACKAGE_PATH`
   - `SUBMISSION_WORK_ROOT`, `SUBMISSION_ZIP_PATH`
2. Cấu hình `OPENAI_API_KEY` và `MODEL_NAME` trong `phantichquery/analyzer.py`.
3. Điều chỉnh các tham số retrieval, temporal, VLM và TRAKE trong `config.py` nếu cần.

> **Bảo mật:** không commit API key thật vào repository. Code hiện đọc key từ `phantichquery/analyzer.py`; nên thu hồi key đã lộ và chuyển sang đọc biến môi trường `OPENAI_API_KEY` trước khi chia sẻ hoặc triển khai.

## Dữ liệu truy vấn

`QUERY_PACKAGE_PATH` có thể là file ZIP hoặc thư mục chứa các file `.txt`. Tác vụ được xác định bắt buộc từ hậu tố tên file:

```text
query-p1-01-kis.txt
query-p1-02-qa.txt
query-p1-03-trake.txt
```

Kho video tại `VIDEOS_ROOT` phải chứa các archive khớp mẫu `Videos_*.zip`; bên trong là các file MP4 có tên theo `video_id`.

## Chạy

Chạy toàn bộ query package:

```bash
python main.py
```

Hoặc gọi trong notebook/Python:

```python
import main

# Chạy một truy vấn
result = main.run_query(
    "Một người đang chuẩn bị món ăn trong bếp",
    task="KIS",
)

# Tiếp tục từ query thứ 16, giữ lại các CSV trước đó
result = main.run_query_package(start_from=16, resume=True)
```

Với Google Colab, đổi tên `main_AIC` thành `main_AIC.ipynb`, mở notebook, mount Google Drive rồi cập nhật các đường dẫn runtime trong notebook.

## Đầu ra

Mỗi query tạo một CSV không có header trong thư mục `submission/`:

```text
KIS:   video_id,frame_id
QA:    video_id,frame_id,answer
TRAKE: video_id,event_1_frame,event_2_frame,...
```

Các CSV được kiểm tra định dạng, giới hạn tối đa 100 dòng và đóng gói thành:

```text
submission.zip
└── submission/
    ├── query-p1-01-kis.csv
    ├── query-p1-02-qa.csv
    └── query-p1-03-trake.csv
```

Khi `SUBMISSION_STRICT = True`, pipeline dừng và không tạo ZIP nếu bất kỳ query nào không sinh được kết quả hợp lệ. Cache video và segment được xóa sau mỗi query theo thiết lập hiện tại trong `main.py`.

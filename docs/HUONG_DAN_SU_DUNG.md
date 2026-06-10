# Hướng dẫn cài đặt & sử dụng DocuMind Workspace

Tài liệu này dành cho **người tải project từ GitHub** — chưa cần biết lập trình nhiều, chỉ cần làm đúng thứ tự bên dưới.

**Repository:** https://github.com/Mphuc310771/NBLM-small

---

## 1. Máy cần có gì?

| Yêu cầu | Ghi chú |
|--------|---------|
| **Windows 10/11** | Hướng dẫn này viết cho Windows (có file `.bat`) |
| **Python 3.10 trở lên** | Tải tại https://www.python.org/downloads/ — khi cài **bật** ☑ *Add python.exe to PATH* |
| **Git** (khuyến nghị) | https://git-scm.com/download/win — hoặc tải ZIP từ GitHub (mục 2b) |
| **RAM ≥ 8 GB** | Lần đầu tải model embedding (~vài trăm MB) |
| **Ổ trống ≥ 3 GB** | Virtualenv + thư viện + model |
| **Ít nhất 1 API key LLM** | Miễn phí/giới hạn: Groq, Gemini, OpenRouter… (mục 4) |

**Không bắt buộc** (chỉ khi cần tính năng đó):

- **Tesseract OCR** — chỉ khi bật chụp màn hình (`SCREEN_CAPTURE_ENABLED=true`)
- **Playwright** — khi cào URL/YouTube: `playwright install chromium`
- **ffmpeg** — khi upload video/audio để Whisper transcribe

---

## 2. Tải code từ GitHub

### Cách A — Git (khuyến nghị)

Mở **CMD** hoặc **PowerShell**:

```powershell
cd D:\Projects
git clone https://github.com/Mphuc310771/NBLM-small.git
cd NBLM-small
```

(Nếu bạn clone vào thư mục tên khác, ví dụ `doan`, thì `cd` vào đúng thư mục đó.)

### Cách B — Tải ZIP (không cài Git)

1. Vào https://github.com/Mphuc310771/NBLM-small  
2. **Code** → **Download ZIP**  
3. Giải nén → mở thư mục vừa giải nén  
4. Mọi lệnh `.bat` bên dưới chạy **trong thư mục đó** (có file `run_app.bat`)

---

## 3. Cài đặt lần đầu (chỉ 1 lần)

### Bước 1: Cài thư viện Python

**Double-click** file:

```text
setup_venv.bat
```

Hoặc PowerShell (phải có `.\`):

```powershell
.\setup_venv.bat
```

Script sẽ:

- Tạo môi trường `venv_win`
- Cài `requirements.txt`
- Tạo file `.env` từ `.env.example` (nếu chưa có)

Đợi đến khi hiện **`[OK] Xong`**. Lần đầu có thể **5–15 phút** tùy mạng.

### Bước 2: Cấu hình API key

Mở file **`.env`** (Notepad) trong thư mục project, điền **ít nhất một** key:

```env
GROQ_API_KEY=gsk_xxxxxxxx
```

Hoặc Gemini / OpenRouter / Mistral / SambaNova — app tự thử provider khác nếu một cái lỗi.

**Khuyến nghị cho máy người khác** (chạy nhẹ, ít log):

```env
SCREEN_CAPTURE_ENABLED=false
```

→ Tắt chụp màn hình + OCR nền (bảo mật hơn, không cần Tesseract, không cần gRPC Vision).

### Bước 3 (tùy chọn): Cào link web / YouTube

Chỉ chạy **một lần** sau `setup_venv.bat`:

```powershell
.\venv_win\Scripts\python.exe -m playwright install chromium
```

---

## 4. Lấy API key (miễn phí / dễ nhất)

| Nhà cung cấp | Đăng ký | Ghi vào `.env` |
|--------------|---------|----------------|
| **Groq** | https://console.groq.com | `GROQ_API_KEY=` |
| **Google Gemini** | https://aistudio.google.com/apikey | `GEMINI_API_KEY=` |
| **OpenRouter** | https://openrouter.ai/keys | `OPENROUTER_API_KEY=` |

Không chia sẻ key, không đưa key lên GitHub. File `.env` đã được git ignore.

---

## 5. Chạy ứng dụng mỗi ngày

### Cách dễ nhất

Double-click một trong các file:

| File | Mô tả |
|------|--------|
| **`MO_APP.bat`** | Mở app (khuyến nghị) |
| **`run_app.bat`** | Chạy server trực tiếp |

Trình duyệt tự mở khi server sẵn sàng:

```text
http://localhost:8000
```

**Tắt app:** vào cửa sổ CMD đang chạy server → **Ctrl+C**.

### Lần đầu chạy chậm?

Bình thường. Server đang tải **model embedding** (Hugging Face). Đợi **30–90 giây**, đừng tắt giữa chừng. Lần sau nhanh hơn.

---

## 6. Cách dùng nhanh (workflow)

```
Tải tài liệu → Hỏi chat RAG → Intelligence Suite (quiz, slide, podcast…)
```

### 6.1. Tải tài liệu

- **PDF / TXT:** kéo thả hoặc nút upload ở sidebar  
- **CSV / Excel:** upload → hỏi “vẽ biểu đồ…”, AI chạy pandas  
- **URL / YouTube:** sidebar → **Thêm URL** (cần Playwright nếu trang phức tạp)

### 6.2. Chat RAG

1. Gõ câu hỏi về nội dung đã upload  
2. **Tìm web** 🌐: chỉ bật khi muốn AI tìm thêm trên Internet (mặc định nên **tắt** nếu chỉ học từ tài liệu lớp)  
3. Xem trích dẫn nguồn dưới câu trả lời  

### 6.3. Intelligence Suite (Studio)

- **Tóm tắt, Quiz, Flashcard, FAQ, Timeline…** — bấm thẻ tương ứng  
- **Quiz:** hỏi số câu (1–10) khi tạo  
- **Slide:** chọn số slide → **Trình chiếu** (tab mới) → **Export PDF**  
- **Podcast:** tạo kịch bản 2 người + nghe TTS  
- **GraphRAG:** đồ thị thực thể–quan hệ  

Phím tắt: **`Ctrl+K`** — command palette (tìm lệnh nhanh).

### 6.4. Sổ tay (Notebook)

- Tạo/xóa sổ ở sidebar — mỗi sổ có tài liệu + chat riêng  
- Dữ liệu lưu **`app_data.db`** + **`chroma_db/`** trên máy local  

### 6.5. Xuất / lưu

| Thao tác | Ở đâu |
|----------|--------|
| Xuất chat | Header **⬇ Chat** |
| Slide PDF | Trang slide preview → **Export to PDF** |
| Ghi chú | **Pin Ghi Chú** trên tin nhắn AI |

---

## 7. Chạy mượt trên máy khác — checklist

1. **`SCREEN_CAPTURE_ENABLED=false`** trong `.env` (mặc định khuyến nghị)  
2. Chỉ chạy **một** cửa sổ `run_app.bat` (không mở 2 lần)  
3. Đóng tab trình duyệt cũ trước khi chạy lại nếu port 8000 báo lỗi  
4. Dùng **`venv_win`** — không cài package vào Python global lung tung  
5. Tắt VPN/proxy nếu tải model HF bị treo  
6. Upload PDF **< ~50 trang** lúc demo cho nhanh  
7. Chọn **một** API key ổn định (Groq/Gemini) thay vì để trống hết  

**Máy yếu (4 GB RAM):** chỉ dùng chat + upload TXT/PDF nhỏ; tránh podcast + slide 25 trang cùng lúc.

---

## 8. Xử lý lỗi thường gặp

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `Khong tim thay Python` | Cài Python 3.10+, bật PATH, chạy lại `setup_venv.bat` |
| `pip install that bai` | Mở CMD **Run as Administrator** hoặc tắt antivirus tạm; chạy lại `setup_venv.bat` |
| Trang trắng / không vào được | Đợi log `Uvicorn running`; thử http://127.0.0.1:8000 |
| Chat báo lỗi API | Kiểm tra key trong `.env`, không có dấu cách thừa; thử provider khác |
| Upload URL fail | Chạy `playwright install chromium` |
| Log `screen.png` liên tục | Đặt `SCREEN_CAPTURE_ENABLED=false`, khởi động lại app |
| Slide cũ / lệch layout | Tạo slide mới + Ctrl+F5 trên tab slide (`?v=4`) |
| Muốn xóa sạch dữ liệu | Tắt server → xóa `chroma_db/` và `app_data.db` → chạy lại |

---

## 9. Cấu trúc thư mục quan trọng

```text
NBLM-small/          (hoặc tên thư mục bạn clone)
├── setup_venv.bat   ← Cài lần đầu
├── run_app.bat      ← Chạy app
├── MO_APP.bat       ← Double-click cho người dùng
├── .env             ← API key (tự tạo, không commit)
├── .env.example     ← Mẫu cấu hình
├── app/             ← Mã nguồn
├── docs/            ← Báo cáo + hướng dẫn này
├── chroma_db/       ← Vector DB (tự sinh)
└── app_data.db      ← SQLite chat/sổ tay (tự sinh)
```

---

## 10. Báo cáo đồ án / demo giảng viên

- Báo cáo kỹ thuật: [`BAO_CAO_DO_AN.md`](BAO_CAO_DO_AN.md)  
- Demo gợi ý: upload 1 PDF môn học → 3 câu hỏi RAG → tạo quiz 5 câu → slide 10 trang → GraphRAG  
- Nhấn mạnh: Clean Architecture, RAG, SQLite + ChromaDB, tắt screen capture khi public  

---

## 11. Hỗ trợ

Mở **Issue** trên GitHub kèm:

- Windows version  
- Dòng lỗi trong cửa sổ CMD (copy 10–20 dòng cuối)  
- Đã chạy `setup_venv.bat` chưa, có `.env` + API key chưa  

---

*Tài liệu đồng bộ với README.md — cập nhật khi đổi tên repo hoặc script khởi động.*

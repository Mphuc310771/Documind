# NBLM Research

> A self-hosted NotebookLM-style research workspace: RAG chat over your own documents, data analysis, knowledge graphs, podcasts and cinematic slides — in one FastAPI app.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-RAG-009688)
![License](https://img.shields.io/badge/License-MIT-green)

FastAPI RAG workspace with PDF/TXT upload, **CSV/XLSX data analysis**, **URL & YouTube indexing**, ChromaDB, streaming chat, Intelligence Suite (quiz, mindmap, flashcards…), 2-host podcast, and cinematic slide export. The UI includes live notebook stats, a command palette (`Ctrl+K`), an opt-in **🌐 Web Search** toggle (no silent hallucination), and chat export.

> **Tiếng Việt — hướng dẫn đầy đủ cho người tải GitHub:** [docs/HUONG_DAN_SU_DUNG.md](docs/HUONG_DAN_SU_DUNG.md) (cài đặt, API key, cách dùng, chạy mượt, xử lý lỗi).

## Quick start (Windows)

| Bước | Việc cần làm |
|------|----------------|
| 1 | `git clone https://github.com/Mphuc310771/NBLM-small.git` → `cd NBLM-small` |
| 2 | Double-click **`setup_venv.bat`** (một lần) |
| 3 | Sửa **`.env`**: ít nhất một `GROQ_API_KEY` (hoặc Gemini/OpenRouter…) |
| 4 | Double-click **`MO_APP.bat`** hoặc **`run_app.bat`** |
| 5 | Mở http://localhost:8000 — upload PDF → hỏi chat → Studio |

Khuyến nghị trong `.env`: `SCREEN_CAPTURE_ENABLED=false` (nhẹ hơn, bảo mật hơn cho người mới).

## ✨ Features

- **RAG chat** over your documents with citations and a streaming UI.
- **Multi-format ingest**: TXT, PDF, CSV/XLSX, web URLs, YouTube transcripts, and audio/video (Whisper).
- **Data analysis**: ask about a spreadsheet → the AI runs pandas/matplotlib in a sandbox and renders charts inline.
- **Intelligence Suite**: summaries, quizzes, flashcards, FAQ, timelines, study guides, mindmaps.
- **GraphRAG**: interactive, zoomable knowledge graph of entities & relationships.
- **Canvas Workspace**: edit AI output in place ("rewrite this selection").
- **2-host podcast** generation (Edge-TTS) and **cinematic slide export**.
- **Multi-LLM** with automatic fallback (Groq, Gemini, OpenRouter, SambaNova, Mistral).
- **Optional**: JWT multi-tenant auth, Celery+Redis task queue — both off by default.

## 🧱 Tech Stack

FastAPI · ChromaDB (vector search) · SQLite (relational store) · sentence-transformers · gRPC + Tesseract (OCR) · Playwright · Edge-TTS · vanilla JS frontend (marked, KaTeX, Mermaid, vis-network).

Upload a `.csv`/`.xlsx` then ask e.g. *"vẽ biểu đồ điểm theo tên"* — the AI writes pandas/matplotlib in the sandbox and renders the chart inline.

**Canvas Workspace** (header `🎨 Canvas` or `Ctrl+K`): a slide-over editor with live Markdown preview. Select any text and click **✨ Sửa với AI** to rewrite that selection in place (or the whole doc if nothing is selected). Export `.md`, copy, or save into the Intelligence Suite.

**GraphRAG — Đồ Thị Tri Thức** (Intelligence Suite card 🧠): the LLM extracts entities + relationships from your documents and renders an interactive, zoomable knowledge graph (vis-network) with color-coded entity types.

**Clipboard quick-summary** (header `📋 Clip` or `Ctrl+K`): reads your clipboard (on localhost) and offers instant AI summary, add-to-notes, or ask-in-chat.

**Audio/Video transcription (Whisper)** — upload `.mp3/.mp4/.wav/.m4a`; the transcript is indexed for RAG. Optional, install when needed:

```bash
pip install faster-whisper   # plus ffmpeg in PATH for video
```

**Background task queue (Celery + Redis)** — optional, disabled by default. Enable in `.env` (`CELERY_ENABLED=true`, `REDIS_URL=...`), install `celery redis`, run a Redis server, then start a worker:

```bash
celery -A app.infrastructure.task_queue.celery_app worker --loglevel=info
```

When disabled, heavy jobs run inline (no behavior change). Task status: `GET /tasks/{id}`.

## Requirements

- Python 3.10 or newer
- Git
- At least one LLM API key for AI generation features
The app can start without API keys, but chat/slides/quiz/summary generation will fail until at least one provider key is configured.

For the screen capture OCR feature, install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki):

```powershell
winget install UB-Mannheim.TesseractOCR
```

Or download the installer from https://github.com/UB-Mannheim/tesseract/wiki and install to the default path `C:\Program Files\Tesseract-OCR\`.

## Setup

Clone the repository:

```powershell
git clone https://github.com/Mphuc310771/NBLM-small.git
cd NBLM-small
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional, for Playwright browser automation:

```powershell
playwright install chromium
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and add one or more API keys:

```env
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
SAMBANOVA_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

## Run

Windows — double-click hoặc CMD:

```bat
setup_venv.bat
run_app.bat
```

PowerShell (phải có `.\`):

```powershell
.\setup_venv.bat
.\run_app.bat
```

Manual run:

```powershell
python app/workers/vision_grpc_server.py
python -m uvicorn app.main:app --port 8000
```

Open:

```text
http://localhost:8000
```

## Common Workflow

1. Open the web app at `http://localhost:8000`.
2. Upload `.txt`/`.pdf` or paste a **web/YouTube URL** (sidebar → Thêm URL).
3. Ask questions against your sources (RAG + web fallback).
4. Use **Intelligence Suite** or `Ctrl+K` for summaries, quizzes, podcasts, slides.
5. Open slide preview and **Export to PDF**. Export chat via header **⬇ Chat**.

## Architecture

- **ChromaDB** (`chroma_db/`) — semantic vector search only.
- **SQLite** (`app_data.db`) — relational source of truth for Notebooks, Documents, ChatHistory and Users (survives browser cache clears, shared across devices). The frontend uses localStorage as an offline cache and syncs to the server.

## Optional authentication (multi-tenant)

Disabled by default — the app runs without any login. To enable private per-user workspaces, set in `.env`:

```env
AUTH_ENABLED=true
JWT_SECRET=<a long random string>
```

Then each visitor must register/login; notebooks, documents and chats are scoped to their account (JWT bearer tokens, PBKDF2-hashed passwords — stdlib only, no extra deps).

## Local Data

The app creates local runtime data that is intentionally not committed:

- `.env`
- `.venv/`, `venv/`, `venv_win/`
- `chroma_db/`
- `app_data.db` (SQLite)
- `scratch/`
- `app/static/outputs/`, `app/static/datasets/`

Delete `chroma_db/` if you want to reset all indexed documents.

## 📁 Project Structure

```
app/
├── domain/          # Entities & interfaces (models.py, interfaces.py)
├── application/     # Use cases (RAG, upload, podcast, slides, graph, auth flows…)
├── infrastructure/  # Adapters & stores (LLM adapters, ChromaDB, SQLite, auth, task queue)
├── presentation/    # FastAPI routes (api.py)
├── workers/         # gRPC vision/OCR worker
├── core/            # config, events, protobufs
└── static/          # Single-page frontend (index.html)
```

The codebase follows **Clean Architecture** (domain → application → infrastructure → presentation) and uses Adapter, Strategy/Fallback, Repository, Dependency Injection, Producer–Consumer and Facade patterns.

## 🔒 Privacy

By default **`.env.example` sets `SCREEN_CAPTURE_ENABLED=false`** so new clones do not screenshot the desktop. If you enable it, a background daemon OCRs the screen into ChromaDB (privacy-sensitive; excluded from chat citations but still processed locally). See [docs/HUONG_DAN_SU_DUNG.md](docs/HUONG_DAN_SU_DUNG.md) §7 for a smooth setup on other PCs.

## Notes

- First startup can be slow because `sentence-transformers` downloads the embedding model.
- Screen capture OCR requires Tesseract to be installed. Without it, screen capture indexing is disabled (a warning is logged).
- Never commit real API keys. Keep them only in `.env`.

## 📄 License

Released under the [MIT License](LICENSE).

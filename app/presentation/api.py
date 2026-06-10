import re
import json
import logging
import uuid
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from app.core.config import settings
from app.infrastructure.auth import hash_password, verify_password, create_token, decode_token
from app.domain.models import QueryRequest, SynthesizeNotesRequest, StudioGenerateRequest, PodcastGenerateRequest, SlideGenerateRequest, UploadURLRequest, NotebookCreateRequest, ChatMessageRequest, CanvasEditRequest, AuthRequest, QuickSummarizeRequest
from app.application.podcast_usecase import PodcastUseCase
from app.application.synthesis_usecase import SynthesisUseCase
from app.application.studio_usecase import StudioUseCase
from app.application.slide_usecase import SlideGeneratorUseCase
from app.infrastructure.vector_store import ChromaDBStore
from app.infrastructure.sql_store import SQLStore
from app.infrastructure.groq_adapter import GroqAdapter
from app.infrastructure.gemini_adapter import GeminiAdapter
from app.infrastructure.openrouter_adapter import OpenRouterAdapter
from app.infrastructure.sambanova_adapter import SambaNovaAdapter
from app.infrastructure.mistral_adapter import MistralAdapter
from app.infrastructure.fallback_llm import FallbackLLMService
from app.application.rag_usecase import RAGUseCase
from app.application.upload_usecase import UploadUseCase
from app.application.data_upload_usecase import DataUploadUseCase
from app.application.transcription_usecase import TranscriptionUseCase
from app.application.upload_url_usecase import UploadURLUseCase
from app.application.quiz_usecase import QuizGeneratorUseCase
from app.application.summary_usecase import DocumentSummaryUseCase
from app.application.graph_usecase import GraphUseCase
from app.application.knowledge_graph_usecase import KnowledgeGraphUseCase
from app.application.delete_usecase import DeleteUseCase

logger = logging.getLogger(__name__)
router = APIRouter()

# ====================================================================
# Dependency Injection — Singleton Pattern
# Instantiate dependencies once at module level so the embedding model
# and ChromaDB client are reused across all requests.
# ====================================================================
vector_store = ChromaDBStore()
sql_store = SQLStore()
groq_provider = GroqAdapter()
gemini_provider = GeminiAdapter()
openrouter_provider = OpenRouterAdapter()
sambanova_provider = SambaNovaAdapter()
mistral_provider = MistralAdapter()
llm_service = FallbackLLMService(
    groq_adapter=groq_provider, 
    gemini_adapter=gemini_provider, 
    openrouter_adapter=openrouter_provider,
    sambanova_adapter=sambanova_provider,
    mistral_adapter=mistral_provider
)


def get_rag_use_case() -> RAGUseCase:
    return RAGUseCase(vector_store=vector_store, llm_service=llm_service)


def get_upload_use_case() -> UploadUseCase:
    return UploadUseCase(vector_store=vector_store)


def get_data_upload_use_case() -> DataUploadUseCase:
    return DataUploadUseCase(vector_store=vector_store)


def get_transcription_use_case() -> TranscriptionUseCase:
    return TranscriptionUseCase(vector_store=vector_store)


def get_upload_url_use_case() -> UploadURLUseCase:
    return UploadURLUseCase(vector_store=vector_store)



def get_quiz_use_case() -> QuizGeneratorUseCase:
    return QuizGeneratorUseCase(vector_store=vector_store, llm_service=llm_service)


def get_summary_use_case() -> DocumentSummaryUseCase:
    return DocumentSummaryUseCase(vector_store=vector_store, llm_service=llm_service)


def get_graph_use_case() -> GraphUseCase:
    return GraphUseCase(vector_store=vector_store, llm_service=llm_service)


def get_knowledge_graph_use_case() -> KnowledgeGraphUseCase:
    return KnowledgeGraphUseCase(vector_store=vector_store, llm_service=llm_service)


def get_delete_use_case() -> DeleteUseCase:
    return DeleteUseCase(vector_store=vector_store)


def get_podcast_use_case() -> PodcastUseCase:
    return PodcastUseCase(vector_store=vector_store, llm_service=llm_service)


def get_slide_use_case() -> SlideGeneratorUseCase:
    return SlideGeneratorUseCase(vector_store=vector_store, llm_service=llm_service)


def get_synthesis_use_case() -> SynthesisUseCase:
    return SynthesisUseCase(llm_service=llm_service)


def get_studio_use_case() -> StudioUseCase:
    return StudioUseCase(vector_store=vector_store, llm_service=llm_service)


# ====================================================================
# Optional authentication (multi-tenant). Controlled by settings.AUTH_ENABLED.
# When disabled (default), all routes behave exactly as before.
# ====================================================================
def get_current_user_id(authorization: str | None = Header(default=None)) -> str | None:
    """Returns the authenticated user id, or None when auth is disabled."""
    if not settings.AUTH_ENABLED:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Yêu cầu đăng nhập.")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token, settings.JWT_SECRET)
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn.")
    return payload["sub"]


def _assert_notebook_access(notebook_id: str, user_id: str | None):
    """When auth is on, ensure the notebook belongs to the current user."""
    if not settings.AUTH_ENABLED or user_id is None:
        return
    owner = sql_store.notebook_owner(notebook_id)
    # Allow access to own notebooks or brand-new ids (created on first write)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập sổ tay này.")


@router.get("/auth/config")
def auth_config():
    return {"auth_enabled": settings.AUTH_ENABLED}


@router.post("/auth/register")
def auth_register(request: AuthRequest):
    if not settings.AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Xác thực đang tắt trên máy chủ.")
    username = (request.username or "").strip().lower()
    if len(username) < 3 or len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Tên đăng nhập ≥ 3 ký tự, mật khẩu ≥ 6 ký tự.")
    if sql_store.get_user_by_username(username):
        raise HTTPException(status_code=409, detail="Tên đăng nhập đã tồn tại.")
    user_id = "u_" + uuid.uuid4().hex[:12]
    sql_store.create_user(user_id, username, hash_password(request.password))
    # Give the new user a personal default notebook
    sql_store.create_notebook("nb_" + uuid.uuid4().hex[:9], "Sổ tay của tôi", owner=user_id)
    token = create_token({"sub": user_id, "username": username}, settings.JWT_SECRET, settings.JWT_EXPIRE_HOURS)
    return {"token": token, "username": username}


@router.post("/auth/login")
def auth_login(request: AuthRequest):
    if not settings.AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Xác thực đang tắt trên máy chủ.")
    username = (request.username or "").strip().lower()
    user = sql_store.get_user_by_username(username)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")
    token = create_token({"sub": user["id"], "username": username}, settings.JWT_SECRET, settings.JWT_EXPIRE_HOURS)
    return {"token": token, "username": username}


@router.get("/auth/me")
def auth_me(user_id: str | None = Depends(get_current_user_id)):
    if user_id is None:
        return {"auth_enabled": False}
    user = sql_store.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Người dùng không tồn tại.")
    return {"auth_enabled": True, "id": user["id"], "username": user["username"]}


@router.get("/tasks/config")
def tasks_config():
    from app.infrastructure import task_queue
    return {"celery_enabled": task_queue.is_enabled(), "redis_url": settings.REDIS_URL if settings.CELERY_ENABLED else None}


@router.get("/tasks/{task_id}")
def task_status(task_id: str):
    from app.infrastructure import task_queue
    return task_queue.get_status(task_id)


@router.get("/api/status")
def api_status():
    """Health check used by IDE/tools polling the running server."""
    chromadb_ok = True
    try:
        vector_store.collection.count()
    except Exception:
        chromadb_ok = False
    return {
        "status": "ok" if chromadb_ok else "degraded",
        "service": "DocuMind Workspace",
        "chromadb": chromadb_ok,
    }


@router.post("/ask")
def ask_question(request: QueryRequest, use_case: RAGUseCase = Depends(get_rag_use_case)):
    """
    Endpoint to process a user query using Retrieval-Augmented Generation.
    Returns a stream of JSON messages (either citation metadata or response tokens)
    formatted as a Server-Sent Event (SSE) stream.
    """
    def event_generator():
        try:
            for chunk in use_case.execute(request.query, provider=request.provider, notebook_id=request.notebook_id, search_web=request.search_web):
                # Format chunk as a standard Server-Sent Event (SSE) message
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error in ask event stream: {e}", exc_info=True)
            err_msg = {"type": "error", "content": f"Internal server error: {str(e)}"}
            yield f"data: {json.dumps(err_msg, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/upload")
async def upload_document(
    notebook_id: str = "default",
    file: UploadFile = File(...),
    use_case: UploadUseCase = Depends(get_upload_use_case),
    data_use_case: DataUploadUseCase = Depends(get_data_upload_use_case),
    transcription_use_case: TranscriptionUseCase = Depends(get_transcription_use_case),
):
    """
    Endpoint to upload a document. Text/PDF are chunked into ChromaDB;
    tabular data (CSV/XLSX) is persisted for pandas analysis; audio/video
    (MP3/MP4/WAV/M4A) is transcribed with Whisper and indexed.
    """
    name = (file.filename or "").lower()
    audio_exts = ('.mp3', '.mp4', '.wav', '.m4a', '.mpeg', '.mpga', '.webm')
    if not name.endswith(('.txt', '.pdf', '.csv', '.xlsx', '.xls') + audio_exts):
        raise HTTPException(status_code=400, detail="Hỗ trợ .txt, .pdf, .csv, .xlsx, .mp3, .mp4, .wav, .m4a.")

    try:
        await file.seek(0)
        file_content = await file.read()
        if name.endswith(audio_exts):
            result = transcription_use_case.execute(file_content=file_content, filename=file.filename, notebook_id=notebook_id)
            doc_type = "media"
        elif name.endswith(('.csv', '.xlsx', '.xls')):
            result = data_use_case.execute(file_content=file_content, filename=file.filename, notebook_id=notebook_id)
            doc_type = "dataset"
        else:
            result = use_case.execute(file_content=file_content, filename=file.filename, notebook_id=notebook_id)
            doc_type = "pdf" if name.endswith(".pdf") else "text"
        try:
            sql_store.add_document(notebook_id, file.filename, doc_type, result.get("total_chunks", 0))
        except Exception as e:
            logger.warning(f"SQLStore add_document failed: {e}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing /upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload-url")
async def upload_url(request: UploadURLRequest, use_case: UploadURLUseCase = Depends(get_upload_url_use_case)):
    """
    Endpoint to download a Webpage URL or YouTube Video transcript,
    chunk it, and store embeddings in ChromaDB.
    """
    try:
        result = await use_case.execute(url=request.url, notebook_id=request.notebook_id)
        try:
            sql_store.add_document(request.notebook_id, result.get("filename", request.url), "url", result.get("total_chunks", 0))
        except Exception as e:
            logger.warning(f"SQLStore add_document (url) failed: {e}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing /upload-url: {e}", exc_info=True)
        msg = str(e)
        if "Không thể trích xuất" in msg or "HTTP Error" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=500, detail=f"Upload URL failed: {msg}")



@router.get("/generate-quiz")
def generate_quiz(notebook_id: str = "default", chat_context: str | None = None, num_questions: int = 5, use_case: QuizGeneratorUseCase = Depends(get_quiz_use_case)):
    """
    Endpoint to generate a multiple choice quiz (default 5 questions, 1-10) from document chunks.
    """
    result = use_case.execute(notebook_id=notebook_id, chat_context=chat_context, num_questions=num_questions)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result.get("questions")


@router.get("/summarize")
def summarize_document(notebook_id: str = "default", use_case: DocumentSummaryUseCase = Depends(get_summary_use_case)):
    """
    Endpoint to generate a concise summary from the uploaded documents.
    """
    result = use_case.execute(notebook_id=notebook_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("summary"))
    return result


@router.get("/generate-graph")
def generate_graph(notebook_id: str = "default", chat_context: str | None = None, use_case: GraphUseCase = Depends(get_graph_use_case)):
    """
    Endpoint to generate a Mermaid.js visual mindmap from the uploaded documents.
    """
    result = use_case.execute(notebook_id=notebook_id, chat_context=chat_context)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("graph"))
    return result


@router.get("/generate-knowledge-graph")
def generate_knowledge_graph(notebook_id: str = "default", chat_context: str | None = None, use_case: KnowledgeGraphUseCase = Depends(get_knowledge_graph_use_case)):
    """
    GraphRAG: extract entities + relationships into an interactive knowledge graph.
    """
    result = use_case.execute(notebook_id=notebook_id, chat_context=chat_context)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.delete("/delete-document")
def delete_document(filename: str, notebook_id: str = "default", use_case: DeleteUseCase = Depends(get_delete_use_case)):
    """
    Endpoint to delete all chunks of a document from ChromaDB.
    """
    result = use_case.execute(filename=filename, notebook_id=notebook_id)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("message"))
    try:
        sql_store.remove_document(notebook_id, filename)
    except Exception as e:
        logger.warning(f"SQLStore remove_document failed: {e}")
    return result


# ====================================================================
# Notebooks & Chat history (SQLite-backed relational layer)
# ====================================================================
@router.get("/notebooks")
def list_notebooks(user_id: str | None = Depends(get_current_user_id)):
    return sql_store.list_notebooks(owner=user_id)


@router.post("/notebooks")
def create_notebook(request: NotebookCreateRequest, user_id: str | None = Depends(get_current_user_id)):
    name = (request.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Tên sổ tay không được để trống.")
    notebook_id = request.id or ("nb_" + uuid.uuid4().hex[:9])
    return sql_store.create_notebook(notebook_id, name, owner=user_id)


@router.delete("/notebooks/{notebook_id}")
def delete_notebook(notebook_id: str, user_id: str | None = Depends(get_current_user_id)):
    if notebook_id == "default":
        raise HTTPException(status_code=400, detail="Không thể xóa DocuMind Workspace mặc định.")
    _assert_notebook_access(notebook_id, user_id)
    # Remove vector chunks for this notebook as well
    try:
        if vector_store.collection.count() > 0:
            vector_store.collection.delete(where={"notebook_id": notebook_id})
    except Exception as e:
        logger.warning(f"Failed to delete ChromaDB chunks for notebook {notebook_id}: {e}")
    sql_store.delete_notebook(notebook_id)
    return {"success": True, "notebook_id": notebook_id}


@router.get("/documents")
def list_documents(notebook_id: str = "default"):
    return sql_store.list_documents(notebook_id)


@router.post("/quick-summarize")
def quick_summarize(request: QuickSummarizeRequest):
    """
    Clipboard helper: summarize an arbitrary piece of text into a few concise
    Vietnamese bullet points. Used by the clipboard quick-summary popup.
    """
    text = (request.text or "").strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Nội dung quá ngắn để tóm tắt.")
    if len(text) > 12000:
        text = text[:12000]

    system_prompt = (
        "You are a concise summarizer. Summarize the provided text into 3-5 short bullet points "
        "in the SAME language as the text (Vietnamese if the text is Vietnamese). Return only the bullets."
    )
    try:
        tokens = list(llm_service.generate_answer(
            context=text, query="Tóm tắt nội dung này.", system_prompt=system_prompt, provider=request.provider
        ))
        summary = "".join(tokens).strip()
        if not summary:
            raise HTTPException(status_code=502, detail="AI không trả về nội dung.")
        return {"success": True, "summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick summarize failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi tóm tắt: {str(e)}")


@router.post("/canvas/edit")
def canvas_edit(request: CanvasEditRequest):
    """
    Interactive Canvas: rewrite the selected text (or the whole document if no
    selection) according to the user's instruction. Returns the rewritten text only.
    """
    instruction = (request.instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="Cần có chỉ dẫn chỉnh sửa.")

    selection = (request.selection or "").strip()
    target = selection if selection else request.content
    if not target.strip():
        raise HTTPException(status_code=400, detail="Không có nội dung để chỉnh sửa.")

    system_prompt = (
        "You are a precise writing assistant editing a document on an interactive canvas. "
        "Rewrite ONLY the target text according to the user's instruction, preserving the original "
        "language unless asked otherwise. Return the rewritten target text as plain Markdown with NO "
        "preamble, NO explanations, and NO surrounding code fences."
    )
    context = f"FULL DOCUMENT (context only):\n{request.content}\n\nTARGET TEXT TO REWRITE:\n{target}"
    try:
        tokens = list(llm_service.generate_answer(
            context=context, query=instruction, system_prompt=system_prompt, provider=request.provider
        ))
        result = "".join(tokens).strip()
        if result.startswith("```"):
            result = re.sub(r"^```[a-zA-Z]*\n?", "", result)
            result = re.sub(r"\n?```$", "", result).strip()
        if not result:
            raise HTTPException(status_code=502, detail="AI không trả về nội dung.")
        return {"success": True, "result": result, "replaced_selection": bool(selection)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Canvas edit failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi chỉnh sửa Canvas: {str(e)}")


@router.get("/chat-history")
def get_chat_history(notebook_id: str = "default"):
    return sql_store.get_chat_messages(notebook_id)


@router.post("/chat-history")
def add_chat_message(request: ChatMessageRequest):
    sql_store.add_chat_message(request.notebook_id, request.role, request.content, request.citations)
    return {"success": True}


@router.delete("/chat-history")
def clear_chat_history(notebook_id: str = "default"):
    sql_store.clear_chat(notebook_id)
    return {"success": True, "notebook_id": notebook_id}


@router.get("/notebook-stats")
def get_notebook_stats(notebook_id: str = "default"):
    """
    Dashboard metrics: document count, chunk count, and estimated context size.
    """
    try:
        results = vector_store.collection.get(
            where={"notebook_id": notebook_id},
            include=["metadatas", "documents"],
        )
        metadatas = results.get("metadatas", []) or []
        documents = results.get("documents", []) or []

        sources = set()
        for meta in metadatas:
            if meta and meta.get("source") and meta["source"] != "screen_capture":
                sources.add(meta["source"])

        char_count = sum(len(doc or "") for doc in documents)
        return {
            "notebook_id": notebook_id,
            "document_count": len(sources),
            "chunk_count": len(metadatas),
            "char_count": char_count,
            "estimated_tokens": 0 if not metadatas else char_count // 4,
        }
    except Exception as e:
        logger.error(f"Error fetching notebook stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uploaded-files")
def get_uploaded_files(notebook_id: str = "default"):
    """
    Get the list of all unique document names currently in ChromaDB (excluding screen_capture) for this notebook.
    """
    try:
        results = vector_store.collection.get(where={"notebook_id": notebook_id}, include=["metadatas"])
        metadatas = results.get("metadatas", []) or []
        
        source_counts = {}
        for meta in metadatas:
            if meta and "source" in meta:
                src = meta["source"]
                if src != "screen_capture":
                    source_counts[src] = source_counts.get(src, 0) + 1
                
        file_items = []
        for src, chunks_count in sorted(source_counts.items()):
            file_items.append({
                "name": src,
                "status": f"✓ {chunks_count} chunks",
                "type": "success"
               })
        return file_items
    except Exception as e:
        logger.error(f"Error fetching uploaded files list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-podcast")
async def generate_podcast(request: PodcastGenerateRequest, use_case: PodcastUseCase = Depends(get_podcast_use_case)):
    """
    Generate an audio podcast briefing dialogue script from current documents.
    """
    result = await use_case.execute(
        provider=request.provider, 
        notebook_id=request.notebook_id, 
        custom_instructions=request.custom_instructions
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.post("/generate-slides")
def generate_slides(request: SlideGenerateRequest, use_case: SlideGeneratorUseCase = Depends(get_slide_use_case)):
    """
    Generate interactive presentation slide content (JSON deck) from documents.
    """
    result = use_case.execute(
        provider=request.provider,
        notebook_id=request.notebook_id,
        num_slides=request.num_slides,
        chat_context=request.chat_context
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result



@router.post("/synthesize-notes")
def synthesize_notes(request: SynthesizeNotesRequest, use_case: SynthesisUseCase = Depends(get_synthesis_use_case)):
    """
    Synthesize user notes into study guides, essays, summaries, or contradiction logs.
    """
    result = use_case.execute(notes=request.notes, action=request.action, provider=request.provider)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("result"))
    return result


@router.post("/studio/generate")
def studio_generate(request: StudioGenerateRequest, use_case: StudioUseCase = Depends(get_studio_use_case)):
    """
    Generate Studio content (flashcards, FAQ, timeline, study guide, briefing) from documents.
    """
    result = use_case.execute(content_type=request.content_type, provider=request.provider, notebook_id=request.notebook_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result

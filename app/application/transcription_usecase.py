import os
import re
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)

MEDIA_DIR = os.path.join("app", "static", "datasets")
_WHISPER_MODEL = None


class TranscriptionUseCase:
    """
    Transcribe uploaded audio/video (mp3/mp4/wav/m4a) with faster-whisper (offline)
    and index the transcript into ChromaDB.

    faster-whisper is an optional, heavy dependency. It is imported lazily so the
    rest of the app runs without it; a clear, actionable error is raised if missing.
    """

    def __init__(self, vector_store: ChromaDBStore, model_size: str = None):
        self.vector_store = vector_store
        self.model_size = model_size or os.environ.get("WHISPER_MODEL", "base")

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name))

    def _get_model(self):
        global _WHISPER_MODEL
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ValueError(
                "Tính năng phiên âm cần thư viện 'faster-whisper'. Cài đặt: "
                "pip install faster-whisper  (và cần ffmpeg trong PATH để xử lý video)."
            )
        logger.info(f"Loading faster-whisper model '{self.model_size}' (CPU/int8)...")
        _WHISPER_MODEL = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return _WHISPER_MODEL

    def execute(self, file_content: bytes, filename: str, notebook_id: str = "default") -> dict:
        model = self._get_model()  # raises ValueError if not installed

        nb_dir = os.path.join(MEDIA_DIR, self._safe_name(notebook_id))
        os.makedirs(nb_dir, exist_ok=True)
        media_path = os.path.join(nb_dir, self._safe_name(filename))
        with open(media_path, "wb") as f:
            f.write(file_content)

        logger.info(f"Transcribing '{filename}'...")
        try:
            segments, info = model.transcribe(media_path, beam_size=1)
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            raise ValueError(f"Không thể phiên âm tệp (có thể thiếu ffmpeg cho video): {e}")

        if not transcript or len(transcript) < 5:
            raise ValueError("Không trích xuất được lời thoại từ tệp âm thanh/video.")

        header = f"[TRANSCRIPT] {filename} (ngôn ngữ: {getattr(info, 'language', '?')})\n\n"
        full_text = header + transcript

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(full_text)
        if chunks:
            metadatas = [
                {"source": filename, "chunk_index": i, "notebook_id": notebook_id, "is_transcript": True}
                for i in range(len(chunks))
            ]
            self.vector_store.add_documents(texts=chunks, metadatas=metadatas)

        logger.info(f"Transcribed '{filename}': {len(transcript)} chars, {len(chunks)} chunks.")
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "language": getattr(info, "language", "?"),
            "message": f"Đã phiên âm '{filename}' thành {len(chunks)} phân đoạn và lập chỉ mục.",
        }

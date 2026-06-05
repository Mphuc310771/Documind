import io
import os
import re
import logging

import pandas as pd

from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)

DATASETS_DIR = os.path.join("app", "static", "datasets")


class DataUploadUseCase:
    """
    Handles tabular data files (.csv / .xlsx).

    Unlike plain text/PDF, the raw file is persisted to disk so the Code
    Interpreter sandbox can load it with pandas for real data analysis.
    A schema + preview summary is indexed into ChromaDB so the RAG model
    knows the columns and the file path it can analyze.
    """

    def __init__(self, vector_store: ChromaDBStore):
        self.vector_store = vector_store

    @staticmethod
    def _safe_name(filename: str) -> str:
        base = os.path.basename(filename)
        return re.sub(r"[^A-Za-z0-9._-]", "_", base)

    def _read_dataframe(self, file_content: bytes, filename: str) -> pd.DataFrame:
        lower = filename.lower()
        if lower.endswith(".csv"):
            try:
                return pd.read_csv(io.BytesIO(file_content))
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(file_content), encoding="latin-1")
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(file_content))
        raise ValueError("Định dạng dữ liệu không được hỗ trợ (chỉ .csv, .xlsx).")

    def _build_summary(self, df: pd.DataFrame, filename: str, data_path: str) -> str:
        rel_path = data_path.replace("\\", "/")
        lines = [
            f"[DATASET] {filename}",
            f"File path for pandas analysis: {rel_path}",
            f"Rows: {df.shape[0]} | Columns: {df.shape[1]}",
            "",
            "Columns and dtypes:",
        ]
        for col in df.columns:
            lines.append(f"- {col} ({df[col].dtype})")

        lines.append("")
        lines.append("Preview (first 5 rows):")
        try:
            lines.append(df.head(5).to_markdown(index=False))
        except Exception:
            lines.append(df.head(5).to_string(index=False))

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            lines.append("")
            lines.append("Numeric statistics:")
            try:
                lines.append(numeric.describe().to_markdown())
            except Exception:
                lines.append(numeric.describe().to_string())

        return "\n".join(lines)

    def execute(self, file_content: bytes, filename: str, notebook_id: str = "default") -> dict:
        df = self._read_dataframe(file_content, filename)

        nb_dir = os.path.join(DATASETS_DIR, self._safe_name(notebook_id))
        os.makedirs(nb_dir, exist_ok=True)
        safe_file = self._safe_name(filename)
        data_path = os.path.join(nb_dir, safe_file)
        with open(data_path, "wb") as f:
            f.write(file_content)

        summary = self._build_summary(df, filename, data_path)

        # Store summary as a small number of chunks (keep schema in one piece)
        chunks = [summary[i:i + 3000] for i in range(0, len(summary), 3000)] or [summary]
        metadatas = [
            {
                "source": filename,
                "chunk_index": i,
                "notebook_id": notebook_id,
                "is_dataset": True,
                "data_path": data_path.replace("\\", "/"),
            }
            for i in range(len(chunks))
        ]
        self.vector_store.add_documents(texts=chunks, metadatas=metadatas)

        logger.info(f"Indexed dataset '{filename}' ({df.shape[0]}x{df.shape[1]}) for notebook '{notebook_id}'.")
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "message": f"Đã nạp dữ liệu '{filename}' ({df.shape[0]} dòng × {df.shape[1]} cột). Hãy hỏi để AI phân tích.",
        }

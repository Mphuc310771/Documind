import io
import logging

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class UploadUseCase:
    def __init__(self, vector_store: ChromaDBStore):
        self.vector_store = vector_store

    def execute(self, file_content: bytes, filename: str, notebook_id: str = "default") -> dict:
        logger.debug(f"File size received: {len(file_content)} bytes")
        combined_text = ""

        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                logger.debug(f"Total pages parsed: {len(pdf.pages)}")
                for page in pdf.pages:
                    # 1. Safely extract normal text
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n\n"

                    # 2. Safely extract tables
                    tables = page.extract_tables() or []
                    for table in tables:
                        if not table:
                            continue
                        for row in table:
                            # CRITICAL: Drop None rows entirely
                            if row is None:
                                continue

                            # CRITICAL: Sanitize every cell and force to string. Remove all newlines.
                            clean_row = []
                            for cell in row:
                                if cell is None:
                                    clean_row.append(" ")
                                else:
                                    clean_str = str(cell).replace('\n', ' ').replace('\r', '').strip()
                                    clean_row.append(clean_str if clean_str else " ")

                            # Join row into Markdown format
                            markdown_row = "| " + " | ".join(clean_row) + " |\n"
                            combined_text += markdown_row
                        combined_text += "\n"
        else:
            combined_text = file_content.decode("utf-8")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(combined_text)
        logger.debug(f"Total chunks generated: {len(chunks)}")

        if chunks:
            metadatas = [{"source": filename, "chunk_index": i, "notebook_id": notebook_id} for i in range(len(chunks))]
            self.vector_store.add_documents(texts=chunks, metadatas=metadatas)

        logger.info(f"Processed '{filename}': {len(chunks)} chunks stored for notebook '{notebook_id}'.")
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "message": f"Successfully processed and stored {len(chunks)} chunks."
        }

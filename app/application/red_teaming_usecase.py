import os
import json
import re
import logging
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class RedTeamingUseCase:
    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        """
        Use case to retrieve the content of a document and perform fact-checking
        and bias analysis (red-teaming) on the text content.
        """
        self.vector_store = vector_store
        self.llm_service = llm_service

    def get_document_content(self, notebook_id: str, filename: str) -> dict:
        """
        Retrieves all text chunks of a document, sorts them by chunk_index,
        and returns the consolidated document content.
        """
        try:
            # Query ChromaDB for all chunks matching the filename and notebook_id
            results = self.vector_store.collection.get(
                where={"$and": [{"source": filename}, {"notebook_id": notebook_id}]},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("documents"):
                return {
                    "success": False,
                    "message": "Không tìm thấy nội dung tài liệu."
                }

            docs = results["documents"]
            metas = results["metadatas"] or []

            # Sort chunks by chunk_index
            sorted_chunks = []
            for doc, meta in zip(docs, metas):
                idx = meta.get("chunk_index", 0) if meta else 0
                sorted_chunks.append((idx, doc))

            sorted_chunks.sort(key=lambda x: x[0])
            full_text = "\n\n".join([chunk[1] for chunk in sorted_chunks])

            return {
                "success": True,
                "filename": filename,
                "content": full_text
            }
        except Exception as e:
            logger.error(f"Error getting document content: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Lỗi đọc nội dung tài liệu: {str(e)}"
            }

    def analyze(self, notebook_id: str, filename: str, content: str = None, provider: str = "auto") -> dict:
        """
        Performs red-teaming analysis on the document content using the LLM.
        """
        try:
            # 1. Load content if not provided
            if not content:
                res = self.get_document_content(notebook_id, filename)
                if not res.get("success"):
                    return res
                content = res.get("content")

            # Limit content size for LLM call
            analyze_text = content
            if len(analyze_text) > 15000:
                analyze_text = analyze_text[:15000]

            # 2. Formulate system and user prompts
            system_prompt = (
                "You are an expert fact-checker and critical red-teaming investigator. "
                "Your job is to read the provided text and identify factual errors or unsupported claims. "
                "Be objective, strict, and precise."
            )

            query_prompt = (
                f"Hãy đóng vai trò là một Thanh tra độc lập (Red-Teaming) để kiểm định sự thật và thiên kiến cho tài liệu sau.\n\n"
                f"NỘI DUNG TÀI LIỆU:\n\"\"\"\n{analyze_text}\n\"\"\"\n\n"
                f"NHIỆM VỤ:\n"
                f"Phân tích tài liệu và phát hiện:\n"
                f"1. **Factual Error (Lỗi sự thật)**: Các tuyên bố sai lệch thông tin, số liệu không khớp thực tế lịch sử/khoa học hoặc lỗi thời. Phân loại là 'factual_error'.\n"
                f"2. **Unsupported Claim (Nhận định chủ quan)**: Nhận định phiến diện của tác giả, suy diễn cá nhân mà hoàn toàn không có số liệu, dẫn chứng, trích dẫn hay lập luận logic đi kèm. Phân loại là 'unsupported_claim'.\n\n"
                f"Định dạng kết quả trả về là một mảng JSON chứa danh sách các điểm phát hiện, mỗi điểm gồm:\n"
                f"- 'text': Chuỗi ký tự (câu hoặc cụm từ) chính xác xuất hiện trong tài liệu gốc bị lỗi.\n"
                f"- 'category': Nhãn phân loại ('factual_error' hoặc 'unsupported_claim').\n"
                f"- 'explanation': Giải thích chi tiết bằng tiếng Việt lý do tại sao câu này bị lỗi, cung cấp lập luận phản bác hoặc thông tin đúng cần có.\n\n"
                f"Chỉ trả về chuỗi JSON thô dạng mảng, không bọc trong markdown block, không thêm bất kỳ văn bản giải thích nào khác."
            )

            # 3. Call LLM Service
            response_tokens = []
            stream = self.llm_service.generate_answer(
                context=analyze_text,
                query=query_prompt,
                system_prompt=system_prompt,
                provider=provider
            )
            for token in stream:
                response_tokens.append(token)

            raw_response = "".join(response_tokens).strip()

            # Clean markdown code blocks if any
            cleaned_response = raw_response
            if cleaned_response.startswith("```"):
                cleaned_response = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_response)
            if cleaned_response.endswith("```"):
                cleaned_response = re.sub(r"\n?```$", "", cleaned_response)
            cleaned_response = cleaned_response.strip()

            # Attempt to parse the JSON array
            try:
                parsed_findings = json.loads(cleaned_response)
                if not isinstance(parsed_findings, list):
                    parsed_findings = []
            except Exception:
                # Fallback extraction using regex
                match = re.search(r'\[[\s\S]*\]', cleaned_response)
                if match:
                    try:
                        parsed_findings = json.loads(match.group())
                    except Exception:
                        parsed_findings = []
                else:
                    parsed_findings = []

            return {
                "success": True,
                "filename": filename,
                "findings": parsed_findings
            }

        except Exception as e:
            logger.error(f"Error running red teaming analysis: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Không thể hoàn thành kiểm định tài liệu: {str(e)}"
            }

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

            if not content or not content.strip():
                return {
                    "success": True,
                    "filename": filename,
                    "findings": []
                }

            # Split into chunks of max 8000 characters with 1000 characters overlap
            chunk_size = 8000
            overlap = 1000
            chunks = []
            
            if len(content) <= chunk_size:
                chunks.append(content)
            else:
                start = 0
                while start < len(content):
                    end = start + chunk_size
                    chunks.append(content[start:end])
                    if end >= len(content):
                        break
                    start += chunk_size - overlap

            # Cap chunks to avoid excessive API calls (max 5 chunks)
            chunks = chunks[:5]

            all_findings = []
            seen_texts = set()

            for chunk_idx, analyze_text in enumerate(chunks):
                # 2. Formulate system and user prompts
                system_prompt = (
                    "Bạn là một chuyên gia kiểm chứng thông tin (Fact-Checker) và nhà khoa học kiểm định độc lập (Red-Teaming) cực kỳ cẩn trọng và chính xác. "
                    "Nhiệm vụ của bạn là kiểm tra văn bản để phát hiện SAI SÓT THỰC SỰ. "
                    "⚠️ QUY TẮC QUAN TRỌNG NHẤT (CHỐNG BÁO ĐỘNG GIẢ):\n"
                    "1. TẤT CẢ các kiến thức giáo trình khoa học, công thức toán học/xử lý ảnh chuẩn (như khoảng cách lưới √2, bộ lọc Gaussian, thuật toán Canny, Sobel, Laplacian, Chain Code, Splines...) ĐỀU LÀ ĐÚNG. KHÔNG ĐƯỢC đánh dấu chúng là 'Lỗi sự thật' hay 'Nhận định chủ quan'.\n"
                    "2. KHÔNG bắt lỗi các câu tóm tắt dạng slide bài giảng hay các quy ước thuật toán tiêu chuẩn.\n"
                    "3. CHỈ báo cáo khi có LỖI SAI SỰ THẬT HIỂN NHIÊN (ví dụ: công thức toán bị viết sai bản chất, số liệu sai thực tế) hoặc MÂU THUẪN TRỰC TIẾP ngay trong tài liệu.\n"
                    "4. Nếu đoạn văn bản hoàn toàn đúng hoặc là kiến thức tiêu chuẩn, hãy trả về mảng rỗng `[]`. Tuyệt đối KHÔNG cố bới vết tìm vết hay bịa ra lỗi."
                )

                query_prompt = (
                    f"Hãy kiểm định tài liệu sau một cách cẩn trọng, chính xác, tránh báo động giả.\n\n"
                    f"NỘI DUNG PHÂN ĐOẠN (Đoạn {chunk_idx + 1}/{len(chunks)}):\n\"\"\"\n{analyze_text}\n\"\"\"\n\n"
                    f"QUY TẮC PHÂN LOẠI CHI TIẾT:\n"
                    f"1. **Factual Error (Lỗi sự thật)**: CHỈ áp dụng cho các tuyên bố sai lệch sự thật nghiêm trọng, sai công thức toán học hiển nhiên. KHÔNG bắt lỗi các công thức hay định lý xử lý ảnh đúng.\n"
                    f"2. **Unsupported Claim (Nhận định chủ quan)**: CHỈ áp dụng cho các suy diễn cá nhân phiến diện của tác giả không có cơ sở. KHÔNG phân loại các quy ước hay mô tả thuật toán chuẩn là chủ quan.\n"
                    f"3. **Context Mismatch (Mâu thuẫn thông tin)**: CHỈ áp dụng khi 2 phát biểu trong cùng tài liệu đá nhau trực tiếp.\n\n"
                    f"Định dạng kết quả trả về BẮT BUỘC là một mảng JSON chứa danh sách các điểm phát hiện thực sự (nếu không có lỗi nào, trả về `[]`):\n"
                    f"[\n"
                    f"  {{\n"
                    f"    \"text\": \"Chuỗi ký tự chính xác bị lỗi trong tài liệu\",\n"
                    f"    \"category\": \"factual_error\" | \"unsupported_claim\" | \"context_mismatch\",\n"
                    f"    \"explanation\": \"Giải thích rõ ràng bằng tiếng Việt tại sao đây là lỗi thực sự\",\n"
                    f"    \"confidence\": 0.90\n"
                    f"  }}\n"
                    f"]\n\n"
                    f"Chỉ trả về duy nhất chuỗi JSON thô dạng mảng, không bọc trong markdown block, không thêm bất kỳ văn bản giải thích nào khác."
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
                parsed_findings = []
                try:
                    parsed_findings = json.loads(cleaned_response)
                except Exception:
                    # Fallback extraction using regex
                    match = re.search(r'\[[\s\S]*\]', cleaned_response)
                    if match:
                        try:
                            parsed_findings = json.loads(match.group())
                        except Exception:
                            pass

                if isinstance(parsed_findings, list):
                    for finding in parsed_findings:
                        if not isinstance(finding, dict) or "text" not in finding:
                            continue
                        
                        finding_text = finding["text"].strip()
                        if not finding_text:
                            continue
                        
                        # Basic validation & defaults
                        category = finding.get("category", "factual_error")
                        if category not in ["factual_error", "unsupported_claim", "context_mismatch"]:
                            category = "factual_error"
                        finding["category"] = category
                        
                        finding["explanation"] = finding.get("explanation", "Phát hiện điểm đáng ngờ trong câu này.")
                        try:
                            finding["confidence"] = round(float(finding.get("confidence", 0.7)), 2)
                        except Exception:
                            finding["confidence"] = 0.7
                        
                        # Deduplication check
                        text_lower = finding_text.lower()
                        if text_lower not in seen_texts:
                            seen_texts.add(text_lower)
                            all_findings.append(finding)

            return {
                "success": True,
                "filename": filename,
                "findings": all_findings
            }

        except Exception as e:
            logger.error(f"Error running red teaming analysis: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Không thể hoàn thành kiểm định tài liệu: {str(e)}"
            }

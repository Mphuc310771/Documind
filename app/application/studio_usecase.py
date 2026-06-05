import logging
import json
import re
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)

STUDIO_PROMPTS = {
    "flashcards": (
        "Dựa trên tài liệu bên dưới, hãy tạo 8–12 thẻ ghi nhớ (flashcard) để giúp người đọc ôn tập kiến thức.\n"
        "Mỗi thẻ gồm mặt trước (câu hỏi hoặc thuật ngữ) và mặt sau (câu trả lời hoặc định nghĩa).\n"
        "Trả về dạng mảng JSON: [{\"front\": \"...\", \"back\": \"...\"}]\n"
        "Chỉ trả về JSON, không thêm bất kỳ văn bản nào khác, không dùng markdown."
    ),
    "faq": (
        "Dựa trên nội dung tài liệu bên dưới, hãy tạo danh sách 6–10 câu hỏi thường gặp (FAQ) kèm câu trả lời chi tiết.\n"
        "Trả về dạng mảng JSON: [{\"question\": \"...\", \"answer\": \"...\"}]\n"
        "Chỉ trả về JSON, không thêm bất kỳ văn bản nào khác, không dùng markdown."
    ),
    "timeline": (
        "Dựa trên tài liệu bên dưới, hãy trích xuất và sắp xếp các sự kiện, mốc thời gian hoặc các bước quan trọng theo thứ tự logic/thời gian.\n"
        "Nếu tài liệu không có mốc thời gian rõ ràng, hãy tạo dòng thời gian dựa trên thứ tự các ý chính được trình bày.\n"
        "Trả về dạng mảng JSON: [{\"time\": \"...\", \"event\": \"...\", \"detail\": \"...\"}]\n"
        "Chỉ trả về JSON, không thêm bất kỳ văn bản nào khác, không dùng markdown."
    ),
    "study_guide": (
        "Bạn là một gia sư AI chuyên nghiệp. Dựa trên tài liệu bên dưới, hãy tạo một hướng dẫn học tập (Study Guide) "
        "chi tiết bằng tiếng Việt với cấu trúc rõ ràng:\n"
        "1. **Tổng quan chủ đề**: Mô tả ngắn gọn nội dung chính\n"
        "2. **Các khái niệm cốt lõi**: Liệt kê và giải thích từng khái niệm quan trọng\n"
        "3. **Mối liên hệ giữa các khái niệm**: Phân tích sự liên kết\n"
        "4. **Câu hỏi ôn tập**: 5-7 câu hỏi tự kiểm tra\n"
        "5. **Lưu ý quan trọng**: Những điểm dễ nhầm lẫn hoặc cần đặc biệt chú ý\n"
        "Trả kết quả bằng Markdown định dạng đẹp."
    ),
    "briefing": (
        "Bạn là trợ lý AI chuyên tạo báo cáo tóm tắt chuyên nghiệp. "
        "Dựa trên tài liệu bên dưới, hãy viết một bản báo cáo tóm tắt (Briefing Document) bằng tiếng Việt với cấu trúc:\n"
        "## Tiêu đề Báo cáo\n"
        "### Tóm tắt điều hành (Executive Summary)\n"
        "- Điểm chính 1...\n"
        "### Phân tích chi tiết\n"
        "...\n"
        "### Kết luận & Khuyến nghị\n"
        "...\n"
        "Trả kết quả bằng Markdown định dạng đẹp, chuyên nghiệp."
    ),
}


class StudioUseCase:
    """
    Generic Studio content generation use case.
    Handles flashcards, FAQ, timeline, study_guide, and briefing
    content types from uploaded documents.
    """

    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    def execute(self, content_type: str, provider: str = "auto", notebook_id: str = "default") -> dict:
        if content_type not in STUDIO_PROMPTS:
            return {
                "success": False,
                "content_type": content_type,
                "message": f"Loại nội dung '{content_type}' không được hỗ trợ."
            }

        # Retrieve document chunks filtered by notebook_id
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=25
        )
        if not results or not results.get("documents"):
            return {
                "success": False,
                "content_type": content_type,
                "message": "Chưa có tài liệu nào cho ghi chú này. Vui lòng tải tài liệu lên trước."
            }

        # Filter out screen captures
        documents = []
        metadatas = results.get("metadatas", []) or []
        for doc, meta in zip(results["documents"], metadatas):
            if meta and meta.get("source") == "screen_capture":
                continue
            documents.append(doc)
        if not documents:
            documents = results["documents"]

        context = "\n\n".join(documents)
        if len(context) > 18000:
            context = context[:18000]

        prompt = STUDIO_PROMPTS[content_type]

        try:
            response_tokens = []
            stream = self.llm_service.generate_answer(context=context, query=prompt, provider=provider)
            for token in stream:
                response_tokens.append(token)

            raw_response = "".join(response_tokens).strip()

            # For JSON content types, attempt to parse
            if content_type in ("flashcards", "faq", "timeline"):
                parsed = self._parse_json_response(raw_response)
                return {
                    "success": True,
                    "content_type": content_type,
                    "result": parsed,
                    "raw": raw_response
                }
            else:
                # Markdown content types (study_guide, briefing)
                return {
                    "success": True,
                    "content_type": content_type,
                    "result": raw_response
                }
        except Exception as e:
            logger.error(f"Error generating studio content '{content_type}': {e}", exc_info=True)
            return {
                "success": False,
                "content_type": content_type,
                "message": f"Không thể tạo nội dung: {str(e)}"
            }

    @staticmethod
    def _parse_json_response(raw: str) -> list:
        """Attempt to parse JSON from LLM response, with fallback."""
        cleaned = raw.strip()
        # Remove markdown code blocks
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON array from the response
        match = re.search(r'\[[\s\S]*\]', cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: return as single-item list
        return [{"content": raw}]

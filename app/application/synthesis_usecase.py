import logging
from app.domain.interfaces import ILLMService

logger = logging.getLogger(__name__)


class SynthesisUseCase:
    def __init__(self, llm_service: ILLMService):
        """
        Use case to synthesize, compare, and study-guide compile from notes selected by the user.
        """
        self.llm_service = llm_service

    def execute(self, notes: list, action: str, provider: str = "auto") -> dict:
        if not notes:
            return {
                "success": False,
                "result": "Danh sách ghi chú rỗng. Hãy chọn ít nhất một ghi chú để tổng hợp."
            }

        # Combine notes with index labeling
        context = "\n\n".join([f"Ghi chú {i+1}:\n{note}" for i, note in enumerate(notes)])

        action_prompts = {
            "study_guide": (
                "Hãy tạo một bộ Hướng dẫn Học tập (Study Guide) chi tiết, khoa học bằng tiếng Việt dựa trên các ghi chú trên.\n"
                "Cấu trúc bản hướng dẫn học tập gồm:\n"
                "1. Thuật ngữ & Khái niệm cốt lõi (Core Glossary)\n"
                "2. Các điểm kiến thức quan trọng nhất dưới dạng danh sách chi tiết\n"
                "3. Phần Hỏi & Đáp (FAQ) tự ôn tập dựa trên các ý tưởng này."
            ),
            "summary": (
                "Hãy phân tích và viết một bản Tóm tắt kết nối (Synthesis Summary) bằng tiếng Việt cho các ghi chú trên.\n"
                "Nhiệm vụ của bạn là kết nối các ý tưởng từ các ghi chú khác nhau thành một luồng nội dung logic, mạch lạc, chỉ ra mối quan hệ giữa chúng."
            ),
            "contradictions": (
                "Hãy đóng vai một chuyên gia tư duy phản biện.\n"
                "Phân tích các ghi chú trên và chỉ ra những điểm Mâu thuẫn (Contradictions), các kẽ hở lập luận (Gaps) hoặc sự thiếu nhất quán về mặt logic giữa các thông tin này.\n"
                "Nếu không có mâu thuẫn lớn nào, hãy nêu ra các điểm giả định cần kiểm chứng hoặc các câu hỏi phản biện thêm bằng tiếng Việt."
            ),
            "essay": (
                "Hãy viết một Bài luận học thuật (Academic Essay) hoàn chỉnh, sâu sắc bằng tiếng Việt kết hợp tất cả các ghi chú trên.\n"
                "Bài luận cần có Mở bài, Thân bài phân tích đa chiều và Kết luận rõ ràng. Sử dụng văn phong mạch lạc và chuyên nghiệp."
            )
        }

        prompt = action_prompts.get(action, action_prompts["summary"])

        try:
            response_tokens = []
            stream = self.llm_service.generate_answer(context=context, query=prompt, provider=provider)
            for token in stream:
                response_tokens.append(token)

            result = "".join(response_tokens).strip()
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error synthesizing notes: {e}", exc_info=True)
            return {
                "success": False,
                "result": f"Lỗi từ hệ thống AI khi tổng hợp ghi chú: {str(e)}"
            }

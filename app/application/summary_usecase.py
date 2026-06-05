import logging
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class DocumentSummaryUseCase:
    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        """
        Use case to generate bullet-point summaries of stored documents.
        """
        self.vector_store = vector_store
        self.llm_service = llm_service

    def execute(self, notebook_id: str = "default") -> dict:
        """
        Retrieves the first few chunks of stored text and prompts the LLM
        to compile a concise bullet-point summary.
        """
        # Retrieve the first few document chunks to extract context filtered by notebook_id
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=10
        )
        if not results or not results.get("documents"):
            return {
                "success": False,
                "summary": "Không có tài liệu nào cho ghi chú này. Vui lòng tải tài liệu lên trước."
            }

        # Combine text chunks
        context = "\n\n".join(results["documents"])

        prompt = (
            "Hãy viết một bản tóm tắt ngắn gọn dưới dạng danh sách các đầu dòng (bullet points) bằng tiếng Việt "
            "cho tài liệu được cung cấp dưới đây. Tập trung vào các ý quan trọng nhất."
        )

        response_tokens = []
        try:
            # Consume streaming LLM response
            stream = self.llm_service.generate_answer(context=context, query=prompt)
            for token in stream:
                response_tokens.append(token)
            
            summary = "".join(response_tokens).strip()
            return {
                "success": True,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            return {
                "success": False,
                "summary": f"Không thể tạo tóm tắt tài liệu từ AI: {str(e)}"
            }

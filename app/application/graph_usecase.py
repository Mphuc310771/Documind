import logging
import re
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class GraphUseCase:
    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        """
        Use case to generate Mermaid.js visual mindmaps from document contents.
        """
        self.vector_store = vector_store
        self.llm_service = llm_service

    def execute(self, notebook_id: str = "default", chat_context: str | None = None) -> dict:
        """
        Extracts key concepts from database and generates valid Mermaid graph code.
        """
        # Retrieve a selection of document chunks to extract relations filtered by notebook_id
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=15
        )
        
        doc_context = ""
        if results and results.get("documents"):
            doc_context = "\n\n".join(results["documents"])

        context_parts = []
        if chat_context:
            context_parts.append(f"Ngữ cảnh hội thoại trước đó:\n{chat_context}")
        if doc_context:
            context_parts.append(f"Ngữ cảnh tài liệu:\n{doc_context}")

        context = "\n\n".join(context_parts)
        if not context:
            return {
                "success": False,
                "graph": "graph TD\n    A[Chua co tai lieu] --> B[Vui long chat hoac tai len tai lieu truoc]"
            }

        prompt = (
            "Hãy phân tích ngữ cảnh được cung cấp và tạo ra một sơ đồ tư duy (mindmap) bằng tiếng Việt biểu diễn mối quan hệ giữa các khái niệm chính.\n"
            "Bạn phải trả về kết quả dưới dạng chuỗi mã nguồn Mermaid.js duy nhất hợp lệ.\n"
            "QUY TẮC CỰC KỲ QUAN TRỌNG: NEVER use double quotes, parentheses, or special characters inside node labels. "
            "Use ONLY simple alphanumeric text. Example: A[Context Window] --> B[Processing].\n"
            "Ví dụ:\n"
            "graph TD\n"
            "    A[Khái niệm chính] --> B[Ý phụ 1]\n"
            "    A --> C[Ý phụ 2]\n"
            "    B --> D[Chi tiết 1]\n"
            "Không bao gồm khối code markdown (ví dụ: ```mermaid), không thêm bất kỳ văn bản giải thích nào khác ngoài mã nguồn Mermaid."
        )

        response_tokens = []
        try:
            stream = self.llm_service.generate_answer(context=context, query=prompt)
            for token in stream:
                response_tokens.append(token)
            
            graph_code = "".join(response_tokens).strip()
            
            # Robust extraction stripping any markdown code block wrappers
            if "```mermaid" in graph_code:
                graph_code = graph_code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in graph_code:
                graph_code = graph_code.split("```")[1].split("```")[0].strip()

            # Remove double quotes and single quotes from the generated graph code
            lines = []
            for line in graph_code.split('\n'):
                # Strip out quote marks and parenthesis inside labels
                cleaned_line = line.replace('"', '').replace("'", "").replace('(', ' ').replace(')', ' ')
                lines.append(cleaned_line)
            graph_code = "\n".join(lines)
            
            # If the output doesn't start with graph, add it
            if not graph_code.strip().startswith("graph"):
                graph_code = "graph TD\n" + graph_code

            return {
                "success": True,
                "graph": graph_code
            }
        except Exception as e:
            logger.error(f"Error generating graph Mermaid source: {e}", exc_info=True)
            return {
                "success": False,
                "graph": f"graph TD\n    A[Loi AI] --> B[Loi thuc thi]"
            }

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
            "QUY TẮC BỐ CỤC: Sử dụng cấu trúc từ trái sang phải (graph LR) để tối ưu không gian màn hình ngang. "
            "Hãy xây dựng sơ đồ có chiều sâu phân cấp (Gốc -> Khái niệm con -> Chi tiết) thay vì dàn phẳng quá nhiều nút ở cùng một cấp.\n"
            "GIỚI HẠN SỐ LƯỢNG NÚT: Chỉ tạo tối đa từ 8 đến 12 nút khái niệm để sơ đồ rõ ràng, đẹp mắt và dễ đọc trên màn hình, tránh tình trạng quá dày đặc.\n"
            "QUY TẮC QUAN TRỌNG: NEVER use double quotes, parentheses, or special characters inside node labels. "
            "Use ONLY simple alphanumeric text. Example: A[Context Window] --> B[Processing].\n"
            "Ví dụ:\n"
            "graph LR\n"
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
            
            # Force layout direction to graph LR
            graph_code = graph_code.strip()
            if graph_code.startswith("graph TD"):
                graph_code = graph_code.replace("graph TD", "graph LR", 1)
            elif not graph_code.startswith("graph"):
                graph_code = "graph LR\n" + graph_code

            # Apply automatic class labels to nodes based on hierarchy
            try:
                graph_code = self.apply_hierarchical_classes(graph_code)
            except Exception as pe:
                logger.error(f"Error post-processing graph classes: {pe}", exc_info=True)

            return {
                "success": True,
                "graph": graph_code
            }
        except Exception as e:
            logger.error(f"Error generating graph Mermaid source: {e}", exc_info=True)
            return {
                "success": False,
                "graph": f"graph LR\n    A[Loi AI] --> B[Loi thuc thi]"
            }

    def apply_hierarchical_classes(self, graph_code: str) -> str:
        """
        Parses Mermaid flowchart code, determines the node depth/hierarchy,
        and appends style class assignments (rootNode, branchNode, leafNode)
        for custom premium CSS styling.
        """
        lines = graph_code.split('\n')
        
        # Regex to match node definitions, e.g. A[Label] or B(Label)
        node_def_pattern = re.compile(
            r'\b([a-zA-Z0-9_-]+)\s*(?:\[[^\]]+\]|\([^)]+\)|\{\{[^}]+\}\}|\(\([^)]+\)\)|\{\[^}]+\}|\["[^"]+"\]|\("[^"]+"\))'
        )
        # Regex to match connections, e.g. A --> B or A --- B after label cleaning
        connection_pattern = re.compile(
            r'\b([a-zA-Z0-9_-]+)\s*(?:-->|---|==>|-.->)\s*([a-zA-Z0-9_-]+)\b'
        )
        
        nodes = set()
        parent_map = {}  # child -> set of parents
        child_map = {}   # parent -> set of children
        
        for line in lines:
            if "classDef" in line or "class " in line:
                continue
                
            # Find all nodes defined in the line
            for match in node_def_pattern.finditer(line):
                nodes.add(match.group(1))
            
            # Remove label brackets to simplify connection parsing
            cleaned_line = re.sub(
                r'(?:\[[^\]]+\]|\([^)]+\)|\{\{[^}]+\}\}|\(\([^)]+\)\)|\{\[^}]+\}|\["[^"]+"\]|\("[^"]+"\))',
                '',
                line
            )
            
            # Find all connections in the cleaned line
            for match in connection_pattern.finditer(cleaned_line):
                parent = match.group(1)
                child = match.group(2)
                nodes.add(parent)
                nodes.add(child)
                
                if parent not in child_map:
                    child_map[parent] = set()
                child_map[parent].add(child)
                
                if child not in parent_map:
                    parent_map[child] = set()
                parent_map[child].add(parent)
                
        if not nodes:
            return graph_code
            
        roots = []
        branches = []
        leaves = []
        
        for node in nodes:
            has_parents = node in parent_map and len(parent_map[node]) > 0
            has_children = node in child_map and len(child_map[node]) > 0
            
            if has_children and not has_parents:
                roots.append(node)
            elif has_parents and has_children:
                branches.append(node)
            elif has_parents and not has_children:
                leaves.append(node)
            else:
                # Isolated node
                roots.append(node)
                
        # Fallback: if no root is found (e.g. cyclic graph), select the node with the most children
        if not roots:
            best_root = max(nodes, key=lambda n: len(child_map.get(n, [])))
            roots.append(best_root)
            if best_root in branches:
                branches.remove(best_root)
            if best_root in leaves:
                leaves.remove(best_root)
                
        # Build clean flowchart lines (filtering out old class/style rules)
        cleaned_lines = []
        for line in lines:
            if "classDef" in line or "class " in line:
                continue
            cleaned_lines.append(line)
            
        # Append class assignments
        class_lines = []
        if roots:
            class_lines.append(f"    class {','.join(roots)} rootNode;")
        if branches:
            class_lines.append(f"    class {','.join(branches)} branchNode;")
        if leaves:
            class_lines.append(f"    class {','.join(leaves)} leafNode;")
            
        return "\n".join(cleaned_lines) + "\n" + "\n".join(class_lines)

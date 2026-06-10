import json
import re
import logging

from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)

VALID_GROUPS = {
    "concept", "person", "organization", "technology",
    "event", "place", "document", "period", "other",
}


class KnowledgeGraphUseCase:
    """
    GraphRAG: extract entities + relationships from notebook documents
    for an interactive knowledge map (vis-network on the frontend).
    """

    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    def execute(self, notebook_id: str = "default", chat_context: str | None = None) -> dict:
        results = self.vector_store.collection.get(where={"notebook_id": notebook_id}, limit=35)

        documents = []
        metadatas = results.get("metadatas", []) or []
        for doc, meta in zip(results.get("documents", []) or [], metadatas):
            if meta and meta.get("source") == "screen_capture":
                continue
            documents.append(doc)
        if not documents:
            documents = results.get("documents", []) or []

        doc_context = "\n\n".join(documents)
        if len(doc_context) > 22000:
            doc_context = doc_context[:22000]

        context_parts = []
        if chat_context:
            context_parts.append(f"Ngữ cảnh hội thoại:\n{chat_context}")
        if doc_context:
            context_parts.append(f"Ngữ cảnh tài liệu:\n{doc_context}")
        context = "\n\n".join(context_parts)

        if not context.strip():
            return {"success": False, "message": "Chưa có tài liệu để dựng đồ thị tri thức."}

        prompt = (
            "Bạn là hệ thống trích xuất đồ thị tri thức (GraphRAG). Đọc ngữ cảnh và trích xuất thực thể + quan hệ "
            "CHI TIẾT, có chiều sâu (không chỉ vài nút trung tâm).\n"
            "Trả về STRICT JSON duy nhất (không markdown, không giải thích):\n"
            '{"nodes":[{"id":"n1","label":"Tên","group":"person","description":"Mô tả 1-2 câu","importance":4}],'
            '"edges":[{"from":"n1","to":"n2","label":"quan hệ ngắn","detail":"giải thích thêm"}]}\n'
            "Quy tắc:\n"
            "- 28 đến 50 nodes. Bao phủ chủ đề chính, sự kiện con, nhân vật phụ, địa danh, văn bản, khái niệm, mốc thời gian.\n"
            "- Ngôn ngữ label/description giống nguồn (tiếng Việt nếu tài liệu tiếng Việt).\n"
            "- group: concept | person | organization | technology | event | place | document | period | other\n"
            "- description: BẮT BUỘC 2-4 câu tiếng Việt (40-220 ký tự), giải thích vai trò, thời gian, ý nghĩa lịch sử.\n"
            "- importance: số nguyên 1-5 (5 = trung tâm nhất).\n"
            "- Ít nhất 35 edges; label edge là cụm động từ; detail BẮT BUỘC 1 câu giải thích quan hệ.\n"
            "- Mọi edge from/to phải trùng id node. id ngắn, duy nhất (n1, n2, ...).\n"
            "- Nối các thực thể phụ với nhau, không chỉ nối vào một nút trung tâm."
        )

        try:
            tokens = list(self.llm_service.generate_answer(context=context, query=prompt))
            raw = "".join(tokens).strip()
            graph = self._parse_graph(raw)
            if not graph["nodes"]:
                return {"success": False, "message": "AI không trích xuất được thực thể nào từ tài liệu."}
            graph = self._enrich_graph(graph)
            return {"success": True, "graph": graph}
        except Exception as e:
            logger.error(f"KnowledgeGraph extraction failed: {e}", exc_info=True)
            return {"success": False, "message": f"Lỗi dựng đồ thị tri thức: {str(e)}"}

    @staticmethod
    def _parse_graph(raw: str) -> dict:
        text = raw.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]

        data = json.loads(text)
        raw_nodes = data.get("nodes", []) or []
        raw_edges = data.get("edges", []) or []

        nodes = []
        seen_ids = set()
        for n in raw_nodes:
            nid = str(n.get("id") or n.get("label") or "").strip()
            label = str(n.get("label") or n.get("id") or "").strip()
            if not nid or nid in seen_ids or not label:
                continue
            seen_ids.add(nid)
            group = str(n.get("group") or "other").strip().lower()
            if group not in VALID_GROUPS:
                group = "other"
            try:
                importance = int(n.get("importance") or 3)
            except (TypeError, ValueError):
                importance = 3
            importance = max(1, min(5, importance))
            nodes.append({
                "id": nid,
                "label": label,
                "group": group,
                "description": str(n.get("description") or "").strip(),
                "importance": importance,
            })

        edges = []
        seen_edge = set()
        for e in raw_edges:
            src = str(e.get("from") or e.get("source") or "").strip()
            dst = str(e.get("to") or e.get("target") or "").strip()
            if src not in seen_ids or dst not in seen_ids or src == dst:
                continue
            key = (src, dst, str(e.get("label") or ""))
            if key in seen_edge:
                continue
            seen_edge.add(key)
            edges.append({
                "from": src,
                "to": dst,
                "label": str(e.get("label") or "").strip(),
                "detail": str(e.get("detail") or "").strip(),
            })

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _enrich_graph(graph: dict) -> dict:
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []
        by_id = {n["id"]: n for n in nodes}

        for node in nodes:
            desc = (node.get("description") or "").strip()
            if len(desc) >= 40:
                continue
            snippets = []
            for edge in edges:
                if edge["from"] == node["id"]:
                    other = by_id.get(edge["to"], {})
                    label = edge.get("label") or "liên quan"
                    detail = (edge.get("detail") or "").strip()
                    other_name = other.get("label") or edge["to"]
                    line = f"{node['label']} {label} {other_name}"
                    if detail:
                        line += f" ({detail})"
                    snippets.append(line + ".")
                elif edge["to"] == node["id"]:
                    other = by_id.get(edge["from"], {})
                    label = edge.get("label") or "liên quan"
                    detail = (edge.get("detail") or "").strip()
                    other_name = other.get("label") or edge["from"]
                    line = f"{other_name} {label} {node['label']}"
                    if detail:
                        line += f" ({detail})"
                    snippets.append(line + ".")
            if snippets:
                node["description"] = " ".join(snippets[:5])
            elif desc:
                node["description"] = desc

        return graph

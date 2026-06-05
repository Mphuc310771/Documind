import json
import re
import logging

from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class KnowledgeGraphUseCase:
    """
    GraphRAG: use the LLM to extract Entities and Relationships from the
    notebook's documents and return a structured node/edge graph for an
    interactive, zoomable knowledge map (vis-network on the frontend).
    """

    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    def execute(self, notebook_id: str = "default", chat_context: str | None = None) -> dict:
        results = self.vector_store.collection.get(where={"notebook_id": notebook_id}, limit=20)

        documents = []
        metadatas = results.get("metadatas", []) or []
        for doc, meta in zip(results.get("documents", []) or [], metadatas):
            if meta and meta.get("source") == "screen_capture":
                continue
            documents.append(doc)
        if not documents:
            documents = results.get("documents", []) or []

        doc_context = "\n\n".join(documents)
        if len(doc_context) > 14000:
            doc_context = doc_context[:14000]

        context_parts = []
        if chat_context:
            context_parts.append(f"Ngữ cảnh hội thoại:\n{chat_context}")
        if doc_context:
            context_parts.append(f"Ngữ cảnh tài liệu:\n{doc_context}")
        context = "\n\n".join(context_parts)

        if not context.strip():
            return {"success": False, "message": "Chưa có tài liệu để dựng đồ thị tri thức."}

        prompt = (
            "You are a knowledge graph extraction engine. Read the provided context and extract the most "
            "important entities and the relationships between them.\n"
            "Return STRICT JSON only (no markdown, no code fences, no commentary) with this exact shape:\n"
            '{"nodes":[{"id":"n1","label":"Tên thực thể","group":"concept"}],'
            '"edges":[{"from":"n1","to":"n2","label":"quan hệ"}]}\n'
            "Rules:\n"
            "- 8 to 22 nodes. Labels in the SAME language as the source (Vietnamese if source is Vietnamese).\n"
            "- 'group' is a short category: concept | person | organization | technology | event | place | other.\n"
            "- Every edge 'from'/'to' MUST reference an existing node id.\n"
            "- 'label' on edges is a short verb phrase describing the relationship.\n"
            "- Keep ids short and unique (n1, n2, ...)."
        )

        try:
            tokens = list(self.llm_service.generate_answer(context=context, query=prompt))
            raw = "".join(tokens).strip()
            graph = self._parse_graph(raw)
            if not graph["nodes"]:
                return {"success": False, "message": "AI không trích xuất được thực thể nào từ tài liệu."}
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
        # Fallback: slice from first { to last }
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

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
            nodes.append({"id": nid, "label": label, "group": str(n.get("group") or "other").strip().lower()})

        edges = []
        for e in raw_edges:
            src = str(e.get("from") or e.get("source") or "").strip()
            dst = str(e.get("to") or e.get("target") or "").strip()
            if src in seen_ids and dst in seen_ids and src != dst:
                edges.append({"from": src, "to": dst, "label": str(e.get("label") or "").strip()})

        return {"nodes": nodes, "edges": edges}

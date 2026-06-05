import json
import logging
import re
from typing import Any

from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class SlideGeneratorUseCase:
    ALLOWED_LAYOUTS = [
        "COVER",
        "EXECUTIVE_SUMMARY",
        "ARCHITECTURE_TWO_COLUMN",
        "DATA_METRICS",
        "CONCLUSION",
    ]
    MIDDLE_SEQUENCE = [
        "EXECUTIVE_SUMMARY",
        "DATA_METRICS",
        "ARCHITECTURE_TWO_COLUMN",
        "EXECUTIVE_SUMMARY",
        "DATA_METRICS",
        "ARCHITECTURE_TWO_COLUMN",
    ]

    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    def execute(
        self,
        provider: str = "auto",
        notebook_id: str = "default",
        num_slides: int = 10,
        chat_context: str | None = None,
    ) -> dict:
        target_count = max(3, min(int(num_slides or 10), 25))
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=15,
        )

        doc_context = ""
        if results and results.get("documents"):
            doc_context = "\n\n".join(results["documents"])

        context_parts = []
        if chat_context:
            context_parts.append(f"CURRENT CHAT CONVERSATION HISTORY:\n{chat_context}")
        if doc_context:
            context_parts.append(f"UPLOADED DOCUMENT CONTEXT:\n{doc_context}")

        context = "\n\n".join(context_parts)
        if not context:
            return {
                "success": False,
                "message": "No document or chat context is available for slide generation.",
            }

        if len(context) > 15000:
            context = context[:15000]

        system_prompt = self._build_system_prompt(target_count)

        try:
            response_tokens = []
            stream = self.llm_service.generate_answer(context=context, query=system_prompt, provider=provider)
            for token in stream:
                response_tokens.append(token)

            raw_response = "".join(response_tokens).strip()
            cleaned = self._clean_json_response(raw_response)
            slides = self._parse_slides(cleaned, raw_response)
            slides = self._normalize_deck(slides, context, target_count)

            return {
                "success": True,
                "slides": slides,
            }
        except Exception as e:
            logger.error(f"Error generating slides: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Could not generate presentation slides: {str(e)}",
            }

    def _build_system_prompt(self, num_slides: int) -> str:
        layouts = json.dumps(self.ALLOWED_LAYOUTS)
        recommended_sequence = self._recommended_sequence(num_slides)
        return (
            "You are a principal enterprise presentation strategist and SaaS pitch-deck writer.\n"
            "Transform the supplied context into a polished, boardroom-ready presentation.\n\n"
            "Return ONLY a valid JSON array. Do not include markdown fences, comments, or explanatory text.\n"
            f"The JSON array MUST contain exactly {num_slides} slide objects. This count is mandatory.\n"
            f"Use ONLY these layout values: {layouts}\n"
            f"Recommended layout sequence for this deck: {json.dumps(recommended_sequence)}\n\n"
            "Each slide object MUST contain exactly these keys:\n"
            "- layout: one of COVER, EXECUTIVE_SUMMARY, ARCHITECTURE_TWO_COLUMN, DATA_METRICS, CONCLUSION.\n"
            "- title: concise professional headline, maximum 8 words.\n"
            "- kicker: short uppercase section label, maximum 4 words.\n"
            "- content: array of concise bullet strings. No paragraphs. No walls of text.\n"
            "- visual_prompt: concrete visual direction for a modern dark enterprise SaaS deck.\n\n"
            "Slide content rules:\n"
            "- Slide 1 must use COVER. The final slide must use CONCLUSION.\n"
            "- Use EXECUTIVE_SUMMARY, ARCHITECTURE_TWO_COLUMN, and DATA_METRICS repeatedly as middle-slide templates when more than 5 slides are requested.\n"
            "- COVER: 2-3 strategic subtitle bullets.\n"
            "- EXECUTIVE_SUMMARY: exactly 4 bullets, each maximum 12 words.\n"
            "- ARCHITECTURE_TWO_COLUMN: exactly 2 bullets formatted as 'Left label: insight' and 'Right label: insight'.\n"
            "- DATA_METRICS: exactly 4 bullets formatted as 'Metric label: VALUE | short implication', where VALUE "
            "is a SHORT quantitative token only (e.g. '3x', '95%', '40%', '5', '24/7') — never a sentence. "
            "Prefer real numbers from the context; the value is shown as a large headline number.\n"
            "- CONCLUSION: exactly 3 action-oriented bullets, each maximum 12 words.\n"
            "- Write in the same language as the strongest user/document context.\n"
            "- Use concrete claims grounded in the context. Avoid filler and generic advice.\n\n"
            "Output shape example:\n"
            "[\n"
            "  {\n"
            "    \"layout\": \"COVER\",\n"
            "    \"title\": \"AI Knowledge Platform Strategy\",\n"
            "    \"kicker\": \"BOARD BRIEF\",\n"
            "    \"content\": [\"Unified retrieval for trusted answers\", \"Enterprise workflow acceleration\"],\n"
            "    \"visual_prompt\": \"Dark SaaS command center with luminous knowledge graph and precise data lines\"\n"
            "  }\n"
            "]"
        )

    @staticmethod
    def _clean_json_response(raw_response: str) -> str:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = re.sub(r"\n?```$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _parse_slides(cleaned: str, raw_response: str) -> list[dict[str, Any]]:
        try:
            slides = json.loads(cleaned)
            if isinstance(slides, list):
                return slides
            raise ValueError("Parsed JSON is not a list")
        except Exception as e:
            logger.warning(f"Failed to parse strict slide JSON: {e}. Attempting regex fallback.")

        slides = []
        slide_blocks = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw_response)
        for block in slide_blocks:
            try:
                slide_data = json.loads(block)
                if isinstance(slide_data, dict):
                    slides.append(slide_data)
            except Exception:
                continue
        return slides

    def _normalize_deck(self, slides: list[dict[str, Any]], context: str, target_count: int) -> list[dict[str, Any]]:
        normalized = []
        valid_slides = [slide for slide in slides if isinstance(slide, dict)]

        for index in range(target_count):
            source = valid_slides[index] if index < len(valid_slides) else {}
            layout = self._layout_for_slide(source, index, target_count)
            normalized.append(self._normalize_slide(source, layout, index, context, target_count))
        return normalized

    def _layout_for_slide(self, slide: dict[str, Any], index: int, total: int) -> str:
        if index == 0:
            return "COVER"
        if index == total - 1:
            return "CONCLUSION"

        layout = str(slide.get("layout", "")).strip().upper()
        if layout in self.ALLOWED_LAYOUTS and layout not in {"COVER", "CONCLUSION"}:
            return layout
        return self.MIDDLE_SEQUENCE[(index - 1) % len(self.MIDDLE_SEQUENCE)]

    @classmethod
    def _recommended_sequence(cls, total: int) -> list[str]:
        sequence = []
        for index in range(total):
            if index == 0:
                sequence.append("COVER")
            elif index == total - 1:
                sequence.append("CONCLUSION")
            else:
                sequence.append(cls.MIDDLE_SEQUENCE[(index - 1) % len(cls.MIDDLE_SEQUENCE)])
        return sequence

    def _normalize_slide(self, slide: dict[str, Any], layout: str, index: int, context: str, total: int) -> dict[str, Any]:
        fallback = self._fallback_slide(layout, index, context, total)
        title = self._short_text(slide.get("title"), fallback["title"], max_words=8)
        kicker = self._short_text(slide.get("kicker"), fallback["kicker"], max_words=4).upper()
        content = self._coerce_content(slide.get("content"), fallback["content"], layout)
        if layout == "DATA_METRICS":
            content = self._sanitize_metric_bullets(content)
        visual_prompt = self._short_text(slide.get("visual_prompt"), fallback["visual_prompt"], max_words=22)

        return {
            "layout": layout,
            "title": title,
            "kicker": kicker,
            "content": content,
            "visual_prompt": visual_prompt,
        }

    def _coerce_content(self, value: Any, fallback: list[str], layout: str) -> list[str]:
        if isinstance(value, list):
            items = [self._short_text(item, "", max_words=16) for item in value]
        elif value is None:
            items = []
        else:
            items = [self._short_text(value, "", max_words=16)]

        items = [item for item in items if item]
        required_counts = {
            "COVER": 3,
            "EXECUTIVE_SUMMARY": 4,
            "ARCHITECTURE_TWO_COLUMN": 2,
            "DATA_METRICS": 4,
            "CONCLUSION": 3,
        }
        required = required_counts[layout]

        merged = (items + fallback)[:required]
        while len(merged) < required:
            merged.append(fallback[len(merged) % len(fallback)])
        return merged

    @staticmethod
    def _clamp_metric_token(value: str) -> str:
        """Keep metric headline short so the slide renderer can show large numbers."""
        text = re.sub(r"\s+", " ", (value or "").strip())
        if not text:
            return "—"
        match = re.search(
            r"(\d+(?:[.,]\d+)?\s*%|\d+(?:[.,]\d+)?\s*x|24/7|low|medium|high|cao|thấp|trung bình)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        words = text.split()
        if len(words) <= 3:
            return text[:20]
        return words[0][:16]

    def _sanitize_metric_bullets(self, content: list[str]) -> list[str]:
        cleaned: list[str] = []
        for raw in content:
            line = str(raw or "").strip()
            if ":" not in line:
                cleaned.append(line)
                continue
            label, rest = line.split(":", 1)
            label = label.strip()
            if "|" in rest:
                value_part, note_part = rest.split("|", 1)
                value_part = self._clamp_metric_token(value_part.strip())
                cleaned.append(f"{label}: {value_part} | {note_part.strip()}")
            else:
                value_part = self._clamp_metric_token(rest.strip())
                cleaned.append(f"{label}: {value_part}")
        return cleaned

    @staticmethod
    def _short_text(value: Any, fallback: str, max_words: int) -> str:
        text = str(value if value is not None else fallback).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            text = fallback
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]).rstrip(".,;:") + "..."
        return text

    def _fallback_slide(self, layout: str, index: int, context: str, total: int) -> dict[str, Any]:
        topic = self._derive_topic(context)
        section_no = index + 1
        middle_variants = [
            {
                "title": "Strategic Context",
                "kicker": f"SECTION {section_no:02d}",
                "content": [
                    "Core opportunity is clear and actionable",
                    "Stakeholders need faster synthesis cycles",
                    "Current knowledge flow remains fragmented",
                    "Decision quality improves with grounded retrieval",
                ],
            },
            {
                "title": "Value Signals",
                "kicker": f"SECTION {section_no:02d}",
                "content": [
                    "Speed: 3x | Faster executive preparation",
                    "Quality: 95% | Higher answer confidence",
                    "Coverage: 5 flows | Broader workflow activation",
                    "Risk: Low | Controlled validation path",
                ],
            },
            {
                "title": "System Blueprint",
                "kicker": f"SECTION {section_no:02d}",
                "content": [
                    "Input layer: Uploaded documents and conversation context",
                    "Decision layer: Curated retrieval transforms evidence into answers",
                ],
            },
            {
                "title": "Operating Priorities",
                "kicker": f"SECTION {section_no:02d}",
                "content": [
                    "Prioritize workflows with repeated information demand",
                    "Measure answer quality before scaling usage",
                    "Keep source traceability visible to users",
                    "Automate only after confidence is proven",
                ],
            },
            {
                "title": "Adoption Economics",
                "kicker": f"SECTION {section_no:02d}",
                "content": [
                    "Time saved: 40% | Less manual synthesis",
                    "Reuse: 2x | More leverage from uploaded assets",
                    "Accuracy: High | Grounded response generation",
                    "Payback: Fast | Immediate workflow compression",
                ],
            },
            {
                "title": "Implementation Model",
                "kicker": f"SECTION {section_no:02d}",
                "content": [
                    "Pilot scope: Start with high-value document workflows",
                    "Scale path: Expand after measurable quality thresholds",
                ],
            },
        ]
        variant = middle_variants[(max(index, 1) - 1) % len(middle_variants)]
        fallbacks = {
            "COVER": {
                "title": topic,
                "kicker": "PITCH DECK",
                "content": [
                    "Strategic narrative from uploaded knowledge",
                    "Executive-ready synthesis and recommendation",
                    "Designed for fast decision alignment",
                ],
                "visual_prompt": "Dark enterprise SaaS hero with luminous data network and glass panels",
            },
            "EXECUTIVE_SUMMARY": {
                "title": variant["title"],
                "kicker": variant["kicker"],
                "content": variant["content"],
                "visual_prompt": "Four executive insight cards on dark glass dashboard with lime accents",
            },
            "ARCHITECTURE_TWO_COLUMN": {
                "title": variant["title"],
                "kicker": variant["kicker"],
                "content": variant["content"][:2],
                "visual_prompt": "Two-column technical architecture with connected neon nodes and data paths",
            },
            "DATA_METRICS": {
                "title": variant["title"],
                "kicker": variant["kicker"],
                "content": variant["content"],
                "visual_prompt": "KPI wall with cyber-lime numbers and electric blue trend lines",
            },
            "CONCLUSION": {
                "title": "Recommended Path Forward",
                "kicker": f"SLIDE {total:02d}",
                "content": [
                    "Approve focused pilot scope",
                    "Instrument quality and adoption metrics",
                    "Scale after evidence-based validation",
                ],
                "visual_prompt": "Executive conclusion slide with luminous roadmap path and decisive focal point",
            },
        }
        return fallbacks.get(layout, fallbacks["EXECUTIVE_SUMMARY"])

    @staticmethod
    def _derive_topic(context: str) -> str:
        first_line = next((line.strip() for line in context.splitlines() if len(line.strip()) > 12), "")
        if not first_line:
            return "Strategic Knowledge Platform"
        cleaned = re.sub(r"^(CURRENT CHAT CONVERSATION HISTORY:|UPLOADED DOCUMENT CONTEXT:)", "", first_line).strip()
        words = cleaned.split()
        return " ".join(words[:8]).rstrip(".,;:") or "Strategic Knowledge Platform"

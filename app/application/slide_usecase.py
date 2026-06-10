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
            f"Use ONLY these layout values: {layouts}\n\n"
            "Choose the most appropriate layout for each slide based on the content it represents. For example, use DATA_METRICS when showing statistics or numbers, ARCHITECTURE_TWO_COLUMN when describing technical components or flows, and EXECUTIVE_SUMMARY for concepts or summaries. Do not just blindly repeat a fixed layout sequence; select layouts dynamically to create a well-structured narrative.\n\n"
            "Each slide object MUST contain exactly these keys:\n"
            "- layout: one of COVER, EXECUTIVE_SUMMARY, ARCHITECTURE_TWO_COLUMN, DATA_METRICS, CONCLUSION.\n"
            "- title: concise professional headline, maximum 8 words.\n"
            "- kicker: short uppercase section label, maximum 4 words.\n"
            "- content: array of concise bullet strings. No paragraphs. No walls of text.\n"
            "- visual_prompt: concrete visual direction for a modern dark enterprise SaaS deck.\n\n"
            "Slide content rules:\n"
            "- Slide 1 must use COVER. The final slide must use CONCLUSION.\n"
            "- COVER: 2-3 strategic subtitle bullets.\n"
            "- EXECUTIVE_SUMMARY: 2 to 4 key insight bullets (each maximum 12 words) summarizing main concepts. Be concise and content-focused.\n"
            "- ARCHITECTURE_TWO_COLUMN: 2 to 3 architectural component bullets formatted as 'Left label: insight or Component: description' representing system flow. Capped at 3 components.\n"
            "- DATA_METRICS: 2 to 4 metric bullets formatted as 'Metric label: VALUE | short implication', where VALUE "
            "is a SHORT quantitative token (e.g. '3x', '95%', '40%', '5', '24/7') — never a sentence. Prefer real numbers from the context. Capped at 4 metrics.\n"
            "- CONCLUSION: 2 to 3 action-oriented path-forward bullets, each maximum 12 words.\n"
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
        start_idx = cleaned.find('[')
        end_idx = cleaned.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx+1]
        else:
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
        max_counts = {
            "COVER": 4,
            "EXECUTIVE_SUMMARY": 6,
            "ARCHITECTURE_TWO_COLUMN": 4,
            "DATA_METRICS": 6,
            "CONCLUSION": 4,
        }
        limit = max_counts.get(layout, 4)

        if not items:
            return fallback[:limit]
        return items[:limit]

    @staticmethod
    def _clamp_metric_token(value: str) -> str:
        """Keep metric headline short so the slide renderer can show large numbers."""
        text = re.sub(r"\s+", " ", (value or "").strip())
        if not text:
            return "—"
        
        # Try to find a numeric value with its optional suffix/units or key labels
        match = re.search(
            r"(24/7|\d+(?:[.,]\d+)?\s*(?:%|x|tỷ|triệu|k|M|B|USD|đ|fps|req/s|s|trang)?(?:\s+đồng|\s+USD)?|low|medium|high|cao|thấp|trung bình)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
            
        words = text.split()
        if len(words) <= 3:
            return text[:20]
        return " ".join(words[:2])[:16]

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
        # Try to build a dynamic fallback from context if available
        try:
            if context and len(context.strip()) > 50:
                cleaned_context = re.sub(
                    r"^(CURRENT CHAT CONVERSATION HISTORY:|UPLOADED DOCUMENT CONTEXT:)", 
                    "", 
                    context, 
                    flags=re.IGNORECASE
                ).strip()
                
                # Split into non-empty lines
                lines = [line.strip() for line in cleaned_context.split("\n") if len(line.strip()) > 15]
                
                if len(lines) >= 3:
                    section_no = index + 1
                    kicker = f"NỘI DUNG {section_no:02d}"
                    
                    # Select lines based on index to ensure slides are different
                    line_idx = (index * 2) % len(lines)
                    content_lines = []
                    for i in range(4):
                        idx = (line_idx + i) % len(lines)
                        content_lines.append(lines[idx])
                        
                    # Derive a slide title from the selected line
                    title_line = content_lines[0]
                    words = title_line.split()
                    title = " ".join(words[:6]).rstrip(".,;:-")
                    if len(title) < 8:
                        title = f"Phân tích chuyên sâu {section_no:02d}"
                        
                    if layout == "COVER":
                        topic = self._derive_topic(context)
                        return {
                            "title": topic,
                            "kicker": "BÁO CÁO",
                            "content": [line[:60] for line in content_lines[:3]],
                            "visual_prompt": "Dark enterprise SaaS hero with luminous data network",
                        }
                    elif layout == "CONCLUSION":
                        return {
                            "title": "Kết luận & Định hướng",
                            "kicker": f"SLIDE {total:02d}",
                            "content": [line[:60] for line in content_lines[:3]],
                            "visual_prompt": "Executive conclusion slide with luminous roadmap path",
                        }
                    elif layout == "DATA_METRICS":
                        metric_bullets = []
                        for idx, line in enumerate(content_lines[:4]):
                            # Look for any numbers
                            num_match = re.search(r'(\d+(?:[.,]\d+)?\s*%?|\d+\s*x)', line)
                            if num_match:
                                val = num_match.group(1)
                                label = line.replace(val, "").replace(":", "").strip()
                                lbl_words = label.split()
                                lbl = " ".join(lbl_words[:2]) or "Chỉ số"
                                metric_bullets.append(f"{lbl}: {val} | {line[:60]}")
                            else:
                                lbl_words = line.split()
                                lbl = " ".join(lbl_words[:2]) or "Thông tin"
                                metric_bullets.append(f"{lbl}: 0{idx+1} | {line[:60]}")
                        return {
                            "title": title,
                            "kicker": kicker,
                            "content": metric_bullets,
                            "visual_prompt": "KPI wall with cyber-lime numbers and trend lines",
                        }
                    elif layout == "ARCHITECTURE_TWO_COLUMN":
                        arch_bullets = []
                        for line in content_lines[:2]:
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                arch_bullets.append(f"{parts[0][:15]}: {parts[1][:60]}")
                            else:
                                lbl_words = line.split()
                                label = " ".join(lbl_words[:2]) or "Thành phần"
                                desc = " ".join(lbl_words[2:10]) or line[:50]
                                arch_bullets.append(f"{label}: {desc}")
                        return {
                            "title": title,
                            "kicker": kicker,
                            "content": arch_bullets,
                            "visual_prompt": "Two-column technical architecture layout",
                        }
                    else: # EXECUTIVE_SUMMARY
                        return {
                            "title": title,
                            "kicker": kicker,
                            "content": [line[:70] for line in content_lines[:4]],
                            "visual_prompt": "Four executive insight cards on dark glass dashboard",
                        }
        except Exception as e:
            logger.warning(f"Failed to generate dynamic fallback slide: {e}. Falling back to static templates.")

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

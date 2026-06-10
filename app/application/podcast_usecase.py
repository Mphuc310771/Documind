import logging
import json
import re
import os
import uuid
import asyncio
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore
from app.infrastructure.tts_adapter import clean_tts_text, host_to_voice_key, synthesize_to_file, sanitize_podcast_dialogue

logger = logging.getLogger(__name__)

TURN_DELAY_SEC = 0.35


class AudioBriefingUseCase:
    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        """
        Use case to generate a 2-host audio briefing podcast script from documents and synthesize it into a single MP3 file.
        """
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def execute(self, provider: str = "auto", notebook_id: str = "default", custom_instructions: str = "") -> dict:
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=15
        )
        if not results or not results.get("documents"):
            return {
                "success": False,
                "message": "Không có tài liệu nào cho ghi chú này. Vui lòng tải tài liệu lên trước."
            }

        documents = []
        metadatas = results.get("metadatas", []) or []
        for doc, meta in zip(results["documents"], metadatas):
            if meta and meta.get("source") == "screen_capture":
                continue
            documents.append(doc)

        if not documents:
            documents = results["documents"]

        context = "\n\n".join(documents)
        if len(context) > 15000:
            context = context[:15000]

        prompt = (
            "Bạn là biên kịch podcast tiếng Việt cho chương trình 'DocuMind Podcast'.\n"
            "Hai người dẫn có tên thật: **Lan** (nữ, năng động, gần gũi khán giả) và **Minh** (nam, bình tĩnh, phân tích sâu).\n"
            "Họ thảo luận tự nhiên về tài liệu được cung cấp.\n"
            "Kịch bản gồm 6–10 lượt thoại xen kẽ Lan và Minh.\n\n"
            "QUY TẮC XƯNG HÔ (bắt buộc):\n"
            "- Trong trường 'text', gọi nhau bằng tên: 'Minh', 'Lan', 'anh Minh', 'chị Lan' — tự nhiên như bạn bè đồng nghiệp.\n"
            "- Gọi khán giả: 'các bạn', 'mọi người'.\n"
            "- TUYỆT ĐỐI KHÔNG viết: Host A, Host B, MC A, MC B, Người dẫn A/B.\n"
            "- Không emoji, không markdown, mỗi lượt 2–4 câu (dưới 450 ký tự).\n\n"
            "Trả về JSON array. Mỗi phần tử: 'host' ('A' = Lan, 'B' = Minh) và 'text' (lời thoại).\n"
            "Ví dụ:\n"
            "[\n"
            "  {\"host\": \"A\", \"text\": \"Chào các bạn! Mình là Lan. Hôm nay mình và Minh sẽ cùng tìm hiểu một chủ đề thú vị từ tài liệu này.\"},\n"
            "  {\"host\": \"B\", \"text\": \"Chào Lan, chào mọi người! Đúng vậy, khi đọc kỹ mình thấy có vài điểm rất đáng bàn...\"}\n"
            "]\n"
            "Chỉ trả JSON, không markdown, không giải thích thêm."
        )

        if custom_instructions:
            prompt += f"\nCHỈ DẪN TÙY CHỈNH TỪ NGƯỜI DÙNG: Hãy tập trung cuộc đối thoại theo yêu cầu sau: '{custom_instructions}'.\n"

        try:
            response_tokens = []
            stream = self.llm_service.generate_answer(context=context, query=prompt, provider=provider)
            for token in stream:
                response_tokens.append(token)

            raw_response = "".join(response_tokens).strip()

            cleaned = raw_response
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            if cleaned.endswith("```"):
                cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

            script = []
            try:
                script = json.loads(cleaned)
                if not isinstance(script, list):
                    raise ValueError("Parsed JSON is not a list")
            except Exception as e:
                logger.warning(f"Failed to parse strict JSON: {e}. Attempting Regex fallback...")
                pattern = r'"host"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"(.*?)"'
                matches = re.findall(pattern, raw_response, re.DOTALL)
                for host, text in matches:
                    text_clean = text.replace('\\"', '"').replace('\\n', '\n').strip()
                    script.append({"host": host.strip(), "text": text_clean})

                if not script:
                    pattern_speaker = r'"speaker"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"(.*?)"'
                    matches_speaker = re.findall(pattern_speaker, raw_response, re.DOTALL)
                    for speaker, text in matches_speaker:
                        host_val = "A" if speaker.strip().lower() in ("lan", "a", "female") else "B"
                        text_clean = text.replace('\\"', '"').replace('\\n', '\n').strip()
                        script.append({"host": host_val, "text": text_clean})

                if not script:
                    lines = [l.strip() for l in raw_response.split('\n') if l.strip()]
                    for i, line in enumerate(lines):
                        host_val = "A" if i % 2 == 0 else "B"
                        script.append({"host": host_val, "text": line})

            final_script = []
            for item in script:
                host_val = item.get("host", item.get("speaker", "A"))
                if host_val not in ("A", "B"):
                    host_val = "A" if str(host_val).lower() in ("lan", "a", "female", "nữ", "nu") else "B"
                text_val = clean_tts_text(sanitize_podcast_dialogue(item.get("text", "")))
                if text_val:
                    speaker = "Lan" if host_val == "A" else "Minh"
                    final_script.append({"host": host_val, "speaker": speaker, "text": text_val})

            if not final_script:
                return {
                    "success": False,
                    "message": "Không tạo được kịch bản podcast từ tài liệu. Hãy thử lại.",
                }

            os.makedirs("app/static/outputs", exist_ok=True)
            temp_files = []
            session_id = uuid.uuid4().hex[:12]
            enriched_script = []
            tts_failures = 0

            for idx, turn in enumerate(final_script):
                host = turn.get("host", "A")
                clean_text = turn.get("text", "")
                voice_key = host_to_voice_key(host)
                turn_filename = f"podcast_{session_id}_turn_{idx}.mp3"
                turn_filepath = os.path.join("app/static/outputs", turn_filename)

                entry = {**turn, "audio_url": None}
                try:
                    ok = await synthesize_to_file(clean_text, turn_filepath, voice_key=voice_key)
                    if ok and os.path.getsize(turn_filepath) > 0:
                        temp_files.append(turn_filepath)
                        entry["audio_url"] = f"/static/outputs/{turn_filename}"
                    else:
                        tts_failures += 1
                        logger.error("Edge TTS returned empty audio for turn %s", idx)
                except Exception as tts_err:
                    tts_failures += 1
                    logger.error("Edge TTS failed for turn %s: %s", idx, tts_err)

                enriched_script.append(entry)
                if idx + 1 < len(final_script):
                    await asyncio.sleep(TURN_DELAY_SEC)

            combined_filename = f"briefing_{session_id}.mp3"
            combined_filepath = os.path.join("app/static/outputs", combined_filename)
            audio_url = None

            if temp_files:
                with open(combined_filepath, "wb") as outfile:
                    for temp_file in temp_files:
                        with open(temp_file, "rb") as infile:
                            outfile.write(infile.read())
                audio_url = f"/static/outputs/{combined_filename}"

            if not audio_url:
                return {
                    "success": False,
                    "message": (
                        "Không tạo được audio Neural (Edge TTS). "
                        "Kiểm tra kết nối Internet, thử lại sau vài giây hoặc rút ngắn tài liệu."
                    ),
                    "script": enriched_script,
                    "tts_failures": tts_failures,
                }

            result = {
                "success": True,
                "audio_url": audio_url,
                "script": enriched_script,
            }
            if tts_failures:
                result["tts_warning"] = (
                    f"Chỉ tạo được audio cho {len(temp_files)}/{len(final_script)} lượt thoại."
                )
            return result

        except Exception as e:
            logger.error(f"Error generating audio briefing: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Không thể tạo podcast từ tài liệu: {str(e)}"
            }


PodcastUseCase = AudioBriefingUseCase

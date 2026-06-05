import logging
import json
import re
import os
import uuid
import asyncio
import edge_tts
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class AudioBriefingUseCase:
    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        """
        Use case to generate a 2-host audio briefing podcast script from documents and synthesize it into a single MP3 file.
        """
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def execute(self, provider: str = "auto", notebook_id: str = "default", custom_instructions: str = "") -> dict:
        # Retrieve document chunks filtered by notebook_id
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
            "Bạn là biên kịch cho một chương trình Podcast đối thoại trực tuyến nổi tiếng gọi là 'AI Podcast Briefing'.\n"
            "Hãy tạo một kịch bản thảo luận cực kỳ sôi nổi, cuốn hút và dễ hiểu bằng tiếng Việt giữa hai người dẫn chương trình: "
            "Host A (Giọng nữ, năng động, kết nối khán giả) và Host B (Giọng nam, thông thái, phân tích sâu, trầm ấm).\n"
            "Họ sẽ thảo luận, phân tích và chia sẻ về các tài liệu được cung cấp dưới đây.\n"
            "Kịch bản phải gồm khoảng 6 đến 10 lượt hội thoại xen kẽ giữa Host A và Host B.\n"
            "Bắt buộc trả về kết quả dưới dạng một mảng JSON các đối tượng. Mỗi đối tượng có hai trường: "
            "'host' (chỉ nhận giá trị 'A' hoặc 'B') và 'text' (nội dung đối thoại).\n"
            "Ví dụ:\n"
            "[\n"
            "  {\"host\": \"A\", \"text\": \"Chào bạn nghe đài! Hôm nay chúng ta sẽ tìm hiểu về một chủ đề rất hay...\"},\n"
            "  {\"host\": \"B\", \"text\": \"Chào bạn! Đúng vậy, tài liệu này có nhiều điểm vô cùng sâu sắc...\"}\n"
            "]\n"
            "Không thêm bất kỳ chữ nào ngoài mã JSON trên, không dùng ký tự định dạng markdown ```json."
        )

        if custom_instructions:
            prompt += f"\nCHỈ DẪN TÙY CHỈNH TỪ NGƯỜI DÙNG: Hãy tập trung cuộc đối thoại theo yêu cầu sau: '{custom_instructions}'.\n"

        try:
            response_tokens = []
            stream = self.llm_service.generate_answer(context=context, query=prompt, provider=provider)
            for token in stream:
                response_tokens.append(token)
            
            raw_response = "".join(response_tokens).strip()
            
            # Clean up markdown blocks if present
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
                    script.append({
                        "host": host.strip(),
                        "text": text_clean
                    })
                
                if not script:
                    # Alternative regex check with speaker
                    pattern_speaker = r'"speaker"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"(.*?)"'
                    matches_speaker = re.findall(pattern_speaker, raw_response, re.DOTALL)
                    for speaker, text in matches_speaker:
                        host_val = "A" if speaker.strip().lower() in ("lan", "a", "female") else "B"
                        text_clean = text.replace('\\"', '"').replace('\\n', '\n').strip()
                        script.append({
                            "host": host_val,
                            "text": text_clean
                        })

                if not script:
                    # Split lines fallback
                    lines = [l.strip() for l in raw_response.split('\n') if l.strip()]
                    for i, line in enumerate(lines):
                        host_val = "A" if i % 2 == 0 else "B"
                        script.append({"host": host_val, "text": line})

            # Ensure all entries have 'host' and 'text' and translate speakers
            final_script = []
            for item in script:
                host_val = item.get("host", item.get("speaker", "A"))
                if host_val not in ("A", "B"):
                    host_val = "A" if host_val in ("Lan", "female", "nữ") else "B"
                final_script.append({
                    "host": host_val,
                    "text": item.get("text", "")
                })

            # Synthesize host turns and combine into briefing.mp3
            os.makedirs("app/static/outputs", exist_ok=True)
            temp_files = []
            
            # Map voices: Host A -> vi-VN-HoaiAnNeural, Host B -> vi-VN-NamMinhNeural
            voice_map = {
                "A": "vi-VN-HoaiAnNeural",
                "B": "vi-VN-NamMinhNeural"
            }

            for idx, turn in enumerate(final_script):
                text = turn.get("text", "")
                clean_text = re.sub(r'\[.*?\]|\(.*?\)', '', text).strip()
                if not clean_text:
                    clean_text = text
                
                host = turn.get("host", "A")
                voice = voice_map.get(host, "vi-VN-HoaiAnNeural")
                temp_filename = f"temp_briefing_turn_{idx}.mp3"
                temp_filepath = os.path.join("app/static/outputs", temp_filename)
                
                try:
                    communicate = edge_tts.Communicate(clean_text, voice)
                    await communicate.save(temp_filepath)
                    temp_files.append(temp_filepath)
                except Exception as tts_err:
                    logger.error(f"Error during edge-tts synthesis: {tts_err}")

            # Combine temporary files into briefing.mp3
            combined_filename = "briefing.mp3"
            combined_filepath = os.path.join("app/static/outputs", combined_filename)
            
            if temp_files:
                with open(combined_filepath, "wb") as outfile:
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            with open(temp_file, "rb") as infile:
                                outfile.write(infile.read())
                            # Clean up temp file
                            try:
                                os.remove(temp_file)
                            except Exception as rm_err:
                                logger.warning(f"Failed to remove temp file {temp_file}: {rm_err}")
                
                audio_url = f"/static/outputs/{combined_filename}"
            else:
                audio_url = None

            return {
                "success": True,
                "audio_url": audio_url,
                "script": final_script
            }
        except Exception as e:
            logger.error(f"Error generating audio briefing: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Không thể tạo podcast từ tài liệu: {str(e)}"
            }


# Backwards compatibility alias
PodcastUseCase = AudioBriefingUseCase

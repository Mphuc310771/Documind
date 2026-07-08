import os
import uuid
import json
import re
import logging
from datetime import datetime
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class AppGeneratorUseCase:
    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        """
        Use case to read document context and generate an executable single-page HTML/CSS/JS application.
        The generated app is hosted in a local sandbox directory.
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.apps_dir = "app/static/sandbox_apps"
        os.makedirs(self.apps_dir, exist_ok=True)

    def execute(self, notebook_id: str, app_prompt: str, provider: str = "auto") -> dict:
        """
        Retrieves context, instructs LLM to write a single-page app, saves it, and logs metadata.
        """
        # 1. Retrieve context document chunks from ChromaDB
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=25
        )

        context = ""
        if results and results.get("documents"):
            documents = []
            metadatas = results.get("metadatas", []) or []
            for doc, meta in zip(results["documents"], metadatas):
                if meta and meta.get("source") == "screen_capture":
                    continue
                documents.append(doc)
            if not documents:
                documents = results["documents"]

            context = "\n\n".join(documents)
            if len(context) > 18000:
                context = context[:18000]

        # 2. Formulate the prompt for HTML/JS app generation
        system_prompt = (
            "You are a master full-stack software engineer and expert UI/UX designer. "
            "You write complete, highly functional, beautiful, single-file HTML5 applications (SPAs). "
            "Your output must be 100% valid HTML, using modern Tailwind CSS and FontAwesome, and containing all game/logic scripts locally."
        )

        query_prompt = (
            f"Dựa trên các tài liệu nghiên cứu dưới đây, hãy thiết kế và lập trình một ứng dụng web (Single Page Application) "
            f"hoàn chỉnh, tương tác tốt và chạy độc lập trong một file HTML duy nhất theo yêu cầu sau:\n"
            f"YÊU CẦU DỰ ÁN: {app_prompt}\n\n"
            f"TÀI LIỆU THAM KHẢO NGỮ CẢNH:\n{context}\n\n"
            f"QUY TẮC BẮT BUỘC:\n"
            f"1. Tạo một trang HTML5 hoàn chỉnh, sử dụng TailwindCSS qua CDN (link: https://cdn.tailwindcss.com) "
            f"và FontAwesome cho icon (link: https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css).\n"
            f"2. Sử dụng Google Fonts (như Inter, Outfit, hoặc Plus Jakarta Sans) để ứng dụng trông hiện đại và chuyên nghiệp.\n"
            f"3. Ứng dụng phải có thiết kế cao cấp, hiện đại (glassmorphic, gradient màu sắc, bo góc mịn màng, hiệu ứng hover, hiệu ứng động mượt mà).\n"
            f"4. Viết toàn bộ code CSS bổ sung và Javascript trong các thẻ <style> và <script> bên trong file HTML đó.\n"
            f"5. **QUAN TRỌNG**: Ứng dụng phải HOÀN TOÀN CHẠY ĐƯỢC và có logic tương tác thực tế (sử dụng Javascript). "
            f"Không viết mã giả (pseudo-code), không để lại comment TODO, không bỏ dở tính năng hay dùng dữ liệu giả không tương tác. "
            f"Mọi nút bấm, biểu mẫu, sự kiện đều phải hoạt động trơn tru.\n"
            f"6. Lưu giữ trạng thái ứng dụng (state) qua localStorage để khi người dùng tải lại trang không bị mất dữ liệu.\n"
            f"7. Trả về mã nguồn HTML hoàn chỉnh bắt đầu bằng <!DOCTYPE html>. Chỉ xuất ra mã nguồn HTML, không bao quanh bằng ký tự markdown code block (không dùng ```html ... ```)."
        )

        try:
            # 3. Generate HTML code using LLM
            response_tokens = []
            stream = self.llm_service.generate_answer(
                context=context,
                query=query_prompt,
                system_prompt=system_prompt,
                provider=provider
            )
            for token in stream:
                response_tokens.append(token)

            raw_code = "".join(response_tokens).strip()

            # Clean markdown code blocks if any
            cleaned_code = raw_code
            if cleaned_code.startswith("```"):
                cleaned_code = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_code)
            if cleaned_code.endswith("```"):
                cleaned_code = re.sub(r"\n?```$", "", cleaned_code)
            cleaned_code = cleaned_code.strip()

            if not cleaned_code.startswith("<!DOCTYPE html>") and "<html" not in cleaned_code:
                # Fallback if AI didn't wrap it properly
                cleaned_code = f"<!DOCTYPE html>\n<html>\n<head>\n<title>{app_prompt[:30]}</title>\n<script src='https://cdn.tailwindcss.com'></script>\n</head>\n<body>\n{cleaned_code}\n</body>\n</html>"

            # 4. Generate app ID and save file
            app_id = f"app_{uuid.uuid4().hex[:12]}"
            app_filename = f"{app_id}.html"
            app_file_path = os.path.join(self.apps_dir, app_filename)

            with open(app_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_code)

            # 5. Extract a friendly title and description using regex or defaults
            title = app_prompt[:40]
            title_match = re.search(r"<title>(.*?)</title>", cleaned_code, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()

            description = f"Ứng dụng web được lập trình tự động dựa trên yêu cầu: '{app_prompt}'."

            # 6. Update metadata JSON
            metadata = self._load_metadata()
            new_app = {
                "id": app_id,
                "notebook_id": notebook_id,
                "name": title,
                "description": description,
                "url": f"/static/sandbox_apps/{app_filename}",
                "created_at": datetime.now().isoformat(),
                "prompt": app_prompt
            }
            metadata.append(new_app)
            self._save_metadata(metadata)

            return {
                "success": True,
                "app": new_app
            }

        except Exception as e:
            logger.error(f"Error generating sandbox app: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Lỗi tạo ứng dụng: {str(e)}"
            }

    def list_apps(self, notebook_id: str) -> list:
        """
        Lists all generated apps for a specific notebook.
        """
        metadata = self._load_metadata()
        return [app for app in metadata if app.get("notebook_id") == notebook_id]

    def delete_app(self, app_id: str) -> bool:
        """
        Deletes a generated app by ID.
        """
        metadata = self._load_metadata()
        app_to_delete = None
        for app in metadata:
            if app.get("id") == app_id:
                app_to_delete = app
                break

        if not app_to_delete:
            return False

        # Remove from metadata list
        metadata = [app for app in metadata if app.get("id") != app_id]
        self._save_metadata(metadata)

        # Delete HTML file
        app_filename = os.path.basename(app_to_delete.get("url"))
        file_path = os.path.join(self.apps_dir, app_filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Failed to delete app file {file_path}: {e}")

        return True

    def _load_metadata(self) -> list:
        meta_path = os.path.join(self.apps_dir, "apps_metadata.json")
        if not os.path.exists(meta_path):
            return []
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_metadata(self, metadata: list):
        meta_path = os.path.join(self.apps_dir, "apps_metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save apps metadata: {e}")

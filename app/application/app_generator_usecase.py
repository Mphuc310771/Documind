import os
import uuid
import json
import re
import logging
from datetime import datetime
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)

TEMPLATES = [
    {
        "id": "doc_visualizer",
        "name": "📖 Trực Quan Hóa Tài Liệu Bài Học",
        "description": "Trực quan hóa tài liệu bài học dưới dạng giao diện học tập tương tác, phân chia chương mục thông minh, tóm tắt ý chính và sơ đồ tóm lược khái niệm.",
        "type": "doc",
        "prompt": "Tạo một ứng dụng trực quan hóa tài liệu bài học (Lesson Document Visualizer). Ứng dụng phải đọc toàn bộ nội dung tài liệu học tập được nhúng, tự động phân tích và chia nhỏ thành các chương/mục học tập sinh động, có thanh tìm kiếm khái niệm nhanh, bảng tóm tắt ý chính (key takeaways), danh mục thuật ngữ (glossary) tương tác, và phần ghi chú cá nhân giúp người học ghi chép lại các điểm quan trọng."
    },
    {
        "id": "data_analyzer",
        "name": "📊 Thống Kê Phân Tích Dữ Liệu",
        "description": "Ứng dụng chuyên sâu để phân tích, tính toán số liệu và vẽ biểu đồ thống kê các cột dữ liệu dựa trên tài liệu/dataset đã gửi lên.",
        "type": "data",
        "prompt": "Tạo một ứng dụng thống kê phân tích dữ liệu dựa trên tài liệu đã gửi lên (Data Analyzer Dashboard). Ứng dụng thực hiện gọi fetch dữ liệu thực từ dataset thông qua API được cung cấp, tự động tính toán các chỉ số thống kê mô tả (Min, Max, Trung bình, Trung vị, Tổng, Số lượng), vẽ biểu đồ cột/đường/tròn phân phối bằng Chart.js, cho phép người dùng chọn cột số và cột phân loại để tạo biểu đồ tương tác, và xuất dữ liệu báo cáo dạng bảng cực đẹp."
    }
]


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

    def execute(self, notebook_id: str, app_prompt: str, provider: str = "auto", template_id: str = None) -> dict:
        """
        Retrieves smart context, selects prompt strategy, calls LLM, injects runtime config,
        saves HTML app, and logs metadata.
        """
        # 1. Build smart context (separated by datasets vs documents)
        context_data = self._build_smart_context(notebook_id)
        datasets = context_data.get("datasets", [])
        documents = context_data.get("documents", [])

        # Get text for documents
        document_text = "\n\n".join([f"Source: {d['source']}\n{d['text']}" for d in documents])
        if len(document_text) > 15000:
            document_text = document_text[:15000]

        # Determine prompt strategy
        has_dataset = len(datasets) > 0
        
        # 2. Formulate prompt based on data availability
        system_prompt = (
            "You are a master full-stack software engineer and expert UI/UX designer. "
            "You write complete, highly functional, beautiful, single-file HTML5 applications (SPAs). "
            "Your output must be 100% valid HTML, using modern Tailwind CSS and FontAwesome, and containing all game/logic scripts locally."
        )

        if has_dataset:
            # Data visualization strategy
            dataset = datasets[0]
            filename = dataset["filename"]
            api_url = dataset["api_url"]
            columns = dataset["columns"]
            preview_data = dataset["preview_data"]

            query_prompt = (
                f"Dựa trên các tài liệu nghiên cứu và cấu trúc dữ liệu dưới đây, hãy thiết kế và lập trình một ứng dụng web (Single Page Application) "
                f"hoàn chỉnh, tương tác tốt và chạy độc lập trong một file HTML duy nhất theo yêu cầu sau:\n"
                f"YÊU CẦU DỰ ÁN: {app_prompt}\n\n"
                f"CẤU TRÚC DỮ LIỆU ĐƯỢC CUNG CẤP (DATASET MANIFEST):\n"
                f"- Tên tệp: {filename}\n"
                f"- API truy cập dữ liệu thực: {api_url}\n"
                f"- Các cột: {', '.join(columns)}\n\n"
                f"TÀI LIỆU THAM KHẢO NGỮ CẢNH KHÁC (NẾU CÓ):\n{document_text}\n\n"
                f"QUY TẮC BẮT BUỘC:\n"
                f"1. Tạo một trang HTML5 hoàn chỉnh, sử dụng TailwindCSS qua CDN (link: https://cdn.tailwindcss.com) "
                f"và thư viện biểu đồ Chart.js qua CDN (link: https://cdn.jsdelivr.net/npm/chart.js) để vẽ biểu đồ trực quan hóa.\n"
                f"2. Sử dụng Google Fonts (như Inter, Outfit, hoặc Plus Jakarta Sans) để ứng dụng trông hiện đại và chuyên nghiệp.\n"
                f"3. Ứng dụng phải có thiết kế cao cấp, hiện đại (glassmorphic, gradient màu sắc, bo góc mịn màng, hiệu ứng hover, hiệu ứng động mượt mà).\n"
                f"4. Viết toàn bộ code CSS bổ sung và Javascript trong các thẻ <style> và <script> bên trong file HTML đó.\n"
                f"5. **QUAN TRỌNG VỀ DỮ LIỆU THỰC**: Để lấy đầy đủ dữ liệu thực, ứng dụng JavaScript phải thực hiện gọi fetch API động đến API URL. "
                f"Hãy viết code Javascript thực hiện fetch dữ liệu này dạng JSON, ví dụ:\n"
                f"   const notebookId = window.__DOCUMIND_CONFIG__?.notebook_id || 'default';\n"
                f"   const filename = '{filename}';\n"
                f"   fetch(`/sandbox/api/data?notebook_id=${{notebookId}}&filename=${{encodeURIComponent(filename)}}`)\n"
                f"     .then(r => r.json())\n"
                f"     .then(data => {{ \n"
                f"         // Cập nhật biểu đồ, bảng xếp hạng và giao diện với dữ liệu thực\n"
                f"         updateAppWithData(data);\n"
                f"     }});\n"
                f"6. Nhúng dữ liệu xem trước (preview) dưới dạng một mảng JS tĩnh làm fallback: `const fallbackData = {json.dumps(preview_data, ensure_ascii=False)};`. "
                f"Nếu fetch API thất bại hoặc chạy offline, dùng fallbackData này để ứng dụng vẫn hiển thị hoạt động tốt.\n"
                f"7. Hỗ trợ thay đổi loại biểu đồ (Bar, Line, Pie), bộ lọc dữ liệu, tìm kiếm và thống kê tự động các giá trị số (Min, Max, Avg, Sum).\n"
                f"8. Trả về mã nguồn HTML hoàn chỉnh bắt đầu bằng <!DOCTYPE html>. Chỉ xuất ra mã nguồn HTML, không bao quanh bằng ký tự markdown code block (không dùng ```html ... ```)."
            )
        else:
            # Document visualization strategy
            query_prompt = (
                f"Dựa trên tài liệu nghiên cứu dưới đây, hãy thiết kế và lập trình một ứng dụng web (Single Page Application) "
                f"hoàn chỉnh, tương tác tốt và chạy độc lập trong một file HTML duy nhất theo yêu cầu sau:\n"
                f"YÊU CẦU DỰ ÁN: {app_prompt}\n\n"
                f"TÀI LIỆU THAM KHẢO NGỮ CẢNH:\n{document_text}\n\n"
                f"QUY TẮC BẮT BUỘC:\n"
                f"1. Tạo một trang HTML5 hoàn chỉnh, sử dụng TailwindCSS qua CDN (link: https://cdn.tailwindcss.com) "
                f"và FontAwesome cho icon (link: https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css). "
                f"Đặc biệt, nếu sử dụng bất kỳ thư viện vẽ biểu đồ, đồ thị, sơ đồ tư duy nào như D3.js hay Chart.js, bạn BẮT BUỘC phải nhúng script CDN của thư viện đó trong thẻ <head> (Ví dụ: <script src=\"https://cdn.jsdelivr.net/npm/d3@7\"></script>).\n"
                f"2. Sử dụng Google Fonts (như Inter, Outfit, hoặc Plus Jakarta Sans) để ứng dụng trông hiện đại và chuyên nghiệp.\n"
                f"3. Ứng dụng phải có thiết kế cao cấp, hiện đại (glassmorphic, gradient màu sắc, bo góc mịn màng, hiệu ứng hover, hiệu ứng động mượt mà).\n"
                f"4. Viết toàn bộ code CSS bổ sung và Javascript trong duy nhất một thẻ <style> và một thẻ <script> bên trong file HTML đó. KHÔNG tách ra nhiều thẻ <script> riêng biệt để tránh lỗi thứ tự khai báo (ReferenceError) hoặc khai báo lại biến (Redeclaration SyntaxError).\n"
                f"5. **TÍCH HỢP TÀI LIỆU**: Nhúng trực tiếp tài liệu nghiên cứu ở trên vào mã nguồn JS dưới dạng các hằng số hoặc cấu trúc JSON có tổ chức. "
                f"Thiết kế các chức năng tương tác dựa trên tài liệu này: thanh tìm kiếm khái niệm, trắc nghiệm ôn tập (Quiz), thẻ học tập lật mặt (Flashcards) hoặc sơ đồ tư duy Mindmap tương tác vẽ bằng SVG/Canvas/HTML.\n"
                f"6. Lưu giữ trạng thái ứng dụng (state) qua localStorage để khi người dùng tải lại trang không bị mất tiến trình học tập. Đặt tên khóa (key) lưu trong localStorage duy nhất theo đường dẫn file (ví dụ: 'my_app_state_' + window.location.pathname.split('/').pop()) để tránh xung đột dữ liệu giữa các ứng dụng khác nhau.\n"
                f"7. Trả về mã nguồn HTML hoàn chỉnh bắt đầu bằng <!DOCTYPE html>. Chỉ xuất ra mã nguồn HTML, không bao quanh bằng ký tự markdown code block (không dùng ```html ... ```)."
            )

        try:
            # 3. Generate HTML code using LLM (excluding redundant context in context param to save tokens)
            response_tokens = []
            stream = self.llm_service.generate_answer(
                context="",  # Already embedded in query_prompt to avoid redundancy
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

            # Inject window.__DOCUMIND_CONFIG__ for runtime info
            config_script = (
                f"\n<script>\n"
                f"  window.__DOCUMIND_CONFIG__ = {{\n"
                f"    notebook_id: {json.dumps(notebook_id)},\n"
                f"    created_at: {json.dumps(datetime.now().isoformat())}\n"
                f"  }};\n"
                f"</script>\n"
            )
            
            if "</head>" in cleaned_code:
                cleaned_code = cleaned_code.replace("</head>", f"{config_script}</head>", 1)
            elif "<body>" in cleaned_code:
                cleaned_code = cleaned_code.replace("<body>", f"<body>{config_script}", 1)
            else:
                cleaned_code = config_script + cleaned_code

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

    def get_suggested_templates(self, notebook_id: str) -> list:
        """
        Decides and filters suggested app templates based on notebook data.
        """
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=50
        )
        
        has_dataset = False
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                if meta and (meta.get("is_dataset") or "data_path" in meta):
                    has_dataset = True
                    break
        
        if not has_dataset and results and results.get("documents"):
            for doc in results["documents"]:
                if doc and doc.strip().startswith("[DATASET]"):
                    has_dataset = True
                    break

        suggested = []
        for t in TEMPLATES:
            # Always include the featured visualizer and analyzer templates at the top
            if t["id"] in ("doc_visualizer", "data_analyzer"):
                suggested.append(t)
            # Include other templates matching data availability
            elif t["type"] == "data" and has_dataset:
                suggested.append(t)
            elif t["type"] == "doc":
                suggested.append(t)
                
        return suggested

    def _build_smart_context(self, notebook_id: str) -> dict:
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=50
        )
        
        datasets = []
        documents = []
        seen_datasets = set()
        seen_documents = set()
        
        if results and results.get("documents"):
            metadatas = results.get("metadatas", []) or []
            for doc, meta in zip(results["documents"], metadatas):
                if not doc:
                    continue
                if meta and meta.get("source") == "screen_capture":
                    continue
                
                # Check if it is a dataset
                is_dataset = False
                data_path = None
                filename = None
                
                if meta:
                    is_dataset = meta.get("is_dataset", False)
                    data_path = meta.get("data_path")
                    filename = meta.get("source")
                
                # Double-check text content for dataset format
                if not is_dataset and doc.strip().startswith("[DATASET]"):
                    is_dataset = True
                    # Try parsing filename and path from text
                    fn_match = re.search(r"\[DATASET\]\s*(.*)", doc)
                    if fn_match:
                        filename = fn_match.group(1).strip()
                    path_match = re.search(r"File path for pandas analysis:\s*(.*)", doc)
                    if path_match:
                        data_path = path_match.group(1).strip()
                
                if is_dataset:
                    if filename and filename not in seen_datasets:
                        seen_datasets.add(filename)
                        
                        # Parse preview or columns from doc
                        columns = []
                        preview_json = []
                        
                        # Extract Columns and dtypes
                        col_section = re.search(r"Columns and dtypes:(.*?)(Preview \(first 5 rows\):|$)", doc, re.DOTALL)
                        if col_section:
                            for col_line in col_section.group(1).strip().split("\n"):
                                if col_line.strip().startswith("-"):
                                    col_name = col_line.strip()[1:].strip().split("(")[0].strip()
                                    columns.append(col_name)
                        
                        # Try to build a preview JSON by parsing the preview markdown table
                        preview_section = re.search(r"Preview \(first 5 rows\):(.*?)(Numeric statistics:|$)", doc, re.DOTALL)
                        if preview_section:
                            preview_text = preview_section.group(1).strip()
                            preview_json = self._parse_markdown_table_to_json(preview_text)
                            
                        datasets.append({
                            "filename": filename,
                            "data_path": data_path,
                            "columns": columns,
                            "preview_data": preview_json,
                            "api_url": f"/sandbox/api/data?notebook_id={notebook_id}&filename={filename}"
                        })
                else:
                    # Normal document text
                    doc_source = meta.get("source", "Unknown") if meta else "Unknown"
                    if doc_source not in seen_documents:
                        seen_documents.add(doc_source)
                        documents.append({
                            "source": doc_source,
                            "text": doc[:3000]
                        })
                    else:
                        # Append text to existing document
                        for d in documents:
                            if d["source"] == doc_source and len(d["text"]) < 9000:
                                d["text"] += "\n\n" + doc[:3000]
                                break
        
        return {
            "datasets": datasets,
            "documents": documents
        }

    @staticmethod
    def _parse_markdown_table_to_json(table_text: str) -> list:
        rows = [line.strip() for line in table_text.split("\n") if line.strip()]
        if len(rows) < 3:
            return []
            
        def split_row(row_str: str) -> list:
            if row_str.startswith("|"):
                row_str = row_str[1:]
            if row_str.endswith("|"):
                row_str = row_str[:-1]
            return [cell.strip() for cell in row_str.split("|")]
            
        headers = split_row(rows[0])
        data_rows = []
        for row in rows[2:]:
            cells = split_row(row)
            if len(cells) == len(headers):
                row_dict = {}
                for h, c in zip(headers, cells):
                    try:
                        if '.' in c:
                            val = float(c)
                        else:
                            val = int(c)
                    except ValueError:
                        val = c
                    row_dict[h] = val
                data_rows.append(row_dict)
        return data_rows

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

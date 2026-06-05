import re
import json
import logging
import requests
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class UploadURLUseCase:
    def __init__(self, vector_store: ChromaDBStore):
        self.vector_store = vector_store

    async def execute(self, url: str, notebook_id: str = "default") -> dict:
        url = self._normalize_url(url)
        logger.info(f"Processing URL upload: {url} (Notebook: {notebook_id})")
        combined_text = ""
        filename = url

        is_youtube = "youtube.com" in url or "youtu.be" in url
        
        try:
            if is_youtube:
                combined_text = await self._scrape_youtube_transcript(url)
                filename = f"YouTube Video ({url})"
            else:
                combined_text = await self._scrape_general_url(url)
                filename = url
        except Exception as scrape_err:
            logger.error(f"Failed to scrape URL {url}: {scrape_err}", exc_info=True)
            raise scrape_err

        if not combined_text or len(combined_text.strip()) < 20:
            raise ValueError("Không thể trích xuất nội dung văn bản hữu ích từ URL được cung cấp.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(combined_text)
        logger.info(f"Total URL chunks generated: {len(chunks)}")

        if chunks:
            metadatas = [{"source": filename, "chunk_index": i, "notebook_id": notebook_id} for i in range(len(chunks))]
            self.vector_store.add_documents(texts=chunks, metadatas=metadatas)

        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "message": f"Đã cào dữ liệu thành công từ URL và lưu trữ thành {len(chunks)} phân đoạn."
        }

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = (url or "").strip()
        if not url:
            raise ValueError("URL không được để trống.")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("URL phải bắt đầu bằng http:// hoặc https://")
        return url

    async def _scrape_general_url(self, url: str) -> str:
        try:
            logger.info("Initializing Playwright browser agent for general scraping...")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                
                title = await page.title()
                content = await page.evaluate("() => document.body.innerText")
                await browser.close()
                return f"Source Webpage: {title}\nURL: {url}\n\nContent:\n{content}"
        except Exception as e:
            logger.warning(f"Playwright scraping failed: {e}. Falling back to requests...")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200:
                raise Exception(f"HTTP Error {resp.status_code} while fetching website: {resp.reason}")
            
            html = resp.text
            # Remove scripts, styles and get text
            clean_text = re.sub(r'<(script|style|iframe)[^>]*>([\s\S]*?)<\/\1>', '', html, flags=re.IGNORECASE)
            clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return f"Source URL: {url}\n\nContent:\n{clean_text}"

    async def _scrape_youtube_transcript(self, url: str) -> str:
        video_id = None
        m1 = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        if m1:
            video_id = m1.group(1)
        else:
            m2 = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
            if m2:
                video_id = m2.group(1)
                
        if not video_id:
            logger.warning(f"Could not extract video ID from YouTube URL: {url}. Scraping as generic URL.")
            return await self._scrape_general_url(url)
            
        video_page_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Scraping YouTube captions for video ID: {video_id}...")
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(video_page_url, wait_until="domcontentloaded", timeout=15000)
                
                title = await page.title()
                html_content = await page.content()
                await browser.close()
                
                # Check for caption tracks JSON block in youtube player response
                match = re.search(r'"captionTracks":\s*(\[.*?\])', html_content)
                if match:
                    caption_tracks = json.loads(match.group(1))
                    if caption_tracks:
                        track_url = caption_tracks[0].get("baseUrl")
                        if track_url:
                            resp = requests.get(track_url, timeout=10)
                            if resp.status_code == 200:
                                xml_content = resp.text
                                # Extract all text nodes
                                text_lines = re.findall(r'<text[^>]*>(.*?)</text>', xml_content)
                                import html as html_lib
                                clean_lines = [html_lib.unescape(line) for line in text_lines]
                                transcript_text = " ".join(clean_lines)
                                return f"YouTube Video: {title}\nURL: {url}\n\nTranscript:\n{transcript_text}"
                
                # If no captions match, fall back to video metadata extraction
                desc_match = re.search(r'"shortDescription":"(.*?)"', html_content)
                desc = desc_match.group(1).replace("\\n", "\n").replace('\\"', '"') if desc_match else ""
                return f"YouTube Video: {title}\nURL: {url}\n\nDescription:\n{desc}\n\n(Không tìm thấy phụ đề tiếng Việt/Tiếng Anh tự động trực tiếp trên video này.)"
        except Exception as e:
            logger.error(f"Playwright YouTube extraction error: {e}. Falling back to requests...")
            # Requests fallback for metadata
            resp = requests.get(video_page_url, timeout=10)
            if resp.status_code == 200:
                html = resp.text
                title_m = re.search(r"<title>(.*?)</title>", html)
                title = title_m.group(1) if title_m else "YouTube Video"
                return f"YouTube Video: {title}\nURL: {url}\n\n(Lỗi tự động tải phụ đề hoặc Playwright: {str(e)})"
            raise e

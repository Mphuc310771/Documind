import os
import uuid
import logging
import tempfile
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


class BrowserAgent:
    def __init__(self, download_dir: str = None):
        """
        Playwright Headless Browser Agent designed under the Command Pattern
        to automate file download operations.
        """
        if download_dir is None:
            download_dir = os.path.join(tempfile.gettempdir(), "downloads")
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    async def execute_download(self, url: str, search_query: str) -> str | None:
        """
        Launches a headless browser, navigates to the URL, clicks the target download links,
        saves the downloaded file, and returns the absolute file path.
        Returns None if no matching download link is found or an error occurs.
        """
        logger.info(f"Navigating to {url} to download document matching query: '{search_query}'")
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(accept_downloads=True)
                page = await context.new_page()

                # Open target page
                await page.goto(url, timeout=15000)
                await page.wait_for_timeout(2000)

                # Selectors prioritizing specific matching query text
                selectors = [
                    f"text={search_query}",
                    f"a:has-text('{search_query}')",
                    f"button:has-text('{search_query}')",
                    "a[href$='.pdf']",
                    "a:has-text('download')",
                    "a:has-text('tải xuống')"
                ]

                target_element = None
                for sel in selectors:
                    try:
                        element = page.locator(sel)
                        if await element.count() > 0:
                            target_element = element.first
                            break
                    except Exception:
                        continue

                # Fallback checking general anchor tags text content
                if not target_element:
                    anchors = await page.query_selector_all("a")
                    for anchor in anchors:
                        text = await anchor.text_content()
                        if text and search_query.lower() in text.lower():
                            target_element = anchor
                            break

                # If no matching element found, return None
                if not target_element:
                    logger.warning(f"No matching download elements found for query: '{search_query}' on {url}")
                    await browser.close()
                    return None

                # Trigger download and wait
                async with page.expect_download(timeout=10000) as download_info:
                    await target_element.click()
                download = await download_info.value

                filename = f"{uuid.uuid4().hex[:8]}_{download.suggested_filename}"
                filepath = os.path.join(self.download_dir, filename)
                await download.save_as(filepath)

                await browser.close()
                logger.info(f"Playwright download success: {filepath}")
                return filepath

            except Exception as e:
                logger.error(f"Playwright download error: {e}")
                return None

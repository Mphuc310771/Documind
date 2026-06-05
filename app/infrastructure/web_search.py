import urllib.parse
import logging
import requests
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class IWebSearch(ABC):
    @abstractmethod
    def search(self, query: str) -> str:
        """
        Execute web search and return a unified string containing the summarized search text.
        """
        pass


class DuckDuckGoSearch(IWebSearch):
    def search(self, query: str) -> str:
        """
        Search using the official DuckDuckGo Instant Answer API.
        Does not get blocked by anomaly modals.
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"
            headers = {"User-Agent": "RAGAIHub/1.0 (contact@example.com)"}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                abstract = data.get("AbstractText", "")
                if abstract:
                    return f"[DuckDuckGo Abstract] {abstract}"
                
                # Check related topics if abstract is empty
                related = data.get("RelatedTopics", [])
                snippets = []
                for item in related[:3]:
                    text = item.get("Text")
                    if text:
                        snippets.append(text)
                if snippets:
                    return "[DuckDuckGo Related] " + " | ".join(snippets)
            return ""
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return ""


class WikipediaSearch(IWebSearch):
    def search(self, query: str) -> str:
        """
        Search Wikipedia using query terms and format the results.
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://vi.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&utf8="
            headers = {"User-Agent": "RAGAIHub/1.0 (contact@example.com)"}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                search_results = data.get("query", {}).get("search", [])
                snippets = []
                for res in search_results[:3]:
                    # Remove search highlighting span tags
                    snippet = res.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', '')
                    snippets.append(f"- {res.get('title')}: {snippet}")
                if snippets:
                    return "[Wikipedia Search]\n" + "\n".join(snippets)
            return ""
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
            return ""


class FallbackWebSearch(IWebSearch):
    def __init__(self):
        self.strategies = [DuckDuckGoSearch(), WikipediaSearch()]

    def search(self, query: str) -> str:
        """
        Try strategies in order, returning the first non-empty search result.
        """
        for strategy in self.strategies:
            result = strategy.search(query)
            if result:
                return result
        return ""

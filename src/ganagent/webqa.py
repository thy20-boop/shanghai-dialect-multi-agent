from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen


DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/?q={query}&kp=-1"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class WebAnswer:
    question: str
    answer: str
    results: list[WebSearchResult]
    status: str = "ok"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [item.as_dict() for item in self.results]
        return payload


def answer_question_from_web(
    question: str,
    max_results: int = 5,
    fetch_pages: bool = True,
) -> WebAnswer:
    query = question.strip()
    if not query:
        return WebAnswer(question=question, answer="没有收到可检索的问题。", results=[], status="empty")
    try:
        results = search_duckduckgo_html(query, max_results=max_results)
    except Exception as exc:
        return WebAnswer(
            question=question,
            answer=f"联网检索失败：{exc}",
            results=[],
            status="failed",
            error=str(exc),
        )

    if not results:
        return WebAnswer(question=question, answer="没有检索到足够可靠的网页结果。", results=[], status="no_results")

    evidence: list[str] = []
    for index, result in enumerate(results, start=1):
        snippet = result.snippet.strip()
        excerpt = fetch_page_excerpt(result.url) if fetch_pages and len(evidence) < 3 else ""
        text = excerpt or snippet
        if not text:
            continue
        evidence.append(f"{index}. {result.title}：{text}")

    if not evidence:
        evidence = [
            f"{index}. {result.title}：{result.snippet or result.url}"
            for index, result in enumerate(results, start=1)
        ]
    answer = "根据联网检索结果，我能给出的回答是：\n" + "\n".join(evidence)
    answer += "\n\n来源：" + "；".join(f"[{index}] {item.title} {item.url}" for index, item in enumerate(results, start=1))
    return WebAnswer(question=question, answer=answer, results=results)


def search_duckduckgo_html(query: str, max_results: int = 5) -> list[WebSearchResult]:
    url = DUCKDUCKGO_HTML_URL.format(query=quote_plus(query))
    html = _http_get(url)
    parser = DuckDuckGoHTMLParser()
    parser.feed(html)
    results: list[WebSearchResult] = []
    seen: set[str] = set()
    for item in parser.results:
        clean_url = clean_duckduckgo_url(item.url)
        if not clean_url or clean_url in seen:
            continue
        seen.add(clean_url)
        results.append(WebSearchResult(title=clean_text(item.title), url=clean_url, snippet=clean_text(item.snippet)))
        if len(results) >= max_results:
            break
    return results


def fetch_page_excerpt(url: str, max_chars: int = 600) -> str:
    try:
        html = _http_get(url, timeout=8)
    except Exception:
        return ""
    parser = PageTextParser()
    parser.feed(html)
    text = clean_text(" ".join(parser.parts))
    if not text:
        return ""
    return text[:max_chars].rstrip()


def clean_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.path.startswith("/l/") or parsed.netloc.endswith("duckduckgo.com"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return url


def clean_text(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _http_get(url: str, timeout: int = 12) -> str:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebSearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_url = ""
        self._last_result_index: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        css_class = attrs_dict.get("class", "")
        if tag == "a" and ("result__a" in css_class or "result-link" in css_class):
            self._in_title = True
            self._current_title = []
            self._current_url = attrs_dict.get("href", "")
        elif "result__snippet" in css_class or "result-snippet" in css_class:
            self._in_snippet = True
            self._current_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._in_title = False
            title = clean_text("".join(self._current_title))
            if title and self._current_url:
                self.results.append(WebSearchResult(title=title, url=self._current_url))
                self._last_result_index = len(self.results) - 1
        elif self._in_snippet and tag in {"a", "td", "div"}:
            self._in_snippet = False
            snippet = clean_text("".join(self._current_snippet))
            if snippet and self._last_result_index is not None:
                current = self.results[self._last_result_index]
                self.results[self._last_result_index] = WebSearchResult(
                    title=current.title,
                    url=current.url,
                    snippet=snippet,
                )

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)


class PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._capture = False
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip = True
        elif tag in {"title", "p", "h1", "h2", "h3", "li"}:
            self._capture = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip = False
        elif tag in {"title", "p", "h1", "h2", "h3", "li"}:
            self._capture = False

    def handle_data(self, data: str) -> None:
        if self._capture and not self._skip:
            text = clean_text(data)
            if len(text) >= 12:
                self.parts.append(text)

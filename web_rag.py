import re
import time
from datetime import datetime
from urllib.parse import urlparse
from typing import List

from tavily import TavilyClient

from config import (
    TAVILY_API_KEY,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_DEPTH,
)


class WebRAG:

    OFFICIAL_DOMAINS = [
        "learn.microsoft.com",
        "powerbi.microsoft.com",
        "community.fabric.microsoft.com",
    ]

    # High-value DAX concepts used to expand multi-function questions.
    DAX_FUNCTIONS = {
        "if", "switch", "calculate", "calculatetable", "filter", "all",
        "removefilters", "keepfilters", "sum", "sumx", "average", "averagex",
        "count", "countrows", "distinctcount", "date", "datediff", "datevalue",
        "day", "month", "year", "today", "now", "format", "left", "right",
        "mid", "search", "find", "related", "relatedtable", "userelationship",
        "values", "distinct", "summarize", "summarizecolumns", "addcolumns",
        "rankx", "divide", "coalesce", "hasonevalue", "isblank"
    }

    def __init__(self):
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not configured.")
        self.client = TavilyClient(api_key=TAVILY_API_KEY)
        print("Live web RAG initialized.")

    @staticmethod
    def is_temporal_query(question: str) -> bool:
        q = question.lower()
        temporal_terms = [
            "latest", "current", "today", "this week", "this month", "this year",
            "recent", "recently", "what's new", "whats new", "new update",
            "new updates", "new feature", "new features", "what changed",
            "what has changed"
        ]
        return any(term in q for term in temporal_terms)

    @classmethod
    def _extract_dax_functions(cls, question: str):
        q = question.lower()
        found = []
        for function in cls.DAX_FUNCTIONS:
            if re.search(r"\b" + re.escape(function) + r"\b", q):
                found.append(function.upper())
        return sorted(set(found))

    @staticmethod
    def _is_comparison_query(question: str) -> bool:
        q = question.lower()
        patterns = [
            r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bcomparison\b",
            r"\bdifference\s+between\b", r"\bdifferences\s+between\b",
            r"\bwhich\s+is\s+better\b", r"\bbetter\s+than\b",
            r"\bwhy\s+(?:choose|use)\b"
        ]
        return any(re.search(pattern, q) for pattern in patterns)

    @staticmethod
    def build_search_query(question: str) -> str:
        now = datetime.now()
        current_month = now.strftime("%B")
        current_year = now.strftime("%Y")
        q = question.strip()

        if WebRAG.is_temporal_query(q):
            return (
                "Power BI "
                f"{current_month} {current_year} "
                "latest update Microsoft Power BI What's New"
            )

        dax_functions = WebRAG._extract_dax_functions(q)
        is_dax = bool(dax_functions) or bool(re.search(r"\bDAX\b", q, re.I))

        if is_dax and WebRAG._is_comparison_query(q):
            entities = " ".join(dax_functions)
            return (
                f"site:learn.microsoft.com/en-us/dax "
                f"DAX function comparison {entities} "
                "IF SWITCH DAX function reference"
            )

        if is_dax:
            entities = " ".join(dax_functions)
            return (
                f"site:learn.microsoft.com/en-us/dax "
                f"DAX function reference {entities}"
            )

        if WebRAG._is_comparison_query(q):
            return (
                f"Power BI comparison {q} "
                "Microsoft Learn Power BI Tableau Qlik"
            )

        # Foundational Power BI questions need a canonical overview rather than
        # an arbitrary related page such as refresh or gateway documentation.
        if re.search(r"\bwhat\s+is\s+power\s*bi\b", q, re.I):
            return (
                "site:learn.microsoft.com/en-us/power-bi "
                "What is Power BI Power BI overview Microsoft"
            )

        if re.search(r"\bwho\s+(invented|created|developed|built|made)\b", q, re.I):
            return (
                f"Power BI history development Microsoft "
                f"{q} official Power BI"
            )

        return q

    @staticmethod
    def normalize_url(url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        path = re.sub(
            r"^/(?:[a-z]{2}(?:-[a-z]{2})?)/",
            "/",
            parsed.path,
            flags=re.IGNORECASE
        )
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        return normalized.rstrip("/")

    @classmethod
    def is_official_domain(cls, url: str) -> bool:
        try:
            hostname = (urlparse(url).hostname or "").lower()
            return any(
                hostname == domain or hostname.endswith("." + domain)
                for domain in cls.OFFICIAL_DOMAINS
            )
        except Exception:
            return False

    @staticmethod
    def is_update_page(url: str) -> bool:
        url = url.lower()
        update_paths = [
            "/power-bi/fundamentals/whats-new",
            "/power-bi/whats-new",
            "/blog/tag/power-bi-desktop",
            "/power-bi-updates-blog",
        ]
        return any(path in url for path in update_paths)

    def filter_results(self, results):
        cleaned = []
        seen_urls = set()
        for item in results:
            url = item.get("url", "")
            if not url:
                continue
            normalized_url = self.normalize_url(url)
            if not normalized_url or normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            if not self.is_official_domain(url):
                continue

            title = item.get("title", "").lower()
            content = item.get("content", "").lower()
            combined = title + " " + content + " " + url.lower()

            old_years = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]
            if any(year in combined for year in old_years) and not self.is_update_page(url):
                continue

            item["normalized_url"] = normalized_url
            cleaned.append(item)
        return cleaned

    def rank_results(self, results, question=""):
        current_year = datetime.now().year
        current_month = datetime.now().strftime("%B").lower()
        q_tokens = set(re.findall(r"\w+", question.lower()))
        dax_functions = set(x.lower() for x in self._extract_dax_functions(question))
        comparison = self._is_comparison_query(question)

        def score(item):
            url = item.get("url", "").lower()
            title = item.get("title", "").lower()
            content = item.get("content", "").lower()
            text = title + " " + content + " " + url
            value = float(item.get("score", 0.0))

            if "learn.microsoft.com" in url:
                value += 5.0
            if self.is_update_page(url):
                value += 4.0
            if str(current_year) in text:
                value += 2.0
            if current_month in text:
                value += 1.0

            # Reward evidence containing the concepts explicitly requested.
            text_tokens = set(re.findall(r"\w+", text))
            value += min(3.0, 0.35 * len(q_tokens & text_tokens))

            if dax_functions:
                matched = sum(1 for fn in dax_functions if fn in text_tokens)
                value += min(6.0, 2.0 * matched)

            # Comparison questions should not be dominated by a page about only
            # one side. Reward pages containing multiple requested entities.
            if comparison:
                competitors = {"power", "bi", "tableau", "qlik", "fabric"}
                value += min(3.0, 0.5 * len(competitors & text_tokens))

            return value

        return sorted(results, key=score, reverse=True)

    def search_official(self, question: str):
        search_query = self.build_search_query(question)
        start = time.perf_counter()

        response = self.client.search(
            query=search_query,
            search_depth=WEB_SEARCH_DEPTH,
            max_results=max(WEB_SEARCH_MAX_RESULTS, 8),
            include_answer=False,
            include_raw_content=False,
            include_domains=self.OFFICIAL_DOMAINS,
        )

        elapsed = time.perf_counter() - start
        raw_results = response.get("results", [])
        cleaned_results = self.filter_results(raw_results)
        ranked_results = self.rank_results(cleaned_results, question)

        final_results = []
        for rank, item in enumerate(ranked_results, start=1):
            final_results.append({
                "rank": rank,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "source": "official_web"
            })

        return {
            "results": final_results,
            "search_time": elapsed,
            "query": search_query,
            "search_type": "official"
        }

    def search_general(self, question: str):
        start = time.perf_counter()
        response = self.client.search(
            query=self.build_search_query(question),
            search_depth=WEB_SEARCH_DEPTH,
            max_results=WEB_SEARCH_MAX_RESULTS,
            include_answer=False,
            include_raw_content=False,
        )
        elapsed = time.perf_counter() - start
        results = []
        for rank, item in enumerate(response.get("results", []), start=1):
            results.append({
                "rank": rank,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "source": "web"
            })
        return {
            "results": results,
            "search_time": elapsed,
            "query": self.build_search_query(question),
            "search_type": "general"
        }

    def search(self, question: str):
        official = self.search_official(question)
        if official.get("results"):
            return official
        return self.search_general(question)

    @staticmethod
    def build_context(results: List[dict]) -> str:
        parts = []
        for result in results:
            parts.append(
                f"""
WEB SOURCE {result["rank"]}

Title:
{result["title"]}

URL:
{result["url"]}

Source:
{result.get("source", "web")}

Content:
{result["content"]}
""".strip()
            )
        return "\n\n".join(parts)

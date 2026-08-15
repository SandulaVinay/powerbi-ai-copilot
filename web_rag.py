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

    # ========================================================
    # OFFICIAL POWER BI SOURCES
    # ========================================================

    OFFICIAL_DOMAINS = [
        "learn.microsoft.com",
        "powerbi.microsoft.com",
        "community.fabric.microsoft.com",
    ]

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        if not TAVILY_API_KEY:

            raise RuntimeError(
                "TAVILY_API_KEY is not configured."
            )

        self.client = TavilyClient(
            api_key=TAVILY_API_KEY
        )

        print(
            "Live web RAG initialized."
        )

    # ========================================================
    # DETECT TEMPORAL QUERY
    # ========================================================

    @staticmethod
    def is_temporal_query(
        question: str
    ) -> bool:

        q = question.lower()

        temporal_terms = [
            "latest",
            "current",
            "today",
            "this week",
            "this month",
            "this year",
            "recent",
            "recently",
            "what's new",
            "whats new",
            "new update",
            "new updates",
            "new feature",
            "new features",
            "what changed",
            "what has changed",
        ]

        return any(
            term in q
            for term in temporal_terms
        )

    # ========================================================
    # BUILD SEARCH QUERY
    # ========================================================

    @staticmethod
    def build_search_query(
        question: str
    ) -> str:

        now = datetime.now()

        current_month = (
            now.strftime("%B")
        )

        current_year = (
            now.strftime("%Y")
        )

        q = question.lower()

        # ----------------------------------------------------
        # Specific Power BI update query
        # ----------------------------------------------------

        if WebRAG.is_temporal_query(
            question
        ):

            return (
                "Power BI "
                f"{current_month} "
                f"{current_year} "
                "latest update "
                "Microsoft "
                "Power BI What's New"
            )

        return question

    # ========================================================
    # URL NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_url(
        url: str
    ) -> str:

        if not url:
            return ""

        parsed = urlparse(url)

        # ----------------------------------------------------
        # Remove language variants where possible
        # ----------------------------------------------------

        path = parsed.path

        path = re.sub(
            r"^/(?:[a-z]{2}(?:-[a-z]{2})?)/",
            "/",
            path,
            flags=re.IGNORECASE
        )

        normalized = (
            f"{parsed.scheme}://"
            f"{parsed.netloc}"
            f"{path}"
        )

        return normalized.rstrip("/")

    # ========================================================
    # CHECK OFFICIAL DOMAIN
    # ========================================================

    @classmethod
    def is_official_domain(
        cls,
        url: str
    ) -> bool:

        try:

            hostname = (
                urlparse(url)
                .hostname
                or ""
            ).lower()

            return any(
                hostname == domain
                or hostname.endswith(
                    "." + domain
                )
                for domain
                in cls.OFFICIAL_DOMAINS
            )

        except Exception:

            return False

    # ========================================================
    # CHECK POWER BI UPDATE PAGE
    # ========================================================

    @staticmethod
    def is_update_page(
        url: str
    ) -> bool:

        url = url.lower()

        update_paths = [

            "/power-bi/fundamentals/whats-new",

            "/power-bi/whats-new",

            "/blog/tag/power-bi-desktop",

            "/power-bi-updates-blog",

        ]

        return any(
            path in url
            for path in update_paths
        )

    # ========================================================
    # FILTER RESULTS
    # ========================================================

    def filter_results(
        self,
        results
    ):

        cleaned = []

        seen_urls = set()

        for item in results:

            url = item.get(
                "url",
                ""
            )

            if not url:
                continue

            normalized_url = (
                self.normalize_url(
                    url
                )
            )

            if not normalized_url:
                continue

            # ------------------------------------------------
            # Deduplicate
            # ------------------------------------------------

            if normalized_url in seen_urls:
                continue

            seen_urls.add(
                normalized_url
            )

            # ------------------------------------------------
            # Official-domain requirement
            # ------------------------------------------------

            if not self.is_official_domain(
                url
            ):
                continue

            # ------------------------------------------------
            # Reject obviously old results
            # for temporal queries
            # ------------------------------------------------

            title = (
                item.get(
                    "title",
                    ""
                )
                .lower()
            )

            content = (
                item.get(
                    "content",
                    ""
                )
                .lower()
            )

            combined = (
                title
                + " "
                + content
                + " "
                + url.lower()
            )

            # ------------------------------------------------
            # Reject very old explicit years
            # ------------------------------------------------

            old_years = [

                "2019",
                "2020",
                "2021",
                "2022",
                "2023",
                "2024",
                "2025",

            ]

            if any(
                year in combined
                for year in old_years
            ):

                # Allow the general What's New
                # page because it contains archive
                # information.

                if not self.is_update_page(
                    url
                ):

                    continue

            item["normalized_url"] = (
                normalized_url
            )

            cleaned.append(
                item
            )

        return cleaned

    # ========================================================
    # RANK RESULTS
    # ========================================================

    def rank_results(
        self,
        results
    ):

        current_year = (
            datetime.now().year
        )

        current_month = (
            datetime.now().strftime(
                "%B"
            ).lower()
        )

        def score(item):

            url = (
                item.get(
                    "url",
                    ""
                )
                .lower()
            )

            title = (
                item.get(
                    "title",
                    ""
                )
                .lower()
            )

            content = (
                item.get(
                    "content",
                    ""
                )
                .lower()
            )

            text = (
                title
                + " "
                + content
                + " "
                + url
            )

            value = float(
                item.get(
                    "score",
                    0.0
                )
            )

            # ------------------------------------------------
            # Strong preference for Microsoft Learn
            # ------------------------------------------------

            if (
                "learn.microsoft.com"
                in url
            ):

                value += 5.0

            # ------------------------------------------------
            # Strong preference for What's New
            # ------------------------------------------------

            if self.is_update_page(
                url
            ):

                value += 4.0

            # ------------------------------------------------
            # Current year
            # ------------------------------------------------

            if str(
                current_year
            ) in text:

                value += 2.0

            # ------------------------------------------------
            # Current month
            # ------------------------------------------------

            if current_month in text:

                value += 1.0

            return value

        return sorted(
            results,
            key=score,
            reverse=True
        )

    # ========================================================
    # OFFICIAL SEARCH
    # ========================================================

    def search_official(
        self,
        question: str
    ):

        search_query = (
            self.build_search_query(
                question
            )
        )

        start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Special query for latest updates
        # ----------------------------------------------------

        if self.is_temporal_query(
            question
        ):

            search_query = (
                f"Power BI "
                f"{datetime.now().strftime('%B')} "
                f"{datetime.now().year} "
                "What's New latest update"
            )

        response = (
            self.client.search(

                query=search_query,

                search_depth=
                    WEB_SEARCH_DEPTH,

                max_results=
                    WEB_SEARCH_MAX_RESULTS,

                include_answer=False,

                include_raw_content=False,

                include_domains=
                    self.OFFICIAL_DOMAINS,

            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        raw_results = (
            response.get(
                "results",
                []
            )
        )

        cleaned_results = (
            self.filter_results(
                raw_results
            )
        )

        ranked_results = (
            self.rank_results(
                cleaned_results
            )
        )

        final_results = []

        for rank, item in enumerate(
            ranked_results,
            start=1
        ):

            final_results.append({

                "rank":
                    rank,

                "title":
                    item.get(
                        "title",
                        ""
                    ),

                "url":
                    item.get(
                        "url",
                        ""
                    ),

                "content":
                    item.get(
                        "content",
                        ""
                    ),

                "score":
                    item.get(
                        "score",
                        0.0
                    ),

                "source":
                    "official_web"

            })

        return {

            "results":
                final_results,

            "search_time":
                elapsed,

            "query":
                search_query,

            "search_type":
                "official"

        }

    # ========================================================
    # GENERAL WEB FALLBACK
    # ========================================================

    def search_general(
        self,
        question: str
    ):

        start = (
            time.perf_counter()
        )

        response = (
            self.client.search(

                query=question,

                search_depth=
                    WEB_SEARCH_DEPTH,

                max_results=
                    WEB_SEARCH_MAX_RESULTS,

                include_answer=False,

                include_raw_content=False,

            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        results = []

        for rank, item in enumerate(
            response.get(
                "results",
                []
            ),
            start=1
        ):

            results.append({

                "rank":
                    rank,

                "title":
                    item.get(
                        "title",
                        ""
                    ),

                "url":
                    item.get(
                        "url",
                        ""
                    ),

                "content":
                    item.get(
                        "content",
                        ""
                    ),

                "score":
                    item.get(
                        "score",
                        0.0
                    ),

                "source":
                    "web"

            })

        return {

            "results":
                results,

            "search_time":
                elapsed,

            "query":
                question,

            "search_type":
                "general"

        }

    # ========================================================
    # MAIN SEARCH
    # ========================================================

    def search(
        self,
        question: str
    ):

        # ----------------------------------------------------
        # First attempt: official Microsoft sources
        # ----------------------------------------------------

        official = (
            self.search_official(
                question
            )
        )

        official_results = (
            official.get(
                "results",
                []
            )
        )

        # ----------------------------------------------------
        # Return official results if available
        # ----------------------------------------------------

        if official_results:

            return official

        # ----------------------------------------------------
        # Fallback to general web
        # ----------------------------------------------------

        general = (
            self.search_general(
                question
            )
        )

        return general

    # ========================================================
    # BUILD CONTEXT FOR LLM
    # ========================================================

    @staticmethod
    def build_context(
        results: List[dict]
    ) -> str:

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

        return "\n\n".join(
            parts
        )
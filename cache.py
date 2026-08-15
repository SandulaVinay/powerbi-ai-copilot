import hashlib
import json
import threading
import time
from pathlib import Path


class AnswerCache:

    def __init__(
        self,
        cache_file,
        ttl_seconds=604800
    ):

        self.cache_file = Path(cache_file)
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()

        self.cache_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._ensure_file()

    # ============================================================
    # INITIALIZE CACHE FILE
    # ============================================================

    def _ensure_file(self):

        if not self.cache_file.exists():

            self._save({})

    # ============================================================
    # LOAD
    # ============================================================

    def _load(self):

        try:

            with open(
                self.cache_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

                if isinstance(data, dict):
                    return data

                return {}

        except (
            FileNotFoundError,
            json.JSONDecodeError
        ):

            return {}

    # ============================================================
    # SAVE
    # ============================================================

    def _save(self, data):

        temporary_file = (
            self.cache_file.with_suffix(".tmp")
        )

        with open(
            temporary_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

        temporary_file.replace(
            self.cache_file
        )

    # ============================================================
    # NORMALIZE QUESTION
    # ============================================================

    @staticmethod
    def normalize(question):

        return " ".join(
            str(question)
            .lower()
            .strip()
            .split()
        )

    # ============================================================
    # CREATE CACHE KEY
    # ============================================================

    @classmethod
    def make_key(cls, question):

        normalized = cls.normalize(
            question
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    # ============================================================
    # CHECK WHETHER AN ANSWER IS CACHEABLE
    # ============================================================

    @staticmethod
    def is_cacheable(
        route,
        answer,
        retrieval_type=None
    ):
        """
        Cache policy:

        LOCAL:
            Cache successful grounded answers.

        WEB:
            Do NOT cache because web information can change.

        REJECT:
            Do NOT cache.

        FAILED / UNKNOWN:
            Do NOT cache.

        """

        # --------------------------------------------------------
        # Only LOCAL answers can enter the persistent cache.
        # --------------------------------------------------------

        if route != "local":
            return False

        # --------------------------------------------------------
        # Empty answer
        # --------------------------------------------------------

        if not answer:
            return False

        answer_text = str(
            answer
        ).strip()

        if not answer_text:
            return False

        # --------------------------------------------------------
        # Never cache insufficient-information responses.
        # --------------------------------------------------------

        blocked_messages = [

            "I don't have enough information in the provided knowledge base.",

            "I don't have enough information in the retrieved web sources.",

            "I don't have enough information",

            "I don't know",

        ]

        answer_lower = (
            answer_text.lower()
        )

        for message in blocked_messages:

            if message.lower() in answer_lower:

                return False

        # --------------------------------------------------------
        # Explicit retrieval type protection
        # --------------------------------------------------------

        if retrieval_type:

            if retrieval_type != "local_hybrid":

                return False

        return True

    # ============================================================
    # GET
    # ============================================================

    def get(self, question):

        key = self.make_key(
            question
        )

        with self.lock:

            data = self._load()

            item = data.get(
                key
            )

            if item is None:

                return None

            # ----------------------------------------------------
            # Safety check:
            #
            # Old cache entries created by previous versions
            # may contain web/rejected answers.
            #
            # Remove them automatically.
            # ----------------------------------------------------

            cached_route = item.get(
                "route",
                "local"
            )

            cached_retrieval_type = item.get(
                "retrieval_type",
                "local_hybrid"
            )

            if cached_route != "local":

                del data[key]

                self._save(
                    data
                )

                return None

            if cached_retrieval_type != "local_hybrid":

                del data[key]

                self._save(
                    data
                )

                return None

            # ----------------------------------------------------
            # Expiration
            # ----------------------------------------------------

            created_at = item.get(
                "created_at",
                0
            )

            age = (
                time.time()
                -
                created_at
            )

            if age > self.ttl_seconds:

                del data[key]

                self._save(
                    data
                )

                return None

            return item

    # ============================================================
    # SET
    # ============================================================

    def set(
        self,
        question,
        answer,
        sources,
        route="local",
        retrieval_type="local_hybrid"
    ):
        """
        Store an answer only when the answer is safe to cache.

        Default behavior is LOCAL + local_hybrid.

        Web answers are intentionally not cached.
        Rejected questions are intentionally not cached.
        """

        # --------------------------------------------------------
        # CACHE POLICY
        # --------------------------------------------------------

        if not self.is_cacheable(
            route=route,
            answer=answer,
            retrieval_type=retrieval_type
        ):

            return False

        key = self.make_key(
            question
        )

        item = {

            "question":
                question,

            "answer":
                answer,

            "sources":
                sources,

            "route":
                route,

            "retrieval_type":
                retrieval_type,

            "created_at":
                time.time()

        }

        with self.lock:

            data = self._load()

            data[key] = item

            self._save(
                data
            )

        return True

    # ============================================================
    # DELETE ONE ANSWER
    # ============================================================

    def delete(
        self,
        question
    ):

        key = self.make_key(
            question
        )

        with self.lock:

            data = self._load()

            if key not in data:

                return False

            del data[key]

            self._save(
                data
            )

            return True

    # ============================================================
    # CLEAR ALL CACHE
    # ============================================================

    def clear(self):

        with self.lock:

            self._save({})

    # ============================================================
    # COUNT
    # ============================================================

    def count(self):

        with self.lock:

            data = self._load()

            return len(
                data
            )
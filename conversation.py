import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


FOLLOW_UP_PATTERNS = [
    r"\bwhat about\b", r"\bhow about\b", r"\band what about\b",
    r"\bwhat if\b", r"\bdoes that\b", r"\bdoes this\b", r"\bwill that\b",
    r"\bwhat does that\b", r"\bwhat does this\b", r"\bhow does that\b",
    r"\bhow does this\b", r"\bthat update\b", r"\bthis update\b",
    r"\bthat feature\b", r"\bthis feature\b", r"\bregarding\b",
    r"\brelated to\b", r"\bany update(?:s)?\b", r"\bwhat changed\b",
]

POWER_BI_ANCHORS = [
    "power bi", "fabric", "power query", "dax", "semantic model",
    "rls", "row level security", "ols", "object level security",
    "copilot", "tmdl", "directquery", "direct lake", "incremental refresh",
    "gateway", "embedded", "power bi service", "power bi desktop",
]

TEMPORAL_ANCHORS = [
    "latest", "recent", "recently", "current", "today", "this month",
    "this week", "this year", "new update", "new updates", "what's new",
    "whats new", "august 2026", "july 2026", "june 2026", "may 2026",
]


@dataclass
class ConversationState:
    conversation_id: str
    turns: List[Dict[str, str]] = field(default_factory=list)
    active_topic: str = ""


class ConversationManager:
    """Conversation understanding layer used before retrieval.

    This layer is intentionally deterministic: it handles conversation state,
    follow-up detection, query rewriting, and retrieval planning without adding
    another LLM/network round trip.
    """

    MAX_TURNS = 8
    MAX_CONTEXT_CHARS = 700

    def __init__(self):
        self._sessions: Dict[str, ConversationState] = {}

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _is_follow_up(question: str) -> bool:
        q = question.lower()
        return any(re.search(pattern, q) for pattern in FOLLOW_UP_PATTERNS)

    @staticmethod
    def _has_domain_anchor(question: str) -> bool:
        q = question.lower()
        return any(anchor in q for anchor in POWER_BI_ANCHORS)

    @staticmethod
    def _has_temporal_anchor(question: str) -> bool:
        q = question.lower()
        return any(anchor in q for anchor in TEMPORAL_ANCHORS)

    @staticmethod
    def _has_security_entity(question: str) -> bool:
        q = question.lower()
        return bool(re.search(r"\brls\b|\brow[- ]level security\b|\bols\b|\bobject[- ]level security\b", q))

    @staticmethod
    def _has_dax_entity(question: str) -> bool:
        q = question.lower()
        return bool(re.search(r"\bdax\b|\bcalculate\b|\bsumx?\b|\bswitch\b|\bfilter\b", q))

    @classmethod
    def _extract_topic(cls, turns: List[Dict[str, str]]) -> str:
        user_turns = [t.get("content", "") for t in turns if t.get("role") == "user"]
        if not user_turns:
            return ""
        recent = user_turns[-3:]
        selected = []
        for text in recent:
            q = cls._clean(text)
            if q and q not in selected:
                selected.append(q)
        return " | ".join(selected)[-cls.MAX_CONTEXT_CHARS :]

    @classmethod
    def infer_plan(cls, question: str, is_follow_up: bool) -> Dict[str, Any]:
        q = question.lower()
        temporal = cls._has_temporal_anchor(q)
        security = cls._has_security_entity(q)
        dax = cls._has_dax_entity(q)

        if temporal:
            intent = "latest_update"
            requires_web = True
            source_preference = "official_web"
            retrieval_strategy = "live_official_web"
        elif security:
            intent = "security"
            requires_web = is_follow_up
            source_preference = "official_web" if requires_web else "local_first"
            retrieval_strategy = "official_web_plus_local" if requires_web else "adaptive_local"
        elif dax:
            intent = "dax"
            requires_web = False
            source_preference = "local_first"
            retrieval_strategy = "adaptive_local"
        elif is_follow_up:
            intent = "contextual_follow_up"
            requires_web = False
            source_preference = "local_first"
            retrieval_strategy = "adaptive_local"
        else:
            intent = "general_power_bi"
            requires_web = False
            source_preference = "local_first"
            retrieval_strategy = "adaptive_local"

        return {
            "intent": intent,
            "requires_web": requires_web,
            "source_preference": source_preference,
            "retrieval_strategy": retrieval_strategy,
        }

    def get_state(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = ConversationState(conversation_id=conversation_id)
        return self._sessions[conversation_id]

    def set_history(self, conversation_id: str, history: Optional[List[Dict[str, Any]]]) -> ConversationState:
        state = self.get_state(conversation_id)
        if history:
            cleaned = []
            for item in history[-self.MAX_TURNS :]:
                role = str(item.get("role", "")).lower().strip()
                content = self._clean(item.get("content", ""))
                if role in {"user", "assistant"} and content:
                    cleaned.append({"role": role, "content": content})
            state.turns = cleaned
        state.active_topic = self._extract_topic(state.turns)
        return state

    def contextualize(self, question: str, state: ConversationState) -> Dict[str, Any]:
        question = self._clean(question)
        previous_users = [t["content"] for t in state.turns if t.get("role") == "user"]
        previous_question = previous_users[-1] if previous_users else ""

        if not previous_question:
            plan = self.infer_plan(question, False)
            return {
                "query": question,
                "is_follow_up": False,
                "active_topic": "",
                "rewrite_reason": "no_conversation_context",
                **plan,
            }

        follow_up = self._is_follow_up(question)
        short_or_ambiguous = len(question.split()) <= 8 and not self._has_domain_anchor(question)
        temporal_context = self._has_temporal_anchor(previous_question)

        if follow_up or short_or_ambiguous or temporal_context:
            topic = state.active_topic or previous_question
            rewritten = (
                f"Power BI conversation context: {topic}. "
                f"Current user question: {question}. "
                "Answer the current question, using the previous context only when relevant."
            )
            plan = self.infer_plan(rewritten, True)
            return {
                "query": rewritten,
                "is_follow_up": True,
                "active_topic": topic,
                "rewrite_reason": "contextual_rewrite",
                **plan,
            }

        plan = self.infer_plan(question, False)
        return {
            "query": question,
            "is_follow_up": False,
            "active_topic": state.active_topic,
            "rewrite_reason": "standalone_question",
            **plan,
        }

    def add_turn(self, conversation_id: str, role: str, content: str) -> None:
        state = self.get_state(conversation_id)
        content = self._clean(content)
        if not content:
            return
        state.turns.append({"role": role, "content": content})
        state.turns = state.turns[-self.MAX_TURNS :]
        state.active_topic = self._extract_topic(state.turns)

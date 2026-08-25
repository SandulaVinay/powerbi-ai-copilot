"""Lightweight intent detection for retrieval planning."""
import re


def detect_intent(question: str) -> str:
    q = question.lower().strip()
    if re.search(r"\b(latest|recent|recently|current|what'?s new|new update|what changed|announced|released)\b", q):
        return "latest_update"
    if re.search(r"\b(how to|how do i|implement|configure|setup|set up|install)\b", q):
        return "implementation"
    if re.search(r"\b(difference between|compare|versus|vs\.?|better than)\b", q):
        return "comparison"
    if re.search(r"\b(why|error|failed|failure|timeout|not working|issue|problem)\b", q):
        return "troubleshooting"
    if re.search(r"\b(what is|what are|explain|meaning|how does)\b", q):
        return "concept_explanation"
    return "general"

"""Source selection policy based on intent, entities and freshness."""
from dataclasses import dataclass


@dataclass
class RetrievalPlan:
    local: bool
    web: bool
    official_only: bool
    reason: str


def plan(intent: str, entities=None, topic_relation: str = "new_topic") -> RetrievalPlan:
    entities = entities or []
    names = {getattr(e, "canonical", e) for e in entities}
    release_intent = intent == "latest_update"
    security_or_embedded = bool(names & {"Row-Level Security", "Object-Level Security", "Embedded Analytics"})
    if release_intent:
        return RetrievalPlan(local=True, web=True, official_only=True, reason="current_information_requires_fresh_official_evidence")
    if security_or_embedded and topic_relation in {"same_topic", "related_topic"}:
        return RetrievalPlan(local=True, web=True, official_only=True, reason="contextual_security_or_embedded_follow_up")
    return RetrievalPlan(local=True, web=False, official_only=False, reason="evergreen_local_first")

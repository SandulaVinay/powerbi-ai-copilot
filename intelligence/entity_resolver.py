"""Generic Power BI entity resolution.

Fast path: exact aliases. The resolver is domain-generic and intentionally
keeps ambiguity/context handling separate from retrieval and answer generation.
"""
from dataclasses import dataclass
import re
from typing import Dict, List, Optional


@dataclass
class EntityMatch:
    canonical: str
    entity_type: str
    alias: str
    confidence: float
    reason: str


ENTITY_CATALOG = {
    "Row-Level Security": ("security", ["rls", "row level security", "row-level security"]),
    "Object-Level Security": ("security", ["ols", "object level security", "object-level security"]),
    "Incremental Refresh": ("data_refresh", ["incremental refresh", "incremental refresh policy"]),
    "Direct Lake": ("storage_mode", ["direct lake", "direct lake mode"]),
    "DirectQuery": ("storage_mode", ["directquery", "direct query", "direct-query"]),
    "On-premises Data Gateway": ("connectivity", ["gateway", "on-premises gateway", "on premises gateway", "data gateway"]),
    "Semantic Model": ("modeling", ["semantic model", "semantic models"]),
    "Visual Calculations": ("calculation", ["visual calculations", "visual calculation"]),
    "LOOKUP": ("calculation", ["lookup function", "lookup"]),
    "TMDL": ("developer", ["tmdl", "tabular model definition language"]),
    "Copilot": ("ai", ["copilot", "power bi copilot"]),
    "Embedded Analytics": ("embedded", ["embedded", "power bi embedded", "embedded analytics"]),
    "Slicer": ("reporting", ["slicer", "slicers"]),
    "Power Query": ("data_transformation", ["power query", "m language", "m query"]),
}


def _matches(question: str, alias: str) -> bool:
    alias = alias.lower().strip()
    if " " in alias or "-" in alias:
        return alias in question.lower()
    return bool(re.search(r"\b" + re.escape(alias) + r"\b", question.lower()))


def resolve_entities(question: str, context: str = "") -> List[EntityMatch]:
    """Resolve Power BI entities without an LLM call on the common path."""
    matches: List[EntityMatch] = []
    for canonical, (entity_type, aliases) in ENTITY_CATALOG.items():
        for alias in aliases:
            if _matches(question, alias):
                matches.append(EntityMatch(canonical, entity_type, alias, 0.99, "exact_alias"))
                break

    # Context can disambiguate short aliases. OLS is shown here because it is
    # the current regression case, but the API is generic for future entities.
    context_lower = context.lower()
    if _matches(question, "ols") and any(x in context_lower for x in ["rls", "security", "power bi"]):
        for match in matches:
            if match.canonical == "Object-Level Security":
                match.confidence = 0.999
                match.reason = "contextual_security_disambiguation"

    return sorted(matches, key=lambda m: m.confidence, reverse=True)

"""Conversation topic switching decisions."""


def classify_topic_relation(current_entities, previous_entities, question: str, has_follow_up_signal: bool) -> str:
    current = {x.canonical for x in current_entities}
    previous = {x.get("name") for x in previous_entities}
    if current & previous:
        return "same_topic"
    if has_follow_up_signal or len(question.split()) <= 8:
        return "related_topic"
    if current:
        return "new_topic"
    return "ambiguous"
